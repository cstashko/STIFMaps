# This file includes code adapted from:
# https://github.com/utkuozbulak/pytorch-cnn-visualizations
# Copyright (c) 2025 utkuozbulak
# Licensed under the MIT License

from collections import OrderedDict
from typing import Dict, Callable
import torch
from skimage import io
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F
from torch.nn import ReLU


def remove_all_hooks(model: torch.nn.Module) -> None:
    for name, child in model._modules.items():
        if child is not None:
            if hasattr(child, "_forward_hooks"):
                child._forward_hooks: Dict[int, Callable] = OrderedDict()
            elif hasattr(child, "_forward_pre_hooks"):
                child._forward_pre_hooks: Dict[int, Callable] = OrderedDict()
            elif hasattr(child, "_backward_hooks"):
                child._backward_hooks: Dict[int, Callable] = OrderedDict()
            remove_all_hooks(child)


################################################## GradCam
class GradCamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        for module_pos, module in self.model.features._modules.items():
            x = module(x)  # Forward
            #print(self.target_layer)
            if module_pos == self.target_layer:
                #print('target layer found')
                x.register_hook(self.save_gradient)
                conv_output = x  # Save the convolution output on that layer
        return conv_output, x

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        x = x.view(x.size(0), -1)  # Flatten
        # Forward pass on the classifier
        x = self.model.classifier(x)
        return conv_output, x


class GradCam():
    """
        Produces class activation map
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = GradCamExtractor(self.model, target_layer)

    def generate_cam(self, input_image, target_class=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(input_image)
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())
        # Zero grads
        self.model.features.zero_grad()
        self.model.classifier.zero_grad()
        # Backward pass with specified target

        model_output.backward(retain_graph=True)
        # Get hooked gradients
        guided_gradients = self.extractor.gradients.data.cpu().numpy()[0]
        # Get convolution outputs
        target = conv_output.data.cpu().numpy()[0]
        # Get weights from gradients
        weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # Have a look at issue #11 to check why the above is np.ones and not np.zeros
        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                       input_image.shape[3]), Image.ANTIALIAS))/255
        
        return cam


################################################## ScoreCam
class ScoreCamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        for module_pos, module in self.model.features._modules.items():
            x = module(x)  # Forward
            if module_pos == self.target_layer:
                conv_output = x  # Save the convolution output on that layer
        return conv_output, x

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        x = x.view(x.size(0), -1)  # Flatten
        # Forward pass on the classifier
        x = self.model.classifier(x)
        return conv_output, x


class ScoreCam():
    """
        Produces class activation map
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = ScoreCamExtractor(self.model, target_layer)

    def generate_cam(self, input_image, target_class=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(input_image)
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())
        # Get convolution outputs
        target = conv_output[0]
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        for i in range(len(target)):
            # Unsqueeze to 4D
            saliency_map = torch.unsqueeze(torch.unsqueeze(target[i, :, :],0),0)
            # Upsampling to input size
            saliency_map = F.interpolate(saliency_map, size=(224, 224), mode='bilinear', align_corners=False)
            if saliency_map.max() == saliency_map.min():
                continue
            # Scale between 0-1
            norm_saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
            # Get the target score
            w = F.softmax(self.extractor.forward_pass(input_image*norm_saliency_map)[1],dim=1)[0][target_class]
            cam += w.data.cpu().numpy() * target[i, :, :].data.cpu().numpy()
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                       input_image.shape[3]), Image.ANTIALIAS))/255
        return cam

################################################## Guided Backprop
class GuidedBackprop():
    """
       Produces gradients generated with guided back propagation from the given image
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.hook_list = []
        self.forward_relu_outputs = []
        # Put model in evaluation mode
        self.model.eval()
        self.update_relus()
        self.hook_list.append(self.hook_layers())

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]
        # Register hook to the first layer
        first_layer = list(self.model.features._modules.items())[0][1]
        return first_layer.register_backward_hook(hook_function)

    def update_relus(self):
        """
            Updates relu activation functions so that
                1- stores output in forward pass
                2- imputes zero for gradient values that are less than zero
        """
        def relu_backward_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, change it to zero
            """
            # Get last forward output
            corresponding_forward_output = self.forward_relu_outputs[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)
            del self.forward_relu_outputs[-1]  # Remove last forward output
            return (modified_grad_out,)

        def relu_forward_hook_function(module, ten_in, ten_out):
            """
            Store results of forward pass
            """
            self.forward_relu_outputs.append(ten_out)

        # Loop through layers, hook up ReLUs
        for pos, module in self.model.features._modules.items():
            if isinstance(module, ReLU):
                self.hook_list.append(module.register_backward_hook(relu_backward_hook_function))
                self.hook_list.append(module.register_forward_hook(relu_forward_hook_function))

    def generate_gradients(self, input_image, target_class):
        # Forward pass
        model_output = self.model(input_image)
        # Zero gradients
        self.model.zero_grad()
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Backward pass
        model_output.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = self.gradients.data.cpu().numpy()[0]
        return gradients_as_arr
    
    def remove_all_hooks(self):
        for hook in self.hook_list:
            hook.remove()
    
################################################## Guided GradCam
def guided_grad_cam(grad_cam_mask, guided_backprop_mask):
    """
        Guided grad cam is just pointwise multiplication of cam mask and
        guided backprop mask

    Args:
        grad_cam_mask (np_arr): Class activation map mask
        guided_backprop_mask (np_arr):Guided backprop mask
    """
    cam_gb = np.multiply(grad_cam_mask, guided_backprop_mask)
    return cam_gb

def visualize_sample(network, img_path, dest):
   
    fig2, axs = plt.subplots(2, 4, figsize=(20,10))
    
    # validation transform
    side_length_crop = 224
    valid_transform = transforms.Compose(
        [transforms.ToPILImage(),
        transforms.CenterCrop(side_length_crop),
        transforms.ToTensor(),
        #transforms.Lambda(lambda x: x[0:2]) # Remove the blank channel
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    t = io.imread(img_path)

    t3 = valid_transform(t)

    # Need to set requires_grad to true to get the gradients during backpropagation
    t3.requires_grad = True

    # Adds one dimension
    prep_img = t3[None, :]

    ##### Plotting 
    t4 = t3.detach().cpu().numpy()
    zeros = np.zeros(t4[0].shape)
    plt.subplot(241)
    io.imshow(np.stack([t4[0], zeros, t4[1]], axis=2))
    plt.title('Input Image')
    plt.grid(None)

    ############################################# Saliency Map
    score = network(t3[None])[0]
    score.backward()

    sal = t3.grad.data.abs()

    ################## Plotting
    im_new = np.stack([sal[0], sal[1], zeros.astype('float32')], axis=2)
    im_new = im_new / np.max(im_new)
    #im_new = (255*im_new).astype(np.uint8)
    plt.subplot(242)
    plt.title('Saliency Map')
    io.imshow(im_new)
    plt.grid(None)




    ############################################# GradCam
    gcv2 = GradCam(network, target_layer='3')
    # Generate cam mask
    cam = gcv2.generate_cam(prep_img, target_class=0)
    #print('Grad cam completed')

    plt.subplot(243)
    plt.title('GradCam')
    io.imshow(cam, cmap='viridis')
    plt.grid(None)
    
    '''
    ######### Save the colorbar
    c_min = np.min(cam)
    c_max = np.max(cam)
    a = np.array([[c_min,c_max]])
    plt.figure(figsize=(10, 10))
    img = plt.imshow(a, cmap="viridis")
    plt.gca().set_visible(False)
    cax = plt.axes([0.1, 0.2, 0.1, 0.6])
    cbar = plt.colorbar(orientation="vertical", cax=cax)
    cbar.ax.tick_params(labelsize=18)
    #cbar.ax.set_ylabel('log(elasticity) (Pa)', fontsize=18)
    plt.savefig(dest + img_path.split('/')[-1][:-4] + '_gradcam_colorbar.svg',
        format = 'svg', transparent=True, bbox_inches='tight')#, dpi=1000)
    #plt.savefig(out_dir + name + "_colorbar.png")
    '''
    
    ############################################# ScoreCam
    score_cam = ScoreCam(network, target_layer='3')
    # Generate cam mask
    cam = score_cam.generate_cam(prep_img, target_class=0)

    plt.subplot(244)
    plt.title('ScoreCam')
    io.imshow(cam, cmap='viridis')
    plt.grid(None)
    
    '''
    ######### Save the colorbar
    c_min = np.min(cam)
    c_max = np.max(cam)
    a = np.array([[c_min,c_max]])
    plt.figure(figsize=(10, 10))
    img = plt.imshow(a, cmap="viridis")
    plt.gca().set_visible(False)
    cax = plt.axes([0.1, 0.2, 0.1, 0.6])
    cbar = plt.colorbar(orientation="vertical", cax=cax)
    cbar.ax.tick_params(labelsize=18)
    #cbar.ax.set_ylabel('log(elasticity) (Pa)', fontsize=18)
    plt.savefig(dest + img_path.split('/')[-1][:-4] + '_scorecam_colorbar.svg',
        format = 'svg', transparent=True, bbox_inches='tight')#, dpi=1000)
    #plt.savefig(out_dir + name + "_colorbar.png")
    '''

    
    ############################################# Guided backprop
    GBP = GuidedBackprop(network)
    # Get gradients
    guided_grads = GBP.generate_gradients(prep_img, target_class=0)
    #print('Guided backpropagation completed')

    input_image = prep_img
    target_class = 0


    model_output = GBP.model(input_image)
    # Zero gradients
    GBP.model.zero_grad()
    # Backward pass
    model_output.backward()
    # Convert Pytorch variable to numpy array
    # [0] to get rid of the first channel (1,3,224,224)
    gradients_as_arr = GBP.gradients.data.cpu().numpy()[0]

    # Convert to grayscale
    grayscale_guided_grads = convert_to_grayscale(guided_grads)
    # Positive and negative saliency maps
    pos_sal, neg_sal = get_positive_negative_saliency(guided_grads)


    # Remove the added hooks from the network
    GBP.remove_all_hooks()

    plt.subplot(245)
    plt.title('Guided BackProp')
    io.imshow(grayscale_guided_grads[0])
    plt.grid(None)

    plt.subplot(246)
    plt.title('Positive Saliency')
    io.imshow(pos_sal[0])
    plt.grid(None)

    plt.subplot(247)
    plt.title('Negative Saliency')
    io.imshow(neg_sal[0])
    plt.grid(None)


    ############################################# Guided Grad cam
    cam_gb = guided_grad_cam(cam, guided_grads)
    grayscale_cam_gb = convert_to_grayscale(cam_gb)

    plt.subplot(248)
    plt.title('Guided GradCam')
    io.imshow(grayscale_cam_gb[0])
    plt.grid(None)

    fig2.savefig(dest + img_path.split('/')[-1][:-4] + '_VIS.png',
                 facecolor='white', edgecolor='none')
    
    fig2.savefig(dest + img_path.split('/')[-1][:-4] + '_VIS.svg',
                 format='svg', facecolor='white', edgecolor='none')
    
    return

# Helper functions from /pytorch_cnn_visualizations_master/src/misc_functions.py

def convert_to_grayscale(im_as_arr):
    """
        Converts 3d image to grayscale

    Args:
        im_as_arr (numpy arr): RGB image with shape (D,W,H)

    returns:
        grayscale_im (numpy_arr): Grayscale image with shape (1,W,D)
    """
    grayscale_im = np.sum(np.abs(im_as_arr), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return grayscale_im

def get_positive_negative_saliency(gradient):
    """
        Generates positive and negative saliency maps based on the gradient
    Args:
        gradient (numpy arr): Gradient of the operation to visualize

    returns:
        pos_saliency ( )
    """
    pos_saliency = (np.maximum(0, gradient) / gradient.max())
    neg_saliency = (np.maximum(0, -gradient) / -gradient.min())
    return pos_saliency, neg_saliency


