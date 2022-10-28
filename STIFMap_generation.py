# imports 
import os
import numpy as np
import pandas as pd
from skimage import io
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import mannwhitneyu
from scipy import interpolate

import time

# Import STIFMaps helper functions
from .misc import norm_pic, get_step

#https://pytorch.org/vision/0.8/_modules/torchvision/models/alexnet.html
class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1, dropout: float = 0.5) -> None:
        super().__init__()
        #_log_api_usage_once(self)
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def generate_STIFMap(dapi, collagen, name, step, models, 
                    mask=False, batch_size=100, square_side=224, save_dir=False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device is ' + str(device))
    # For each collagen-DAPI pair, generate the STIFMap for each model provided

    # Get the half step size
    half_step = int(step/2)
    # and the half side length
    half_side = int(square_side/2)


    then = time.time()

    # Transformation to apply to every square; 
    # Resize the square to the appropriate dimensions for the network
    #      and convert to a tensor
    valid_transform = transforms.Compose(
        [transforms.ToPILImage(),
        transforms.Resize(size=(224,224)),
        transforms.ToTensor(),
        #transforms.Lambda(lambda x: x[0:2]) # Remove the blank channel
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])



    ############################################ Import the images
    #
    # FIX: CHECK THAT THE FILE EXISTS
    # FIX: ALSO THAT THE FILE IS THE CORRECT FORMAT
    #
    dapi = io.imread(dapi)
    collagen = io.imread(collagen)

    # If no mask was provided, then no regions should be excluded
    if mask == False:
        mask = np.ones(dapi.shape)
    # Otherwise, read in the mask
    else:
        mask = io.imread(mask)

    # Normalize the images
    dapi = norm_pic(dapi)
    collagen = norm_pic(collagen)
    # Convert the datatype
    dapi = dapi.astype(np.float32)
    collagen = collagen.astype(np.float32)


    '''    
    ######################### Save the dapi norm pic
    #io.imsave(save_dir + name + '_DAPI_norm.TIF', dapi)
    ######################### Save the collagen norm pic
    #io.imsave(save_dir + name + '_collagen_norm.TIF', collagen)
    '''

    # Combine DAPI and collagen into one RGB image with a layer of zeros
    blank = np.zeros(shape = collagen.shape, dtype = np.float32)
    im = np.stack((collagen,dapi,blank), axis=0)

    # Show the input image to the model
    io.imshow(np.moveaxis(im, 0, -1))

    print('Image shape is ' + str(im.shape))
    # Remove the original layers for memory purposes
    del dapi
    del collagen
    del blank

    ######################### Split the image into squares
    x_range = int((im.shape[2] - square_side) / (step)) + 1
    y_range = int((im.shape[1] - square_side) / (step)) + 1
    
    ######################### Create the list that will eventually be returned
    out = []

    print('Num squares in x direction is ' + str(x_range))
    print('Num squares in y direction is ' + str(y_range))

    # Generate two lists of the (x,y) indices for every square
    x = np.array(list(range(x_range)) * y_range)
    y = [[i]*x_range for i in list(range(y_range))]
    y = np.array(y)
    y = y.flatten()

    then2 = time.time()

    # If there is only one model, then make it a list for the upcoming 'for' loop
    if type(models) == str:
        models = [models]

    # Fit the squares with each model
    for model in models:
        # The iteration number
        model_name = model[-7:-3]

        # One list to keep track of input squares to the model
        squares = []
        # and another to keep track of the model prediction values
        outputs = []

        # Load the model
        network = AlexNet()
        network.load_state_dict(torch.load(model))
        network.eval()

        # Keep track of which values should be masked out
        masked_value_tracker = []

        # Go through each square one-by-one
        for i in range(len(x)):

            # Get the current (X,Y) position from the mask
            masked_value = mask[step*y[i] + half_side, step*x[i] + half_side]
            # and add it to the list
            masked_value_tracker.append(masked_value)

            # If a model is outside of the masked region, then we don't want to use it
            if masked_value != 0:
                # Otherwise, (1) get the square
                im_sub = im[:,step*y[i]:step*y[i] + square_side, step*x[i]:step*x[i] + square_side].copy()
                # (2) convert to a tensor
                im_sub = torch.from_numpy(im_sub)
                # (3) transform it to the right dimensions
                im_sub = valid_transform(im_sub)
                # (4) add it to the list of squares
                squares.append(im_sub)


            # Once we have <batch_size> worth of squares in the list, fit them using the model
            if (i % batch_size == 0) & (i != 0) & (len(squares) != 0):

                outputs_sub = network(torch.stack(squares))
                outputs_sub = torch.reshape(outputs_sub, (-1,))
                # Append the model prediction values onto the list of outputs
                outputs += list(outputs_sub.detach().cpu().numpy().flatten())

                # Reset the list of squares
                squares = [] 

        # Fit the last group of squares too, but only if theres something there
        if len(squares) != 0:

            outputs_sub = network(torch.stack(squares))
            outputs_sub = torch.reshape(outputs_sub, (-1,))
            outputs += list(outputs_sub.detach().cpu().numpy().flatten())

            squares = [] 

        # Save memory
        del network 

        now2 = time.time()
        print('Time taken to predict squares is ' + str(now2-then2))

        ################## Finally, reform the model predictions into the correct shape of the original
        ################## data by taking into account which values should be masked
        z = outputs
        z = np.array(z)

        z_out = []
        j = 0
        # If a value was masked, then it's a zero. Otherwise, it was predicted using the model
        for i in range(len(masked_value_tracker)):
            if masked_value_tracker[i] == 0:
                z_out.append(0)
            else:
                z_out.append(z[j])
                j += 1

        z_out = np.array(z_out)
        z_out = z_out.reshape((y_range,x_range))

        # Add the new layer onto the output
        out.append(z_out)

        # Save the predictions as a numpy array
        if save_dir != False:
            np.save(save_dir + '/' + name + '_model_' + model_name + '.npy', z_out)

    # Save memory
    del im

    now = time.time()
    print('Total time taken is ' + str(now-then))
    
    return np.array(out)

def collagen_paint(dapi, collagen, z, name, step,
                    mask=False, square_side=224,
                    scale_percent=100, save_dir=False):

    then = time.time()
    
    print(name)

    # Get the half step size
    half_step = int(step/2)
    # and the half side length
    half_side = int(square_side/2)

    #############################################################################
    ########################################## Load the STIFFMap
    #############################################################################
    z = np.mean(z, axis=0)

    ##### Get the min and max regions of the original STIFMap (not resized)
    z_range = z.flatten()
    # Remove masked regions
    z_range = [i for i in z_range if i != 0]
    z_min = min(z_range)
    z_max = max(z_range)

    y_range = z.shape[0]
    x_range = z.shape[1]

    ############################################ Import the images
    #
    # FIX: CHECK THAT THE FILE EXISTS
    # FIX: ALSO THAT THE FILE IS THE CORRECT FORMAT
    #
    dapi = io.imread(dapi)
    collagen = io.imread(collagen)

    # If no mask was provided, then no regions should be excluded
    if mask == False:
        mask = np.ones(dapi.shape)
    # Otherwise, read in the mask
    else:
        mask = io.imread(mask)
        
    #############################################################################
    ########################################## Reduce to remove regions not sampled with the STIFMap
    #############################################################################
    collagen = collagen[half_side - half_step : (y_range-1)*step + half_side + half_step,
         half_side - half_step : (x_range-1)*step + half_side + half_step]
    mask = mask[half_side - half_step : (y_range-1)*step + half_side + half_step,
         half_side - half_step : (x_range-1)*step + half_side + half_step]
    dapi = dapi[half_side - half_step : (y_range-1)*step + half_side + half_step,
         half_side - half_step : (x_range-1)*step + half_side + half_step]
    

    #############################################################################
    ########################################## Interpolate the stiffmap to make it the same dimensions as the collagen image
    #############################################################################
    x = np.arange(half_step, x_range*step+half_step, step)
    y = np.arange(half_step, y_range*step+half_step, step)

    f = interpolate.interp2d(x, y, z, kind='cubic')

    xnew = np.arange(0,mask.shape[1],1) #np.arange(0,y_range*step,1)
    ynew = np.arange(0,mask.shape[0],1) #np.arange(0,x_range*step,1)
    znew = f(xnew, ynew)
    # Clip to remove negative values
    znew = np.clip(znew, 0, 100000)
    
    #############################################################################
    ########################################## Mask out empty regions
    #############################################################################
    ###### Normalize inputs
    collagen = norm_pic(collagen)
    dapi = norm_pic(dapi)

    
    
    #############################################################################
    ########################################## Downsample
    ############ (scale_percent can be used to downsample the image if the full image uses too much memory)
    #############################################################################
    # Find the new dimensions    
    width = int(dapi.shape[1] * scale_percent / 100)
    height = int(dapi.shape[0] * scale_percent / 100)
    dim = (width, height)
    # Downsample the images
    dapi = cv2.resize(dapi, dim, interpolation = cv2.INTER_AREA)
    collagen = cv2.resize(collagen, dim, interpolation = cv2.INTER_AREA)
    znew = cv2.resize(znew, dim, interpolation = cv2.INTER_AREA)
    mask = cv2.resize(mask, dim, interpolation = cv2.INTER_AREA)
    

    
    
    ############################# Collagen paint
    
    # Mask out unused regions
    znew = znew * mask
    collagen = collagen * mask
    #del mask

    # Clip znew to only include the stiffness range in the original STIFMap
    znew_color = np.clip(znew, z_min, z_max)
    #del znew

    # Get the scaled image in RGB                 
    collagen3 = np.stack([collagen,collagen,collagen], axis=2)
    #del collagen
    # How many colors do we want to use
    n_colors=1000
    hues = sns.color_palette("viridis", n_colors+1)

    # For every pixel, get the index of the hue we want to use
    hue_indices = (n_colors * (znew_color - z_min) / (z_max - z_min)).astype(int)
    hues = np.array(hues)

    # Take the hue for each pixel and multiply by the intensity of the collagen image to get the final output
    znew_color = np.take(hues, hue_indices, axis=0)
    del hue_indices

    collagen_colored = collagen3 * znew_color
    del collagen3
    del znew_color

    if save_dir != False:
        plt.imsave(save_dir + '/' + name + '_STIFMap.png', collagen_colored)


    ######### Save the colorbar
    a = np.array([[z_min,z_max]])
    plt.figure(figsize=(10, 10))
    img = plt.imshow(a, cmap="viridis")
    plt.gca().set_visible(False)
    cax = plt.axes([0.1, 0.2, 0.1, 0.6])
    cbar = plt.colorbar(orientation="vertical", cax=cax)
    cbar.ax.set_ylabel('log(elasticity) (Pa)', fontsize=18)
    if save_dir != False:
        plt.savefig(save_dir + '/' + name + "_colorbar.png")
    
    
    
    now = time.time()
    print('Time taken is ' + str(now-then))

    return collagen_colored




def correlate_signals_with_stain(dapi, collagen, z, stain, step,
                    mask=False, square_side=224, 
                    scale_percent=100, quantile=0.99):

    then = time.time()



    # Read in the images
    dapi = io.imread(dapi)
    collagen = io.imread(collagen)
    stain = io.imread(stain)


    # If no mask was provided, then no regions should be excluded
    if mask == False:
        mask = np.ones(dapi.shape)
    # Otherwise, read in the mask
    else:
        mask = io.imread(mask)

    # Get the half step size
    half_step = int(step/2)
    # and the half side length
    half_side = int(square_side/2)





    z = np.mean(z, axis=0)

    ##### Get the min and max regions of the original STIFMap (not resized)
    z_range = z.flatten()
    # Remove masked regions
    z_range = [i for i in z_range if i != 0]
    z_min = min(z_range)
    z_max = max(z_range)

    y_range = z.shape[0]
    x_range = z.shape[1]




    #############################################################################
    ########################################## Reduce to remove regions not sampled with the STIFMap
    #############################################################################
    collagen = collagen[half_side - half_step : (y_range-1)*step + half_side + half_step,
         half_side - half_step : (x_range-1)*step + half_side + half_step]
    mask = mask[half_side - half_step : (y_range-1)*step + half_side + half_step,
         half_side - half_step : (x_range-1)*step + half_side + half_step]
    dapi = dapi[half_side - half_step : (y_range-1)*step + half_side + half_step,
         half_side - half_step : (x_range-1)*step + half_side + half_step]
    stain = stain[half_side - half_step : (y_range-1)*step + half_side + half_step,
         half_side - half_step : (x_range-1)*step + half_side + half_step]





    #############################################################################
    ########################################## Interpolate the stiffmap to make it the same dimensions as the collagen image
    #############################################################################
    x = np.arange(half_step, x_range*step+half_step, step)
    y = np.arange(half_step, y_range*step+half_step, step)

    f = interpolate.interp2d(x, y, z, kind='cubic')

    xnew = np.arange(0,mask.shape[1],1) #np.arange(0,y_range*step,1)
    ynew = np.arange(0,mask.shape[0],1) #np.arange(0,x_range*step,1)
    znew = f(xnew, ynew)
    # Clip to remove negative values
    znew = np.clip(znew, 0, 100000)

    #############################################################################
    ########################################## Mask out empty regions
    #############################################################################
    ###### Normalize inputs
    stain = norm_pic(stain)
    collagen = norm_pic(collagen)
    dapi = norm_pic(dapi)

    #############################################################################
    ########################################## Downsample
    ############ (scale_percent can be used to downsample the image if the full image uses too much memory)
    #############################################################################
    # Find the new dimensions    
    width = int(dapi.shape[1] * scale_percent / 100)
    height = int(dapi.shape[0] * scale_percent / 100)
    dim = (width, height)
    # Downsample the images
    dapi = cv2.resize(dapi, dim, interpolation = cv2.INTER_AREA)
    collagen = cv2.resize(collagen, dim, interpolation = cv2.INTER_AREA)
    stain = cv2.resize(stain, dim, interpolation = cv2.INTER_AREA)
    znew = cv2.resize(znew, dim, interpolation = cv2.INTER_AREA)
    mask = cv2.resize(mask, dim, interpolation = cv2.INTER_AREA)

    ########################################### Evaluate the correlations between different signals    

    dapi = dapi.flatten()
    collagen = collagen.flatten()
    stain = stain.flatten()
    znew = znew.flatten()
    mask = mask.flatten()

    df = np.zeros((len(collagen),5))

    df[:,0] = dapi
    df[:,1] = collagen
    df[:,2] = stain
    df[:,3] = znew
    df[:,4] = mask

    del dapi
    del collagen
    del stain
    del znew
    del mask


    df = pd.DataFrame(df)
    df.columns = ['dapi','collagen','stain','z','mask']

    # Remove regions that are masked out
    df = df.loc[df['mask'] > .2]

    ######################################### Looking at 99th stain percentiles for each z interval
    z_min = int(np.min(df['z'])*100)
    z_max = int((np.max(df['z'])*100)+1)

    intervals = np.array(range(z_min,z_max,1))/100

    stain_tracker = []
    intervals_used = []

    for interval in intervals:
        df_sub = df.loc[df['z'] < interval + .01].loc[df['z'] > interval]
        if df_sub.shape[0] != 0:
            stain_tracker.append(np.quantile(df_sub['stain'], quantile))
            intervals_used.append(interval)

    z_stain_spearman = spearmanr(intervals_used, stain_tracker)
    print('Spearman correlation between stain and predicted stiffness is ' + str(z_stain_spearman[0]))
    
    ######################################### Looking at 99th stain percentiles for each collagen interval
    intervals = np.array(range(1,100))/100

    stain_tracker = []
    intervals_used = []

    for interval in intervals:
        df_sub = df.loc[df['collagen'] < interval + .01].loc[df['collagen'] > interval]
        if df_sub.shape[0] != 0:
            stain_tracker.append(np.quantile(df_sub['stain'], quantile))
            intervals_used.append(interval)

    collagen_stain_spearman = spearmanr(intervals_used, stain_tracker)
    print('Spearman correlation between stain and collagen is ' + str(collagen_stain_spearman[0]))

    ######################################### Looking at 99th stain percentiles for each dapi interval
    intervals = np.array(range(1,100))/100

    stain_tracker = []
    intervals_used = []

    for interval in intervals:
        df_sub = df.loc[df['dapi'] < interval + .01].loc[df['dapi'] > interval]
        if df_sub.shape[0] != 0:
            stain_tracker.append(np.quantile(df_sub['stain'], quantile))
            intervals_used.append(interval)

    dapi_stain_spearman = spearmanr(intervals_used, stain_tracker)
    print('Spearman correlation between stain and dapi is ' + str(dapi_stain_spearman[0]))


    now = time.time()
    print('time taken is ' + str(now-then))
    
    return z_stain_spearman, collagen_stain_spearman, dapi_stain_spearman
