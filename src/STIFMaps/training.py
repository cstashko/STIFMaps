import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from tqdm.notebook import tqdm
from skimage import io
from scipy.stats import pearsonr

from .vis import visualize_sample

class CustomImageDataset(torch.utils.data.Dataset):
    ''' Class for importing custom images to use during network training '''
    
    # Initialize the directory containing the images, the annotation file, and both transforms
    def __init__(self, df, img_dir, transform=None, target_transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform # this transform is applied to the labels

    # Returns the number of samples in our dataset
    def __len__(self):
        return len(self.df)

    # Loads and returns a sample from the dataset at the given index idx
    # Based on the index, it identifies the images location on the disk, converts it to a tensor using torch.from_numpy
    # retrieves the corresponding label/value from the csv data in self.df, calls the transform function (if applicable)
    # and returns the tensor image and corresponding label in a tuple
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.df.iloc[idx]['Key'])
        # Read the image
        im = io.imread(img_path)
        # Change the order of dimensions so the channels comes first
        im = np.moveaxis(im, -1, 0)
        # Convert to a tensor
        image = torch.from_numpy(im)#.float()
        label = self.df.iloc[idx]['Stiffness']
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


# AlexNet class code from: https://pytorch.org/vision/0.8/_modules/torchvision/models/alexnet.html
class AlexNet(nn.Module):
    ''' Class to define the network used during training itself ''' 
    
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
    

def train_model(df, img_dir, name,
    brightness_range, contrast_range, sharpness_range,
    # Parameters for the model
    batch_size = 16,
    n_epochs = 100,
    learning_rate = 4e-6,
    weight_decay = 4e-7,
    # Loss function
    criterion = nn.MSELoss(),
    save_directory=False,
    save_visualizations=False):

    '''
    Trains the model to reproduce results from the manuscript

    Arguments:
     - df: The dataframe 'stiffnesses.csv'
     - img_dir: The directory where the squares are stored 
     - name: The name that will be used during saving (if applicable)
     - brightness_range: The upper and lower bounds of brightness adjustments. Used to augment data to artificially 'increase' the size of the training data, emphasize relevant features, and prevent overfitting of the training data
     - contrast_range: The upper and lower bounds of contrast adjustments
     - sharpness_range: The upper and lower bounds of sharpness adjustments
     - batch_size: How many samples to run through the model at once when computing the direction to step the model
     - n_epochs: The number of epochs that the model will be trained over
     - learning_rate: The step size used to change the model parameters in the direction of the error gradient
     - weight_decay: Regularization parameter used to prevent model overfitting by reducing all of the network weights each epoch
     - criterion: The loss/cost function used to compute model errors
     - save_directory: The directory where training statistics and summary plots will be stored. Note that a value of 'False' means that these will no be saved
     - save_visualizations: If 'True', will save saliency plots for the best and worst fits in the training and validation data sets. Note that 'save_directory' must be specified for plots to be saved
    
    Returns:
    In addition to saving output plots (if specified), `train_model` returns the trained network.
    '''

    # Side lengths for the original images and the sidelength used for training
    side_length_source = 448
    side_length_crop = 224

    # device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device

    ########################################## Define transforms
    # Training dataset transformation
    train_transform = transforms.Compose(
        [transforms.ToPILImage(),
        # (optional) Resize to account for different magnification (Data is 0.2405 um/pixel)
        #transforms.Resize(size=(248,248)),
        #transforms.Resize(size=(448,448)),
            
        # Transformations
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=360),
        transforms.CenterCrop(side_length_crop),
        transforms.ColorJitter(brightness=brightness_range, contrast=contrast_range),
        transforms.RandomAdjustSharpness(sharpness_range[0], p=.5),
        transforms.RandomAdjustSharpness(sharpness_range[1], p=.5),
        #transforms.GaussianBlur(kernel_size=(5,9), sigma=(.9,1.1)),
        transforms.ToTensor(),
        #transforms.Lambda(lambda x: x[0:2]) # Remove the blank channel
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    # Used as the validation transform during model training
    valid_transform = transforms.Compose(
        [transforms.ToPILImage(),
        transforms.CenterCrop(side_length_crop),
        transforms.ToTensor(),
        #transforms.Lambda(lambda x: x[0:2]) # Remove the blank channel
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])


    ################################################## Split the dataset into validation and training
    validation_split = 0.2

    # Get a list of unique samples and randomize them
    samples = df['Sample'].unique()
    np.random.shuffle(samples)

    # Split the samples into a training and validation set
    train_samples = samples[round(len(samples)*validation_split):]
    val_samples = samples[:round(len(samples)*validation_split)]

    ########## Split the dataframe into train and validation
    df_val = df.loc[df['Sample'].isin(val_samples)]
    df_train = df.loc[df['Sample'].isin(train_samples)]

    # Reset the index
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)

    print('Length of training df is ' + str(df_train.shape[0]))
    print('Length of validation df is ' + str(df_val.shape[0]))
    print(df_val['Sample'].unique())


    ######### Specify the training and validation datasets
    training_set = CustomImageDataset(
        df = df_train,
        img_dir = img_dir,
        transform = train_transform
    )
    validation_set = CustomImageDataset(
        df = df_val,
        img_dir = img_dir,
        transform = valid_transform
    )


    trainloader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True)
    validloader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=False)



    #################################################################### Initialize the network
    network = AlexNet()

    print('Num of parameters is ' +str(sum(p.numel() for p in network.parameters())))
    #print([p.numel() for p in network.parameters()])




    ######################################################### TRAIN THE NETWORK
    then = time.time()

    network_training_loss = []
    network_training_pval = []
    network_training_rval = []
    network_valid_loss = []
    network_valid_pval = []
    network_valid_rval = []


    network.to(device) 

    #optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
    optimizer = optim.Adam(network.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # train for n_epochs 
    for epoch in tqdm(range(n_epochs)):  # loop over the dataset multiple times
        running_loss = 0.0

        then2 = time.time()

        # NEED TO SWITCH BETWEEN EVALUATION AND TRAINING MODE FOR THE DROPOUT LAYERS TO WORK AS INTENDED
        network.train()

        train_labels = []
        train_outputs = []
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            # labels provides the ground truth answers
            inputs, labels = data[0].to(device), data[1].to(device)  
            #print('TRAINING')
            #print(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = network(inputs)
            outputs = torch.reshape(outputs, (-1,))

            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            # loss accumulation 
            running_loss += loss.item()



            # Getting correlation coefficients
            labels = list(labels.cpu().numpy())
            outputs = list(outputs.detach().cpu().numpy().flatten())

            train_labels = train_labels + labels
            train_outputs = train_outputs + outputs

        r_val, p_val = pearsonr(train_labels, train_outputs)

        network_training_loss.append(running_loss/(i+1))
        network_training_pval.append(p_val)
        network_training_rval.append(r_val)

        ########################## Check the validation dataset
        # NEED TO SWITCH BETWEEN EVALUATION AND TRAINING MODE FOR THE DROPOUT LAYERS TO WORK AS INTENDED
        network.eval()

        valid_loss = 0.0

        val_labels = []
        val_outputs = []
        for i, data in enumerate(validloader, 0):
        #for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            # labels provides the ground truth answers
            inputs, labels = data[0].to(device), data[1].to(device)          
            #print('VALIDATION')
            #print(labels)

            # forward + backward + optimize
            outputs = network(inputs)
            outputs = torch.reshape(outputs, (-1,))

            loss = criterion(outputs, labels)
            valid_loss += loss.item()

            # Getting correlation coefficients
            labels = list(labels.cpu().numpy())
            outputs = list(outputs.detach().cpu().numpy().flatten())

            val_labels = val_labels + labels
            val_outputs = val_outputs + outputs

        r_val, p_val = pearsonr(val_labels, val_outputs)

        network_valid_loss.append(valid_loss/(i+1))
        network_valid_pval.append(p_val)
        network_valid_rval.append(r_val)
        print(r_val)



        now2 = time.time()
        print('Epoch time is ' + str(now2-then2))

    now = time.time()

    print('Training time is ' + str(now-then))

    if save_directory != False:
        # Add a backslash in case it's not included
        save_directory = save_directory + '/'

        ################################# Save output plots
        fig = plt.figure(figsize=(10,10))
        plt.figure(fig)

        plt.rcParams.update({'font.size': 22})

        ####### Loss
        plt.scatter(list(range(len(network_training_loss))), network_training_loss, label='Training')
        plt.scatter(list(range(len(network_valid_loss))), network_valid_loss, label='Validation')
        plt.legend(fontsize=22)
        plt.xlabel('Epoch', fontsize=26)
        plt.ylabel('Loss', fontsize=26)
        fig.savefig(save_directory + name + '_loss.TIF',  bbox_inches='tight')
        fig.clf()

        ####### pval
        plt.scatter(list(range(len(network_training_pval))), -np.log10(network_training_pval), label='Training')
        plt.scatter(list(range(len(network_valid_pval))), -np.log10(network_valid_pval), label='Validation')
        plt.legend(fontsize=22)
        plt.xlabel('Epoch', fontsize=26)
        plt.ylabel('-log10(p-value)', fontsize=26)
        fig.savefig(save_directory + name + '_pval.TIF', bbox_inches='tight')
        fig.clf()

        ####### rval
        plt.scatter(list(range(len(network_training_rval))), network_training_rval, label='Training')
        plt.scatter(list(range(len(network_valid_rval))), network_valid_rval, label='Validation')
        plt.legend(fontsize=22)
        plt.xlabel('Epoch', fontsize=26)
        plt.ylabel('Correlation (r)', fontsize=26)
        fig.savefig(save_directory + name + '_rval.TIF', bbox_inches='tight')
        fig.clf()

        ####### scatter
        val_labels = np.array(val_labels)
        val_outputs = np.array(val_outputs)

        print('mse is ' + str(((val_labels - val_outputs)**2).mean()))
        print(pearsonr(val_labels,val_outputs))

        plt.scatter(val_labels, val_outputs, s=5)
        plt.gca().set_aspect('equal')
        plt.xlabel('log(actual stiffness) (Pa)', fontsize=22)
        plt.ylabel('log(predicted stiffness) (Pa)', fontsize=22)
        fig.savefig(save_directory + name + '_scatter.TIF',  bbox_inches='tight')
        fig.clf()

        ####### dataframe of QC metrics 
        df_out = pd.DataFrame({'network_training_loss':network_training_loss, 
            'network_training_pval':network_training_pval,
            'network_training_rval':network_training_rval,
            'network_valid_loss':network_valid_loss,
            'network_valid_pval':network_valid_pval,
            'network_valid_rval':network_valid_rval})
        df_out.to_csv(save_directory + name + '_TRAINING.csv')


        ####### dataframe of predicted vs actual values
        df_val['Predicted Stiffness'] = val_outputs
        df_val['Error'] = abs(df_val['Stiffness'] - df_val['Predicted Stiffness'])**2 # / df_val['Stiffness'] # 
        df_val = df_val.sort_values('Error', ascending=False)
        df_out2 = df_val[['Sample','Forceplot','Stiffness','Predicted Stiffness','Error']]
        df_out2.to_csv(save_directory + name + '_PREDICTIONS.csv')

        ####### dataframe of error by sample
        #key_list = df_val.sort_values('Error', ascending=False).head(10)['Key'].tolist()
        df_out3 = df_val.groupby('Sample').mean()['Error'].to_frame()
        df_out3 = df_out3.sort_values('Error', ascending=False)
        df_out3.to_csv(save_directory + name + '_ERRORS.csv')


        ############################# Write out the parameters
        with open(save_directory + name + '_log.txt', 'a') as f:
            f.write('brightness_range:' + str(brightness_range) + '\n')
            f.write('contrast_range:' + str(contrast_range) + '\n')
            f.write('sharpness_range:' + str(sharpness_range) + '\n')

        
        #############################
        #############################
        ###################################################### Visualizations
        ############################# Save saliency maps for the 5 best and 5 worst samples
        ############################# for the training and validation datasets
        if save_visualizations == True:
            network.to("cpu")

            ############################# Plot the top 10 best and worst predicted plots
            # Make directories
            if not os.path.isdir(save_directory + name):
                os.makedirs(save_directory + name)
                os.makedirs(save_directory + name + '/training/')
                os.makedirs(save_directory + name + '/training/worst_fits/')
                os.makedirs(save_directory + name + '/training/best_fits/')
                os.makedirs(save_directory + name + '/validation/')
                os.makedirs(save_directory + name + '/validation/worst_fits/')
                os.makedirs(save_directory + name + '/validation/best_fits/')
    
            ###################### Validation set
            # First the worst fits
            dest = save_directory + name + '/validation/worst_fits/'
    
            key_list = df_val.head(4)['Key'].tolist()
            print('Saving worst validation fits')
            for key in key_list:
                ####### Visualizations
                img_path = os.path.join(img_dir, key)
                print('Saving ' + img_path)
                visualize_sample(network, img_path, dest)
    
    
    
    
            # Then the best fits
            dest = save_directory + name + '/validation/best_fits/'
    
            key_list = df_val.tail(4)['Key'].tolist()
            print('Saving best validation fits')
            for key in key_list:
                ####### Visualizations
                img_path = os.path.join(img_dir, key)
                print('Saving ' + img_path)
                visualize_sample(network, img_path, dest)
     
    
    
    
            ###################### Training set
            train_predictions = pd.DataFrame({'Ground Truth':train_labels,
                                              'Predicted Stiffness':train_outputs})
    
            df_train = pd.merge(df_train, train_predictions, left_on='Stiffness', right_on='Ground Truth')
    
            df_train['Error'] = abs(df_train['Stiffness'] - df_train['Predicted Stiffness'])**2 # / df_train['Stiffness'] # 
            df_train = df_train.sort_values('Error', ascending=False)
    
    
            # First the worst fits
            dest = save_directory + name + '/training/worst_fits/'
    
            key_list = df_train.head(4)['Key'].tolist()
            for key in key_list:
                ####### Visualizations
                img_path = os.path.join(img_dir, key)
                print('Saving ' + img_path)
                visualize_sample(network, img_path, dest)
    
    
    
    
            # Then the best fits
            dest = save_directory + name + '/training/best_fits/'
    
            key_list = df_train.tail(4)['Key'].tolist()
            for key in key_list:
                ####### Visualizations
                img_path = os.path.join(img_dir, key)
                print('Saving ' + img_path)
                visualize_sample(network, img_path, dest)
    if save_directory == False:
        if save_visualizations == True:
            print('No visualizations were saved because no save_directory was specified')

    return network
























