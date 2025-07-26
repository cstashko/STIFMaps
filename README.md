![image](https://user-images.githubusercontent.com/25675422/235388288-5c2ab9c5-de8a-4b72-b6c9-1d1f141e5b25.png)

# Spatially Transformed Inferential Force Maps (STIFMaps)

STIFMaps predict the stiffness of breast tissue using the morphology of collagen fibers and nuclei.  


![Graphical Abstract](./assets/graphical_abstract.png)

Shown is a triple negative breast tumor. Scale bar, 50 microns. Stiffness values are natural log-transformed.

Code from the manuscript: "STIFMap employs a convolutional neural network to reveal 
spatial mechanical heterogeneity and tension-dependent activation of an epithelial 
to mesenchymal transition within human breast cancers"

## Contents

-[Repo Contents](#repo-contents)  
-[System Requirements](#system-requirements)  
-[Installation Guide](#installation-guide)  
-[Getting Started](#getting-started)  
-[Reproducing Manuscript Results](#reproducing-manuscript-results)  
-[License](#license)  
-[Acknowledgements](#acknowledgements)  
-[Contact](#contact)  

## Repo Contents
-[network_training.ipynb](./network_training.ipynb): Jupyter notebook for reproducing the trained models presented in the manuscript  
-[STIFMaps.ipynb](./STIFMaps.ipynb): Jupyter notebook for creating STIFMaps out of input DAPI and collagen images using trained networks  
-[tests](./tests): Contains test code and example image fixtures (e.g., `fixtures/no_stain`, `fixtures/with_stain`) used to validate and demonstrate the STIFMap generation pipeline in [STIFMaps.ipynb](./STIFMaps.ipynb)

## System Requirements

STIFMaps should run on any standard computer capable of running Jupyter and PyTorch, though 16 GB of RAM is required to enable CUDA optimization. Note that the computer must have enough RAM to support in-memory operations and the extent of memory usage depends on the size of the image that the user is trying to characterize using STIFMaps. Within [STIFMaps.ipynb](./STIFMaps.ipynb), the user may downsample the image prior to stiffness predictions to reduce memory consumption. 

Running STIFMaps.ipynb on the example images provided should only take a few minutes (on a computer with 16 GB of RAM, 12 cores@1.10 GHz, and running Ubuntu 18.04). The runtime to reproduce training for one network should take about 45 minutes to an hour (using a computer with 64 GB of RAM, 16 cores@3.60 GHz, and running Ubuntu 18.04).

## Installation Guide

It's recommended to run STIFMaps in a designated virtual environment. Create a virtual environment in Python 3.7 or later that includes pip using the following:
```bash
conda create -n STIFMaps python=3.10 pip
```

Then enter the virtual environment and install the STIFMaps PyPI package:
```bash
conda activate STIFMaps
python3 -m pip install STIFMaps
```

Alternatively, a virtual environment can be created from the provided `environment.yml` file:
```bash
conda env create -f environment.yml
```

## Getting Started

Once the STIFMaps package has been installed, run the [STIFMaps.ipynb](./STIFMaps.ipynb) notebook using paired collagen and DAPI images. Example images are available within the [fixtures](./tests/fixtures) directory. Trained models are available at https://data.mendeley.com/datasets/vw2bb5jy99/2

(Optional) A mask image of zeros and ones may be used to indicate which regions of an image should be excluded from analysis  

(Optional) An additional staining image may be used to compute colocalization between the stain and collagen, DAPI, and predicted stiffness  

## Reproducing Manuscript Results

Data for reproducing manuscript results is available via https://data.mendeley.com/datasets/vw2bb5jy99/2  

-**raw_squares**: The images used for training the neural networks  
-**stiffnesses.csv**: The table of ground truth stiffness values for each square used for model training. Note that stiffness values are natural-log transformed to limit the influence of ourliers  
-**trained_models**: 25 completed, trained models for use with STIFMaps.ipynb to predict elasticity values on an unknown tissue  
-**output**: Statistics regarding the training and accuracy for the trained models  

To reproduce manuscript results, the Jupyter notebook used for building and training the neural networks is available via [network_training.ipynb](./network_training.ipynb). As inputs, the Jupyter notebook needs the elasticity values contained in 'stiffnesses.csv' as well as the image files from 'raw_squares'.  

## License

This project is covered under the **MIT License**.

## Acknowledgements

Code for visualizing activation and saliency maps was modified from https://github.com/utkuozbulak/pytorch-cnn-visualizations, as credited within the publication.

## Contact

Please direct any questions to cstashko@hmc.edu
