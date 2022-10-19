# Spatially Transformed Inferential Force Maps (STIFMaps)

STIFMaps predicts the stiffness of breast tissue using the morphology of collagen fibers and nuclei.  


![alt text](https://github.com/cstashko/STIFMaps/blob/master/test_cases/example_image_2.png)

Shown is a triple negative breast tumor. Scale bar, 50 microns  

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
-[Contact](#contact)  

## Repo Contents
-[network_training.ipynb](./network_training.ipynb): Jupyter notebook for reproducing the trained models presented in the manuscript  
-[STIFMaps.ipynb](./STIFMaps.ipynb): Jupyter notebook for creating STIFMaps out of an input image using trained networks  
-[test_cases](./test_cases): Example images to use with 'STIFMaps.ipynb' to see the pipeline in action

## System Requirements

STIFMaps should run on any standard computer capable of running Jupyter and PyTorch, though 16 GB of RAM is required to enable CUDA optimization. Note that the computer must have enough RAM to support in-memory operations and the extent of memory usage depends on the size of the image that the user is trying to characterize using STIFMaps. Within 'STIFMaps.ipynb', the user may downsample the image prior to stiffness predictions to reduce memory consumption. 

Running STIFMaps.ipynb on the example images provided should only take a few minutes (on a computer with 16 GB of RAM, 12 cores@1.10 GHz, and running Ubuntu 18.04). The runtime to reproduce the network training should take several hours (using a computer with 64 GB of RAM, 16 cores@3.60 GHz, and running Ubuntu 18.04.

## Installation Guide

Create a virtual environment in Python 3.7 or later and install the PyPI package using the following:
```bash
python3 -m pip install STIFMaps
```

## Getting Started

Once the STIFMaps package has been installed, run the [STIFMaps.ipynb](./STIFMaps.ipynb) notebook using paired collagen and DAPI images. Example images are available via the [test_cases](./test_cases) directory.  

(Optional) A mask image of zeros and ones may be used to indicate which regions of an image should be excluded from analysis  

(Optional) An additional staining image may be used to compute colocalization between the stain and collagen, DAPI, and predicted stiffness  

## Reproducing Manuscript Results

Data for reproducing manuscript results is available via https://data.mendeley.com/datasets/vw2bb5jy99/2  
-**raw_squares**: The images used for training the neural networks  
-**stiffnesses.csv: The table of ground truth stiffness values for each square used for model training  
-**trained_models**: 25 completed, trained models for use with STIFMaps.ipynb to predict elasticity values on an unknown tissue  
-**output**: Statistics regarding the training and accuracy for the trained models  

To reproduce manuscript results, the Jupyter notebook used for building the neural networks is available via 'network_training.ipynb'. As inputs, the Jupyter notebook needs the elasticity values contained in 'stiffnesses.csv' as well as the image files from 'raw_squares'.  

## License

This project is covered under the **MIT License**.

## Contact

Please direct any questions to cstashko@berkeley.edu
