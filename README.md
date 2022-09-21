# STIFMaps

Code from the manuscript: "STIFMap employs a convolutional neural network to reveal 
spatial mechanical heterogeneity and tension-dependent activation of an epithelial 
to mesenchymal transition within human breast cancers"

To generate STIFMaps, create a virtual environment using the 'env.yml' file. Then, refer to the Jupyter notebook 'STIFMaps.ipynb' to create STIFMaps using paired GFP and DAPI images. Additional staining images may optionally be included to evaluate the overlap between stain intensity and STIFMaps.

To reproduce manuscript results, the Jupyter notebook used for building the neural networks is available via 'network_training.ipynb'. As inputs, the Jupyter notebook needs the elasticity values contained in 'stiffnesses.csv' as well as the image files available at data.mendeley.com with reserved DOI: 10.17632/vw2bb5jy99.2

25 completed models are available at https://data.mendeley.com/datasets/vw2bb5jy99/2 via 'trained_models' and output plots can be found in the 'output' directory.

Please direct any questions to cstashko@berkeley.edu
