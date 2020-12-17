# Files

- archive: This folder contains old notebooks and files that are not used for the final product
- models: This folder contains all models that have been trained and tested on Fashion-MNIST
	- file name logic: model_i{INPUT WIDTH}_w{WEIGHT WIDTH}_a{ACTIVATION WIDTH}.pth
- onnx: This folder contains all onnx files acquired during FINN transformation and deploy process for each final and preliminary model
- QuantLenetV2.py: This file contains the structure of our custom convolutional neural network that is used throughout the project and deployed on Pynq
- rain_utils.py: This file contains the training and testing functions that are used in the training and testing notebooks
- Train_i{INPUT WIDTH}_w{WEIGHT WIDTH}_a{ACTIVATION WIDTH}.ipynb
	- These notebooks create, train, and test the respective bit width models on Fashion-MNIST
- Deployw{WEIGHT WIDTH}_a{ACTIVATION WIDTH}.ipynb
	- These notebooks perform the FINN and HLS transformations and deploy our models onto Pynq
