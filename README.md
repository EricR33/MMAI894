# MMAI894
Please save the two datasets to the local repository:


Concrete Crack Images for Classification - our main dataset (approved) ("https://data.mendeley.com/datasets/5y9wdsg2zt/2")
-- Please download the dataset and save it in a local folder A


SDNET2018 - for transferrability ("https://digitalcommons.usu.edu/all_datasets/48/")
-- Please download the dataset and save it in the local folder A



Libraries required packages For CNN & Transfer Learning Models:

- matplotlib
- numpy
- torch
- torchvision
- tensorflow
- os
- cv2
- sklearn
- random
- collections
- tqdm
- skimgage
- scipy


Instructions for Mask R-CNN File:
1) Download Mask R-CNN folder and its entirety to your desktop
2) Set current working directory to the Mask R-CNN folder on your desktop
3) Code is located under the Mask R-CNN/Samples/Cracks/Mask R-CNN.py 
4) Ensure the following libraries are in your IDE environment


Mask R-CNN Libraries
1) Scikit Learn 0.21.3
2) Python 3.7
3) Tensorflow 1.14.0
4) Matplotlib 3.1.1
5) Keras 2.3.1
6) Numpy 1.17.2
7) Pandas 0.25.1
8) Scikit-Image 0.15.0


Description of each file:

1. Keras_Model_Benchmark
-- Keras Model functions as performance benchmark

2. model.pt
-- Saved Pytorch model with preprocessing method 1
-- Please save it to the local folder A

3. Model_Preprocessing_Method_1
-- Main Pytorch model with preprocessing method 1
-- data_dir = 'path to folder A'
-- model = torch.load('path to folder A')

4. model1.pt
-- Saved Pytorch model with preprocessing method 2
-- Please save it to the local folder A

5. Model_Preprocessing_Method_2
-- Main Pytorch model with preprocessing method 2
-- data_dir = 'path to folder A'
-- model = torch.load('path to folder A')

6. Model Structure
-- Architecture of both Pytroch models

7. Transfer_Learning_VGG16
-- Main Transfer Learning model VGG16

8. Transfer_Learning_AlexNet
-- Main Transfer Learning model AlexNet

9. Transfer_Learning_ResNet50
-- Main Transfer Learning model ResNet50

10. Mask R-CNN
-- Contains all the relevant code to run the Mask R-CNN Transfer Learning Code
