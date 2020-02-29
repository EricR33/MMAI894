# MMAI894
Please save the two datasets to the local repository:


Concrete Crack Images for Classification - our main dataset (approved) ("https://data.mendeley.com/datasets/5y9wdsg2zt/2")
-- Please download the dataset and save it in a local folder A


SDNET2018 - for transferrability ("https://digitalcommons.usu.edu/all_datasets/48/")
-- Please download the dataset and save it in the local folder A



Python required packages:

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
