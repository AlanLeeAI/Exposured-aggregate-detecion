# Exposured-aggregate-detecion
This repository is using for Exposed aggregate recognize and semantic segmentation.The functions included are as follows:

•	Training and testing modes

•	Data augmentation

•	Able to use other dataset

•	Evaluation including precision, recall, f1 score, average accuracy, per-class accuracy, and mean IoU

•	Plotting of loss function precision, recall, f1 score, average accuracy and mean IoU over epochs

# Files and Directories
train.py: Training on the dataset of your choice. 

test.py: Testing on the dataset of your choice. 

Utils/helper.py: Quick helper functions for data preparation and visualization

Utils/utils.py: Utilities for printing, debugging, testing, and evaluation

Utils/unet.py: network model files. 

checkpoints: Checkpoint files for each epoch during training

# Usage
The only thing you have to do to get started is set up the folders in the following structure:

├── "dataset_name"        

|   ├── train

|   ├── train_labels

|   ├── val

|   ├── val_labels

|   ├── test

|   ├── test_labels

Put a text file under the dataset directory called "class_dict.csv" which contains the list of classes along with the R, G, B colour labels to visualize the segmentation results. This kind of dictionairy is usually supplied with the dataset. 

Here is an example for the Exposed aggregate dataset:

name,r,g,b

rock,128,0,0

background,0,0,0


# Citing
https://github.com/GeorgeSeif/Semantic-Segmentation-Suite





