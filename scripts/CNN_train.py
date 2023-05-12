#!/usr/bin/env python
# coding: utf-8

# <h3>Importing the Libraries

# In[1]:


import os
import cv2
import argparse
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import torchvision
import torchvision.transforms as transforms

from efficientnet_pytorch import EfficientNet


# <h3>Unzipping the Dataset

# In[2]:


# main_directory = r'C:\Users\Akhash\Documents\Fall 2021\Neural Networks\Homework 6'
# zipfile_location = main_directory + "/geometry_dataset.zip"
# extractedfile_location = main_directory + "/geometry_dataset/"


# # In[3]:


# #Unzips the Contents of File present in zipfile_location onto the folder extractedfile_location
# with zipfile.ZipFile(zipfile_location, 'r') as reference:
#     reference.extractall(extractedfile_location)


# <h3>Resizing the Images to Smaller Resolution Grayscale Images

# In[4]:


resize_folder = r"C:\Users\shari\Downloads\2D geometric shapes dataset\2D geometric shapes dataset\dataset\output"
for file_name in os.listdir(resize_folder):
    image_name = resize_folder + "/" + file_name
    #Reading the File as an Image
    color_image = cv2.imread(image_name)
    #Converting the Color Image to Grayscale Image
    grayscale_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    #Resizing the 200*200 Images to 100*100 Images
    resized_image = cv2.resize(grayscale_image, (100, 100))
    cv2.imwrite(image_name, resized_image)


# <h3>Function to Assign Class Labels to Images 

# In[5]:


def class_initiator():
    folder_path = r'C:\Users\shari\Downloads\2D geometric shapes dataset\2D geometric shapes dataset\dataset\output'
    file_array = []
    class_value = []
    for file_name in os.listdir(folder_path):
        file_array.append(file_name)
        if "Circle" in file_name:
            class_value.append(0)
        elif "Square" in file_name:
            class_value.append(1)
        elif "Octagon" in file_name:
            class_value.append(2)
        elif "Heptagon" in file_name:
            class_value.append(3)
        elif "Nonagon" in file_name:
            class_value.append(4)
        elif "Star" in file_name:
            class_value.append(5)
        elif "Hexagon" in file_name:
            class_value.append(6)
        elif "Pentagon" in file_name:
            class_value.append(7)
        elif "Triangle" in file_name:
            class_value.append(8)
    return file_array, class_value


# <h3>Creating CSV File for Dataset

# In[6]:


file_array, class_value = class_initiator()
class_dictionary = {
            'file_name': file_array,
            'file_class': class_value
          }

df = pd.DataFrame(class_dictionary)
df.to_csv(r'C:\Users\shari\Downloads\Geometric-Shapes-Identification-using-CNN-main\Geometric-Shapes-Identification-using-CNN-main\dataset.csv')


# <h3>Custom Dataset Class

# In[7]:


class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform = None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        img_path = os.path.join(str(self.root_dir), str(self.annotations.iloc[index, 1]))
        image = plt.imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 2]))
        
        if self.transform:
            image = self.transform(image)
        return (image, y_label)


# <h3>Using Dataloader function to create Train Loader and Test Loader

# In[8]:


root_directory = r"C:\Users\shari\Downloads\Geometric-Shapes-Identification-using-CNN-main\Geometric-Shapes-Identification-using-CNN-main"
dataset = CustomDataset(csv_file = "dataset.csv", root_dir = root_directory, transform = transforms.ToTensor())
batch_size = 500
#Split Dataset into Training and Testing Set
train_set, test_set = torch.utils.data.random_split(dataset, [72000, 18000])
train_loader = DataLoader(dataset = train_set, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(dataset = test_set, batch_size = batch_size, shuffle = True)


# <h3>Declaring Neural Networks Class using nn.Module

# In[9]:


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 5, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(10368, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, 128)
        self.fc4 = nn.Linear(128, 9)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)

        return x


# <h3>Function to Calculate Accuracy

# In[10]:


def calculate_accuracy(loader, model):
    correct_predictions = 0
    total_samples = 0
    model.eval()
    
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device=device)
            target = target.to(device=device)
            
            scores = model(data)
            _, predictions = scores.max(1)
            correct_predictions += (predictions == target).sum()
            total_samples += predictions.size(0)
        return float(correct_predictions)/(total_samples)*100
    model.train()


# <h3>Training the Model

# In[ ]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Net().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = (1e-3))

training_cost_array = []
testing_cost_array = []
training_accuracy_array = []
testing_accuracy_array = []
epoch_array = []

for epoch in range(1, 26):
    training_losses = []
    testing_losses = []
    
    for batch_idx, (train_data, train_targets) in enumerate(train_loader):
        train_data = train_data.to(device=device)
        train_targets = train_targets.to(device=device)
        training_score = model(train_data)
        training_loss = criterion(training_score, train_targets)
        training_losses.append(training_loss.item())
        optimizer.zero_grad()
        training_loss.backward()
        optimizer.step()
        
    with torch.no_grad():
        for test_data, test_target in test_loader:
            test_data = test_data.to(device=device)
            test_target = test_target.to(device=device)
            testing_score = model(test_data)
            testing_loss = criterion(testing_score, test_target)  # sum up batch loss
            testing_losses.append(testing_loss.item())
            
    train_accuracy = calculate_accuracy(train_loader, model)
    test_accuracy = calculate_accuracy(test_loader, model)
    
    printable_training_loss = sum(training_losses)/len(training_losses)
    printable_testing_loss = sum(testing_losses)/len(testing_losses)
    
    training_cost_array.append(printable_training_loss)
    testing_cost_array.append(printable_testing_loss)
    training_accuracy_array.append(train_accuracy)
    testing_accuracy_array.append(test_accuracy)
    epoch_array.append(epoch)
    print(f'| Epoch: {epoch} |  Training Cost: {round(printable_training_loss, 3)} | Testing Cost: {round(printable_testing_loss, 3)} | Training Accuracy: {round(train_accuracy, 3)}% | Testing Accuracy: {round(test_accuracy, 3)}% |')   


# <h3>Plotting Graphs

# In[ ]:


plt.plot(epoch_array, training_cost_array, color = 'red', label = "Training")
plt.plot(epoch_array, testing_cost_array, color = 'blue', label = "Testing")
plt.title("Variation of Cost with Epoch")
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.legend()
plt.show()


# In[ ]:


plt.plot(epoch_array, training_accuracy_array, color = 'red', label = "Training")
plt.plot(epoch_array, testing_accuracy_array, color = 'blue', label = "Testing")
plt.title("Variation of Accuracy with Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# <h3>Saving the Trained Model

# In[ ]:


torch.save(model.state_dict(), r"C:\Users\shari\Downloads\Geometric-Shapes-Identification-using-CNN-main\Geometric-Shapes-Identification-using-CNN-main\model.pt")

