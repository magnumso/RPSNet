# You can download the datasets here:

# Dataset-5
#from roboflow import Roboflow
#rf = Roboflow(api_key="5GziaYTm2SplmQeD3zAI")
#project = rf.workspace("workspace-gfwiw").project("rpc2-nle8g")
#version = project.version(22)
#dataset = version.download("yolov4scaled")


# Datast-4
#from roboflow import Roboflow
#rf = Roboflow(api_key="5GziaYTm2SplmQeD3zAI")
#project = rf.workspace("workspace-gfwiw").project("rpc2-nle8g")
#version = project.version(21)
#dataset = version.download("yolov4scaled")


# Dataset-3
#from roboflow import Roboflow
#rf = Roboflow(api_key="5GziaYTm2SplmQeD3zAI")
#project = rf.workspace("workspace-gfwiw").project("rpc2-nle8g")
#version = project.version(20)
#dataset = version.download("yolov4scaled")


# Dataset-2
#from roboflow import Roboflow
#rf = Roboflow(api_key="5GziaYTm2SplmQeD3zAI")
#project = rf.workspace("workspace-gfwiw").project("rpc2-nle8g")
#version = project.version(7)
#dataset = version.download("yolov4scaled")


# Dataset-1
#from roboflow import Roboflow
#rf = Roboflow(api_key="5GziaYTm2SplmQeD3zAI")
#project = rf.workspace("workspace-gfwiw").project("rpc2-nle8g")
#version = project.version(5)
#dataset = version.download("yolov4scaled")


import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder


class RPSDataset(Dataset):
    def __init__(self, img_dir, labels_dir):
        self.img_dir = img_dir
        self.labels_dir = labels_dir
        self.img_filenames = self.load_filenames()
        self.one_hot_encoder = OneHotEncoder(sparse_output=False)
        self.one_hot_encoder.fit(np.array([[0], [1], [2], [3]]))  # Assuming 4 classes including 'None'
        
    # Get files and account for both jpg and png
    def load_filenames(self):
        return [f for f in os.listdir(self.img_dir) if f.endswith('.jpg') or f.endswith('.png')]

    def __len__(self):
        return len(self.img_filenames)

    def __getitem__(self, idx):
        filename = self.img_filenames[idx]
        img_path = os.path.join(self.img_dir, filename)
        label_path = os.path.join(self.labels_dir, filename.replace('.jpg', '.txt').replace('.png', '.txt'))
        
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image {img_path} not found")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if not os.path.exists(label_path):
            label = 3  # Assign 'None' class if label file does not exist since images of backgounds have no labels so we create them
        else:
            with open(label_path, 'r') as f:
                label_line = f.readline().strip()
                if label_line:
                    label = int(label_line.split()[0])  # Assuming label is the first element in the line
                else:
                    label = 3  # Assign 'None' class if label file is blank

        # One-hot encode the class label
        one_hot_label = self.one_hot_encoder.transform([[label]])[0]

        # Convert image to torch tensor
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0 # Normalize iamges to be between 0-1
        one_hot_label = torch.tensor(one_hot_label, dtype=torch.float32)  # Ensure label is a float tensor for BCEWithLogitsLoss

        return image, one_hot_label

