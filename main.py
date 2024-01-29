import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class PCBDataSet(Dataset):
    def __init__(self, root_dir, annotation_folders, image_folders, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            annotation_folders (list): List of folders where annotations are stored.
            image_folders (list): List of folders where images are stored.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.annotation_folders = annotation_folders
        self.image_folders = image_folders
        self.transform = transform
        self.annotations = self.load_annotations()

    def load_annotations(self):
            file_count = 0 
            for folder, image_folder in zip(self.annotation_folders, self.image_folders):
                annotation_path = os.path.join(self.root_dir, folder)
                for file in os.listdir(annotation_path):
                    if file.endswith('.txt'):
                        # if file_count >= 10:  # Check files processed 
                        #     break 
                        file_path = os.path.join(annotation_path, file)
                        print(f"Annotation file path: {file_path}")  # Print annotation file path
                        # file_count += 1

                        # Path for the damaged image (test.jpg)
                        image_name_test = file.replace('_not', '').replace('.txt', '_test.jpg')
                        image_path_test = os.path.join(self.root_dir, image_folder, image_name_test)
                        print(f"Damaged Image file path: {image_path_test}")  # Print damaged image file path

                        # Path for the undamaged image (temp.jpg)
                        image_name_temp = file.replace('_not', '').replace('.txt', '_temp.jpg')
                        image_path_temp = os.path.join(self.root_dir, image_folder, image_name_temp)
                        print(f"Undamaged Image file path: {image_path_temp}")  # Print undamaged image file path


    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        image_path, bbox = self.annotations[idx]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # more processing can be added here
        if self.transform:
            image = self.transform(image)
        return image, bbox


root_dir = 'C:/Users/DARYL/Schoolwork/FYP/Surface-Defect-Detection/DeepPCB/PCBData'
groups = ["group00041", "group12000", "group12100", "group12300", "group13000", 
          "group20085", "group44000", "group50600", "group77000", "group90100", "group92000"]
annotation_folders = [os.path.join(group, group.split('group')[-1] + '_not') for group in groups]
image_folders = [os.path.join(group, group.split('group')[-1]) for group in groups]

pcb_dataset = PCBDataSet(root_dir, annotation_folders, image_folders)
data_loader = DataLoader(pcb_dataset, batch_size=4, shuffle=True)

for i, (image, bbox) in enumerate(data_loader):
    print(f"Sample {i+1}:")
    print("Image shape:", image.shape)
    print("Bounding Box:", bbox)
    # Print only the first few samples
    if i == 2:
        break