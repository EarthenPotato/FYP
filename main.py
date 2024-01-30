import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.optim as optim
import torchvision.models as models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data._utils.collate import default_collate
from PIL import Image

class PCBDataSet(Dataset):
    def __init__(self, root_dir, annotation_folders, image_folders, transform=None):
        self.root_dir = root_dir
        self.annotation_folders = annotation_folders
        self.image_folders = image_folders
        self.transform = transform
        self.annotations = self.load_annotations()
        self.images = self.load_images()
        self.data = self.combine_annotations_with_images()

    def load_annotations(self):
        annotations = {}
        for folder in self.annotation_folders:
            annotation_path = os.path.join(self.root_dir, folder)
            for file in os.listdir(annotation_path):
                if file.endswith('.txt'):
                    file_id = os.path.splitext(file)[0]
                    file_path = os.path.join(annotation_path, file)
                    with open(file_path, 'r') as f:
                        lines = f.readlines()
                        bbox_data = []
                        for line in lines:
                            x1, y1, x2, y2, damage_type = map(int, line.split())
                            bbox_data.append((x1, y1, x2, y2, damage_type))
                        annotations[file_id] = bbox_data
        return annotations

    def load_images(self):
        images = {}
        for image_folder in self.image_folders:
            image_path = os.path.join(self.root_dir, image_folder)
            for file in os.listdir(image_path):
                if file.endswith('_test.jpg') or file.endswith('_temp.jpg'):
                    file_id = file.split('_')[0]  # Get file identifier
                    if file_id not in images:
                        images[file_id] = {}
                    if file.endswith('_test.jpg'):
                        images[file_id]['test'] = os.path.join(image_path, file)
                    else:
                        images[file_id]['_temp'] = os.path.join(image_path, file)
        return images

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_id = list(self.data.keys())[idx]
        entry = self.data[file_id]
        annotations = entry['annotations']

        image_path = entry['images']['test']
        
        # Load the image with OpenCV
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

        # Convert the image from a NumPy array to a PIL image
        image = Image.fromarray(image)

        # Apply the transformations
        if self.transform:
            image = self.transform(image)  

        # Process annotations
        boxes = []
        labels = []
        for ann in annotations:
            x1, y1, x2, y2, damage_type = self.scale_annotations(ann, original_size=(640, 640), new_size=(640, 640))
            boxes.append([x1, y1, x2, y2])
            labels.append(damage_type)

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64)
        }

        return image, target


    
    def scale_annotations(self, annotation, original_size, new_size):
        # Scale the annotations according to the new image size
        x_scale = new_size[0] / original_size[1]
        y_scale = new_size[1] / original_size[0]
        x1, y1, x2, y2, damage_type = annotation
        scaled_annotation = [x1 * x_scale, y1 * y_scale, x2 * x_scale, y2 * y_scale, damage_type]
        return scaled_annotation


    def combine_annotations_with_images(self):
        data = {}
        for file_id in self.annotations:
            if file_id in self.images:
                # Create a new entry in the data dictionary for each file_id
                data[file_id] = {
                    'annotations': self.annotations[file_id],
                    'images': self.images[file_id]
                }
        # print(data)
        return data

def collate_fn(batch):
    # Separate images and targets
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    # Use default_collate for images
    images = default_collate(images)

    # Targets are already a list of tensors, so no need to stack
    return images, targets

def get_model(num_classes):
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

root_dir = 'C:/Users/DARYL/Schoolwork/FYP/Surface-Defect-Detection/DeepPCB/PCBData'
# groups = ["group00041", "group12000", "group12100", "group12300", "group13000", 
#           "group20085", "group44000", "group50600", "group77000", "group90100", "group92000"]

groups = ["group00041"]
annotation_folders = [os.path.join(group, group.split('group')[-1] + '_not') for group in groups]
image_folders = [os.path.join(group, group.split('group')[-1]) for group in groups]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToPILImage(),   # If your images are not PIL Images
    transforms.ToTensor(),     # Converts to Torch Tensor and scales to [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

pcb_dataset = PCBDataSet(root_dir, annotation_folders, image_folders)
data_loader = DataLoader(pcb_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)


model = get_model(num_classes=7).to(device)  
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {losses.item()}')


torch.save(model.state_dict(), 'pcb_defect_detection_model.pth')
