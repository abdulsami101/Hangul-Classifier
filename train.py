import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np



# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
num_epochs = 50
batch_size = 2
learning_rate = 0.001

# Custom Dataset root directory
custom_data_root = './dataset/'

# Transformer configuration
transform = transforms.Compose([
    # transforms.Resize(size=(32,32)),
    transforms.Resize((256, 256), antialias=True),  # Add antialias=True
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Loading custom train dataset
custom_dataset = datasets.ImageFolder(root=os.path.join(custom_data_root, 'train_imgs'), transform=transform)


# Split the dataset into training and validation sets (80% train, 20% test)
train_size = 0.8 
train_dataset, validation_dataset = train_test_split(custom_dataset, train_size=train_size, test_size=1-train_size)

# Creating Dataloaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

# Define classes
Classes = custom_dataset.classes
print(Classes)
# Implement ConvNet

class AlexNet(nn.Module):
    def __init__(self, num_classes=38):
        super(AlexNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(9216, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(4096, num_classes))
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


#main method
if __name__ == "__main__":

    model = AlexNet().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    n_total_steps = len(train_loader)
    print(n_total_steps)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):

            # Move tensors to the configured device  
            images = images.to(device)
            labels = labels.to(device)

            # Forward Pass 
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward Pass and Optimize 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            
            print(f'EPOCH [{epoch + 1} / {num_epochs}], STEP [{i + 1} / {n_total_steps}], LOSS = {loss.item():.4f}')

    print('Finished Training')

    # Saving the Model
    FILE = "hangul_font.pth"
    torch.save(model.state_dict(), FILE)