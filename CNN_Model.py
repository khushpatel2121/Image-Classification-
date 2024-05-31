import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torchvision.models as models
from torch.optim.lr_scheduler import StepLR
import os


# Define data transforms for training and validation
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(64),
        transforms.RandomResizedCrop(64),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
        transforms.TrivialAugmentWide(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

# Define your data directories for training and validation
train_dir = '/Users/khushpatel/Desktop/E-yantra/task_2B/training'
val_dir = '/Users/khushpatel/Desktop/E-yantra/task_2B/testing'

# Create datasets using ImageFolder for training and validation
train_dataset = ImageFolder(train_dir, transform=data_transforms['train'])
val_dataset = ImageFolder(val_dir, transform=data_transforms['val'])

# Create DataLoader for training and validation
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Calculate the number of classes based on the dataset
class_names = ["combat", "humanitarianaid", "militaryvehicles", "fire", "destroyedbuilding"]
num_classes = 5

# Define the CNN model (you can use a pre-trained model like ResNet or VGG)
class CustomCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 16 * 16, 256),  # Adjust the input dimension based on the output size of the convolutional layers
            nn.ReLU(inplace=True),
            
            
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            
            
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Define loss function and optimizer
model = CustomCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# scheduler = StepLR(optimizer, step_size=1, gamma=0.05)  # Adjust step_size and gamma as needed


num_epochs = 50
# model.train()  # Set the model to training mode
print_every = 1
total_loss = 0.0
num_batches = len(train_loader)
# Training loop
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
                                                                                  
        if (batch_idx + 1) % print_every == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{num_batches}], Loss: {total_loss / print_every:.4f}')
            total_loss = 0.0
            
    # scheduler.step()  # Adjust learning rate

# Evaluation loop
model.eval()
class_correct = [0] * num_classes
class_total = [0] * num_classes
class_names = ["combat", "humanitarianaid", "militaryvehicles", "fire", "destroyedbuilding"]

with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total = labels.size(0)
        correct = (predicted == labels).squeeze()

        # Track correct predictions for each class
        for i in range(total):
            label = labels[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

# Calculate and report accuracy for each class
for i in range(num_classes):
    accuracy = 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
    print(f'Accuracy for class {class_names[i]}: {accuracy:.2f}%')

#print total and correct predictions
print(f'Total number of images: {sum(class_total)}')
print(f'Number of correct predictions: {sum(class_correct)}')


# Calculate overall accuracy
overall_accuracy = 100 * sum(class_correct) / sum(class_total)
print(f'Overall validation accuracy: {overall_accuracy:.2f}%')

model_path = '/Users/khushpatel/Desktop/E-yantra/task_2B/model.pth'

torch.save(model, model_path)
