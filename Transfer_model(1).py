import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torchvision.models as models


# Define data transforms for training and validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0), ratio=(1.0, 1.0)),
        transforms.RandomCrop(224, padding = 2),
        transforms.RandomRotation(35),
        transforms.ColorJitter(brightness=(0.5, 1.8), contrast=1, saturation=0.5, hue=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

# Define your data directories for training and validation
train_dir = '/Users/khushpatel/Desktop/E-yantra/task_2B/training'
val_dir = '/Users/khushpatel/Desktop/E-yantra/task_2B/testing'

# Create datasets using ImageFolder for training and validation
train_dataset = ImageFolder(train_dir, transform=data_transforms['train'])
val_dataset = ImageFolder(val_dir, transform=data_transforms['val'])

# Create DataLoader for training and validation
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Calculate the number of classes based on the dataset
class_names = ["combat", "humanitarianaid", "militaryvehicles", "fire", "destroyedbuilding"]
num_classes = 5

# Define the CNN model (you can use a pre-trained model like ResNet or VGG)
model = models.resnet18(pretrained=True)  # Example: ResNet-18

# Modify the last fully connected layer to match the number of classes
num_ftrs = model.fc.in_features
print(num_ftrs)
model.fc = nn.Linear(num_ftrs, num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001,weight_decay=0.0001)

# Train the model
num_epochs = 10

for epoch in range(num_epochs):
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        corrects = 0

        # Track class-wise accuracy
        class_corrects = [0] * num_classes
        class_totals = [0] * num_classes

        data_loader = train_loader if phase == 'train' else val_loader

        for inputs, labels in data_loader:
            optimizer.zero_grad()
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

            if phase == 'train':
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            corrects += torch.sum(preds == labels.data)

            # Update class-wise accuracy
            for i in range(len(labels)):
                class_totals[labels[i]] += 1
                if labels[i] == preds[i]:
                    class_corrects[labels[i]] += 1

        epoch_loss = running_loss / len(data_loader.dataset)
        epoch_acc = corrects.double() / len(data_loader.dataset)

        print(f'Phase: {phase}, Epoch: {epoch + 1}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')

        # Calculate and print class-wise accuracy
        class_accuracy = [class_corrects[i] / class_totals[i] for i in range(num_classes)]
        print(f'Class-wise accuracy in {phase}: {class_accuracy}')
        
total_correct = sum(class_corrects)
total_samples = sum(class_totals)
total_accuracy = total_correct / total_samples * 100
print(f'Total Accuracy: {total_accuracy:.2f}%')

print('Training complete.')

# Save the trained model
torch.save(model.state_dict(), 'model.pth')
