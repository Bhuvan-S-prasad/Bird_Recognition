import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Define data transformations for training and testing
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),   
    transforms.RandomHorizontalFlip(),     # Randomly flip images horizontally
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),  
    transforms.ToTensor(),                 # Convert image to PyTorch Tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize using ImageNet values
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),           # Center crop for test data
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load training and testing data using ImageFolder
train_data = datasets.ImageFolder(r'C:/Users/bhuva/Desktop/birds_data/train_data', transform=train_transform)
test_data = datasets.ImageFolder(r'C:/Users/bhuva/Desktop/birds_data/test_data', transform=test_transform)

# Create DataLoaders to load data in batches
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Get number of classes (bird species)
num_classes = len(train_data.classes)
print(f'Number of bird species: {num_classes}')

# Load pre-trained ResNet50 model
model = models.resnet50(weights='IMAGENET1K_V1')

# Modify the final fully connected layer to match the number of bird species (num_classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Model training
num_epochs = 2
early_stopping_threshold = 0.01
previous_loss = float('inf')

for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)  # Move data to GPU
        
        optimizer.zero_grad()  # Clear gradients from the last step
        outputs = model(images)  # Forward pass: compute predictions
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backward pass: compute gradients
        optimizer.step()  # Update model parameters
        
        running_loss += loss.item()

    average_loss = running_loss / len(train_loader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {average_loss:.4f}')

    if abs(previous_loss - average_loss) < early_stopping_threshold:
        print("Stopping early due to no significant improvement.")
        break

    previous_loss = average_loss

    # Save the model, overwriting the previous version
    torch.save(model.state_dict(), 'bird_model.pth')

    torch.save(model, 'bird_model.pth')


    print(f'Model saved after epoch {epoch + 1}.')


'''
# Evaluation
model.eval()  # Set model to evaluation mode
correct = 0
total = 0

with torch.no_grad():  # No need to compute gradients during evaluation
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)  # Move to GPU
        outputs = model(images)  # Forward pass
        _, predicted = torch.max(outputs.data, 1)  # Get the predicted class
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Test Accuracy: {100 * correct / total:.2f}%') 

'''


#Rectified Linear Unit-->After the convolution, the result passes through a function called ReLU. This removes any negative values by turning them into zeros.