import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm  # For real-time progress bars
import os
import matplotlib.pyplot as plt

print(torch.cuda.is_available())
print(torch.cuda.device_count())  # Should return the number of GPUs detected
print(torch.cuda.get_device_name(0))

# Define data transformations for training, validation, and testing
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the dataset
data_dir = r'C:/Users/bhuva/Desktop/birds_data/train_data'
dataset = datasets.ImageFolder(data_dir, transform=train_transform)

# Split dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoaders
batch_size = 32  # Adjusted to optimize GPU usage
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Testing dataset
test_data = datasets.ImageFolder(r'C:/Users/bhuva/Desktop/birds_data/test_data', transform=test_transform)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Get number of classes (bird species)
num_classes = len(dataset.classes)
print(f'Number of bird species: {num_classes}')

# Load pre-trained ResNet50 model
model = models.resnet50(weights='IMAGENET1K_V1')

# Modify the final fully connected layer for bird species classification
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Using mixed precision training for faster computations and memory optimization
scaler = torch.cuda.amp.GradScaler()

# Checkpoint loading and saving
checkpoint_path = "bird_model_checkpoint.pth"

def save_checkpoint(epoch, model, optimizer, val_loss, checkpoint_path=checkpoint_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss
    }, checkpoint_path)
    print(f"Checkpoint saved after epoch {epoch + 1}.")

def load_checkpoint(model, optimizer, checkpoint_path=checkpoint_path):
    if os.path.exists(checkpoint_path):
        print("Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        val_loss = checkpoint['val_loss']
        print(f"Resuming from epoch {start_epoch}. Validation loss at checkpoint: {val_loss:.4f}")
        return start_epoch
    return 0  # Start from epoch 0 if no checkpoint is found

# Early stopping parameters
early_stopping_patience = 3
best_val_loss = float('inf')
epochs_no_improve = 0

# Training and validation loop with real-time progress display
num_epochs = 100  # Set number of epochs
start_epoch = load_checkpoint(model, optimizer)  # Load checkpoint if available

# Function to check for overfitting
def check_overfitting(train_losses, val_losses, train_accuracies, val_accuracies):
    """
    Indicates if the model is overfitting or training safely.
    """
    # Compare the latest validation loss and training loss
    if len(train_losses) > 1:
        if val_losses[-1] > val_losses[-2] and train_losses[-1] < train_losses[-2]:
            print("Warning: Overfitting detected! Validation loss is increasing while training loss is decreasing.")
        elif val_accuracies[-1] < val_accuracies[-2] and train_accuracies[-1] > train_accuracies[-2]:
            print("Warning: Overfitting detected! Validation accuracy is dropping while training accuracy is improving.")
        else:
            print("Model is training safely. No signs of overfitting yet.")
    else:
        print("Not enough data to assess overfitting yet.")



# Create lists to store the training and validation losses and accuracies
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(start_epoch, num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    print(f'Epoch {epoch + 1}/{num_epochs}')

    # Training loop with real-time progress bar
    train_loader_len = len(train_loader)  # Total number of batches
    progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}", total=train_loader_len)
    
    for batch_idx, (images, labels) in enumerate(progress_bar):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()  # Clear gradients
        with torch.cuda.amp.autocast():  # Mixed precision forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()  # Mixed precision backward pass
        scaler.step(optimizer)  # Optimizer step with scaling
        scaler.update()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)    #None
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

        # Update progress bar with current loss and accuracy
        progress_bar.set_postfix({
            'Loss': running_loss / (batch_idx + 1),
            'Accuracy': 100 * correct_train / total_train,
            'Progress': f'{batch_idx + 1}/{train_loader_len}'
        })

    train_accuracy = 100 * correct_train / total_train
    average_loss = running_loss / len(train_loader)
    train_losses.append(average_loss)
    train_accuracies.append(train_accuracy)
    print(f'Training Loss: {average_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%')

    # Validation loop with real-time progress bar
    model.eval()  # Set model to evaluation mode
    correct_val = 0
    total_val = 0
    val_loss = 0.0

    val_loader_len = len(val_loader)  # Total number of validation batches
    val_progress_bar = tqdm(val_loader, desc="Validating", total=val_loader_len)
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_progress_bar):
            images, labels = images.to(device), labels.to(device)
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

            # Update validation progress bar
            val_progress_bar.set_postfix({
                'Val Loss': val_loss / (batch_idx + 1),
                'Val Accuracy': 100 * correct_val / total_val
            })

    val_accuracy = 100 * correct_val / total_val
    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

    # Save the model checkpoint after each epoch
    save_checkpoint(epoch, model, optimizer, val_loss)

    # Check for overfitting after each epoch
    check_overfitting(train_losses, val_losses, train_accuracy, val_accuracy)

    # Early stopping logic
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    if epochs_no_improve == early_stopping_patience:
        print(f"Early stopping triggered after {epoch + 1} epochs.")
        break

'''
# Plot training and validation loss over epochs for visualization
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
'''


# Test the model accuracy
correct_test = 0
total_test = 0

model.eval()  # Set model to evaluation mode
test_loader_len = len(test_loader)
test_progress_bar = tqdm(test_loader, desc="Testing", total=test_loader_len)

with torch.no_grad():
    for batch_idx, (images, labels) in enumerate(test_progress_bar):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total_test += labels.size(0)
        correct_test += (predicted == labels).sum().item()

        # Update testing progress bar
        test_progress_bar.set_postfix({
            'Test Accuracy': 100 * correct_test / total_test
        })

test_accuracy = 100 * correct_test / total_test
print(f'Test Accuracy: {test_accuracy:.2f}%')                                                                                                        