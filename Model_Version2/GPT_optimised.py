import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Check CUDA availability
def check_cuda():
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))

# Define data transformations for training and testing
def get_data_transforms():
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

    return train_transform, test_transform

# Load datasets
def load_datasets(data_dir, test_data_dir, train_transform, batch_size):
    dataset = datasets.ImageFolder(data_dir, transform=train_transform)

    # Split dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Testing dataset
    test_data = datasets.ImageFolder(test_data_dir, transform=test_transform)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, len(dataset.classes)

# Initialize model
def initialize_model(num_classes):
    model = models.resnet50(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# Checkpoint functions
def save_checkpoint(epoch, model, optimizer, val_loss, checkpoint_path="bird_model_checkpoint.pth"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss
    }, checkpoint_path)
    print(f"Checkpoint saved after epoch {epoch + 1}.")

def load_checkpoint(model, optimizer, checkpoint_path="bird_model_checkpoint.pth"):
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

# Save metrics to CSV
def save_metrics(train_acc, val_acc, train_loss, val_loss, epoch):
    metrics = {
        'Epoch': list(range(1, epoch + 1))
    }

    # Ensure each metric list is of length `epoch`
    metrics['Train Accuracy'] = train_acc + [np.nan] * (epoch - len(train_acc))
    metrics['Validation Accuracy'] = val_acc + [np.nan] * (epoch - len(val_acc))
    metrics['Train Loss'] = train_loss + [np.nan] * (epoch - len(train_loss))
    metrics['Validation Loss'] = val_loss + [np.nan] * (epoch - len(val_loss))

    df = pd.DataFrame(metrics)
    df.to_csv('training_metrics.csv', index=False)
    print("Metrics saved to training_metrics.csv")


# Training and validation loop
# Training and validation loop
def train_and_validate(model, train_loader, val_loader, criterion, optimizer, scaler, num_epochs=100):
    train_losses = []
    val_losses = [] 
    train_accuracies= []
    val_accuracies = []
    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stopping_patience = 3
    start_epoch = load_checkpoint(model, optimizer)

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0
        print(f'Epoch {epoch + 1}/{num_epochs}')

        train_loader_len = len(train_loader)
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}", total=train_loader_len)

        for batch_idx, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(images)

                # Check for NaN values
                if torch.any(torch.isnan(outputs)):
                    print("NaN values in outputs.")
                    return None, None  # Return None if NaN found
                
                loss = criterion(outputs, labels)

                if torch.isnan(loss):
                    print("Loss is NaN.")
                    return None, None  # Return None if loss is NaN

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            progress_bar.set_postfix({
                'Loss': running_loss / (batch_idx + 1),
                'Accuracy': 100 * correct_train / total_train,
                'Progress': f'{batch_idx + 1}/{train_loader_len}'
            })

        # Store training metrics
        train_accuracy = 100 * correct_train / total_train
        average_loss = running_loss / len(train_loader)
        train_losses.append(average_loss)
        train_accuracies.append(train_accuracy)
        print(f'Training Loss: {average_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%')

        # Validation loop
        model.eval()
        correct_val = 0
        total_val = 0
        val_loss = 0.0

        val_loader_len = len(val_loader)
        val_progress_bar = tqdm(val_loader, desc="Validating", total=val_loader_len)

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(val_progress_bar):
                images, labels = images.to(device), labels.to(device)

                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    if torch.isnan(loss):
                        print("Validation Loss is NaN.")
                        return None, None  # Return None if loss is NaN

                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()

                val_progress_bar.set_postfix({
                    'Val Loss': val_loss / (batch_idx + 1),
                    'Val Accuracy': 100 * correct_val / total_val
                })

        # Store validation metrics
        val_accuracy = 100 * correct_val / total_val
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

        # Save checkpoint and metrics
        save_checkpoint(epoch, model, optimizer, val_loss)
        save_metrics(train_accuracies, val_accuracies, train_losses, val_losses, epoch)

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == early_stopping_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

    # Return the losses
    return train_losses, val_losses



# Test the model accuracy
def test_model(model, test_loader):
    correct_test, total_test = 0, 0
    model.eval()
    test_loader_len = len(test_loader)
    test_progress_bar = tqdm(test_loader, desc="Testing", total=test_loader_len)

    with torch.no_grad():
        for images, labels in test_progress_bar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
            test_progress_bar.set_postfix({'Test Accuracy': 100 * correct_test / total_test})

    test_accuracy = 100 * correct_test / total_test
    print(f'Test Accuracy: {test_accuracy:.2f}%')

# Main execution
if __name__ == "__main__":
    check_cuda()
    batch_size = 32
    data_dir = r'C:/Users/bhuva/Desktop/birds_data/train_data'
    test_data_dir = r'C:/Users/bhuva/Desktop/birds_data/test_data'
    train_transform, test_transform = get_data_transforms()
    train_loader, val_loader, test_loader, num_classes = load_datasets(data_dir, test_data_dir, train_transform, batch_size)
    model = initialize_model(num_classes)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Using mixed precision training
    scaler = torch.amp.GradScaler()

    # Train and validate the model
    train_losses, val_losses = train_and_validate(model, train_loader, val_loader, criterion, optimizer, scaler)

    # Test the model
    test_model(model, test_loader)
