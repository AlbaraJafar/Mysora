import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import matplotlib.pyplot as plt
import copy
from sklearn.metrics import classification_report
from PIL import Image, ImageFile

# Ensure that truncated images are loaded properly
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Set device to MPS (Metal Performance Shaders) if available, otherwise CPU
if torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

print(f'Using device: {device}')

# Data augmentation for the training set (no rotation or flipping)
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomApply([transforms.ColorJitter(
        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)], p=0.5),
    transforms.RandomGrayscale(p=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1),
                            scale=(0.9, 1.1), shear=10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Transforms for validation and test sets
val_test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Custom dataset class to apply different transforms to subsets
class SubsetDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        try:
            x, y = self.subset[index]
            if self.transform:
                x = self.transform(x)
            return x, y
        except Exception as e:
            # Handle any exceptions (e.g., corrupted images)
            print(f'Error loading image at index {index}: {e}')
            # Return a zero tensor and -1 label (will be filtered out)
            return torch.zeros(3, 224, 224), -1

    def __len__(self):
        return len(self.subset)

# Main execution block
if __name__ == '__main__':
    # Load the dataset
    data_dir = '/Users/wolf7031/Documents/Quran/RGB ArSL dataset'  # Update with your data directory

    # Filter out any corrupted images during dataset creation
    def is_valid_file(path):
        try:
            Image.open(path).verify()
            return True
        except Exception:
            return False

    dataset = datasets.ImageFolder(root=data_dir, is_valid_file=is_valid_file)

    # Split dataset into train (70%), validation (15%), and test (15%)
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )

    # Apply transforms to the datasets
    train_dataset = SubsetDataset(train_dataset, transform=train_transform)
    val_dataset = SubsetDataset(val_dataset, transform=val_test_transform)
    test_dataset = SubsetDataset(test_dataset, transform=val_test_transform)

    # Create data loaders
    batch_size = 500
    num_workers = 0  # Set to 0 to avoid multiprocessing issues on macOS
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)

    # Load the pretrained ResNet50 model
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Freeze early layers to prevent overfitting
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last few layers
    for param in list(model.parameters())[-10:]:
        param.requires_grad = True

    # Modify the final layer to output 31 classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 31)

    # Move the model to the appropriate device
    model = model.to(device)

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Initialize optimizer and scheduler
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)

    # Initialize variables to track training progress
    num_epochs = 500  # Adjust as needed
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    early_stopping_patience = 7
    epochs_no_improve = 0
    start_epoch = 0  # Start from epoch 0

    # Path to your saved model or checkpoint
    checkpoint_path = '/Users/wolf7031/Documents/Quran/best_checkpoint.pth'

    # Check if a checkpoint exists and load it
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        best_val_loss = checkpoint['best_val_loss']
        epochs_no_improve = checkpoint['epochs_no_improve']
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        train_accuracies = checkpoint['train_accuracies']
        val_accuracies = checkpoint['val_accuracies']
        start_epoch = checkpoint['epoch'] + 1  # Start from the next epoch
        print(f"Loaded checkpoint from '{checkpoint_path}' at epoch {start_epoch}")
    else:
        # Check if only model weights are saved
        model_weights_path = '/Users/wolf7031/Documents/Quran/best_model_mps.pth'
        if os.path.exists(model_weights_path):
            model.load_state_dict(torch.load(model_weights_path, map_location=device))
            print(f"Loaded model weights from '{model_weights_path}'")
            # Optimizer and scheduler states are reinitialized
            start_epoch = 0
        else:
            print("No checkpoint or model weights found, starting training from scratch")
            start_epoch = 0

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch consists of a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0
            samples = 0  # To account for any samples skipped due to errors

            # Progress bar for the current phase
            pbar = tqdm(dataloader, desc=f'{phase} Epoch {epoch+1}', ncols=80)
            for inputs, labels in pbar:
                # Filter out samples with label -1 (from corrupted images)
                mask = labels != -1
                inputs = inputs[mask]
                labels = labels[mask]
                if labels.size(0) == 0:
                    continue  # Skip if no valid samples

                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass and optimization
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Update running loss and correct predictions
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                samples += inputs.size(0)
                # Calculate batch accuracy
                batch_acc = torch.sum(preds == labels.data).float() / labels.size(0)
                # Update progress bar with loss and accuracy
                pbar.set_postfix({'loss': loss.item(), 'acc': batch_acc.item()})

            # Calculate epoch loss and accuracy
            epoch_loss = running_loss / samples
            epoch_acc = running_corrects.float() / samples

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Store losses and accuracies for plotting
            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accuracies.append(epoch_acc.item())
                scheduler.step()
            else:
                val_losses.append(epoch_loss)
                val_accuracies.append(epoch_acc.item())

                # Check for improvement
                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    # Save the checkpoint
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_val_loss': best_val_loss,
                        'epochs_no_improve': epochs_no_improve,
                        'train_losses': train_losses,
                        'val_losses': val_losses,
                        'train_accuracies': train_accuracies,
                        'val_accuracies': val_accuracies,
                    }, checkpoint_path)
                    print(f"Checkpoint saved at '{checkpoint_path}'")
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                # Early stopping
                if epochs_no_improve >= early_stopping_patience:
                    print('Early stopping!')
                    break

        else:
            continue  # only executed if the inner loop did NOT break
        break  # inner loop did break, so we break the outer

    # Plot training and validation losses
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot training and validation accuracies
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Load the best model weights
    # Load from the best checkpoint
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from '{checkpoint_path}' for evaluation")
    else:
        print("No checkpoint found for evaluation, using current model")

    # Evaluate the model on the test set
    model.eval()
    test_loss = 0.0
    test_corrects = 0
    test_samples = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Testing', ncols=80):
            # Filter out samples with label -1 (from corrupted images)
            mask = labels != -1
            inputs = inputs[mask]
            labels = labels[mask]
            if labels.size(0) == 0:
                continue  # Skip if no valid samples

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * inputs.size(0)
            test_corrects += torch.sum(preds == labels.data)
            test_samples += inputs.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_loss = test_loss / test_samples
    test_acc = test_corrects.float() / test_samples

    print(f'Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}')

    # Generate classification report
    print('Classification Report:')
    print(classification_report(all_labels, all_preds, target_names=dataset.classes))