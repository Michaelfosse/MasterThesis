import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
import nibabel as nib
import pandas as pd
from torchvision import transforms
from tqdm import tqdm
import os
import datetime
import pandas as pd
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
import random

# Set seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Ensure deterministic behavior in PyTorch (may impact performance)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class NiiDataset(Dataset):
    def __init__(self, df, image_type='MRI_PET', transform=None):
        """
        Initializes the dataset object.
        :param df: DataFrame containing file paths, labels, and subject IDs.
        :param image_type: Type of images to load. This should directly correspond to the column names in the DataFrame.
        :param transform: A function or a series of transforms to apply to the images.
        """
        self.image_type = image_type
        # Dictionary mapping the image types to DataFrame column names, directly using the column names as keys
        column_mapping = {
            'Co-registered PET': 'Co-registered PET',
            'Fused Images': 'Fused Images',
            'Masked PET': 'Masked PET',
            'Spatial Normalization': 'Spatial Normalization',
            'Resampled Images(Co-registered PET)': 'Resampled Images(Co-registered PET)',
            'Resampled Images(Masked PET)': 'Resampled Images(Masked PET)',
            'Resampled Images(Spatial Normalization)': 'Resampled Images(Spatial Normalization)',
            'Resampled Images_fused': 'Resampled Images_fused'
        }

        # Check if the image type exists in the mapping and assign paths
        if image_type in column_mapping:
            self.paths = df[column_mapping[image_type]].tolist()
        else:
            raise ValueError(f"Unknown image type: {image_type}")
        
        self.labels = pd.Categorical(df['Research Group']).codes
        self.subjects = df['Subject'].tolist()
        self.transform = transform

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.paths)

    def __getitem__(self, idx):
        """
        Retrieve the nth sample from the dataset.
        """
        path = self.paths[idx]
        image = self.load_nii(path)
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        subject = self.subjects[idx]
        return image, label, path, subject

    def load_nii(self, path):
        """
        Load a NIfTI file and normalize its intensity.
        """
        image = nib.load(path).get_fdata(dtype=np.float32)
        image = np.expand_dims(image, axis=0)  # Add a channel dimension
        return image




def load_datasets(df, image_type, sample_size=None):
    train_df = df[df['dataset_split'] == 'train']
    val_df = df[df['dataset_split'] == 'validation']
    test_df = df[df['dataset_split'] == 'test']

    # If sample_size is specified, randomly select 'sample_size' samples from train and validation sets
    if sample_size is not None:
        # Ensure there are enough samples in each subset to sample from
        if len(train_df) >= sample_size:
            train_df = train_df.sample(sample_size, random_state=42)  # For reproducibility
        if len(val_df) >= sample_size:
            val_df = val_df.sample(sample_size, random_state=42)

    # Creating dataset objects
    train_dataset = NiiDataset(train_df, image_type=image_type)
    val_dataset = NiiDataset(val_df, image_type=image_type)
    test_dataset = NiiDataset(test_df, image_type=image_type)
    
    return train_dataset, val_dataset, test_dataset



# Function to initialize the workers of the DataLoader
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# Create data loaders with seed settings for reproducibility
def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=4, num_workers=0):
    # Creating a generator for seeding (Recommended for PyTorch >= 1.6)
    g = torch.Generator()
    g.manual_seed(42)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              drop_last=True, num_workers=num_workers, worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            drop_last=True, num_workers=num_workers, worker_init_fn=seed_worker, generator=g)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             drop_last=True, num_workers=num_workers, worker_init_fn=seed_worker, generator=g)
    
    return train_loader, val_loader, test_loader



def train_and_validate(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, patience=5, device='cuda'):
    model.to(device)
    train_accuracies = []
    val_accuracies = []
    val_losses = []
    training_summary = []
    
    best_val_loss = float('inf')
    best_val_accuracy = 0
    epochs_no_improve_loss = 0
    epochs_no_improve_acc = 0

    for epoch in range(num_epochs):
        model.train()
        train_correct = 0
        train_total = 0
        train_epoch_losses = []

        for images, labels, _, _ in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Train'):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_epoch_losses.append(loss.item())

            _, predicted_indices = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted_indices == labels).sum().item()

        train_avg_loss = sum(train_epoch_losses) / len(train_epoch_losses)
        train_accuracy = 100 * train_correct / train_total
        train_accuracies.append(train_accuracy)
        print(f'Epoch {epoch+1}: Train Loss: {train_avg_loss:.4f} - Train Accuracy: {train_accuracy:.2f}%')

        model.eval()
        val_correct = 0
        val_total = 0
        val_epoch_losses = []
        with torch.no_grad():
            for images, labels, _, _ in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Validate'):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss = criterion(outputs, labels)
                val_epoch_losses.append(val_loss.item())
                _, predicted_indices = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted_indices == labels).sum().item()

        val_avg_loss = sum(val_epoch_losses) / len(val_epoch_losses)
        val_accuracy = 100 * val_correct / val_total
        val_accuracies.append(val_accuracy)
        val_losses.append(val_avg_loss)
        training_summary.append((epoch+1, val_avg_loss, val_accuracy))
        print(f'Epoch {epoch+1}: Validation Loss: {val_avg_loss:.4f} - Validation Accuracy: {val_accuracy:.2f}%')

        if val_avg_loss < best_val_loss:
            best_val_loss = val_avg_loss
            epochs_no_improve_loss = 0  # Reset counter on improvement
        else:
            epochs_no_improve_loss += 1

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_no_improve_acc = 0  # Reset counter on improvement
        else:
            epochs_no_improve_acc += 1

        if epochs_no_improve_loss >= patience or epochs_no_improve_acc >= patience:
            print(f'Early stopping triggered after {epoch + 1} epochs due to no improvement in validation loss or accuracy.')
            break

    results_df = pd.DataFrame(training_summary, columns=['Epoch', 'Validation Loss', 'Validation Accuracy'])
    return train_accuracies, val_accuracies, val_losses, results_df


# Example usage:
# Assuming model, train_loader, val_loader, criterion, and optimizer are defined
# train_acc, val_acc, val_loss, results_df = train_and_validate(model, train_loader, val_loader, criterion, optimizer)
# print(results_df)



def test_model(model, test_loader, label_mapping, device='cuda'):
    model.to(device)
    model.eval()
    test_results = []
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels, paths, subjects in tqdm(test_loader, desc='Testing'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted_indices = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted_indices == labels).sum().item()
            predicted_labels = [label_mapping[code] for code in predicted_indices.cpu().numpy()]

            for label, pred, path, subject in zip(labels.cpu().numpy(), predicted_labels, paths, subjects):
                test_results.append({
                    'Subject': subject,
                    'Path': path,
                    'Actual Label': label_mapping[label.item()],
                    'Prediction': pred,
                    'Type': 'Test'
                })
    accuracy = 100 * correct / total
    return test_results, accuracy





