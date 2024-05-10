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

class NiiDataset(Dataset):
    def __init__(self, df, image_type='MRI_PET', transform=None):
        """
        Initializes the dataset object.
        :param df: DataFrame containing file paths, labels, and subject IDs.
        :param image_type: Type of images to load ('MRI_PET', 'MRI', or 'PET').
        :param transform: A function or a series of transforms to apply to the images.
        """
        self.image_type = image_type
        if image_type == 'MRI_PET':
            self.paths = df['PATH_MRI_PET'].tolist()
        elif image_type == 'MRI':
            self.paths = df['PATH_MRI'].tolist()
        elif image_type == 'PET':
            self.paths = df['PATH_PET'].tolist()
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
        image = self.normalize_intensity(image)
        image = np.expand_dims(image, axis=0)  # Add a channel dimension
        return image

    @staticmethod
    def normalize_intensity(image):
        """
        Normalize the image data to zero mean and unit variance.
        """
        mean_intensity = np.mean(image)
        std_intensity = np.std(image)
        normalized_image = (image - mean_intensity) / std_intensity
        return normalized_image


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



def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=4):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    return train_loader, val_loader, test_loader


import torch
from tqdm import tqdm

def train_and_validate(model, train_loader, val_loader, criterion, optimizer, label_mapping, num_epochs=10, patience=5, device='cuda'):
    model.to(device)
    train_accuracies = []
    val_accuracies = []
    val_losses = []  # To store validation losses for monitoring
    best_val_loss = float('inf')
    best_val_accuracy = 0  # Initialize the best validation accuracy
    epochs_no_improve_loss = 0  # Counter for epochs with no improvement in loss
    epochs_no_improve_acc = 0  # Counter for epochs with no improvement in accuracy

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
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
        
        # Validation phase at the end of each epoch
        model.eval()  # Set model to evaluation mode
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
        val_losses.append(val_avg_loss)
        val_accuracy = 100 * val_correct / val_total
        val_accuracies.append(val_accuracy)
        print(f'Epoch {epoch+1}: Validation Loss: {val_avg_loss:.4f} - Validation Accuracy: {val_accuracy:.2f}%')
        
        # Early stopping logic based on loss
        if val_avg_loss < best_val_loss:
            best_val_loss = val_avg_loss
            epochs_no_improve_loss = 0
        else:
            epochs_no_improve_loss += 1
        
        # Early stopping logic based on accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_no_improve_acc = 0
        else:
            epochs_no_improve_acc += 1
        
        # Check for early stop condition
        if epochs_no_improve_loss >= patience or epochs_no_improve_acc >= patience:
            print(f'Early stopping triggered after {epoch + 1} epochs due to no improvement in validation loss or accuracy.')
            break  # Break out of the loop if no improvement for 'patience' consecutive epochs

    return train_accuracies, val_accuracies, val_losses




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





