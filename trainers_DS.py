import os
import numpy as np
import glob
import PIL.Image as Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import wandb
from IDLCV_VC.models_DS import save_checkpoint

from sklearn.metrics import precision_score, recall_score, f1_score,confusion_matrix

if torch.cuda.is_available():
    print("The code will run on GPU.")
else:
    print("The code will run on CPU. Go to Edit->Notebook Settings and choose GPU as the hardware accelerator")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_one_epoch(model, optimizer, criterion, loader, device):
    model.train()
    total_loss = 0.0
    correct = 0

    for iteration, (image_data, flow_data, target) in enumerate(loader):
        # print(iteration)
        # Move data and target to the specified device
        # data, target = data.to(device), target.to(device).float().unsqueeze(1)
        image_data, flow_data, target = image_data.to(device), flow_data.to(device), target.to(device).long()  # Target should be long for CrossEntropyLoss

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(image_data, flow_data)

        # Compute loss
        loss = criterion(output, target)

        # Backward pass and optimizer step
        loss.backward()
        optimizer.step()

        # Accumulate loss
        total_loss += loss.item()

        # Compute predictions and update correct count
        predicted = torch.argmax(output, dim=1)  # Multi-class prediction
        correct += (target == predicted).sum().item()

    # Compute average loss and accuracy
    avg_loss = total_loss / len(loader)
    accuracy = correct / len(loader.dataset)
    

    return avg_loss, accuracy


def evaluate(model, criterion, loader, device):
    model.eval()
    total_loss = 0.0
    correct = 0

    true_labels = []
    pred_labels = []
    with torch.no_grad():  # Disable gradient computation for evaluation
        for image_data, flow_data, target in loader:
            # Move data and target to the specified device
            image_data, flow_data, target = image_data.to(device), flow_data.to(device), target.to(device).long()

            # Forward pass
            output = model(image_data, flow_data)

            # Compute loss*[torch.tensor(1)]
            loss = criterion(output, target)

            # Accumulate loss
            total_loss += loss.item()

            # Compute predictions
            predicted = torch.argmax(output, dim=1)  # Multi-class prediction
            correct += (target == predicted).sum().item()


            # Collect true and predicted labels
            true_labels.extend(target.cpu().numpy())
            pred_labels.extend(predicted.cpu().numpy())
    # Compute metrics
    avg_loss = total_loss / len(loader)
    accuracy = correct / len(loader.dataset)
    precision = precision_score(true_labels, pred_labels, average='macro')
    recall = recall_score(true_labels, pred_labels, average='macro')
    f1 = f1_score(true_labels, pred_labels, average='macro')
    conf_matrix = confusion_matrix(true_labels, pred_labels)

    return avg_loss, accuracy, precision, recall, f1, conf_matrix



def _train_every_frame(model, optimizer, criterion, train_loader, validation_loader, test_loader, trainset, validationset, testset, num_epochs=20, run_id=""):
    out_dict = {
        'train_acc': [],
        'validation_acc': [],
        'test_acc': [],
        'train_loss': [],
        'validation_loss': [],
    }
    previous_val_acc = 1000000

    for epoch in tqdm(range(num_epochs), unit='epoch'):
        # Training
        train_loss, train_acc = train_one_epoch(model, optimizer, criterion, train_loader, device)

        # Validation
        val_loss, val_acc, val_precision, val_recall, val_f1, val_conf_matrix = evaluate(model, criterion, validation_loader, device)

        # Testing (optional)
        # _, test_acc = evaluate(model, criterion, test_loader, device)
        _, test_acc, test_precision, test_recall, test_f1, test_conf_matrix = evaluate(model, criterion, test_loader, device)

        # Log metrics
        out_dict['train_acc'].append(train_acc)
        out_dict['validation_acc'].append(val_acc)
        out_dict['test_acc'].append(test_acc)
        out_dict['train_loss'].append(train_loss)
        out_dict['validation_loss'].append(val_loss)

        if wandb.run:
            wandb.log({
                "train_acc": train_acc,
                "validation_acc": val_acc,
                "test_acc": test_acc,
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "epoch": epoch,
                "run_id": run_id,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "val_f1": val_f1,
                "val_confusion_matrix": val_conf_matrix,
                "test_precision": test_precision,
                "test_recall": test_recall,
                "test_f1": test_f1,
                "test_confusion_matrix": test_conf_matrix
            })

        # Save model checkpoint
        if previous_val_acc > val_acc:
            save_dir = "./checkpoints"
            save_checkpoint(model, optimizer, epoch, f"{save_dir}/checkpoint_{run_id}.pth")

        previous_val_acc = val_acc
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss: {train_loss:.3f}, Val Loss: {val_loss:.3f}, "
              f"Train Acc: {train_acc*100:.1f}%, Val Acc: {val_acc*100:.1f}%")

    return out_dict
