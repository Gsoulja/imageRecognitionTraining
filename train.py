import os
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Set the MLflow tracking URI - uncomment and modify as needed
# mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Set the experiment name
mlflow.set_experiment("image-recognition-finetuning")

def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=5):
    """Train the model and return training history"""
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update statistics
            running_loss += loss.item() * inputs.size(0)
            train_bar.set_postfix(loss=loss.item())
            
        epoch_train_loss = running_loss / len(train_loader.dataset)
        history['train_loss'].append(epoch_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Valid]")
        
        with torch.no_grad():
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Update statistics
                val_loss += loss.item() * inputs.size(0)
                
                # Get predictions
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                val_bar.set_postfix(loss=loss.item())
        
        # Calculate epoch validation metrics
        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_accuracy = accuracy_score(all_labels, all_preds)
        
        history['val_loss'].append(epoch_val_loss)
        history['val_accuracy'].append(epoch_val_accuracy)
        
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {epoch_train_loss:.4f}, "
              f"Val Loss: {epoch_val_loss:.4f}, "
              f"Val Accuracy: {epoch_val_accuracy:.4f}")
    
    return history, all_preds, all_labels

def plot_training_history(history):
    """Plot the training and validation loss and accuracy"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot losses
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    
    # Plot accuracy
    ax2.plot(history['val_accuracy'], label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig("training_history.png")
    return fig

def plot_confusion_matrix(y_true, y_pred, classes):
    """Create a confusion matrix visualization"""
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("confusion_matrix.png")
    return plt.gcf()

def main():
    # Set hyperparameters
    params = {
        "model_name": "resnet18",  # Options: resnet18, resnet34, resnet50, vgg16, mobilenet_v2
        "pretrained": True,
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 5,
        "num_workers": 4,
        "optimizer": "Adam",  # Options: Adam, SGD
        "weight_decay": 1e-4,
        "momentum": 0.9,  # Only used if optimizer is SGD
        "freeze_backbone": True  # Whether to freeze the backbone layers
    }
    
    # Define data directories
    data_dir = "data"  # Replace with your data directory
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    
    # Set up data transformations
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    try:
        train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
        val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)
        
        # Get class names
        class_names = train_dataset.classes
        params["num_classes"] = len(class_names)
        
        print(f"Classes: {class_names}")
        print(f"Number of training samples: {len(train_dataset)}")
        print(f"Number of validation samples: {len(val_dataset)}")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=params["batch_size"],
            shuffle=True,
            num_workers=params["num_workers"],
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=params["batch_size"],
            shuffle=False,
            num_workers=params["num_workers"],
            pin_memory=True
        )
    except Exception as e:
        print(f"Error loading datasets: {e}")
        # If no data is available, we'll use a dummy dataset for demonstration
        print("Using dummy data for demonstration...")
        # Create dummy datasets with 10 classes and random data
        train_dataset = datasets.FakeData(
            size=1000,
            image_size=(3, 224, 224),
            num_classes=10,
            transform=train_transforms
        )
        val_dataset = datasets.FakeData(
            size=200,
            image_size=(3, 224, 224),
            num_classes=10,
            transform=val_transforms
        )
        
        class_names = [f"class_{i}" for i in range(10)]
        params["num_classes"] = 10
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=params["batch_size"],
            shuffle=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=params["batch_size"],
            shuffle=False
        )
    
    # Set device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params["device"] = str(device)
    print(f"Using device: {device}")
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"{params['model_name']}_finetuning"):
        # Log hyperparameters
        mlflow.log_params(params)
        
        # Get model architecture
        if params["model_name"] == "resnet18":
            model = models.resnet18(pretrained=params["pretrained"])
        elif params["model_name"] == "resnet34":
            model = models.resnet34(pretrained=params["pretrained"])
        elif params["model_name"] == "resnet50":
            model = models.resnet50(pretrained=params["pretrained"])
        elif params["model_name"] == "vgg16":
            model = models.vgg16(pretrained=params["pretrained"])
        elif params["model_name"] == "mobilenet_v2":
            model = models.mobilenet_v2(pretrained=params["pretrained"])
        else:
            raise ValueError(f"Unknown model name: {params['model_name']}")
        
        # Freeze backbone layers if specified
        if params["freeze_backbone"]:
            for param in model.parameters():
                param.requires_grad = False
        
        # Modify the final layer for our specific number of classes
        if params["model_name"].startswith("resnet"):
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, params["num_classes"])
        elif params["model_name"] == "vgg16":
            num_features = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_features, params["num_classes"])
        elif params["model_name"] == "mobilenet_v2":
            num_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_features, params["num_classes"])
        
        # Move model to the appropriate device
        model = model.to(device)
        
        # Define loss function
        criterion = nn.CrossEntropyLoss()
        
        # Define optimizer
        if params["optimizer"] == "Adam":
            optimizer = optim.Adam(
                [p for p in model.parameters() if p.requires_grad],
                lr=params["learning_rate"],
                weight_decay=params["weight_decay"]
            )
        elif params["optimizer"] == "SGD":
            optimizer = optim.SGD(
                [p for p in model.parameters() if p.requires_grad],
                lr=params["learning_rate"],
                momentum=params["momentum"],
                weight_decay=params["weight_decay"]
            )
        else:
            raise ValueError(f"Unknown optimizer: {params['optimizer']}")
        
        # Train the model
        print("Starting training...")
        history, final_preds, final_labels = train_model(
            model, 
            train_loader,
            val_loader,
            criterion,
            optimizer,
            device,
            epochs=params["epochs"]
        )
        
        # Calculate and log final metrics
        final_accuracy = accuracy_score(final_labels, final_preds)
        final_precision = precision_score(final_labels, final_preds, average='weighted', zero_division=0)
        final_recall = recall_score(final_labels, final_preds, average='weighted', zero_division=0)
        final_f1 = f1_score(final_labels, final_preds, average='weighted', zero_division=0)
        
        mlflow.log_metric("accuracy", final_accuracy)
        mlflow.log_metric("precision", final_precision)
        mlflow.log_metric("recall", final_recall)
        mlflow.log_metric("f1_score", final_f1)
        
        print(f"Final Metrics:")
        print(f"  Accuracy: {final_accuracy:.4f}")
        print(f"  Precision: {final_precision:.4f}")
        print(f"  Recall: {final_recall:.4f}")
        print(f"  F1 Score: {final_f1:.4f}")
        
        # Create and log visualizations
        # Training history plot
        history_fig = plot_training_history(history)
        mlflow.log_artifact("training_history.png")
        
        # Confusion matrix
        cm_fig = plot_confusion_matrix(final_labels, final_preds, class_names)
        mlflow.log_artifact("confusion_matrix.png")
        
        # Log the model
        mlflow.pytorch.log_model(model, "model")
        
        # Register the model in MLflow Model Registry
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        registered_model_name = f"image_classifier_{params['model_name']}"
        
        try:
            mlflow.register_model(model_uri, registered_model_name)
            print(f"Model registered as: {registered_model_name}")
        except Exception as e:
            print(f"Warning: Could not register model - {e}")
            
        print("Training complete!")

if __name__ == "__main__":
    main()
