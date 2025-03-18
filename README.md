# Fine-tuning an Off-the-Shelf Image Recognition Model with MLflow

This guide walks you through the process of fine-tuning a pre-trained image recognition model using MLflow to track your experiments.

## Project Setup

First, let's set up the project structure:

```
image-recognition-project/
├── data/
│   ├── train/
│   │   ├── class_1/
│   │   ├── class_2/
│   │   └── ...
│   └── val/
│       ├── class_1/
│       ├── class_2/
│       └── ...
├── scripts/
│   └── train.py
├── requirements.txt
└── README.md
```

## Installing Dependencies

Create a `requirements.txt` file with the following dependencies:

```
torch>=1.8.0
torchvision>=0.9.0
mlflow>=1.15.0
scikit-learn>=0.24.1
matplotlib>=3.3.4
numpy>=1.19.5
pandas>=1.2.3
tqdm>=4.59.0
Pillow>=8.2.0
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Preparing Your Dataset

For image recognition, your data should be organized in the following structure:

```
data/
├── train/
│   ├── class_1/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── class_2/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── ...
└── val/
    ├── class_1/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── class_2/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    └── ...
```

Each class should have its own folder, with all images of that class inside. This structure is compatible with PyTorch's `ImageFolder` dataset.

## Training Script

I've created a comprehensive training script (`train.py`) that you can find in the "Fine-tuning an Image Recognition Model with MLflow" artifact. This script:

1. Loads a pre-trained model (ResNet, VGG, or MobileNet)
2. Fine-tunes it on your custom dataset
3. Tracks the entire process with MLflow

## Running the Training

To train the model:

```bash
python scripts/train.py
```

By default, this will:
- Use ResNet18 as the base model
- Freeze the backbone layers
- Train only the classifier layers for 5 epochs
- Use the Adam optimizer with a learning rate of 0.001

## Customizing the Training Process

You can modify the hyperparameters in the `params` dictionary in the script:

```python
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
```

## Viewing Results in MLflow

Start the MLflow UI to view your experiment results:

```bash
mlflow ui
```

Then open your browser and navigate to [http://localhost:5000](http://localhost:5000) to see your experiment runs.

## Advanced Customization Options

### Unfreezing Layers

To fine-tune deeper layers of the model, you can set `freeze_backbone` to `False`:

```python
params["freeze_backbone"] = False
```

This will allow all parameters to be updated during training, which can lead to better performance but may require more data and training time.

### Using Different Pre-trained Models

The script supports several popular architectures. Change the model by updating:

```python
params["model_name"] = "resnet50"  # Or another supported model
```

### Learning Rate Scheduling

For better convergence, you can add a learning rate scheduler. Add the following to the script:

```python
# Add to imports
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Add to the main function after optimizer definition
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5, verbose=True)

# Modify the train_model function to use the scheduler
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=5):
    # ... existing code ...
    
    for epoch in range(epochs):
        # ... existing training code ...
        
        # Calculate epoch validation metrics
        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_accuracy = accuracy_score(all_labels, all_preds)
        
        # Step the scheduler
        scheduler.step(epoch_val_loss)
        
        # ... rest of existing code ...
```

### Data Augmentation

The script already includes basic data augmentation. You can enhance it by modifying the `train_transforms`:

```python
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),  # Increased rotation
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),  # Added perspective transformation
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Increased jitter
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

## Deploying Your Model

After training, MLflow registers your model in the Model Registry. You can deploy it using:

```python
import mlflow.pytorch

# Load the model from the registry
model_name = "image_classifier_resnet18"
stage = "Production"  # or "Staging", "None", etc.
model = mlflow.pytorch.load_model(f"models:/{model_name}/{stage}")

# Use the model for inference
import torch
from PIL import Image
from torchvision import transforms

def predict(image_path, model, device="cpu"):
    # Prepare the image
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
    
    return predicted.item()
```

## Hyperparameter Tuning

For a hackathon setting, you might want to quickly find good hyperparameters. Here's how to integrate hyperparameter tuning with Optuna:

```python
import optuna

def objective(trial):
    # Define hyperparameters to tune
    params = {
        "model_name": trial.suggest_categorical("model_name", ["resnet18", "mobilenet_v2"]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "optimizer": trial.suggest_categorical("optimizer", ["Adam", "SGD"]),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
        "freeze_backbone": trial.suggest_categorical("freeze_backbone", [True, False])
    }
    
    # ... rest of your training code ...
    
    # Return the validation accuracy
    return final_accuracy

# Create and optimize the study
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)  # Adjust number of trials based on time constraints

print(f"Best parameters: {study.best_params}")
print(f"Best accuracy: {study.best_value:.4f}")
```

## Troubleshooting

### Out of Memory Errors

If you encounter GPU memory issues:
1. Reduce the batch size
2. Use a smaller model architecture
3. Use mixed precision training (requires PyTorch 1.6+)

### Slow Training

To speed up training:
1. Increase `num_workers` for faster data loading
2. Use a smaller model
3. Use fewer epochs or implement early stopping

### Overfitting

If your model performs well on training data but poorly on validation:
1. Increase data augmentation
2. Add dropout layers
3. Use stronger weight decay
4. Collect more training data

## Next Steps

Once you have a well-performing model, you can:

1. Export it for production use
2. Create a simple API using Flask or FastAPI
3. Deploy it to cloud services like AWS, Azure, or Google Cloud
4. Create a demo UI to showcase your model

MLflow makes all of these steps easier by tracking your experiments and providing a simple interface for model management and deployment.

Good luck with your hackathon!
