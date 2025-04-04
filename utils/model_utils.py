import os 
import logging
import torch
import torchvision.models as models
import torch.nn as nn


def load_model(model_name: str, num_classes: int, pretrained: bool, checkpoint_path: str):
    if model_name == 'resnet18':
        return load_resnet18_model(num_classes, pretrained, checkpoint_path)
    else:
        raise ValueError(f"Unknown model: {model_name}. Please choose 'resnet18' or create your own function.")


def load_resnet18_model(num_classes: int, pretrained: bool, checkpoint_path: str):
    """
    num_classes: Number of output classes
    pretrained: If True, load model with weights pre-trained on ImageNet
    checkpoint: Path to a model checkpoint

    Load a ResNet18 model with the specified number of output classes.
    If pretrained is True, the model is loaded with weights pre-trained on ImageNet.
    If a checkpoint is provided, the model is loaded from the checkpoint.
    """

    if os.path.exists(checkpoint_path):
        logging.info(f"Loading ResNet18 model with {num_classes} classes, pretrained={pretrained}, checkpoint_path={checkpoint_path}")
        return load_resnet18_model_checkpoint(checkpoint_path, num_classes)
    
    if pretrained:
        model = models.resnet18(weights='IMAGENET1K_V1')
        logging.info("Loaded ResNet18 model with IMAGENET1K_V1 weights")
    else:
        model = models.resnet18(weights=None)
        logging.info("Loaded ResNet18 model with random weights")

    # Change the last fully connected layer for num classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model


def load_resnet18_model_checkpoint(checkpoint_path, num_classes):
    if os.path.exists(checkpoint_path):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = models.resnet18(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from checkpoint: {checkpoint_path}")
        return model
    else:
        raise ValueError(f"Checkpoint path is not valid: {checkpoint_path}")
