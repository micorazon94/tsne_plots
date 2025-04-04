import argparse
from logging import config
import time
import torch
import torch.optim as optim
import torch.nn as nn
import logging
from pathlib import Path
import os
import datetime
import sys
import yaml
from load_dataset import CustomDataset, create_dataloader
from utils.lr_scheduler import initialize_lr_scheduler
from utils.model_utils import load_model



class Trainer:
    def __init__(self, output_path, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        train_dataset = CustomDataset(dataset_name=config['dataset_name'], 
                                      split="train", 
                                      max_classes=config['max_classes'])
        chosen_classes = train_dataset.chosen_classes
        val_dataset = CustomDataset(dataset_name=config['dataset_name'], 
                                    split="val", 
                                    max_classes=config['max_classes'], 
                                    chosen_classes=chosen_classes)
        self.class_to_idx = train_dataset.class_to_idx
        self.train_loader = create_dataloader(train_dataset, batch_size=config['batch_size'], num_workers=config['num_workers'], train=True)
        self.val_loader = create_dataloader(val_dataset, batch_size=config['batch_size'], num_workers=config['num_workers'], train=False)
        
        self.model = load_model(model_name=config['model'], 
                                num_classes=len(self.class_to_idx), 
                                pretrained=config['pretrained'],
                                checkpoint_path=config['checkpoint']).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.lr_scheduler = initialize_lr_scheduler(scheduler_name=config['lr_scheduler'], optimizer=self.optimizer)
        self.loss_function = nn.CrossEntropyLoss()

        self.output_path = Path(output_path)
        self.optimizer_step_counter = 0
        self.best_loss = float('inf')
        self.best_accuracy = 0.0

    def run_training(self, num_epochs):

        for epoch in range(num_epochs):
            start_time = time.time()
            
            train_loss = self.train_epoch()
            val_loss, val_accuracy = self.validate_epoch()
            
            epoch_time = time.time() - start_time
            
            logging.info(f"Epoch [{epoch+1}/{num_epochs}], "
                         f"Train Loss: {train_loss:.4f}, "
                         f"Val Loss: {val_loss:.4f}, "
                         f"Val Accuracy: {val_accuracy:.4f}, "
                         f"Val Accuracy %: {val_accuracy*100:.2f}, "
                         f"Epoch Time: {epoch_time:.2f}s")
        
        # Save final model after training for further fine-tuning
        model_path = f"{output_path}/final.pth"
        self.save_model(model_path)

        logging.info(f"\n\nCheckpoint path with best results: '{self.output_path}/best.pth'\n"
                     f"\tbest val. loss: {self.best_loss}\n"
                     f"\tbest val. accuracy: {self.best_accuracy*100:.2f}%")
        print(f"\n\nCheckpoint path with best results: '{self.output_path}/best.pth'\n"
              f"\tbest val. loss: {self.best_loss}\n"
              f"\tbest val. accuracy: {self.best_accuracy*100:.2f}%")
        
    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        self.optimizer.zero_grad()
        
        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)  # tensor[B, 3, 224, 224], tensor[B]
            self.optimizer.zero_grad()
            outputs = self.model(inputs)  # tensor[B, num_classes]
            
            if torch.isnan(outputs).any():
                print("NaN detected in model output")
            
            loss = self.loss_function(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            running_loss += loss.item()
            self.optimizer_step_counter += 1

        return running_loss / len(self.train_loader)

    def validate_epoch(self):
        self.model.eval()
        validation_loss = 0.0
        correct = 0
        total = 0
        avg_loss = 0

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)

                loss = self.loss_function(outputs, labels)
                validation_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        avg_loss = validation_loss / len(self.val_loader)
        accuracy = correct / total

        logging.info(f"[VAL] avg loss = {avg_loss:.6f} \t acc = {accuracy:.6f}")

        # Update learning rate scheduler
        self.lr_scheduler.step(avg_loss)
        logging.info(f"Learning rate now at {self.lr_scheduler._last_lr[0]}")

        # Save best model
        if avg_loss < self.best_loss:
            self.best_loss = avg_loss
            self.best_accuracy = accuracy
            output_path = self.output_path / "best.pth"

            self.save_model(output_path)

            logging.info(f"Got a new minimum loss of {self.best_loss}")
            logging.info(f"Saving model to {output_path}")

        return avg_loss, accuracy


    def save_model(self, model_output_path):
        # Save model state_dict and additional metadata
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'class_to_idx': self.class_to_idx,
        }, model_output_path)
        logging.info(f"Save {model_output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description='Train a neural network on CIFAR-10')
    parser.add_argument('-c', '--config', type=str, help='path to the config file', required=True)
    return parser.parse_args()
    


if __name__ == "__main__":
    args = parse_args()
    config_path = args.config
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print(config)

    # LOGGING
    now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_path = config['output_folder'] + '/' + now
    os.makedirs(output_path, exist_ok=True) 

    log_file = f"{output_path}/training.log"
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                        level="INFO",
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=log_file)
    print(f"Redirecting log output to {log_file}")

    logging.info(f"Training started at {now}")
    logging.info(f"Config path: {config_path}")
    logging.info(f"Number of epochs: {config['epochs']}")
    logging.info(f"Batch size: {config['batch_size']}")
    logging.info(f"Number of workers: {config['num_workers']}")
    logging.info(f"Learning rate: {config['learning_rate']}")

    # copy config file (config_path) to output folder
    config_file_name = os.path.basename(config_path)
    config_file_output_path = f"{output_path}/{config_file_name}"
    logging.info(f"Copying config file to {config_file_output_path}")
    os.system(f"cp {config_path} {config_file_output_path}")

    # INITIALIZE TRAINER
    trainer = Trainer(output_path, config)

    logging.info("Trainer initialized")
    logging.info(f"Number of classes: {trainer.model.fc.out_features}")

    # START TRAINING
    logging.info("Start training")
    start_training_time = time.time()

    try:
        # ... run training
        trainer.run_training(config['epochs'])
    # this special block makes sure that if a training is interrupted, we save the current state
    # this allows us to interrupt a long running training if there is some maintenance work to be
    # done or similar
    except KeyboardInterrupt:
        if trainer.model is not None:
            outputPath = trainer.output_path / \
                         str(datetime.datetime.now().strftime("%Y%m%dT%H%M%S") + '_INTERRUPTED.pth')
            torch.save(trainer.model.state_dict(), outputPath)
            logging.info('Saved after interrupt')
        else:
            logging.warning("Could not save working model")
        sys.exit(0)

    training_time_seconds = time.time() - start_training_time
    training_time_minutes = training_time_seconds / 60
    logging.info(f"Training took {training_time_minutes:.2f} minutes")

    logging.info("Training finished")
    print(f"Training finished.\nTraining log: {log_file}")


