# src/training/classification_trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import time
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from ..utils.logger import setup_logger

logger = logging.getLogger(__name__)

class ClassificationTrainer:
    """
    Trainer class for glomeruli classification.
    """
    def __init__(self, 
                 model,
                 train_dataset,
                 val_dataset,
                 config):
        """
        Initialize the trainer.
        
        Args:
            model: PyTorch model
            train_dataset: Training dataset
            val_dataset: Validation dataset
            config: Configuration dictionary
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        
        # Setup device
        self.device = torch.device(config['training']['device'] 
                                  if torch.cuda.is_available() 
                                  else 'cpu')
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Setup data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=config['training']['num_workers'],
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=config['training']['num_workers'],
            pin_memory=True
        )
        
        # Setup loss function
        if config['training']['loss'] == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss()
        elif config['training']['loss'] == 'focal':
            from ..models.segmentation.losses import FocalLoss
            self.criterion = FocalLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
            
        # Setup optimizer
        if config['training']['optimizer'] == 'adam':
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=config['training']['learning_rate'],
                weight_decay=config['training']['weight_decay']
            )
        elif config['training']['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(
                model.parameters(),
                lr=config['training']['learning_rate'],
                momentum=0.9,
                weight_decay=config['training']['weight_decay']
            )
        else:
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=config['training']['learning_rate'],
                weight_decay=config['training']['weight_decay']
            )
            
        # Setup learning rate scheduler
        if config['training']['scheduler'] == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=config['training']['scheduler_step_size'],
                gamma=config['training']['scheduler_gamma']
            )
        elif config['training']['scheduler'] == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=config['training']['scheduler_gamma'],
                patience=config['training']['scheduler_patience'],
                verbose=True
            )
        else:
            self.scheduler = None
            
        # Setup early stopping
        self.best_val_loss = float('inf')
        self.early_stopping_patience = config['training']['early_stopping_patience']
        self.early_stopping_counter = 0
        
        # Setup checkpoint directory
        self.checkpoint_dir = Path(config['logging']['save_dir']) / 'classification'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        setup_logger()
        
    def train_epoch(self):
        """
        Train for one epoch.
        
        Returns:
            dict: Training metrics
        """
        self.model.train()
        
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(self.train_loader, desc='Training')
        
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            self.optimizer.step()
            
            # Compute accuracy
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update metrics
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'acc': 100. * correct / total
            })
            
        # Compute epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    def validate(self):
        """
        Validate the model.
        
        Returns:
            dict: Validation metrics
        """
        self.model.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        
        all_targets = []
        all_preds = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader, desc='Validation'):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Compute accuracy
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Update metrics
                total_loss += loss.item()
                
                # Store predictions and targets for confusion matrix
                all_targets.extend(targets.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                
        # Compute epoch metrics
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        # Compute confusion matrix
        cm = confusion_matrix(all_targets, all_preds)
        class_names = list(self.train_dataset.class_map.keys())
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'targets': np.array(all_targets),
            'predictions': np.array(all_preds),
            'class_names': class_names
        }
    
    def train(self, epochs):
        """
        Train the model for multiple epochs.
        
        Args:
            epochs: Number of epochs to train
            
        Returns:
            dict: Training history
        """
        logger.info(f"Starting training for {epochs} epochs")
        
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch+1}/{epochs}")
            
            # Train
            start_time = time.time()
            train_metrics = self.train_epoch()
            train_time = time.time() - start_time
            
            # Validate
            val_metrics = self.validate()
            
            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
                    
            # Update history
            history['train_loss'].append(train_metrics['loss'])
            history['train_acc'].append(train_metrics['accuracy'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_acc'].append(val_metrics['accuracy'])
            
            # Log metrics
            logger.info(f"Train Loss: {train_metrics['loss']:.4f}, "
                       f"Train Acc: {train_metrics['accuracy']:.2f}%, "
                       f"Val Loss: {val_metrics['loss']:.4f}, "
                       f"Val Acc: {val_metrics['accuracy']:.2f}%, "
                       f"Time: {train_time:.2f}s")
            
            # Check for improvement
            if val_metrics['loss'] < self.best_val_loss:
                logger.info(f"Validation loss improved from {self.best_val_loss:.4f} to {val_metrics['loss']:.4f}")
                self.best_val_loss = val_metrics['loss']
                self.early_stopping_counter = 0
                
                # Save checkpoint
                self._save_checkpoint(epoch, val_metrics)
                
                # Plot confusion matrix
                self._plot_confusion_matrix(val_metrics, epoch)
            else:
                self.early_stopping_counter += 1
                logger.info(f"Validation loss did not improve, counter: {self.early_stopping_counter}/{self.early_stopping_patience}")
                
                if self.early_stopping_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
                    
        # Plot training history
        self._plot_training_history(history)
        
        return history
    
    def _save_checkpoint(self, epoch, metrics):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            metrics: Validation metrics
        """
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch{epoch}_acc{metrics['accuracy']:.2f}.pth"
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': metrics['loss'],
            'accuracy': metrics['accuracy']
        }, checkpoint_path)
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def _plot_confusion_matrix(self, metrics, epoch):
        """
        Plot and save confusion matrix.
        
        Args:
            metrics: Validation metrics containing confusion matrix
            epoch: Current epoch
        """
        cm = metrics['confusion_matrix']
        class_names = metrics['class_names']
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix - Epoch {epoch+1}')
        
        cm_path = self.checkpoint_dir / f"confusion_matrix_epoch{epoch}.png"
        plt.savefig(cm_path, bbox_inches='tight')
        plt.close()
    
    def _plot_training_history(self, history):
        """
        Plot and save training history.
        
        Args:
            history: Training history dictionary
        """
        plt.figure(figsize=(12, 5))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train')
        plt.plot(history['val_loss'], label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        
        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Train')
        plt.plot(history['val_acc'], label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        
        plt.tight_layout()
        
        history_path = self.checkpoint_dir / "training_history.png"
        plt.savefig(history_path, bbox_inches='tight')
        plt.close()
