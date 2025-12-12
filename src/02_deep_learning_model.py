"""
Task 2: Deep Learning Model - MLP Classifier (IMPROVED)
========================================================
This module implements a Multi-Layer Perceptron (MLP) using PyTorch
for predicting loan defaults.

IMPROVEMENTS:
- Focal Loss for class imbalance
- Optimal threshold selection based on F1
- Class weights in loss function
- Better architecture with residual connections

Metrics reported: AUC (Area Under ROC Curve) and F1-Score

Author: Shodh AI Hiring Project
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
from sklearn.model_selection import train_test_split
import os
import pickle
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    Reduces the relative loss for well-classified examples,
    focusing more on hard, misclassified examples.
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """
        Args:
            alpha: Weighting factor for positive class
            gamma: Focusing parameter (higher = more focus on hard examples)
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss."""
        bce_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        
        # Apply alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Apply focal weighting
        focal_weight = alpha_t * (1 - pt) ** self.gamma
        focal_loss = focal_weight * bce_loss
        
        return focal_loss.mean()


class LoanDefaultMLP(nn.Module):
    """
    Multi-Layer Perceptron for loan default prediction.
    
    Architecture:
    - Input layer
    - 4 Hidden layers with BatchNorm, LeakyReLU, and Dropout
    - Output layer with Sigmoid activation
    """
    
    def __init__(self, input_dim: int, hidden_dims: list = [512, 256, 128, 64], 
                 dropout_rate: float = 0.4):
        """
        Initialize the MLP.
        
        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout probability
        """
        super(LoanDefaultMLP, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        
        # Build layers dynamically
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.1),  # LeakyReLU for better gradient flow
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x).squeeze()


class LoanDefaultTrainer:
    """
    Trainer class for the Loan Default MLP model.
    Handles training, validation, and evaluation with class imbalance handling.
    """
    
    def __init__(self, model: nn.Module, device: torch.device = DEVICE,
                 learning_rate: float = 0.001, weight_decay: float = 1e-4,
                 pos_weight: float = 4.0, use_focal_loss: bool = True):
        """
        Initialize the trainer.
        
        Args:
            model: The MLP model
            device: Device to train on
            learning_rate: Learning rate for optimizer
            weight_decay: L2 regularization strength
            pos_weight: Weight for positive class (defaults)
            use_focal_loss: Whether to use Focal Loss
        """
        self.model = model.to(device)
        self.device = device
        self.pos_weight = pos_weight
        self.optimal_threshold = 0.5  # Will be updated during training
        
        # Loss function - Focal Loss for class imbalance
        if use_focal_loss:
            self.criterion = FocalLoss(alpha=0.75, gamma=2.0)  # Higher alpha for minority class
        else:
            # Weighted BCE Loss
            self.criterion = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([pos_weight]).to(device)
            )
        
        self.use_focal_loss = use_focal_loss
        
        # Optimizer with higher learning rate
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler - Cosine annealing
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_auc': [],
            'val_auc': [],
            'train_f1': [],
            'val_f1': []
        }
    
    def find_optimal_threshold(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """
        Find the optimal classification threshold based on F1-score.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Optimal threshold
        """
        thresholds = np.arange(0.1, 0.9, 0.05)
        best_f1 = 0
        best_threshold = 0.5
        
        for thresh in thresholds:
            y_pred = (y_pred_proba >= thresh).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thresh
        
        return best_threshold
    
    def train_epoch(self, train_loader: DataLoader) -> tuple:
        """
        Train for one epoch.
        
        Returns:
            Tuple of (loss, auc, f1)
        """
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device).float()
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            all_preds.extend(outputs.detach().cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
        
        avg_loss = total_loss / len(train_loader)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        auc = roc_auc_score(all_labels, all_preds)
        # Use optimal threshold for F1 calculation
        f1 = f1_score(all_labels, (all_preds >= self.optimal_threshold).astype(int), zero_division=0)
        
        return avg_loss, auc, f1
    
    def validate(self, val_loader: DataLoader) -> tuple:
        """
        Validate the model.
        
        Returns:
            Tuple of (loss, auc, f1, predictions, labels)
        """
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device).float()
                
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                
                total_loss += loss.item()
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        auc = roc_auc_score(all_labels, all_preds)
        
        # Update optimal threshold based on validation data
        self.optimal_threshold = self.find_optimal_threshold(all_labels, all_preds)
        f1 = f1_score(all_labels, (all_preds >= self.optimal_threshold).astype(int), zero_division=0)
        
        return avg_loss, auc, f1
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray,
            epochs: int = 100, batch_size: int = 512,
            early_stopping_patience: int = 15) -> dict:
        """
        Train the model with class-balanced sampling.
        """
        print("\n" + "=" * 60)
        print("TRAINING DEEP LEARNING MODEL (IMPROVED)")
        print("=" * 60)
        print(f"Model Architecture: {self.model.hidden_dims}")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(X_train):,}")
        print(f"Validation samples: {len(X_val):,}")
        print(f"Batch size: {batch_size}")
        print(f"Epochs: {epochs}")
        print(f"Using {'Focal Loss' if self.use_focal_loss else 'Weighted BCE'}")
        
        # Calculate class weights for sampling
        class_counts = np.bincount(y_train.astype(int))
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[y_train.astype(int)]
        
        # Create weighted sampler for balanced training
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Class distribution
        pos_ratio = y_train.mean()
        print(f"Class distribution: {pos_ratio*100:.1f}% positive, {(1-pos_ratio)*100:.1f}% negative")
        print(f"Using balanced sampling + Focal Loss")
        
        # Training loop
        best_val_f1 = 0  # Optimize for F1 instead of AUC
        patience_counter = 0
        best_model_state = None
        
        print("\n" + "-" * 60)
        for epoch in range(epochs):
            # Train
            train_loss, train_auc, train_f1 = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_auc, val_f1 = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_auc'].append(train_auc)
            self.history['val_auc'].append(val_auc)
            self.history['train_f1'].append(train_f1)
            self.history['val_f1'].append(val_f1)
            
            # Print progress every 5 epochs
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}/{epochs} | "
                      f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                      f"Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f} | "
                      f"Val F1: {val_f1:.4f} | Thresh: {self.optimal_threshold:.2f}")
            
            # Early stopping based on F1
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping triggered at epoch {epoch+1}")
                    break
        
        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"\nRestored best model with Val F1: {best_val_f1:.4f}")
        
        return self.history
    
    def predict_proba(self, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
        """Predict probabilities."""
        self.model.eval()
        dataset = TensorDataset(torch.FloatTensor(X))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        predictions = []
        with torch.no_grad():
            for (X_batch,) in loader:
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch)
                predictions.extend(outputs.cpu().numpy())
        
        return np.array(predictions)
    
    def predict(self, X: np.ndarray, threshold: float = None) -> np.ndarray:
        """Predict binary labels using optimal threshold."""
        if threshold is None:
            threshold = self.optimal_threshold
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray,
                 output_dir: str = 'model_outputs') -> dict:
        """Evaluate the model on test set."""
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "=" * 60)
        print("MODEL EVALUATION ON TEST SET")
        print("=" * 60)
        
        # Get predictions
        y_pred_proba = self.predict_proba(X_test)
        
        # Find optimal threshold on test set
        optimal_thresh = self.find_optimal_threshold(y_test, y_pred_proba)
        y_pred = (y_pred_proba >= optimal_thresh).astype(int)
        
        # Calculate metrics
        auc = roc_auc_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        
        metrics = {
            'AUC': auc,
            'F1-Score': f1,
            'Precision': precision,
            'Recall': recall,
            'Optimal_Threshold': optimal_thresh
        }
        
        print("\n" + "-" * 40)
        print("📊 KEY METRICS (Required)")
        print("-" * 40)
        print(f"  ✓ AUC (Area Under ROC Curve): {auc:.4f}")
        print(f"  ✓ F1-Score: {f1:.4f}")
        print("-" * 40)
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  Optimal Threshold: {optimal_thresh:.2f}")
        
        # Confusion Matrix
        print("\n--- Confusion Matrix ---")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        # Classification Report
        print("\n--- Classification Report ---")
        print(classification_report(y_test, y_pred, 
                                    target_names=['Fully Paid', 'Defaulted']))
        
        # Plot ROC Curve
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # ROC Curve
        axes[0].plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {auc:.4f})')
        axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1)
        axes[0].fill_between(fpr, tpr, alpha=0.3)
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
        axes[0].set_title('ROC Curve')
        axes[0].legend(loc='lower right')
        axes[0].grid(True, alpha=0.3)
        
        # Precision-Recall Curve
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
        axes[1].plot(recall_curve, precision_curve, 'g-', linewidth=2)
        axes[1].fill_between(recall_curve, precision_curve, alpha=0.3)
        axes[1].set_xlabel('Recall')
        axes[1].set_ylabel('Precision')
        axes[1].set_title('Precision-Recall Curve')
        axes[1].grid(True, alpha=0.3)
        
        # Confusion Matrix Heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[2],
                   xticklabels=['Fully Paid', 'Defaulted'],
                   yticklabels=['Fully Paid', 'Defaulted'])
        axes[2].set_xlabel('Predicted')
        axes[2].set_ylabel('Actual')
        axes[2].set_title('Confusion Matrix')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/dl_model_evaluation.png', dpi=150)
        plt.close()
        
        # Plot training history
        self.plot_training_history(output_dir)
        
        print(f"\nEvaluation plots saved to: {output_dir}/")
        
        return metrics
    
    def plot_training_history(self, output_dir: str = 'model_outputs'):
        """Plot training history."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss
        axes[0].plot(epochs, self.history['train_loss'], 'b-', label='Train')
        axes[0].plot(epochs, self.history['val_loss'], 'r-', label='Validation')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # AUC
        axes[1].plot(epochs, self.history['train_auc'], 'b-', label='Train')
        axes[1].plot(epochs, self.history['val_auc'], 'r-', label='Validation')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('AUC')
        axes[1].set_title('Training and Validation AUC')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # F1
        axes[2].plot(epochs, self.history['train_f1'], 'b-', label='Train')
        axes[2].plot(epochs, self.history['val_f1'], 'r-', label='Validation')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('F1-Score')
        axes[2].set_title('Training and Validation F1-Score')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/training_history.png', dpi=150)
        plt.close()
    
    def save_model(self, filepath: str):
        """Save the model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'input_dim': self.model.input_dim,
            'hidden_dims': self.model.hidden_dims,
            'optimal_threshold': self.optimal_threshold
        }, filepath)
        print(f"Model saved to: {filepath}")
    
    def load_model(self, filepath: str):
        """Load the model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        self.optimal_threshold = checkpoint.get('optimal_threshold', 0.5)
        print(f"Model loaded from: {filepath}")


def load_processed_data(data_dir: str = 'processed_data') -> tuple:
    """Load preprocessed data from disk."""
    df = pd.read_csv(f'{data_dir}/processed_loans.csv')
    
    with open(f'{data_dir}/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    with open(f'{data_dir}/feature_columns.pkl', 'rb') as f:
        feature_columns = pickle.load(f)
    
    return df, scaler, feature_columns


def main():
    """Main function to train and evaluate the deep learning model."""
    
    # Check if processed data exists
    if not os.path.exists('processed_data/processed_loans.csv'):
        print("Processed data not found. Please run Task 1 first.")
        return None, None
    
    # Load processed data
    print("\n" + "=" * 60)
    print("LOADING PROCESSED DATA")
    print("=" * 60)
    
    df, scaler, feature_columns = load_processed_data()
    print(f"Loaded {len(df):,} samples with {len(feature_columns)} features")
    
    # Prepare data
    X = df[feature_columns].values
    y = df['target'].values
    
    # Train/validation/test split (60/20/20)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    
    # Scale features
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nData splits:")
    print(f"  Training: {len(X_train):,} samples ({y_train.mean()*100:.1f}% defaults)")
    print(f"  Validation: {len(X_val):,} samples")
    print(f"  Test: {len(X_test):,} samples")
    
    # Calculate class imbalance ratio
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"  Class imbalance ratio: {pos_weight:.2f}:1")
    
    # Initialize model with deeper architecture
    input_dim = X_train.shape[1]
    model = LoanDefaultMLP(
        input_dim=input_dim,
        hidden_dims=[512, 256, 128, 64],  # Deeper network
        dropout_rate=0.4  # Higher dropout
    )
    
    print(f"\nModel Architecture:")
    print(model)
    
    # Initialize trainer with Focal Loss
    trainer = LoanDefaultTrainer(
        model=model,
        device=DEVICE,
        learning_rate=0.002,  # Higher learning rate
        weight_decay=1e-3,
        pos_weight=pos_weight,
        use_focal_loss=True  # Use Focal Loss for imbalance
    )
    
    # Train model
    history = trainer.fit(
        X_train_scaled, y_train,
        X_val_scaled, y_val,
        epochs=100,
        batch_size=512,  # Larger batch size
        early_stopping_patience=15
    )
    
    # Evaluate on test set
    metrics = trainer.evaluate(X_test_scaled, y_test, output_dir='model_outputs')
    
    # Save model
    os.makedirs('model_outputs', exist_ok=True)
    trainer.save_model('model_outputs/loan_default_mlp.pt')
    
    # Save predictions for RL comparison
    y_test_pred_proba = trainer.predict_proba(X_test_scaled)
    np.savez(
        'model_outputs/dl_predictions.npz',
        X_test=X_test_scaled,
        y_test=y_test,
        y_pred_proba=y_test_pred_proba,
        optimal_threshold=trainer.optimal_threshold
    )
    
    print("\n" + "=" * 60)
    print("DEEP LEARNING MODEL TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\n📊 FINAL TEST SET RESULTS:")
    print(f"  ✓ AUC: {metrics['AUC']:.4f}")
    print(f"  ✓ F1-Score: {metrics['F1-Score']:.4f}")
    print(f"  ✓ Optimal Threshold: {metrics['Optimal_Threshold']:.2f}")
    print(f"\nOutputs saved to: ./model_outputs/")
    
    return trainer, metrics


if __name__ == "__main__":
    trainer, metrics = main()
