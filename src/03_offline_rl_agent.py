"""
Task 3: Offline Reinforcement Learning Agent (IMPROVED)
========================================================
This module implements an Offline RL agent for loan approval decisions.

IMPROVEMENTS:
- Counterfactual learning with synthetic deny experiences
- Better reward normalization
- Higher CQL alpha for more conservative policy
- Reward shaping that encourages selective denials

RL Formulation:
- State (s): Preprocessed feature vector of loan applicant
- Action (a): {0: Deny Loan, 1: Approve Loan}
- Reward (r):
  - Deny: Small positive reward for avoiding potential defaults
  - Approve + Paid: +loan_amnt * int_rate (profit from interest)
  - Approve + Default: -loan_amnt (loss of principal)

Author: Shodh AI Hiring Project
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import pickle
import os
import warnings
from typing import Tuple, Dict, Optional
from dataclasses import dataclass

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Random seeds
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LoanApprovalEnvironment:
    """
    Simulated loan approval environment for offline RL.
    IMPROVED: Now includes counterfactual reward estimation.
    """
    
    def __init__(self, 
                 states: np.ndarray,
                 loan_amounts: np.ndarray,
                 interest_rates: np.ndarray,
                 outcomes: np.ndarray):
        """
        Initialize the environment.
        """
        self.states = states
        self.loan_amounts = loan_amounts
        self.interest_rates = interest_rates / 100.0  # Convert to decimal
        self.outcomes = outcomes
        self.n_samples = len(states)
        
        self.state_dim = states.shape[1]
        self.action_dim = 2
        
        print(f"Environment initialized with {self.n_samples:,} loan applications")
        print(f"State dimension: {self.state_dim}")
        print(f"Default rate: {outcomes.mean()*100:.1f}%")
    
    def calculate_reward(self, action: int, loan_amount: float, 
                        interest_rate: float, outcome: int) -> float:
        """
        Calculate reward based on action and outcome.
        
        IMPROVED reward structure:
        - Deny: Small reward proportional to avoided loss potential
        - Approve + Paid: Profit from interest
        - Approve + Default: Loss of principal
        """
        if action == 0:  # Deny
            # Small reward for being cautious with risky loans
            # Higher for loans that would have defaulted
            if outcome == 1:  # Would have defaulted
                return loan_amount * 0.1  # Reward for avoiding loss
            else:  # Would have been paid
                return -loan_amount * 0.05  # Small penalty for missing opportunity
        else:  # Approve
            if outcome == 0:  # Fully Paid
                term_years = 3
                profit = loan_amount * interest_rate * term_years
                return profit
            else:  # Defaulted
                return -loan_amount
    
    def create_offline_dataset_with_counterfactuals(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create offline RL dataset with counterfactual experiences.
        
        IMPROVED: Generates both approve and deny experiences to help
        the agent learn when to deny loans.
        """
        print("\n" + "=" * 60)
        print("CREATING OFFLINE RL DATASET (WITH COUNTERFACTUALS)")
        print("=" * 60)
        
        # Original data: all approved
        approve_actions = np.ones(self.n_samples, dtype=np.int32)
        approve_rewards = np.zeros(self.n_samples, dtype=np.float32)
        
        for i in range(self.n_samples):
            approve_rewards[i] = self.calculate_reward(
                action=1,
                loan_amount=self.loan_amounts[i],
                interest_rate=self.interest_rates[i],
                outcome=self.outcomes[i]
            )
        
        # Counterfactual data: simulate deny actions
        deny_actions = np.zeros(self.n_samples, dtype=np.int32)
        deny_rewards = np.zeros(self.n_samples, dtype=np.float32)
        
        for i in range(self.n_samples):
            deny_rewards[i] = self.calculate_reward(
                action=0,
                loan_amount=self.loan_amounts[i],
                interest_rate=self.interest_rates[i],
                outcome=self.outcomes[i]
            )
        
        # Combine datasets (use more deny examples for defaulted loans)
        # This helps the agent learn to identify high-risk loans
        
        # Add all approve experiences
        all_states = [self.states.copy()]
        all_actions = [approve_actions]
        all_rewards = [approve_rewards]
        
        # Add deny experiences - oversample denials for defaulted loans
        # This is key: we want the agent to learn that denying defaulted loans is good
        default_mask = self.outcomes == 1
        paid_mask = self.outcomes == 0
        
        # Add deny experiences for ALL defaulted loans (multiple times to reinforce)
        for _ in range(3):  # Triple the deny experiences for defaults
            all_states.append(self.states[default_mask])
            all_actions.append(np.zeros(default_mask.sum(), dtype=np.int32))
            all_rewards.append(deny_rewards[default_mask])
        
        # Add deny experiences for some paid loans (to show opportunity cost)
        paid_sample = np.random.choice(np.where(paid_mask)[0], 
                                       size=min(int(paid_mask.sum() * 0.3), 10000), 
                                       replace=False)
        all_states.append(self.states[paid_sample])
        all_actions.append(np.zeros(len(paid_sample), dtype=np.int32))
        all_rewards.append(deny_rewards[paid_sample])
        
        # Concatenate all
        states = np.vstack(all_states)
        actions = np.concatenate(all_actions)
        rewards = np.concatenate(all_rewards)
        next_states = states.copy()
        dones = np.ones(len(states), dtype=np.float32)
        
        # Shuffle
        perm = np.random.permutation(len(states))
        states = states[perm]
        actions = actions[perm]
        rewards = rewards[perm]
        next_states = next_states[perm]
        dones = dones[perm]
        
        # Statistics
        print(f"\nDataset size: {len(states):,} experiences")
        print(f"Approve actions: {(actions == 1).sum():,} ({(actions == 1).mean()*100:.1f}%)")
        print(f"Deny actions: {(actions == 0).sum():,} ({(actions == 0).mean()*100:.1f}%)")
        
        print(f"\nReward Statistics:")
        print(f"  Mean: {rewards.mean():.2f}")
        print(f"  Std: {rewards.std():.2f}")
        print(f"  Min: {rewards.min():.2f}")
        print(f"  Max: {rewards.max():.2f}")
        
        return states, actions, rewards, next_states, dones


class ConservativeQLearning(nn.Module):
    """
    Conservative Q-Learning (CQL) implementation.
    IMPROVED: Deeper network with better initialization.
    """
    
    def __init__(self, state_dim: int, action_dim: int = 2, 
                 hidden_dims: list = [256, 256, 128]):
        super(ConservativeQLearning, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Q-network with deeper architecture
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LeakyReLU(0.1),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.q_network = nn.Sequential(*layers)
        
        # Target network
        self.target_network = self._create_target_network(layers)
        
        # Initialize
        self._initialize_weights()
    
    def _create_target_network(self, layers):
        """Create target network as a copy."""
        target_layers = []
        for layer in layers:
            if isinstance(layer, nn.Linear):
                new_layer = nn.Linear(layer.in_features, layer.out_features)
                new_layer.weight.data = layer.weight.data.clone()
                new_layer.bias.data = layer.bias.data.clone()
                target_layers.append(new_layer)
            elif isinstance(layer, nn.LayerNorm):
                target_layers.append(nn.LayerNorm(layer.normalized_shape))
            elif isinstance(layer, nn.Dropout):
                target_layers.append(nn.Dropout(layer.p))
            else:
                target_layers.append(layer)
        return nn.Sequential(*target_layers)
    
    def _initialize_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.q_network(state)
    
    def get_q_values(self, state: torch.Tensor) -> torch.Tensor:
        return self.q_network(state)
    
    def get_target_q_values(self, state: torch.Tensor) -> torch.Tensor:
        return self.target_network(state)
    
    def update_target_network(self, tau: float = 0.005):
        """Soft update target network."""
        for param, target_param in zip(self.q_network.parameters(), 
                                       self.target_network.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


class OfflineRLTrainer:
    """
    Trainer for Offline RL algorithms.
    IMPROVED: Higher CQL alpha, better reward scaling.
    """
    
    def __init__(self, state_dim: int, action_dim: int = 2,
                 hidden_dims: list = [256, 256, 128],
                 learning_rate: float = 1e-4,
                 gamma: float = 0.0,  # Single-step decisions
                 cql_alpha: float = 5.0):  # Higher CQL regularization
        """
        Initialize the trainer.
        """
        self.device = DEVICE
        self.gamma = gamma
        self.cql_alpha = cql_alpha
        self.action_dim = action_dim
        
        self.q_network = ConservativeQLearning(
            state_dim, action_dim, hidden_dims
        ).to(self.device)
        
        self.optimizer = optim.AdamW(
            self.q_network.parameters(), 
            lr=learning_rate,
            weight_decay=1e-4
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=200
        )
        
        self.training_history = {
            'q_loss': [],
            'cql_loss': [],
            'total_loss': [],
            'mean_q_value': [],
            'q_diff': []
        }
    
    def compute_cql_loss(self, q_values: torch.Tensor, 
                         actions: torch.Tensor) -> torch.Tensor:
        """Compute CQL regularization loss."""
        # Log-sum-exp of Q-values
        logsumexp_q = torch.logsumexp(q_values, dim=1)
        
        # Q-values for actions in dataset
        q_for_actions = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # CQL loss
        cql_loss = (logsumexp_q - q_for_actions).mean()
        
        return cql_loss
    
    def train_step(self, states: torch.Tensor, actions: torch.Tensor,
                   rewards: torch.Tensor, next_states: torch.Tensor,
                   dones: torch.Tensor) -> Dict[str, float]:
        """Single training step."""
        # Get current Q-values
        q_values = self.q_network.get_q_values(states)
        current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q-values (single-step, so just rewards)
        with torch.no_grad():
            if self.gamma > 0:
                next_q_values = self.q_network.get_target_q_values(next_states)
                max_next_q = next_q_values.max(dim=1)[0]
                target_q = rewards + self.gamma * max_next_q * (1 - dones)
            else:
                target_q = rewards  # Single-step, no discounting
        
        # TD loss with Huber loss for stability
        td_loss = nn.functional.smooth_l1_loss(current_q, target_q)
        
        # CQL regularization loss
        cql_loss = self.compute_cql_loss(q_values, actions)
        
        # Total loss
        total_loss = td_loss + self.cql_alpha * cql_loss
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self.q_network.update_target_network()
        
        # Calculate Q-value difference (approve - deny)
        q_diff = (q_values[:, 1] - q_values[:, 0]).mean().item()
        
        return {
            'q_loss': td_loss.item(),
            'cql_loss': cql_loss.item(),
            'total_loss': total_loss.item(),
            'mean_q_value': current_q.mean().item(),
            'q_diff': q_diff
        }
    
    def train(self, states: np.ndarray, actions: np.ndarray,
              rewards: np.ndarray, next_states: np.ndarray,
              dones: np.ndarray, epochs: int = 200,
              batch_size: int = 512) -> Dict[str, list]:
        """Train the offline RL agent."""
        print("\n" + "=" * 60)
        print("TRAINING OFFLINE RL AGENT (IMPROVED CQL)")
        print("=" * 60)
        print(f"Training samples: {len(states):,}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"CQL alpha: {self.cql_alpha}")
        print(f"Gamma: {self.gamma}")
        
        # Normalize rewards for stable training
        reward_mean = rewards.mean()
        reward_std = rewards.std() + 1e-8
        rewards_normalized = (rewards - reward_mean) / reward_std
        
        print(f"\nReward normalization:")
        print(f"  Mean: {reward_mean:.2f}, Std: {reward_std:.2f}")
        
        # Create dataset
        dataset = TensorDataset(
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards_normalized),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones)
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        print("\n" + "-" * 60)
        for epoch in range(epochs):
            epoch_losses = {'q_loss': [], 'cql_loss': [], 'total_loss': [], 
                           'mean_q_value': [], 'q_diff': []}
            
            for batch in dataloader:
                s, a, r, ns, d = [x.to(self.device) for x in batch]
                losses = self.train_step(s, a, r, ns, d)
                
                for key in epoch_losses:
                    epoch_losses[key].append(losses[key])
            
            # Average losses
            for key in epoch_losses:
                avg_loss = np.mean(epoch_losses[key])
                self.training_history[key].append(avg_loss)
            
            # Update scheduler
            self.scheduler.step()
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1:3d}/{epochs} | "
                      f"Q-Loss: {self.training_history['q_loss'][-1]:.4f} | "
                      f"CQL-Loss: {self.training_history['cql_loss'][-1]:.4f} | "
                      f"Q-Diff: {self.training_history['q_diff'][-1]:.4f}")
        
        return self.training_history
    
    def get_policy_actions(self, states: np.ndarray) -> np.ndarray:
        """Get policy actions for given states."""
        self.q_network.eval()
        
        with torch.no_grad():
            states_tensor = torch.FloatTensor(states).to(self.device)
            q_values = self.q_network.get_q_values(states_tensor)
            actions = q_values.argmax(dim=1).cpu().numpy()
        
        return actions
    
    def get_q_values(self, states: np.ndarray) -> np.ndarray:
        """Get Q-values for all actions."""
        self.q_network.eval()
        
        with torch.no_grad():
            states_tensor = torch.FloatTensor(states).to(self.device)
            q_values = self.q_network.get_q_values(states_tensor)
            return q_values.cpu().numpy()
    
    def evaluate_policy(self, states: np.ndarray, 
                        loan_amounts: np.ndarray,
                        interest_rates: np.ndarray,
                        outcomes: np.ndarray,
                        output_dir: str = 'model_outputs') -> Dict:
        """Evaluate the learned policy."""
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "=" * 60)
        print("EVALUATING OFFLINE RL POLICY")
        print("=" * 60)
        
        # Get policy actions
        policy_actions = self.get_policy_actions(states)
        
        # Create environment for reward calculation
        env = LoanApprovalEnvironment(
            states=states,
            loan_amounts=loan_amounts,
            interest_rates=interest_rates,
            outcomes=outcomes
        )
        
        # Calculate rewards
        policy_rewards = np.zeros(len(states))
        historical_rewards = np.zeros(len(states))
        
        for i in range(len(states)):
            policy_rewards[i] = env.calculate_reward(
                policy_actions[i], loan_amounts[i], 
                interest_rates[i]/100, outcomes[i]
            )
            historical_rewards[i] = env.calculate_reward(
                1, loan_amounts[i], interest_rates[i]/100, outcomes[i]
            )
        
        # Statistics
        n_approved = policy_actions.sum()
        n_denied = len(policy_actions) - n_approved
        approval_rate = n_approved / len(policy_actions)
        
        approved_mask = policy_actions == 1
        denied_mask = policy_actions == 0
        
        approved_default_rate = outcomes[approved_mask].mean() if approved_mask.sum() > 0 else 0
        denied_default_rate = outcomes[denied_mask].mean() if denied_mask.sum() > 0 else 0
        historical_default_rate = outcomes.mean()
        
        # Financial metrics
        policy_total_return = policy_rewards.sum()
        historical_total_return = historical_rewards.sum()
        improvement = policy_total_return - historical_total_return
        improvement_pct = (improvement / abs(historical_total_return)) * 100 if historical_total_return != 0 else 0
        
        metrics = {
            'approval_rate': approval_rate,
            'n_approved': n_approved,
            'n_denied': n_denied,
            'approved_default_rate': approved_default_rate,
            'denied_default_rate': denied_default_rate,
            'historical_default_rate': historical_default_rate,
            'policy_total_return': policy_total_return,
            'historical_total_return': historical_total_return,
            'improvement': improvement,
            'improvement_pct': improvement_pct
        }
        
        # Print results
        print("\n--- Policy Actions ---")
        print(f"Approved: {n_approved:,} ({approval_rate*100:.1f}%)")
        print(f"Denied: {n_denied:,} ({(1-approval_rate)*100:.1f}%)")
        
        print("\n--- Default Rates ---")
        print(f"Historical (all approved): {historical_default_rate*100:.1f}%")
        print(f"RL Policy (approved only): {approved_default_rate*100:.1f}%")
        print(f"Would-be denied loans: {denied_default_rate*100:.1f}%")
        
        print("\n--- Financial Performance ---")
        print(f"Historical Policy Return: ${historical_total_return:,.0f}")
        print(f"RL Policy Return: ${policy_total_return:,.0f}")
        print(f"Improvement: ${improvement:,.0f} ({improvement_pct:+.1f}%)")
        
        # Plot results
        self._plot_evaluation_results(
            policy_actions, outcomes, policy_rewards, historical_rewards,
            loan_amounts, output_dir
        )
        
        return metrics
    
    def _plot_evaluation_results(self, policy_actions, outcomes, 
                                  policy_rewards, historical_rewards,
                                  loan_amounts, output_dir):
        """Plot evaluation visualizations."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # 1. Policy action distribution
        action_counts = pd.Series(policy_actions).value_counts().sort_index()
        axes[0, 0].bar(['Deny', 'Approve'], 
                      [action_counts.get(0, 0), action_counts.get(1, 0)],
                      color=['#e74c3c', '#27ae60'])
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('RL Policy Action Distribution')
        
        # 2. Reward comparison
        policies = ['Historical\n(Approve All)', 'RL Policy']
        total_returns = [historical_rewards.sum(), policy_rewards.sum()]
        colors = ['#3498db', '#2ecc71']
        bars = axes[0, 1].bar(policies, total_returns, color=colors)
        axes[0, 1].set_ylabel('Total Return ($)')
        axes[0, 1].set_title('Total Financial Return Comparison')
        for bar, val in zip(bars, total_returns):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'${val:,.0f}', ha='center', va='bottom')
        
        # 3. Default rate comparison
        approved_mask = policy_actions == 1
        denied_mask = policy_actions == 0
        
        categories = ['Historical', 'Approved', 'Denied']
        default_rates = [
            outcomes.mean() * 100,
            outcomes[approved_mask].mean() * 100 if approved_mask.sum() > 0 else 0,
            outcomes[denied_mask].mean() * 100 if denied_mask.sum() > 0 else 0
        ]
        colors = ['#95a5a6', '#27ae60', '#e74c3c']
        bars = axes[0, 2].bar(categories, default_rates, color=colors)
        axes[0, 2].set_ylabel('Default Rate (%)')
        axes[0, 2].set_title('Default Rate by Decision')
        for bar, val in zip(bars, default_rates):
            axes[0, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'{val:.1f}%', ha='center', va='bottom')
        
        # 4. Reward distribution
        axes[1, 0].hist(historical_rewards, bins=50, alpha=0.5, label='Historical', color='#3498db')
        axes[1, 0].hist(policy_rewards, bins=50, alpha=0.5, label='RL Policy', color='#2ecc71')
        axes[1, 0].set_xlabel('Reward ($)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Reward Distribution')
        axes[1, 0].legend()
        
        # 5. Loan amount by decision
        if denied_mask.sum() > 0 and approved_mask.sum() > 0:
            axes[1, 1].hist(loan_amounts[approved_mask], bins=50, alpha=0.5, 
                           label='Approved', color='#27ae60')
            axes[1, 1].hist(loan_amounts[denied_mask], bins=50, alpha=0.5, 
                           label='Denied', color='#e74c3c')
        axes[1, 1].set_xlabel('Loan Amount ($)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Loan Amount by Decision')
        axes[1, 1].legend()
        
        # 6. Training history
        if self.training_history['q_diff']:
            epochs = range(1, len(self.training_history['q_diff']) + 1)
            axes[1, 2].plot(epochs, self.training_history['q_diff'], 'b-', linewidth=2)
            axes[1, 2].axhline(y=0, color='r', linestyle='--', alpha=0.5)
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Q(Approve) - Q(Deny)')
            axes[1, 2].set_title('Q-Value Difference During Training')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/rl_policy_evaluation.png', dpi=150)
        plt.close()
        
        print(f"\nPlots saved to: {output_dir}/")
    
    def save(self, filepath: str):
        """Save the trained model."""
        torch.save({
            'q_network_state': self.q_network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'training_history': self.training_history,
            'config': {
                'state_dim': self.q_network.state_dim,
                'action_dim': self.q_network.action_dim,
                'gamma': self.gamma,
                'cql_alpha': self.cql_alpha
            }
        }, filepath)
        print(f"Model saved to: {filepath}")


def load_processed_data(data_dir: str = 'processed_data') -> tuple:
    """Load preprocessed data."""
    df = pd.read_csv(f'{data_dir}/processed_loans.csv')
    
    with open(f'{data_dir}/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    with open(f'{data_dir}/feature_columns.pkl', 'rb') as f:
        feature_columns = pickle.load(f)
    
    return df, scaler, feature_columns


def main():
    """Main function to train and evaluate the offline RL agent."""
    
    if not os.path.exists('processed_data/processed_loans.csv'):
        print("Processed data not found. Please run Task 1 first.")
        return None, None
    
    # Load data
    print("\n" + "=" * 60)
    print("LOADING DATA FOR OFFLINE RL")
    print("=" * 60)
    
    df, scaler, feature_columns = load_processed_data()
    print(f"Loaded {len(df):,} samples")
    
    # Extract features and metadata
    X = df[feature_columns].values
    y = df['target'].values
    loan_amounts = df['loan_amnt'].values
    interest_rates = df['int_rate'].values
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Train/test split
    X_train, X_test, y_train, y_test, \
    amounts_train, amounts_test, \
    rates_train, rates_test = train_test_split(
        X_scaled, y, loan_amounts, interest_rates,
        test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining: {len(X_train):,} samples ({y_train.mean()*100:.1f}% defaults)")
    print(f"Test: {len(X_test):,} samples")
    
    # Create environment
    env = LoanApprovalEnvironment(
        states=X_train,
        loan_amounts=amounts_train,
        interest_rates=rates_train,
        outcomes=y_train
    )
    
    # Create offline dataset WITH counterfactuals
    states, actions, rewards, next_states, dones = env.create_offline_dataset_with_counterfactuals()
    
    # Initialize trainer
    trainer = OfflineRLTrainer(
        state_dim=X_train.shape[1],
        action_dim=2,
        hidden_dims=[256, 256, 128],
        learning_rate=1e-4,
        gamma=0.0,  # Single-step decisions
        cql_alpha=5.0  # Strong conservative regularization
    )
    
    # Train
    history = trainer.train(
        states, actions, rewards, next_states, dones,
        epochs=200,
        batch_size=512
    )
    
    # Evaluate on test set
    metrics = trainer.evaluate_policy(
        states=X_test,
        loan_amounts=amounts_test,
        interest_rates=rates_test,
        outcomes=y_test,
        output_dir='model_outputs'
    )
    
    # Save model and results
    os.makedirs('model_outputs', exist_ok=True)
    trainer.save('model_outputs/offline_rl_agent.pt')
    
    # Save predictions
    policy_actions = trainer.get_policy_actions(X_test)
    q_values = trainer.get_q_values(X_test)
    
    np.savez(
        'model_outputs/rl_predictions.npz',
        X_test=X_test,
        y_test=y_test,
        policy_actions=policy_actions,
        q_values=q_values,
        loan_amounts=amounts_test,
        interest_rates=rates_test
    )
    
    print("\n" + "=" * 60)
    print("OFFLINE RL TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\n📊 KEY RESULTS:")
    print(f"  Approval Rate: {metrics['approval_rate']*100:.1f}%")
    print(f"  Policy Return: ${metrics['policy_total_return']:,.0f}")
    print(f"  Improvement: {metrics['improvement_pct']:+.1f}%")
    
    return trainer, metrics


if __name__ == "__main__":
    trainer, metrics = main()
