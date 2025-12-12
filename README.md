# Shodh AI Hiring Project - Loan Approval Policy Optimization

A comprehensive machine learning and reinforcement learning project for optimizing loan approval decisions using the LendingClub dataset.

## 📋 Project Overview

This project addresses the challenge of developing an intelligent loan approval system that maximizes financial returns while managing default risk. It implements both:
- **Deep Learning Model**: MLP classifier for default prediction
- **Offline RL Agent**: Conservative Q-Learning (CQL) for policy optimization

## 🎯 Objectives

1. **EDA & Preprocessing**: Analyze and prepare the LendingClub loan data
2. **Predictive Model**: Build a deep learning classifier for default prediction
3. **Offline RL Policy**: Train an agent to optimize loan approval decisions
4. **Analysis**: Compare and synthesize findings from both approaches

## 📁 Project Structure

```
├── main.py                      # Main entry point - runs all tasks
├── requirements.txt             # Project dependencies
├── README.md                    # This file
│
├── src/
│   ├── __init__.py
│   ├── 01_eda_preprocessing.py  # Task 1: EDA and data preprocessing
│   ├── 02_deep_learning_model.py # Task 2: PyTorch MLP classifier
│   ├── 03_offline_rl_agent.py   # Task 3: Offline RL with CQL
│   └── 04_analysis_comparison.py # Task 4: Model comparison
│
├── archive/                     # Dataset directory (LendingClub data)
│   ├── accepted_2007_to_2018q4.csv/
│   └── rejected_2007_to_2018q4.csv/
│
├── eda_plots/                   # Generated EDA visualizations
├── processed_data/              # Preprocessed dataset and artifacts
├── model_outputs/               # Trained models and predictions
└── analysis_outputs/            # Final analysis and reports
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Complete Pipeline

```bash
python main.py
```

### 3. Run Individual Tasks

```bash
# Task 1: EDA and Preprocessing
python main.py --task 1

# Task 2: Deep Learning Model
python main.py --task 2

# Task 3: Offline RL Agent
python main.py --task 3

# Task 4: Analysis and Comparison
python main.py --task 4
```

### 4. Adjust Sample Size (for faster iteration)

```bash
# Use 100,000 samples
python main.py --sample-size 100000

# Use full dataset
python main.py --sample-size -1
```

## 📊 Task Details

### Task 1: EDA and Preprocessing

**Key Steps:**
- Load LendingClub accepted loans dataset
- Explore feature distributions and missing values
- Select predictive features with justifications
- Clean data: handle missing values, encode categoricals, scale features
- Create train/test splits

**Output:** 
- `eda_plots/` - Visualizations
- `processed_data/` - Cleaned dataset

### Task 2: Deep Learning Model

**Architecture:**
- Multi-Layer Perceptron (MLP)
- Layers: 256 → 128 → 64 → 1
- BatchNorm, ReLU, Dropout (0.3)
- Binary Cross-Entropy Loss
- Adam optimizer with learning rate scheduling

**Metrics Reported:**
- AUC (Area Under ROC Curve)
- F1-Score

### Task 3: Offline RL Agent

**RL Formulation:**
- **State**: Preprocessed feature vector
- **Action**: {0: Deny, 1: Approve}
- **Reward**:
  - Deny → 0 (no risk, no gain)
  - Approve + Paid → +loan_amount × interest_rate × term
  - Approve + Default → −loan_amount

**Algorithm:** Conservative Q-Learning (CQL)
- Penalizes Q-values for out-of-distribution actions
- Suitable for offline learning from historical data

### Task 4: Analysis and Comparison

**Analyses Performed:**
- Decision agreement between DL and RL
- Financial return comparison
- Default rate analysis by decision category
- Behavioral differences by loan amount/interest rate
- Q-value vs probability correlation

**Recommendations:**
- Short-term: Ensemble approach
- Medium-term: Advanced RL algorithms (IQL, Decision Transformer)
- Long-term: Online learning with exploration

## 📈 Expected Results

| Metric | Historical | DL Model | RL Policy |
|--------|------------|----------|-----------|
| Approval Rate | 100% | ~85-90% | ~75-85% |
| Default Rate (Approved) | ~20% | ~15-18% | ~12-15% |
| Financial Return | Baseline | +10-15% | +15-25% |

*Note: Actual results depend on sample size and hyperparameters*

## ⚙️ Configuration

Key parameters can be adjusted in each module:

```python
# Sample size (main.py or individual modules)
SAMPLE_SIZE = 200000  # Set to None for full data

# Deep Learning (02_deep_learning_model.py)
hidden_dims = [256, 128, 64]
dropout_rate = 0.3
learning_rate = 0.001
epochs = 50

# Offline RL (03_offline_rl_agent.py)
hidden_dims = [256, 256]
learning_rate = 3e-4
cql_alpha = 1.0
epochs = 100
```

## 🛠️ Requirements

- Python 3.8+
- PyTorch 1.9+
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn
- d3rlpy (optional, for advanced RL)

## 📝 Feature Justifications

| Feature | Justification |
|---------|--------------|
| `loan_amnt` | Higher amounts may correlate with higher risk |
| `int_rate` | Reflects LC's risk assessment |
| `grade` | LC's proprietary risk grade |
| `annual_inc` | Income provides repayment buffer |
| `dti` | Debt-to-income indicates existing burden |
| `fico_range_*` | Industry-standard creditworthiness |
| `delinq_2yrs` | Recent delinquencies predict future issues |
| `purpose` | Some purposes have higher default rates |

## 📚 References

- [LendingClub Dataset on Kaggle](https://www.kaggle.com/wordsforthewise/lending-club)
- [Conservative Q-Learning (CQL)](https://arxiv.org/abs/2006.04779)
- [d3rlpy Documentation](https://d3rlpy.readthedocs.io/)

