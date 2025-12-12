"""
Task 1: Exploratory Data Analysis (EDA) and Preprocessing
==========================================================
This module handles data loading, exploration, feature engineering,
and preprocessing for the LendingClub loan dataset.

Author: Shodh AI Hiring Project
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
import os
import pickle

warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class LoanDataProcessor:
    """
    Handles all data loading, EDA, and preprocessing operations
    for the LendingClub loan dataset.
    """
    
    # Features selected based on domain knowledge and predictive power
    # These are features available at loan origination (before approval)
    SELECTED_FEATURES = [
        # Loan characteristics
        'loan_amnt',           # Loan amount requested
        'term',                # Loan term (36 or 60 months)
        'int_rate',            # Interest rate
        'installment',         # Monthly payment
        'grade',               # LC assigned loan grade
        'sub_grade',           # LC assigned loan subgrade
        
        # Borrower characteristics
        'emp_length',          # Employment length in years
        'home_ownership',      # Home ownership status
        'annual_inc',          # Annual income
        'verification_status', # Income verification status
        'purpose',             # Purpose of loan
        'dti',                 # Debt-to-income ratio
        
        # Credit history
        'delinq_2yrs',         # Delinquencies in past 2 years
        'fico_range_low',      # Lower boundary of FICO range
        'fico_range_high',     # Upper boundary of FICO range
        'inq_last_6mths',      # Inquiries in last 6 months
        'open_acc',            # Number of open credit lines
        'pub_rec',             # Number of derogatory public records
        'revol_bal',           # Total credit revolving balance
        'revol_util',          # Revolving line utilization rate
        'total_acc',           # Total number of credit lines
        
        # Additional useful features
        'mort_acc',            # Number of mortgage accounts
        'pub_rec_bankruptcies', # Number of public record bankruptcies
        
        # Target variable
        'loan_status',
    ]
    
    # Feature justifications for documentation
    FEATURE_JUSTIFICATIONS = {
        'loan_amnt': 'Higher loan amounts may correlate with higher default risk',
        'term': 'Longer terms (60 months) historically have higher default rates',
        'int_rate': 'Interest rate reflects LCs risk assessment - higher rates = higher risk',
        'installment': 'Monthly payment burden relative to income affects repayment ability',
        'grade': 'LCs proprietary risk grade - strong predictor of default',
        'sub_grade': 'Finer granularity than grade for risk assessment',
        'emp_length': 'Employment stability indicates income reliability',
        'home_ownership': 'Homeowners may have more financial stability',
        'annual_inc': 'Higher income provides more buffer for loan repayment',
        'verification_status': 'Verified income adds credibility to application',
        'purpose': 'Some loan purposes (small business) have higher default rates',
        'dti': 'Debt-to-income ratio indicates existing debt burden',
        'delinq_2yrs': 'Recent delinquencies predict future payment issues',
        'fico_range_low': 'FICO score is industry-standard creditworthiness measure',
        'fico_range_high': 'Upper FICO bound for complete picture',
        'inq_last_6mths': 'Many recent inquiries may indicate financial stress',
        'open_acc': 'Number of active accounts indicates credit utilization',
        'pub_rec': 'Public records indicate past serious credit issues',
        'revol_bal': 'Existing revolving debt affects ability to pay',
        'revol_util': 'High utilization indicates potential overextension',
        'total_acc': 'Credit history depth and experience',
        'mort_acc': 'Mortgage holders may be more financially established',
        'pub_rec_bankruptcies': 'Bankruptcies are strong negative credit indicators',
    }
    
    def __init__(self, data_path: str, sample_size: int = None):
        """
        Initialize the data processor.
        
        Args:
            data_path: Path to the CSV file
            sample_size: Optional sample size for faster iteration
        """
        self.data_path = data_path
        self.sample_size = sample_size
        self.df = None
        self.df_processed = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        
    def load_data(self) -> pd.DataFrame:
        """Load the dataset from CSV file."""
        print("=" * 60)
        print("LOADING DATA")
        print("=" * 60)
        
        # Check if file exists
        if not os.path.exists(self.data_path):
            # Try gzipped version
            gz_path = self.data_path + '.gz'
            if os.path.exists(gz_path):
                self.data_path = gz_path
            else:
                raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        print(f"Loading data from: {self.data_path}")
        
        # Load with appropriate compression
        if self.data_path.endswith('.gz'):
            self.df = pd.read_csv(self.data_path, compression='gzip', low_memory=False)
        else:
            self.df = pd.read_csv(self.data_path, low_memory=False)
        
        # Sample if specified
        if self.sample_size and len(self.df) > self.sample_size:
            print(f"Sampling {self.sample_size:,} rows from {len(self.df):,} total rows")
            self.df = self.df.sample(n=self.sample_size, random_state=42)
        
        print(f"\nDataset Shape: {self.df.shape}")
        print(f"Total Rows: {len(self.df):,}")
        print(f"Total Columns: {len(self.df.columns)}")
        print(f"Memory Usage: {self.df.memory_usage(deep=True).sum() / 1e9:.2f} GB")
        
        return self.df
    
    def perform_eda(self, save_plots: bool = True, output_dir: str = 'eda_plots'):
        """
        Perform Exploratory Data Analysis.
        
        Args:
            save_plots: Whether to save plots to files
            output_dir: Directory to save plots
        """
        print("\n" + "=" * 60)
        print("EXPLORATORY DATA ANALYSIS")
        print("=" * 60)
        
        if save_plots:
            os.makedirs(output_dir, exist_ok=True)
        
        # 1. Basic Info
        print("\n--- Basic Dataset Information ---")
        print(f"Shape: {self.df.shape}")
        print(f"\nData Types:\n{self.df.dtypes.value_counts()}")
        
        # 2. Missing Values Analysis
        print("\n--- Missing Values Analysis ---")
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Missing Count': missing,
            'Missing %': missing_pct
        }).sort_values('Missing %', ascending=False)
        
        print("\nTop 20 columns with missing values:")
        print(missing_df[missing_df['Missing Count'] > 0].head(20))
        
        # 3. Loan Status Distribution (Target Variable)
        print("\n--- Loan Status Distribution ---")
        if 'loan_status' in self.df.columns:
            status_counts = self.df['loan_status'].value_counts()
            print(status_counts)
            
            # Plot loan status distribution
            fig, ax = plt.subplots(figsize=(12, 6))
            colors = sns.color_palette("husl", len(status_counts))
            bars = ax.bar(range(len(status_counts)), status_counts.values, color=colors)
            ax.set_xticks(range(len(status_counts)))
            ax.set_xticklabels(status_counts.index, rotation=45, ha='right')
            ax.set_xlabel('Loan Status')
            ax.set_ylabel('Count')
            ax.set_title('Distribution of Loan Status')
            
            # Add value labels on bars
            for bar, val in zip(bars, status_counts.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
                       f'{val:,}', ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            if save_plots:
                plt.savefig(f'{output_dir}/loan_status_distribution.png', dpi=150)
            plt.close()
        
        # 4. Numerical Features Distribution
        print("\n--- Numerical Features Distribution ---")
        numerical_cols = ['loan_amnt', 'int_rate', 'annual_inc', 'dti', 
                         'fico_range_low', 'revol_util', 'installment']
        available_num_cols = [col for col in numerical_cols if col in self.df.columns]
        
        if available_num_cols:
            print(f"\nDescriptive Statistics for key numerical features:")
            print(self.df[available_num_cols].describe())
            
            # Plot distributions
            n_cols = 2
            n_rows = (len(available_num_cols) + 1) // 2
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4*n_rows))
            axes = axes.flatten()
            
            for i, col in enumerate(available_num_cols):
                if i < len(axes):
                    data = self.df[col].dropna()
                    # Cap outliers for visualization
                    q99 = data.quantile(0.99)
                    data_capped = data[data <= q99]
                    
                    axes[i].hist(data_capped, bins=50, edgecolor='black', alpha=0.7)
                    axes[i].set_xlabel(col)
                    axes[i].set_ylabel('Frequency')
                    axes[i].set_title(f'Distribution of {col}')
            
            # Hide unused subplots
            for j in range(i+1, len(axes)):
                axes[j].set_visible(False)
            
            plt.tight_layout()
            if save_plots:
                plt.savefig(f'{output_dir}/numerical_distributions.png', dpi=150)
            plt.close()
        
        # 5. Categorical Features Distribution
        print("\n--- Categorical Features Distribution ---")
        categorical_cols = ['term', 'grade', 'home_ownership', 'verification_status', 'purpose']
        available_cat_cols = [col for col in categorical_cols if col in self.df.columns]
        
        if available_cat_cols:
            n_cols = 2
            n_rows = (len(available_cat_cols) + 1) // 2
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4*n_rows))
            axes = axes.flatten()
            
            for i, col in enumerate(available_cat_cols):
                if i < len(axes):
                    value_counts = self.df[col].value_counts().head(10)
                    axes[i].barh(range(len(value_counts)), value_counts.values)
                    axes[i].set_yticks(range(len(value_counts)))
                    axes[i].set_yticklabels(value_counts.index)
                    axes[i].set_xlabel('Count')
                    axes[i].set_title(f'Distribution of {col}')
            
            for j in range(i+1, len(axes)):
                axes[j].set_visible(False)
            
            plt.tight_layout()
            if save_plots:
                plt.savefig(f'{output_dir}/categorical_distributions.png', dpi=150)
            plt.close()
        
        # 6. Correlation Analysis
        print("\n--- Correlation Analysis ---")
        if available_num_cols:
            corr_matrix = self.df[available_num_cols].corr()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0,
                       fmt='.2f', square=True, ax=ax)
            ax.set_title('Correlation Matrix of Numerical Features')
            plt.tight_layout()
            if save_plots:
                plt.savefig(f'{output_dir}/correlation_matrix.png', dpi=150)
            plt.close()
        
        # 7. Default Rate by Grade
        print("\n--- Default Rate by Grade ---")
        if 'grade' in self.df.columns and 'loan_status' in self.df.columns:
            # Create binary default indicator
            default_statuses = ['Charged Off', 'Default', 'Does not meet the credit policy. Status:Charged Off']
            self.df['is_default'] = self.df['loan_status'].isin(default_statuses).astype(int)
            
            default_by_grade = self.df.groupby('grade')['is_default'].agg(['mean', 'count'])
            default_by_grade.columns = ['Default Rate', 'Count']
            print(default_by_grade.sort_index())
            
            fig, ax = plt.subplots(figsize=(10, 6))
            grades = default_by_grade.index.tolist()
            bars = ax.bar(grades, default_by_grade['Default Rate'].values * 100)
            ax.set_xlabel('Loan Grade')
            ax.set_ylabel('Default Rate (%)')
            ax.set_title('Default Rate by Loan Grade')
            
            for bar, rate in zip(bars, default_by_grade['Default Rate'].values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{rate*100:.1f}%', ha='center', va='bottom')
            
            plt.tight_layout()
            if save_plots:
                plt.savefig(f'{output_dir}/default_rate_by_grade.png', dpi=150)
            plt.close()
        
        print(f"\nEDA plots saved to: {output_dir}/")
        return self
    
    def select_features(self) -> pd.DataFrame:
        """
        Select and justify features for modeling.
        
        Returns:
            DataFrame with selected features
        """
        print("\n" + "=" * 60)
        print("FEATURE SELECTION")
        print("=" * 60)
        
        # Find available features from our selection
        available_features = [col for col in self.SELECTED_FEATURES if col in self.df.columns]
        missing_features = [col for col in self.SELECTED_FEATURES if col not in self.df.columns]
        
        print(f"\nSelected {len(available_features)} features for modeling:")
        print("-" * 50)
        
        for feature in available_features:
            if feature in self.FEATURE_JUSTIFICATIONS:
                print(f"  • {feature}: {self.FEATURE_JUSTIFICATIONS[feature]}")
        
        if missing_features:
            print(f"\nNote: {len(missing_features)} features not found in dataset:")
            for f in missing_features:
                print(f"  - {f}")
        
        # Select features
        self.df_processed = self.df[available_features].copy()
        print(f"\nDataset reduced from {self.df.shape[1]} to {self.df_processed.shape[1]} columns")
        
        return self.df_processed
    
    def clean_data(self) -> pd.DataFrame:
        """
        Clean and preprocess the data.
        Handles missing values, encodes categoricals, and scales features.
        
        Returns:
            Cleaned and preprocessed DataFrame
        """
        print("\n" + "=" * 60)
        print("DATA CLEANING AND PREPROCESSING")
        print("=" * 60)
        
        df = self.df_processed.copy()
        
        # 1. Filter for final loan statuses only
        print("\n--- Step 1: Filtering for Final Loan Statuses ---")
        valid_statuses = ['Fully Paid', 'Charged Off', 'Default',
                         'Does not meet the credit policy. Status:Fully Paid',
                         'Does not meet the credit policy. Status:Charged Off']
        
        initial_count = len(df)
        df = df[df['loan_status'].isin(valid_statuses)]
        print(f"Filtered from {initial_count:,} to {len(df):,} loans with final statuses")
        
        # 2. Create binary target variable
        print("\n--- Step 2: Creating Binary Target Variable ---")
        default_statuses = ['Charged Off', 'Default', 'Does not meet the credit policy. Status:Charged Off']
        df['target'] = df['loan_status'].isin(default_statuses).astype(int)
        
        target_dist = df['target'].value_counts()
        print(f"Target Distribution:")
        print(f"  Fully Paid (0): {target_dist.get(0, 0):,} ({target_dist.get(0, 0)/len(df)*100:.1f}%)")
        print(f"  Defaulted (1): {target_dist.get(1, 0):,} ({target_dist.get(1, 0)/len(df)*100:.1f}%)")
        
        # 3. Handle term column
        print("\n--- Step 3: Processing Term Column ---")
        if 'term' in df.columns:
            df['term'] = df['term'].str.extract(r'(\d+)').astype(float)
            print(f"Term values: {df['term'].unique()}")
        
        # 4. Handle employment length
        print("\n--- Step 4: Processing Employment Length ---")
        if 'emp_length' in df.columns:
            emp_map = {
                '< 1 year': 0.5, '1 year': 1, '2 years': 2, '3 years': 3,
                '4 years': 4, '5 years': 5, '6 years': 6, '7 years': 7,
                '8 years': 8, '9 years': 9, '10+ years': 10
            }
            df['emp_length'] = df['emp_length'].map(emp_map)
            # Fill missing with median
            df['emp_length'].fillna(df['emp_length'].median(), inplace=True)
            print(f"Employment length range: {df['emp_length'].min()} - {df['emp_length'].max()}")
        
        # 5. Handle categorical variables
        print("\n--- Step 5: Encoding Categorical Variables ---")
        categorical_cols = ['grade', 'sub_grade', 'home_ownership', 
                           'verification_status', 'purpose']
        
        for col in categorical_cols:
            if col in df.columns:
                # Create dummy variables
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
                df.drop(col, axis=1, inplace=True)
                print(f"  Created {len(dummies.columns)} dummy variables for '{col}'")
        
        # 6. Handle missing values in numerical columns
        print("\n--- Step 6: Handling Missing Values ---")
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numerical_cols = [c for c in numerical_cols if c != 'target']
        
        for col in numerical_cols:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                # Fill with median
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                print(f"  Filled {missing_count:,} missing values in '{col}' with median: {median_val:.2f}")
        
        # 7. Drop original loan_status and any remaining object columns
        print("\n--- Step 7: Dropping Non-Numeric Columns ---")
        cols_to_drop = df.select_dtypes(include=['object']).columns.tolist()
        if cols_to_drop:
            df.drop(cols_to_drop, axis=1, inplace=True)
            print(f"Dropped columns: {cols_to_drop}")
        
        # 8. Store feature columns (excluding target)
        self.feature_columns = [c for c in df.columns if c != 'target']
        print(f"\nFinal feature count: {len(self.feature_columns)}")
        
        self.df_processed = df
        print(f"\nFinal dataset shape: {df.shape}")
        
        return df
    
    def scale_features(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Scale features using StandardScaler.
        
        Args:
            X: Feature array
            fit: Whether to fit the scaler (True for training, False for test)
            
        Returns:
            Scaled feature array
        """
        if fit:
            return self.scaler.fit_transform(X)
        else:
            return self.scaler.transform(X)
    
    def prepare_train_test_split(self, test_size: float = 0.2, 
                                  random_state: int = 42) -> tuple:
        """
        Prepare train/test split with scaled features.
        
        Args:
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        print("\n" + "=" * 60)
        print("TRAIN/TEST SPLIT")
        print("=" * 60)
        
        X = self.df_processed[self.feature_columns].values
        y = self.df_processed['target'].values
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scale_features(X_train, fit=True)
        X_test_scaled = self.scale_features(X_test, fit=False)
        
        print(f"Training set: {X_train.shape[0]:,} samples")
        print(f"Test set: {X_test.shape[0]:,} samples")
        print(f"Features: {X_train.shape[1]}")
        print(f"\nTraining set class distribution:")
        print(f"  Class 0 (Paid): {(y_train == 0).sum():,} ({(y_train == 0).mean()*100:.1f}%)")
        print(f"  Class 1 (Default): {(y_train == 1).sum():,} ({(y_train == 1).mean()*100:.1f}%)")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def get_loan_amounts_and_rates(self) -> tuple:
        """
        Get loan amounts and interest rates for RL reward calculation.
        
        Returns:
            Tuple of (loan_amounts, interest_rates) aligned with processed data
        """
        return (
            self.df_processed['loan_amnt'].values if 'loan_amnt' in self.df_processed.columns else None,
            self.df_processed['int_rate'].values if 'int_rate' in self.df_processed.columns else None
        )
    
    def save_processed_data(self, output_dir: str = 'processed_data'):
        """
        Save processed data and preprocessing artifacts.
        
        Args:
            output_dir: Directory to save outputs
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save processed DataFrame
        self.df_processed.to_csv(f'{output_dir}/processed_loans.csv', index=False)
        
        # Save scaler
        with open(f'{output_dir}/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save feature columns
        with open(f'{output_dir}/feature_columns.pkl', 'wb') as f:
            pickle.dump(self.feature_columns, f)
        
        print(f"\nProcessed data saved to: {output_dir}/")
        
    def get_feature_justifications(self) -> dict:
        """Return feature justifications for documentation."""
        return self.FEATURE_JUSTIFICATIONS


def main():
    """Main function to run the EDA and preprocessing pipeline."""
    
    # Configuration
    DATA_PATH = 'archive/accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv'
    SAMPLE_SIZE = 200000  # Use sample for faster iteration, set to None for full data
    
    # Initialize processor
    processor = LoanDataProcessor(DATA_PATH, sample_size=SAMPLE_SIZE)
    
    # Step 1: Load data
    processor.load_data()
    
    # Step 2: Perform EDA
    processor.perform_eda(save_plots=True, output_dir='eda_plots')
    
    # Step 3: Select features
    processor.select_features()
    
    # Step 4: Clean and preprocess data
    processor.clean_data()
    
    # Step 5: Prepare train/test split
    X_train, X_test, y_train, y_test = processor.prepare_train_test_split()
    
    # Step 6: Save processed data
    processor.save_processed_data()
    
    print("\n" + "=" * 60)
    print("EDA AND PREPROCESSING COMPLETE!")
    print("=" * 60)
    print(f"\nOutputs:")
    print(f"  • EDA plots: ./eda_plots/")
    print(f"  • Processed data: ./processed_data/")
    print(f"\nNext steps:")
    print(f"  • Run 02_deep_learning_model.py for Task 2")
    print(f"  • Run 03_offline_rl_agent.py for Task 3")
    
    return processor, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    processor, X_train, X_test, y_train, y_test = main()
