"""
Shodh AI Hiring Project - Main Runner
======================================
This script runs the complete pipeline for the Loan Approval ML/RL project.

Tasks:
1. EDA and Preprocessing
2. Deep Learning Model (MLP Classifier)
3. Offline RL Agent (CQL)
4. Analysis and Comparison

Author: Shodh AI Hiring Project
"""

import os
import sys
import argparse
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))


def run_task1():
    """Run Task 1: EDA and Preprocessing."""
    print("\n" + "=" * 70)
    print(" TASK 1: EDA AND PREPROCESSING")
    print("=" * 70)
    
    # Import with correct module name (file is 01_eda_preprocessing.py)
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "eda_preprocessing", 
        os.path.join(os.path.dirname(__file__), "src", "01_eda_preprocessing.py")
    )
    eda_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eda_module)
    
    LoanDataProcessor = eda_module.LoanDataProcessor
    
    # Configuration - adjust paths and sample size as needed
    DATA_PATH = 'archive/accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv'
    SAMPLE_SIZE = 200000  # Set to None for full data
    
    processor = LoanDataProcessor(DATA_PATH, sample_size=SAMPLE_SIZE)
    processor.load_data()
    processor.perform_eda(save_plots=True, output_dir='eda_plots')
    processor.select_features()
    processor.clean_data()
    X_train, X_test, y_train, y_test = processor.prepare_train_test_split()
    processor.save_processed_data()
    
    print("\n✅ Task 1 Complete!")
    return processor


def run_task2():
    """Run Task 2: Deep Learning Model."""
    print("\n" + "=" * 70)
    print(" TASK 2: DEEP LEARNING MODEL")
    print("=" * 70)
    
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "deep_learning_model", 
        os.path.join(os.path.dirname(__file__), "src", "02_deep_learning_model.py")
    )
    dl_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dl_module)
    
    trainer, metrics = dl_module.main()
    
    print("\n✅ Task 2 Complete!")
    print(f"   AUC: {metrics['AUC']:.4f}")
    print(f"   F1-Score: {metrics['F1-Score']:.4f}")
    return trainer, metrics


def run_task3():
    """Run Task 3: Offline RL Agent."""
    print("\n" + "=" * 70)
    print(" TASK 3: OFFLINE RL AGENT")
    print("=" * 70)
    
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "offline_rl_agent", 
        os.path.join(os.path.dirname(__file__), "src", "03_offline_rl_agent.py")
    )
    rl_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rl_module)
    
    trainer, metrics = rl_module.main()
    
    print("\n✅ Task 3 Complete!")
    print(f"   Approval Rate: {metrics['approval_rate']*100:.1f}%")
    print(f"   Policy Return: ${metrics['policy_total_return']:,.0f}")
    return trainer, metrics


def run_task4():
    """Run Task 4: Analysis and Comparison."""
    print("\n" + "=" * 70)
    print(" TASK 4: ANALYSIS AND COMPARISON")
    print("=" * 70)
    
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "analysis_comparison", 
        os.path.join(os.path.dirname(__file__), "src", "04_analysis_comparison.py")
    )
    analysis_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analysis_module)
    
    results = analysis_module.main()
    
    print("\n✅ Task 4 Complete!")
    return results


def run_all():
    """Run all tasks sequentially."""
    print("\n" + "=" * 70)
    print(" SHODH AI HIRING PROJECT - LOAN APPROVAL ML/RL")
    print(" Running Complete Pipeline")
    print("=" * 70)
    print(f"\n Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Task 1
    processor = run_task1()
    
    # Task 2
    dl_trainer, dl_metrics = run_task2()
    
    # Task 3
    rl_trainer, rl_metrics = run_task3()
    
    # Task 4
    analysis_results = run_task4()
    
    # Final Summary
    print("\n" + "=" * 70)
    print(" ALL TASKS COMPLETE!")
    print("=" * 70)
    print(f" Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n📊 FINAL RESULTS SUMMARY")
    print("-" * 50)
    print(f"Deep Learning Model:")
    print(f"  • AUC: {dl_metrics['AUC']:.4f}")
    print(f"  • F1-Score: {dl_metrics['F1-Score']:.4f}")
    print(f"\nOffline RL Policy:")
    print(f"  • Approval Rate: {rl_metrics['approval_rate']*100:.1f}%")
    print(f"  • Financial Return: ${rl_metrics['policy_total_return']:,.0f}")
    print(f"  • Improvement over Historical: {rl_metrics['improvement_pct']:+.1f}%")
    
    print("\n📁 OUTPUT FILES")
    print("-" * 50)
    print("  ./eda_plots/           - EDA visualizations")
    print("  ./processed_data/      - Preprocessed dataset")
    print("  ./model_outputs/       - Trained models and predictions")
    print("  ./analysis_outputs/    - Comparison analysis")
    
    return {
        'processor': processor,
        'dl_trainer': dl_trainer,
        'dl_metrics': dl_metrics,
        'rl_trainer': rl_trainer,
        'rl_metrics': rl_metrics,
        'analysis': analysis_results
    }


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Shodh AI Hiring Project - Loan Approval ML/RL Pipeline'
    )
    parser.add_argument(
        '--task', 
        type=int, 
        choices=[1, 2, 3, 4],
        help='Run a specific task (1-4). If not specified, runs all tasks.'
    )
    parser.add_argument(
        '--sample-size',
        type=int,
        default=200000,
        help='Sample size for data processing (default: 200000). Use -1 for full data.'
    )
    
    args = parser.parse_args()
    
    if args.task:
        task_funcs = {
            1: run_task1,
            2: run_task2,
            3: run_task3,
            4: run_task4
        }
        task_funcs[args.task]()
    else:
        run_all()


if __name__ == "__main__":
    main()
