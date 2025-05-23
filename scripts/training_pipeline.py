def plot_confusion_matrix(cm, labels, title, save_path=None):
    """
    Plot and optionally save a confusion matrix.
    
    Args:
        cm (array): Confusion matrix
        labels (list): Class labels
        title (str): Plot title
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(8, 6))
    
    # Create the confusion matrix display
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap='Blues', values_format='d')
    
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.show()
    plt.close()


def save_results_to_file(results, file_path):
    """
    Save results to a text file.
    
    Args:
        results (dict): Results dictionary
        file_path (str): Path to save the results
    """
    with open(file_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("SENTIMENT ANALYSIS TRAINING RESULTS\n")
        f.write("="*60 + "\n\n")
        
        if 'baseline' in results:
            f.write("BASELINE MODEL RESULTS:\n")
            f.write("-"*30 + "\n")
            f.write(f"F1-score: {results['baseline']['f1_score']:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(results['baseline']['classification_report'])
            f.write("\n\n")
        
        if 'cross_validation' in results:
            f.write("CROSS-VALIDATION RESULTS:\n")
            f.write("-"*30 + "\n")
            cv = results['cross_validation']
            for i, score in enumerate(cv['scores_array']):
                f.write(f"Split {i + 1}: {score:.4f}\n")
            f.write(f"\nMean F1-score: {cv['mean_f1']:.4f}\n")
            f.write(f"Standard Deviation: {cv['std_f1']:.4f}\n\n")
            
            if 'baseline' in results:
                improvement = cv['mean_f1'] - results['baseline']['f1_score']
                f.write(f"Improvement over baseline: {improvement:.4f}\n")
    
    print(f"Results saved to: {file_path}")


#!/usr/bin/env python3
"""
Sentiment Analysis Training Pipeline

This script trains a sentiment analysis model using a pre-trained RoBERTa model
fine-tuned on Twitter data. It performs k-fold cross-validation and evaluates
the model performance.


"""

# =============================================================================
# CONFIGURATION - Edit these values for your specific setup
# =============================================================================

# Data settings
DATA_PATH = "data/all_tweets.xlsx"  # Path to your Excel file
TEXT_COLUMN = "tweet"               # Column name containing the tweets
LABEL_COLUMN = "Verdict"            # Column name containing the sentiment labels

# Model settings
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"  # Pre-trained model to use
OUTPUT_DIR = "./outputs"            # Directory to save trained models

# Training settings
N_SPLITS = 5                        # Number of cross-validation folds
MAX_STEPS = 100                     # Maximum training steps per fold
LEARNING_RATE = 1e-5               # Learning rate for fine-tuning

# Evaluation settings
EVALUATE_BASELINE = True            # Whether to evaluate the baseline model first
RUN_FINE_TUNING = True             # Whether to run fine-tuning (set False for baseline only)

# Output settings
SAVE_RESULTS = True                 # Whether to save detailed results
PLOT_CONFUSION_MATRIX = True        # Whether to plot and save confusion matrices
SAVE_BEST_MODEL = True             # Whether to save the best performing model

# =============================================================================
# IMPORTS
# =============================================================================

import os
import pandas as pd
import numpy as np
import warnings

# ML and NLP libraries
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer, 
    DataCollatorWithPadding, 
    AutoConfig,
    EarlyStoppingCallback
)
from sklearn.metrics import classification_report, confusion_matrix, f1_score, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold

# Plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def setup_model_config(model_name):
    """
    Load model configuration and get label mappings.
    
    Args:
        model_name (str): Name of the pre-trained model
        
    Returns:
        dict: Label to ID mapping
    """
    print(f"Loading model configuration from: {model_name}")
    config = AutoConfig.from_pretrained(model_name)
    label2id = config.label2id
    print(f"Model labels: {label2id}")
    return label2id


def load_tokenizer(model_name):
    """
    Load the tokenizer for the pre-trained model.
    
    Args:
        model_name (str): Name of the pre-trained model
        
    Returns:
        AutoTokenizer: Loaded tokenizer
    """
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


def preprocess_text(text):
    """
    Preprocess text according to the model's requirements.
    This follows the preprocessing used by the Cardiff NLP Twitter RoBERTa model.
    
    Args:
        text (str): Raw text to preprocess
        
    Returns:
        str: Preprocessed text
    """
    new_text = []
    for token in text.split(" "):
        # Replace mentions with @user
        token = '@user' if token.startswith('@') and len(token) > 1 else token
        # Replace URLs with http
        token = 'http' if token.startswith('http') else token
        new_text.append(token)
    return " ".join(new_text)


def load_and_prepare_data(data_path, text_column, label_column, label2id):
    """
    Load and prepare the dataset for training.
    
    Args:
        data_path (str): Path to the Excel file containing the data
        text_column (str): Name of the column containing text data
        label_column (str): Name of the column containing labels
        label2id (dict): Label to ID mapping
        
    Returns:
        Dataset: Processed dataset ready for training
    """
    print(f"Loading data from: {data_path}")
    
    # Load data
    df = pd.read_excel(data_path)
    print(f"Loaded {len(df)} samples")
    
    # Show label distribution
    print(f"Label distribution:")
    print(df[label_column].value_counts())
    
    # Preprocess text and labels
    print("Preprocessing text...")
    df['text'] = df[text_column].apply(preprocess_text)
    df['label'] = df[label_column].apply(lambda l: label2id[l])
    
    # Create dataset
    dataset = Dataset.from_pandas(df[['text', 'label']]).class_encode_column('label')
    
    return dataset


def tokenize_dataset(dataset, tokenizer):
    """
    Tokenize the dataset.
    
    Args:
        dataset (Dataset): Raw dataset to tokenize
        tokenizer (AutoTokenizer): Tokenizer to use
        
    Returns:
        Dataset: Tokenized dataset
    """
    print("Tokenizing dataset...")
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset


def evaluate_baseline_model(dataset, tokenizer, model_name, label2id):
    """
    Evaluate the pre-trained model without fine-tuning.
    
    Args:
        dataset (Dataset): Dataset to evaluate on
        tokenizer (AutoTokenizer): Tokenizer to use
        model_name (str): Name of the pre-trained model
        label2id (dict): Label to ID mapping
        
    Returns:
        dict: Evaluation metrics
    """
    print("\n" + "="*60)
    print("EVALUATING BASELINE MODEL (without fine-tuning)")
    print("="*60)
    
    # Prepare dataset
    preprocessed_dataset = DatasetDict({
        'train': dataset,
        'test': dataset,
    })
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    # Setup trainer for evaluation only
    training_args = TrainingArguments(output_dir='./temp_baseline')
    data_collator = DataCollatorWithPadding(tokenizer)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=preprocessed_dataset['train'],
        eval_dataset=preprocessed_dataset['test'],
        data_collator=data_collator,
    )
    
    # Make predictions
    eval_predictions = trainer.predict(preprocessed_dataset['test'])
    y_pred = eval_predictions.predictions.argmax(axis=-1)
    y_true = eval_predictions.label_ids
    
    # Calculate metrics
    f1 = f1_score(y_true, y_pred, average='macro')
    report = classification_report(y_true, y_pred, target_names=list(label2id.keys()))
    cm = confusion_matrix(y_true, y_pred)
    
    print("Baseline Model Performance:")
    print(report)
    print(f"Macro F1-score: {f1:.4f}")
    
    # Plot confusion matrix if requested
    if PLOT_CONFUSION_MATRIX:
        save_path = os.path.join(OUTPUT_DIR, "baseline_confusion_matrix.png") if SAVE_RESULTS else None
        plot_confusion_matrix(cm, list(label2id.keys()), "Baseline Model - Confusion Matrix", save_path)
    
    return {
        'f1_score': f1,
        'classification_report': report,
        'confusion_matrix': cm,
        'predictions': y_pred,
        'true_labels': y_true
    }


def train_single_fold(dataset, train_idxs, test_idxs, tokenizer, model_name, 
                     fold_output_dir, max_steps, learning_rate, label2id):
    """
    Train a model on a single fold of cross-validation.
    
    Args:
        dataset (Dataset): Full tokenized dataset
        train_idxs (list): Indices for training data
        test_idxs (list): Indices for test data
        tokenizer (AutoTokenizer): Tokenizer to use
        model_name (str): Name of the pre-trained model
        fold_output_dir (str): Directory to save this fold's model
        max_steps (int): Maximum training steps
        learning_rate (float): Learning rate for training
        label2id (dict): Label to ID mapping
        
    Returns:
        tuple: (f1_score, confusion_matrix, predictions, true_labels)
    """
    # Data split
    train_dataset = dataset.select(train_idxs)
    preprocessed_dataset = train_dataset.train_test_split(
        test_size=0.25, seed=1, stratify_by_column='label'
    )
    preprocessed_dataset['validation'] = preprocessed_dataset['test']
    preprocessed_dataset['test'] = dataset.select(test_idxs)
    
    print(f"Train size: {len(preprocessed_dataset['train'])}")
    print(f"Validation size: {len(preprocessed_dataset['validation'])}")
    print(f"Test size: {len(preprocessed_dataset['test'])}")
    
    # Load fresh model for this fold
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    # Training arguments
    training_args = TrainingArguments(
        learning_rate=learning_rate,
        output_dir=fold_output_dir,
        save_strategy='steps',
        save_steps=10,
        evaluation_strategy='steps',
        eval_steps=10,
        logging_strategy='steps',
        logging_steps=10,
        max_steps=max_steps,
        save_total_limit=1,
        num_train_epochs=10,
        logging_first_step=True,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
    )
    
    # Setup trainer
    data_collator = DataCollatorWithPadding(tokenizer)
    callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=preprocessed_dataset['train'],
        eval_dataset=preprocessed_dataset['validation'],
        data_collator=data_collator,
        callbacks=callbacks,
    )
    
    # Train model
    print("Training...")
    trainer.train()
    
    # Evaluate on test set
    print("Evaluating...")
    eval_predictions = trainer.predict(preprocessed_dataset['test'])
    y_pred = eval_predictions.predictions.argmax(axis=-1)
    y_true = eval_predictions.label_ids
    
    # Calculate metrics
    f1 = f1_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"F1-score: {f1:.4f}")
    print(classification_report(y_true, y_pred, target_names=list(label2id.keys())))
    
    return f1, cm, y_pred, y_true


def train_with_cross_validation(dataset, tokenizer, model_name, output_dir, 
                               n_splits, max_steps, learning_rate, label2id):
    """
    Train the model using k-fold cross-validation.
    
    Args:
        dataset (Dataset): Tokenized dataset for training
        tokenizer (AutoTokenizer): Tokenizer to use
        model_name (str): Name of the pre-trained model
        output_dir (str): Directory to save model outputs
        n_splits (int): Number of folds for cross-validation
        max_steps (int): Maximum training steps per fold
        learning_rate (float): Learning rate for training
        label2id (dict): Label to ID mapping
        
    Returns:
        dict: Cross-validation results
    """
    print("\n" + "="*60)
    print(f"STARTING {n_splits}-FOLD CROSS-VALIDATION")
    print("="*60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    all_scores = {}
    all_cms = {}
    
    # K-fold cross-validation
    folds = StratifiedKFold(n_splits=n_splits)
    splits = folds.split(np.zeros(dataset.num_rows), dataset["label"])
    
    best_score = 0
    best_fold = 0
    
    for split_id, (train_idxs, test_idxs) in enumerate(splits):
        print(f'\n--- Split {split_id + 1}/{n_splits} ---')
        
        # Train on this fold
        fold_output_dir = os.path.join(output_dir, f'split_{split_id}')
        f1, cm, y_pred, y_true = train_single_fold(
            dataset, train_idxs, test_idxs, tokenizer, model_name,
            fold_output_dir, max_steps, learning_rate, label2id
        )
        
        all_scores[split_id] = f1
        all_cms[split_id] = cm
        
        # Track best performing fold
        if f1 > best_score:
            best_score = f1
            best_fold = split_id
    
    # Plot confusion matrix for best fold
    if PLOT_CONFUSION_MATRIX and best_fold in all_cms:
        save_path = os.path.join(output_dir, f"best_fold_confusion_matrix.png") if SAVE_RESULTS else None
        plot_confusion_matrix(
            all_cms[best_fold], 
            list(label2id.keys()), 
            f"Best Fold (Split {best_fold + 1}) - Confusion Matrix", 
            save_path
        )
    
    # Calculate overall statistics
    scores_array = np.array(list(all_scores.values()))
    mean_f1 = scores_array.mean()
    std_f1 = scores_array.std()
    
    print(f'\n--- CROSS-VALIDATION SUMMARY ---')
    for i, score in enumerate(scores_array):
        print(f'Split {i + 1}: {score:.4f}')
    print(f'Mean F1-score: {mean_f1:.4f}')
    print(f'Standard Deviation: {std_f1:.4f}')
    
    # Save best model if requested
    if SAVE_BEST_MODEL:
        best_model_path = os.path.join(output_dir, f'best_model_split_{best_fold}')
        if os.path.exists(os.path.join(output_dir, f'split_{best_fold}')):
            # Copy the best model to a dedicated folder
            import shutil
            if os.path.exists(best_model_path):
                shutil.rmtree(best_model_path)
            shutil.copytree(os.path.join(output_dir, f'split_{best_fold}'), best_model_path)
            print(f"Best model (Split {best_fold + 1}, F1: {best_score:.4f}) saved to: {best_model_path}")
    
    return {
        'individual_scores': all_scores,
        'confusion_matrices': all_cms,
        'mean_f1': mean_f1,
        'std_f1': std_f1,
        'scores_array': scores_array,
        'best_fold': best_fold,
        'best_score': best_score
    }


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main function to run the training pipeline."""
    
    print("="*60)
    print("SENTIMENT ANALYSIS TRAINING PIPELINE")
    print("="*60)
    print(f"Data path: {DATA_PATH}")
    print(f"Model: {MODEL_NAME}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Cross-validation folds: {N_SPLITS}")
    print(f"Max steps per fold: {MAX_STEPS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print("="*60)
    
    # Step 1: Setup model configuration and tokenizer
    label2id = setup_model_config(MODEL_NAME)
    tokenizer = load_tokenizer(MODEL_NAME)
    
    # Step 2: Load and prepare data
    dataset = load_and_prepare_data(DATA_PATH, TEXT_COLUMN, LABEL_COLUMN, label2id)
    
    # Step 3: Tokenize dataset
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)
    
    # Step 4: Evaluate baseline model if requested
    baseline_results = None
    if EVALUATE_BASELINE:
        baseline_results = evaluate_baseline_model(
            tokenized_dataset, tokenizer, MODEL_NAME, label2id
        )
    
    # Step 5: Run fine-tuning if requested  
    cv_results = None
    if RUN_FINE_TUNING:
        cv_results = train_with_cross_validation(
            tokenized_dataset, tokenizer, MODEL_NAME, OUTPUT_DIR,
            N_SPLITS, MAX_STEPS, LEARNING_RATE, label2id
        )
    
    # Step 6: Print final comparison
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    if baseline_results:
        print(f"Baseline F1-score: {baseline_results['f1_score']:.4f}")
    
    if cv_results:
        print(f"Fine-tuned Mean F1-score: {cv_results['mean_f1']:.4f} ± {cv_results['std_f1']:.4f}")
        
        if baseline_results:
            improvement = cv_results['mean_f1'] - baseline_results['f1_score']
            print(f"Improvement: {improvement:.4f}")
    
    print("="*60)
    print("Training completed!")
    if RUN_FINE_TUNING:
        print(f"Trained models saved in: {OUTPUT_DIR}")
        if SAVE_BEST_MODEL and cv_results:
            print(f"Best model: Split {cv_results['best_fold'] + 1} (F1: {cv_results['best_score']:.4f})")
    
    # Save detailed results if requested
    if SAVE_RESULTS:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        results_dict = {}
        
        if baseline_results:
            results_dict['baseline'] = baseline_results
        
        if cv_results:
            results_dict['cross_validation'] = cv_results
        
        results_file = os.path.join(OUTPUT_DIR, "training_results.txt")
        save_results_to_file(results_dict, results_file)
        
        # Save scores to CSV for further analysis
        if cv_results:
            scores_df = pd.DataFrame({
                'Split': [f'Split_{i+1}' for i in range(len(cv_results['scores_array']))],
                'F1_Score': cv_results['scores_array']
            })
            csv_path = os.path.join(OUTPUT_DIR, "cross_validation_scores.csv")
            scores_df.to_csv(csv_path, index=False)
            print(f"Cross-validation scores saved to: {csv_path}")


if __name__ == "__main__":
    main()