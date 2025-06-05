import os
import json
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from classifier import load_model, predict_hate_speech, combined_classification
import numpy as np
from tqdm import tqdm
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

# Paths
BASE_DIR = os.path.dirname(__file__)
DATASET_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', 'Dynamically-Generated-Hate-Speech-Dataset-main', 'Dynamically Generated Hate Dataset v0.2.3.csv'))
EVALUATION_PATH = os.path.join(BASE_DIR, 'evaluation_results.json')

# Configuration
TEST_SET_SIZE = 100  # Number of samples to evaluate
RANDOM_SEED = 42     # For reproducibility

def create_example_table(text, true_label, trad_pred, llm_pred, final_pred):
    """Create a rich table for displaying example results."""
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    # Truncate text if too long
    display_text = text if len(text) < 100 else text[:97] + "..."
    
    # Convert boolean predictions to Yes/No
    def bool_to_yn(b):
        return "Yes" if b else "No"
    
    table.add_row("Text", display_text)
    table.add_row("True Label", bool_to_yn(true_label))
    table.add_row("Traditional", bool_to_yn(trad_pred))
    table.add_row("LLM", bool_to_yn(llm_pred))
    table.add_row("Final", bool_to_yn(final_pred))
    
    # Add agreement status
    agreement = "✓" if trad_pred == llm_pred else "✗"
    table.add_row("Agreement", agreement)
    
    return table

def evaluate_classifiers():
    console = Console()
    
    # Read dataset
    console.print("[bold blue]Loading dataset...[/bold blue]")
    df = pd.read_csv(DATASET_PATH, usecols=['text', 'label', 'type', 'split'])
    
    # Only use test set
    df_test = df[df['split'] == 'test']
    console.print(f"Total test set size: {len(df_test)} samples")
    
    # Sample a subset of the test set
    df_test = df_test.sample(n=min(TEST_SET_SIZE, len(df_test)), random_state=RANDOM_SEED)
    console.print(f"Using {len(df_test)} samples for evaluation")
    
    # Get distribution of labels
    true_labels = df_test['label'].map(lambda x: x == 'hate')
    label_counts = true_labels.value_counts()
    console.print("\n[bold]Label distribution in test set:[/bold]")
    console.print(f"Hate Speech: {label_counts.get(True, 0)} samples")
    console.print(f"Non-Hate: {label_counts.get(False, 0)} samples")
    
    texts = df_test['text'].astype(str)
    
    # Initialize predictions
    traditional_preds = []
    llm_preds = []
    combined_preds = []
    
    # Load traditional model
    console.print("\n[bold blue]Loading traditional model...[/bold blue]")
    vectorizer, clf = load_model()
    
    # Make predictions
    console.print("\n[bold blue]Making predictions...[/bold blue]")
    
    # Create a live display for the current example
    with Live(console=console, refresh_per_second=4) as live:
        for i, (text, true_label) in enumerate(zip(texts, true_labels)):
            # Traditional classifier
            trad_pred = bool(predict_hate_speech(text, vectorizer, clf))
            traditional_preds.append(trad_pred)
            
            # Combined classifier
            combined_result = combined_classification(text)
            llm_preds.append(combined_result['llm_prediction'])
            combined_preds.append(combined_result['is_hate_speech'])
            
            # Update the live display
            example_table = create_example_table(
                text, true_label, trad_pred,
                combined_result['llm_prediction'],
                combined_result['is_hate_speech']
            )
            
            progress = f"Processing example {i+1}/{len(texts)}"
            live.update(Panel(example_table, title=progress))
    
    # Calculate metrics
    console.print("\n[bold blue]Calculating metrics...[/bold blue]")
    traditional_report = classification_report(true_labels, traditional_preds, output_dict=True)
    llm_report = classification_report(true_labels, llm_preds, output_dict=True)
    combined_report = classification_report(true_labels, combined_preds, output_dict=True)
    
    # Create confusion matrices
    trad_cm = confusion_matrix(true_labels, traditional_preds)
    llm_cm = confusion_matrix(true_labels, llm_preds)
    combined_cm = confusion_matrix(true_labels, combined_preds)
    
    # Save results
    console.print("\n[bold blue]Saving results...[/bold blue]")
    results = {
        'test_set_size': len(df_test),
        'label_distribution': {
            'hate_speech': int(label_counts.get(True, 0)),
            'non_hate': int(label_counts.get(False, 0))
        },
        'traditional_classifier': {
            'classification_report': traditional_report,
            'confusion_matrix': trad_cm.tolist()
        },
        'llm_classifier': {
            'classification_report': llm_report,
            'confusion_matrix': llm_cm.tolist()
        },
        'combined_classifier': {
            'classification_report': combined_report,
            'confusion_matrix': combined_cm.tolist()
        }
    }
    
    with open(EVALUATION_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot confusion matrices
    console.print("[bold blue]Generating visualizations...[/bold blue]")
    plot_confusion_matrices(trad_cm, llm_cm, combined_cm)
    
    # Print summary
    console.print("\n[bold]Evaluation Summary:[/bold]")
    
    console.print("\n[bold cyan]Traditional Classifier:[/bold cyan]")
    console.print(f"Accuracy: {traditional_report['accuracy']:.3f}")
    console.print(f"Precision: {traditional_report['weighted avg']['precision']:.3f}")
    console.print(f"Recall: {traditional_report['weighted avg']['recall']:.3f}")
    console.print(f"F1-Score: {traditional_report['weighted avg']['f1-score']:.3f}")
    
    console.print("\n[bold cyan]LLM Classifier:[/bold cyan]")
    console.print(f"Accuracy: {llm_report['accuracy']:.3f}")
    console.print(f"Precision: {llm_report['weighted avg']['precision']:.3f}")
    console.print(f"Recall: {llm_report['weighted avg']['recall']:.3f}")
    console.print(f"F1-Score: {llm_report['weighted avg']['f1-score']:.3f}")
    
    console.print("\n[bold cyan]Combined Classifier:[/bold cyan]")
    console.print(f"Accuracy: {combined_report['accuracy']:.3f}")
    console.print(f"Precision: {combined_report['weighted avg']['precision']:.3f}")
    console.print(f"Recall: {combined_report['weighted avg']['recall']:.3f}")
    console.print(f"F1-Score: {combined_report['weighted avg']['f1-score']:.3f}")
    
    # Calculate improvements
    accuracy_improvement = combined_report['accuracy'] - traditional_report['accuracy']
    f1_improvement = combined_report['weighted avg']['f1-score'] - traditional_report['weighted avg']['f1-score']
    
    console.print("\n[bold]Improvements over Traditional Classifier:[/bold]")
    console.print(f"Accuracy: {accuracy_improvement:+.3f}")
    console.print(f"F1-Score: {f1_improvement:+.3f}")
    
    console.print(f"\nDetailed results saved to: {EVALUATION_PATH}")
    console.print(f"Confusion matrices saved to: {os.path.join(BASE_DIR, 'confusion_matrices.png')}")

def plot_confusion_matrices(trad_cm, llm_cm, combined_cm):
    """Plot confusion matrices for all classifiers."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    
    # Plot traditional classifier confusion matrix
    sns.heatmap(trad_cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_title('Traditional Classifier')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')
    ax1.set_xticklabels(['Non-Hate', 'Hate'])
    ax1.set_yticklabels(['Non-Hate', 'Hate'])
    
    # Plot LLM classifier confusion matrix
    sns.heatmap(llm_cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
    ax2.set_title('LLM Classifier')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('True')
    ax2.set_xticklabels(['Non-Hate', 'Hate'])
    ax2.set_yticklabels(['Non-Hate', 'Hate'])
    
    # Plot combined classifier confusion matrix
    sns.heatmap(combined_cm, annot=True, fmt='d', cmap='Blues', ax=ax3)
    ax3.set_title('Combined Classifier')
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('True')
    ax3.set_xticklabels(['Non-Hate', 'Hate'])
    ax3.set_yticklabels(['Non-Hate', 'Hate'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'confusion_matrices.png'))
    plt.close()

if __name__ == "__main__":
    evaluate_classifiers() 