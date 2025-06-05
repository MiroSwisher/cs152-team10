import os
import json
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from classifier import load_model, predict_severity, combined_classification
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

def create_example_table(text, true_sev, trad_sev, llm_sev, final_sev):
    """Create a rich table for displaying example results."""
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    # Truncate text if too long
    display_text = text if len(text) < 100 else text[:97] + "..."
    
    table.add_row("Text", display_text)
    table.add_row("True Severity", str(true_sev))
    table.add_row("Traditional Severity", str(trad_sev))
    table.add_row("LLM Severity", str(llm_sev))
    table.add_row("Final Severity", str(final_sev))
    
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
    
    # Get distribution of severity levels
    severity_counts = df_test.apply(lambda row: severity_mapping(row['label'], row['type']), axis=1).value_counts().sort_index()
    console.print("\n[bold]Severity level distribution in test set:[/bold]")
    for severity, count in severity_counts.items():
        console.print(f"Severity {severity}: {count} samples")
    
    texts = df_test['text'].astype(str)
    true_severities = df_test.apply(lambda row: severity_mapping(row['label'], row['type']), axis=1)
    
    # Initialize predictions
    traditional_preds = []
    combined_preds = []
    
    # Load traditional model
    console.print("\n[bold blue]Loading traditional model...[/bold blue]")
    vectorizer, clf = load_model()
    
    # Make predictions
    console.print("\n[bold blue]Making predictions...[/bold blue]")
    
    # Create a live display for the current example
    with Live(console=console, refresh_per_second=4) as live:
        for i, (text, true_sev) in enumerate(zip(texts, true_severities)):
            # Traditional classifier
            trad_severity = predict_severity(text, vectorizer, clf)
            traditional_preds.append(trad_severity)
            
            # Combined classifier
            combined_result = combined_classification(text)
            combined_preds.append(combined_result['severity'])
            
            # Update the live display
            example_table = create_example_table(
                text, true_sev, trad_severity, 
                combined_result['llm_severity'], 
                combined_result['severity']
            )
            
            progress = f"Processing example {i+1}/{len(texts)}"
            live.update(Panel(example_table, title=progress))
    
    # Calculate metrics
    console.print("\n[bold blue]Calculating metrics...[/bold blue]")
    traditional_report = classification_report(true_severities, traditional_preds, output_dict=True)
    combined_report = classification_report(true_severities, combined_preds, output_dict=True)
    
    # Create confusion matrices
    trad_cm = confusion_matrix(true_severities, traditional_preds)
    combined_cm = confusion_matrix(true_severities, combined_preds)
    
    # Save results
    console.print("\n[bold blue]Saving results...[/bold blue]")
    results = {
        'test_set_size': len(df_test),
        'severity_distribution': severity_counts.to_dict(),
        'traditional_classifier': {
            'classification_report': traditional_report,
            'confusion_matrix': trad_cm.tolist()
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
    plot_confusion_matrices(trad_cm, combined_cm)
    
    # Print summary
    console.print("\n[bold]Evaluation Summary:[/bold]")
    console.print("\n[bold cyan]Traditional Classifier:[/bold cyan]")
    console.print(f"Accuracy: {traditional_report['accuracy']:.3f}")
    console.print(f"Macro F1: {traditional_report['macro avg']['f1-score']:.3f}")
    
    console.print("\n[bold cyan]Combined Classifier:[/bold cyan]")
    console.print(f"Accuracy: {combined_report['accuracy']:.3f}")
    console.print(f"Macro F1: {combined_report['macro avg']['f1-score']:.3f}")
    
    # Calculate improvement
    accuracy_improvement = combined_report['accuracy'] - traditional_report['accuracy']
    f1_improvement = combined_report['macro avg']['f1-score'] - traditional_report['macro avg']['f1-score']
    
    console.print("\n[bold]Improvements:[/bold]")
    console.print(f"Accuracy: {accuracy_improvement:+.3f}")
    console.print(f"Macro F1: {f1_improvement:+.3f}")
    
    console.print(f"\nDetailed results saved to: {EVALUATION_PATH}")
    console.print(f"Confusion matrices saved to: {os.path.join(BASE_DIR, 'confusion_matrices.png')}")

def plot_confusion_matrices(trad_cm, combined_cm):
    """Plot confusion matrices for both classifiers."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot traditional classifier confusion matrix
    sns.heatmap(trad_cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_title('Traditional Classifier')
    ax1.set_xlabel('Predicted Severity')
    ax1.set_ylabel('True Severity')
    
    # Plot combined classifier confusion matrix
    sns.heatmap(combined_cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
    ax2.set_title('Combined Classifier')
    ax2.set_xlabel('Predicted Severity')
    ax2.set_ylabel('True Severity')
    
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'confusion_matrices.png'))
    plt.close()

def severity_mapping(label, type_str):
    """Map dataset label and type to severity levels."""
    if label != 'hate':
        return 0
    if type_str == 'animosity':
        return 1
    elif type_str in ['derogation', 'dehumanization']:
        return 2
    elif type_str == 'threatening':
        return 3
    elif type_str == 'support':
        return 4
    else:
        return 1

if __name__ == "__main__":
    evaluate_classifiers() 