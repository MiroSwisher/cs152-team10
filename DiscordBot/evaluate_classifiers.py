import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from classifier import load_model, predict_severity, llm_classification, tuned_llm_classification
from hate_speech_classifier import HateSpeechClassifier
import logging
from tqdm import tqdm
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation.log'),
        logging.StreamHandler()
    ]
)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, 'small_val_data.jsonl')  # Using validation set for evaluation
METRICS_DIR = os.path.join(BASE_DIR, 'evaluation_metrics')
os.makedirs(METRICS_DIR, exist_ok=True)

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

def load_test_data(sample_size=100):
    """Load and prepare test data."""
    logging.info("Loading test data...")
    df = pd.read_json(DATASET_PATH, lines=True)
    
    # Extract text from nested structure
    texts = []
    labels = []
    
    for _, row in df.iterrows():
        # Get the text from the user's message
        text = row['contents'][0]['parts'][0]['text'].replace("Analyze this text for hate speech: ", "")
        
        # Get the severity from the model's response
        response = row['contents'][1]['parts'][0]['text']
        response_json = json.loads(response)
        severity = response_json['severity']
        
        texts.append(text)
        labels.append(severity)
    
    # Convert to DataFrame and sample
    df = pd.DataFrame({'text': texts, 'label': labels})
    if sample_size and sample_size < len(df):
        return df.sample(n=sample_size, random_state=42)
    return df

def evaluate_classifier(texts, true_labels, classifier_type='traditional'):
    """
    Evaluate a classifier on the test set.
    
    Args:
        texts (list): List of texts to classify
        true_labels (list): True labels (0-4)
        classifier_type (str): 'traditional', 'llm', or 'tuned_llm'
        
    Returns:
        tuple: (predictions, classification_report_dict)
    """
    predictions = []
    
    if classifier_type == 'traditional':
        vectorizer, clf = load_model()
        for text in tqdm(texts, desc="Traditional ML"):
            severity = predict_severity(text, vectorizer, clf)
            predictions.append(severity)
            
    elif classifier_type == 'llm':
        classifier = HateSpeechClassifier()
        for text in tqdm(texts, desc="LLM"):
            try:
                result = classifier.classify_message(text, verbose=True)
                if isinstance(result, str):
                    result = json.loads(result)
                predictions.append(result['severity'])
            except Exception as e:
                logging.error(f"Error classifying text with LLM: {str(e)}\nText: {text[:100]}...")
                predictions.append(0)  # Default to non-hateful on error
                
    else:  # tuned_llm
        for text in tqdm(texts, desc="Tuned LLM"):
            try:
                result = tuned_llm_classification(text)
                if isinstance(result, str):
                    result = json.loads(result)
                predictions.append(result['severity'])
            except Exception as e:
                logging.error(f"Error classifying text with Tuned LLM: {str(e)}\nText: {text[:100]}...")
                predictions.append(0)  # Default to non-hateful on error
    
    # Convert predictions to numpy array for metrics calculation
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    
    # Generate classification report
    report = classification_report(
        true_labels, 
        predictions, 
        labels=range(5),  # 0-4 severity levels
        output_dict=True
    )
    
    return predictions, report

def plot_confusion_matrix(y_true, y_pred, title, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=range(5))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['0', '1', '2', '3', '4'],
                yticklabels=['0', '1', '2', '3', '4'])
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_metrics(report, classifier_type, predictions, true_labels):
    """Save metrics to JSON file."""
    metrics = {
        'classification_report': report,
        'classifier_type': classifier_type,
        'total_samples': len(true_labels),
        'label_distribution': {
            str(i): int(np.sum(true_labels == i)) for i in range(5)
        }
    }
    
    # Save individual metrics
    output_path = os.path.join(METRICS_DIR, f'{classifier_type}_metrics.json')
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Plot confusion matrix
    plot_confusion_matrix(
        true_labels, 
        predictions,
        f'Confusion Matrix - {classifier_type.upper()}',
        os.path.join(METRICS_DIR, f'{classifier_type}_confusion_matrix.png')
    )
    
    return metrics

def main():
    # Load test data (sample 100 examples)
    df = load_test_data(sample_size=100)
    texts = df['text'].tolist()
    true_labels = df['label'].tolist()
    
    logging.info(f"Evaluating on {len(texts)} test samples")
    
    # Evaluate traditional ML classifier
    logging.info("Evaluating traditional ML classifier...")
    trad_preds, trad_report = evaluate_classifier(texts, true_labels, 'traditional')
    trad_metrics = save_metrics(trad_report, 'traditional', trad_preds, true_labels)
    
    # Evaluate LLM classifier
    logging.info("Evaluating LLM classifier...")
    llm_preds, llm_report = evaluate_classifier(texts, true_labels, 'llm')
    llm_metrics = save_metrics(llm_report, 'llm', llm_preds, true_labels)
    
    # Evaluate tuned LLM classifier
    logging.info("Evaluating tuned LLM classifier...")
    tuned_preds, tuned_report = evaluate_classifier(texts, true_labels, 'tuned_llm')
    tuned_metrics = save_metrics(tuned_report, 'tuned_llm', tuned_preds, true_labels)
    
    # Print summary
    logging.info("\nEvaluation Summary:")
    
    logging.info("\nTraditional ML Classifier:")
    logging.info(f"Accuracy: {trad_report['accuracy']:.3f}")
    logging.info(f"Macro F1: {trad_report['macro avg']['f1-score']:.3f}")
    
    logging.info("\nLLM Classifier:")
    logging.info(f"Accuracy: {llm_report['accuracy']:.3f}")
    logging.info(f"Macro F1: {llm_report['macro avg']['f1-score']:.3f}")
    
    logging.info("\nTuned LLM Classifier:")
    logging.info(f"Accuracy: {tuned_report['accuracy']:.3f}")
    logging.info(f"Macro F1: {tuned_report['macro avg']['f1-score']:.3f}")
    
    # Save combined metrics
    combined_metrics = {
        'traditional_ml': trad_metrics,
        'llm': llm_metrics,
        'tuned_llm': tuned_metrics,
        'comparison': {
            'accuracy': {
                'traditional_ml': trad_report['accuracy'],
                'llm': llm_report['accuracy'],
                'tuned_llm': tuned_report['accuracy']
            },
            'macro_f1': {
                'traditional_ml': trad_report['macro avg']['f1-score'],
                'llm': llm_report['macro avg']['f1-score'],
                'tuned_llm': tuned_report['macro avg']['f1-score']
            }
        }
    }
    
    with open(os.path.join(METRICS_DIR, 'combined_metrics.json'), 'w') as f:
        json.dump(combined_metrics, f, indent=2)
    
    logging.info(f"\nDetailed metrics saved to: {METRICS_DIR}")

if __name__ == "__main__":
    main() 