import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
from classifier import llm_classification, combined_classification, tuned_llm_classification
import vertexai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = os.path.dirname(__file__)
DATASET_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', 'Dynamically-Generated-Hate-Speech-Dataset-main', 'Dynamically Generated Hate Dataset v0.2.3.csv'))
METRICS_DIR = os.path.join(BASE_DIR, 'evaluation_metrics')

def initialize_vertex_ai():
    """Initialize Vertex AI with proper project configuration."""
    try:
        project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
        location = os.getenv('VERTEX_LOCATION', 'us-west1')
        endpoint_id = os.getenv('VERTEX_ENDPOINT_ID')
        
        if not project_id:
            logger.error("GOOGLE_CLOUD_PROJECT environment variable not set")
            return False
            
        if not endpoint_id:
            logger.error("VERTEX_ENDPOINT_ID environment variable not set")
            return False
            
        vertexai.init(project=project_id, location=location)
        logger.info(f"Initialized Vertex AI with project: {project_id}, location: {location}, endpoint: {endpoint_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize Vertex AI: {str(e)}")
        return False

def load_test_data():
    """Load and prepare test data."""
    df = pd.read_csv(DATASET_PATH)
    test_df = df[df['split'] == 'test'].copy()
    test_df['is_hate'] = test_df.apply(lambda row: 1 if row['label'] == 'hate' else 0, axis=1)
    return test_df

def evaluate_classifier(classifier_func, test_df, classifier_name, endpoint_id=None, project_id=None):
    """Evaluate a classifier and return metrics."""
    logger.info(f"Evaluating {classifier_name}...")
    
    predictions = []
    true_labels = []
    errors = []
    
    # Process each test example
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"Evaluating {classifier_name}"):
        text = row['text']
        true_label = row['is_hate']
        
        try:
            if classifier_name == "Tuned LLM":
                if not endpoint_id or not project_id:
                    raise ValueError("Endpoint ID and Project ID required for Tuned LLM")
                # Use the endpoint directly for tuned model
                result = classifier_func(text, endpoint_id, project_id)
                pred = result['is_hate_speech']
            else:
                # For non-tuned models, use the base classifier
                pred = classifier_func(text)
            
            predictions.append(int(pred))
            true_labels.append(true_label)
            
        except Exception as e:
            error_msg = f"Error processing example: {str(e)}"
            logger.error(error_msg)
            errors.append({
                'text': text,
                'error': str(e)
            })
            continue
    
    # Calculate metrics
    if not predictions:
        logger.error(f"No successful predictions for {classifier_name}")
        return None
        
    report = classification_report(true_labels, predictions, output_dict=True)
    conf_matrix = confusion_matrix(true_labels, predictions)
    
    # Calculate additional metrics
    total = len(predictions)
    correct = sum(1 for p, t in zip(predictions, true_labels) if p == t)
    accuracy = correct / total if total > 0 else 0
    
    # Calculate confidence scores
    confidence_scores = []
    for pred, true in zip(predictions, true_labels):
        if pred == true:
            confidence_scores.append(1.0)
        else:
            confidence_scores.append(0.0)
    
    avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
    
    metrics = {
        'classifier_name': classifier_name,
        'classification_report': report,
        'confusion_matrix': conf_matrix.tolist(),
        'accuracy': accuracy,
        'average_confidence': avg_confidence,
        'total_examples': total,
        'correct_predictions': correct,
        'error_rate': 1 - accuracy,
        'errors': errors[:10]  # Store first 10 errors for analysis
    }
    
    return metrics

def plot_confusion_matrix(conf_matrix, classifier_name, save_path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {classifier_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()

def save_metrics(metrics, classifier_name):
    """Save metrics to JSON file."""
    if metrics is None:
        logger.error(f"No metrics to save for {classifier_name}")
        return
        
    os.makedirs(METRICS_DIR, exist_ok=True)
    output_path = os.path.join(METRICS_DIR, f'{classifier_name.lower().replace(" ", "_")}_metrics.json')
    
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Also save confusion matrix plot
    plot_confusion_matrix(
        np.array(metrics['confusion_matrix']),
        classifier_name,
        os.path.join(METRICS_DIR, f'{classifier_name.lower().replace(" ", "_")}_confusion_matrix.png')
    )

def compare_classifiers(metrics_list):
    """Compare and summarize results from all classifiers."""
    comparison = {
        'accuracy': {},
        'f1_score': {},
        'precision': {},
        'recall': {},
        'error_rate': {},
        'average_confidence': {}
    }
    
    for metrics in metrics_list:
        if metrics is None:
            continue
            
        name = metrics['classifier_name']
        report = metrics['classification_report']
        
        comparison['accuracy'][name] = metrics['accuracy']
        comparison['f1_score'][name] = report['weighted avg']['f1-score']
        comparison['precision'][name] = report['weighted avg']['precision']
        comparison['recall'][name] = report['weighted avg']['recall']
        comparison['error_rate'][name] = metrics['error_rate']
        comparison['average_confidence'][name] = metrics['average_confidence']
    
    # Save comparison
    with open(os.path.join(METRICS_DIR, 'classifier_comparison.json'), 'w') as f:
        json.dump(comparison, f, indent=2)
    
    return comparison

def main():
    """Main evaluation function."""
    # Initialize Vertex AI
    vertex_initialized = initialize_vertex_ai()
    
    # Load test data
    test_df = load_test_data()
    
    # Evaluate each classifier
    metrics_list = []
    
    # Get endpoint and project info
    endpoint_id = os.getenv('VERTEX_ENDPOINT_ID')
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
    
    # 1. LLM Classifier (only if we have proper initialization)
    if vertex_initialized:
        llm_metrics = evaluate_classifier(llm_classification, test_df, "LLM Classifier")
        if llm_metrics:
            metrics_list.append(llm_metrics)
            save_metrics(llm_metrics, "LLM Classifier")
    else:
        logger.warning("Skipping LLM Classifier due to Vertex AI initialization failure")
    
    # 2. Combined Classifier
    combined_metrics = evaluate_classifier(combined_classification, test_df, "Combined Classifier")
    if combined_metrics:
        metrics_list.append(combined_metrics)
        save_metrics(combined_metrics, "Combined Classifier")
    
    # 3. Tuned LLM Classifier
    if vertex_initialized and endpoint_id and project_id:
        tuned_metrics = evaluate_classifier(
            tuned_llm_classification,
            test_df,
            "Tuned LLM",
            endpoint_id=endpoint_id,
            project_id=project_id
        )
        if tuned_metrics:
            metrics_list.append(tuned_metrics)
            save_metrics(tuned_metrics, "Tuned LLM")
    else:
        logger.warning("Skipping Tuned LLM Classifier - missing endpoint_id or project_id")
    
    # Compare and summarize results
    if metrics_list:
        comparison = compare_classifiers(metrics_list)
        
        # Print summary
        logger.info("\nClassifier Comparison Summary:")
        for metric, values in comparison.items():
            logger.info(f"\n{metric.upper()}:")
            for classifier, value in values.items():
                logger.info(f"{classifier}: {value:.4f}")
    else:
        logger.error("No classifiers were successfully evaluated")

if __name__ == "__main__":
    main() 