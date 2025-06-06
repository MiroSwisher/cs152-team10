import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from classifier import load_model, predict_severity
from hate_speech_classifier import HateSpeechClassifier
import logging
from tqdm import tqdm
from vertexai import init
from vertexai.generative_models import GenerativeModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation.log'),
        logging.StreamHandler()
    ]
)

# Paths and Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SMALL_VAL_PATH = os.path.join(BASE_DIR, 'small_val_data.jsonl')
VAL_DATASET_PATH = os.path.join(BASE_DIR, 'val_data.jsonl')
TEST_DATASET_PATH = os.path.join(BASE_DIR, 'test_data.jsonl')
METRICS_DIR = os.path.join(BASE_DIR, 'evaluation_metrics')
os.makedirs(METRICS_DIR, exist_ok=True)

# Tuned model configuration
PROJECT = "cs152-bot-461705"
REGION = "us-west1"
ENDPOINT = "projects/cs152-bot-461705/locations/us-west1/endpoints/8783387065837420544"

# Initialize Vertex AI
init(project=PROJECT, location=REGION)
tuned_model = GenerativeModel(model_name=ENDPOINT)

def load_jsonl_data(file_path):
    """
    Load and process data from a JSONL file.
    
    Args:
        file_path (str): Path to the JSONL file
        
    Returns:
        pd.DataFrame: DataFrame with 'text' and 'label' columns
    """
    if not os.path.exists(file_path):
        logging.warning(f"File not found: {file_path}")
        return pd.DataFrame(columns=['text', 'label'])
        
    texts = []
    labels = []
    
    try:
        df = pd.read_json(file_path, lines=True)
        
        for _, row in df.iterrows():
            # Get the text from the user's message
            text = row['contents'][0]['parts'][0]['text'].replace("Analyze this text for hate speech: ", "")
            
            # Get the severity from the model's response
            response = row['contents'][1]['parts'][0]['text']
            response_json = json.loads(response)
            severity = response_json['severity']
            
            texts.append(text)
            labels.append(severity)
            
        return pd.DataFrame({'text': texts, 'label': labels})
    except Exception as e:
        logging.error(f"Error loading {file_path}: {str(e)}")
        return pd.DataFrame(columns=['text', 'label'])

def load_test_data(sample_size=250, balanced=True):
    """
    Load and prepare test data from both small validation and full validation sets.
    
    Args:
        sample_size (int): Total number of samples to use
        balanced (bool): If True, try to create a balanced dataset with equal samples per severity level
        
    Returns:
        pd.DataFrame: DataFrame with 'text' and 'label' columns
    """
    logging.info("Loading test data...")
    
    # Load both validation datasets
    small_val_df = load_jsonl_data(SMALL_VAL_PATH)
    val_df = load_jsonl_data(VAL_DATASET_PATH)
    
    # Combine datasets
    df = pd.concat([small_val_df, val_df], ignore_index=True)
    
    # Remove duplicates if any
    df = df.drop_duplicates(subset=['text'])
    
    # Print distribution of labels
    label_dist = df['label'].value_counts().sort_index()
    logging.info("\nInitial combined label distribution:")
    for severity, count in label_dist.items():
        logging.info(f"Severity {severity}: {count} samples")
    
    if balanced:
        # Calculate samples per class
        n_classes = 5  # 0-4 severity levels
        if sample_size < n_classes:
            logging.warning(f"Sample size {sample_size} is less than number of classes {n_classes}. Setting to {n_classes}.")
            sample_size = n_classes
            
        target_per_class = min(sample_size // n_classes, label_dist.min())
        logging.info(f"\nBalancing dataset with {target_per_class} samples per severity level")
        
        # Sample equally from each class
        balanced_samples = []
        for severity in range(5):
            severity_samples = df[df['label'] == severity]
            if len(severity_samples) > target_per_class:
                severity_samples = severity_samples.sample(n=target_per_class, random_state=42)
            balanced_samples.append(severity_samples)
        
        # Combine balanced samples
        df = pd.concat(balanced_samples, ignore_index=True)
        
        # Print final distribution
        final_dist = df['label'].value_counts().sort_index()
        logging.info("\nFinal balanced label distribution:")
        for severity, count in final_dist.items():
            logging.info(f"Severity {severity}: {count} samples")
            
    elif sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    
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
                # Use non-verbose mode to get simpler output
                result = classifier.classify_message(text, verbose=False)
                if isinstance(result, str):
                    result = json.loads(result)
                predictions.append(result['severity'])
            except Exception as e:
                logging.error(f"Error classifying text with LLM: {str(e)}\nText: {text[:100]}...")
                predictions.append(0)  # Default to non-hateful on error
                
    else:  # tuned_llm
        for text in tqdm(texts, desc="Tuned LLM"):
            try:
                # Create the prompt for severity classification
                prompt = f"""Analyze this text for hate speech: {text}

Please classify the severity of hate speech in this text on a scale from 0-4:
0: Non-Hateful
1: Mild Hate (animosity)
2: Moderate Hate (derogation, dehumanization)
3: Severe Hate (threatening)
4: Extremist Hate (support for hate)

Return ONLY a single number (0-4) representing the severity level, with no additional text or explanation."""

                # Get response from tuned model
                response = tuned_model.generate_content(
                    prompt,
                    generation_config={"temperature": 0.2, "max_output_tokens": 128}
                )
                
                # Parse response - expect just a number
                try:
                    severity = int(response.text.strip())
                    if 0 <= severity <= 4:
                        predictions.append(severity)
                    else:
                        logging.warning(f"Invalid severity {severity} from tuned LLM for text: {text[:100]}...")
                        predictions.append(0)
                except ValueError as e:
                    logging.error(f"Failed to parse tuned LLM response as integer: {response.text}\nError: {str(e)}")
                    predictions.append(0)
                    
            except Exception as e:
                logging.error(f"Error in tuned LLM classification: {str(e)}\nText: {text[:100]}...")
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
    # Load test data (sample 250 examples)
    df = load_test_data(sample_size=250)
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