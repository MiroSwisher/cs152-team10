import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import logging
from classifier import load_model, predict_severity, llm_classification, tuned_llm_classification
from vertexai import init
from vertexai.generative_models import GenerativeModel
import time
from tqdm import tqdm
from datetime import datetime
import concurrent.futures
from typing import List, Dict, Any
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = os.path.dirname(__file__)
DATASET_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', 'Dynamically-Generated-Hate-Speech-Dataset-main', 'Dynamically Generated Hate Dataset v0.2.3.csv'))
RESULTS_DIR = os.path.join(BASE_DIR, 'evaluation_metrics')

def load_test_data(sample_size: int = None) -> pd.DataFrame:
    """Load the test split from the original dataset."""
    logger.info("Loading test data...")
    df = pd.read_csv(DATASET_PATH, usecols=['text', 'label', 'type', 'split'])
    df_test = df[df['split'] == 'test']
    
    if sample_size:
        df_test = df_test.sample(n=sample_size, random_state=42)
        logger.info(f"Using {sample_size} random samples from test set")
    
    logger.info(f"Loaded {len(df_test)} test examples")
    return df_test

def evaluate_traditional_classifier(df_test: pd.DataFrame) -> Dict[str, Any]:
    """Evaluate the traditional ML classifier."""
    logger.info("Evaluating traditional classifier...")
    start_time = time.time()
    
    vectorizer, clf = load_model()
    
    # Get predictions
    X_test = df_test['text'].astype(str)
    y_true = df_test['label'].apply(lambda x: 1 if x == 'hate' else 0)
    
    # Use tqdm for progress bar
    y_pred = []
    for text in tqdm(X_test, desc="Traditional Classifier"):
        y_pred.append(predict_severity(text, vectorizer, clf))
    
    # Calculate metrics
    report = classification_report(y_true, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Traditional classifier evaluation completed in {elapsed_time:.2f} seconds")
    
    return {
        'classification_report': report,
        'confusion_matrix': conf_matrix.tolist(),
        'evaluation_time': elapsed_time
    }

def process_llm_batch(texts: List[str], model_type: str) -> List[int]:
    """Process a batch of texts using either base or tuned LLM."""
    predictions = []
    for text in texts:
        try:
            if model_type == 'base':
                result = llm_classification(text)
                if isinstance(result, str):
                    result = json.loads(result)
                predictions.append(result.get('severity', 0))
            else:  # tuned
                prompt = f"""Analyze this text for hate speech: {text}

Please classify the severity of hate speech in this text on a scale from 0-4:
0: Non-Hateful
1: Mild Hate (animosity)
2: Moderate Hate (derogation, dehumanization)
3: Severe Hate (threatening)
4: Extremist Hate (support for hate)

Return ONLY a single number (0-4) representing the severity level, with no additional text or explanation."""

                response = model.generate_content(
                    prompt,
                    generation_config={"temperature": 0.2, "max_output_tokens": 128}
                )
                
                response_text = response.text.strip()
                if response_text:
                    numbers = re.findall(r'\d+', response_text)
                    if numbers:
                        severity = int(numbers[0])
                        predictions.append(severity if 0 <= severity <= 4 else 0)
                    else:
                        predictions.append(0)
                else:
                    predictions.append(0)
            
            time.sleep(0.2)  # Reduced rate limiting
        except Exception as e:
            logger.error(f"Error processing text: {str(e)}")
            predictions.append(0)
    
    return predictions

def evaluate_llm(df_test: pd.DataFrame, model_type: str = 'base') -> Dict[str, Any]:
    """Evaluate either base or tuned LLM classifier."""
    logger.info(f"Evaluating {model_type} LLM classifier...")
    start_time = time.time()
    
    if model_type == 'tuned':
        # Load configuration and initialize tuned model
        with open('tokens.json', 'r') as f:
            config = json.load(f)
            project_id = config['PROJECT']
            region = config['REGION']
            endpoint = config['ENDPOINT']
        init(project=project_id, location=region)
        global model
        model = GenerativeModel(model_name=endpoint)
    
    y_true = df_test['label'].apply(lambda x: 1 if x == 'hate' else 0)
    y_pred = []
    
    # Process in smaller batches with progress bar
    batch_size = 20  # Smaller batches for more frequent progress updates
    n_batches = (len(df_test) + batch_size - 1) // batch_size
    
    with tqdm(total=len(df_test), desc=f"{model_type.capitalize()} LLM") as pbar:
        for i in range(0, len(df_test), batch_size):
            batch = df_test.iloc[i:i+batch_size]
            batch_texts = batch['text'].tolist()
            
            batch_predictions = process_llm_batch(batch_texts, model_type)
            y_pred.extend(batch_predictions)
            
            pbar.update(len(batch_texts))
            
            # Log progress
            if (i + batch_size) % 100 == 0:
                elapsed = time.time() - start_time
                remaining = (elapsed / (i + batch_size)) * (len(df_test) - (i + batch_size))
                logger.info(f"Processed {i + batch_size}/{len(df_test)} examples. "
                          f"Elapsed: {elapsed:.1f}s, Remaining: {remaining:.1f}s")
    
    # Calculate metrics
    report = classification_report(y_true, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    elapsed_time = time.time() - start_time
    logger.info(f"{model_type.capitalize()} LLM evaluation completed in {elapsed_time:.2f} seconds")
    
    return {
        'classification_report': report,
        'confusion_matrix': conf_matrix.tolist(),
        'evaluation_time': elapsed_time
    }

def main():
    # Create results directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Load test data (optionally with sample size for faster testing)
    sample_size = 500  # Set to a number (e.g., 100) for faster testing
    df_test = load_test_data(sample_size)
    
    # Evaluate each classifier
    traditional_metrics = evaluate_traditional_classifier(df_test)
    base_llm_metrics = evaluate_llm(df_test, 'base')
    tuned_llm_metrics = evaluate_llm(df_test, 'tuned')
    
    # Save results with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        'timestamp': timestamp,
        'sample_size': sample_size if sample_size else len(df_test),
        'traditional_classifier': traditional_metrics,
        'base_llm': base_llm_metrics,
        'tuned_llm': tuned_llm_metrics
    }
    
    output_file = os.path.join(RESULTS_DIR, f'test_split_evaluation_{timestamp}.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Evaluation complete! Results saved to {output_file}")
    
    # Print summary
    print("\nEvaluation Summary:")
    print(f"Total examples evaluated: {len(df_test)}")
    print(f"Traditional Classifier time: {traditional_metrics['evaluation_time']:.2f}s")
    print(f"Base LLM time: {base_llm_metrics['evaluation_time']:.2f}s")
    print(f"Tuned LLM time: {tuned_llm_metrics['evaluation_time']:.2f}s")

if __name__ == "__main__":
    main() 