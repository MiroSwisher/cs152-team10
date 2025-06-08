import pandas as pd
import json
import time
import re
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
from DiscordBot.classifier import load_model, predict_severity, llm_classification, combined_classification

def plot_confusion_matrix(y_true, y_pred, labels, title, filename):
    """Plot confusion matrix with seaborn."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(filename)
    plt.close()

def plot_confidence_histogram(confidences, correct_predictions, title, filename):
    """Plot histogram of confidence scores for correct vs incorrect predictions."""
    plt.figure(figsize=(10, 6))
    
    # Separate confidences for correct and incorrect predictions
    correct_conf = [conf for conf, correct in zip(confidences, correct_predictions) if correct]
    incorrect_conf = [conf for conf, correct in zip(confidences, correct_predictions) if not correct]
    
    # Plot histograms
    plt.hist(correct_conf, bins=20, alpha=0.5, label='Correct Predictions', color='green')
    plt.hist(incorrect_conf, bins=20, alpha=0.5, label='Incorrect Predictions', color='red')
    
    plt.title(title)
    plt.xlabel('Confidence Score')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig(filename)
    plt.close()

def evaluate_traditional_classifier(test_df, verbose=False):
    """Evaluate the traditional ML classifier."""
    print("\nEvaluating Traditional ML Classifier...")
    
    vectorizer, clf = load_model()
    true_labels = []
    predicted_labels = []
    confidences = []
    correct_predictions = []
    
    for index, row in test_df.iterrows():
        text = row['text']
        true_label = (row['label'] == 'hate')
        
        # Get prediction
        severity = predict_severity(text, vectorizer, clf)
        predicted_hate = (severity > 0)
        
        # Store results
        true_labels.append(true_label)
        predicted_labels.append(predicted_hate)
        
        if verbose:
            # Use severity as confidence (normalized to 0-1)
            confidence = min(severity / 4.0, 1.0)
            confidences.append(confidence)
            correct_predictions.append(predicted_hate == true_label)
    
    # Calculate metrics
    accuracy = sum(1 for t, p in zip(true_labels, predicted_labels) if t == p) / len(true_labels) * 100
    
    print(f"Accuracy: {accuracy:.2f}%")
    print("\nClassification Report:")
    print(classification_report(true_labels, predicted_labels, target_names=['Not Hate', 'Hate']))
    
    # Plot confusion matrix
    plot_confusion_matrix(true_labels, predicted_labels, ['Not Hate', 'Hate'],
                         'Traditional ML Classifier Confusion Matrix',
                         'traditional_confusion_matrix.png')
    
    if verbose and confidences:
        plot_confidence_histogram(confidences, correct_predictions,
                                'Traditional ML Classifier Confidence Distribution',
                                'traditional_confidence_histogram.png')

def evaluate_llm_classifier(test_df, verbose=False):
    """Evaluate the LLM classifier."""
    print("\nEvaluating LLM Classifier...")
    
    true_labels = []
    predicted_labels = []
    confidences = []
    correct_predictions = []
    
    for index, row in test_df.iterrows():
        text = row['text']
        true_label = (row['label'] == 'hate')
        
        try:
            # Get prediction
            result = llm_classification(text)
            predicted_hate = result
            
            # Store results
            true_labels.append(true_label)
            predicted_labels.append(predicted_hate)
            
            if verbose:
                # For LLM, we don't have confidence scores in non-verbose mode
                confidences.append(1.0)  # Placeholder
                correct_predictions.append(predicted_hate == true_label)
            
            # Add a small delay to avoid rate limiting
            time.sleep(1)
            
        except Exception as e:
            print(f"Error processing example {index + 1}: {str(e)}")
            continue
    
    # Calculate metrics
    accuracy = sum(1 for t, p in zip(true_labels, predicted_labels) if t == p) / len(true_labels) * 100
    
    print(f"Accuracy: {accuracy:.2f}%")
    print("\nClassification Report:")
    print(classification_report(true_labels, predicted_labels, target_names=['Not Hate', 'Hate']))
    
    # Plot confusion matrix
    plot_confusion_matrix(true_labels, predicted_labels, ['Not Hate', 'Hate'],
                         'LLM Classifier Confusion Matrix',
                         'llm_confusion_matrix.png')

def evaluate_combined_classifier(test_df, verbose=False):
    """Evaluate the combined classifier."""
    print("\nEvaluating Combined Classifier...")
    
    true_labels = []
    predicted_labels = []
    confidences = []
    correct_predictions = []
    
    for index, row in test_df.iterrows():
        text = row['text']
        true_label = (row['label'] == 'hate')
        
        try:
            # Get prediction
            result = combined_classification(text, verbose=verbose)
            predicted_hate = result['is_hate_speech']
            
            # Store results
            true_labels.append(true_label)
            predicted_labels.append(predicted_hate)
            
            if verbose and 'combined_confidence' in result:
                confidences.append(result['combined_confidence'])
                correct_predictions.append(predicted_hate == true_label)
            
            # Add a small delay to avoid rate limiting
            time.sleep(1)
            
        except Exception as e:
            print(f"Error processing example {index + 1}: {str(e)}")
            continue
    
    # Calculate metrics
    accuracy = sum(1 for t, p in zip(true_labels, predicted_labels) if t == p) / len(true_labels) * 100
    
    print(f"Accuracy: {accuracy:.2f}%")
    print("\nClassification Report:")
    print(classification_report(true_labels, predicted_labels, target_names=['Not Hate', 'Hate']))
    
    # Plot confusion matrix
    plot_confusion_matrix(true_labels, predicted_labels, ['Not Hate', 'Hate'],
                         'Combined Classifier Confusion Matrix',
                         'combined_confusion_matrix.png')
    
    if verbose and confidences:
        plot_confidence_histogram(confidences, correct_predictions,
                                'Combined Classifier Confidence Distribution',
                                'combined_confidence_histogram.png')

def evaluate_models(num_examples=50, verbose=False):
    """
    Evaluate all three classifiers on a specified number of examples.
    
    Args:
        num_examples (int): Number of examples to evaluate (default: 50)
        verbose (bool): Whether to show detailed output (default: False)
    """
    # Load the CSV file
    df = pd.read_csv("assets/Dynamically Generated Hate Dataset v0.2.2.csv")
    
    # Take specified number of examples
    test_df = df.head(num_examples)
    
    print(f"\nStarting evaluation on {num_examples} examples...")
    print("Press Ctrl+C at any time to stop and see results.")
    
    try:
        # Evaluate each classifier
        evaluate_traditional_classifier(test_df, verbose)
        evaluate_llm_classifier(test_df, verbose)
        evaluate_combined_classifier(test_df, verbose)
        
    except KeyboardInterrupt:
        print("\nEvaluation stopped by user.")
        sys.exit(0)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Evaluate all hate speech classifiers on a specified number of examples.')
    parser.add_argument('-n', '--num_examples', type=int, default=50,
                      help='Number of examples to evaluate (default: 50)')
    parser.add_argument('-v', '--verbose', action='store_true',
                      help='Show detailed output including confidence scores')
    
    # Parse arguments
    args = parser.parse_args()
    
    try:
        # Run evaluation with specified number of examples
        evaluate_models(args.num_examples, args.verbose)
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
        sys.exit(0)

if __name__ == "__main__":
    main() 