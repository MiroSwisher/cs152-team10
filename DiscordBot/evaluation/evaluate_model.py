import pandas as pd
import json
from hate_speech_classifier import HateSpeechClassifier
import time
import re
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
import signal

def extract_json_from_text(text):
    """Extract JSON from text that might contain additional content."""
    # Remove markdown code block markers if present
    text = text.replace('```json', '').replace('```', '').strip()
    
    # Find JSON-like structure in the text
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return None
    return None

def plot_confusion_matrix(y_true, y_pred, labels):
    """Plot confusion matrix with seaborn."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()

def plot_confidence_histogram(confidences, correct_predictions):
    """Plot histogram of confidence scores for correct vs incorrect predictions."""
    plt.figure(figsize=(10, 6))
    
    # Separate confidences for correct and incorrect predictions
    correct_conf = [conf for conf, correct in zip(confidences, correct_predictions) if correct]
    incorrect_conf = [conf for conf, correct in zip(confidences, correct_predictions) if not correct]
    
    # Plot histograms
    plt.hist(correct_conf, bins=20, alpha=0.5, label='Correct Predictions', color='green')
    plt.hist(incorrect_conf, bins=20, alpha=0.5, label='Incorrect Predictions', color='red')
    
    plt.title('Distribution of Confidence Scores')
    plt.xlabel('Confidence Score')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig('confidence_histogram.png')
    plt.close()

def evaluate_model(num_examples=50, verbose=False):
    """
    Evaluate the model on a specified number of examples.
    
    Args:
        num_examples (int): Number of examples to evaluate (default: 50)
        verbose (bool): Whether to show detailed LLM output (default: False)
    """
    # Load the CSV file
    df = pd.read_csv("assets/Dynamically Generated Hate Dataset v0.2.2.csv")
    
    # Take specified number of examples
    test_df = df.head(num_examples)
    
    # Initialize the classifier
    classifier = HateSpeechClassifier()
    
    # Initialize counters and lists for confusion matrix
    correct = 0
    total = 0
    true_labels = []
    predicted_labels = []
    confidences = []
    correct_predictions = []
    
    print("\nStarting evaluation... Press Ctrl+C at any time to stop and see results.")
    
    # Process each example
    for index, row in test_df.iterrows():
        text = row['text']
        true_label = row['label']
        
        try:
            # Get model prediction
            result = classifier.classify_message(text, verbose=verbose)
            prediction = extract_json_from_text(result)
            
            if prediction is None:
                print(f"\nExample {index + 1}/{num_examples}: Failed to parse model response")
                if verbose:
                    print(f"Response: {result}")
                continue
            
            # Compare prediction with true label
            predicted_hate = prediction['is_hate_speech']
            true_hate = (true_label == 'hate')
            
            # Store labels for confusion matrix
            true_labels.append(true_hate)
            predicted_labels.append(predicted_hate)
            
            # Store confidence if in verbose mode
            if verbose and 'confidence' in prediction:
                confidences.append(prediction['confidence'])
                correct_predictions.append(predicted_hate == true_hate)
            
            if predicted_hate == true_hate:
                correct += 1
            total += 1
            
            # Print progress
            print(f"\nExample {index + 1}/{num_examples}:")
            print(f"Text: {text}")
            print(f"True label: {true_label}")
            if verbose:
                print(f"Full prediction: {prediction}")
                if 'confidence' in prediction:
                    print(f"Confidence: {prediction['confidence']:.2f}")
            print(f"Predicted: {1 if predicted_hate else 0}")
            print(f"Correct: {predicted_hate == true_hate}")
            
            # Add a small delay to avoid rate limiting
            time.sleep(1)
            
        except KeyboardInterrupt:
            print("\nEvaluation stopped by user.")
            break
        except Exception as e:
            print(f"Error processing example {index + 1}: {str(e)}")
            continue
    
    # Calculate and print accuracy
    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"\nFinal Results:")
    print(f"Correct predictions: {correct}")
    print(f"Total examples: {total}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    if total > 0:  # Only create confusion matrix if we have results
        # Get unique classes in the data
        unique_classes = sorted(set(true_labels + predicted_labels))
        labels = ['Not Hate', 'Hate']
        
        # Only create confusion matrix if we have at least one example of each class
        if len(unique_classes) > 1:
            plot_confusion_matrix(true_labels, predicted_labels, labels)
            print("\nClassification Report:")
            print(classification_report(true_labels, predicted_labels, target_names=labels))
            
            # If in verbose mode and we have confidence scores, analyze them
            if verbose and confidences:
                print("\nConfidence Analysis:")
                avg_conf_correct = np.mean([conf for conf, correct in zip(confidences, correct_predictions) if correct])
                avg_conf_incorrect = np.mean([conf for conf, correct in zip(confidences, correct_predictions) if not correct])
                print(f"Average confidence for correct predictions: {avg_conf_correct:.2f}")
                print(f"Average confidence for incorrect predictions: {avg_conf_incorrect:.2f}")
                
                # Plot confidence histogram
                plot_confidence_histogram(confidences, correct_predictions)
                print("\nConfidence histogram saved as 'confidence_histogram.png'")
        else:
            print("\nNote: Only one class was present in the evaluated examples.")
            print("Classification report and confusion matrix not generated.")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Evaluate hate speech classifier on a specified number of examples.')
    parser.add_argument('-n', '--num_examples', type=int, default=50,
                      help='Number of examples to evaluate (default: 50)')
    parser.add_argument('-v', '--verbose', action='store_true',
                      help='Show detailed LLM output including confidence scores and explanations')
    
    # Parse arguments
    args = parser.parse_args()
    
    try:
        # Run evaluation with specified number of examples
        evaluate_model(args.num_examples, args.verbose)
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
        sys.exit(0)

if __name__ == "__main__":
    main() 