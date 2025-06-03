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

def evaluate_model(num_examples=50):
    """
    Evaluate the model on a specified number of examples.
    
    Args:
        num_examples (int): Number of examples to evaluate (default: 50)
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
    
    # Process each example
    for index, row in test_df.iterrows():
        text = row['text']
        true_label = row['label']
        
        try:
            # Get model prediction
            result = classifier.classify_message(text)
            prediction = extract_json_from_text(result)
            
            if prediction is None:
                print(f"\nExample {index + 1}/{num_examples}: Failed to parse model response")
                print(f"Response: {result}")
                continue
            
            # Compare prediction with true label
            predicted_hate = prediction['is_hate_speech']
            true_hate = (true_label == 'hate')
            
            # Store labels for confusion matrix
            true_labels.append(true_hate)
            predicted_labels.append(predicted_hate)
            
            if predicted_hate == true_hate:
                correct += 1
            total += 1
            
            # Print progress
            print(f"\nExample {index + 1}/{num_examples}:")
            print(f"Text: {text}")
            print(f"True label: {true_label}")
            print(f"Predicted: {prediction}")
            print(f"Correct: {predicted_hate == true_hate}")
            
            # Add a small delay to avoid rate limiting
            time.sleep(1)
            
        except Exception as e:
            print(f"Error processing example {index + 1}: {str(e)}")
            continue
    
    # Calculate and print accuracy
    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"\nFinal Results:")
    print(f"Correct predictions: {correct}")
    print(f"Total examples: {total}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    # Create and plot confusion matrix
    labels = ['Not Hate', 'Hate']
    plot_confusion_matrix(true_labels, predicted_labels, labels)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(true_labels, predicted_labels, target_names=labels))

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Evaluate hate speech classifier on a specified number of examples.')
    parser.add_argument('-n', '--num_examples', type=int, default=50,
                      help='Number of examples to evaluate (default: 50)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run evaluation with specified number of examples
    evaluate_model(args.num_examples)

if __name__ == "__main__":
    main() 