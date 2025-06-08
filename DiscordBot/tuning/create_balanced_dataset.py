import json
import random
from pathlib import Path
import pandas as pd

# System instruction that will be included with each example
SYSTEM_INSTRUCTION = {
    "systemInstruction": {
        "parts": [{
            "text": "You are a hate speech detection model. Your task is to analyze text and determine if it contains hate speech, and if so, classify its severity level.\n\nSeverity levels:\n0: No hate speech\n1: Animosity (mild hate)\n2: Derogation or Dehumanization (moderate hate)\n3: Threatening (severe hate)\n4: Support for hate (endorsement of hate)\n\nProvide your analysis in JSON format with:\n1. is_hate_speech: true/false\n2. severity: 0-4 (as defined above)"
        }]
    }
}

def map_type_to_severity(row):
    """Map hate speech type to severity level."""
    type_to_severity = {
        'none': 0,
        'animosity': 1,
        'derogation': 2,
        'dehumanization': 2,
        'threatening': 3,
        'support': 4
    }
    return type_to_severity[row['type']]

def create_example(text, severity):
    """Create a single training example in the correct format."""
    # Determine if it's hate speech based on severity
    is_hate_speech = severity > 0
    
    # Create the user message
    user_message = {
        "role": "user",
        "parts": [{
            "text": f"Analyze this text for hate speech: {text}"
        }]
    }
    
    # Create the model response
    model_response = {
        "role": "model",
        "parts": [{
            "text": json.dumps({
                "is_hate_speech": is_hate_speech,
                "severity": severity
            })
        }]
    }
    
    # Combine into final format
    example = {
        "systemInstruction": SYSTEM_INSTRUCTION["systemInstruction"],
        "contents": [user_message, model_response]
    }
    
    return example

def main():
    # Read the filtered dataset
    df = pd.read_csv('DiscordBot/hate_speech_dataset_filtered.csv')
    
    # Map types to severity levels
    df['severity'] = df.apply(map_type_to_severity, axis=1)
    
    # Print initial statistics
    print("\nInitial Dataset Statistics:")
    print(f"Total examples: {len(df)}")
    print("\nDistribution by Type:")
    print(df['type'].value_counts())
    print("\nDistribution by Severity:")
    print(df['severity'].value_counts().sort_index())
    
    # Create balanced dataset with specified counts
    samples_per_class = {
        0: 1000,  # No hate speech
        1: 1000,  # Mild hate (animosity)
        2: 1000,  # Moderate hate (derogation + dehumanization)
        3: 500,   # Severe hate (threatening)
        4: 154    # Support for hate (all available)
    }
    
    balanced_data = []
    for severity, count in samples_per_class.items():
        # Get all samples for this severity
        severity_data = df[df['severity'] == severity]
        
        # If we have more samples than needed, randomly sample
        if len(severity_data) > count:
            severity_data = severity_data.sample(n=count, random_state=42)
        else:
            print(f"\nWarning: Only {len(severity_data)} samples available for severity {severity}, using all of them")
        
        balanced_data.append(severity_data)
    
    # Combine all samples
    balanced_df = pd.concat(balanced_data)
    
    # Create output directory if it doesn't exist
    output_dir = Path('DiscordBot/trainer')
    output_dir.mkdir(exist_ok=True)
    
    # Create examples and write to JSONL file
    with open(output_dir / 'train_data.jsonl', 'w') as f:
        for _, row in balanced_df.iterrows():
            example = create_example(row['text'], row['severity'])
            f.write(json.dumps(example) + '\n')
    
    # Print statistics
    print("\nBalanced Dataset Statistics:")
    print("Distribution by Severity:")
    for severity, count in balanced_df['severity'].value_counts().sort_index().items():
        print(f"Severity {severity}: {count} samples")
    print(f"Total samples: {len(balanced_df)}")

if __name__ == "__main__":
    main() 