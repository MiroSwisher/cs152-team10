import os
import pandas as pd
import json
import logging
from collections import Counter

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('prepare_dynamic.log'),
        logging.StreamHandler()
    ]
)

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DYNAMIC_DATA_PATH = os.path.join(BASE_DIR, 'Dynamically-Generated-Hate-Speech-Dataset-main', 'Dynamically Generated Hate Dataset v0.2.3.csv')
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weighted_train_data.jsonl')

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

def create_jsonl_example(text, severity):
    """Create a JSONL example in the format expected by our model."""
    return {
        "contents": [
            {
                "parts": [
                    {
                        "text": f"Analyze this text for hate speech: {text}"
                    }
                ]
            },
            {
                "parts": [
                    {
                        "text": json.dumps({
                            "severity": severity,
                            "explanation": "Generated from Dynamic Hate Dataset"
                        })
                    }
                ]
            }
        ]
    }

def process_dynamic_dataset():
    """Process the Dynamic Hate Dataset and convert to our format."""
    logging.info("Loading Dynamic Hate Dataset...")
    
    try:
        # Read the CSV file
        df = pd.read_csv(DYNAMIC_DATA_PATH)
        
        # Filter for training data only
        train_df = df[df['split'] == 'train'].copy()
        
        # Map severity levels
        train_df['severity'] = train_df.apply(
            lambda row: severity_mapping(row['label'], row['type']), 
            axis=1
        )
        
        # Count samples per severity level before sampling
        initial_dist = Counter(train_df['severity'])
        logging.info("\nInitial distribution:")
        for severity, count in sorted(initial_dist.items()):
            logging.info(f"Severity {severity}: {count} samples")
        
        # Create unbalanced dataset with specified distribution
        samples = []
        target_samples = {
            0: 1000,  # 1000 samples for level 0
            1: 1000,  # 1000 samples for level 1
            2: 1000,  # 1000 samples for level 2
            3: initial_dist[3],  # All samples for level 3
            4: initial_dist[4]   # All samples for level 4
        }
        
        for severity, target_count in target_samples.items():
            severity_samples = train_df[train_df['severity'] == severity]
            if len(severity_samples) >= target_count:
                samples.append(severity_samples.sample(n=target_count, random_state=42))
            else:
                logging.warning(
                    f"Severity {severity} has only {len(severity_samples)} samples, "
                    f"using all available samples."
                )
                samples.append(severity_samples)
        
        # Combine samples
        final_df = pd.concat(samples, ignore_index=True)
        
        # Calculate class weights for training
        total_samples = len(final_df)
        class_weights = {}
        final_dist = Counter(final_df['severity'])
        
        logging.info("\nFinal distribution and class weights:")
        for severity, count in sorted(final_dist.items()):
            weight = total_samples / (len(final_dist) * count)
            class_weights[severity] = weight
            logging.info(f"Severity {severity}: {count} samples, weight = {weight:.2f}")
        
        # Save class weights
        weights_path = os.path.join(os.path.dirname(OUTPUT_PATH), 'class_weights.json')
        with open(weights_path, 'w') as f:
            json.dump(class_weights, f, indent=2)
        logging.info(f"\nClass weights saved to {weights_path}")
        
        # Convert to JSONL format
        logging.info("\nConverting to JSONL format...")
        with open(OUTPUT_PATH, 'w') as f:
            for _, row in final_df.iterrows():
                example = create_jsonl_example(row['text'], row['severity'])
                f.write(json.dumps(example) + '\n')
        
        logging.info(f"Dataset saved to {OUTPUT_PATH}")
        
    except Exception as e:
        logging.error(f"Error processing dataset: {str(e)}")
        return False
    
    return True

def main():
    if process_dynamic_dataset():
        logging.info("\nDone! You can now use weighted_train_data.jsonl for fine-tuning.")
    else:
        logging.error("Failed to process dataset.")

if __name__ == "__main__":
    main() 