import os
import json
import pandas as pd
from tqdm import tqdm
import logging
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_preparation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = os.path.dirname(__file__)
DATASET_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', 'Dynamically-Generated-Hate-Speech-Dataset-main', 'Dynamically Generated Hate Dataset v0.2.3.csv'))
TRAIN_JSONL = os.path.join(BASE_DIR, 'train_data.jsonl')
VAL_JSONL = os.path.join(BASE_DIR, 'val_data.jsonl')
SMALL_TRAIN_JSONL = os.path.join(BASE_DIR, 'small_train_data.jsonl')
SMALL_VAL_JSONL = os.path.join(BASE_DIR, 'small_val_data.jsonl')

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

def validate_example_format(example):
    """Validate that the example follows Gemini's required format."""
    # Check top-level structure
    required_keys = ["systemInstruction", "contents"]
    if not all(key in example for key in required_keys):
        raise ValueError(f"Example missing required keys: {required_keys}")
    
    # Validate systemInstruction
    system_instruction = example["systemInstruction"]
    if not isinstance(system_instruction, dict):
        raise ValueError("systemInstruction must be a dictionary")
    if "parts" not in system_instruction:
        raise ValueError("systemInstruction must have parts")
    if not isinstance(system_instruction["parts"], list):
        raise ValueError("systemInstruction parts must be a list")
    for part in system_instruction["parts"]:
        if not isinstance(part, dict) or "text" not in part:
            raise ValueError("Each systemInstruction part must be a dictionary with a text field")
    
    # Validate contents
    if not isinstance(example["contents"], list):
        raise ValueError("contents must be a list")
    
    valid_roles = {"user", "model", "function"}
    for message in example["contents"]:
        if not isinstance(message, dict):
            raise ValueError("Each message must be a dictionary")
        
        if "role" not in message:
            raise ValueError("Each message must have a role")
        
        if message["role"] not in valid_roles:
            raise ValueError(f"Invalid role: {message['role']}. Must be one of: {valid_roles}")
        
        if "parts" not in message:
            raise ValueError("Each message must have parts")
        
        if not isinstance(message["parts"], list):
            raise ValueError("parts must be a list")
        
        for part in message["parts"]:
            if not isinstance(part, dict) or "text" not in part:
                raise ValueError("Each part must be a dictionary with a text field")
    
    return True

def create_tuning_example(text, label, type_str):
    """Create a single tuning example in the required format."""
    # Create the system instruction
    system_instruction = """You are a hate speech detection model. Your task is to analyze text and determine if it contains hate speech, and if so, classify its severity level.

Severity levels:
0: No hate speech
1: Animosity (mild hate)
2: Derogation or Dehumanization (moderate hate)
3: Threatening (severe hate)
4: Support for hate (endorsement of hate)

Provide your analysis in JSON format with:
1. is_hate_speech: true/false
2. severity: 0-4 (as defined above)
3. explanation: brief explanation of your classification"""

    # Create the user prompt
    user_prompt = f"Analyze this text for hate speech: {text}"

    # Create the expected response
    response = {
        "is_hate_speech": label == 'hate',
        "severity": severity_mapping(label, type_str),
        "explanation": f"This text is classified as {'hate speech' if label == 'hate' else 'not hate speech'} with severity level {severity_mapping(label, type_str)}."
    }

    # Format the example according to Gemini's requirements
    example = {
        "systemInstruction": {
            "parts": [{"text": system_instruction}]
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_prompt}]
            },
            {
                "role": "model",
                "parts": [{"text": json.dumps(response)}]
            }
        ]
    }

    return example

def write_examples_to_file(examples, output_file):
    """Write examples to a JSONL file with validation."""
    logger.info(f"Writing {len(examples)} examples to {output_file}")
    with open(output_file, 'w') as f:
        for i, example in enumerate(examples):
            try:
                # Validate format
                validate_example_format(example)
                # Write to file
                f.write(json.dumps(example) + '\n')
            except Exception as e:
                logger.error(f"Error in example {i}: {str(e)}")
                raise

def prepare_small_datasets(df):
    """Prepare smaller datasets for cost-effective testing."""
    logger.info("Preparing small datasets for cost-effective testing...")
    
    # Calculate sizes for ~$10 cost
    SMALL_TRAIN_SIZE = 3000  # 75% of 4K
    SMALL_VAL_SIZE = 1000    # 25% of 4K
    
    # Split hate and non-hate examples
    hate_df = df[df['label'] == 'hate']
    non_hate_df = df[df['label'] != 'hate']
    
    # Sample from each class proportionally
    train_hate = hate_df.sample(n=int(SMALL_TRAIN_SIZE * 0.5), random_state=42)
    train_non_hate = non_hate_df.sample(n=int(SMALL_TRAIN_SIZE * 0.5), random_state=42)
    val_hate = hate_df.drop(train_hate.index).sample(n=int(SMALL_VAL_SIZE * 0.5), random_state=42)
    val_non_hate = non_hate_df.drop(train_non_hate.index).sample(n=int(SMALL_VAL_SIZE * 0.5), random_state=42)
    
    # Combine and shuffle
    small_train_df = pd.concat([train_hate, train_non_hate]).sample(frac=1, random_state=42)
    small_val_df = pd.concat([val_hate, val_non_hate]).sample(frac=1, random_state=42)
    
    # Create examples
    train_examples = [create_tuning_example(row['text'], row['label'], row['type']) 
                     for _, row in tqdm(small_train_df.iterrows(), desc="Creating small training examples")]
    val_examples = [create_tuning_example(row['text'], row['label'], row['type']) 
                   for _, row in tqdm(small_val_df.iterrows(), desc="Creating small validation examples")]
    
    # Write examples to files
    write_examples_to_file(train_examples, SMALL_TRAIN_JSONL)
    write_examples_to_file(val_examples, SMALL_VAL_JSONL)
    
    # Print statistics
    logger.info("\nSmall Dataset Statistics:")
    logger.info(f"Small training set: {len(train_hate)} hate, {len(train_non_hate)} non-hate examples")
    logger.info(f"Small validation set: {len(val_hate)} hate, {len(val_non_hate)} non-hate examples")
    
    # Calculate estimated cost
    total_examples = len(small_train_df) + len(small_val_df)
    estimated_cost = (total_examples / 1000) * 2.50  # $2.50 per 1000 examples
    logger.info(f"\nEstimated cost for small dataset: ${estimated_cost:.2f}")

def prepare_datasets():
    """Prepare training and validation datasets for Gemini tuning."""
    logger.info("Loading dataset...")
    df = pd.read_csv(DATASET_PATH, usecols=['text', 'label', 'type', 'split'])
    
    # Split into train and validation sets (80-20 split)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    
    logger.info(f"Training set size: {len(train_df)}")
    logger.info(f"Validation set size: {len(val_df)}")
    
    # Create examples
    train_examples = [create_tuning_example(row['text'], row['label'], row['type']) 
                     for _, row in tqdm(train_df.iterrows(), desc="Creating training examples")]
    val_examples = [create_tuning_example(row['text'], row['label'], row['type']) 
                   for _, row in tqdm(val_df.iterrows(), desc="Creating validation examples")]
    
    # Write examples to files
    write_examples_to_file(train_examples, TRAIN_JSONL)
    write_examples_to_file(val_examples, VAL_JSONL)
    
    # Print full dataset statistics
    train_hate = len(train_df[train_df['label'] == 'hate'])
    train_non_hate = len(train_df[train_df['label'] != 'hate'])
    val_hate = len(val_df[val_df['label'] == 'hate'])
    val_non_hate = len(val_df[val_df['label'] != 'hate'])
    
    logger.info("\nFull Dataset Statistics:")
    logger.info(f"Full training set: {train_hate} hate, {train_non_hate} non-hate examples")
    logger.info(f"Full validation set: {val_hate} hate, {val_non_hate} non-hate examples")
    
    # Prepare small datasets
    prepare_small_datasets(df)
    
    # Print example of the format
    logger.info("\nExample format:")
    example = create_tuning_example(
        "This is an example text",
        "nothate",
        "none"
    )
    logger.info(json.dumps(example, indent=2))

if __name__ == "__main__":
    prepare_datasets() 