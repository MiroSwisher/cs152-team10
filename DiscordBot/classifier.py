# Classifier for hate-speech severity.
# Dataset: Vidgen et al. (2021) "Dynamically Generated Hate Speech Dataset", ACL 2021.
# CC-BY 4.0 license.
# Please cite: Bertie Vidgen et al., Proceedings of ACL 2021: https://arxiv.org/abs/2012.15761
import os
import json
from sklearn.metrics import classification_report
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from hate_speech_classifier import HateSpeechClassifier
import logging
import vertexai
from vertexai.preview.generative_models import GenerativeModel

# Paths
BASE_DIR = os.path.dirname(__file__)
DATASET_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', 'Dynamically-Generated-Hate-Speech-Dataset-main', 'Dynamically Generated Hate Dataset v0.2.3.csv'))
VECTORIZER_PATH = os.path.join(BASE_DIR, 'vectorizer.pkl')
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')
METRICS_PATH = os.path.join(BASE_DIR, 'metrics.json')

# Configure logger
logger = logging.getLogger(__name__)


def binary_mapping(label, type_str):
    # Map dataset label to binary classification (0: non-hate, 1: hate)
    return 1 if label == 'hate' else 0


def train_and_save():
    # Read dataset including split
    df = pd.read_csv(DATASET_PATH, usecols=['text', 'label', 'type', 'split'])
    # Map to binary
    df['is_hate'] = df.apply(lambda row: binary_mapping(row['label'], row['type']), axis=1)
    # Use provided splits: train for training, test for evaluation
    df_train = df[df['split'] == 'train']
    df_test = df[df['split'] == 'test']
    X_train = df_train['text'].astype(str)
    y_train = df_train['is_hate']
    X_test = df_test['text'].astype(str)
    y_test = df_test['is_hate']
    # Train vectorizer and classifier
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2), stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(X_train_vec, y_train)
    # Save model and vectorizer
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(clf, MODEL_PATH)
    # Evaluate on test set and save metrics
    X_test_vec = vectorizer.transform(X_test)
    preds = clf.predict(X_test_vec)
    report_dict = classification_report(y_test, preds, output_dict=True)
    # Serialize parameters as strings to ensure JSON compatibility
    classifier_params = {k: str(v) for k, v in clf.get_params().items()}
    vectorizer_params = {k: str(v) for k, v in vectorizer.get_params().items()}
    metrics = {
        'classification_report': report_dict,
        'classifier_params': classifier_params,
        'vectorizer_params': vectorizer_params
    }
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)


def load_model():
    # Load or train model if not present
    if not os.path.isfile(VECTORIZER_PATH) or not os.path.isfile(MODEL_PATH):
        train_and_save()
    vectorizer = joblib.load(VECTORIZER_PATH)
    clf = joblib.load(MODEL_PATH)
    return vectorizer, clf


def predict_hate_speech(text, vectorizer, clf):
    # Predict if text contains hate speech (binary)
    vec = vectorizer.transform([text])
    return int(clf.predict(vec)[0])


def combined_classification(text: str, verbose: bool = False) -> dict:
    """
    Combines predictions from both traditional ML and LLM classifiers.
    
    Args:
        text (str): The message to classify
        verbose (bool): Whether to return detailed output with confidence scores
        
    Returns:
        dict: Combined classification result containing:
            - is_hate_speech (bool): Whether the message is classified as hate speech
            - traditional_prediction (bool): Prediction from traditional classifier
            - llm_prediction (bool): Prediction from LLM classifier
            - confidence (str): 'high' if both classifiers agree, 'medium' if they disagree
            - combined_confidence (float): Combined confidence score (only in verbose mode)
    """
    # Get traditional ML prediction
    vectorizer, clf = load_model()
    trad_pred = bool(predict_hate_speech(text, vectorizer, clf))
    
    # Get LLM prediction with verbose output if requested
    vertex_classifier = HateSpeechClassifier()
    vertex_result = vertex_classifier.classify_message(text, verbose=verbose)
    
    # Log the raw response for debugging
    logger.info(f"Raw LLM response: {vertex_result}")
    
    # Parse Vertex AI result
    try:
        if isinstance(vertex_result, str):
            import json
            # Clean the response - remove markdown code block syntax
            cleaned_response = vertex_result.strip()
            if cleaned_response.startswith('```'):
                # Remove the opening ```json or ``` and closing ```
                cleaned_response = cleaned_response.split('\n', 1)[1]  # Remove first line
                cleaned_response = cleaned_response.rsplit('\n', 1)[0]  # Remove last line
                cleaned_response = cleaned_response.strip()
            
            logger.info(f"Attempting to parse JSON: {cleaned_response}")
            vertex_result = json.loads(cleaned_response)
        
        # Get severity from result
        llm_severity = vertex_result.get('severity', 0)
        llm_pred = bool(llm_severity > 0)
        
        # Smart combination strategy:
        # 1. If both classifiers agree, use their prediction
        # 2. If they disagree, prefer the LLM prediction since it has better accuracy
        # 3. For edge cases (very short messages or specific patterns), use traditional ML
        is_hate_speech = llm_pred if trad_pred != llm_pred else trad_pred
        
        # Edge case handling
        if len(text.split()) <= 2:  # Very short messages
            is_hate_speech = trad_pred  # Traditional ML often better with short texts
        
        result = {
            'traditional_prediction': trad_pred,
            'llm_prediction': llm_pred,
            'is_hate_speech': is_hate_speech,
            'confidence': 'high' if trad_pred == llm_pred else 'medium'
        }
        
        if verbose:
            # Calculate combined confidence based on agreement and LLM confidence
            if trad_pred == llm_pred:
                # Full agreement: higher confidence
                result['combined_confidence'] = 0.9
            else:
                # Use LLM confidence if available, otherwise default to medium
                llm_confidence = vertex_result.get('confidence', 0.5)
                result['combined_confidence'] = llm_confidence
        
        return result
        
    except Exception as e:
        # If there's any error in parsing or processing, fall back to traditional ML
        logger.error(f"Error in combined classification: {str(e)}")
        logger.error(f"Raw response was: {vertex_result}")
        return {
            'traditional_prediction': trad_pred,
            'llm_prediction': False,
            'is_hate_speech': trad_pred,
            'confidence': 'low'
        }


def llm_classification(text: str) -> bool:
    """
    Uses only the Vertex AI (LLM) classifier to determine if a message contains hate speech.
    
    Args:
        text (str): The message to classify
        
    Returns:
        bool: True if the message is classified as hate speech, False otherwise
    """
    vertex_classifier = HateSpeechClassifier()
    result = vertex_classifier.classify_message(text)
    
    # Parse the result string into a dictionary
    if isinstance(result, str):
        import json
        result = json.loads(result)
    
    # Convert severity to binary
    severity = result.get('severity', 0)
    return bool(severity > 0)


def tuned_llm_classification(text: str, endpoint_id: str, project_id: str, location: str = "us-central1") -> dict:
    """
    Uses a tuned Vertex AI model endpoint to classify text for hate speech.
    
    Args:
        text (str): The message to classify
        endpoint_id (str): The ID of the tuned model endpoint
        project_id (str): Google Cloud project ID
        location (str): Location of the endpoint (default: "us-central1")
        
    Returns:
        dict: Classification result containing:
            - is_hate_speech (bool): Whether the message is classified as hate speech
            - severity (int): Severity level (0-4)
            - explanation (str): Explanation of the classification
    """
    try:
        # Initialize Vertex AI
        vertexai.init(project=project_id, location=location)
        
        # Get the model from the endpoint
        model = GenerativeModel.from_tuned_model(endpoint_id)
        
        # Create the prompt
        prompt = f"Analyze this text for hate speech: {text}"
        
        # Get response from the model
        response = model.generate_content(prompt)
        
        # Parse the response
        try:
            result = json.loads(response.text)
        except json.JSONDecodeError:
            # If response is not valid JSON, try to extract JSON from text
            import re
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                raise ValueError(f"Could not parse response as JSON: {response.text}")
        
        # Ensure the result has the expected format
        if not isinstance(result, dict):
            raise ValueError(f"Expected dictionary response, got: {type(result)}")
        
        # Extract required fields with defaults
        is_hate_speech = result.get('is_hate_speech', False)
        severity = result.get('severity', 0)
        explanation = result.get('explanation', 'No explanation provided')
        
        return {
            'is_hate_speech': bool(is_hate_speech),
            'severity': int(severity),
            'explanation': str(explanation)
        }
        
    except Exception as e:
        logger.error(f"Error in tuned model classification: {str(e)}")
        # Return a safe default response
        return {
            'is_hate_speech': False,
            'severity': 0,
            'explanation': f"Error in classification: {str(e)}"
        } 