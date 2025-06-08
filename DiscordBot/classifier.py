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
import logging
import vertexai
from vertexai.preview.generative_models import GenerativeModel
from hate_speech_classifier import HateSpeechClassifier


# Paths
BASE_DIR = os.path.dirname(__file__)
DATASET_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', 'Dynamically-Generated-Hate-Speech-Dataset-main', 'Dynamically Generated Hate Dataset v0.2.3.csv'))
VECTORIZER_PATH = os.path.join(BASE_DIR, 'vectorizer.pkl')
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')
METRICS_PATH = os.path.join(BASE_DIR, 'metrics.json')

# Configure logger
logger = logging.getLogger(__name__)

# were we aware of how bad this dataset is? a ton of examples are simply not given granularity levels
def severity_mapping(label, type_str):
    # Map dataset label and type to severity levels
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


def predict_severity(text, vectorizer, clf):
    # Predict severity for a single text input
    vec = vectorizer.transform([text])
    return int(clf.predict(vec)[0])


def llm_classification(text: str) -> dict:
    """
    Uses only the Vertex AI (LLM) classifier to determine if a message contains hate speech.
    
    Args:
        text (str): The message to classify
        
    Returns:
        dict: Classification result containing:
            - severity (int): Severity level (0-4):
                0: Non-Hateful
                1: Mild Hate (animosity)
                2: Moderate Hate (derogation, dehumanization)
                3: Severe Hate (threatening)
                4: Extremist Hate (support for hate)
            - is_hate_speech (bool): Whether the message is classified as hate speech
            - confidence (float): Confidence score of the classification
    """
    vertex_classifier = HateSpeechClassifier()
    result = vertex_classifier.classify_message(text)
    
    # Parse the result string into a dictionary if needed
    if isinstance(result, str):
        import json
        result = json.loads(result)
    
    return result


def combined_classification(text: str, verbose: bool = False) -> dict:
    """
    Combines predictions from both traditional ML and LLM classifiers.
    
    Args:
        text (str): The message to classify
        verbose (bool): Whether to return detailed output with confidence scores
        
    Returns:
        dict: Combined classification result containing:
            - severity (int): Severity level (0-4)
            - traditional_severity (int): Severity from traditional classifier
            - llm_severity (int): Severity from LLM classifier
            - confidence (str): 'high' if both classifiers agree, 'medium' if they disagree
            - combined_confidence (float): Combined confidence score (only in verbose mode)
    """
    # Get traditional ML prediction
    vectorizer, clf = load_model()
    trad_severity = predict_severity(text, vectorizer, clf)
    
    # Get LLM prediction
    llm_result = llm_classification(text)
    if isinstance(llm_result, str):
        import json
        llm_result = json.loads(llm_result)
    llm_severity = llm_result.get('severity', 0)
    
    # Take max severity between classifiers
    final_severity = max(trad_severity, llm_severity)
    
    result = {
        'traditional_severity': trad_severity,
        'llm_severity': llm_severity,
        'severity': final_severity,
        'confidence': 'high' if trad_severity == llm_severity else 'medium'
    }
    
    if verbose:
        # Calculate combined confidence based on agreement
        result['combined_confidence'] = 0.9 if trad_severity == llm_severity else 0.5
    
    return result


def tuned_llm_classification(text: str) -> dict:
    """
    Classify text using the tuned Gemini model on Vertex AI.
    Configuration is loaded from tokens.json.
    
    Args:
        text (str): Text to classify
        
    Returns:
        dict: Classification results with keys:
            - severity (int): Severity level (0-4)
            - explanation (str): Explanation of the classification
    """
    try:
        # Load configuration from tokens.json
        with open('tokens.json', 'r') as f:
            config = json.load(f)
            project_id = config['PROJECT']
            region = config['REGION']
            endpoint = config['ENDPOINT']
        
        # Initialize Vertex AI
        vertexai.init(project=project_id, location=region)
        
        # Load the tuned model
        model = GenerativeModel(model_name=endpoint)
        
        # Create the prompt
        prompt = f"""Analyze this text for hate speech: {text}

Please classify the severity of hate speech in this text on a scale from 0-4:
0: Non-Hateful
1: Mild Hate (animosity)
2: Moderate Hate (derogation, dehumanization)
3: Severe Hate (threatening)
4: Extremist Hate (support for hate)

Respond in JSON format with:
- severity (int): The severity level (0-4)
- explanation (str): Brief explanation of the classification"""
        
        # Get response from model
        response = model.generate_content(prompt)
        
        # Parse response
        try:
            result = json.loads(response.text)
            # Ensure severity is an integer
            result['severity'] = int(result.get('severity', 0))
            return result
        except json.JSONDecodeError:
            logger.error(f"Failed to parse tuned LLM response as JSON: {response.text}")
            return {'severity': 0, 'explanation': 'Failed to parse model response'}
            
    except Exception as e:
        logger.error(f"Error in tuned LLM classification: {str(e)}")
        return {'severity': 0, 'explanation': f'Error: {str(e)}'} 