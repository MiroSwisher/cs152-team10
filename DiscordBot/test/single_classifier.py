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

# Paths
BASE_DIR = os.path.dirname(__file__)
DATASET_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', 'Dynamically-Generated-Hate-Speech-Dataset-main', 'Dynamically Generated Hate Dataset v0.2.3.csv'))
VECTORIZER_PATH = os.path.join(BASE_DIR, 'vectorizer.pkl')
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')
METRICS_PATH = os.path.join(BASE_DIR, 'metrics.json')


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
    # Map severity
    df['severity'] = df.apply(lambda row: severity_mapping(row['label'], row['type']), axis=1)
    # Use provided splits: train for training, test for evaluation
    df_train = df[df['split'] == 'train']
    df_test = df[df['split'] == 'test']
    X_train = df_train['text'].astype(str)
    y_train = df_train['severity']
    X_test = df_test['text'].astype(str)
    y_test = df_test['severity']
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