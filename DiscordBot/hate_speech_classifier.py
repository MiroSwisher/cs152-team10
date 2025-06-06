import os
import json
import logging
from typing import Dict, Any
from google.cloud import aiplatform
from vertexai.preview.generative_models import GenerativeModel
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HateSpeechClassifier:
    def __init__(self, project_id=None, location='us-west1'):
        """Initialize the classifier with the endpoint model."""
        self.project_id = project_id or os.getenv('GOOGLE_CLOUD_PROJECT')
        self.location = location
        self.endpoint_id = os.getenv('VERTEX_ENDPOINT_ID')
        self.model = self.load_model()
        
    def load_model(self):
        """Load the model from the endpoint."""
        try:
            # Initialize Vertex AI
            aiplatform.init(project=self.project_id, location=self.location)
            
            if not self.endpoint_id:
                raise ValueError("VERTEX_ENDPOINT_ID environment variable not set")
            
            logger.info(f"Loading model from endpoint: {self.endpoint_id}")
            return GenerativeModel.from_tuned_model(self.endpoint_id)
                
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def classify_message(self, message: str, verbose: bool = False) -> Dict[str, Any]:
        """
        Classify if a message contains hate speech and determine its severity level.
        
        Args:
            message (str): The message to classify
            verbose (bool): Whether to return detailed output with confidence scores
            
        Returns:
            Dict[str, Any]: Classification result containing:
                - severity (int): Severity level (0-4):
                    0: Non-Hateful
                    1: Mild Hate (animosity)
                    2: Moderate Hate (derogation, dehumanization)
                    3: Severe Hate (threatening)
                    4: Extremist Hate (support for hate)
                - confidence (float): Confidence score of the classification
        """
        try:
            # Format the prompt for the model
            prompt = f"""Classify the following message for hate speech:
            Message: {message}
            
            Return a JSON object with:
            - is_hate_speech (boolean)
            - severity (integer 0-4)
            - confidence (float 0-1)
            """
            
            # Get prediction from model
            response = self.model.generate_content(prompt)
            
            # Parse the response
            try:
                # Clean the response - remove markdown code block syntax if present
                cleaned_response = response.text.strip()
                if cleaned_response.startswith('```'):
                    cleaned_response = cleaned_response.split('\n', 1)[1]  # Remove first line
                    cleaned_response = cleaned_response.rsplit('\n', 1)[0]  # Remove last line
                    cleaned_response = cleaned_response.strip()
                
                result = json.loads(cleaned_response)
                
                # Ensure all required fields are present
                if not all(k in result for k in ['is_hate_speech', 'severity', 'confidence']):
                    raise ValueError("Missing required fields in model response")
                
                return {
                    'severity': int(result['severity']),
                    'is_hate_speech': bool(result['is_hate_speech']),
                    'confidence': float(result['confidence'])
                }
                
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing model response: {str(e)}")
                logger.error(f"Raw response: {response.text}")
                return {'severity': 0, 'is_hate_speech': False, 'confidence': 0.5}
                
        except Exception as e:
            logger.error(f"Error in classify_message: {str(e)}")
            return {'severity': 0, 'is_hate_speech': False, 'confidence': 0.5}

def main():
    # Example usage
    try:
        # You can either set the project ID here
        # os.environ["GOOGLE_CLOUD_PROJECT"] = "your-project-id"
        
        classifier = HateSpeechClassifier()
        
        # Example messages to test
        test_messages = [
            "I love this community!",
            "You're all worthless and should be eliminated",
            "Let's work together to make things better",
            "I don't like your kind around here",
            "We should remove all of them from our society",
            "I hate this weather",
            "I hate when that happens",
            "I disagree with your opinion",
            "This is a bad idea",
            "You're an idiot",
            "Your people are inferior",
            "I hope you die",
            "Death to all of them"
        ]

        for message in test_messages:
            print(f"\nAnalyzing message: {message}")
            result = classifier.classify_message(message)
            print(f"Classification result: {result}")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Make sure you have set up Google Cloud authentication:")
        print("   gcloud auth application-default login")
        print("2. Verify your project ID is set:")
        print("   gcloud config get-value project")
        print("3. Ensure you have enabled the Vertex AI API in your project")
        print("   gcloud services enable aiplatform.googleapis.com")

if __name__ == "__main__":
    main() 