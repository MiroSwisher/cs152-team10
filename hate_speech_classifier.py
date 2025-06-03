import os
from typing import Dict, Any
from google.cloud import aiplatform
from vertexai.preview.generative_models import GenerativeModel, Part
from dotenv import load_dotenv

load_dotenv()

class HateSpeechClassifier:
    def __init__(self, project_id: str = None, location: str = "us-east4"):
        """
        Initialize the hate speech classifier using Vertex AI.
        
        Args:
            project_id (str, optional): Google Cloud project ID. If None, will try to get from environment.
            location (str): Google Cloud location/region
        """
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        if not self.project_id:
            raise ValueError(
                "Project ID not found. Please set it in one of these ways:\n"
                "1. Pass it to the constructor: HateSpeechClassifier(project_id='your-project-id')\n"
                "2. Set environment variable: export GOOGLE_CLOUD_PROJECT='your-project-id'\n"
                "3. Use gcloud: gcloud config set project your-project-id"
            )
        
        self.location = location
        
        # Initialize Vertex AI
        aiplatform.init(project=self.project_id, location=location)
        
        # Initialize the Gemini model
        self.model = GenerativeModel("gemini-2.0-flash-001")

    def classify_message(self, message: str) -> Dict[str, Any]:
        """
        Classify if a message contains hate speech.
        
        Args:
            message (str): The message to classify
            
        Returns:
            Dict[str, Any]: Classification result containing:
                - is_hate_speech (bool): Whether the message is classified as hate speech
                - confidence (float): Confidence score of the classification
        """
        prompt = f"""Analyze the following message and determine if it contains hate speech.
        Return ONLY a JSON object in this exact format, with no additional text or explanation:
        {{
            "is_hate_speech": true/false,
            "confidence": 0.0-1.0
        }}

        Message to analyze: {message}
        """

        try:
            response = self.model.generate_content(prompt)
            # Parse the response and return the classification
            # Note: In a production environment, you'd want to add more robust parsing
            return response.text
        except Exception as e:
            raise Exception(f"Error during classification: {str(e)}")

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
            "Let's work together to make things better"
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