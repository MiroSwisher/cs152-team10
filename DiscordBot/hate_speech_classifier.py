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
                - confidence (float): Confidence score of the classification (only in verbose mode)
        """
        if verbose:
            prompt = f"""Analyze the following message and determine if it contains hate speech.
            Return ONLY a JSON object in this exact format, with no additional text or explanation:
            {{
                "severity": 0-4,
                "confidence": 0.0-1.0
            }}

            Severity levels:
            0: Non-Hateful - No hate speech detected
            1: Mild Hate - Contains animosity or mild hostility
            2: Moderate Hate - Contains derogation or dehumanization
            3: Severe Hate - Contains threats or severe hostility
            4: Extremist Hate - Expresses support for hate or extremist views

            Message to analyze: {message}
            """
        else:
            prompt = f"""Analyze the following message and determine if it contains hate speech.
            Return ONLY a single number (0-4) representing the severity level, with no additional text or explanation.

            Severity levels:
            0: Non-Hateful - No hate speech detected
            1: Mild Hate - Contains animosity or mild hostility
            2: Moderate Hate - Contains derogation or dehumanization
            3: Severe Hate - Contains threats or severe hostility
            4: Extremist Hate - Expresses support for hate or extremist views

            Message to analyze: {message}
            """

        try:
            response = self.model.generate_content(prompt)
            if verbose:
                return response.text
            else:
                # For non-verbose mode, convert the single number response to our standard format
                result = response.text.strip()
                try:
                    severity = int(result)
                    if 0 <= severity <= 4:
                        return f'{{"severity": {severity}}}'
                    else:
                        return '{"severity": 0}'  # Default to non-hateful if invalid severity
                except ValueError:
                    return '{"severity": 0}'  # Default to non-hateful if parsing fails
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
            "Let's work together to make things better",
            "I don't like your kind around here",
            "We should remove all of them from our society"
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