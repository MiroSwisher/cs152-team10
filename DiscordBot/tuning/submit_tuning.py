import os
import time
import logging
import vertexai
from vertexai.tuning import sft
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tuning_submission.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def upload_to_gcs(local_file, gcs_bucket, project_id):
    """Upload a file to Google Cloud Storage."""
    gcs_path = f"gs://{gcs_bucket}/{os.path.basename(local_file)}"
    cmd = f"gsutil cp {local_file} {gcs_path}"
    logger.info(f"Uploading {local_file} to {gcs_path}")
    os.system(cmd)
    return gcs_path

def submit_tuning_job(project_id, train_data_gcs, val_data_gcs, location="us-central1"):
    """Submit a tuning job to Vertex AI."""
    logger.info("Initializing Vertex AI...")
    vertexai.init(project=project_id, location=location)
    
    logger.info("Submitting tuning job...")
    sft_tuning_job = sft.train(
        source_model="gemini-1.5-pro-preview-0409",
        train_dataset=train_data_gcs,
        validation_dataset=val_data_gcs,
        tuning_config={
            "max_training_steps": 500,
            "batch_size": 4,
            "learning_rate": 1e-5,
            "early_stopping_patience": 10,
            "evaluation_interval": 50
        }
    )
    
    logger.info(f"Tuning job submitted: {sft_tuning_job.name}")
    return sft_tuning_job

def monitor_job(tuning_job):
    """Monitor the tuning job progress."""
    logger.info("Monitoring tuning job progress...")
    while not tuning_job.has_ended:
        time.sleep(60)  # Check every minute
        tuning_job.refresh()
        logger.info(f"Job status: {tuning_job.state}")
    
    logger.info("Tuning job completed!")
    logger.info(f"Tuned model name: {tuning_job.tuned_model_name}")
    logger.info(f"Tuned model endpoint: {tuning_job.tuned_model_endpoint_name}")
    logger.info(f"Experiment details: {tuning_job.experiment}")

def main():
    parser = argparse.ArgumentParser(description="Submit Gemini tuning job")
    parser.add_argument('--project-id', required=True, help='Google Cloud project ID')
    parser.add_argument('--gcs-bucket', required=True, help='GCS bucket name (without gs:// prefix)')
    parser.add_argument('--location', default='us-central1', help='Vertex AI location')
    args = parser.parse_args()
    
    # Upload datasets to GCS
    train_data_gcs = upload_to_gcs('train_data.jsonl', args.gcs_bucket, args.project_id)
    val_data_gcs = upload_to_gcs('val_data.jsonl', args.gcs_bucket, args.project_id)
    
    # Submit and monitor tuning job
    tuning_job = submit_tuning_job(
        project_id=args.project_id,
        train_data_gcs=train_data_gcs,
        val_data_gcs=val_data_gcs,
        location=args.location
    )
    
    monitor_job(tuning_job)

if __name__ == "__main__":
    main() 