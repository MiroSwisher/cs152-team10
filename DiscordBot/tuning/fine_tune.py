
import time

import vertexai
from vertexai.tuning import sft

# TODO(developer): Update and un-comment below line
# PROJECT_ID = "your-project-id"
vertexai.init(project=PROJECT_ID, location="us-central1")

sft_tuning_job = sft.train(
    source_model="gemini-2.0-flash-001",
    # 1.5 and 2.0 models use the same JSONL format
    train_dataset="gs://cloud-samples-data/ai-platform/generative_ai/gemini-1_5/text/sft_train_data.jsonl",
)

# Polling for job completion
while not sft_tuning_job.has_ended:
    time.sleep(60)
    sft_tuning_job.refresh()

print(sft_tuning_job.tuned_model_name)
print(sft_tuning_job.tuned_model_endpoint_name)
print(sft_tuning_job.experiment)
# Example response:
# projects/123456789012/locations/us-central1/models/1234567890@1
# projects/123456789012/locations/us-central1/endpoints/123456789012345
# <google.cloud.aiplatform.metadata.experiment_resources.Experiment object at 0x7b5b4ae07af0>
