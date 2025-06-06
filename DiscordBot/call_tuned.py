from vertexai import init
from vertexai.generative_models import GenerativeModel
from vertexai.tuning import sft         # only needed if you want to look up the job

PROJECT = "cs152-bot-461705"
REGION  = "us-west1"                 # same region you tuned in
ENDPOINT = "projects/cs152-bot-461705/locations/us-west1/endpoints/8783387065837420544"

# 1) Initialise once
init(project=PROJECT, location=REGION)

# (Optional)  Grab the endpoint programmatically from the tuning‑job ID
# job = sft.SupervisedTuningJob("projects/.../tuningJobs/TUNING_JOB_ID")
# ENDPOINT = job.tuned_model_endpoint_name    # prints the same string as above

# 2) Load the tuned model *by endpoint*
model = GenerativeModel(model_name=ENDPOINT)

# 3) Call it
response = model.generate_content(
    "Why is the sky blue?",
    generation_config={"temperature": 0.2, "max_output_tokens": 128},
)
print(response.text)
