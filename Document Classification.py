from azure.ai.ml import MLClient, Input, Output
from azure.ai.ml.entities import Data, Environment, CodeConfiguration
from azure.ai.ml.constants import AssetTypes

# Set up data and environment
data = Data(
    path="data/",
    type=AssetTypes.URI_FOLDER,
    description="Document classification dataset",
)

env = Environment(
    name="doc-classification-env",
    conda_file="conda.yml",
    image="mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04",
)

# Set up code configuration
code = CodeConfiguration(
    code="src/",
    scoring_script="score.py",
)

# Create a job to train the model
job = ml_client.jobs.create_or_update(
    "doc-classification-job",
    inputs=dict(data=data.as_input()),
    code=code,
    environment=env,
)

# Wait for the job to complete
ml_client.jobs.stream(job.name)

# Register the trained model
model = ml_client.models.create_or_update(
    "doc-classification-model",
    path=job.outputs.model_output,
    type=AssetTypes.MLFLOW_MODEL,
)
