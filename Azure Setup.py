from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

# Set up Azure Machine Learning Workspace
credential = DefaultAzureCredential()
ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)

# Set up Cognitive Services resources
from azure.ai.formrecognizer import FormRecognizerClient
from azure.ai.textanalytics import TextAnalyticsClient
from azure.cognitiveservices.vision.computervision import ComputerVisionClient

form_recognizer_client = FormRecognizerClient(endpoint, credential)
text_analytics_client = TextAnalyticsClient(endpoint, credential)
computer_vision_client = ComputerVisionClient(endpoint, credential)
