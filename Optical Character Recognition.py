from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from azure.core.exceptions import ResourceNotFoundError
import time

def ocr_document(image_url):
    read_results = []
    operation_location = None

    # Call the Computer Vision API to perform OCR
    read_results = computer_vision_client.read(image_url, raw=True)

    # Get the operation location and wait for the operation to complete
    operation_location = read_results.headers["Operation-Location"]
    operation_id = operation_location.split("/")[-1]

    while True:
        read_result = computer_vision_client.get_read_result(operation_id)
        if read_result.status not in [OperationStatusCodes.running]:
            break
        time.sleep(1)

    # Get the text from the OCR results
    if read_result.status == OperationStatusCodes.succeeded:
        for text_result in read_result.analyze_result.read_results:
            for line in text_result.lines:
                read_results.append(line.text)

    return "\n".join(read_results)
