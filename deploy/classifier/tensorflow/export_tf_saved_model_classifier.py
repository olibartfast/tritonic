import tensorflow as tf
from tensorflow import keras

# Load the pretrained ResNet model
model = keras.applications.ResNet50(weights='imagenet')

# Specify the path where the SavedModel will be stored (no file extension needed)
saved_model_path = 'model.savedmodel'

model.export(saved_model_path)

print(f"Model successfully exported to {saved_model_path}")


# import tritonclient.http as httpclient
# from tritonclient.utils import InferenceServerException
# import numpy as np

# # Connect to Triton Inference Server
# triton_client = httpclient.InferenceServerClient(url="localhost:8000")

# # Prepare input data (example for a single image input)
# input_data = np.random.rand(1, 224, 224, 3).astype(np.float32)  # Dummy image data

# # Create input and output tensors
# input_tensor = httpclient.InferInput("inputs", input_data.shape, "FP32")
# input_tensor.set_data_from_numpy(input_data)

# output_tensor = httpclient.InferRequestedOutput("output_0")

# # Perform inference
# response = triton_client.infer("resnet50_tensorflow", inputs=[input_tensor], outputs=[output_tensor])

# # Get the results
# result = response.as_numpy("output_0")
# print(result)