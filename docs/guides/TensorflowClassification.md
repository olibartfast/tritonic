
## Models from Tensorflow/Keras API
  * Select a model from https://keras.io/api/applications/, for example resnet50, then export to saved model format

```python
import tensorflow as tf
from tensorflow import keras

# Load the pretrained ResNet model
model = keras.applications.ResNet50(weights='imagenet')

# Specify the path where the SavedModel will be stored (no file extension needed)
saved_model_path = 'model.savedmodel'

model.export(saved_model_path)
```