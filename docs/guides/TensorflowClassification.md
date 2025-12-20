
## Models from Tensorflow/Keras API

**⚠️ Export tools have moved to vision-core for reusability across inference engines.**

Select a model from https://keras.io/api/applications/, for example resnet50, then export to saved model format:

```bash
# Clone vision-core if not already available
git clone https://github.com/olibartfast/vision-core.git
cd vision-core/export/classification/tensorflow

# Use the provided export script
python export_tf_saved_model_classifier.py
```

Or manually:

```python
import tensorflow as tf
from tensorflow import keras

# Load the pretrained ResNet model
model = keras.applications.ResNet50(weights='imagenet')

# Specify the path where the SavedModel will be stored (no file extension needed)
saved_model_path = 'model.savedmodel'

model.export(saved_model_path)
```

See [vision-core export documentation](https://github.com/olibartfast/vision-core/tree/main/export) for detailed usage.