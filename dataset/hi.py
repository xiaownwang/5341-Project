import tensorflow as tf
from tensorflow.python import keras
# from tensorflow.keras import layers
# from tensorflow.keras.applications import ResNet50V2
from tensorflow.python.keras.layers import Input, GlobalAveragePooling2D, Dense
from keras import layers
from tensorflow.keras.applications import ResNet50V2
# from keras.layers import Input, GlobalAveragePooling2D, Dense
#
# # Load pre-trained ResNet50V2 without top layers
base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(96, 128, 3))
print(base_model.summary())
import tensorflow as tf
print(tf.__version__)
from tensorflow.keras.applications import ResNet50V2
base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(96, 128, 3))