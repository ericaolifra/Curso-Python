from google.colab import drive
drive.mount('/content/drive')
Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).
In [2]:
# Load the TensorBoard notebook extension
%load_ext tensorboard
In [3]:
#Import tensorflow and all depencies
import tensorflow as tf
import numpy as np
import datetime

print(tf.__version__)
2.5.0
In [4]:
# Clear any logs from previous runs
!rm -rf ./logs/
In [5]:
#import fashion MNIST and make the split
fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Normalize the data
train_images = train_images.astype(np.float32) / 255
test_images = test_images.astype(np.float32) / 255
train_images = np.expand_dims(train_images, -1)
test_images = np.expand_dims(test_images, -1)
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-labels-idx1-ubyte.gz
32768/29515 [=================================] - 0s 0us/step
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-images-idx3-ubyte.gz
26427392/26421880 [==============================] - 0s 0us/step
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-labels-idx1-ubyte.gz
8192/5148 [===============================================] - 0s 0us/step
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-images-idx3-ubyte.gz
4423680/4422102 [==============================] - 0s 0us/step
In [6]:
#build the actual model
from tensorflow.keras import layers

input_shape = (28, 28, 1)

model = tf.keras.Sequential(
    [
        tf.keras.Input(shape=input_shape),
        layers.Conv2D(6, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(6, kernel_size=(3, 3), activation="relu"),
        layers.Flatten(),
        layers.Dense(10, activation="softmax"),
    ]
)

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(train_images, train_labels, epochs=20, validation_split=0.2)
WARNING:tensorflow:Please add `keras.layers.InputLayer` instead of `keras.Input` to Sequential model. `keras.Input` is intended to be used by Functional model.
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 26, 26, 6)         60        
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 13, 13, 6)         0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 11, 11, 6)         330       
_________________________________________________________________
flatten (Flatten)            (None, 726)               0         
_________________________________________________________________
dense (Dense)                (None, 10)                7270      
=================================================================
Total params: 7,660
Trainable params: 7,660
Non-trainable params: 0
_________________________________________________________________
Epoch 1/20
/usr/local/lib/python3.7/dist-packages/tensorflow/python/keras/backend.py:4930: UserWarning: "`sparse_categorical_crossentropy` received `from_logits=True`, but the `output` argument was produced by a sigmoid or softmax activation and thus does not represent logits. Was this intended?"
  '"`sparse_categorical_crossentropy` received `from_logits=True`, but '
1500/1500 [==============================] - 22s 14ms/step - loss: 0.5776 - accuracy: 0.7942 - val_loss: 0.4500 - val_accuracy: 0.8442
Epoch 2/20
1500/1500 [==============================] - 21s 14ms/step - loss: 0.4102 - accuracy: 0.8554 - val_loss: 0.3899 - val_accuracy: 0.8662
Epoch 3/20
1500/1500 [==============================] - 21s 14ms/step - loss: 0.3719 - accuracy: 0.8686 - val_loss: 0.3849 - val_accuracy: 0.8657
Epoch 4/20
1500/1500 [==============================] - 21s 14ms/step - loss: 0.3496 - accuracy: 0.8761 - val_loss: 0.3526 - val_accuracy: 0.8771
Epoch 5/20
1500/1500 [==============================] - 22s 15ms/step - loss: 0.3334 - accuracy: 0.8826 - val_loss: 0.3711 - val_accuracy: 0.8671
Epoch 6/20
1500/1500 [==============================] - 21s 14ms/step - loss: 0.3229 - accuracy: 0.8844 - val_loss: 0.3357 - val_accuracy: 0.8827
Epoch 7/20
1500/1500 [==============================] - 21s 14ms/step - loss: 0.3110 - accuracy: 0.8907 - val_loss: 0.3327 - val_accuracy: 0.8806
Epoch 8/20
1500/1500 [==============================] - 21s 14ms/step - loss: 0.3044 - accuracy: 0.8913 - val_loss: 0.3169 - val_accuracy: 0.8883
Epoch 9/20
1500/1500 [==============================] - 21s 14ms/step - loss: 0.2945 - accuracy: 0.8949 - val_loss: 0.3299 - val_accuracy: 0.8821
Epoch 10/20
1500/1500 [==============================] - 21s 14ms/step - loss: 0.2907 - accuracy: 0.8962 - val_loss: 0.3100 - val_accuracy: 0.8896
Epoch 11/20
1500/1500 [==============================] - 21s 14ms/step - loss: 0.2820 - accuracy: 0.8993 - val_loss: 0.3143 - val_accuracy: 0.8882
Epoch 12/20
1500/1500 [==============================] - 22s 14ms/step - loss: 0.2781 - accuracy: 0.9009 - val_loss: 0.3229 - val_accuracy: 0.8854
Epoch 13/20
1500/1500 [==============================] - 21s 14ms/step - loss: 0.2713 - accuracy: 0.9037 - val_loss: 0.3192 - val_accuracy: 0.8866
Epoch 14/20
1500/1500 [==============================] - 21s 14ms/step - loss: 0.2686 - accuracy: 0.9032 - val_loss: 0.3105 - val_accuracy: 0.8908
Epoch 15/20
1500/1500 [==============================] - 21s 14ms/step - loss: 0.2638 - accuracy: 0.9053 - val_loss: 0.2981 - val_accuracy: 0.8949
Epoch 16/20
1500/1500 [==============================] - 21s 14ms/step - loss: 0.2594 - accuracy: 0.9073 - val_loss: 0.3015 - val_accuracy: 0.8934
Epoch 17/20
1500/1500 [==============================] - 21s 14ms/step - loss: 0.2548 - accuracy: 0.9089 - val_loss: 0.3093 - val_accuracy: 0.8906
Epoch 18/20
1500/1500 [==============================] - 21s 14ms/step - loss: 0.2530 - accuracy: 0.9101 - val_loss: 0.3060 - val_accuracy: 0.8921
Epoch 19/20
1500/1500 [==============================] - 21s 14ms/step - loss: 0.2503 - accuracy: 0.9100 - val_loss: 0.3009 - val_accuracy: 0.8927
Epoch 20/20
1500/1500 [==============================] - 23s 16ms/step - loss: 0.2478 - accuracy: 0.9116 - val_loss: 0.3111 - val_accuracy: 0.8889
Out[6]:
<tensorflow.python.keras.callbacks.History at 0x7f818040e290>
In [7]:
#check logs
!kill 320
%tensorboard --logdir logs/fit
/bin/bash: line 0: kill: (320) - No such process
<IPython.core.display.Javascript object>
In [8]:
model.save("model1")
INFO:tensorflow:Assets written to: model1/assets