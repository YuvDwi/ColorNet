import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dropout, LeakyReLU, BatchNormalization, Input, Concatenate, Activation, concatenate
from keras.initializers import RandomNormal
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
import cv2
import PIL
from PIL import Image
import random
import h5py
import plotly.graph_objects as go  

# Source: https://github.com/emilwallner/Coloring-greyscale-images/blob/master/Full-version/full_version.ipynb

# This project uses a transfer learning model that is generalized on a wide range of images
# Below is the model architecture which has taken partially from the above links due to my lack of experience with such models

# Importing the transfer learning model and preparing the kernel initializer values to initialise the weights and distribute them 
from tensorflow.keras.applications import MobileNetV2
weight_init = RandomNormal(stddev=0.02) 

# Preparing the Input layer
input_layer = Input((224, 224, 3))

# Downloading mobile net  trained on the imagenet dataset containing 1.4 million images and using it as the base of the model
# In using this model, we utilize its weights, but do not use its connected Dense Layer (#include_top=False), and instead build our own layer  atop the model
pre_trained_model = MobileNetV2(
    input_shape= (224, 224, 3),
    include_top=False, 
    weights='imagenet'
)
mobilenet = pre_trained_model(input_layer)

# Source: https://github.com/emilwallner/Coloring-greyscale-images/blob/master/Full-version/full_version.ipynb
# Defining the model architecture, (decoders and encoders)
# Encoder
# 224x224
conv1 = Conv2D(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer=weight_init)(input_layer)
conv1 = LeakyReLU(alpha=0.2)(conv1)

# 112x112
conv2 = Conv2D(128, (3, 3), strides=(1, 1), padding='same', kernel_initializer=weight_init)(conv1)
conv2 = LeakyReLU(alpha=0.2)(conv2)

# 112x112
conv3 = Conv2D(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer=weight_init)(conv2)
conv3 =  Activation('relu')(conv3)

# 56x56
conv4 = Conv2D(256, (3, 3), strides=(2, 2), padding='same', kernel_initializer=weight_init)(conv3)
conv4 = Activation('relu')(conv4)

# 28x28
conv4_ = Conv2D(256, (3, 3), strides=(1, 1), padding='same', kernel_initializer=weight_init)(conv4)
conv4_ = Activation('relu')(conv4_)

# 28x28
conv5 = Conv2D(512, (3, 3), strides=(2, 2), padding='same', kernel_initializer=weight_init)(conv4_)
conv5 = Activation('relu')(conv5)

# 14x14
conv5_ = Conv2D(256, (3, 3), strides=(2, 2), padding='same', kernel_initializer=weight_init)(conv5)
conv5_ = Activation('relu')(conv5_)

#7x7
# Fusion layer - Connects MobileNet with our encoder
conc = concatenate([mobilenet, conv5_])
fusion = Conv2D(512, (1, 1), padding='same', kernel_initializer=weight_init)(conc)
fusion = Activation('relu')(fusion)

# Skip fusion layer
skip_fusion = concatenate([fusion, conv5_])

# Decoder 
# 7x7
decoder = Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same', kernel_initializer=weight_init)(skip_fusion)
decoder = Activation('relu')(decoder)
decoder = Dropout(0.25)(decoder)

# Skip layer from conv5 (with added dropout)
skip_4_drop = Dropout(0.25)(conv5) # Drops 1/4 of the neurons to prevent overfitting 
skip_4 = concatenate([decoder, skip_4_drop])

# 14x14
decoder = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same', kernel_initializer=weight_init)(skip_4)
decoder = Activation('relu')(decoder)
decoder = Dropout(0.25)(decoder)

# Skip layer from conv4_ (with added dropout)
skip_3_drop = Dropout(0.25)(conv4_)
skip_3 = concatenate([decoder, skip_3_drop])

# 28x28
decoder = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', kernel_initializer=weight_init)(skip_3)
decoder = Activation('relu')(decoder)
decoder = Dropout(0.25)(decoder)

# 56x56
decoder = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', kernel_initializer=weight_init)(decoder)
decoder = Activation('relu')(decoder)
decoder = Dropout(0.25)(decoder)

# 112x112
decoder = Conv2DTranspose(64, (3, 3), strides=(1, 1), padding='same', kernel_initializer=weight_init)(decoder)
decoder = Activation('relu')(decoder)

# 112x112
decoder = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', kernel_initializer=weight_init)(decoder)
decoder = Activation('relu')(decoder)

# 224x224
# Ooutput layer, with 2 channels (a and b)
output_layer = Conv2D(2, (1, 1), activation='tanh')(decoder)

model = Model(input_layer, output_layer)
model.compile(Adam(lr=0.0002), loss='mse', metrics=['accuracy'])

model.load_weights('/Users/hamadsultan/Downloads/Image Colorization/saved_model_weights.h5')