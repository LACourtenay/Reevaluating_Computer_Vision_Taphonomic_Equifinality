import tensorflow as tf
from tensorflow.keras.applications import ResNet50, DenseNet201, VGG16, InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.initializers import HeUniform
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, BatchNormalization
from tensorflow.keras.models import Model

import sklearn
import os
import numpy as np
import os
import shutil
import random
import matplotlib

print("\nLibraries\n-----------------------------")
print(f"Tensorflow: {tf.__version__}")
print(f"Scikit Learn: {sklearn.__version__}")
print(f"Numpy: {np.__version__}")
print(f"Matplotlib: {matplotlib.__version__}")

def count_files_in_directory(directory_path):
    total_files = 0

    for root, _, files in os.walk(directory_path):
        total_files += len(files)

    return total_files

def create_test_set(source_dir, test_dir, class_names, test_percent):
    os.makedirs(test_dir, exist_ok = True)
    
    for class_name in class_names:
        class_source_dir = os.path.join(source_dir, class_name)
        class_test_dir = os.path.join(test_dir, class_name)
        os.makedirs(class_test_dir, exist_ok = True)
        
        images = os.listdir(class_source_dir)
        num_images = len(images)
        num_test_images = int(test_percent * num_images)
        
        test_indices = random.sample(range(num_images), num_test_images)
        
        for index in test_indices:
            image_name = images[index]
            source_path = os.path.join(class_source_dir, image_name)
            target_path = os.path.join(class_test_dir, image_name)
            shutil.move(source_path, target_path)

# Define image dimensions and other parameters
img_height, img_width = 80, 400
num_classes = 3

source_data_dir = "./MDR Dataset/Train"
test_data_dir = "./MDR Dataset/Testset"
valid_data_dir = "./MDR Dataset/Validset"
class_names = os.listdir(source_data_dir)

first_execution = True

if first_execution:
    create_test_set(source_data_dir, test_data_dir, class_names, 0.3)
    create_test_set(source_data_dir, valid_data_dir, class_names, 0.2)

datagen = ImageDataGenerator(
    rotation_range = 40,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    vertical_flip = True,
    rescale = 1.0 / 255.0,
    preprocessing_function = preprocess_input
)

data_generator = datagen.flow_from_directory(
    source_data_dir,
    target_size = (img_height, img_width),
    batch_size = 32,
    class_mode = 'categorical',
    shuffle = True,
    classes = class_names
)

validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_directory(
        valid_data_dir,
        target_size = (img_height, img_width),
        batch_size = 32,
        class_mode = 'categorical',
        shuffle = True,
        classes = class_names
)

base_model = VGG16(include_top = False, input_shape = (img_height, img_width, 3))

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = Flatten()(x)
x = Dense(128, activation = 'relu', kernel_initializer = 'uniform')(x)
predictions = Dense(num_classes, activation = 'softmax')(x)

model = Model(inputs = base_model.input, outputs = predictions)

model.compile(optimizer = SGD(learning_rate = 0.001, momentum = 0.9),
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

history = model.fit(data_generator, epochs = 100,
                    validation_data = validation_generator,
                   verbose = False)

model.save_weights("VGG16_Model.h5")

base_model = ResNet50(include_top = False, input_shape = (img_height, img_width, 3))

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = Flatten()(x)
x = Dense(128, activation = 'relu', kernel_initializer = 'uniform')(x)
predictions = Dense(num_classes, activation = 'softmax')(x)

model = Model(inputs = base_model.input, outputs = predictions)

model.compile(optimizer = SGD(learning_rate = 0.001, momentum = 0.9),
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

history = model.fit(data_generator, epochs = 100,
                    validation_data = validation_generator)

model.save_weights("ResNet50_Model.h5")

base_model = InceptionV3(include_top = False, input_shape = (img_height, img_width, 3))

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = Flatten()(x)
x = Dense(128, activation = 'relu', kernel_initializer = 'uniform')(x)
predictions = Dense(num_classes, activation = 'softmax')(x)

model = Model(inputs = base_model.input, outputs = predictions)

model.compile(optimizer = SGD(learning_rate = 0.001, momentum = 0.9),
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

history = model.fit(data_generator, epochs = 100,
                    validation_data = validation_generator,
                   verbose = False)

model.save_weights("InceptionV3_Model.h5")

base_model = DenseNet201(include_top = False, input_shape = (img_height, img_width, 3))

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = Flatten()(x)
x = Dense(128, activation = 'relu', kernel_initializer = 'uniform')(x)
predictions = Dense(num_classes, activation = 'softmax')(x)

model = Model(inputs = base_model.input, outputs = predictions)

model.compile(optimizer = SGD(learning_rate = 0.001, momentum = 0.9),
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])

history = model.fit(data_generator, epochs = 100,
                    validation_data = validation_generator,
                   verbose = False)

model.save_weights("DenseNet201_Model.h5")

input_layer = Input(shape = (img_height, img_width, 3))

# first block

conv1 = Conv2D(
    filters = 32, 
    kernel_size = (3, 3), 
    strides = (2, 2), 
    padding = 'same',
    kernel_initializer = HeUniform()
)(input_layer)
conv1 = Activation('relu')(conv1)
norm1 = BatchNormalization()(conv1)

conv2 = Conv2D(
    filters = 32, 
    kernel_size = (3, 3), 
    strides = (2, 2), 
    padding = 'same',
    kernel_initializer = HeUniform()
)(norm1)
conv2 = Activation('relu')(conv2)
norm2 = BatchNormalization()(conv2)

pool1 = MaxPooling2D(
    pool_size = (2, 2), 
    strides = (1, 1), 
    padding = 'same'
)(norm2)

dropout1 = Dropout(0.2)(pool1)

# second block

conv3 = Conv2D(
    filters = 64, 
    kernel_size = (3, 3), 
    strides = (2, 2), 
    padding = 'same',
    kernel_initializer = HeUniform()
)(dropout1)
conv3 = Activation('relu')(conv3)
norm3 = BatchNormalization()(conv3)

conv4 = Conv2D(
    filters = 64, 
    kernel_size = (3, 3), 
    strides = (2, 2), 
    padding = 'same',
    kernel_initializer = HeUniform()
)(norm3)
conv4 = Activation('relu')(conv4)
norm4 = BatchNormalization()(conv4)

pool2 = MaxPooling2D(
    pool_size = (2, 2), 
    strides = (1, 1),
    padding = 'same'
)(norm4)

dropout2 = Dropout(0.3)(pool2)

# third block

conv5 = Conv2D(
    filters = 128, 
    kernel_size = (3, 3), 
    strides = (2, 2), 
    padding = 'same',
    kernel_initializer = HeUniform()
)(dropout2)
conv5 = Activation('relu')(conv5)
norm5 = BatchNormalization()(conv5)

conv6 = Conv2D(
    filters = 128, 
    kernel_size = (3, 3), 
    strides = (2, 2), 
    padding = 'same',
    kernel_initializer = HeUniform()
)(norm5)
conv6 = Activation('relu')(conv6)
norm6 = BatchNormalization()(conv6)

pool3 = MaxPooling2D(
    pool_size = (2, 2), 
    strides = (1, 1), 
    padding = 'same'
)(norm6)

dropout3 = Dropout(0.4)(pool3)

# fourth block

conv7 = Conv2D(
    filters = 512, 
    kernel_size = (3, 3), 
    strides = (2, 2),  
    padding = 'same',
    kernel_initializer = HeUniform()
)(dropout3)
conv7 = Activation('relu')(conv7)
norm7 = BatchNormalization()(conv7)

conv8 = Conv2D(
    filters = 512, 
    kernel_size = (3, 3), 
    strides = (2, 2), 
    padding = 'same',
    kernel_initializer = HeUniform()
)(norm7)
conv8 = Activation('relu')(conv8)
norm8 = BatchNormalization()(conv8)

pool4 = MaxPooling2D(
    pool_size = (2, 2), 
    strides = (1, 1), 
    padding = 'same'
)(norm8)

dropout4 = Dropout(0.4)(pool4)

flatten = Flatten()(dropout4)

dense1 = Dense(128, activation = 'relu')(flatten)
norm9 = BatchNormalization()(dense1)
dropout5 = Dropout(0.5)(norm9)
output_layer = Dense(3,
                      activation = 'softmax')(dropout5)

model = Model(inputs = input_layer, outputs = output_layer)

model.compile(optimizer = SGD(learning_rate = 0.001, momentum = 0.9),
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])

history = model.fit(data_generator, epochs = 100,
                    validation_data = validation_generator,
                   verbose = False)

model.save_weights("Jason2_Model.h5")