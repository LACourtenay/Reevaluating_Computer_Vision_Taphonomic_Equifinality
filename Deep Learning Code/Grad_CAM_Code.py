import argparse

import tensorflow as tf
from tensorflow.keras.applications import DenseNet201, VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
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
from matplotlib import pyplot as plt
import matplotlib
import cv2

print("\nLibraries\n-----------------------------")
print(f"Tensorflow: {tf.__version__}")
print(f"Scikit Learn: {sklearn.__version__}")
print(f"Numpy: {np.__version__}")
print(f"Matplotlib: {matplotlib.__version__}")
print(f"OpenCV: {cv2.__version__}")

def count_files_in_directory(directory_path):
    total_files = 0

    for root, _, files in os.walk(directory_path):
        total_files += len(files)

    return total_files

class GradCAM:

    def __init__(self, model, classIdx, dataset):
        self.model = model
        self.classIdx = classIdx

        if dataset == 1:
            self.layer_name = "conv2d_3"
        elif dataset == 2:
            self.layer_name = "block3_conv1"
        else:
            self.layer_name = "conv5_block32_1_conv"

    def compute_heatmap(self, image, eps=1e-8):

        gradModel = Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layer_name).output, self.model.output])

        with tf.GradientTape() as tape:
            
            inputs = tf.cast(image, tf.float32)
            (convOutputs, predictions) = gradModel(inputs)
            
            loss = predictions[:, tf.argmax(predictions[0])]

        grads = tape.gradient(loss, convOutputs)

        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads

        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]

        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

        (w, h) = (image.shape[2], image.shape[1])
        heatmap = cv2.resize(cam.numpy(), (w, h))

        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")

        return heatmap
    
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

parser = argparse.ArgumentParser(description = "Process different datasets, DS1 (1), DS2 (2) or DS3 (3)")
parser.add_argument("--dataset", type = int, choices = [1, 2, 3], help = "Choose dataset (1, 2, or 3)")
args = parser.parse_args()

target_dataset = f"DS{args.dataset}"

if target_dataset == "DS1":
    
    img_height, img_width = 80, 400

    data_dir = "./DS1/Train"

    train_data_dir = "./DS1/Train"
    test_data_dir = "./DS1/Test"
    valid_data_dir = "./DS1/Validset"

    class_names = os.listdir(train_data_dir)

    first_execution = True

    if first_execution:
        create_test_set(train_data_dir, test_data_dir, class_names, 0.3)
        create_test_set(train_data_dir, valid_data_dir, class_names, 0.2)
    
        datagen = ImageDataGenerator(
        rotation_range = 40,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True,
        vertical_flip = True,
        rescale = 1.0 / 255.0
    )

    data_generator = datagen.flow_from_directory(
        train_data_dir,
        target_size = (img_height, img_width),
        batch_size = 32,
        class_mode = 'binary',
        shuffle = True
    )

    validation_datagen = ImageDataGenerator(
        rescale=1./255
    )

    validation_generator = validation_datagen.flow_from_directory(
            valid_data_dir,
            target_size = (img_height, img_width),
            batch_size = 14,
            class_mode = 'binary',
            shuffle = True
    )

    test_datagen = ImageDataGenerator(
        rescale=1./255
    )
    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size = (img_height, img_width),
        batch_size = count_files_in_directory(test_data_dir),
        class_mode = 'binary',
        shuffle = False
    )

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
    output_layer = Dense(1,
                        activation = 'sigmoid')(dropout5)

    model = Model(inputs = input_layer, outputs = output_layer)

    model.compile(optimizer = SGD(learning_rate = 0.001, momentum = 0.9),
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])

    history = model.fit(data_generator, epochs = 100,
                        validation_data = validation_generator)

elif target_dataset == "DS2":
    
    img_height, img_width = 80, 400
    num_classes = 3

    data_dir = "./DS2/Train"
    class_names = os.listdir(data_dir)

    # test set
    source_data_dir = "./DS2/Train"
    test_data_dir = "./DS2/Testset"
    valid_data_dir = "./DS2/Validset"
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
        data_dir,
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

    test_datagen = ImageDataGenerator(
        rescale=1./255
    )
    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size = (img_height, img_width),
        class_mode = 'categorical',
        batch_size = count_files_in_directory(test_data_dir),
        shuffle = False,
        classes = class_names
    )

    base_model = VGG16(include_top = False, input_shape = (img_height, img_width, 3))

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)
    x = Dense(128, activation = 'relu', kernel_initializer = 'uniform')(x)
    #x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation = 'softmax')(x)

    model = Model(inputs = base_model.input, outputs = predictions)

    model.compile(optimizer = SGD(learning_rate = 0.001, momentum = 0.9),
                loss = 'categorical_crossentropy',
                metrics = ['accuracy'])

    history = model.fit(data_generator, epochs = 100,
                        validation_data = validation_generator)

else:

    img_height, img_width = 250, 200
    num_classes = 3

    data_dir = "./DS3/Train"
    class_names = os.listdir(data_dir)

    # test set
    source_data_dir = "./DS3/Train"
    test_data_dir = "./DS3/Test"
    valid_data_dir = "./DS3/Validset"
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
        rescale = 1.0 / 255.0
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
            batch_size = 14,
            class_mode = 'categorical',
            shuffle = True,
            classes = class_names
    )

    base_model = DenseNet201(include_top = False, input_shape = (img_height, img_width, 3))

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation = 'relu', kernel_initializer = 'uniform')(x)
    predictions = Dense(num_classes, activation = 'softmax')(x)

    model = Model(inputs = base_model.input, outputs = predictions)

    model.compile(optimizer = SGD(learning_rate = 0.001, momentum = 0.9),
                loss = 'categorical_crossentropy',
                metrics = ['accuracy'])

    history = model.fit(data_generator, epochs = 100,
                        validation_data = validation_generator)

return_files = True

if target_dataset == "DS1":
    
    if return_files:

        source_folders = ["./DS1/Test", "./DS1/Validset"]
        destination_folder = "./DS1/Train"

        for source_folder in source_folders:
            croc_source_path = os.path.join(source_folder, "Croc")
            cut_mark_source_path = os.path.join(source_folder, "Cut Mark")

            croc_destination_path = os.path.join(destination_folder, "Croc")
            cut_mark_destination_path = os.path.join(destination_folder, "Cut Mark")

            for filename in os.listdir(croc_source_path):
                source_file = os.path.join(croc_source_path, filename)
                destination_file = os.path.join(croc_destination_path, filename)
                shutil.move(source_file, destination_file)

            for filename in os.listdir(cut_mark_source_path):
                source_file = os.path.join(cut_mark_source_path, filename)
                destination_file = os.path.join(cut_mark_destination_path, filename)
                shutil.move(source_file, destination_file)

        print("Files moved successfully.")

        classification_type = "binary"

elif target_dataset == "DS2":

    if return_files:

        source_folders = ["./DS2/Testset", "./DS2/Validset"]
        destination_folder = "./DS2/Train"

        for source_folder in source_folders:
            source_path = os.path.join(source_folder, "Cut-marks")
            destination_path = os.path.join(destination_folder, "Cut-marks")

            for filename in os.listdir(source_path):
                source_file = os.path.join(source_path, filename)
                destination_file = os.path.join(destination_path, filename)
                shutil.move(source_file, destination_file)

            source_path = os.path.join(source_folder, "Scores")
            destination_path = os.path.join(destination_folder, "Scores")

            for filename in os.listdir(source_path):
                source_file = os.path.join(source_path, filename)
                destination_file = os.path.join(destination_path, filename)
                shutil.move(source_file, destination_file)

            source_path = os.path.join(source_folder, "Tramplings")
            destination_path = os.path.join(destination_folder, "Tramplings")

            for filename in os.listdir(source_path):
                source_file = os.path.join(source_path, filename)
                destination_file = os.path.join(destination_path, filename)
                shutil.move(source_file, destination_file)

        print("Files moved successfully.")

        classification_type = "categorical"
else:

    if return_files:

        source_folders = ["./DS3/Test", "./DS3/Validset"]
        destination_folder = "./DS3/Train"

        for source_folder in source_folders:
            source_path = os.path.join(source_folder, "Cut-marks")
            destination_path = os.path.join(destination_folder, "Cut-marks")

            for filename in os.listdir(source_path):
                source_file = os.path.join(source_path, filename)
                destination_file = os.path.join(destination_path, filename)
                shutil.move(source_file, destination_file)

            source_path = os.path.join(source_folder, "Scores")
            destination_path = os.path.join(destination_folder, "Scores")

            for filename in os.listdir(source_path):
                source_file = os.path.join(source_path, filename)
                destination_file = os.path.join(destination_path, filename)
                shutil.move(source_file, destination_file)

            source_path = os.path.join(source_folder, "Tramplings")
            destination_path = os.path.join(destination_folder, "Tramplings")

            for filename in os.listdir(source_path):
                source_file = os.path.join(source_path, filename)
                destination_file = os.path.join(destination_path, filename)
                shutil.move(source_file, destination_file)

        print("Files moved successfully.")

        classification_type = "categorical"

gradcam_datagen = ImageDataGenerator(rescale=1./255)

gradcam_generator = gradcam_datagen.flow_from_directory(
        data_dir,
        target_size = (img_height, img_width),
        batch_size = count_files_in_directory(data_dir),
        class_mode = classification_type,
        shuffle = True,
        classes = class_names
)

os.makedirs("./Grad_CAM_Results", exist_ok = True)

if target_dataset == "DS1":

    os.makedirs("./Grad_CAM_Results/DS1", exist_ok = True)

    target_result_path = "./Grad_CAM_Results/DS1"

elif target_dataset == "DS2":

    os.makedirs("./Grad_CAM_Results/DS2", exist_ok = True)

    target_result_path = "./Grad_CAM_Results/DS2"

else:

    os.makedirs("./Grad_CAM_Results/DS3", exist_ok = True)

    target_result_path = "./Grad_CAM_Results/DS3"

batch_images, batch_labels = next(gradcam_generator)

for iteration in range(batch_images.shape[0]):

    single_image_array = batch_images[iteration]
    single_image_label = batch_labels[iteration]

    target_image = tf.convert_to_tensor(np.expand_dims(single_image_array, axis=0))

    plt.figure(figsize = (15, 5))

    plt.subplot(1, 3, 1)
    original_pil_image = image.array_to_img(single_image_array)
    plt.imshow(original_pil_image)
    plt.axis("off")

    target_image = tf.convert_to_tensor(np.expand_dims(single_image_array, axis=0))
    with tf.GradientTape() as tape:
        tape.watch(target_image)
        prediction = model(target_image)
        loss = tf.keras.losses.binary_crossentropy(single_image_label, prediction[0])
    gradient = tape.gradient(loss, target_image)
    perturbed_image = target_image + 0.007 * tf.sign(gradient)

    icam = GradCAM(model, 1, args.dataset)
    heatmap = icam.compute_heatmap(target_image)
    heatmap = cv2.resize(heatmap, (img_height, img_width))
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap)
    plt.axis("off")
    plt.title("Grad-CAM")

    heatmap2 = icam.compute_heatmap(perturbed_image)
    heatmap2 = cv2.resize(heatmap2, (img_height, img_width))
    plt.subplot(1, 3, 3)
    plt.imshow(heatmap2)
    plt.axis("off")
    plt.title("Grad-CAM (Adversarial)")

    grad_cam_path = os.path.join(target_result_path, f"grad_cam_{iteration}.png")

    plt.savefig(grad_cam_path, bbox_inches = "tight")
    plt.close()