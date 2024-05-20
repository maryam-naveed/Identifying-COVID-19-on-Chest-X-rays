# configuring the path of Kaggle.json file
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json 
!kaggle datasets download -d alifrahman/covid19-chest-xray-image-dataset
!ls

# extracting the compressed dataset
from zipfile import ZipFile
dataset = '/content/covid19-chest-xray-image-dataset.zip'
with ZipFile(dataset, 'r') as zip:
  zip.extractall()
  print('The dataset is extracted')

import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,BatchNormalization,Dropout

# generators
train_ds = keras.utils.image_dataset_from_directory(
    directory = '/content/dataset',
    labels='inferred',
    label_mode = 'int',
    batch_size=32,
    image_size=(256,256)
)

validation_ds = keras.utils.image_dataset_from_directory(
    directory = '/content/dataset',
    labels='inferred',
    label_mode = 'int',
    batch_size=32,
    image_size=(256,256)
)

import os
file_names = os.listdir('/content/dataset/covid')
print(len(file_names))
file_names = os.listdir('/content/dataset/normal')
print(file_names)

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Set the path to the train and val folders
train_folder = '/content/dataset'
val_folder = '/content/dataset'

# Define image parameters
image_size = (224, 224)
input_shape = (224, 224, 3)
batch_size = 16

# Create data generators for train and val sets
train_datagen = ImageDataGenerator(rescale=1.0/255)
val_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    train_folder,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    val_folder,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# Create the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=10, validation_data=val_generator)

# Evaluate the model on the validation set
loss, accuracy = model.evaluate(val_generator)
print(f'Validation Loss: {loss:.4f}')
print(f'Validation Accuracy: {accuracy:.4f}')

import os
import random
import shutil
from tensorflow import keras

# Define the directory of your dataset
dataset_dir = '/content/dataset'

# Define the directory for the training dataset
train_dir = '/content/train'

# Define the directory for the validation dataset
validation_dir = '/content/validation'

# Create the training and validation directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(os.path.join(train_dir, 'covid'), exist_ok=True)
os.makedirs(os.path.join(train_dir, 'normal'), exist_ok=True)
os.makedirs(validation_dir, exist_ok=True)
os.makedirs(os.path.join(validation_dir, 'covid'), exist_ok=True)
os.makedirs(os.path.join(validation_dir, 'normal'), exist_ok=True)

# Define the subdirectory for COVID images
covid_dir = os.path.join(dataset_dir, 'covid')
normal_dir = os.path.join(dataset_dir, 'normal')

# Get the list of COVID image filenames
covid_filenames = [filename for filename in os.listdir(covid_dir) if filename.endswith('.jpeg')]

# Get the list of normal image filenames
normal_filenames = [filename for filename in os.listdir(normal_dir) if filename.endswith('.jpeg')]

# Set the random seed for reproducibility
random.seed(42)

# Shuffle the COVID filenames
random.shuffle(covid_filenames)

# Shuffle the normal filenames
random.shuffle(normal_filenames)

# Define the number of COVID images for training and validation
num_train_covid = 44
num_val_covid = len(covid_filenames) - num_train_covid

# Define the number of normal images for training and validation
num_train_normal = 16
num_val_normal = len(normal_filenames) - num_train_normal

# Move COVID images to the training directory
for filename in covid_filenames[:num_train_covid]:
    src = os.path.join(covid_dir, filename)
    dst = os.path.join(train_dir, 'covid', filename)
    shutil.copyfile(src, dst)

# Move COVID images to the validation directory
for filename in covid_filenames[num_train_covid:]:
    src = os.path.join(covid_dir, filename)
    dst = os.path.join(validation_dir, 'covid', filename)
    shutil.copyfile(src, dst)

# Move normal images to the training directory
for filename in normal_filenames[:num_train_normal]:
    src = os.path.join(normal_dir, filename)
    dst = os.path.join(train_dir, 'normal', filename)
    shutil.copyfile(src, dst)

# Move normal images to the validation directory
for filename in normal_filenames[num_train_normal:]:
    src = os.path.join(normal_dir, filename)
    dst = os.path.join(validation_dir, 'normal', filename)
    shutil.copyfile(src, dst)

# Create the training dataset using the training directory
train_ds = keras.utils.image_dataset_from_directory(
    directory=train_dir,
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(256, 256)
)

# Create the validation dataset using the validation directory
validation_ds = keras.utils.image_dataset_from_directory(
    directory=validation_dir,
    labels='inferred',
    label_mode='int',
    batch_size=32,
    image_size=(256, 256)
)

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.metrics import classification_report, accuracy_score, precision_score, f1_score, recall_score

# Set the path to the train and val folders
train_folder = '/content/train'
val_folder = '/content/validation'

# Define image parameters
image_size = (224, 224)
input_shape = (224, 224, 3)
batch_size = 16

# Create data generators for train and val sets
train_datagen = ImageDataGenerator(rescale=1.0/255)
val_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    train_folder,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    val_folder,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# Create the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=10, validation_data=val_generator)

# Generate predictions for the validation set
y_pred = model.predict(val_generator)
y_pred = np.round(y_pred).flatten()  # Convert probabilities to binary predictions

# Get the true labels for the validation set
y_true = val_generator.labels

# Calculate the accuracy
accuracy = accuracy_score(y_true, y_pred)

# Calculate the precision
precision = precision_score(y_true, y_pred)

# Calculate the recall
recall = recall_score(y_true, y_pred)

# Calculate the F1 score
f1 = f1_score(y_true, y_pred)

# Print the results
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1 Score:', f1)
