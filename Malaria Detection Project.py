#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:





# ## Data Preprocessing

# ### Import libraries

# In[1]:



import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import random
from pathlib import Path


# ### Load the dataset and inspect the data

# In[2]:


base_dir = Path("Malaria_Detection_Dataset")
train_dir = base_dir / 'train'
test_dir = base_dir / 'test'


# In[3]:


# Inspect directories
print(f"Train categories: {list(train_dir.iterdir())}")
print(f"Test categories: {list(test_dir.iterdir())}")


# In[5]:


def show_dataset_images(data_dir, category, num_images=5):
    category_path = data_dir / category
    dataset_images = os.listdir(category_path)[:num_images]
    
    plt.figure(figsize=(10, 3))
    for i, image_name in enumerate(dataset_images):
        image_path = category_path / image_name
        image = Image.open(image_path)
        plt.subplot(1, num_images, i + 1)
        plt.imshow(image)
        plt.title(category)
    plt.show()


# In[ ]:





# ### Preprocess the images:

# Resize images to a uniform size (e.g., 128x128 or 64x64 pixels).
# 
# Normalize pixel values to a range of [0, 1]

# In[6]:


# Create a function to Load and Preprocessing images
def load_and_preprocess_images(data_dir, target_size=(64, 64)):
    images, labels = [], []
    for category in ['Parasite', 'Uninfected']:
        category_path = data_dir / category
        label = 1 if category == 'Parasite' else 0
        
        for img_name in os.listdir(category_path):
            img_path = category_path / img_name
            with Image.open(img_path) as img:
                img = img.resize(target_size)
                img_array = np.array(img) / 255.0  # Normalize to [0, 1]
                images.append(img_array)
                labels.append(label)
    
    return np.array(images), np.array(labels)


# In[6]:


# Load and preprocess train and test images
train_images, train_labels = load_and_preprocess_images(train_dir)
test_images, test_labels = load_and_preprocess_images(test_dir)


# In[7]:


plt.imshow(train_images[2])


# In[8]:


train_images[0]


# In[9]:


test_images[0]


# In[10]:


test_images.shape


# In[11]:


train_images.shape


# ### Perform data augmentation:

# In[12]:


print(f"Number of images in train data: {len(train_images)}")


# In[13]:


data_augmentation = keras.Sequential(
    [
        layers.RandomRotation(0.1),
        layers.RandomFlip('horizontal'),
        layers.RandomZoom(0.2)
    ]
)


# In[14]:


# Loop through the first 5 images
for i in range(5):
    plt.figure(figsize=(3,3))
    img = train_images[i]  # Get image
    img_array = tf.expand_dims(img, 0)
    
    # Display original image
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title("Original Image   ")
    
    # Apply data augmentation
    augmented_image = data_augmentation(img_array)  # Augment image
    plt.subplot(1, 2, 2)
    plt.imshow(augmented_image[0].numpy())  # Remove batch dimension and display
    plt.axis('off')
    plt.title("    Augmented Image")

    plt.show()


# In[15]:


augmented_train_images = []  # To store augmented images

# Apply augmentation and store augmented images
for img in train_images:  # Iterate through original training images
    # Add batch dimension (1, height, width, channels)
    
    # Apply data augmentation
    augmented_image = data_augmentation(img)  # Augment image
    
    # Append original image and augmented image
    augmented_train_images.append(augmented_image)

# Convert list to a numpy array or tensor
len(augmented_train_images)


# In[25]:


import os
import numpy as np
from PIL import Image
import tensorflow as tf

# Define the path where you want to save the images
train_images_dir = Path(r"C:\Users\akash\OneDrive\Documents\DeepLearning\Computer_Vision(CV)\FINAL_PROJECT_DataScience\Malaria_Detection_Dataset\Train")  # Change to your pat
save_directory = train_images_dir / 'augmented'
# Ensure the directory exists
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# Assuming `augmented_train_images` is a list or array of images (numpy arrays or tensors)
for i, img in enumerate(augmented_train_images):
    # Convert the image back to uint8 format if necessary (since it could be float32)
    img_uint8 = np.array(img * 255, dtype=np.uint8)  # Assuming img is normalized (0-1)
    
    # Convert the numpy array to a PIL image
    pil_img = Image.fromarray(img_uint8)
    
    # Create a PIL image from the numpy array
    # pil_img = Image.fromarray(img_uint8)
    
    # Save the image with a filename (use index or any naming convention)
    img_name = f"augmented_image_{i}.jpg"  # You can customize the file name
    img_path = os.path.join(save_directory, img_name)
    
    # Save the image
    pil_img.save(img_path)


# In[24]:


plt.imshow(augmented_train_images[5])


# ## Data Splitting

# - Split the dataset into training, validation, and test sets (e.g., 70% training, 15% validation, 15% test).

# In[64]:


# import os
# import shutil
# import random
# from pathlib import Path

# # Define paths
# base_dir = Path(r"C:\Users\akash\OneDrive\Documents\DeepLearning\Computer_Vision(CV)\FINAL_PROJECT_DataScience\Malaria_Detection_Dataset")
# train_dir = base_dir / 'train'
# test_dir = base_dir / 'test'

# # Create the validation directory
# validation_dir = base_dir / 'validation'
# validation_dir.mkdir(exist_ok=True)

# # Create 'Parasite' and 'Uninfected' subdirectories inside validation
# for category in ['Parasite', 'Uninfected']:
#     (validation_dir / category).mkdir(parents=True, exist_ok=True)

# # Helper function to move files
# def split_and_move_files(source_dir, dest_dir, split_ratio):
#     files = os.listdir(source_dir)
#     files_to_move = random.sample(files, int(len(files) * split_ratio))
    
#     for file_name in files_to_move:
#         src_path = source_dir / file_name
#         dest_path = dest_dir / file_name
#         shutil.move(src_path, dest_path)

# # Split the data: Move 15% of the total files from train and test into validation
# for category in ['Parasite', 'Uninfected']:
#     train_path = train_dir / category
#     test_path = test_dir / category
#     validation_path = validation_dir / category

#     # Move 15% from train
#     split_and_move_files(train_path, validation_path, 0.15)
#     # Move 15% from test
#     split_and_move_files(test_path, validation_path, 0.15)


# Now the Directory looks like 
# Malaria_Detection_Dataset/
#
#         train/
#
#         test/
#
#         validation/

# In[7]:


base_dir = Path("Malaria_Detection_Dataset")


# In[8]:


base_dir = Path("Malaria_Detection_Dataset")
train_dir = base_dir / 'train'
test_dir = base_dir / 'test'
val_dir = base_dir / 'validation'


# In[9]:


train_images, train_labels = load_and_preprocess_images(train_dir)
test_images, test_labels = load_and_preprocess_images(test_dir)
val_images, val_labels = load_and_preprocess_images(val_dir)


# In[10]:


from tensorflow.keras.utils import image_dataset_from_directory


# In[11]:


train_dataset = image_dataset_from_directory(base_dir / 'train', image_size=(64,64),batch_size=8)
validation_dataset = image_dataset_from_directory(base_dir / 'validation', image_size=(64,64),batch_size=8)
test_dataset = image_dataset_from_directory(base_dir / 'test', image_size=(64,64),batch_size=8)


# In[11]:


# import os

# train_dir = r"C:\Users\akash\OneDrive\Documents\DeepLearning\Computer_Vision(CV)\FINAL_PROJECT_DataScience\Malaria_Detection_Dataset\train"
# print("Contents of train directory:", os.listdir(train_dir))


# In[12]:


# import shutil
# import os

# # Define the path to the augmented folder
# augmented_dir = r"C:\Users\akash\OneDrive\Documents\DeepLearning\Computer_Vision(CV)\FINAL_PROJECT_DataScience\Malaria_Detection_Dataset\train\augmented"

# # Check if the folder exists
# if os.path.exists(augmented_dir):
#     # Remove the 'augmented' folder
#     shutil.rmtree(augmented_dir)
#     print(f"'{augmented_dir}' has been removed successfully.")
# else:
#     print(f"'{augmented_dir}' does not exist.")


# In[14]:


train_dataset = image_dataset_from_directory(base_dir / 'train', image_size=(64,64),batch_size=8)
validation_dataset = image_dataset_from_directory(base_dir / 'validation', image_size=(64,64),batch_size=8)
test_dataset = image_dataset_from_directory(base_dir / 'test', image_size=(64,64),batch_size=8)


# ## Build the CNN Model

# In[15]:


inputs = keras.Input(shape=(64,64,3),name='Malaria_Detection_model')                              # Input layer
x = layers.Conv2D(filters = 512,kernel_size=3,activation='relu', padding='same')(inputs)                 # Convolutional layers with ReLU activation
x = layers.MaxPool2D(pool_size=2)(x)                                                                    # Pooling layers (e.g., MaxPooling)
x = layers.Dropout(0.2)(x)
x = layers.Conv2D(filters = 256,kernel_size=3,activation='relu', padding='same')(x)
x = layers.MaxPool2D(pool_size=2)(x)
x = layers.Dropout(0.2)(x)                                            
x = layers.Conv2D(filters = 256,kernel_size=3,activation='relu', padding='same')(x)
x = layers.Flatten()(x)

outputs = layers.Dense(1,activation='softmax')(x)                      # Fully connected layers
model = keras.Model(inputs=inputs,outputs=outputs,name = 'Malaria_Detection_model')


# In[16]:


model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[17]:


model.summary()


# In[12]:


from tensorflow.keras import mixed_precision
# Set policy to mixed float16
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)


# In[19]:


# history = model.fit(train_images, train_labels,batch_size=32, epochs=20, validation_data=(val_images, val_labels))


# In[24]:


# import matplotlib.pyplot as plt
# accuracy = history.history["accuracy"]
# val_accuracy = history.history["val_accuracy"]
# loss = history.history["loss"]
# val_loss = history.history["val_loss"]
# epochs = range(1, len(accuracy) + 1)
# plt.plot(epochs, accuracy, "bo", label="Training accuracy")
# plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
# plt.title("Training and validation accuracy")
# plt.legend()
# plt.figure()
# plt.plot(epochs, loss, "bo", label="Training loss")
# plt.plot(epochs, val_loss, "b", label="Validation loss")
# plt.title("Training and validation loss")
# plt.legend()
# plt.show()


# #### Pre-trained model

# In[13]:


con_base = keras.applications.vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))


# In[14]:


for layer in con_base.layers:
    layer.trainable = False


# In[15]:


con_base.summary()


# In[16]:


# Create a new model that takes the extracted features
model = tf.keras.Sequential([
    layers.InputLayer(input_shape=(2, 2, 512)),  # The shape of the features from the convolutional base
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])


# In[17]:


# callbacks = [
# keras.callbacks.ModelCheckpoint(
# filepath="maleria_detection_model.tf",
# save_best_only=True,
# monitor="val_loss")
# ]


# In[18]:


model.compile(loss="binary_crossentropy",optimizer="rmsprop",metrics=["accuracy"])


# In[19]:


# Function to extract features from a dataset
def get_features(dataset):
    features = con_base.predict(dataset)  # Get features from the convolutional base
    return features
train_features = get_features(train_images)
val_features = get_features(val_images)


# In[20]:


# history = model.fit(train_features, train_labels, epochs=20, validation_data=(val_features, val_labels),callbacks=callbacks)


# In[40]:


# accuracy = history.history["accuracy"]
# val_accuracy = history.history["val_accuracy"]
# loss = history.history["loss"]
# val_loss = history.history["val_loss"]
# epochs = range(1, len(accuracy) + 1)
# plt.plot(epochs, accuracy, "bo", label="Training accuracy")
# plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
# plt.title("Training and validation accuracy")
# plt.legend()
# plt.figure()
# plt.plot(epochs, loss, "bo", label="Training loss")
# plt.plot(epochs, val_loss, "b", label="Validation loss")
# plt.title("Training and validation loss")
# plt.legend()
# plt.show()


# In[41]:


# callbacks = [
# keras.callbacks.ModelCheckpoint(
# filepath="maleria_detection_model_2.h5",
# save_best_only=True,
# monitor="val_loss")
# ]


# In[42]:


# history = model.fit(train_features, train_labels, epochs=12, validation_data=(val_features, val_labels),callbacks=callbacks)


# In[44]:


# history = model.fit(train_features, train_labels, epochs=20, validation_data=(val_features, val_labels),callbacks=callbacks)


# In[45]:


# accuracy = history.history["accuracy"]
# val_accuracy = history.history["val_accuracy"]
# loss = history.history["loss"]
# val_loss = history.history["val_loss"]
# epochs = range(1, len(accuracy) + 1)
# plt.plot(epochs, accuracy, "bo", label="Training accuracy")
# plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
# plt.title("Training and validation accuracy")
# plt.legend()
# plt.figure()
# plt.plot(epochs, loss, "bo", label="Training loss")
# plt.plot(epochs, val_loss, "b", label="Validation loss")
# plt.title("Training and validation loss")
# plt.legend()
# plt.show()


# ## Evaluate the Model

# In[21]:


test_model = keras.models.load_model('maleria_detection_model.tf')


# In[34]:


# Assuming 'con_base' is your convolutional base (e.g., VGG16)
def extract_features_from_images(images):
    # Preprocess the images and extract features using the convolutional base
    processed_images = keras.applications.vgg16.preprocess_input(images)
    features = con_base.predict(processed_images)
    return features

test_features = extract_features_from_images(test_images)  # Extract features
print(test_features.shape)  # Should print a shape like (99, 5, 5, 512) for VGG16


# In[35]:


# Flatten the features to a 2D array: (num_samples, feature_size)
test_features_flattened = test_features.reshape(test_features.shape[0], -1)
print(test_features_flattened.shape)  # Now it should be (99, 2048) or similar, depending on the model


# In[36]:


test_model = keras.models.load_model('maleria_detection_model.tf')


# In[38]:


# Ensure you pass the test features (without flattening) and the test labels
test_loss, test_Acc = test_model.evaluate(test_features, test_labels)

print(f"Test Loss: {test_loss:.3f}")
print(f"Test Accuracy: {test_Acc:.3f}")


# In[ ]:


test_model = keras.models.load_model('maleria_detection_model_2.h5')


# In[47]:


test_features = extract_features_from_images(test_images)  # Extract features
print(test_features.shape)  # Should print a shape like (99, 5, 5, 512) for VGG16
test_features_flattened = test_features.reshape(test_features.shape[0], -1)
print(test_features_flattened.shape)


# In[48]:


# Ensure you pass the test features (without flattening) and the test labels
test_loss, test_Acc = test_model.evaluate(test_features, test_labels)

print(f"Test Loss: {test_loss:.3f}")
print(f"Test Accuracy: {test_Acc:.3f}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




