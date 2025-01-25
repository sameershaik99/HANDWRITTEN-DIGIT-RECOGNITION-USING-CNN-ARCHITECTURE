import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Paths to our dataset folders
train_data_dir = r'D:\IAI PROJECT\HANDWRITTEN_DIGIT_RECOGNITION\DIGIT_DATASET\TRAINING'  # training path
test_data_dir = r'D:\IAI PROJECT\HANDWRITTEN_DIGIT_RECOGNITION\DIGIT_DATASET\TEST'       # test path

# Image dimensions (64x64 for handwritten digit dataset)
img_width, img_height = 64, 64
input_shape = (img_width, img_height, 1)

# Parameters
batch_size = 32
num_classes = 10  # recognizing digits from 0 to 9
epochs = 30   # Increased epochs for more training

# ImageDataGenerator for training and test sets
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    brightness_range=(0.8, 1.2),
    zoom_range=0.2,
    width_shift_range=0.2,
    shear_range=0.2
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Training and testing data generators
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    color_mode="grayscale",  # For grayscale images
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode='categorical'
)

#CNN model with padding to prevent dimension reduction issues
model = Sequential()
model.add(Conv2D(8, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(num_classes, activation='softmax'))

# learning rate
model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              metrics=['accuracy'])

# Train the model
hist = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=test_generator.samples // batch_size
)

# Evaluate the model
score = model.evaluate(test_generator, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Save the model locally
model.save('handwritten_digit_recognition_model.h5')
print("Model saved to disk")
