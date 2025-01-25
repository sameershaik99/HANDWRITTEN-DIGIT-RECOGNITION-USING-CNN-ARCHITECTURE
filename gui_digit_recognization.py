import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# Load the pre-trained model
model = load_model('handwritten_digit_recognition_model.h5')

# Image dimensions (matching the model's input)
img_width, img_height = 64, 64

# Function to load an image, preprocess, and make a prediction
def load_and_predict_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = load_img(file_path, target_size=(img_width, img_height), color_mode="grayscale")
        img_array = img_to_array(img) / 255.0  # Rescale image as the model was trained on rescaled images
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make a prediction
        prediction = model.predict(img_array)
        predicted_digit = np.argmax(prediction)

        # Display the result
        result_text.set(f"Predicted Digit: {predicted_digit}")

        # Display the selected image in the GUI
        display_img = ImageTk.PhotoImage(img.resize((150, 150)))
        image_label.config(image=display_img)
        image_label.image = display_img

# Initialize the GUI
root = tk.Tk()
root.title("Handwritten Digit Recognition")

# Set up GUI layout
result_text = tk.StringVar()
tk.Label(root, text="Handwritten Digit Recognition", font=("Helvetica", 16)).pack(pady=10)
tk.Button(root, text="Select Image", command=load_and_predict_image).pack(pady=10)
tk.Label(root, textvariable=result_text, font=("Helvetica", 14)).pack(pady=10)

# Image display area
image_label = tk.Label(root)
image_label.pack()

# Start the GUI event loop
root.geometry("300x400")
root.mainloop()
