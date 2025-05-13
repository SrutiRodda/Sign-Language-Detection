import tkinter as tk
from tkinter import Label
import cv2
from PIL import Image, ImageTk
import threading
import time
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Global variable to store the latest detection result
current_sign = "None"

# Callback function to process detection results
def result_callback(result, output_image, timestamp_ms):
    global current_sign
    if result and result.gestures and len(result.gestures) > 0:
        # Get the top gesture
        current_sign = result.gestures[0][0].category_name

# Load model
MODEL_PATH = "sign_language_recognizer_25-04-2023.task"
BaseOptions = python.BaseOptions
GestureRecognizer = vision.GestureRecognizer
GestureRecognizerOptions = vision.GestureRecognizerOptions
VisionRunningMode = vision.RunningMode

# Create recognizer with proper callback
options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=result_callback,
)
recognizer = GestureRecognizer.create_from_options(options)

# Create GUI window
window = tk.Tk()
window.title("Sign Language Detector")
window.geometry("800x600")

video_label = Label(window)
video_label.pack(pady=10)

text_label = Label(window, text="Sign: None", font=("Arial", 24))
text_label.pack(pady=10)

# Video capture
cap = cv2.VideoCapture(0)

def update_frame():
    global current_sign
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        # Flip and convert for display
        display_frame = cv2.flip(frame, 1)
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        try:
            # Process the frame with timestamp
            timestamp_ms = int(time.time() * 1000)
            recognizer.recognize_async(mp_image, timestamp_ms)
            
            # Update the label with the current sign
            text_label.config(text=f"Sign: {current_sign}")
            
        except Exception as e:
            print(f"Error in recognition: {e}")

        # Display video
        img = Image.fromarray(rgb_frame)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)
        
        # Update the window
        window.update_idletasks()
        window.update()
        
        # Small delay to control frame rate
        time.sleep(0.03)

# Start the application
print("Starting Sign Language Detector...")
update_thread = threading.Thread(target=update_frame)
update_thread.daemon = True
update_thread.start()

window.mainloop()

# Clean up
cap.release()
