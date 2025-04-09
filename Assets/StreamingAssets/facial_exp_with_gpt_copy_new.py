from typing import List
import numpy as np
import cv2
from PIL import Image
from openai import AzureOpenAI
from gtts import gTTS
import os
import playsound
import time
import torch
from typing import List
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN
from emotiefflib.facial_analysis import EmotiEffLibRecognizer, get_model_list
from pygame import mixer
import io
from mutagen.mp3 import MP3

def recognize_faces(frame: np.ndarray, device: str) -> List[np.array]:
    """
    Detects faces in the given image and returns the facial images cropped from the original.

    This function reads an image from the specified path, detects faces using the MTCNN
    face detection model, and returns a list of cropped face images.

    Args:
        frame (numpy.ndarray): The image frame in which faces need to be detected.
        device (str): The device to run the MTCNN face detection model on, e.g., 'cpu' or 'cuda'.

    Returns:
        list: A list of numpy arrays, representing a cropped face image from the original image.
    """

    def detect_face(frame: np.ndarray):
        mtcnn = MTCNN(keep_all=False, post_process=False, min_face_size=40, device=device)
        bounding_boxes, probs = mtcnn.detect(frame, landmarks=False)
        if probs[0] is None:
            return []
        bounding_boxes = bounding_boxes[probs > 0.9]
        return bounding_boxes

    bounding_boxes = detect_face(frame)
    facial_images = []
    for bbox in bounding_boxes:
        box = bbox.astype(int)
        x1, y1, x2, y2 = box[0:4]
        facial_images.append(frame[y1:y2, x1:x2, :])
    return facial_images

##############Combine emotion reconition with gpt and voice####################

# Initialize OpenAI Azure client
client = AzureOpenAI(
    api_key="84bf82be4c204e7182860feeee5c3c26",  # Get from https://hkust.developer.azure-api.net/  
    api_version="2024-10-21",
    azure_endpoint="https://hkust.azure-api.net"
)

# Function to get a motivational quote based on emotion
def get_motivational_quote(emotion):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": """You are a agent from a fitness game called Fitness Fugitive, where the mechanics are similar to Temple Run. 
                During gameplay, the camera captures the player's emotions and body motions. Assume the camera can always detect emotions accurately using an algorithm to classify them as Angry, Contempt, Disgusted, Fear, Happy, Sad, Surprise, or Neutral. 
                When certain emotions are detected (e.g., Happy, Sad, Tired, Angry), generate encouraging and motivational quotes or dictums that align with the player's emotional state and the fitness theme of the game. 
                The quotes or dictums should be positive, inspiring, and relevant to the gameplay. 
                For example: If the player is angry, the quote could be: 'Channel that fire into your run! You're unstoppable!' If the player is tired, the quote could be: 'Every step counts! You're stronger than you think!'.
                Note that sadness oftenly is similar to tiredness, the quote of dictum should related to sadness also should consider that the it involves tiredness"""
            },
            {
                "role": "user",
                "content": f"Generate a quote for the emotion without quotation mark: {emotion}."
            }
        ]
    )
    print(response.choices[0].message.content)
    return response.choices[0].message.content

import tempfile
# Function to speak the quote
def speak_quote(quote):
    filename = "quote.mp3"

    tts = gTTS(text=quote, lang='en')
    tts.save(filename)

    audio = MP3(filename)
###############
    # Load the audio data from quote.mp3
    mixer.init()
    mixer.music.load(filename)
    mixer.music.play()

    print(f"audio length: {audio.info.length}")
    print("sleeping")
    time.sleep(audio.info.length) # sleep for this duration in seconds
    print("finish sleeping")

    mixer.music.stop()
    mixer.quit()
###############
    os.remove(filename)

    return time.time()
    
def main():
    # Initialize variables
    device = 'cuda' if torch.cuda.is_available() else 'cpu'   
    model_name = get_model_list()[0]  # Load the first model from the available list
    fer = EmotiEffLibRecognizer(engine="onnx", model_name=model_name, device=device)  # Initialize recognizer
    frame_interval = 3  # Process every 3rd frame for more frequent emotion sampling
    frame_count = 0  # Counter to track frames

    # Emotion tracking variables
    emotion_window_size = 10  # 10-second window
    emotion_history = []  # List to track emotions in the window
    last_quote_time = time.time()  # Track the last time a quote was generated

    # Set desired resolution
    desired_width = 960
    desired_height = 720

    # Initialize video capture
    cap = cv2.VideoCapture(0)  # Use the default webcam

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

    # Create a real-time display window
    cv2.namedWindow("Emotion Recognition", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Emotion Recognition", desired_width, desired_height)

    try:
        while True:
            # Read frame from webcam
            success, frame_bgr = cap.read()
            if not success:
                print("Error: Could not read frame from webcam.")
                break

            # Resize the frame to the desired resolution (in case the camera doesn't support it)
            frame_bgr = cv2.resize(frame_bgr, (desired_width, desired_height))

            # Increment frame count
            frame_count += 1
            
            # Current time for time-based operations
            current_time = time.time()

            # Process every nth frame
            if frame_count % frame_interval == 0:
                # Convert frame to RGB for processing
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                # Detect faces in the frame
                facial_images = recognize_faces(frame_rgb, device)

                if len(facial_images) > 0:
                    # We'll focus on the first detected face for simplicity
                    face_img = facial_images[0]
                    
                    # Predict emotions 
                    emotion, _ = fer.predict_emotions(face_img, logits=True)
                    detected_emotion = emotion[0]

                    print(detected_emotion)
                    
                    # Add timestamp and emotion to history
                    emotion_history.append((current_time, detected_emotion))
                    
                    # Show current emotion on the frame
                    cv2.putText(frame_bgr, f"Current: {detected_emotion}", (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    # Clean up old emotion records (older than window size)
                    while emotion_history and (current_time - emotion_history[0][0] > emotion_window_size):
                        emotion_history.pop(0)
                    
                    emotions_in_window = [e[1] for e in emotion_history]
                    total_samples = len(emotions_in_window)
                    
                    if total_samples > 0:
                        print("Have samples")
                        emotion_counts = {}
                        for e in emotions_in_window: # Count occurrences of each emotion
                            emotion_counts[e] = emotion_counts.get(e, 0) + 1
                        
                        # Calculate portions (frequencies)
                        emotion_portions = {e: count/total_samples for e, count in emotion_counts.items()}
                        
                        # Find emotions that meet the 2/10 threshold
                        threshold_emotions = {e: portion for e, portion in emotion_portions.items() 
                                            if portion >= 0.2}  # 2/10 = 0.2
                        
                        for i, (e, p) in enumerate(sorted(emotion_portions.items(), 
                                                        key=lambda x: x[1], reverse=True)):
                            percent = int(p * 100)
                            cv2.putText(frame_bgr, f"{e}: {percent}%", (50, 100 + i * 30), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        
                        # If any emotions meet the threshold and it's been at least 10 seconds since last quote
                        if threshold_emotions and (current_time - last_quote_time >= 10):
                            # Get the emotion with highest portion above threshold
                            dominant_emotion = max(threshold_emotions.items(), key=lambda x: x[1])[0]
                            if dominant_emotion == "Neutral":
                                print("skip neutraul")
                                continue

                            # Generate and speak a quote for the dominant emotion
                            quote = get_motivational_quote(dominant_emotion)
                            print(f"Dominant Emotion: {dominant_emotion} ({int(threshold_emotions[dominant_emotion]*100)}%), Quote: {quote}")
                            last_quote_time = speak_quote(quote) # Update the last quote time and generate mp3
                    else:
                        print("No samples")     

            cv2.imshow("Emotion Recognition", frame_bgr)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting...")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Emotion recognition stopped.")

if __name__ == "__main__":
    main()