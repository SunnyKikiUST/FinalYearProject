import io
from typing import List
import numpy as np
import cv2
from PIL import Image
from openai import AzureOpenAI
from gtts import gTTS
import os
import io
import playsound
import time
import sys
import torch
from typing import List
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN
from emotiefflib.facial_analysis import EmotiEffLibRecognizer, get_model_list
from multiprocessing import shared_memory
import time
import io
from mutagen.mp3 import MP3
from socket import *
import struct
from model_common_utils import listen_for_exit_signal, get_setting_openAI_key

HOST = '127.0.0.1'
PORT = 65452
API_KEY = get_setting_openAI_key()

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

    Example:
        faces = recognize_faces('image.jpg', 'cuda')
        # faces contains the cropped face images detected in 'image.jpg'.
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


# Function to get a motivational quote based on emotion
def get_motivational_quote(emotion):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": """You are a agent from a fitness game called Fitness Fugitive, where the mechanics are similar to Temple Run. 
                During gameplay, the camera captures the player's emotions and body motions. Assume the camera can always detect emotions accurately using an algorithm to classify them as Angry, Disgusted, Fear, Happy, Sad, Surprise, or Neutral. 
                When certain emotions are detected (e.g., Happy, Sad, Tired, Angry), generate encouraging and motivational quotes or dictums that align with the player's emotional state and the fitness theme of the game. 
                The quotes or dictums should be positive, inspiring, and relevant to the gameplay. 
                For example: If the player is angry, the quote could be: 'Channel that fire into your run! You're unstoppable!' If the player is tired, the quote could be: 'Every step counts! You're stronger than you think!'."""
            },
            {
                "role": "user",
                "content": f"Generate a quote for the emotion without quotation mark: {emotion}."
            }
        ]
    )
    return response.choices[0].message.content

def check_api_validity(client):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": "test"
            }
        ]
    )
    return response.choices[0].message.content


import tempfile
# Function to speak the quote
def speak_quote(quote, client_socket):
    filename = "quote.mp3"

    tts = gTTS(text=quote, lang='en')
    tts.save(filename)

    audio = MP3(filename)
    # Inform unity side that the newest quote is ready
    inform_message = struct.pack("i", 1)
    # try:
    #     client_socket.sendall(inform_message) 
    # except Exception as e:
    #     print(f"Error: {e}")

    print(f"audio length: {audio.info.length}")
    print("sleeping")
    time.sleep(audio.info.length) # sleep for this duration in seconds
    print("finish sleeping")

###############
    os.remove(filename)

    return time.time()

def main():
    # Initialize variables
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = get_model_list()[0]  # Load the first model from the available list
    fer = EmotiEffLibRecognizer(engine="onnx", model_name=model_name, device=device)  # Initialize recognizer
    frame_interval = 30  # Process every 30th frame
    frame_count = 0  # Counter to track frames
    
    # Emotion tracking variables
    emotion_window_size = 10  # 10-second window
    emotion_history = []  # List to track emotions in the window
    last_quote_time = time.time()  # Track the last time a quote was generated

    # Shared memory parameters
    high_res_shm_name = "fitness_fat_camera_high_res"
    # high_res = (960, 720)  # Default high resolution - might be (960, 720) if fallback is used
    # high_res_size = high_res[0] * high_res[1] * 3  # RGB format

    actual_webcam_resolution_name = "actual_resolution"
    channels = 3

    # Wait for shared memory to be available
    while True:
        try:
            actual_webcam_resolution_shm = shared_memory.SharedMemory(name=actual_webcam_resolution_name, create=False)
            resolution_array = np.ndarray((2,), dtype=np.float32, buffer=actual_webcam_resolution_shm.buf)
            desired_width = int(resolution_array[0])  # Store width
            desired_height = int(resolution_array[1]) # Store height
            break
        except FileNotFoundError:
            print("Shared memory for resolution detail not found. Waiting...")
            time.sleep(1)
        except Exception as e:
            print(f"Error: {e}")

    print("Waiting for shared memory to be available...")
    while True:
        try:
            high_res_shm = shared_memory.SharedMemory(name=high_res_shm_name, create=False)
            frame_array = np.ndarray((desired_height, desired_width, channels), dtype=np.uint8, buffer=high_res_shm.buf)
            break
        except FileNotFoundError:
            print("Shared memory not found. Waiting...")
            time.sleep(1)
        except Exception as e:
            print(f"Error: {e}")
    
    print(f"Connected to shared memory: {high_res_shm_name}")

    print("Seeing if openai api key in setting is empty...")
    if not API_KEY or API_KEY == "":
        print("API key not found. Exiting script.")
        return

    try:
        # Initialize OpenAI Azure client "84bf82be4c204e7182860feeee5c3c26"
        client = AzureOpenAI(
            api_key=API_KEY,  # Get from https://hkust.developer.azure-api.net/  
            api_version="2024-10-21",
            azure_endpoint="https://hkust.azure-api.net"
        )
        result = check_api_validity(client)
        print(f"Simple testing of api key. The successful response result: {result}")
    except Exception:
        print("Error while constructing AzureOpenAI")
        return 
    
    print("Openai api key in setting is not empty.")

    #Socket communcation setup for communicating with C# game section
    server_socket = socket(AF_INET, SOCK_STREAM) # Sunny
    server_socket.bind((HOST, PORT))
    server_socket.listen(1)
    print("Waiting for Unity to connect emotion recognition model...")

    client_socket, addr = server_socket.accept() # Accept connect when cap is opened
    client_socket.settimeout(5)
    print(f"Emotion recognition model: Connected to Unity at {addr}") # Sunny

    threading.Thread(target=listen_for_exit_signal, args=(client_socket,), daemon = True)
    
    try:
        while True:

            # Resize the frame to the desired resolution (in case the camera doesn't support it)
            #frame_bgr = cv2.resize(frame_array, (desired_width, desired_height))
            frame_bgr = frame_array

            # Increment frame count
            frame_count += 1
            
            # Current time for time-based operations
            current_time = time.time()

            # Process every nth frame
            if frame_count % frame_interval == 0:
                # Convert frame to RGB for processing
                
                frame_count = 0

                frame_rgb = cv2.cvtColor(frame_array, cv2.COLOR_BGR2RGB)

                # Detect faces in the frame
                facial_images = recognize_faces(frame_rgb, device)

                valid_faces = [face for face in facial_images if all(dim > 0 for dim in face.shape)]

                if len(valid_faces) > 0:
                    # We'll focus on the first detected face for simplicity
                    face_img = facial_images[0]
                    # Predict emotions 
                    print("test 1")
                    emotion, _ = fer.predict_emotions(face_img, logits=True)
                    detected_emotion = emotion[0]
                    print("test 2")
                    print(detected_emotion)

                    # Add timestamp and emotion to history
                    emotion_history.append((current_time, detected_emotion))
                    
                    # Show current emotion on the frame
                    cv2.putText(frame_bgr, f"Current: {detected_emotion}", (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    # Clean up old emotion records (older than window size)
                    while emotion_history and (current_time - emotion_history[0][0] > emotion_window_size):
                        emotion_history.pop(0)
                    print("test 3")
                    emotions_in_window = [e[1] for e in emotion_history]
                    total_samples = len(emotions_in_window)
                    
                    if total_samples > 0:
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
                        if threshold_emotions and (current_time - last_quote_time >= 1000000):
                            # Get the emotion with highest portion above threshold
                            dominant_emotion = max(threshold_emotions.items(), key=lambda x: x[1])[0]
                            if dominant_emotion == "Neutral":
                                print("skip neutraul")
                                continue

                            # Generate and speak a quote for the dominant emotion
                            quote = get_motivational_quote(dominant_emotion)
                            print(f"Dominant Emotion: {dominant_emotion} ({int(threshold_emotions[dominant_emotion]*100)}%), Quote: {quote}")
                            last_quote_time = speak_quote(quote, client_socket) # Update the last quote time and generate mp3
                    else:
                        print("No samples")     

            #cv2.imshow("Emotion Recognition", frame_bgr)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting...")
                break
    except Exception as e:
        print(f"Error: {e}")

    finally:
        cv2.destroyAllWindows()
        print("Emotion recognition stopped.")

if __name__ == "__main__":
    main()