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

# Initialize OpenAI Azure client
client = AzureOpenAI(
    api_key="84bf82be4c204e7182860feeee5c3c26",  # Get from https://hkust.developer.azure-api.net/  
    api_version="2024-10-21",
    azure_endpoint="https://hkust.azure-api.net"
)

# Function to get a motivational quote based on emotion
def get_motivational_quote(emotion):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": f"You are a agent from a fitness game called Fitness Fugitive, where the mechanics are similar to Temple Run. 
                During gameplay, the camera captures the player's emotions and body motions. Assume the camera can always detect emotions accurately using an algorithm to classify them as Angry, Disgusted, Fear, Happy, Sad, Surprise, or Neutral. 
                When certain emotions are detected (e.g., Happy, Sad, Tired, Angry), generate encouraging and motivational quotes or dictums that align with the player's emotional state and the fitness theme of the game. 
                The quotes or dictums should be positive, inspiring, and relevant to the gameplay. 
                For example: If the player is angry, the quote could be: 'Channel that fire into your run! You're unstoppable!' If the player is tired, the quote could be: 'Every step counts! You're stronger than you think!'."
            },
            {
                "role": "user",
                "content": f"Generate a quote for the emotion without quotation mark: {emotion}."
            }
        ]
    )
    return response.choices[0].message.content


# Function to speak the quote
def speak_quote(quote):
    tts = gTTS(text=quote, lang='en')
    filename = "quote.mp3"
    tts.save(filename)
    playsound.playsound(filename)
    os.remove(filename) 

def main():
    # Initialize variables
    device = 'cuda' if torch.cuda.is_available() else 'cpu'   
    model_name = get_model_list()[0]  # Load the first model from the available list
    fer = EmotiEffLibRecognizer(engine="onnx", model_name=model_name, device=device)  # Initialize recognizer
    frame_interval = 30  # Process every 30th frame
    frame_count = 0  # Counter to track frames

    # Set desired resolution
    desired_width = 960
    desired_height = 720

    # Initialize video capture
    cap = cv2.VideoCapture(0)  # Use the default webcam

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    # Create a real-time display window
    cv2.namedWindow("Emotion Recognition", cv2.WINDOW_NORMAL)

    try:
        while True:
            success, frame_bgr = cap.read()
            if not success:
                print("Error: Could not read frame from webcam.")
                break

            # Resize the frame to the desired resolution
            frame_bgr = cv2.resize(frame_bgr, (desired_width, desired_height))

            # Increment frame count
            frame_count += 1

            # Process every 30th frame
            if frame_count % frame_interval == 0:
                # Convert frame to RGB for processing
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                # Detect faces in the frame
                facial_images = recognize_faces(frame_rgb, device)

                if len(facial_images) > 0:
                    emotions = []
                    for face_img in facial_images:
                        # Predict emotions for each detected face
                        emotion, _ = fer.predict_emotions(face_img, logits=True)
                        emotions.append(emotion[0])

                        # Display the detected face and emotion in Matplotlib (optional)
                        plt.figure(figsize=(3, 3))
                        plt.axis('off')
                        plt.imshow(face_img)
                        plt.title(emotion[0])
                        plt.show()

                        # Generate and speak a quote every 30 frames
                        quote = get_motivational_quote(emotion)
                        print(f"Emotion: {emotion}, Quote: {quote}")
                        speak_quote(quote)  # Speak the generated quote

                    # Annotate the original frame with emotions
                    for idx, face_img in enumerate(facial_images):
                        cv2.putText(frame_bgr, f"{emotions[idx]}", (50, 50 + idx * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # D'isplay the video stream with annotations
            cv2.imshow("Emotion Recognition", frame_bgr)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting...")
                break

    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()