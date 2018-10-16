from statistics import mode

import cv2, time
from keras.models import load_model
import numpy as np
from naoqi import ALProxy
import vision_definitions
from PIL import Image
import numpy as np
import sys, os

from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input

# parameters for loading data and images
detection_model_path = '../trained_models/detection_models/haarcascade_frontalface_default.xml'
emotion_model_path = '../trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
emotion_labels = get_labels('fer2013')

# hyper-parameters for bounding boxes shape
frame_window = 10
emotion_offsets = (20, 40)

# loading models
face_detection = load_detection_model(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# starting lists for calculating modes
emotion_window = []

# starting video streaming
camProxy = ALProxy("ALVideoDevice", "127.0.0.1", 51421)
cameraIndex = 0
resolution = vision_definitions.kVGA
colorSpace = vision_definitions.kRGBColorSpace
resolution = 2
colorSpace =3
cv2.namedWindow('window_frame')
video_capture = cv2.VideoCapture(0)
if video_capture.isOpened():
 frame = video_capture.read()
else:
 rval = False
while True:
    bgr_image = video_capture.read()[1]
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    faces = detect_faces(face_detection, gray_image)
    rval, frame = video_capture.read()
    frame=cv2.resize(frame, (640, 480))
    key = cv2.waitKey(1)
    time.sleep(0.15)
    b,g,r = cv2.split(frame) # get b,g,r
    rgb_img = cv2.merge([r,g,b]) # switch it to rgb
    set=camProxy.putImage(0,640,480,rgb_img.tobytes())

    for face_coordinates in faces:

        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue

        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_prediction = emotion_classifier.predict(gray_face)
        emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_labels[emotion_label_arg]
        emotion_window.append(emotion_text)

        if len(emotion_window) > frame_window:
            emotion_window.pop(0)
        try:
            emotion_mode = mode(emotion_window)
        except:
            continue

        if emotion_text == 'angry':
            color = emotion_probability * np.asarray((255, 0, 0))
            tts = ALProxy("ALTextToSpeech", "127.0.0.1", 51421)
            tts.say("Why u so angry?")
        elif emotion_text == 'sad':
            color = emotion_probability * np.asarray((0, 0, 255))
            tts = ALProxy("ALTextToSpeech", "127.0.0.1", 51421)
            tts.say("Sad Slut")
        elif emotion_text == 'fear':
            color = emotion_probability * np.asarray((0, 0, 255))
            tts = ALProxy("ALTextToSpeech", "127.0.0.1", 51421)
            tts.say("Get me out of here!!")
        elif emotion_text == 'happy':
            color = emotion_probability * np.asarray((255, 255, 0))
            tts = ALProxy("ALTextToSpeech", "127.0.0.1", 51421)
            tts.say("Good to see you happy!")
            print("Happy")
        elif emotion_text == 'surprise':
            color = emotion_probability * np.asarray((0, 255, 255))
            tts = ALProxy("ALTextToSpeech", "127.0.0.1", 51421)
            tts.say("Surprised ?")
            print("Surprised")
        else:
            color = emotion_probability * np.asarray((0, 255, 0))

        color = color.astype(int)
        color = color.tolist()

        draw_bounding_box(face_coordinates, rgb_image, color)
        draw_text(face_coordinates, rgb_image, emotion_mode,
                  color, 0, -45, 1, 1)

    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('window_frame', bgr_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
