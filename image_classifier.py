import cv2
import numpy as np
from PIL import Image

from ultralytics import YOLO

from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions
)

yolo_model = YOLO("yolov8n.pt")

mobilenet_model = MobileNetV2(weights="imagenet")

HUMAN_CLASSES = [
    "person"
]

ANIMAL_CLASSES = [

    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe"

]

VEHICLE_CLASSES = [

    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "train",
    "truck",
    "boat",
    "airplane"

]

def preprocess_image(image):

    image = image.convert("RGB")

    img = np.array(image)

    return img

def detect_with_yolo(img):

    """
    Returns:
    Human
    Animal
    Vehicle
    None
    """

    results = yolo_model.predict(
        img,
        verbose=False
    )

    if len(results) == 0:
        return None

    boxes = results[0].boxes

    if boxes is None:
        return None

    for box in boxes:

        class_id = int(box.cls[0])

        class_name = yolo_model.names[class_id]

        print("YOLO Detected :", class_name)

        if class_name in HUMAN_CLASSES:
            return "Human"

        if class_name in ANIMAL_CLASSES:
            return "Animal"

        if class_name in VEHICLE_CLASSES:
            return "Vehicle"

    return None

def mobilenet_prediction(img):

    resized = cv2.resize(img, (224, 224))

    x = np.expand_dims(resized, axis=0)

    x = preprocess_input(x)

    preds = mobilenet_model.predict(
        x,
        verbose=0
    )

    predictions = decode_predictions(
        preds,
        top=5
    )[0]

    print("\nTop 5 MobileNet Predictions\n")

    for item in predictions:
        print(item)

    return predictions

def classify_image(image):

    img = preprocess_image(image)

    yolo_result = detect_with_yolo(img)

    if yolo_result is not None:
        return yolo_result

    predictions = mobilenet_prediction(img)

    labels = []

    for item in predictions:
        labels.append(item[1].lower())

    print("\nLabels :", labels)

    nature_keywords = [

        "mountain",
        "volcano",
        "valley",
        "lakeside",
        "seashore",
        "cliff",
        "alp",
        "forest"

    ]

    for label in labels:
        for keyword in nature_keywords:
            if keyword in label:
                return "Nature"

    logo_keywords = [

        "comic_book",
        "book_jacket",
        "web_site",
        "menu",
        "packet"

    ]

    for label in labels:
        for keyword in logo_keywords:
            if keyword in label:
                return "Logo"

    cartoon_keywords = [

        "comic_book",
        "mask",
        "toyshop",
        "puppet"

    ]

    for label in labels:
        for keyword in cartoon_keywords:
            if keyword in label:
                return "Cartoon"

    human_keywords = [

        "suit",
        "wig",
        "maillot",
        "jersey",
        "academic_gown",
        "bow_tie",
        "neck_brace",
        "sunglasses"

    ]

    for label in labels:
        for keyword in human_keywords:
            if keyword in label:
                return "Human"


    return "Unknown"