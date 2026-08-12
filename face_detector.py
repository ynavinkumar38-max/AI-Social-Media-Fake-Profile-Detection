import cv2

cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

def detect_face(image_path):
    image = cv2.imread(image_path)

    if image is None:
        return False, 0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )

    return len(faces) > 0, len(faces)