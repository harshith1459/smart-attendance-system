try:
    import cv2
except ImportError:
    cv2 = None
import os

class FaceDetector:
    def __init__(self):
        # Load the cascade
        if cv2:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        else:
            self.face_cascade = None

    def detect_faces(self, img):
        if not cv2: return [], None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        return faces, gray

    def extract_face(self, img):
        faces, gray = self.detect_faces(img)
        if len(faces) == 0:
            return None, None
        
        (x, y, w, h) = faces[0]
        face_roi = gray[y:y+h, x:x+w]
        return face_roi, (x, y, w, h)
