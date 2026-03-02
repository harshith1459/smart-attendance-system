try:
    import cv2
except ImportError:
    cv2 = None
import os
try:
    import numpy as np
except ImportError:
    np = None

class FaceRecognizer:
    def __init__(self, db_path):
        self.db_path = db_path
        self.models_dir = os.path.join(os.getcwd(), 'models')
        
        # Load models
        detector_path = os.path.join(self.models_dir, "face_detection_yunet_2023mar.onnx")
        recognizer_path = os.path.join(self.models_dir, "face_recognition_sface_2021dec.onnx")
        
        if not os.path.exists(detector_path) or not os.path.exists(recognizer_path):
            print(f"ERROR: SFace models not found in {self.models_dir}")
            self.detector = None
            self.recognizer = None
        else:
            # Initialize YuNet detector
            # We'll set input size in detect method
            self.detector = cv2.FaceDetectorYN.create(detector_path, "", (320, 320))
            # Initialize SFace recognizer
            self.recognizer = cv2.FaceRecognizerSF.create(recognizer_path, "")
            
        self.student_features = {} # {student_id: [embedding1, ...]}
        self.pkl_path = os.path.join(self.db_path, "representations_sface.pkl")
        self._load_features()

    def _load_features(self):
        """Build feature database from dataset/student_{id}/ folders"""
        if self.recognizer is None:
            return

        if not os.path.exists(self.db_path):
            os.makedirs(self.db_path, exist_ok=True)
            return

        # Try to load from cache
        import pickle
        if os.path.exists(self.pkl_path):
            try:
                with open(self.pkl_path, 'rb') as f:
                    self.student_features = pickle.load(f)
                print(f"Loaded {len(self.student_features)} students from cache.")
                return
            except Exception as e:
                print(f"Cache load failed: {e}")

        print("Building face feature database from scratch...")
        for folder in os.listdir(self.db_path):
            if folder.startswith('student_'):
                student_id = folder.split('_')[1]
                folder_path = os.path.join(self.db_path, folder)
                
                embeddings = []
                for img_name in os.listdir(folder_path):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(folder_path, img_name)
                        img = cv2.imread(img_path)
                        if img is not None:
                            # Detect and extract feature
                            self.detector.setInputSize((img.shape[1], img.shape[0]))
                            _, faces = self.detector.detect(img)
                            if faces is not None:
                                # Use first face
                                aligned_face = self.recognizer.alignCrop(img, faces[0])
                                feature = self.recognizer.feature(aligned_face)
                                embeddings.append(feature)
                
                if embeddings:
                    self.student_features[student_id] = embeddings
        
        # Save to cache
        try:
            with open(self.pkl_path, 'wb') as f:
                pickle.dump(self.student_features, f)
            print("Feature database saved to cache.")
        except Exception as e:
            print(f"Failed to save cache: {e}")

        print(f"Database built. Loaded features for {len(self.student_features)} students.")

    def recognize_face(self, img):
        if cv2 is None or np is None or self.detector is None or self.recognizer is None:
            return []
            
        # 1. Detect faces
        self.detector.setInputSize((img.shape[1], img.shape[0]))
        _, faces = self.detector.detect(img)
        
        results = []
        if faces is not None:
            for face in faces:
                # 2. Extract feature
                aligned_face = self.recognizer.alignCrop(img, face)
                feature = self.recognizer.feature(aligned_face)
                
                best_match = "unknown"
                best_score = 0.0 # Cosine similarity
                
                # 3. Match against database
                for student_id, embeddings in self.student_features.items():
                    for db_feature in embeddings:
                        # SFace uses Cosine Similarity or Norm (default 0 for cosine)
                        score = self.recognizer.match(feature, db_feature, 0)
                        if score > best_score:
                            best_score = score
                            best_match = student_id
                
                # Threshold for SFace Cosine is usually around 0.363 
                # but we'll be more strict if needed. Let's use 0.3
                if best_score < 0.3:
                    best_match = "unknown"
                
                box = face[0:4].astype(int)
                results.append((best_match, best_score, (box[0], box[1], box[2], box[3])))
                
        return results

    def validate_face(self, img):
        """Validates if a single face is visible and detectable for enrollment"""
        if self.detector is None:
            return False, "Engine initialization failed"
        
        self.detector.setInputSize((img.shape[1], img.shape[0]))
        _, faces = self.detector.detect(img)
        
        if faces is None:
            return False, "No face detected. Please ensure you are in a well-lit area."
        
        if len(faces) > 1:
            return False, "Multiple faces detected. Please ensure only you are in the frame."
            
        # Optional: Check face size/confidence
        conf = faces[0][-1]
        if conf < 0.8:
            return False, "Face detection confidence low. Please look directly at the camera."
            
        return True, "Valid face"

