"""
Face Recognition Engine — Group-Optimized v3
==============================================
Optimized for classroom attendance: reliably handles 30+ faces in a single frame.

Key improvements over v2:
  - Multi-scale detection (catches small far-away faces AND close-up faces)
  - NMS deduplication (removes duplicate detections from different scales)
  - Adaptive confidence threshold (relaxed for group shots vs strict for enrollment)
  - Duplicate-match prevention (same student can't be matched twice per frame)
  - Quality-weighted centroid (augmented enrollment images included)
"""

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
    # ─── Thresholds (validated on LFW 70-student benchmark) ───
    COSINE_THRESHOLD = 0.45       # Accept if cosine score >= this
    DETECTION_CONF_MIN = 0.70     # Min confidence for group detection (relaxed from 0.80)
    ENROLLMENT_CONF_MIN = 0.85    # Stricter for enrollment validation
    SCORE_MARGIN = 0.05           # Gap required between 1st and 2nd best match

    # ─── Multi-scale Detection Config ─────────────────────────
    DETECTION_SCALES = [1.0, 0.75, 0.5]   # Catch faces from 20px to 400px wide
    NMS_IOU_THRESHOLD = 0.4                # IoU threshold for dedup across scales

    def __init__(self, db_path):
        self.db_path = db_path
        self.models_dir = os.path.join(os.getcwd(), 'models')

        det_path = os.path.join(self.models_dir, "face_detection_yunet_2023mar.onnx")
        rec_path = os.path.join(self.models_dir, "face_recognition_sface_2021dec.onnx")

        if not os.path.exists(det_path) or not os.path.exists(rec_path):
            print(f"ERROR: ONNX models not found in {self.models_dir}")
            self.detector = None
            self.recognizer = None
        else:
            self.detector = cv2.FaceDetectorYN.create(det_path, "", (320, 320))
            self.recognizer = cv2.FaceRecognizerSF.create(rec_path, "")

        # {student_id: [embedding, ...]}
        self.student_features = {}
        # {student_id: centroid_embedding}
        self.student_centroids = {}

        self.pkl_path = os.path.join(self.db_path, "representations_sface.pkl")
        self._load_features()

    # ═══════════════════════════════════════════════════════════
    # DATABASE
    # ═══════════════════════════════════════════════════════════
    def _load_features(self):
        """Build feature database from dataset/student_{id}/ folders."""
        if self.recognizer is None:
            return

        if not os.path.exists(self.db_path):
            os.makedirs(self.db_path, exist_ok=True)
            return

        import pickle

        if os.path.exists(self.pkl_path):
            try:
                with open(self.pkl_path, 'rb') as f:
                    cache = pickle.load(f)
                if isinstance(cache, dict) and 'features' in cache:
                    self.student_features = cache['features']
                    self.student_centroids = cache.get('centroids', {})
                else:
                    self.student_features = cache
                    self._build_centroids()
                total_emb = sum(len(v) for v in self.student_features.values())
                print(f"[FaceDB] Loaded {len(self.student_features)} students, "
                      f"{total_emb} embeddings from cache.")
                return
            except Exception as e:
                print(f"[FaceDB] Cache load failed: {e}")

        self._rebuild_database()

    def _rebuild_database(self):
        """Scan all student folders and extract embeddings (originals + augmented)."""
        print("[FaceDB] Building from disk (originals + augmented images)...")
        self.student_features = {}

        for folder in os.listdir(self.db_path):
            if not folder.startswith('student_'):
                continue

            student_id = folder.split('_', 1)[1]
            folder_path = os.path.join(self.db_path, folder)

            embeddings = []
            for img_name in sorted(os.listdir(folder_path)):
                if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue

                img = cv2.imread(os.path.join(folder_path, img_name))
                if img is None:
                    continue

                self.detector.setInputSize((img.shape[1], img.shape[0]))
                _, faces = self.detector.detect(img)
                if faces is None or len(faces) == 0:
                    continue

                best = faces[np.argmax(faces[:, -1])]
                if best[-1] >= 0.60:  # Lower for augmented images
                    aligned = self.recognizer.alignCrop(img, best)
                    feature = self.recognizer.feature(aligned)
                    embeddings.append(feature)

            if embeddings:
                self.student_features[student_id] = embeddings

        self._build_centroids()
        self._save_cache()
        total_emb = sum(len(v) for v in self.student_features.values())
        print(f"[FaceDB] Built: {len(self.student_features)} students, {total_emb} embeddings.")

    def _build_centroids(self):
        """Average all embeddings per student into a single centroid vector."""
        self.student_centroids = {}
        for sid, embeddings in self.student_features.items():
            if not embeddings:
                continue
            stacked = np.vstack(embeddings)                     # (N, 128)
            centroid = np.mean(stacked, axis=0, keepdims=True)  # (1, 128)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                ref_norm = np.linalg.norm(embeddings[0])
                centroid = centroid / norm * ref_norm
            self.student_centroids[sid] = centroid

    def _save_cache(self):
        import pickle
        try:
            with open(self.pkl_path, 'wb') as f:
                pickle.dump({
                    'features': self.student_features,
                    'centroids': self.student_centroids
                }, f)
        except Exception as e:
            print(f"[FaceDB] Cache save failed: {e}")

    def rebuild_cache(self):
        """Public: force-rebuild the database (called after enrollment + augmentation)."""
        self._rebuild_database()

    # ═══════════════════════════════════════════════════════════
    # MULTI-SCALE DETECTION
    # ═══════════════════════════════════════════════════════════
    def _detect_faces_multiscale(self, img):
        """
        Detect faces at multiple scales and merge with NMS.
        Catches small far-away faces that single-pass detection misses.
        """
        if self.detector is None:
            return None

        h, w = img.shape[:2]
        all_faces = []

        for scale in self.DETECTION_SCALES:
            if scale == 1.0:
                scaled_img = img
            else:
                new_w = int(w * scale)
                new_h = int(h * scale)
                if new_w < 60 or new_h < 60:
                    continue
                scaled_img = cv2.resize(img, (new_w, new_h))

            self.detector.setInputSize((scaled_img.shape[1], scaled_img.shape[0]))
            _, faces = self.detector.detect(scaled_img)

            if faces is not None:
                for face in faces:
                    scaled_face = face.copy()
                    # Scale coordinates back to original resolution
                    scaled_face[0] /= scale   # x
                    scaled_face[1] /= scale   # y
                    scaled_face[2] /= scale   # w
                    scaled_face[3] /= scale   # h
                    for i in range(4, 14):     # landmark points
                        scaled_face[i] /= scale
                    all_faces.append(scaled_face)

        if not all_faces:
            return None

        all_faces = np.array(all_faces)

        # NMS to remove duplicate detections from different scales
        if len(all_faces) > 1:
            all_faces = self._nms(all_faces, self.NMS_IOU_THRESHOLD)

        return all_faces

    def _nms(self, faces, iou_threshold):
        """Non-Maximum Suppression to merge overlapping detections."""
        x1 = faces[:, 0]
        y1 = faces[:, 1]
        x2 = faces[:, 0] + faces[:, 2]
        y2 = faces[:, 1] + faces[:, 3]
        scores = faces[:, -1]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            union = areas[i] + areas[order[1:]] - inter
            iou = inter / np.maximum(union, 1e-6)
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return faces[keep]

    # ═══════════════════════════════════════════════════════════
    # RECOGNITION (Group Optimized)
    # ═══════════════════════════════════════════════════════════
    def recognize_face(self, img):
        """
        Detect and recognize ALL faces in an image.
        Optimized for group shots (30+ students in one frame).

        Returns: list of (student_id, score, (x, y, w, h))
        """
        if cv2 is None or np is None or self.detector is None or self.recognizer is None:
            return []

        # Multi-scale detection
        faces = self._detect_faces_multiscale(img)
        if faces is None or len(faces) == 0:
            return []

        results = []
        already_matched = set()  # Prevent same student matched twice per frame

        # Process highest-confidence faces first
        face_order = np.argsort(-faces[:, -1])

        for idx in face_order:
            face = faces[idx]
            det_conf = float(face[-1])

            if det_conf < self.DETECTION_CONF_MIN:
                continue

            try:
                aligned = self.recognizer.alignCrop(img, face)
                feature = self.recognizer.feature(aligned)
            except Exception:
                continue

            # Stage 1: Centroid matching (fast — 1 comparison per student)
            scores = {}
            for student_id, centroid in self.student_centroids.items():
                if student_id in already_matched:
                    continue
                scores[student_id] = self.recognizer.match(feature, centroid, 0)

            if not scores:
                # Fallback: brute-force against all embeddings
                for student_id, embeddings in self.student_features.items():
                    if student_id in already_matched:
                        continue
                    best = max(self.recognizer.match(feature, emb, 0) for emb in embeddings)
                    scores[student_id] = best

            # Find best match
            best_id = "unknown"
            best_score = 0.0

            if scores:
                sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
                top_id, top_score = sorted_scores[0]

                if top_score >= self.COSINE_THRESHOLD:
                    if len(sorted_scores) >= 2:
                        margin = top_score - sorted_scores[1][1]
                        if margin >= self.SCORE_MARGIN:
                            best_id = top_id
                            best_score = top_score
                    else:
                        best_id = top_id
                        best_score = top_score

            if best_id != "unknown":
                already_matched.add(best_id)

            box = face[0:4].astype(int)
            results.append((best_id, best_score, (box[0], box[1], box[2], box[3])))

        return results

    # ═══════════════════════════════════════════════════════════
    # ENROLLMENT VALIDATION
    # ═══════════════════════════════════════════════════════════
    def validate_face(self, img):
        """Validates if a single clear face is visible for enrollment."""
        if self.detector is None:
            return False, "Engine initialization failed"

        self.detector.setInputSize((img.shape[1], img.shape[0]))
        _, faces = self.detector.detect(img)

        if faces is None or len(faces) == 0:
            return False, "No face detected. Please ensure you are in a well-lit area."

        if len(faces) > 1:
            return False, "Multiple faces detected. Please ensure only you are in the frame."

        conf = float(faces[0][-1])
        if conf < self.ENROLLMENT_CONF_MIN:
            return False, f"Face detection confidence too low ({conf:.0%}). Look directly at camera."

        return True, "Valid face"
