"""
Face Training Pipeline — Data Augmentation Engine
Generates augmented face images from enrollment frames to build
a more robust embedding database. This runs automatically when
a student completes face enrollment.
"""

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import numpy as np
except ImportError:
    np = None

import os


class FaceTrainer:
    """
    Takes raw enrollment images and generates augmented variants to
    make face recognition robust under real classroom conditions:
    - Different lighting (bright windows, dim rooms)
    - Slight head rotations (looking at notebook, board, phone)
    - Webcam quality variations (blur, noise)
    - Distance variations (front row vs back row)
    """

    # Augmentation config
    AUGMENTATIONS_PER_IMAGE = 6  # Generate 6 variants per original

    def __init__(self, models_dir=None):
        if models_dir is None:
            models_dir = os.path.join(os.getcwd(), 'models')

        det_path = os.path.join(models_dir, "face_detection_yunet_2023mar.onnx")
        if cv2 and os.path.exists(det_path):
            self.detector = cv2.FaceDetectorYN.create(det_path, "", (320, 320))
        else:
            self.detector = None

    def augment_enrollment(self, student_folder):
        """
        Read all original frames in a student folder and generate
        augmented images. Returns number of augmented images created.
        """
        if cv2 is None or np is None:
            return 0

        originals = []
        for fname in sorted(os.listdir(student_folder)):
            if fname.startswith('frame_') and fname.endswith('.jpg'):
                img_path = os.path.join(student_folder, fname)
                img = cv2.imread(img_path)
                if img is not None:
                    originals.append(img)

        if not originals:
            return 0

        aug_count = 0
        for idx, img in enumerate(originals):
            augmented = self._generate_augmentations(img)
            for aug_idx, aug_img in enumerate(augmented):
                # Verify a face is still detectable in the augmented image
                if self._has_face(aug_img):
                    out_path = os.path.join(student_folder,
                                            f'aug_{idx}_{aug_idx}.jpg')
                    cv2.imwrite(out_path, aug_img)
                    aug_count += 1

        return aug_count

    def _generate_augmentations(self, img):
        """Generate diverse augmented versions of a face image."""
        augmented = []
        h, w = img.shape[:2]

        # 1. Brightness increase (bright classroom / window light)
        bright = cv2.convertScaleAbs(img, alpha=1.3, beta=30)
        augmented.append(bright)

        # 2. Brightness decrease (dim room / shadow)
        dark = cv2.convertScaleAbs(img, alpha=0.7, beta=-20)
        augmented.append(dark)

        # 3. Slight rotation left (-8°) — looking at notebook
        M = cv2.getRotationMatrix2D((w // 2, h // 2), 8, 1.0)
        rotL = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        augmented.append(rotL)

        # 4. Slight rotation right (+8°) — looking at neighbor
        M = cv2.getRotationMatrix2D((w // 2, h // 2), -8, 1.0)
        rotR = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        augmented.append(rotR)

        # 5. Gaussian blur (slight, simulating distance / motion)
        blurred = cv2.GaussianBlur(img, (5, 5), 1.5)
        augmented.append(blurred)

        # 6. Random noise (simulating low-quality webcam)
        noise = np.random.normal(0, 12, img.shape).astype(np.int16)
        noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        augmented.append(noisy)

        return augmented

    def _has_face(self, img):
        """Quick check if YuNet can still detect a face in the augmented image."""
        if self.detector is None:
            return True  # If no detector, assume OK

        self.detector.setInputSize((img.shape[1], img.shape[0]))
        _, faces = self.detector.detect(img)
        return faces is not None and len(faces) > 0

    def retrain_student(self, student_folder):
        """
        Full pipeline: augment originals + return stats.
        Called after enrollment completes.
        """
        # Count originals
        originals = len([f for f in os.listdir(student_folder)
                         if f.startswith('frame_') and f.endswith('.jpg')])

        # Remove old augmented files (for re-enrollment)
        for fname in os.listdir(student_folder):
            if fname.startswith('aug_'):
                os.remove(os.path.join(student_folder, fname))

        # Generate new augmentations
        aug_count = self.augment_enrollment(student_folder)

        total = originals + aug_count
        return {
            'originals': originals,
            'augmented': aug_count,
            'total': total
        }
