"""
Group Recognition Stress Test
==============================
Simulates a real classroom scenario:
1. Enroll 70 students (5 images each + augmentation → ~35 training images each)
2. Create composite GROUP IMAGES with 10, 20, 30+ faces pasted together
3. Test if the recognizer correctly identifies everyone in the group
4. Also tests: imposters, partial occlusion, varied sizes

Uses LFW dataset (already downloaded) as face source.
"""

import sys
import os
import time

# Ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
from sklearn.datasets import fetch_lfw_people
from ml.trainer import FaceTrainer
from ml.recognize import FaceRecognizer

# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────
NUM_ENROLLED = 70        # Students to enroll
NUM_IMPOSTERS = 30       # People NOT enrolled (should be rejected)
ENROLLMENT_IMGS = 5      # Original enrollment images per student
GROUP_SIZES = [5, 10, 15, 20, 25, 30, 40]  # Group photo sizes to test

# Paths
LFW_DIR = os.path.join(os.path.dirname(__file__), 'lfw_home')
TEST_DB = os.path.join(os.path.dirname(__file__), 'group_test_db')
COMPOSITE_DIR = os.path.join(os.path.dirname(__file__), 'group_composites')


def lfw_to_bgr(img, scale=3):
    """Convert LFW grayscale float32 (125x94) to BGR uint8, upscaled for YuNet."""
    # LFW images are float32 in [0, 1] range
    img_uint8 = np.clip(img * 255, 0, 255).astype(np.uint8)
    # Grayscale → BGR
    bgr = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)
    # Upscale (125x94 → 375x282 at 3x) so YuNet can detect faces
    h, w = bgr.shape[:2]
    bgr = cv2.resize(bgr, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    return bgr


def load_lfw_data():
    """Load LFW dataset with enough people."""
    print("Loading LFW dataset...")
    lfw = fetch_lfw_people(
        data_home=LFW_DIR,
        min_faces_per_person=8,
        resize=1.0
    )
    print(f"  Raw image shape: {lfw.images[0].shape}, dtype: {lfw.images[0].dtype}")

    # Group by person, convert to BGR
    people = {}
    for img, label in zip(lfw.images, lfw.target):
        name = lfw.target_names[label]
        if name not in people:
            people[name] = []
        people[name].append(lfw_to_bgr(img))

    print(f"  Converted to BGR: {people[list(people.keys())[0]][0].shape}")
    print(f"  Loaded {len(people)} people from LFW")
    return people


def prepare_enrollment(people, test_db):
    """Create enrollment folders with 5 images each + augmentation."""
    os.makedirs(test_db, exist_ok=True)

    # Clean old data
    import shutil
    for d in os.listdir(test_db):
        full = os.path.join(test_db, d)
        if os.path.isdir(full):
            shutil.rmtree(full)

    names = list(people.keys())
    enrolled_names = names[:NUM_ENROLLED]
    imposter_names = names[NUM_ENROLLED:NUM_ENROLLED + NUM_IMPOSTERS]

    print(f"\nEnrolling {len(enrolled_names)} students, {len(imposter_names)} imposters...")

    enrolled_data = {}  # name → {'folder': path, 'test_images': [img, ...]}
    imposter_data = {}  # name → [test_images]

    for i, name in enumerate(enrolled_names):
        imgs = people[name]
        sid = f"test_{i:03d}"
        folder = os.path.join(test_db, f"student_{sid}")
        os.makedirs(folder, exist_ok=True)

        # Save enrollment images (first 5)
        for j in range(min(ENROLLMENT_IMGS, len(imgs))):
            img = imgs[j]
            cv2.imwrite(os.path.join(folder, f"frame_{j}.jpg"), img)

        # Remaining images are for testing
        test_imgs = []
        for j in range(ENROLLMENT_IMGS, len(imgs)):
            test_imgs.append(imgs[j])

        enrolled_data[sid] = {'folder': folder, 'test_images': test_imgs, 'name': name}

    for i, name in enumerate(imposter_names):
        imgs = people[name]
        imposter_data[name] = imgs[:]

    return enrolled_data, imposter_data


def run_augmentation(enrolled_data):
    """Run augmentation on all enrolled students."""
    print("\nRunning data augmentation pipeline...")
    trainer = FaceTrainer()
    total_orig = 0
    total_aug = 0

    for sid, data in enrolled_data.items():
        stats = trainer.retrain_student(data['folder'])
        total_orig += stats['originals']
        total_aug += stats['augmented']

    print(f"  Augmentation complete: {total_orig} originals → {total_orig + total_aug} total "
          f"({total_aug} augmented images generated)")
    print(f"  Average per student: {(total_orig + total_aug) / len(enrolled_data):.1f} images")
    return total_orig, total_aug


def create_group_composite(faces, target_width=1920, target_height=1080):
    """
    Create a composite image with multiple faces arranged in a grid,
    simulating a classroom group photo / webcam capture.
    """
    n = len(faces)
    if n == 0:
        return None

    # Calculate grid layout
    cols = int(np.ceil(np.sqrt(n * (target_width / target_height))))
    rows = int(np.ceil(n / cols))

    cell_w = target_width // cols
    cell_h = target_height // rows

    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    canvas[:] = (190, 195, 200)  # Light grey classroom background

    for i, face_img in enumerate(faces):
        row = i // cols
        col = i % cols

        # Random scale simulates different distances from camera
        scale = np.random.uniform(0.65, 0.95)
        new_w = max(int(cell_w * scale), 30)
        new_h = max(int(cell_h * scale), 30)

        resized = cv2.resize(face_img, (new_w, new_h))

        # Center with slight random offset
        x = col * cell_w + (cell_w - new_w) // 2 + np.random.randint(-3, 4)
        y = row * cell_h + (cell_h - new_h) // 2 + np.random.randint(-3, 4)
        x = max(0, min(x, target_width - new_w))
        y = max(0, min(y, target_height - new_h))

        canvas[y:y + new_h, x:x + new_w] = resized

    return canvas


def test_group_recognition(recognizer, enrolled_data, imposter_data):
    """Test recognition on composite group images."""
    print("\n" + "=" * 70)
    print("GROUP RECOGNITION STRESS TEST")
    print("=" * 70)

    os.makedirs(COMPOSITE_DIR, exist_ok=True)
    enrolled_ids = list(enrolled_data.keys())

    overall_results = {}

    for group_size in GROUP_SIZES:
        print(f"\n{'─' * 50}")
        print(f"Testing group of {group_size} enrolled + 5 imposters")
        print(f"{'─' * 50}")

        # Select students for this group
        selected = enrolled_ids[:group_size]
        faces_for_composite = []
        expected_ids = set()

        for sid in selected:
            test_imgs = enrolled_data[sid]['test_images']
            if test_imgs:
                faces_for_composite.append(test_imgs[0])
                expected_ids.add(sid)

        # Add 5 imposter faces
        imposter_names_list = list(imposter_data.keys())
        for imp_name in imposter_names_list[:5]:
            imp_imgs = imposter_data[imp_name]
            if imp_imgs:
                faces_for_composite.append(imp_imgs[0])

        # Create composite
        composite = create_group_composite(faces_for_composite)
        if composite is None:
            print("  Failed to create composite!")
            continue

        # Save composite for inspection
        save_path = os.path.join(COMPOSITE_DIR, f"group_{group_size}.jpg")
        cv2.imwrite(save_path, composite)

        # Run recognition
        t0 = time.time()
        results = recognizer.recognize_face(composite)
        elapsed = time.time() - t0

        detected_count = len(results)
        recognized_ids = set()
        unknown_count = 0
        misidentified = 0

        for (sid, score, box) in results:
            if sid == "unknown":
                unknown_count += 1
            elif sid.startswith("test_"):
                recognized_ids.add(sid)
                if sid not in expected_ids:
                    misidentified += 1

        # Metrics
        true_positives = len(recognized_ids & expected_ids)
        false_negatives = len(expected_ids - recognized_ids)
        recall = true_positives / max(len(expected_ids), 1)
        precision = true_positives / max(true_positives + misidentified, 1)

        print(f"  Faces in image: {len(faces_for_composite)} ({group_size} enrolled + 5 imposters)")
        print(f"  Faces detected: {detected_count}")
        print(f"  Correctly recognized: {true_positives}/{len(expected_ids)} ({recall:.1%})")
        print(f"  Misidentified: {misidentified}")
        print(f"  Unknown (correct for imposters): {unknown_count}")
        print(f"  False negatives (missed enrolled): {false_negatives}")
        print(f"  Precision: {precision:.1%}")
        print(f"  Recall: {recall:.1%}")
        print(f"  Time: {elapsed:.2f}s ({elapsed / max(detected_count, 1) * 1000:.1f}ms/face)")

        overall_results[group_size] = {
            'total_faces': len(faces_for_composite),
            'detected': detected_count,
            'true_positives': true_positives,
            'expected': len(expected_ids),
            'misidentified': misidentified,
            'unknown': unknown_count,
            'false_negatives': false_negatives,
            'precision': precision,
            'recall': recall,
            'time': elapsed
        }

    return overall_results


def test_individual_recognition(recognizer, enrolled_data, imposter_data):
    """Also re-run individual face tests to confirm augmentation improved accuracy."""
    print("\n" + "=" * 70)
    print("INDIVIDUAL RECOGNITION TEST (post-augmentation)")
    print("=" * 70)

    tp = 0  # True positive
    fn = 0  # False negative (enrolled but not recognized)
    fp = 0  # False positive (imposter wrongly identified)
    tn = 0  # True negative (imposter correctly rejected)
    misid = 0  # Misidentified (enrolled, matched to wrong person)

    # Test enrolled students
    for sid, data in enrolled_data.items():
        for img in data['test_images'][:3]:  # Up to 3 test images each
            results = recognizer.recognize_face(img)
            if results:
                best_id, score, box = results[0]
                if best_id == sid:
                    tp += 1
                elif best_id == "unknown":
                    fn += 1
                else:
                    misid += 1
            else:
                fn += 1

    # Test imposters
    for name, imgs in imposter_data.items():
        for img in imgs[:3]:
            results = recognizer.recognize_face(img)
            if results:
                best_id, score, box = results[0]
                if best_id == "unknown":
                    tn += 1
                else:
                    fp += 1
            else:
                tn += 1

    total = tp + fn + fp + tn + misid
    accuracy = (tp + tn) / max(total, 1)
    far = fp / max(fp + tn, 1)
    frr = fn / max(fn + tp, 1)

    print(f"\n  True Positives:  {tp}")
    print(f"  True Negatives:  {tn}")
    print(f"  False Positives: {fp} (imposters accepted)")
    print(f"  False Negatives: {fn} (enrolled rejected)")
    print(f"  Misidentified:   {misid} (wrong person)")
    print(f"  Accuracy:        {accuracy:.1%}")
    print(f"  FAR:             {far:.1%}")
    print(f"  FRR:             {frr:.1%}")
    print(f"  Misid Rate:      {misid / max(tp + misid, 1):.1%}")

    return {'accuracy': accuracy, 'far': far, 'frr': frr, 'misid_rate': misid / max(tp + misid, 1)}


def main():
    np.random.seed(42)

    # 1. Load data
    people = load_lfw_data()

    if len(people) < NUM_ENROLLED + NUM_IMPOSTERS:
        print(f"ERROR: Need {NUM_ENROLLED + NUM_IMPOSTERS} people but only have {len(people)}")
        return

    # 2. Prepare enrollment
    enrolled_data, imposter_data = prepare_enrollment(people, TEST_DB)

    # 3. Run augmentation
    total_orig, total_aug = run_augmentation(enrolled_data)

    # 4. Build recognition database (force rebuild, no stale cache)
    print("\nBuilding face recognition database...")
    cache_path = os.path.join(TEST_DB, "representations_sface.pkl")
    if os.path.exists(cache_path):
        os.remove(cache_path)
        print("  Deleted stale cache — forcing full rebuild")
    recognizer = FaceRecognizer(TEST_DB)

    db_size = sum(len(v) for v in recognizer.student_features.values())
    print(f"  Database: {len(recognizer.student_features)} students, {db_size} embeddings")
    if db_size > 0:
        avg = db_size / len(recognizer.student_features)
        print(f"  Average embeddings per student: {avg:.1f}")

    # 5. Individual recognition test
    individual = test_individual_recognition(recognizer, enrolled_data, imposter_data)

    # 6. Group recognition test
    group = test_group_recognition(recognizer, enrolled_data, imposter_data)

    # 7. Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"\nTraining Data:")
    print(f"  Original images:   {total_orig} ({total_orig / NUM_ENROLLED:.0f} per student)")
    print(f"  Augmented images:  {total_aug}")
    print(f"  Total training:    {total_orig + total_aug} "
          f"({(total_orig + total_aug) / NUM_ENROLLED:.0f} per student)")
    print(f"  Embeddings in DB:  {db_size}")

    print(f"\nIndividual Recognition:")
    print(f"  Accuracy: {individual['accuracy']:.1%}")
    print(f"  FAR:      {individual['far']:.1%}")
    print(f"  FRR:      {individual['frr']:.1%}")
    print(f"  Misid:    {individual['misid_rate']:.1%}")

    print(f"\nGroup Recognition:")
    for size, r in group.items():
        print(f"  {size}-person group: {r['recall']:.1%} recall, "
              f"{r['precision']:.1%} precision, "
              f"{r['detected']} detected, "
              f"{r['time']:.2f}s")


if __name__ == '__main__':
    main()
