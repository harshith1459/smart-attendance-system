"""
Compare OLD recognize.py vs NEW recognize_v2.py on LFW dataset
"""
import cv2
import numpy as np
import os
import sys
import time
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

LFW_DIR = os.path.join(os.path.dirname(__file__), 'lfw_home', 'lfw_funneled')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
NUM_ENROLLED = 70
NUM_IMPOSTERS = 30
ENROLL_IMAGES = 5
MIN_IMAGES = 8

random.seed(42)
np.random.seed(42)

det_path = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
rec_path = os.path.join(MODELS_DIR, "face_recognition_sface_2021dec.onnx")
detector = cv2.FaceDetectorYN.create(det_path, "", (320, 320))
recognizer = cv2.FaceRecognizerSF.create(rec_path, "")

# ─── Collect people ──────────────────────────────────────────
people = {}
for person in os.listdir(LFW_DIR):
    pdir = os.path.join(LFW_DIR, person)
    if os.path.isdir(pdir):
        imgs = sorted([f for f in os.listdir(pdir) if f.endswith('.jpg')])
        if len(imgs) >= MIN_IMAGES:
            people[person] = imgs

all_people = list(people.keys())
random.shuffle(all_people)
enrolled_people = all_people[:NUM_ENROLLED]
imposter_people = all_people[NUM_ENROLLED:NUM_ENROLLED + NUM_IMPOSTERS]

# ─── Build embeddings ────────────────────────────────────────
def get_embedding(img_path):
    img = cv2.imread(img_path)
    if img is None: return None, 0
    detector.setInputSize((img.shape[1], img.shape[0]))
    _, faces = detector.detect(img)
    if faces is None or len(faces) == 0: return None, 0
    best = faces[np.argmax(faces[:, -1])]
    aligned = recognizer.alignCrop(img, best)
    return recognizer.feature(aligned), float(best[-1])

# Enroll
print("Enrolling 70 students...")
database = {}
for person in enrolled_people:
    embeddings = []
    for img_name in people[person][:ENROLL_IMAGES]:
        emb, _ = get_embedding(os.path.join(LFW_DIR, person, img_name))
        if emb is not None:
            embeddings.append(emb)
    if embeddings:
        database[person] = embeddings

# Build centroids (for V2)
centroids = {}
for sid, embs in database.items():
    stacked = np.vstack(embs)
    centroid = np.mean(stacked, axis=0, keepdims=True)
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm * np.linalg.norm(embs[0])
    centroids[sid] = centroid

# Collect test embeddings
print("Building test set...")
test_data = []
for person in enrolled_people:
    if person not in database: continue
    for img_name in people[person][ENROLL_IMAGES:ENROLL_IMAGES+3]:
        emb, _ = get_embedding(os.path.join(LFW_DIR, person, img_name))
        if emb is not None:
            test_data.append((person, emb, True))

for person in imposter_people:
    for img_name in people[person][:3]:
        emb, _ = get_embedding(os.path.join(LFW_DIR, person, img_name))
        if emb is not None:
            test_data.append((person, emb, False))

print(f"Test set: {len(test_data)} samples ({sum(1 for _,_,e in test_data if e)} genuine, {sum(1 for _,_,e in test_data if not e)} imposter)")

# ─── OLD METHOD: brute force, threshold 0.30 ─────────────────
def match_old(query_emb, threshold=0.30):
    best_person, best_score = "unknown", 0.0
    for person, embs in database.items():
        for db_emb in embs:
            score = recognizer.match(query_emb, db_emb, 0)
            if score > best_score:
                best_score = score
                best_person = person
    if best_score < threshold:
        best_person = "unknown"
    return best_person, best_score

# ─── NEW METHOD: centroid + margin, threshold 0.45 ───────────
def match_new(query_emb, threshold=0.45, margin=0.05):
    scores = {}
    for sid, centroid in centroids.items():
        scores[sid] = recognizer.match(query_emb, centroid, 0)
    
    sorted_s = sorted(scores.items(), key=lambda x: -x[1])
    if not sorted_s:
        return "unknown", 0.0
    
    top_id, top_score = sorted_s[0]
    if top_score < threshold:
        return "unknown", top_score
    
    if len(sorted_s) >= 2:
        second_score = sorted_s[1][1]
        if top_score - second_score < margin:
            return "unknown", top_score  # ambiguous
    
    return top_id, top_score

# ─── RUN COMPARISON ──────────────────────────────────────────
print("\n" + "=" * 70)
print("  COMPARISON: OLD vs NEW RECOGNITION")
print("=" * 70)

for label, match_fn in [("OLD (brute, t=0.30)", match_old), ("NEW (centroid+margin, t=0.45)", match_new)]:
    tp = fp = tn = fn = misid = 0
    
    t0 = time.perf_counter()
    for true_person, emb, is_enrolled in test_data:
        pred, score = match_fn(emb)
        if is_enrolled:
            if pred == true_person: tp += 1
            elif pred != "unknown": fp += 1; misid += 1
            else: fn += 1
        else:
            if pred == "unknown": tn += 1
            else: fp += 1
    elapsed = (time.perf_counter() - t0) * 1000
    
    total = tp + fp + tn + fn
    acc = (tp + tn) / total * 100
    prec = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    far = fp / (fp + tn) * 100 if (fp + tn) > 0 else 0
    frr = fn / (fn + tp) * 100 if (fn + tp) > 0 else 0
    genuine = sum(1 for _,_,e in test_data if e)
    imposter = sum(1 for _,_,e in test_data if not e)
    
    print(f"\n  ── {label} ──")
    print(f"  Accuracy     : {acc:.1f}%")
    print(f"  Precision    : {prec:.1f}%")
    print(f"  Recall       : {rec:.1f}%")
    print(f"  F1 Score     : {f1:.1f}%")
    print(f"  FAR          : {far:.2f}%   (false accepts)")
    print(f"  FRR          : {frr:.1f}%   (false rejects)")
    print(f"  Misidentify  : {misid}/{genuine} ({100*misid/genuine:.1f}%)")
    print(f"  Speed        : {elapsed:.1f} ms for {len(test_data)} samples ({elapsed/len(test_data):.2f} ms/sample)")

# Speed comparison for single face vs 70-student DB
print("\n" + "=" * 70)
print("  SPEED COMPARISON: SINGLE FACE LOOKUP")
print("=" * 70)

test_emb = test_data[0][1]

# Old: brute force
times_old = []
for _ in range(50):
    t0 = time.perf_counter()
    match_old(test_emb)
    times_old.append((time.perf_counter() - t0) * 1000)

# New: centroid
times_new = []
for _ in range(50):
    t0 = time.perf_counter()
    match_new(test_emb)
    times_new.append((time.perf_counter() - t0) * 1000)

print(f"\n  OLD (brute, {sum(len(v) for v in database.values())} comparisons) : {np.mean(times_old):.3f} ms/lookup")
print(f"  NEW (centroid, {len(centroids)} comparisons)  : {np.mean(times_new):.3f} ms/lookup")
print(f"  Speedup: {np.mean(times_old)/np.mean(times_new):.1f}x faster")

print("\n" + "=" * 70)
print("  DONE")
print("=" * 70)
