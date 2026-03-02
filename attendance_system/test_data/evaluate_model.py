"""
=============================================================================
  FACE RECOGNITION PIPELINE — FULL EVALUATION ON LFW DATASET
  Simulates a 70-student classroom with real face images
=============================================================================
"""
import cv2
import numpy as np
import os
import sys
import time
import random
from collections import defaultdict

# Add parent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ─── CONFIG ───────────────────────────────────────────────────
LFW_DIR = os.path.join(os.path.dirname(__file__), 'lfw_home', 'lfw_funneled')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
NUM_ENROLLED = 70       # Simulate 70 enrolled students
NUM_IMPOSTERS = 30      # Non-enrolled people for false positive testing
ENROLL_IMAGES = 5       # Images per person for enrollment
MIN_IMAGES = 8          # Need at least this many images per person (enroll + test)
THRESHOLD = 0.363       # SFace recommended
THRESHOLDS_TO_TEST = [0.20, 0.25, 0.30, 0.363, 0.40, 0.45, 0.50]

random.seed(42)
np.random.seed(42)

# ─── LOAD MODELS ──────────────────────────────────────────────
det_path = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
rec_path = os.path.join(MODELS_DIR, "face_recognition_sface_2021dec.onnx")

detector = cv2.FaceDetectorYN.create(det_path, "", (320, 320))
recognizer = cv2.FaceRecognizerSF.create(rec_path, "")

# ─── COLLECT PEOPLE WITH ENOUGH IMAGES ───────────────────────
print("Scanning LFW dataset...")
people = {}
for person in os.listdir(LFW_DIR):
    pdir = os.path.join(LFW_DIR, person)
    if os.path.isdir(pdir):
        imgs = sorted([f for f in os.listdir(pdir) if f.endswith('.jpg')])
        if len(imgs) >= MIN_IMAGES:
            people[person] = imgs

print(f"Found {len(people)} people with {MIN_IMAGES}+ images")

if len(people) < NUM_ENROLLED + NUM_IMPOSTERS:
    print(f"WARNING: Not enough people. Reducing counts.")
    NUM_ENROLLED = min(NUM_ENROLLED, len(people) - 10)
    NUM_IMPOSTERS = min(NUM_IMPOSTERS, len(people) - NUM_ENROLLED)

all_people = list(people.keys())
random.shuffle(all_people)
enrolled_people = all_people[:NUM_ENROLLED]
imposter_people = all_people[NUM_ENROLLED:NUM_ENROLLED + NUM_IMPOSTERS]

print(f"Enrolled: {NUM_ENROLLED} | Imposters: {NUM_IMPOSTERS}")

# ─── HELPER: Extract face embedding from image ───────────────
def get_embedding(img_path):
    """Returns (embedding, detection_confidence) or (None, 0)"""
    img = cv2.imread(img_path)
    if img is None:
        return None, 0
    
    detector.setInputSize((img.shape[1], img.shape[0]))
    _, faces = detector.detect(img)
    
    if faces is None or len(faces) == 0:
        return None, 0
    
    # Use highest confidence face
    best_face = faces[np.argmax(faces[:, -1])]
    aligned = recognizer.alignCrop(img, best_face)
    feature = recognizer.feature(aligned)
    return feature, float(best_face[-1])


# ═══════════════════════════════════════════════════════════════
#  PHASE 1: ENROLLMENT — Build feature database
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  PHASE 1: ENROLLMENT")
print("=" * 65)

database = {}  # {person_name: [embedding1, ...]}
enroll_stats = {"success": 0, "fail_detect": 0, "total_images": 0}

for person in enrolled_people:
    imgs = people[person][:ENROLL_IMAGES]
    embeddings = []
    for img_name in imgs:
        enroll_stats["total_images"] += 1
        img_path = os.path.join(LFW_DIR, person, img_name)
        emb, conf = get_embedding(img_path)
        if emb is not None:
            embeddings.append(emb)
            enroll_stats["success"] += 1
        else:
            enroll_stats["fail_detect"] += 1
    
    if embeddings:
        database[person] = embeddings

print(f"  Enrolled {len(database)}/{NUM_ENROLLED} people successfully")
print(f"  Images processed: {enroll_stats['total_images']}")
print(f"  Faces detected: {enroll_stats['success']} ({100*enroll_stats['success']/enroll_stats['total_images']:.1f}%)")
print(f"  Failed detections: {enroll_stats['fail_detect']}")

avg_embeddings = sum(len(v) for v in database.values()) / len(database)
print(f"  Avg embeddings per person: {avg_embeddings:.1f}")


# ═══════════════════════════════════════════════════════════════
#  PHASE 2: RECOGNITION TESTING
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  PHASE 2: RECOGNITION TESTING")
print("=" * 65)

def match_face(query_emb, database, threshold):
    """Match a query embedding against the database. Returns (best_person, best_score)"""
    best_person = "unknown"
    best_score = 0.0
    
    for person, embeddings in database.items():
        for db_emb in embeddings:
            score = recognizer.match(query_emb, db_emb, 0)  # cosine similarity
            if score > best_score:
                best_score = score
                best_person = person
    
    if best_score < threshold:
        best_person = "unknown"
    
    return best_person, best_score


# Collect all test results with scores for threshold sweep
test_results = []  # (true_person, best_match, best_score, is_enrolled)

# Test enrolled people (genuine tests)
print("\n  Testing enrolled people (genuine recognition)...")
genuine_count = 0
for person in enrolled_people:
    if person not in database:
        continue
    test_imgs = people[person][ENROLL_IMAGES:]  # Use images NOT used for enrollment
    for img_name in test_imgs[:3]:  # Test with up to 3 images per person
        img_path = os.path.join(LFW_DIR, person, img_name)
        emb, conf = get_embedding(img_path)
        if emb is not None:
            # Find best match (threshold-independent)
            best_person = "unknown"
            best_score = 0.0
            for db_person, embeddings in database.items():
                for db_emb in embeddings:
                    score = recognizer.match(emb, db_emb, 0)
                    if score > best_score:
                        best_score = score
                        best_person = db_person
            
            test_results.append((person, best_person, best_score, True))
            genuine_count += 1

print(f"  Genuine test samples: {genuine_count}")

# Test imposter people (should be rejected)
print("  Testing imposters (should be rejected)...")
imposter_count = 0
for person in imposter_people:
    test_imgs = people[person][:3]
    for img_name in test_imgs:
        img_path = os.path.join(LFW_DIR, person, img_name)
        emb, conf = get_embedding(img_path)
        if emb is not None:
            best_person = "unknown"
            best_score = 0.0
            for db_person, embeddings in database.items():
                for db_emb in embeddings:
                    score = recognizer.match(emb, db_emb, 0)
                    if score > best_score:
                        best_score = score
                        best_person = db_person
            
            test_results.append((person, best_person, best_score, False))
            imposter_count += 1

print(f"  Imposter test samples: {imposter_count}")


# ═══════════════════════════════════════════════════════════════
#  PHASE 3: THRESHOLD SWEEP & METRICS
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  PHASE 3: THRESHOLD ANALYSIS")
print("=" * 65)

print(f"\n  {'Threshold':>10} | {'Accuracy':>8} | {'Precision':>9} | {'Recall':>7} | {'F1':>7} | {'FAR':>7} | {'FRR':>7}")
print("  " + "-" * 72)

best_f1 = 0
best_thresh = 0

for thresh in THRESHOLDS_TO_TEST:
    tp = 0  # Enrolled person correctly identified
    fp = 0  # Imposter falsely accepted OR enrolled person misidentified
    tn = 0  # Imposter correctly rejected
    fn = 0  # Enrolled person not recognized (false reject)
    
    for true_person, best_match, best_score, is_enrolled in test_results:
        if is_enrolled:
            if best_score >= thresh and best_match == true_person:
                tp += 1
            elif best_score >= thresh and best_match != true_person:
                fp += 1  # Misidentification
            else:
                fn += 1  # Not recognized
        else:  # imposter
            if best_score < thresh:
                tn += 1  # Correctly rejected
            else:
                fp += 1  # Falsely accepted
    
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total * 100 if total > 0 else 0
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    far = fp / (fp + tn) * 100 if (fp + tn) > 0 else 0  # False Accept Rate
    frr = fn / (fn + tp) * 100 if (fn + tp) > 0 else 0  # False Reject Rate
    
    marker = ""
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = thresh
    if thresh == 0.30:
        marker = " ← current"
    if thresh == 0.363:
        marker = " ← paper default"
    
    print(f"  {thresh:>10.3f} | {accuracy:>7.1f}% | {precision:>8.1f}% | {recall:>6.1f}% | {f1:>6.1f}% | {far:>6.2f}% | {frr:>6.1f}%{marker}")

print(f"\n  ★ Best F1 score at threshold: {best_thresh}")


# ═══════════════════════════════════════════════════════════════
#  PHASE 4: SCORE DISTRIBUTION ANALYSIS
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  PHASE 4: SCORE DISTRIBUTION")
print("=" * 65)

genuine_scores = [s for _, _, s, enrolled in test_results if enrolled]
imposter_scores = [s for _, _, s, enrolled in test_results if not enrolled]

print(f"\n  Genuine matches (enrolled people):")
print(f"    Count : {len(genuine_scores)}")
print(f"    Mean  : {np.mean(genuine_scores):.4f}")
print(f"    Std   : {np.std(genuine_scores):.4f}")
print(f"    Min   : {np.min(genuine_scores):.4f}")
print(f"    Max   : {np.max(genuine_scores):.4f}")
print(f"    < 0.3 : {sum(1 for s in genuine_scores if s < 0.3)} ({100*sum(1 for s in genuine_scores if s < 0.3)/len(genuine_scores):.1f}%)")
print(f"    < 0.363: {sum(1 for s in genuine_scores if s < 0.363)} ({100*sum(1 for s in genuine_scores if s < 0.363)/len(genuine_scores):.1f}%)")

print(f"\n  Imposter matches (non-enrolled):")
print(f"    Count : {len(imposter_scores)}")
print(f"    Mean  : {np.mean(imposter_scores):.4f}")
print(f"    Std   : {np.std(imposter_scores):.4f}")
print(f"    Min   : {np.min(imposter_scores):.4f}")
print(f"    Max   : {np.max(imposter_scores):.4f}")
print(f"    > 0.3 : {sum(1 for s in imposter_scores if s > 0.3)} ({100*sum(1 for s in imposter_scores if s > 0.3)/len(imposter_scores):.1f}%)")
print(f"    > 0.363: {sum(1 for s in imposter_scores if s > 0.363)} ({100*sum(1 for s in imposter_scores if s > 0.363)/len(imposter_scores):.1f}%)")

separation = np.mean(genuine_scores) - np.mean(imposter_scores)
print(f"\n  Score separation (genuine mean - imposter mean): {separation:.4f}")
if separation > 0.3:
    print(f"  ✅ GOOD separation — model distinguishes well")
elif separation > 0.15:
    print(f"  ⚠️  MODERATE separation — may need improvements")
else:
    print(f"  ❌ POOR separation — model needs significant improvements")


# ═══════════════════════════════════════════════════════════════
#  PHASE 5: SPEED BENCHMARK (70 students)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  PHASE 5: SPEED BENCHMARK (70-STUDENT DATABASE)")
print("=" * 65)

# Pick a random test image
test_person = random.choice(enrolled_people)
test_img_path = os.path.join(LFW_DIR, test_person, people[test_person][ENROLL_IMAGES])
test_img = cv2.imread(test_img_path)

# Detection speed
detector.setInputSize((test_img.shape[1], test_img.shape[0]))
times_det = []
for _ in range(20):
    t0 = time.perf_counter()
    _, faces = detector.detect(test_img)
    times_det.append((time.perf_counter() - t0) * 1000)

# Extraction speed
aligned = recognizer.alignCrop(test_img, faces[0])
times_ext = []
for _ in range(20):
    t0 = time.perf_counter()
    feat = recognizer.feature(aligned)
    times_ext.append((time.perf_counter() - t0) * 1000)

# Matching speed (against full 70-student DB)
times_match = []
for _ in range(10):
    t0 = time.perf_counter()
    for person, embeddings in database.items():
        for db_emb in embeddings:
            recognizer.match(feat, db_emb, 0)
    times_match.append((time.perf_counter() - t0) * 1000)

total_comparisons = sum(len(v) for v in database.values())

avg_det = np.mean(times_det)
avg_ext = np.mean(times_ext)
avg_match = np.mean(times_match)
avg_total = avg_det + avg_ext + avg_match

print(f"\n  Detection     : {avg_det:.2f} ms")
print(f"  Extraction    : {avg_ext:.2f} ms")
print(f"  Matching      : {avg_match:.2f} ms ({total_comparisons} comparisons)")
print(f"  Total/frame   : {avg_total:.2f} ms")
print(f"  FPS           : {1000/avg_total:.1f}")


# ═══════════════════════════════════════════════════════════════
#  PHASE 6: MISIDENTIFICATION ANALYSIS
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  PHASE 6: ERROR ANALYSIS (at threshold 0.363)")
print("=" * 65)

misid = []
false_rejects = []
false_accepts = []

for true_person, best_match, best_score, is_enrolled in test_results:
    if is_enrolled:
        if best_score >= 0.363 and best_match != true_person:
            misid.append((true_person, best_match, best_score))
        elif best_score < 0.363:
            false_rejects.append((true_person, best_score))
    else:
        if best_score >= 0.363:
            false_accepts.append((true_person, best_match, best_score))

print(f"\n  Misidentifications: {len(misid)}")
for true, pred, score in misid[:5]:
    print(f"    {true[:25]:25s} → {pred[:25]:25s}  (score: {score:.4f})")

print(f"\n  False Rejects (enrolled but not recognized): {len(false_rejects)}")
for person, score in sorted(false_rejects, key=lambda x: x[1])[:5]:
    print(f"    {person[:35]:35s}  best_score: {score:.4f}")

print(f"\n  False Accepts (imposter accepted): {len(false_accepts)}")
for imp, matched, score in sorted(false_accepts, key=lambda x: -x[2])[:5]:
    print(f"    {imp[:20]:20s} matched → {matched[:20]:20s}  (score: {score:.4f})")


# ═══════════════════════════════════════════════════════════════
#  SUMMARY
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  FINAL SUMMARY")
print("=" * 65)

# Use threshold 0.363
tp = sum(1 for t, m, s, e in test_results if e and s >= 0.363 and m == t)
total_genuine = sum(1 for _, _, _, e in test_results if e)
total_imposter = sum(1 for _, _, _, e in test_results if not e)
fp_imposter = sum(1 for t, m, s, e in test_results if not e and s >= 0.363)
misid_count = sum(1 for t, m, s, e in test_results if e and s >= 0.363 and m != t)

print(f"""
  Database size      : {len(database)} people ({total_comparisons} embeddings)
  Test samples       : {len(test_results)} ({total_genuine} genuine + {total_imposter} imposter)

  Recognition rate   : {tp}/{total_genuine} = {100*tp/total_genuine:.1f}%
  Misidentification  : {misid_count}/{total_genuine} = {100*misid_count/total_genuine:.1f}%
  False reject rate  : {total_genuine - tp - misid_count}/{total_genuine} = {100*(total_genuine - tp - misid_count)/total_genuine:.1f}%
  False accept rate  : {fp_imposter}/{total_imposter} = {100*fp_imposter/total_imposter:.1f}%
  
  Speed (70 students): {avg_total:.1f} ms/face = {1000/avg_total:.0f} FPS
  Best threshold     : {best_thresh} (F1: {best_f1:.1f}%)
""")

print("=" * 65)
print("  EVALUATION COMPLETE")
print("=" * 65)
