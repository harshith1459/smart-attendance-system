# AttendEase — Smart Attendance Management System

A full-stack web application for managing student attendance using **face recognition**, built with Flask, MongoDB, and OpenCV.

![Python](https://img.shields.io/badge/Python-3.14-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-3.1-green?logo=flask)
![MongoDB](https://img.shields.io/badge/MongoDB-NoSQL-brightgreen?logo=mongodb)
![OpenCV](https://img.shields.io/badge/OpenCV-4.11-red?logo=opencv)

---

## Features

### 👨‍💼 Admin
- Dashboard with system-wide statistics
- Approve / reject new user registrations
- View all students and their attendance history
- Manage weekly timetable (7 periods × 6 days)

### 👩‍🏫 Teacher
- View today's class schedule with live attendance counts
- **Take attendance via webcam** (face recognition powered)
- Manual bulk attendance marking
- Submit special attendance requests

### 🎓 Student
- Personal dashboard with attendance percentage and history
- **Enroll face via webcam** (5-frame capture)
- View attendance status per subject

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Flask 3.1, Jinja2 |
| Database | MongoDB + MongoEngine ODM |
| Auth | Flask-Login + Werkzeug password hashing |
| Forms | Flask-WTF + WTForms |
| Face Detection | OpenCV YuNet (ONNX) |
| Face Recognition | OpenCV SFace (ONNX) |
| Frontend | Custom CSS (dark theme), Vanilla JS |
| Icons/Fonts | Font Awesome 6, Google Fonts (Inter) |

---

## Project Structure

```
attendance_system/
├── app.py                  # App factory & blueprint registration
├── config.py               # MongoDB URI, secrets, upload config
├── extensions.py           # MongoEngine & LoginManager instances
├── forms.py                # WTForms (Login, Register, AttendanceRequest)
├── requirements.txt        # Python dependencies
├── models/                 # MongoEngine document models
│   ├── user.py             # User (with role-based inheritance)
│   ├── student.py          # Student profile + face enrollment
│   ├── teacher.py          # Teacher profile
│   ├── timetable.py        # Weekly schedule entries
│   └── attendance.py       # Attendance records & requests
├── routes/                 # Flask Blueprints
│   ├── auth.py             # Login, Register, Logout
│   ├── admin.py            # Admin dashboard & user management
│   ├── teacher.py          # Attendance marking & bulk ops
│   ├── student.py          # Student dashboard & face enrollment
│   └── main.py             # Role-based redirect
├── services/               # Business logic layer
│   ├── auth_service.py     # Authentication & registration
│   ├── admin_service.py    # Stats, approvals, timetable
│   ├── attendance_service.py # Attendance recording & queries
│   └── student_service.py  # Student stats & biometric frames
├── ml/                     # Face recognition pipeline
│   ├── detector.py         # YuNet face detection
│   └── recognize.py        # SFace embedding & matching
├── templates/              # Jinja2 HTML templates
├── static/                 # CSS, JS, uploads
├── scripts/                # DB seed scripts
└── deployment/             # Production config (nginx, gunicorn)
```

---

## Setup & Installation

### Prerequisites
- Python 3.10+
- MongoDB running on `localhost:27017`

### Steps

```bash
# 1. Clone the repo
git clone https://github.com/harshith1459/smart-attendance-system.git
cd smart-attendance-system

# 2. Create virtual environment
python -m venv .venv
source .venv/Scripts/activate    # Windows
# source .venv/bin/activate      # Mac/Linux

# 3. Install dependencies
cd attendance_system
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env with your SECRET_KEY and MONGODB_URI

# 5. Seed the admin account
python scripts/seed_admin.py

# 6. Run the app
python app.py
```

Open **http://localhost:5000** in your browser.

---



> Create teacher and student accounts via the registration page, then approve them from the admin dashboard.

---

## Face Recognition Pipeline

1. **Enrollment**: Student captures 5 webcam frames → saved as images in `static/uploads/dataset/`
2. **Feature Extraction**: SFace model extracts 128-d face embeddings → cached in `.pkl` file
3. **Recognition**: Camera frame → YuNet detects faces → SFace extracts embedding → cosine similarity match → threshold 0.3 → mark present

**Models used:**
- `face_detection_yunet_2023mar.onnx` — Face detection
- `face_recognition_sface_2021dec.onnx` — Face recognition

> ⚠️ ONNX model files are not included in the repo (gitignored due to size). Download them from [OpenCV Zoo](https://github.com/opencv/opencv_zoo) and place in `attendance_system/models/`.

---

## License

This project is for educational purposes.
