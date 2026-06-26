# 😷 Face Mask Detection System

A deep learning-based Face Mask Detection web application built using **FastAPI**, **TensorFlow**, and **Keras**. The application allows users to upload an image and predicts whether the person in the image is wearing a face mask, along with the prediction confidence.

## 🚀 Live Demo

🔗 **Live Application:**  https://face-mask-detector-mlue.onrender.com

## 📌 Features

- Upload an image for face mask detection
- Predicts **With Mask** or **Without Mask**
- Displays prediction confidence
- FastAPI backend with TensorFlow model
- Simple and user-friendly interface
- Deployed using Docker on Render

## 🛠️ Tech Stack

- Python
- FastAPI
- TensorFlow
- Keras
- NumPy
- Pillow (PIL)
- Jinja2
- HTML/CSS
- Docker
- Render

## 📂 Project Structure

```
face-mask-detector/
│── templates/
│   └── index.html
│── Dockerfile
│── Procfile
│── runtime.txt
│── requirements.txt
│── main.py
│── face_mask_model.h5
│── .gitignore
└── README.md
```

## ⚙️ Installation

### Clone the Repository

```bash
git clone https://github.com/Abhijeet718/face-mask-detector.git
cd face-mask-detector
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
uvicorn main:app --reload
```

Open your browser and visit:

```
http://127.0.0.1:8000
```

## 🧠 Model Details

- Framework: TensorFlow / Keras
- Model Format: `.h5`
- Input Image Size: **128 × 128**
- Classes:
  - 😷 With Mask
  - ❌ Without Mask


## 📈 Future Enhancements

- Real-time webcam detection
- Video-based mask detection
- Multi-face detection
- Improved model accuracy
- Mobile-responsive UI

## 👨‍💻 Author

**Abhijeet Kumar**

GitHub: https://github.com/Abhijeet718

---

⭐ If you found this project useful, consider giving it a star on GitHub!