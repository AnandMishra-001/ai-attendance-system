# 🎓 AI-Based Smart Attendance System

An intelligent attendance management system that uses **Facial Recognition** to automatically detect and mark attendance in real-time — no manual effort required.

---

## 🚀 Features

- 📷 **Real-Time Face Detection** — Automatically detects and recognizes faces using your webcam
- ✅ **Automatic Attendance Marking** — Marks attendance instantly upon face recognition
- 📊 **Attendance Reports** — Generates and exports attendance reports for easy tracking
- 🔒 **Secure & Accurate** — AI-powered recognition ensures reliable identification

---

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Core programming language |
| OpenCV | Real-time face detection & image processing |
| Machine Learning | Face recognition model |

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.x installed
- Webcam/Camera

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/AnandMishra-001/ai-attendance-system.git
   cd ai-attendance-system
   ```

2. **Install required libraries**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python main.py
   ```

---

## 📁 Project Structure

```
ai-attendance-system/
│
├── main.py               # Main application file
├── requirements.txt      # Required dependencies
├── attendance/           # Attendance records/exports
├── dataset/              # Face image dataset
└── README.md             # Project documentation
```

---

## 📸 How It Works

1. **Register** — Capture and store face data for each person
2. **Detect** — System scans faces in real-time via webcam
3. **Recognize** — AI model matches detected face with stored data
4. **Mark** — Attendance is automatically recorded with timestamp
5. **Export** — Generate attendance reports anytime

---

## 📋 Requirements

```
opencv-python
numpy
pandas
face-recognition
```

---

## 🙋‍♂️ Author

**Anand Mishra**
- GitHub: [@AnandMishra-001](https://github.com/AnandMishra-001)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

⭐ If you found this project helpful, please give it a star on GitHub!
