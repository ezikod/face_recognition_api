# 🎭 Face Recognition API

A FastAPI-based face recognition system that can identify faces in photos and videos. The system maintains a database of known people and can recognize them in real-time.

## 🚀 Features

- **Add People**: Register new people with their photos
- **Face Recognition**: Identify people in uploaded photos
- **Video Processing**: Recognize faces in video files
- **People Management**: View, manage, and delete registered people
- **REST API**: Full RESTful API with Swagger documentation
- **Web Interface**: User-friendly HTML interface for testing

## 📋 Requirements

- Python 3.8+
- OpenCV
- face-recognition library
- FastAPI
- See `requirements.txt` for full list

## 🛠️ Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd face-recognition-api
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🏃‍♂️ Running the Application

1. Navigate to the API directory:
```bash
cd API/src
```

2. Start the server:
```bash
python main.py
```

The server will start on `http://localhost:8000`

## 🖥️ Using the Web Interface

Open `test_page.html` in your web browser to access the user interface.

### Available Functions:

1. **Add Person**: Upload a photo with a person's name to register them
2. **Recognize Photo**: Upload a photo to identify people in it
3. **Recognize Video**: Upload a video file to identify people throughout the video
4. **View Database**: See all registered people
5. **Statistics**: View system statistics

## 📡 API Endpoints

### Database Management
- `POST /api/person/add` - Add new person
- `GET /api/person/list` - List all people
- `GET /api/person/{id}` - Get specific person info
- `DELETE /api/person/{id}` - Delete person

### Recognition
- `POST /api/person/recognize` - Recognize faces in photo
- `POST /api/video/recognize` - Recognize faces in video
- `POST /api/person/recognize-base64` - Recognize faces in base64 image

### System
- `GET /` - API health check
- `GET /api/stats` - System statistics

## 📚 API Documentation

When the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 📁 Project Structure

```
API/
├── src/
│   └── main.py          # Main application file
├── uploads/             # Uploaded photos storage
├── persons_data.csv     # Database file
├── test_page.html       # Web interface
├── debug_video.html     # Debug interface for video
└── requirements.txt     # Python dependencies
```

## 🎥 Video Processing

When processing videos:
- **Frame Interval**: Process every Nth frame (default: 15)
  - Lower values = more accurate but slower
  - Higher values = faster but may miss short appearances
- **Key Frames**: Up to 10 annotated frames are returned
- **Results**: Includes timestamps and confidence scores

## 🔧 Configuration

Key settings in `main.py`:
- `CSV_FILE`: Database file location
- `UPLOAD_DIR`: Directory for uploaded files
- `frame_interval`: Video processing interval

## 🐛 Troubleshooting

### Common Issues:

1. **"No module named 'face_recognition'"**
   - Make sure you've installed all requirements
   - On some systems, you may need to install dlib first

2. **Video processing is slow**
   - Increase the frame_interval parameter
   - Use smaller video files
   - Ensure your system has adequate resources

3. **Face not detected**
   - Ensure good lighting in photos
   - Face should be clearly visible and frontal
   - Photo resolution should be adequate

## 📝 Data Storage

- **Database**: CSV file storing person info and face encodings
- **Photos**: Stored in the uploads directory
- **Temporary**: Video files are deleted after processing

## 🔐 Security Notes

- CORS is configured to accept all origins (`*`) - restrict this in production
- No authentication is implemented - add security layers for production use
- Uploaded files are stored locally - implement proper file management

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## 📄 License

This project is open source. Please check the license file for more details.