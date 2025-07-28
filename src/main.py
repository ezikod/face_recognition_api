import os
import io
import csv
import json
import logging
import cv2
import base64
import uvicorn
import numpy as np
from PIL import Image
from datetime import datetime
from typing import List, Optional, Dict
import asyncio
from enum import Enum

from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import face_recognition

from functools import wraps
from typing import Callable
import traceback



# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Processing states
class ProcessingState(Enum):
    IDLE = "idle"
    PROCESSING_IMAGE = "processing_image"
    PROCESSING_VIDEO = "processing_video"

# Global state management
class APIState:
    def __init__(self):
        self.current_state = ProcessingState.IDLE
        self.lock = asyncio.Lock()
        self.current_task_info = None

api_state = APIState()

# --- Pydantic models for validation and documentation ---
# These models will be displayed in Swagger (/docs)

class FaceLocation(BaseModel):
    top: int = Field(..., description="Top coordinate of face bounding box")
    right: int = Field(..., description="Right coordinate of face bounding box")
    bottom: int = Field(..., description="Bottom coordinate of face bounding box")
    left: int = Field(..., description="Left coordinate of face bounding box")

class RecognizedFace(BaseModel):
    person_id: int = Field(..., description="Person ID from database (0 for unknown)")
    name: str = Field(..., description="Person's first name")
    last_name: Optional[str] = Field(None, description="Person's last name")
    workplace: Optional[str] = Field(None, description="Person's workplace")
    email: Optional[str] = Field(None, description="Person's email address")
    phone: Optional[str] = Field(None, description="Person's phone number")
    added_date: Optional[str] = Field(None, description="Date when person was added to database")
    confidence: float = Field(..., description="Recognition confidence (0.0-1.0)")
    location: FaceLocation = Field(..., description="Face coordinates on image")
    encoding: List[float] = Field(..., description="Face feature vector of 128 numbers")

class RecognitionResponseData(BaseModel):
    faces_count: int = Field(..., description="Total number of faces found")
    faces: List[RecognizedFace] = Field(..., description="List of recognized faces with full person data")
    processed_image_base64: str = Field(..., description="Image with drawn bounding boxes in Base64 format")

class PersonInDB(BaseModel):
    id: int
    name: str
    last_name: Optional[str] = None
    workplace: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    added_date: str
    photo_path: str

class AddPersonResponseData(BaseModel):
    """Response data for add person endpoint"""
    id: int = Field(..., description="Assigned person ID")
    name: str = Field(..., description="Person's first name")
    last_name: Optional[str] = Field(None, description="Person's last name")
    workplace: Optional[str] = Field(None, description="Person's workplace")
    email: Optional[str] = Field(None, description="Person's email")
    phone: Optional[str] = Field(None, description="Person's phone")
    faces_found: int = Field(..., description="Number of faces found in uploaded photo")
    photo_path: str = Field(..., description="Path where photo was saved")

class VideoRecognitionFace(BaseModel):
    person_id: int = Field(..., description="Person ID from database (negative for temporary unknown IDs)")
    name: str = Field(..., description="Person's first name or Unknown #N")
    last_name: Optional[str] = Field(None, description="Person's last name")
    workplace: Optional[str] = Field(None, description="Person's workplace")
    email: Optional[str] = Field(None, description="Person's email address")
    phone: Optional[str] = Field(None, description="Person's phone number")
    added_date: Optional[str] = Field(None, description="Date when person was added to database")
    confidence: float = Field(..., description="Recognition confidence (0.0-1.0)")
    timestamp: float = Field(..., description="Time in seconds when face was detected")
    frame_number: int = Field(..., description="Frame number where face was detected")
    location: FaceLocation = Field(..., description="Face coordinates on frame")

class PersonAppearance(BaseModel):
    person_id: int = Field(..., description="Person ID from database")
    name: str = Field(..., description="Person's first name")
    last_name: Optional[str] = Field(None, description="Person's last name")
    workplace: Optional[str] = Field(None, description="Person's workplace")
    email: Optional[str] = Field(None, description="Person's email address")
    phone: Optional[str] = Field(None, description="Person's phone number")
    first_seen: float = Field(..., description="First appearance timestamp in seconds")
    last_seen: float = Field(..., description="Last appearance timestamp in seconds")
    total_appearances: int = Field(..., description="Total number of appearances in video")

class VideoKeyFrame(BaseModel):
    frame_number: int = Field(..., description="Frame number")
    timestamp: float = Field(..., description="Timestamp in seconds")
    faces_count: int = Field(..., description="Number of faces in this frame")
    image_base64: str = Field(..., description="Annotated frame image in Base64 format")

class VideoRecognitionSummary(BaseModel):
    total_frames: int = Field(..., description="Total frames in video")
    processed_frames: int = Field(..., description="Number of frames processed")
    fps: float = Field(..., description="Video frames per second")
    duration_seconds: float = Field(..., description="Video duration in seconds")
    unique_persons: int = Field(..., description="Number of unique persons detected")
    person_appearances: Dict[str, PersonAppearance] = Field(..., description="Statistics for each detected person")

class StatsData(BaseModel):
    total_persons: int
    last_added: Optional[dict] = None

class ApiResponse(BaseModel):
    """Standard API response format"""
    success: bool = Field(..., description="Operation success status")
    message: str = Field(..., description="Human-readable message")
    data: Optional[dict] = Field(None, description="Response data (structure depends on endpoint)")


# Create FastAPI application
app = FastAPI(
    title="Face Recognition API",
    description="API for face recognition with extended person profiles support. Recognizes faces in photos and videos, returns complete person information including name, last name, workplace, email, and phone.",
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS setup for C# client compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Specify specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Request: {request.method} {request.url.path}")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response

# Global variables
CSV_FILE = "persons_data.csv"  # Path to CSV database file
UPLOAD_DIR = "uploads"  # Directory for uploaded photos

# In-memory data storage for fast access
known_encodings = []  # List of numpy arrays with face encodings
known_names = []  # List of person names (first names only)
known_ids = []  # List of person IDs
known_persons_data = {}  # Dictionary mapping person_id to full person data (all fields)

# Create necessary directories
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Data models
class PersonResponse(BaseModel):
    """Complete person profile from database"""
    id: int = Field(..., description="Person ID")
    name: str = Field(..., description="First name")
    last_name: Optional[str] = Field(None, description="Last name")
    workplace: Optional[str] = Field(None, description="Workplace")
    email: Optional[str] = Field(None, description="Email address")
    phone: Optional[str] = Field(None, description="Phone number")
    added_date: str = Field(..., description="Date added to database")
    photo_path: str = Field(..., description="Path to person's photo")

class RecognitionResult(BaseModel):
    """Legacy model - not used in current version"""
    face_id: int
    name: str
    confidence: float
    location: dict

# Helper function to get person data by ID
def get_person_data_by_id(person_id: int) -> Optional[Dict]:
    """
    Get full person data from memory by ID.
    
    Args:
        person_id: Person's database ID
        
    Returns:
        Dictionary with person data or None if not found:
        - id: Person ID
        - name: First name
        - last_name: Last name
        - workplace: Workplace
        - email: Email address
        - phone: Phone number
        - added_date: Date added to database
        - photo_path: Path to person's photo
    """
    return known_persons_data.get(person_id)

# Initialization
def check_and_copy_old_data():
    """Check for data in old location and copy if needed"""
    old_csv = "../persons_data.csv"
    old_uploads = "../uploads"
    
    # First, check if we need to copy from old location
    if os.path.exists(old_csv):
        logger.info(f"Found old CSV at {old_csv}")
        
        if not os.path.exists(CSV_FILE):
            # New CSV doesn't exist, copy from old
            logger.info(f"New CSV doesn't exist, copying from old location")
            try:
                import shutil
                shutil.copy2(old_csv, CSV_FILE)
                logger.info(f"Copied CSV from {old_csv} to {CSV_FILE}")
                
                # Also copy uploads directory if exists
                if os.path.exists(old_uploads):
                    if not os.path.exists(UPLOAD_DIR):
                        shutil.copytree(old_uploads, UPLOAD_DIR)
                        logger.info(f"Copied uploads from {old_uploads} to {UPLOAD_DIR}")
                    else:
                        # Copy individual files that don't exist
                        for file in os.listdir(old_uploads):
                            old_file = os.path.join(old_uploads, file)
                            new_file = os.path.join(UPLOAD_DIR, file)
                            if not os.path.exists(new_file) and os.path.isfile(old_file):
                                shutil.copy2(old_file, new_file)
                                logger.info(f"Copied file {file} to new uploads directory")
            except Exception as e:
                logger.error(f"Error copying old data: {e}")
        else:
            # Both files exist, compare them
            with open(old_csv, 'r') as f:
                old_lines = len(f.readlines())
            with open(CSV_FILE, 'r') as f:
                new_lines = len(f.readlines())
            
            if old_lines > new_lines:
                logger.warning(f"⚠️ Old CSV has {old_lines} lines, new CSV has {new_lines} lines")
                logger.warning(f"The old file has more data! Consider backing up and manually merging.")
                
                # Create a special backup of the old file
                backup_name = f"persons_data_old_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                import shutil
                shutil.copy2(old_csv, backup_name)
                logger.info(f"Created backup of old file: {backup_name}")

def migrate_csv_if_needed():
    """Migrate old CSV format to new format with extended fields"""
    if not os.path.exists(CSV_FILE):
        logger.info(f"CSV file not found at {CSV_FILE}")
        return
    
    try:
        # Create backup
        backup_file = CSV_FILE + ".backup"
        import shutil
        shutil.copy2(CSV_FILE, backup_file)
        logger.info(f"Created backup at {backup_file}")
        
        # Read existing data
        rows = []
        needs_migration = False
        row_count = 0
        
        with open(CSV_FILE, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
                logger.info(f"Current header: {header}")
                
                # Check if migration is needed
                if len(header) < 9 or header[2] != 'Last Name':
                    needs_migration = True
                    header = ['ID', 'Name', 'Last Name', 'Workplace', 'Email', 'Phone', 'Date Added', 'Photo Path', 'Face Encoding']
                    logger.info(f"Migration needed. New header: {header}")
                
                rows.append(header)
                
                for row in reader:
                    row_count += 1
                    if len(row) < 9:
                        # Old format: [ID, Name, Date Added, Photo Path, Face Encoding]
                        if len(row) >= 5:
                            # Update photo path if it points to old location
                            photo_path = row[3]
                            if photo_path.startswith("../uploads/"):
                                photo_path = photo_path.replace("../uploads/", "uploads/")
                                logger.info(f"Updated photo path from {row[3]} to {photo_path}")
                            
                            migrated_row = [
                                row[0],  # ID
                                row[1],  # Name
                                "",      # Last Name (empty)
                                "",      # Workplace (empty)
                                "",      # Email (empty)
                                "",      # Phone (empty)
                                row[2],  # Date Added
                                photo_path,  # Updated Photo Path
                                row[4]   # Face Encoding
                            ]
                            rows.append(migrated_row)
                            logger.info(f"Migrated row {row[0]}: {row[1]}")
                        else:
                            logger.warning(f"Skipping incomplete row: {row}")
                    else:
                        # Even in new format, check and update photo paths
                        if len(row) > 7 and row[7].startswith("../uploads/"):
                            row[7] = row[7].replace("../uploads/", "uploads/")
                            logger.info(f"Updated photo path in new format row")
                        rows.append(row)
                        
            except StopIteration:
                logger.info("CSV file is empty (no data rows)")
        
        logger.info(f"Read {row_count} data rows from CSV")
        
        # Write migrated data if needed
        if needs_migration and len(rows) > 1:
            with open(CSV_FILE, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerows(rows)
            logger.info(f"CSV file migrated successfully. Wrote {len(rows)-1} data rows")
            
            # Verify the migration
            with open(CSV_FILE, 'r', encoding='utf-8') as f:
                verify_rows = list(csv.reader(f))
                logger.info(f"Verification: CSV now has {len(verify_rows)} rows (including header)")
        elif needs_migration:
            logger.warning("Migration needed but no data rows found")
    
    except Exception as e:
        logger.error(f"Error during CSV migration: {e}")
        import traceback
        logger.error(traceback.format_exc())

def init_csv():
    """Initialize CSV file with extended fields"""
    if not os.path.exists(CSV_FILE):
        logger.info(f"Creating new CSV file at {CSV_FILE}")
        with open(CSV_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['ID', 'Name', 'Last Name', 'Workplace', 'Email', 'Phone', 'Date Added', 'Photo Path', 'Face Encoding'])
        logger.info("CSV file created successfully")
    else:
        logger.info(f"CSV file already exists at {CSV_FILE}")

def load_data():
    """
    Load data from CSV file into memory.
    
    Loads:
    - Face encodings into known_encodings list
    - Names into known_names list
    - IDs into known_ids list
    - Full person profiles into known_persons_data dictionary
    
    Supports both old format (5 columns) and new format (9 columns) CSV files.
    """
    global known_encodings, known_names, known_ids, known_persons_data
    known_encodings = []
    known_names = []
    known_ids = []
    known_persons_data = {}
    
    try:
        if os.path.exists(CSV_FILE):
            logger.info(f"Loading data from {CSV_FILE}")
            with open(CSV_FILE, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader)  # Skip header
                logger.info(f"CSV header: {header}")
                
                row_count = 0
                for row in reader:
                    row_count += 1
                    # Handle both old and new formats
                    if len(row) >= 9:  # New format
                        person_id = int(row[0])  # Ensure it's Python int, not numpy
                        name = row[1]
                        encoding_str = row[8]
                        
                        # Store full person data
                        known_persons_data[person_id] = {
                            "id": person_id,
                            "name": name,
                            "last_name": row[2] if row[2] else None,
                            "workplace": row[3] if row[3] else None,
                            "email": row[4] if row[4] else None,
                            "phone": row[5] if row[5] else None,
                            "added_date": row[6],
                            "photo_path": row[7]
                        }
                    elif len(row) >= 5:  # Old format
                        person_id = int(row[0])  # Ensure it's Python int, not numpy
                        name = row[1]
                        encoding_str = row[4]
                        
                        # Store minimal person data for old format
                        known_persons_data[person_id] = {
                            "id": person_id,
                            "name": name,
                            "last_name": None,
                            "workplace": None,
                            "email": None,
                            "phone": None,
                            "added_date": row[2],
                            "photo_path": row[3]
                        }
                    else:
                        logger.warning(f"Skipping invalid row: {row}")
                        continue
                    
                    if encoding_str:
                        try:
                            encoding = np.array(json.loads(encoding_str))
                            known_encodings.append(encoding)
                            known_names.append(name)
                            known_ids.append(person_id)  # Already converted to int above
                        except json.JSONDecodeError:
                            logger.error(f"Failed to decode encoding for {name} (ID: {person_id})")
                
                logger.info(f"Loaded {len(known_names)} valid records from {row_count} total rows")
        else:
            logger.info(f"CSV file not found at {CSV_FILE}, will be created on first add")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        import traceback
        logger.error(traceback.format_exc())

# Helper functions
def convert_numpy_types(obj):
    """
    Convert numpy types to Python native types for JSON serialization.
    
    Args:
        obj: Object that may contain numpy types
        
    Returns:
        Object with numpy types converted to Python types
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def save_uploaded_file(upload_file: UploadFile) -> str:
    """Save uploaded file"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{upload_file.filename}"
        file_path = os.path.join(UPLOAD_DIR, filename)
        
        # Reset file position to beginning
        upload_file.file.seek(0)
        content = upload_file.file.read()
        
        with open(file_path, "wb") as f:
            f.write(content)
        
        logger.info(f"File saved: {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Error saving file: {e}")
        raise

def image_to_base64(image_path: str) -> str:
    """Convert image to base64"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def base64_to_image(base64_string: str) -> np.ndarray:
    """Convert base64 to image"""
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))
    return np.array(img)

# Обновленный декоратор для проверки состояния
def check_api_state(state_type: ProcessingState):
    """Декоратор для проверки и управления состоянием API"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Проверка текущего состояния
            if api_state.current_state != ProcessingState.IDLE:
                return JSONResponse(
                    status_code=503,
                    content={
                        "success": False,
                        "message": f"System is busy: {api_state.current_state.value}. Please wait and try again.",
                        "data": {
                            "current_state": api_state.current_state.value,
                            "current_task": api_state.current_task_info,
                            "retry_after": 5  # Секунды до повторной попытки
                        }
                    }
                )
            
            # Захват блокировки и установка состояния
            async with api_state.lock:
                api_state.current_state = state_type
                api_state.current_task_info = {
                    "type": func.__name__,
                    "started_at": datetime.now().isoformat(),
                    "endpoint": func.__name__
                }
                
                try:
                    # Выполнение функции
                    result = await func(*args, **kwargs)
                    return result
                    
                except Exception as e:
                    # Логирование ошибки
                    logger.error(f"Error in {func.__name__}: {str(e)}")
                    logger.error(traceback.format_exc())
                    
                    # Возврат ошибки
                    return JSONResponse(
                        status_code=500,
                        content={
                            "success": False,
                            "message": f"Internal server error: {str(e)}",
                            "data": None
                        }
                    )
                finally:
                    # Всегда сбрасываем состояние
                    api_state.current_state = ProcessingState.IDLE
                    api_state.current_task_info = None
        
        return wrapper
    return decorator

# Load data on startup
logger.info(f"Starting Face Recognition API...")
logger.info(f"Working directory: {os.getcwd()}")
logger.info(f"CSV file path: {os.path.abspath(CSV_FILE)}")
logger.info(f"Upload directory: {os.path.abspath(UPLOAD_DIR)}")

check_and_copy_old_data()
init_csv()
migrate_csv_if_needed()
load_data()

# Debug: Check CSV content after startup
if os.path.exists(CSV_FILE):
    with open(CSV_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        logger.info(f"CSV file has {len(lines)} lines (including header)")
        if len(lines) > 0:
            logger.info(f"First line (header): {lines[0].strip()}")
        if len(lines) > 1:
            logger.info(f"Sample data row: {lines[1].strip()}")

# API Endpoints
@app.get("/")
async def root():
    """Check API health and get endpoints info"""
    return {
        "status": "active",
        "service": "Face Recognition API",
        "version": "1.1.0",  # Updated version
        "current_state": api_state.current_state.value,
        "features": [
            "Full person data support (name, last name, workplace, email, phone)",
            "Face recognition in photos and videos",
            "Unknown face tracking in videos",
            "Extended person profiles",
            "Concurrent request protection"
        ],
        "endpoints": {
            "POST /api/person/add": "Add person with extended profile",
            "POST /api/person/recognize": "Recognize faces in photo (returns full person data)",
            "POST /api/person/recognize-base64": "Recognize faces in base64 image (returns full person data)",
            "POST /api/video/recognize": "Recognize faces in video (returns full person data)",
            "GET /api/person/list": "List all people with full profiles",
            "GET /api/person/{id}": "Get person info by ID",
            "DELETE /api/person/{id}": "Delete person from database",
            "GET /api/stats": "System statistics",
            "GET /api/status": "Get processing status"
        }
    }

@app.get("/api/status",
         response_model=ApiResponse,
         tags=["System"],
         summary="Get current processing status")
async def get_status():
    """Get current API processing status"""
    return {
        "success": True,
        "message": "Status retrieved",
        "data": {
            "state": api_state.current_state.value,
            "is_busy": api_state.current_state != ProcessingState.IDLE,
            "current_task": api_state.current_task_info
        }
    }

@app.get("/api/debug/files",
         response_model=ApiResponse,
         tags=["System"],
         summary="Debug: Check file locations")
async def debug_files():
    """Debug endpoint to check file locations and contents"""
    try:
        debug_info = {
            "working_directory": os.getcwd(),
            "csv_file_path": os.path.abspath(CSV_FILE),
            "csv_exists": os.path.exists(CSV_FILE),
            "upload_dir_path": os.path.abspath(UPLOAD_DIR),
            "upload_dir_exists": os.path.exists(UPLOAD_DIR)
        }
        
        # Check CSV content
        if os.path.exists(CSV_FILE):
            with open(CSV_FILE, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                debug_info["csv_lines"] = len(lines)
                debug_info["csv_header"] = lines[0].strip() if lines else "No header"
                debug_info["csv_sample"] = lines[1].strip() if len(lines) > 1 else "No data"
        
        # Check for backup file
        backup_file = CSV_FILE + ".backup"
        if os.path.exists(backup_file):
            debug_info["backup_exists"] = True
            with open(backup_file, 'r', encoding='utf-8') as f:
                backup_lines = f.readlines()
                debug_info["backup_lines"] = len(backup_lines)
        
        # Check upload directory
        if os.path.exists(UPLOAD_DIR):
            files = os.listdir(UPLOAD_DIR)
            debug_info["upload_files_count"] = len(files)
            debug_info["upload_files"] = files[:5]  # First 5 files
        
        # Check for old location
        old_csv = "../persons_data.csv"
        if os.path.exists(old_csv):
            debug_info["old_csv_exists"] = True
            with open(old_csv, 'r', encoding='utf-8') as f:
                old_lines = f.readlines()
                debug_info["old_csv_lines"] = len(old_lines)
        
        # Check memory state
        debug_info["loaded_in_memory"] = {
            "known_names": len(known_names),
            "known_encodings": len(known_encodings),
            "known_ids": len(known_ids),
            "sample_names": known_names[:3] if known_names else [],
            "sample_ids": known_ids[:3] if known_ids else []
        }
        
        return {
            "success": True,
            "message": "Debug information retrieved",
            "data": debug_info
        }
    except Exception as e:
        logger.error(f"Error during debug files check: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"Debug error: {str(e)}",
                "data": None
            }
        )
    
@app.post("/api/migrate/old-data",
          response_model=ApiResponse,
          tags=["System"],
          summary="Manually migrate data from old location")
async def migrate_old_data():
    """Manually copy data from old location (../persons_data.csv) to new location"""
    try:
        old_csv = "../persons_data.csv"
        old_uploads = "../uploads"
        
        if not os.path.exists(old_csv):
            return {
                "success": False,
                "message": "No old CSV file found at ../persons_data.csv",
                "data": None
            }
        
        # Backup current data if exists
        if os.path.exists(CSV_FILE):
            backup_name = f"persons_data_before_migration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            import shutil
            shutil.copy2(CSV_FILE, backup_name)
            logger.info(f"Backed up current CSV to {backup_name}")
        
        # Copy CSV file
        import shutil
        shutil.copy2(old_csv, CSV_FILE)
        logger.info(f"Copied {old_csv} to {CSV_FILE}")
        
        # Copy uploads if they exist
        copied_files = 0
        if os.path.exists(old_uploads):
            os.makedirs(UPLOAD_DIR, exist_ok=True)
            for file in os.listdir(old_uploads):
                old_file = os.path.join(old_uploads, file)
                new_file = os.path.join(UPLOAD_DIR, file)
                if os.path.isfile(old_file):
                    shutil.copy2(old_file, new_file)
                    copied_files += 1
        
        # Now migrate the CSV to new format and update paths
        migrate_csv_if_needed()
        
        # Reload data
        load_data()
        
        return {
            "success": True,
            "message": f"Successfully migrated data from old location. Copied {copied_files} image files.",
            "data": {
                "csv_copied": True,
                "images_copied": copied_files,
                "persons_loaded": len(known_names)
            }
        }
        
    except Exception as e:
        logger.error(f"Error during manual migration: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"Migration error: {str(e)}",
                "data": None
            }
        )
    

@app.post("/api/person/add",
          response_model=ApiResponse,
          tags=["Database Management"],
          summary="Add new person to database with extended profile")
@check_api_state(ProcessingState.PROCESSING_IMAGE)
async def add_person(
    name: str = Form(..., description="Name of the person to add."),
    photo: UploadFile = File(..., description="Photo of the person (face should be clearly visible)."),
    last_name: Optional[str] = Form(None, description="Last name (optional)"),
    workplace: Optional[str] = Form(None, description="Workplace (optional)"),
    email: Optional[str] = Form(None, description="Email address (optional)"),
    phone: Optional[str] = Form(None, description="Phone number (optional)")
):
    """
    Add new person to database with extended profile information
    
    Parameters:
    - name: Person's name (required)
    - photo: Photo file (JPG, PNG) (required)
    - last_name: Last name (optional)
    - workplace: Place of work (optional)
    - email: Email address (optional)
    - phone: Phone number (optional)
    """
    # Сохраняем файл
    file_path = save_uploaded_file(photo)
    
    # Загружаем изображение для обработки
    image = face_recognition.load_image_file(file_path)
    face_locations = face_recognition.face_locations(image)
    
    if len(face_locations) == 0:
        os.remove(file_path)  # Удаляем файл если лицо не найдено
        raise HTTPException(
            status_code=400,
            detail="No faces found in the photo"
        )
    
    if len(face_locations) > 1:
        logger.warning(f"Warning: found {len(face_locations)} faces, using the first one")
    
    # Получаем кодировку лица
    face_encoding = face_recognition.face_encodings(image, face_locations)[0]
    
    # Генерируем ID
    next_id = int(max(known_ids) + 1) if known_ids else 1  # Ensure Python int
    
    # Добавляем в память
    known_encodings.append(face_encoding)
    known_names.append(name)
    known_ids.append(next_id)
    
    # Сохраняем полные данные в память
    person_data = {
        "id": next_id,
        "name": name,
        "last_name": last_name,
        "workplace": workplace,
        "email": email,
        "phone": phone,
        "added_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "photo_path": file_path
    }
    known_persons_data[next_id] = person_data
    
    # Сохраняем в CSV с расширенными полями
    with open(CSV_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        encoding_str = json.dumps(face_encoding.tolist())
        writer.writerow([
            next_id,
            name,
            last_name or "",
            workplace or "",
            email or "",
            phone or "",
            person_data["added_date"],
            file_path,
            encoding_str
        ])
    
    # Формируем полное имя для логирования
    full_name = f"{name} {last_name}" if last_name else name
    logger.info(f"Added new person: {full_name} (ID: {next_id})")
    
    return {
        "success": True,
        "message": f"Person '{full_name}' successfully added",
        "data": {
            "id": next_id,
            "name": name,
            "last_name": last_name,
            "workplace": workplace,
            "email": email,
            "phone": phone,
            "faces_found": len(face_locations),
            "photo_path": file_path
        }
    }
 
@app.post("/api/person/recognize",
          response_model=ApiResponse,
          tags=["Recognition"],
          summary="Recognize faces in photo with full person data")
async def recognize_person(photo: UploadFile = File(..., description="Photo for recognition.")):
    """
    Recognizes faces in uploaded photo and returns full person information.
    
    Returns:
    - Processed image with bounding boxes in Base64 format
    - List of recognized faces with complete person data:
      - person_id: Database ID (0 for unknown faces)
      - name: First name
      - last_name: Last name (if available)
      - workplace: Workplace (if available)
      - email: Email address (if available)
      - phone: Phone number (if available)
      - added_date: Date when person was added
      - confidence: Recognition confidence score (0.0-1.0)
      - location: Face bounding box coordinates
      - encoding: 128-dimensional face feature vector
      
    Unknown faces will have person_id=0 and name="Unknown" with null values for other fields.
    """
    # Check if system is busy
    if api_state.current_state != ProcessingState.IDLE:
        return JSONResponse(
            status_code=503,
            content={
                "success": False,
                "message": f"System is busy: {api_state.current_state.value}. Please wait and try again.",
                "data": {
                    "current_state": api_state.current_state.value,
                    "current_task": api_state.current_task_info
                }
            }
        )
    
    async with api_state.lock:
        # Set processing state
        api_state.current_state = ProcessingState.PROCESSING_IMAGE
        api_state.current_task_info = {"type": "image_recognition", "started_at": datetime.now().isoformat()}
        
        try:
            # Read uploaded file into memory as byte array
            image_bytes = await photo.read()
            
            # Convert bytes to NumPy array, then to OpenCV format (BGR)
            np_array = np.frombuffer(image_bytes, np.uint8)
            image_bgr = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
            
            # face_recognition needs RGB format, so convert
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            # Find all faces and their encodings
            face_locations = face_recognition.face_locations(image_rgb)
            face_encodings = face_recognition.face_encodings(image_rgb, face_locations)
            logger.info(f"Found {len(face_locations)} faces in the image.")

            recognition_results = []
            
            # Process each found face
            for i, (top, right, bottom, left) in enumerate(face_locations):
                face_encoding = face_encodings[i]
                
                # Default values for unknown person
                person_data = {
                    "id": 0,
                    "name": "Unknown",
                    "last_name": None,
                    "workplace": None,
                    "email": None,
                    "phone": None,
                    "added_date": None,
                    "photo_path": None
                }
                confidence = 0.0
                color = (0, 0, 255)  # Red color for unknown face bounding box
                
                if known_encodings:
                    matches = face_recognition.compare_faces(known_encodings, face_encoding)
                    face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                    
                    if True in matches:
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            person_id = known_ids[best_match_index]
                            confidence = float(1 - face_distances[best_match_index])
                            
                            # Get full person data from memory
                            full_person_data = get_person_data_by_id(person_id)
                            if full_person_data:
                                person_data = full_person_data.copy()
                                color = (0, 255, 0)  # Green color for known face bounding box
                            else:
                                # Fallback if data not in memory (shouldn't happen)
                                person_data["id"] = person_id
                                person_data["name"] = known_names[best_match_index]
                                color = (0, 255, 0)

                # Draw on image (in BGR format)
                cv2.rectangle(image_bgr, (left, top), (right, bottom), color, 2)
                cv2.rectangle(image_bgr, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                
                # Display full name if last name exists
                display_name = person_data["name"]
                if person_data.get("last_name"):
                    display_name += f" {person_data['last_name']}"
                
                cv2.putText(image_bgr, display_name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

                # Collect result for this face with all person data
                # Convert numpy types to Python types for JSON serialization
                recognition_result = {
                    "person_id": int(person_data["id"]),
                    "name": person_data["name"],
                    "last_name": person_data.get("last_name"),
                    "workplace": person_data.get("workplace"),
                    "email": person_data.get("email"),
                    "phone": person_data.get("phone"),
                    "added_date": person_data.get("added_date"),
                    "confidence": round(float(confidence), 3),
                    "location": {
                        "top": int(top), 
                        "right": int(right), 
                        "bottom": int(bottom), 
                        "left": int(left)
                    },
                    "encoding": face_encoding.tolist()  # Add vector to response
                }
                recognition_results.append(recognition_result)

            # Encode processed image to Base64
            _, buffer = cv2.imencode('.jpg', image_bgr)
            processed_image_base64 = base64.b64encode(buffer).decode('utf-8')

            return {
                "success": True,
                "message": f"Recognition completed.",
                "data": {
                    "faces_count": len(recognition_results),
                    "faces": recognition_results,
                    "processed_image_base64": processed_image_base64
                }
            }

        except Exception as e:
            logger.error(f"Critical error in /api/person/recognize: {e}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={"success": False, "message": f"Internal server error: {str(e)}"}
            )
        finally:
            # Reset state
            api_state.current_state = ProcessingState.IDLE
            api_state.current_task_info = None

@app.get("/api/person/list",
         response_model=ApiResponse,
         tags=["Database Management"],
         summary="Get list of all people in database")
async def list_persons():
    """Get list of all people in database with extended information"""
    try:
        persons = []
        
        if os.path.exists(CSV_FILE):
            with open(CSV_FILE, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                
                for row in reader:
                    if len(row) >= 8:  # New format
                        person_data = {
                            "id": int(row[0]),
                            "name": row[1],
                            "last_name": row[2] if row[2] else None,
                            "workplace": row[3] if row[3] else None,
                            "email": row[4] if row[4] else None,
                            "phone": row[5] if row[5] else None,
                            "added_date": row[6],
                            "photo_path": row[7]
                        }
                        persons.append(person_data)
                    elif len(row) >= 5:  # Old format
                        person_data = {
                            "id": int(row[0]),
                            "name": row[1],
                            "last_name": None,
                            "workplace": None,
                            "email": None,
                            "phone": None,
                            "added_date": row[2],
                            "photo_path": row[3]
                        }
                        persons.append(person_data)
        
        return {
            "success": True,
            "message": f"Found {len(persons)} people",
            "data": {
                "count": len(persons),
                "persons": persons
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting list: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"Server error: {str(e)}",
                "data": None
            }
        )

@app.get("/api/person/{person_id}",
         response_model=ApiResponse,
         tags=["Database Management"],
         summary="Get specific person info")
async def get_person(person_id: int):
    """Get specific person information with all extended fields"""
    try:
        with open(CSV_FILE, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            
            for row in reader:
                if len(row) >= 8 and int(row[0]) == person_id:
                    # Handle both old and new format
                    return JSONResponse(content={
                        "success": True,
                        "message": "Person found",
                        "data": {
                            "id": int(row[0]),
                            "name": row[1],
                            "last_name": row[2] if len(row) > 2 and row[2] else None,
                            "workplace": row[3] if len(row) > 3 and row[3] else None,
                            "email": row[4] if len(row) > 4 and row[4] else None,
                            "phone": row[5] if len(row) > 5 and row[5] else None,
                            "added_date": row[6] if len(row) > 6 else row[2],  # Handle old format
                            "photo_path": row[7] if len(row) > 7 else row[3],  # Handle old format
                            "photo_base64": image_to_base64(row[7] if len(row) > 7 else row[3]) if os.path.exists(row[7] if len(row) > 7 else row[3]) else None
                        }
                    })
        
        raise HTTPException(status_code=404, detail="Person not found")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/person/{person_id}",
            response_model=ApiResponse,
            tags=["Database Management"],
            summary="Delete person from database")
async def delete_person(person_id: int):
    """Delete person from database"""
    try:
        rows = []
        deleted = False
        deleted_name = ""
        photo_path = ""
        
        with open(CSV_FILE, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows.append(next(reader))  # Header
            
            for row in reader:
                if len(row) >= 5 and int(row[0]) != person_id:
                    rows.append(row)
                elif len(row) >= 5 and int(row[0]) == person_id:
                    deleted = True
                    deleted_name = row[1]
                    
                    # Handle different formats
                    if len(row) >= 9:  # New format
                        if row[2]:  # Add last name if exists
                            deleted_name += f" {row[2]}"
                        photo_path = row[7]
                    else:  # Old format
                        photo_path = row[3]
                    
                    # Delete photo file if exists
                    if photo_path and os.path.exists(photo_path):
                        os.remove(photo_path)
        
        if not deleted:
            return JSONResponse(
                status_code=404,
                content={
                    "success": False,
                    "message": "Person not found",
                    "data": None
                }
            )
        
        # Rewrite file
        with open(CSV_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        
        # Reload data
        load_data()
        
        return {
            "success": True,
            "message": f"Person '{deleted_name}' deleted",
            "data": {
                "id": person_id,
                "name": deleted_name
            }
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"Server error: {str(e)}",
                "data": None
            }
        )

@app.get("/api/stats",
         response_model=ApiResponse,
         tags=["Statistics"],
         summary="Get database statistics")
async def get_stats():
    """Get system statistics"""
    try:
        total_persons = len(known_names)
        
        # Last addition with full name
        last_added = None
        if os.path.exists(CSV_FILE):
            with open(CSV_FILE, 'r', encoding='utf-8') as f:
                reader = list(csv.reader(f))
                if len(reader) > 1:
                    last_row = reader[-1]
                    if len(last_row) >= 9:  # New format
                        full_name = last_row[1]
                        if last_row[2]:  # Add last name if exists
                            full_name += f" {last_row[2]}"
                        last_added = {
                            "name": full_name,
                            "date": last_row[6]
                        }
                    elif len(last_row) >= 5:  # Old format
                        last_added = {
                            "name": last_row[1],
                            "date": last_row[2]
                        }
        
        return {
            "success": True,
            "message": "Statistics retrieved",
            "data": {
                "total_persons": total_persons,
                "total_photos": total_persons,
                "last_added": last_added,
                "storage_path": UPLOAD_DIR,
                "api_state": api_state.current_state.value
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"Server error: {str(e)}",
                "data": None
            }
        )

@app.post("/api/person/recognize-base64",
          response_model=ApiResponse,
          tags=["Recognition"],
          summary="Recognize faces in base64 image with full person data")
async def recognize_person_base64(image_base64: str = Form(...)):
    """
    Recognize faces in base64-encoded image and return full person information.
    
    Parameters:
    - image_base64: Image in base64 format
    
    Returns:
    - List of recognized faces with complete person data:
      - face_id: Sequential face number in this image (1, 2, 3...)
      - person_id: Database ID (0 for unknown faces)
      - name: First name
      - last_name: Last name (if available)
      - workplace: Workplace (if available)
      - email: Email address (if available)
      - phone: Phone number (if available)
      - added_date: Date when person was added
      - confidence: Recognition confidence score (0.0-1.0)
      - location: Face bounding box coordinates
      
    Note: This endpoint does not return processed image or face encodings.
    """
    # Check if system is busy
    if api_state.current_state != ProcessingState.IDLE:
        return JSONResponse(
            status_code=503,
            content={
                "success": False,
                "message": f"System is busy: {api_state.current_state.value}. Please wait and try again.",
                "data": {
                    "current_state": api_state.current_state.value,
                    "current_task": api_state.current_task_info
                }
            }
        )
    
    async with api_state.lock:
        api_state.current_state = ProcessingState.PROCESSING_IMAGE
        api_state.current_task_info = {"type": "base64_recognition", "started_at": datetime.now().isoformat()}
        
        try:
            # Convert base64 to image
            image = base64_to_image(image_base64)
            
            # Find faces
            face_locations = face_recognition.face_locations(image)
            
            if len(face_locations) == 0:
                return JSONResponse(content={
                    "success": True,
                    "message": "No faces found in photo",
                    "data": {
                        "faces_count": 0,
                        "faces": []
                    }
                })
            
            # Get encodings
            face_encodings = face_recognition.face_encodings(image, face_locations)
            
            results = []
            
            for i, (face_encoding, face_location) in enumerate(zip(face_encodings, face_locations)):
                # Default values for unknown person
                person_data = {
                    "id": 0,
                    "name": "Unknown",
                    "last_name": None,
                    "workplace": None,
                    "email": None,
                    "phone": None,
                    "added_date": None,
                    "photo_path": None
                }
                confidence = 0.0
                
                if len(known_encodings) > 0:
                    matches = face_recognition.compare_faces(known_encodings, face_encoding)
                    face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                    
                    if True in matches:
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            person_id = known_ids[best_match_index]
                            confidence = float(1 - face_distances[best_match_index])
                            
                            # Get full person data from memory
                            full_person_data = get_person_data_by_id(person_id)
                            if full_person_data:
                                person_data = full_person_data.copy()
                
                top, right, bottom, left = face_location
                
                results.append({
                    "face_id": i + 1,
                    "person_id": int(person_data["id"]),
                    "name": person_data["name"],
                    "last_name": person_data.get("last_name"),
                    "workplace": person_data.get("workplace"),
                    "email": person_data.get("email"),
                    "phone": person_data.get("phone"),
                    "added_date": person_data.get("added_date"),
                    "confidence": round(float(confidence), 3),
                    "location": {
                        "top": int(top),
                        "right": int(right),
                        "bottom": int(bottom),
                        "left": int(left)
                    }
                })
            
            return JSONResponse(content={
                "success": True,
                "message": f"Found {len(results)} faces",
                "data": {
                    "faces_count": len(results),
                    "faces": results
                }
            })
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            api_state.current_state = ProcessingState.IDLE
            api_state.current_task_info = None


@app.post("/api/video/recognize",
          response_model=ApiResponse,
          tags=["Recognition"],
          summary="Recognize faces in video with full person data")
@check_api_state(ProcessingState.PROCESSING_VIDEO)
async def recognize_video(
    video: UploadFile = File(..., description="Video file for recognition (MP4, AVI, MOV)"),
    frame_interval: int = Form(15, description="Process every Nth frame (default 15)")
):
    """
    Recognizes faces in video file and returns complete person information for each detection.
    
    Parameters:
    - video: Video file (MP4, AVI, MOV formats supported)
    - frame_interval: Process every Nth frame (default: 15)
    
    Returns:
    - summary: Video processing statistics including:
      - total_frames: Total number of frames in video
      - processed_frames: Number of frames actually processed
      - fps: Video frame rate
      - duration_seconds: Video duration
      - unique_persons: Number of unique faces detected
      - person_appearances: Dictionary with full data for each person:
        - person_id: Database ID (negative for temporary unknown IDs)
        - name, last_name, workplace, email, phone: Person details
        - first_seen: First appearance timestamp
        - last_seen: Last appearance timestamp
        - total_appearances: Number of detections
    
    - key_frames: Up to 10 annotated frames as Base64 images
    
    - detections: List of all face detections (limited to first 100) with:
      - Full person data (all fields from database)
      - confidence: Recognition confidence
      - timestamp: Time in video when detected
      - frame_number: Frame where face was found
      - location: Face bounding box
    
    Notes:
    - Unknown faces are assigned temporary negative IDs (e.g., -1, -2)
    - Unknown faces are labeled as "Unknown #1", "Unknown #2", etc.
    - Processing large videos may take several minutes
    """
    # Сохраняем видео файл
    video_path = save_uploaded_file(video)
    logger.info(f"Processing video: {video_path}")
    
    # Открываем видео
    video_capture = cv2.VideoCapture(video_path)
    
    if not video_capture.isOpened():
        os.remove(video_path)
        raise HTTPException(
            status_code=400,
            detail="Failed to open video file"
        )
    
    # Получаем информацию о видео
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Временное хранилище для неизвестных лиц
    temp_unknown_encodings = []
    temp_unknown_id_counter = 1
    person_appearances = {}
    
    # Результаты распознавания
    recognition_results = []
    key_frames = []
    processed_frames = 0
    frame_count = 0
    
    logger.info(f"Video: {total_frames} frames, {fps} FPS, processing every {frame_interval} frame")
    
    try:
        while video_capture.isOpened():
            ret, frame = video_capture.read()
            
            if not ret:
                break
            
            # Обрабатываем каждый N-й кадр
            if frame_count % frame_interval == 0:
                # Конвертируем в RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Находим лица
                face_locations = face_recognition.face_locations(rgb_frame)
                
                if face_locations:
                    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                    
                    # Копируем кадр для рисования
                    annotated_frame = frame.copy()
                    frame_results = []
                    
                    for face_encoding, face_location in zip(face_encodings, face_locations):
                        # Default values for unknown person
                        person_data = {
                            "id": 0,
                            "name": "Unknown",
                            "last_name": None,
                            "workplace": None,
                            "email": None,
                            "phone": None,
                            "added_date": None,
                            "photo_path": None
                        }
                        confidence = 0.0
                        color = (0, 0, 255)  # Красный для неизвестного
                        unknown_id = None
                        
                        if known_encodings:
                            matches = face_recognition.compare_faces(known_encodings, face_encoding)
                            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                            
                            if True in matches:
                                best_match_index = np.argmin(face_distances)
                                if matches[best_match_index]:
                                    person_id = known_ids[best_match_index]
                                    confidence = float(1 - face_distances[best_match_index])
                                    color = (0, 255, 0)  # Зеленый для известного
                                    
                                    # Get full person data from memory
                                    full_person_data = get_person_data_by_id(person_id)
                                    if full_person_data:
                                        person_data = full_person_data.copy()
                        
                        if person_data["name"] == "Unknown":
                            # Проверяем среди временных неизвестных
                            if temp_unknown_encodings:
                                unknown_distances = face_recognition.face_distance(temp_unknown_encodings, face_encoding)
                                best_match_index = np.argmin(unknown_distances)
                                if unknown_distances[best_match_index] < 0.6:
                                    unknown_id = best_match_index + 1
                                    person_data["name"] = f"Unknown #{unknown_id}"
                                else:
                                    temp_unknown_encodings.append(face_encoding)
                                    unknown_id = len(temp_unknown_encodings)
                                    person_data["name"] = f"Unknown #{unknown_id}"
                            else:
                                temp_unknown_encodings.append(face_encoding)
                                unknown_id = 1
                                person_data["name"] = f"Unknown #1"
                            
                            # Set temporary negative ID for unknowns
                            person_data["id"] = -int(unknown_id)  # Ensure Python int
                        
                        # Рисуем рамку и имя
                        top, right, bottom, left = face_location
                        cv2.rectangle(annotated_frame, (left, top), (right, bottom), color, 2)
                        cv2.rectangle(annotated_frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                        
                        # Display full name if last name exists
                        display_name = person_data["name"]
                        if person_data.get("last_name"):
                            display_name += f" {person_data['last_name']}"
                        
                        cv2.putText(annotated_frame, display_name, (left + 6, bottom - 6),
                                  cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
                        
                        # Сохраняем результат с полными данными
                        timestamp = frame_count / fps
                        frame_result = {
                            "person_id": int(person_data["id"]),
                            "name": person_data["name"],
                            "last_name": person_data.get("last_name"),
                            "workplace": person_data.get("workplace"),
                            "email": person_data.get("email"),
                            "phone": person_data.get("phone"),
                            "added_date": person_data.get("added_date"),
                            "confidence": round(float(confidence), 3),
                            "timestamp": round(float(timestamp), 2),
                            "frame_number": int(frame_count),
                            "location": {
                                "top": int(top),
                                "right": int(right),
                                "bottom": int(bottom),
                                "left": int(left)
                            }
                        }
                        frame_results.append(frame_result)
                        
                        # Отслеживаем уникальные появления
                        person_key = person_data["name"]
                        if person_data.get("last_name"):
                            person_key += f" {person_data['last_name']}"
                            
                        if person_key not in person_appearances:
                            person_appearances[person_key] = {
                                "person_id": int(person_data["id"]),
                                "name": person_data["name"],
                                "last_name": person_data.get("last_name"),
                                "workplace": person_data.get("workplace"),
                                "email": person_data.get("email"),
                                "phone": person_data.get("phone"),
                                "first_seen": float(timestamp),
                                "last_seen": float(timestamp),
                                "total_appearances": 0
                            }
                        person_appearances[person_key]["last_seen"] = float(timestamp)
                        person_appearances[person_key]["total_appearances"] += 1
                    
                    # Сохраняем ключевой кадр
                    if processed_frames % 5 == 0 and len(key_frames) < 10:
                        _, buffer = cv2.imencode('.jpg', annotated_frame)
                        key_frame_base64 = base64.b64encode(buffer).decode('utf-8')
                        key_frames.append({
                            "frame_number": int(frame_count),
                            "timestamp": round(float(frame_count / fps), 2),
                            "faces_count": len(face_locations),
                            "image_base64": key_frame_base64
                        })
                    
                    recognition_results.extend(frame_results)
                    processed_frames += 1
                
                # Логируем прогресс
                if processed_frames % 30 == 0:
                    logger.info(f"Processed {processed_frames} frames out of {frame_count // frame_interval}")
            
            frame_count += 1
    
    finally:
        video_capture.release()
        
        # Удаляем временный видео файл
        if os.path.exists(video_path):
            os.remove(video_path)
    
    logger.info(f"Processing completed. Found {len(person_appearances)} unique faces")
    
    # Формируем итоговую статистику с полными данными
    summary = {
        "total_frames": int(total_frames),
        "processed_frames": int(frame_count // frame_interval) if frame_interval > 0 else 0,
        "fps": round(float(fps), 2),
        "duration_seconds": round(float(total_frames / fps), 2) if fps > 0 else 0,
        "unique_persons": len(person_appearances),
        "person_appearances": {
            name: {
                "person_id": int(data["person_id"]),
                "name": data["name"],
                "last_name": data.get("last_name"),
                "workplace": data.get("workplace"),
                "email": data.get("email"),
                "phone": data.get("phone"),
                "first_seen": round(float(data["first_seen"]), 2),
                "last_seen": round(float(data["last_seen"]), 2),
                "total_appearances": int(data["total_appearances"])
            }
            for name, data in person_appearances.items()
        }
    }
    
    return {
        "success": True,
        "message": f"Video processed. Found {len(person_appearances)} unique faces",
        "data": {
            "summary": summary,
            "key_frames": key_frames,
            "total_detections": len(recognition_results),
            "detections": recognition_results[:100]  # Ограничиваем размер ответа
        }
    }

# Run server
if __name__ == "__main__":
    print("🚀 Starting Face Recognition API...")
    
    uvicorn.run(
        "main:app",  # Application path as string
        host="0.0.0.0",
        port=8000,
        reload=False,
        reload_dirs=["src"] # Directories to watch for changes
    )