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
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import face_recognition


# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Pydantic модели для валидации и документации ---
# Эти модели будут отображаться в Swagger (/docs)

class FaceLocation(BaseModel):
    top: int
    right: int
    bottom: int
    left: int

class FaceData(BaseModel):
    name: str
    location: FaceLocation
    encoding: List[float] = Field(..., description="Вектор признаков лица из 128 чисел.")

class RecognitionResponseData(BaseModel):
    faces_count: int
    faces: List[FaceData]
    processed_image_base64: str = Field(..., description="Изображение с нарисованными рамками в формате Base64.")

class PersonInDB(BaseModel):
    id: int
    name: str
    added_date: str

class AddPersonResponseData(BaseModel):
    id: int
    name: str
    faces_found: int

class StatsData(BaseModel):
    total_persons: int
    last_added: Optional[dict] = None

class ApiResponse(BaseModel):
    success: bool
    message: str
    data: Optional[dict] = None


# Создаем приложение FastAPI
app = FastAPI(
    title="Face Recognition API",
    description="API для распознавания лиц",
    version="1.0.0"
)

# Настройка CORS для работы с C# клиентом
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене укажите конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware для логирования запросов
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Request: {request.method} {request.url.path}")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response

# Глобальные переменные
CSV_FILE = "persons_data.csv"
UPLOAD_DIR = "uploads"
known_encodings = []
known_names = []
known_ids = []

# Создаем необходимые директории
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Модели данных
class PersonResponse(BaseModel):
    id: int
    name: str
    added_date: str
    photo_path: str

class RecognitionResult(BaseModel):
    face_id: int
    name: str
    confidence: float
    location: dict

class ApiResponse(BaseModel):
    success: bool
    message: str
    data: Optional[dict] = None

# Инициализация
def init_csv():
    """Инициализация CSV файла"""
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['ID', 'Имя', 'Дата добавления', 'Путь к фото', 'Кодировка лица'])

def load_data():
    """Загрузка данных из CSV"""
    global known_encodings, known_names, known_ids
    known_encodings = []
    known_names = []
    known_ids = []
    
    try:
        if os.path.exists(CSV_FILE):
            with open(CSV_FILE, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)  # Пропускаем заголовок
                
                for row in reader:
                    if len(row) >= 5:
                        person_id = int(row[0])
                        name = row[1]
                        encoding_str = row[4]
                        
                        if encoding_str:
                            encoding = np.array(json.loads(encoding_str))
                            known_encodings.append(encoding)
                            known_names.append(name)
                            known_ids.append(person_id)
            logger.info(f"Загружено {len(known_names)} записей из базы данных")
        else:
            logger.info("База данных пуста, создан новый файл")
    except Exception as e:
        logger.error(f"Ошибка при загрузке данных: {e}")

# Вспомогательные функции
def save_uploaded_file(upload_file: UploadFile) -> str:
    """Сохранение загруженного файла"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{upload_file.filename}"
        file_path = os.path.join(UPLOAD_DIR, filename)
        
        content = upload_file.file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        
        logger.info(f"Файл сохранен: {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Ошибка при сохранении файла: {e}")
        raise

def image_to_base64(image_path: str) -> str:
    """Конвертация изображения в base64"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def base64_to_image(base64_string: str) -> np.ndarray:
    """Конвертация base64 в изображение"""
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))
    return np.array(img)

# Загружаем данные при старте
init_csv()
load_data()

# API Endpoints
@app.get("/")
async def root():
    """Проверка работоспособности API"""
    return {
        "status": "active",
        "service": "Face Recognition API",
        "version": "1.0.0",
        "endpoints": {
            "POST /api/person/add": "Добавить человека",
            "POST /api/person/recognize": "Распознать лицо",
            "GET /api/person/list": "Список всех людей",
            "GET /api/person/{id}": "Получить информацию о человеке",
            "DELETE /api/person/{id}": "Удалить человека",
            "GET /api/stats": "Статистика системы"
        }
    }

@app.post("/api/person/add",
          response_model=ApiResponse,
          tags=["Управление базой"],
          summary="Добавить нового человека в базу")
async def add_person(
    name: str = Form(..., description="Имя добавляемого человека."),
    photo: UploadFile = File(..., description="Фотография человека (лицо должно быть хорошо видно).")
):
    """
    Добавить нового человека в базу данных
    
    Parameters:
    - name: Имя человека
    - photo: Фотография (JPG, PNG)
    """
    try:
        # Сохраняем файл
        file_path = save_uploaded_file(photo)
        
        # Загружаем изображение для обработки
        image = face_recognition.load_image_file(file_path)
        face_locations = face_recognition.face_locations(image)
        
        if len(face_locations) == 0:
            os.remove(file_path)  # Удаляем файл если лицо не найдено
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "message": "На фото не найдено лиц",
                    "data": None
                }
            )
        
        if len(face_locations) > 1:
            # Предупреждение, но продолжаем с первым лицом
            print(f"Внимание: найдено {len(face_locations)} лиц, используется первое")
        
        # Получаем кодировку лица
        face_encoding = face_recognition.face_encodings(image, face_locations)[0]
        
        # Генерируем ID
        person_id = len(known_ids) + 1
        
        # Добавляем в память
        known_encodings.append(face_encoding)
        known_names.append(name)
        known_ids.append(person_id)
        
        # Сохраняем в CSV
        with open(CSV_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            encoding_str = json.dumps(face_encoding.tolist())
            writer.writerow([
                person_id,
                name,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                file_path,
                encoding_str
            ])
        
        logger.info(f"Добавлен новый человек: {name} (ID: {person_id})")
        
        return {
            "success": True,
            "message": f"Человек '{name}' успешно добавлен",
            "data": {
                "id": person_id,
                "name": name,
                "faces_found": len(face_locations),
                "photo_path": file_path
            }
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"Ошибка сервера: {str(e)}",
                "data": None
            }
        )
 
@app.post("/api/person/recognize",
          response_model=ApiResponse,
          tags=["Распознавание"],
          summary="Распознать лица на фотографии")
async def recognize_person(photo: UploadFile = File(..., description="Фотография для распознавания.")):
    """
    Распознает лица, рисует на фото рамки и имена,
    и возвращает обработанное изображение и вектор признаков лица в формате Base64.
    """
    try:
        # Читаем загруженный файл в память как массив байт
        image_bytes = await photo.read()
        
        # Конвертируем байты в массив NumPy, а затем в формат OpenCV (BGR)
        np_array = np.frombuffer(image_bytes, np.uint8)
        image_bgr = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        
        # Для face_recognition нужен формат RGB, поэтому конвертируем
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Находим все лица и их кодировки
        face_locations = face_recognition.face_locations(image_rgb)
        face_encodings = face_recognition.face_encodings(image_rgb, face_locations)
        logger.info(f"Найдено {len(face_locations)} лиц на изображении.")

        recognition_results = []
        
        # Проходим по каждому найденному лицу
        for i, (top, right, bottom, left) in enumerate(face_locations):
            face_encoding = face_encodings[i]
            
            name = "Неизвестный"
            color = (0, 0, 255) # Красный цвет для рамки неизвестного
            
            if known_encodings:
                matches = face_recognition.compare_faces(known_encodings, face_encoding)
                face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                
                if True in matches:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_names[best_match_index]
                        color = (0, 255, 0) # Зеленый цвет для рамки известного

            # --- Рисование на изображении (в формате BGR) ---
            cv2.rectangle(image_bgr, (left, top), (right, bottom), color, 2)
            cv2.rectangle(image_bgr, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(image_bgr, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            # Собираем результат для этого лица
            recognition_results.append({
                "name": name, 
                "location": {"top": top, "right": right, "bottom": bottom, "left": left},
                "encoding": face_encoding.tolist()  # Добавляем вектор в ответ
            })

        # --- Кодирование обработанного изображения в Base64 ---
        _, buffer = cv2.imencode('.jpg', image_bgr)
        processed_image_base64 = base64.b64encode(buffer).decode('utf-8')

        return {
            "success": True,
            "message": f"Распознавание завершено.",
            "data": {
                "faces_count": len(recognition_results),
                "faces": recognition_results,
                "processed_image_base64": processed_image_base64
            }
        }

    except Exception as e:
        logger.error(f"Критическая ошибка в /api/person/recognize: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"Внутренняя ошибка сервера: {str(e)}"}
        )

@app.get("/api/person/list",
         response_model=ApiResponse,
         tags=["Управление базой"],
         summary="Получить список всех людей в базе")
async def list_persons():
    """Получить список всех людей в базе данных"""
    try:
        persons = []
        
        if os.path.exists(CSV_FILE):
            with open(CSV_FILE, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)  # Пропускаем заголовок
                
                for row in reader:
                    if len(row) >= 4:
                        persons.append({
                            "id": int(row[0]),
                            "name": row[1],
                            "added_date": row[2],
                            "photo_path": row[3]
                        })
        
        return {
            "success": True,
            "message": f"Найдено {len(persons)} человек",
            "data": {
                "count": len(persons),
                "persons": persons
            }
        }
        
    except Exception as e:
        logger.error(f"Ошибка при получении списка: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"Ошибка сервера: {str(e)}",
                "data": None
            }
        )

@app.get("/api/person/{person_id}",
         response_model=ApiResponse,
         tags=["Управление базой"],
         summary="Получить информацию о конкретном человеке")
async def get_person(person_id: int):
    """Получить информацию о конкретном человеке"""
    try:
        with open(CSV_FILE, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            
            for row in reader:
                if len(row) >= 4 and int(row[0]) == person_id:
                    return JSONResponse(content={
                        "success": True,
                        "message": "Человек найден",
                        "data": {
                            "id": int(row[0]),
                            "name": row[1],
                            "added_date": row[2],
                            "photo_path": row[3],
                            "photo_base64": image_to_base64(row[3]) if os.path.exists(row[3]) else None
                        }
                    })
        
        raise HTTPException(status_code=404, detail="Человек не найден")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/person/{person_id}",
            response_model=ApiResponse,
            tags=["Управление базой"],
            summary="Удалить человека из базы данных")
async def delete_person(person_id: int):
    """Удалить человека из базы данных"""
    try:
        rows = []
        deleted = False
        deleted_name = ""
        
        with open(CSV_FILE, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows.append(next(reader))  # Заголовок
            
            for row in reader:
                if len(row) >= 5 and int(row[0]) != person_id:
                    rows.append(row)
                elif int(row[0]) == person_id:
                    deleted = True
                    deleted_name = row[1]
                    # Удаляем файл фото если существует
                    if os.path.exists(row[3]):
                        os.remove(row[3])
        
        if not deleted:
            return JSONResponse(
                status_code=404,
                content={
                    "success": False,
                    "message": "Человек не найден",
                    "data": None
                }
            )
        
        # Перезаписываем файл
        with open(CSV_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        
        # Перезагружаем данные
        load_data()
        
        return {
            "success": True,
            "message": f"Человек '{deleted_name}' удален",
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
                "message": f"Ошибка сервера: {str(e)}",
                "data": None
            }
        )

@app.get("/api/stats",
         response_model=ApiResponse,
         tags=["Статистика"],
         summary="Получить статистику по базе данных")
async def get_stats():
    """Получить статистику системы"""
    try:
        total_persons = len(known_names)
        
        # Последнее добавление
        last_added = None
        if os.path.exists(CSV_FILE):
            with open(CSV_FILE, 'r', encoding='utf-8') as f:
                reader = list(csv.reader(f))
                if len(reader) > 1:
                    last_row = reader[-1]
                    if len(last_row) >= 3:
                        last_added = {
                            "name": last_row[1],
                            "date": last_row[2]
                        }
        
        return {
            "success": True,
            "message": "Статистика получена",
            "data": {
                "total_persons": total_persons,
                "total_photos": total_persons,
                "last_added": last_added,
                "storage_path": UPLOAD_DIR
            }
        }
        
    except Exception as e:
        logger.error(f"Ошибка при получении статистики: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"Ошибка сервера: {str(e)}",
                "data": None
            }
        )

@app.post("/api/person/recognize-base64",
          response_model=ApiResponse,
          tags=["Распознавание"],
          summary="Распознать лица на изображении в формате base64")
async def recognize_person_base64(image_base64: str = Form(...)):
    """
    Распознать лица на изображении в формате base64
    
    Parameters:
    - image_base64: Изображение в формате base64
    """
    try:
        # Конвертируем base64 в изображение
        image = base64_to_image(image_base64)
        
        # Находим лица
        face_locations = face_recognition.face_locations(image)
        
        if len(face_locations) == 0:
            return JSONResponse(content={
                "success": True,
                "message": "На фото не найдено лиц",
                "data": {
                    "faces_count": 0,
                    "faces": []
                }
            })
        
        # Получаем кодировки
        face_encodings = face_recognition.face_encodings(image, face_locations)
        
        results = []
        
        for i, (face_encoding, face_location) in enumerate(zip(face_encodings, face_locations)):
            person_id = 0
            name = "Неизвестный"
            confidence = 0.0
            
            if len(known_encodings) > 0:
                matches = face_recognition.compare_faces(known_encodings, face_encoding)
                face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                
                if True in matches:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        person_id = known_ids[best_match_index]
                        name = known_names[best_match_index]
                        confidence = float(1 - face_distances[best_match_index])
            
            top, right, bottom, left = face_location
            
            results.append({
                "face_id": i + 1,
                "person_id": person_id,
                "name": name,
                "confidence": round(confidence, 3),
                "location": {
                    "top": top,
                    "right": right,
                    "bottom": bottom,
                    "left": left
                }
            })
        
        return JSONResponse(content={
            "success": True,
            "message": f"Найдено {len(results)} лиц",
            "data": {
                "faces_count": len(results),
                "faces": results
            }
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Запуск сервера
if __name__ == "__main__":
    print("🚀 Запуск Face Recognition API...")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)