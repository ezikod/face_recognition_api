import os
import csv
import json
import base64
import shutil
import tempfile
import asyncio
from datetime import datetime
from pathlib import Path
import numpy as np
import cv2

import pytest
from httpx import AsyncClient
from fastapi.testclient import TestClient
from PIL import Image
from unittest.mock import patch, MagicMock

# Import your app
from main import app, init_csv, load_data, known_encodings, known_names, known_ids, api_state, ProcessingState, CSV_FILE, UPLOAD_DIR


# Test fixtures
@pytest.fixture
def test_client():
    """Create a test client"""
    return TestClient(app)


@pytest.fixture
async def async_client():
    """Create an async test client"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture
def temp_dirs(tmp_path):
    """Create temporary directories for testing"""
    temp_csv = tmp_path / "test_persons.csv"
    temp_uploads = tmp_path / "test_uploads"
    temp_uploads.mkdir()
    
    # Patch the global variables
    with patch("main.CSV_FILE", str(temp_csv)), \
         patch("main.UPLOAD_DIR", str(temp_uploads)):
        init_csv()
        load_data()
        yield temp_csv, temp_uploads
    
    # Cleanup
    if temp_csv.exists():
        temp_csv.unlink()
    if temp_uploads.exists():
        shutil.rmtree(temp_uploads)


@pytest.fixture
def sample_image():
    """Create a sample test image"""
    # Create a simple white image
    img = Image.new('RGB', (100, 100), color='white')
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    img.save(temp_file.name)
    temp_file.close()
    yield temp_file.name
    os.unlink(temp_file.name)


@pytest.fixture
def sample_image_with_face():
    """Create a sample image with a detectable face pattern"""
    # Create an image with a face-like pattern (for testing purposes)
    img = np.ones((200, 200, 3), dtype=np.uint8) * 255
    # Draw a simple face-like pattern
    cv2.circle(img, (100, 100), 50, (200, 200, 200), -1)  # Face
    cv2.circle(img, (80, 80), 10, (0, 0, 0), -1)  # Left eye
    cv2.circle(img, (120, 80), 10, (0, 0, 0), -1)  # Right eye
    cv2.ellipse(img, (100, 120), (20, 10), 0, 0, 180, (0, 0, 0), 2)  # Mouth
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    cv2.imwrite(temp_file.name, img)
    temp_file.close()
    yield temp_file.name
    os.unlink(temp_file.name)


@pytest.fixture
def sample_video():
    """Create a sample test video"""
    # Create a simple video with 10 frames
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    out = cv2.VideoWriter(temp_file.name, fourcc, 1.0, (100, 100))
    
    for i in range(10):
        frame = np.ones((100, 100, 3), dtype=np.uint8) * (i * 25)
        out.write(frame)
    
    out.release()
    temp_file.close()
    yield temp_file.name
    os.unlink(temp_file.name)


# Unit Tests

class TestHelperFunctions:
    """Test helper functions"""
    
    def test_init_csv(self, tmp_path):
        """Test CSV initialization"""
        csv_path = tmp_path / "test.csv"
        with patch("main.CSV_FILE", str(csv_path)):
            init_csv()
            assert csv_path.exists()
            
            # Check header
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                header = next(reader)
                assert header == ['ID', 'Name', 'Date Added', 'Photo Path', 'Face Encoding']
    
    def test_image_to_base64(self, sample_image):
        """Test image to base64 conversion"""
        from main import image_to_base64
        
        base64_str = image_to_base64(sample_image)
        assert isinstance(base64_str, str)
        assert len(base64_str) > 0
        
        # Test decoding
        decoded = base64.b64decode(base64_str)
        assert len(decoded) > 0
    
    def test_base64_to_image(self):
        """Test base64 to image conversion"""
        from main import base64_to_image
        
        # Create a simple image and encode it
        img = Image.new('RGB', (10, 10), color='red')
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        base64_str = base64.b64encode(buffer.getvalue()).decode()
        
        # Test conversion
        result = base64_to_image(base64_str)
        assert isinstance(result, np.ndarray)
        assert result.shape == (10, 10, 3)


# API Endpoint Tests

class TestRootEndpoint:
    """Test root endpoint"""
    
    def test_root(self, test_client):
        """Test root endpoint returns correct info"""
        response = test_client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "active"
        assert data["service"] == "Face Recognition API"
        assert "endpoints" in data
        assert data["current_state"] == "idle"


class TestStatusEndpoint:
    """Test status endpoint"""
    
    def test_get_status(self, test_client):
        """Test status endpoint"""
        response = test_client.get("/api/status")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "state" in data["data"]
        assert "is_busy" in data["data"]
        assert data["data"]["is_busy"] is False


class TestPersonManagement:
    """Test person management endpoints"""
    
    @pytest.mark.asyncio
    async def test_add_person_no_face(self, async_client, temp_dirs, sample_image):
        """Test adding person with image without face"""
        with open(sample_image, "rb") as f:
            response = await async_client.post(
                "/api/person/add",
                data={"name": "Test Person"},
                files={"photo": ("test.jpg", f, "image/jpeg")}
            )
        
        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert "No faces found" in data["message"]
    
    @pytest.mark.asyncio
    @patch('face_recognition.face_locations')
    @patch('face_recognition.face_encodings')
    async def test_add_person_success(self, mock_encodings, mock_locations, 
                                    async_client, temp_dirs, sample_image):
        """Test successfully adding a person"""
        # Mock face detection
        mock_locations.return_value = [(0, 100, 100, 0)]
        mock_encodings.return_value = [np.random.rand(128)]
        
        with open(sample_image, "rb") as f:
            response = await async_client.post(
                "/api/person/add",
                data={"name": "John Doe"},
                files={"photo": ("john.jpg", f, "image/jpeg")}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["name"] == "John Doe"
        assert data["data"]["id"] == 1
        assert data["data"]["faces_found"] == 1
    
    @pytest.mark.asyncio
    async def test_list_persons_empty(self, async_client, temp_dirs):
        """Test listing persons when database is empty"""
        response = await async_client.get("/api/person/list")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["count"] == 0
        assert data["data"]["persons"] == []
    
    @pytest.mark.asyncio
    @patch('face_recognition.face_locations')
    @patch('face_recognition.face_encodings')
    async def test_list_persons_with_data(self, mock_encodings, mock_locations, 
                                        async_client, temp_dirs, sample_image):
        """Test listing persons with data"""
        # Add a person first
        mock_locations.return_value = [(0, 100, 100, 0)]
        mock_encodings.return_value = [np.random.rand(128)]
        
        with open(sample_image, "rb") as f:
            await async_client.post(
                "/api/person/add",
                data={"name": "Jane Doe"},
                files={"photo": ("jane.jpg", f, "image/jpeg")}
            )
        
        # List persons
        response = await async_client.get("/api/person/list")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["count"] == 1
        assert len(data["data"]["persons"]) == 1
        assert data["data"]["persons"][0]["name"] == "Jane Doe"
    
    @pytest.mark.asyncio
    async def test_get_person_not_found(self, async_client, temp_dirs):
        """Test getting non-existent person"""
        response = await async_client.get("/api/person/999")
        assert response.status_code == 404
    
    @pytest.mark.asyncio
    @patch('face_recognition.face_locations')
    @patch('face_recognition.face_encodings')
    async def test_delete_person(self, mock_encodings, mock_locations, 
                                async_client, temp_dirs, sample_image):
        """Test deleting a person"""
        # Add a person first
        mock_locations.return_value = [(0, 100, 100, 0)]
        mock_encodings.return_value = [np.random.rand(128)]
        
        with open(sample_image, "rb") as f:
            response = await async_client.post(
                "/api/person/add",
                data={"name": "Delete Me"},
                files={"photo": ("delete.jpg", f, "image/jpeg")}
            )
        
        person_id = response.json()["data"]["id"]
        
        # Delete the person
        response = await async_client.delete(f"/api/person/{person_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["name"] == "Delete Me"
        
        # Verify deletion
        response = await async_client.get(f"/api/person/{person_id}")
        assert response.status_code == 404


class TestRecognition:
    """Test recognition endpoints"""
    
    @pytest.mark.asyncio
    @patch('face_recognition.face_locations')
    @patch('face_recognition.face_encodings')
    async def test_recognize_person(self, mock_encodings, mock_locations, 
                                  async_client, temp_dirs, sample_image):
        """Test face recognition"""
        # Mock face detection
        mock_locations.return_value = [(0, 100, 100, 0)]
        mock_encodings.return_value = [np.random.rand(128)]
        
        with open(sample_image, "rb") as f:
            response = await async_client.post(
                "/api/person/recognize",
                files={"photo": ("test.jpg", f, "image/jpeg")}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["faces_count"] == 1
        assert "processed_image_base64" in data["data"]
    
    @pytest.mark.asyncio
    async def test_recognize_base64(self, async_client, temp_dirs):
        """Test base64 image recognition"""
        # Create a simple base64 image
        img = Image.new('RGB', (10, 10), color='blue')
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        base64_str = base64.b64encode(buffer.getvalue()).decode()
        
        with patch('face_recognition.face_locations', return_value=[]):
            response = await async_client.post(
                "/api/person/recognize-base64",
                data={"image_base64": base64_str}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["faces_count"] == 0
    
    @pytest.mark.asyncio
    async def test_video_recognition(self, async_client, temp_dirs, sample_video):
        """Test video recognition"""
        with patch('face_recognition.face_locations', return_value=[]):
            with open(sample_video, "rb") as f:
                response = await async_client.post(
                    "/api/video/recognize",
                    data={"frame_interval": "5"},
                    files={"video": ("test.mp4", f, "video/mp4")}
                )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "summary" in data["data"]


class TestConcurrency:
    """Test concurrent request handling"""
    
    @pytest.mark.asyncio
    async def test_concurrent_image_recognition_blocked(self, async_client, temp_dirs, sample_image):
        """Test that concurrent image recognition is blocked"""
        # Start first recognition
        with patch('face_recognition.face_locations', return_value=[(0, 100, 100, 0)]), \
             patch('face_recognition.face_encodings', return_value=[np.random.rand(128)]):
            
            # Simulate slow processing
            async def slow_recognition(*args, **kwargs):
                await asyncio.sleep(0.5)
                return [(0, 100, 100, 0)]
            
            with patch('face_recognition.face_locations', side_effect=slow_recognition):
                # Start first request
                with open(sample_image, "rb") as f:
                    task1 = asyncio.create_task(
                        async_client.post(
                            "/api/person/recognize",
                            files={"photo": ("test1.jpg", f, "image/jpeg")}
                        )
                    )
                
                # Wait a bit and try second request
                await asyncio.sleep(0.1)
                
                with open(sample_image, "rb") as f:
                    response2 = await async_client.post(
                        "/api/person/recognize",
                        files={"photo": ("test2.jpg", f, "image/jpeg")}
                    )
                
                # Second request should be rejected
                assert response2.status_code == 503
                data2 = response2.json()
                assert data2["success"] is False
                assert "System is busy" in data2["message"]
                
                # Wait for first request to complete
                response1 = await task1
                assert response1.status_code == 200
    
    @pytest.mark.asyncio
    async def test_video_blocks_image_recognition(self, async_client, temp_dirs, sample_image, sample_video):
        """Test that video processing blocks image recognition"""
        # Start video processing
        async def slow_video_processing(*args, **kwargs):
            await asyncio.sleep(0.5)
            return []
        
        with patch('face_recognition.face_locations', side_effect=slow_video_processing):
            # Start video processing
            with open(sample_video, "rb") as f:
                video_task = asyncio.create_task(
                    async_client.post(
                        "/api/video/recognize",
                        data={"frame_interval": "5"},
                        files={"video": ("test.mp4", f, "video/mp4")}
                    )
                )
            
            # Wait a bit and try image recognition
            await asyncio.sleep(0.1)
            
            with open(sample_image, "rb") as f:
                image_response = await async_client.post(
                    "/api/person/recognize",
                    files={"photo": ("test.jpg", f, "image/jpeg")}
                )
            
            # Image recognition should be blocked
            assert image_response.status_code == 503
            data = image_response.json()
            assert "processing_video" in data["data"]["current_state"]
            
            # Clean up
            await video_task


class TestStats:
    """Test statistics endpoint"""
    
    @pytest.mark.asyncio
    async def test_stats_empty(self, async_client, temp_dirs):
        """Test stats with empty database"""
        response = await async_client.get("/api/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["total_persons"] == 0
        assert data["data"]["api_state"] == "idle"
    
    @pytest.mark.asyncio
    @patch('face_recognition.face_locations')
    @patch('face_recognition.face_encodings')
    async def test_stats_with_data(self, mock_encodings, mock_locations, 
                                 async_client, temp_dirs, sample_image):
        """Test stats with data"""
        # Add some people
        mock_locations.return_value = [(0, 100, 100, 0)]
        mock_encodings.return_value = [np.random.rand(128)]
        
        for i, name in enumerate(["Alice", "Bob", "Charlie"]):
            with open(sample_image, "rb") as f:
                await async_client.post(
                    "/api/person/add",
                    data={"name": name},
                    files={"photo": (f"{name}.jpg", f, "image/jpeg")}
                )
        
        response = await async_client.get("/api/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["data"]["total_persons"] == 3
        assert data["data"]["last_added"]["name"] == "Charlie"


# Integration Tests

class TestIntegration:
    """Integration tests for complete workflows"""
    
    @pytest.mark.asyncio
    @patch('face_recognition.face_locations')
    @patch('face_recognition.face_encodings')
    @patch('face_recognition.compare_faces')
    @patch('face_recognition.face_distance')
    async def test_add_and_recognize_workflow(self, mock_distance, mock_compare, 
                                            mock_encodings, mock_locations,
                                            async_client, temp_dirs, sample_image):
        """Test complete add and recognize workflow"""
        # Setup mocks
        person_encoding = np.random.rand(128)
        mock_locations.return_value = [(0, 100, 100, 0)]
        mock_encodings.return_value = [person_encoding]
        
        # Add a person
        with open(sample_image, "rb") as f:
            response = await async_client.post(
                "/api/person/add",
                data={"name": "Test Person"},
                files={"photo": ("test.jpg", f, "image/jpeg")}
            )
        
        assert response.status_code == 200
        
        # Setup recognition mocks
        mock_compare.return_value = [True]
        mock_distance.return_value = [0.3]
        
        # Recognize the same person
        with open(sample_image, "rb") as f:
            response = await async_client.post(
                "/api/person/recognize",
                files={"photo": ("recognize.jpg", f, "image/jpeg")}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["data"]["faces_count"] == 1
        assert data["data"]["faces"][0]["name"] == "Test Person"


# Error Handling Tests

class TestErrorHandling:
    """Test error handling"""
    
    @pytest.mark.asyncio
    async def test_invalid_image_format(self, async_client, temp_dirs):
        """Test handling of invalid image format"""
        # Create a text file instead of image
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is not an image")
            temp_file = f.name
        
        try:
            with open(temp_file, "rb") as f:
                with patch('face_recognition.load_image_file', side_effect=Exception("Invalid image")):
                    response = await async_client.post(
                        "/api/person/add",
                        data={"name": "Test"},
                        files={"photo": ("test.txt", f, "text/plain")}
                    )
            
            assert response.status_code == 500
            data = response.json()
            assert data["success"] is False
        finally:
            os.unlink(temp_file)
    
    @pytest.mark.asyncio
    async def test_csv_corruption_handling(self, async_client, temp_dirs):
        """Test handling of corrupted CSV"""
        temp_csv, _ = temp_dirs
        
        # Corrupt the CSV
        with open(temp_csv, 'w') as f:
            f.write("Invalid,CSV,Content\n")
            f.write("Missing,Data\n")
        
        # Try to load data
        with patch('main.load_data', side_effect=Exception("CSV error")):
            response = await async_client.get("/api/person/list")
        
        # Should still return a response
        assert response.status_code in [200, 500]


# Performance Tests

class TestPerformance:
    """Test performance and limits"""
    
    @pytest.mark.asyncio
    @patch('face_recognition.face_locations')
    @patch('face_recognition.face_encodings')
    async def test_large_database_performance(self, mock_encodings, mock_locations,
                                            async_client, temp_dirs, sample_image):
        """Test performance with large number of people"""
        # Mock face detection
        mock_locations.return_value = [(0, 100, 100, 0)]
        
        # Add 100 people
        import time
        start_time = time.time()
        
        for i in range(100):
            mock_encodings.return_value = [np.random.rand(128)]
            with open(sample_image, "rb") as f:
                response = await async_client.post(
                    "/api/person/add",
                    data={"name": f"Person {i}"},
                    files={"photo": (f"person{i}.jpg", f, "image/jpeg")}
                )
            assert response.status_code == 200
        
        elapsed = time.time() - start_time
        assert elapsed < 60  # Should complete within 1 minute
        
        # Test listing performance
        start_time = time.time()
        response = await async_client.get("/api/person/list")
        elapsed = time.time() - start_time
        
        assert response.status_code == 200
        assert elapsed < 1  # Listing should be fast


if __name__ == "__main__":
    pytest.main([__file__, "-v"])