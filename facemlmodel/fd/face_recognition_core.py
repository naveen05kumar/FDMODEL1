import cv2
import numpy as np
import torch
from ultralytics import YOLO
from deep_sort_realtime.deep_sort import nn_matching
from deep_sort_realtime.deep_sort.detection import Detection
from deep_sort_realtime.deep_sort.tracker import Tracker
import os
from datetime import datetime, date
from .models import TempFace
import logging

logger = logging.getLogger(__name__)

class FaceRecognitionCore:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.facemodel = YOLO('yolov8m-face.pt').to(self.device)
        self.tracker = self.initialize_tracker()
        self.current_date = date.today()
        self.face_id_counter = 1
        self.face_id_mapping = {}
        self.frame_save_counter = {}

    def initialize_tracker(self):
        max_cosine_distance = 0.3
        nn_budget = 300
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        return Tracker(metric, max_age=100)

    def detect_faces(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.facemodel(frame_rgb, conf=0.49)
        faces = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf.item()
                faces.append([x1, y1, x2 - x1, y2 - y1, confidence])
        return np.array(faces)

    def generate_simple_feature(self, face, frame):
        x, y, w, h, _ = face.astype(int)
        face_roi = frame[y:y+h, x:x+w]
        if face_roi.size == 0:
            return np.zeros(64*64)
        face_roi = cv2.resize(face_roi, (64, 64))
        feature = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY).flatten()
        norm = np.linalg.norm(feature)
        return feature / norm if norm != 0 else feature

    def get_next_face_id(self):
        face_id = f"unknown_{self.face_id_counter:03d}"
        self.face_id_counter += 1
        return face_id

    def save_face_image(self, frame, track):
        track_id = int(track.track_id)
        if track_id not in self.face_id_mapping:
            self.face_id_mapping[track_id] = self.get_next_face_id()
            self.frame_save_counter[track_id] = 0
        
        self.frame_save_counter[track_id] += 1
        if self.frame_save_counter[track_id] % 7 != 0:
            return
        
        face_id = self.face_id_mapping[track_id]
        today = self.current_date.strftime("%Y-%m-%d")
        directory = os.path.join('media', today, face_id, "images")
        os.makedirs(directory, exist_ok=True)
        
        bbox = track.to_tlbr()
        h, w = frame.shape[:2]
        pad_w, pad_h = 0.2 * (bbox[2] - bbox[0]), 0.2 * (bbox[3] - bbox[1])
        x1, y1 = max(0, int(bbox[0] - pad_w)), max(0, int(bbox[1] - pad_h))
        x2, y2 = min(w, int(bbox[2] + pad_w)), min(h, int(bbox[3] + pad_h))
        
        face_img = frame[y1:y2, x1:x2]
        
        if face_img.size > 0:
            image_count = len([f for f in os.listdir(directory) if f.endswith('.jpg')])
            if image_count < 15:
                filename = os.path.join(directory, f"{face_id}_{image_count:02d}.jpg")
                cv2.imwrite(filename, face_img)
                
                temp_face, created = TempFace.objects.get_or_create(face_id=face_id)
                temp_face.image_paths.append(filename)
                temp_face.processed = False
                temp_face.save()
        
        return face_id

    def process_frame(self, frame):
        faces = self.detect_faces(frame)
        detections = [Detection(face[:4], face[4], self.generate_simple_feature(face, frame)) for face in faces]
        
        self.tracker.predict()
        self.tracker.update(detections)
        
        processed_faces = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            face_id = self.save_face_image(frame, track)
            processed_faces.append({
                'face_id': face_id,
                'bbox': bbox.tolist(),
            })
        
        return frame, processed_faces

face_recognition_core = FaceRecognitionCore()