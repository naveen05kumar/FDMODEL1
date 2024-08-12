import cv2
import numpy as np
import threading
import queue
import time
from .face_recognition_core import face_recognition_core
import logging

logger = logging.getLogger(__name__)

class StreamProcessor:
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=30)
        self.result_queue = queue.Queue(maxsize=30)
        self.terminate_flag = False

    def ensure_connection(self, max_retries=10):
        for attempt in range(max_retries):
            if self.cap is None or not self.cap.isOpened():
                logger.info(f"Attempting to connect to RTSP stream (Attempt {attempt + 1}/{max_retries})...")
                if self.cap is not None:
                    self.cap.release()
                self.cap = cv2.VideoCapture(self.rtsp_url)
                if self.cap.isOpened():
                    logger.info("Successfully connected to RTSP stream.")
                    return True
            time.sleep(2)
        logger.error("Failed to connect to RTSP stream after multiple attempts.")
        return False

    def read_frame(self):
        if self.cap is None or not self.cap.isOpened():
            if not self.ensure_connection():
                return None
        ret, frame = self.cap.read()
        if not ret:
            logger.warning("Failed to read frame. Reconnecting...")
            if self.ensure_connection():
                ret, frame = self.cap.read()
        return frame if ret else None

    def frame_producer(self):
        while not self.terminate_flag:
            frame = self.read_frame()
            if frame is not None:
                if self.frame_queue.full():
                    self.frame_queue.get()
                self.frame_queue.put(frame)
            else:
                time.sleep(0.1)

    def frame_consumer(self):
        while not self.terminate_flag:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                processed_frame, face_data = face_recognition_core.process_frame(frame)
                if self.result_queue.full():
                    self.result_queue.get()
                self.result_queue.put((processed_frame, face_data))
            else:
                time.sleep(0.01)

    def start_streaming(self):
        self.terminate_flag = False
        self.ensure_connection()
        producer_thread = threading.Thread(target=self.frame_producer)
        consumer_thread = threading.Thread(target=self.frame_consumer)
        producer_thread.start()
        consumer_thread.start()
        logger.info("Streaming started")

    def stop_streaming(self):
        self.terminate_flag = True
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        logger.info("Streaming stopped")

    def get_latest_frame(self):
        if not self.result_queue.empty():
            return self.result_queue.get()
        return None, None

stream_processor = StreamProcessor('rtsp://admin:skypler@sriram@210.18.176.33:1024/Streaming/channels/101')