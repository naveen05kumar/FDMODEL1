from django.db import models

class TempFace(models.Model):
    face_id = models.CharField(max_length=100, unique=True)
    image_paths = models.JSONField(default=list)
    embeddings = models.JSONField(default=list)
    timestamp = models.DateTimeField(auto_now_add=True)
    processed = models.BooleanField(default=False)

class PermFace(models.Model):
    name = models.CharField(max_length=100)
    image_paths = models.JSONField(default=list)
    embeddings = models.JSONField(default=list)
    last_seen = models.DateTimeField()

class Analytics(models.Model):
    date = models.DateField(unique=True)
    detected_persons = models.JSONField(default=dict)
    total_detections = models.IntegerField(default=0)
    last_updated = models.DateTimeField(auto_now=True)