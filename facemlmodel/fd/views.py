from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import TempFace, PermFace, Analytics
from .serializers import TempFaceSerializer, PermFaceSerializer, AnalyticsSerializer
from .stream_processor import stream_processor
import cv2
import numpy as np
from datetime import datetime

class StartStreamView(APIView):
    def post(self, request):
        stream_processor.start_streaming()
        return Response({'message': 'Streaming started'}, status=status.HTTP_200_OK)

class StopStreamView(APIView):
    def post(self, request):
        stream_processor.stop_streaming()
        return Response({'message': 'Streaming stopped'}, status=status.HTTP_200_OK)

class GetLatestFrameView(APIView):
    def get(self, request):
        frame, face_data = stream_processor.get_latest_frame()
        if frame is not None:
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            return Response({
                'frame': frame_bytes,
                'face_data': face_data
            }, status=status.HTTP_200_OK)
        else:
            return Response({'message': 'No frame available'}, status=status.HTTP_404_NOT_FOUND)

class TempFaceListView(APIView):
    def get(self, request):
        temp_faces = TempFace.objects.all()
        serializer = TempFaceSerializer(temp_faces, many=True)
        return Response(serializer.data)

class PermFaceListView(APIView):
    def get(self, request):
        perm_faces = PermFace.objects.all()
        serializer = PermFaceSerializer(perm_faces, many=True)
        return Response(serializer.data)

class AnalyticsView(APIView):
    def get(self, request):
        analytics = Analytics.objects.all()
        serializer = AnalyticsSerializer(analytics, many=True)
        return Response(serializer.data)

class UpdateFaceIDView(APIView):
    def post(self, request):
        unknown_id = request.data.get('unknown_id')
        new_name = request.data.get('new_name')
        
        try:
            temp_face = TempFace.objects.get(face_id=unknown_id)
            perm_face = PermFace.objects.create(
                name=new_name,
                image_paths=temp_face.image_paths,
                embeddings=temp_face.embeddings,
                last_seen=temp_face.timestamp
            )
            temp_face.delete()
            return Response({'message': 'Face ID updated successfully'}, status=status.HTTP_200_OK)
        except TempFace.DoesNotExist:
            return Response({'error': 'Face ID not found'}, status=status.HTTP_404_NOT_FOUND)