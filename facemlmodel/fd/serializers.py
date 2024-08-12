from rest_framework import serializers
from .models import TempFace, PermFace, Analytics

class TempFaceSerializer(serializers.ModelSerializer):
    class Meta:
        model = TempFace
        fields = '__all__'

class PermFaceSerializer(serializers.ModelSerializer):
    class Meta:
        model = PermFace
        fields = '__all__'

class AnalyticsSerializer(serializers.ModelSerializer):
    class Meta:
        model = Analytics
        fields = '__all__'