from django.urls import path
from .views import StartStreamView, StopStreamView, GetLatestFrameView, TempFaceListView, PermFaceListView, AnalyticsView, UpdateFaceIDView

urlpatterns = [
    path('start-stream/', StartStreamView.as_view(), name='start_stream'),
    path('stop-stream/', StopStreamView.as_view(), name='stop_stream'),
    path('get-latest-frame/', GetLatestFrameView.as_view(), name='get_latest_frame'),
    path('temp-faces/', TempFaceListView.as_view(), name='temp_face_list'),
    path('perm-faces/', PermFaceListView.as_view(), name='perm_face_list'),
    path('analytics/', AnalyticsView.as_view(), name='analytics'),
    path('update-face-id/', UpdateFaceIDView.as_view(), name='update_face_id'),
]