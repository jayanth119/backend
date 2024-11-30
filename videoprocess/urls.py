from django.urls import path
from .views import VideoApi

urlpatterns = [
    path('api/video/', VideoApi.as_view(), name='video_api'),
]
