from django.urls import path
from .views import home_view, predict_view, dashboard_view

urlpatterns = [
    path('', home_view, name='home'),
    path('predict/', predict_view, name='predict'),
    path('dashboard/', dashboard_view, name='dashboard'),
]