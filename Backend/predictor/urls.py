from django.urls import path
from .views import home_view, predict_view, dashboard_view, dashboard_powerbi_view, dashboard_tableau_view

urlpatterns = [
    path('', home_view, name='home'),
    path('predict/', predict_view, name='predict'),
    path('dashboard/', dashboard_view, name='dashboard'),
    path('dashboard/powerbi/', dashboard_powerbi_view, name='dashboard_powerbi'),
    path('dashboard/tableau/', dashboard_tableau_view, name='dashboard_tableau'),
]