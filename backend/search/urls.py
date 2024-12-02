from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("add-query/", views.add_query, name='add_query'),
    path("query/", views.query, name = "query"),
]