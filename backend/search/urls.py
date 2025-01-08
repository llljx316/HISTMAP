from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("add-query/", views.add_query, name='add_query'),
    path("query/", views.query, name = "query"),
    path("upload-img/", views.upload_image, name = "upload_img"),
    path("id_query/", views.id_query, name="id_query"),
    path("query_num/", views.query_num, name="id_query_num"),
    path("filter_dataset/", views.filter_dataset, name="filter_dataset"),

]