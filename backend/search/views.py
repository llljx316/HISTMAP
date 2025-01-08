from django.shortcuts import render


# Create your views here.
# import sys 
# sys.path.append('.')
from django.http import HttpResponse
from django.http import JsonResponse
from .forms import GetWordQueryForm
from utils import *
import json
import requests
import os
from django.conf import settings
from pathlib import Path


def index(request):
    return HttpResponse("Hello, world. You're at the polls index.")

def add_query(request):
    if request.method == "POST":
        data = json.loads(request.body)
        # form = GetWordQueryForm(request.POST)
        form = GetWordQueryForm(data)
        if form.is_valid():
            # 保存表单数据到数据库
            query = form.save()

            return JsonResponse({"message": "Query added successfully!", "id": query.id}, status=201)
        else:
            return JsonResponse({"errors": form.errors}, status=400)
    else:
        return JsonResponse({"message": "Invalid request method. Use POST."}, status=405)

def query(request):
    # query = json.loads(request.body)
    query = request.GET.get('q')
    num = request.GET.get('num',10)
    on = request.GET.get('on', 'image')
    # full = request.GET.get('full')
    # input_data = {"text": query}
    data =requests.get(MODEL_SERVER_URL +"text_predict", params={'text_input': query, 'num': num, 'on':on}) 
    # data = json.loads(data.text)
    # 
    # return JsonResponse(data, safe=False, status=400) 
    return HttpResponse(data.text) #

def id_query(request):
    query = request.GET.get('id')
    num = request.GET.get('num',10)
    on = request.GET.get('on', 'image')
    # input_data = {"text": query}
    data =requests.get(MODEL_SERVER_URL +"id_predict", params={'id': query, 'num': num, 'on':on}) 
    # data = json.loads(data.text)
    # 
    # return JsonResponse(data, safe=False, status=400) 
    return HttpResponse(data.text) #

def query_num(request):
    query = request.GET.get('text_input')
    threshold = request.GET.get('threshold',30)
    on = request.GET.get('on', 'image')
    # input_data = {"text": query}
    data =requests.get(MODEL_SERVER_URL +"distance_num", params={'text_input': query, 'threshold': threshold, 'on':on}) 
    # data = json.loads(data.text)
    # 
    # return JsonResponse(data, safe=False, status=400) 
    return HttpResponse(data.text) #

def filter_dataset(request):
    dataset_name = json.loads(request.body.decode('utf-8'))

    # input_data = {"text": query}
    print(dataset_name)
    data =requests.post(MODEL_SERVER_URL +"filter_dataset", data=request.body) 
    return HttpResponse(data.text) #



def upload_image(request):
    if request.method == 'POST' and request.FILES.get('file'):
        # 获取上传的文件对象
        image_file = request.FILES['file']
        numEle= int(request.POST['num'])
        try:
            on = request.POST['on']
        except:
            on = 'image'

        # 指定保存路径
        save_path = settings.MEDIA_ROOT/ 'uploads'
        os.makedirs(save_path, exist_ok=True)  # 如果路径不存在，创建目录

        image_file_name = f'query_image{Path(image_file.name).suffix}'
        # 保存文件到服务器
        file_path = save_path/ image_file_name
        with open(file_path, 'wb+') as destination:
            for chunk in image_file.chunks():
                destination.write(chunk)

        # process photo
        data =requests.get(MODEL_SERVER_URL+"image_predict/", params={'image_path': str(file_path.relative_to(file_path.parents[3])), 'num': numEle, 'on': on}) 

        return HttpResponse(data.text) #
        # return JsonResponse({'status': 'success', 'file_path': file_path})
    else:
        return JsonResponse({'status': 'fail', 'message': 'No file uploaded'})