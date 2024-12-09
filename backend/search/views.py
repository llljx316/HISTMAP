from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse
from django.http import JsonResponse
from .forms import GetWordQueryForm
import json
import requests



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
    # input_data = {"text": query}
    input_data = query
    data =requests.get("http://127.0.0.1:8001/text_predict", params={'text_input': query, 'num': num}) 
    # data = json.loads(data.text)
    # 
    # return JsonResponse(data, safe=False, status=400) 
    return HttpResponse(data.text) #