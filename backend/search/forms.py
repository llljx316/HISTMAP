from django import forms
from .models import GetWordQuery, ImageModel

class GetWordQueryForm(forms.ModelForm):
    class Meta:
        model = GetWordQuery
        fields = ['Query_text']

class ImageForm(forms.ModelForm):
    class Meta:
        model = ImageModel
        fields = ['image']