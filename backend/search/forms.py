from django import forms
from .models import GetWordQuery

class GetWordQueryForm(forms.ModelForm):
    class Meta:
        model = GetWordQuery
        fields = ['Query_text']
