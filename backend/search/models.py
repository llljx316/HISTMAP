from django.db import models

# Create your models here.
class GetWordQuery(models.Model):
    Query_text = models.CharField(max_length=500)
    def __str__(self):
            return self.Query_text

class ImageModel(models.Model):
    image = models.ImageField(upload_to='images/')        # 图片存储路径，`upload_to` 指定文件夹
