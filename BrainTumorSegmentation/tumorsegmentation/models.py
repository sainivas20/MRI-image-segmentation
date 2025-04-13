from django.db import models

class ImageModel(models.Model):
    photo =models.ImageField(upload_to="pictures")
    class Meta:
        db_table = "prediction"