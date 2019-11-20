from django.db import models
from django.contrib.auth.models import User
from django.urls import reverse
# Create your models here.

class omr_templates(models.Model):
        pdf_name = models.CharField(max_length = 120)
        pdf = models.FileField(null=False,blank=False,upload_to='templates/' )
        user = models.ForeignKey(User,on_delete=models.CASCADE)


        def __str__(self):
            return self.pdf_name

        def get_absolute_url(self):
            return reverse('all-templates')


class Exam(models.Model):
        user = models.ForeignKey(User,on_delete=models.CASCADE)
        exam_name = models.CharField(max_length = 200)
        ansKey = models.FileField(null=True,blank=True,upload_to='exams/answer_key/' )
        ansKeyImg = models.FileField(default="default.jpg", null=True,blank=True,upload_to='exams/answer_key_img/' )
        template = models.ForeignKey(omr_templates,on_delete=models.PROTECT)


        def __str__(self):
            return self.exam_name

        def get_absolute_url(self):
            return reverse('all-exams')
