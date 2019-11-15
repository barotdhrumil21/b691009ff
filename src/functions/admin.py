from django.contrib import admin

from functions.models import (
                            omr_templates,
                            Exam)
# Register your models here.

admin.site.register(omr_templates)
admin.site.register(Exam)
