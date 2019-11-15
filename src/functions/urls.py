"""omr URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path,include
from django.contrib.auth.decorators import login_required
from .views import (upload_view,
                    upload_exam_view,
                    all_templates,
                    all_exams,
                    exam_detail_view,
                    multiple_inputs,
                    delete_img,
                    eval,
                    ReportEval)

from functions.views import (
                pdf_view,
                delete_template,
                )


urlpatterns = [
    #path('add-pdf/',login_required(upload_view.as_view()),name="pdf-adder"),
    path('pdf-viewer/<str:name>',login_required(pdf_view),name="pdf-templates"),
    path('all-templates/',login_required(all_templates.as_view()),name='all-templates'),
    path('delete/<int:id>',login_required(delete_template),name='delete-template'),


    path('add-exam/',login_required(upload_exam_view.as_view()),name="exam-adder"),
    path('exams/',login_required(all_exams.as_view()),name='all-exams'),
    path('exam-detail/<str:name>',login_required(exam_detail_view.as_view()),name='exam-detail'),
    path('exam-detail/<str:name>/inputs',login_required(multiple_inputs.as_view()),name="exam-detail-input"),
    path('exam-detail/inputs/delete/<str:fname>/<str:exam>',login_required(delete_img),name="exam-input-delete"),

    path('eval/',login_required(eval.as_view()),name='exam-eval'),
    path('report-eval/',login_required (ReportEval) ,name='report-eval')




]
