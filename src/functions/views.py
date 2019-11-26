import os
import glob
from django.http import HttpResponse, Http404
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.shortcuts import render,redirect,get_object_or_404
#from django.core.urlresolvers import reverse
from django.views.generic import(
                                    CreateView,
                                    ListView,
                                    DetailView,
                                    UpdateView,
                                    DeleteView,
                                    FormView)
from django.core.files.storage import FileSystemStorage
from .forms import (upload_form,
                    exam_builder,
                    exam_editor,
                    multiple_files_input)
from .models import (
                    omr_templates,
                    Exam)
from .all import *
import pdfkit
from django.contrib.auth.decorators import login_required

from django.conf import settings

# Create your views here.


#-------------------------------------TEMPLATE VIEWS-------------------------------------------
#def upload_view(request,*args,**kwargs):
#    form = upload_form(request.POST or None,request.FILES or None)
#    if form.is_valid():
#        form.save()
#    templates = omr_templates.objects.all()
#    context ={
#    'templates':templates
#    }
#    return render(request,'templateAdder.html',context)

#global variable for answer key
Answers = {}

#"/functions/add-pdf"

class upload_view(CreateView):
    model = omr_templates
    form_class=upload_form
    template_name = 'templateAdder.html'

    def form_valid(self,form):
        form.instance.user = self.request.user
        return super().form_valid(form)

#"/functions/pdf-viewer/<PDF_NAME>"
def pdf_view(request,name,*args,**kwargs):
    obj = omr_templates.objects.get(pdf_name=name)
    context = {
    'obj':obj,
    'id':obj.id
    }
    #print(obj.id)
    return render(request,'preview.html',context)

#"/functions/all-templates"
class all_templates(ListView):
    model = omr_templates
    template_name = 'templates.html'
    #context_object_name = 'templates'
    def get_queryset(self):
        return omr_templates.objects.filter(user=self.request.user)

#"/functions/delete/<PDF_ID>"
def delete_template(request,id,*args,**kwargs):
    obj = omr_templates.objects.get(id=id)
    obj.delete()
    #print(obj)
    return redirect('pdf-adder')

#------------------------------------EXAMS VIEWS-----------------------------------
def exam_answers(request):
    global Answers
    dataFile = request.FILES['datafile']
    Afile=FileSystemStorage(location=r'./media/temp/')
    Af_name=dataFile.name
    Af_extension = dataFile.content_type.split('/')[1]
    #print(Af_name,Af_extension)
    Afile.save(Af_name,dataFile)
    path = "media/temp/"+Af_name
    Temp = fromImage(imgPath = path)
    if Temp != {}:
        Answers = Temp
    else:
        Answers = {}
    Afile.delete(Af_name)
    #print("Answers:",Answers,"Temp:",Temp)
    return redirect('exam-adder')

#"/functions/add-exam"
class upload_exam_view(CreateView):
    model = Exam
    form_class = exam_builder
    template_name = 'examAdder.html'
    global Answers

    def form_valid(self,form):
        form.instance.user = self.request.user
        return super().form_valid(form)

    def get_context_data(self,*args,**kwargs):
        context = super().get_context_data(**kwargs)
        context['Answers'] = Answers
        return context

#"/functions/exams"
class all_exams(ListView):
    model = Exam
    template_name = 'exam.html'
    def get_queryset(self):
        #print(Exam.objects.filter(user=self.request.user)   )
        return Exam.objects.filter(user=self.request.user)

#"/functions/exam-detail/<EXAM_NAME>"
class exam_detail_view(DetailView):
    template_name = "exam_detail.html/"
    context_object_name = 'object'

    def get_object(self):
        #print(self.kwargs['name'],"thissssssss")
        #do the logic here
        obj = Exam.objects.filter(exam_name=self.kwargs['name'])
        return obj

    def get_context_data(self, **kwargs):
        # Call the base implementation first to get a context
        context = super().get_context_data(**kwargs)
        oprating_object = self.get_object()
        #print(oprating_object.first().ansKey)
        context['Tester'] = "TESTER STRING"
        return context

#"/functions/exam-detail/<EXAM_NAME>/update"
class exam_update_view(LoginRequiredMixin, UserPassesTestMixin,UpdateView):
    model = Exam
    form_class = exam_editor
    template_name = 'examUpdate.html'

    #only user who created can update
    def test_func(self):
        exam = self.get_object()
        if self.request.user == exam.user:
            return True
        return False

class exam_delete_view(LoginRequiredMixin, UserPassesTestMixin,DeleteView):
    model = Exam
    template_name = 'examDelete.html'
    success_url = '/functions/exams'

    #only user who created can delete
    def test_func(self):
        exam = self.get_object()
        if self.request.user == exam.user:
            return True
        return False

#"/functions/exam-detail/<EXAM_NAME>/inputs"
class multiple_inputs(FormView):
    template_name='eval_inputs.html'
    form_class = multiple_files_input
    success_url ='inputs'
    files_list =[]

    def post(self, request, *args, **kwargs):
        form_class = self.get_form_class()
        form = self.get_form(form_class)
        files = request.FILES.getlist('inputs')
        #context = super().get_context_data(**kwargs)
        files_list = []
        if form.is_valid():
            fs=FileSystemStorage(location=r'./functions/inputs/OMR_Files/MobileCameraBased/JE')
            for f in files:
                # Do something with each file.
                f_name = f.name
                f_extension = f.content_type.split('/')[1]
                self.files_list.append(f_name)
                fs.save(f_name,f)
            #print(context)
            return self.form_valid(form)
        else:
            return self.form_invalid(form)


    def request_page(request):
        print(request.GET)

    def get_context_data(self,*args,**kwargs):
        global Answers
        context = super().get_context_data(**kwargs)
        fs =FileSystemStorage(location=r'./functions/inputs/OMR_Files/MobileCameraBased/JE')
        #print(fs.listdir(path=r'./functions/inputs/OMR_Files/MobileCameraBased/JE'))
        #print("path: ", os.listdir(r'./functions/inputs/OMR_Files/MobileCameraBased/JE'))

        uploadedFile = "./media/exams/answer_key/tableExport.csv"
        if len(os.listdir(r'./functions/inputs/OMR_Files/MobileCameraBased/JE'))>0:
            context['files']=os.listdir(r'./functions/inputs/OMR_Files/MobileCameraBased/JE')
        else:
            context['files'] = ''
        context['files'].remove('gitkeep')
        # Add in a QuerySet of all the books
        obj = Exam.objects.filter(exam_name=self.kwargs['name'])
        Answers = extractAnswers(csvPath = obj[0].ansKey, imgPath = obj[0].ansKeyImg)

        context['Tester'] = "dsf"
        context['object'] = obj
        #print(obj[0].ansKey)
        return context


def delete_img(request, *args, **kwargs):
    fs=FileSystemStorage(location = r'./functions/inputs/OMR_Files/MobileCameraBased/JE')
    if kwargs['fname'] == "all":
        imgs = fs.listdir(path = r'.')[1]
        imgs.remove('gitkeep')
        if len(imgs) > 0:
            for file in imgs:
                fs.delete(file)
        return redirect('exam-detail-input',kwargs['exam'])
    fs.delete(kwargs['fname'])
    return redirect('exam-detail-input',kwargs['exam'])
#-----------------------------------------EVALUATION VIEWS-----------------------------------------

#EVALUATION VIEWS "/functions/eval"
class eval(ListView):
        template_name = 'eval_selection.html'
        context_object_name = 'exams'

        def get_queryset(self):
            #print(Exam.objects.filter(user=self.request.user)   )
            return Exam.objects.filter(user=self.request.user)


def ReportEval(request,*args,**kwargs):
    global Answers
    results_dir =FileSystemStorage(location=r'.')
    media_result_dir = FileSystemStorage(location=r'./media/output')

    a=r"./functions/inputs/OMR_Files/MobileCameraBased/JE"

    report_xl, report_pdf, report_csv, table = main(Answers, a, directory = True)

    to_be_copied = results_dir.open(report_xl,mode='rb')
    media_result_dir.save(report_xl,to_be_copied)

    to_be_copied = results_dir.open(report_pdf,mode='rb')
    media_result_dir.save(report_pdf,to_be_copied)

    to_be_copied = results_dir.open(report_csv,mode='rb')
    media_result_dir.save(report_csv,to_be_copied)
    print(report_pdf[1::])
    context ={
    'result_xl': report_xl,
    'result_pdf': report_pdf[1::],
    'result_csv': report_csv,
    'table': table
    }
    #print(context['result'])
    return render(request,'eval_report.html',context)
