import os
import glob
from django.http import HttpResponse, Http404
from django.shortcuts import render,redirect,get_object_or_404
#from django.core.urlresolvers import reverse
from django.views.generic import(
                                    CreateView,
                                    ListView,
                                    DetailView,
                                    FormView)
from django.core.files.storage import FileSystemStorage
from .forms import (upload_form,
                    exam_builder,
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
    print(obj.id)
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
#"/functions/add-exam"
class upload_exam_view(CreateView):
    model = Exam
    form_class =exam_builder
    template_name = 'examAdder.html'
    def form_valid(self,form):
        form.instance.user = self.request.user
        return super().form_valid(form)

#"/functions/exam-detail/<EXAM_NAME>/inputs"
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
        # Add in a QuerySet of all the books
        oprating_object = self.get_object()
        #print(oprating_object.first().ansKey)
        context['Tester'] = "TESTER STRING"
        return context

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
        files_list =[]
        if form.is_valid():
            fs=FileSystemStorage(location=r'.\functions\inputs\OMR_Files\MobileCameraBased\JE')
            for f in files:
                # Do something with each file.
                f_name=f.name
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
        context = super().get_context_data(**kwargs)
        fs =FileSystemStorage(location=r'.\functions\inputs\OMR_Files\MobileCameraBased\JE')
        #print(fs.listdir(path=r'.\functions\inputs\OMR_Files\MobileCameraBased\JE'))
        print("cwd: ", os.getcwd())
        context['files']=os.listdir(r'.\functions\inputs\OMR_Files\MobileCameraBased\JE')[1]
        print("cwd: ", os.listdir('.'))
        # Add in a QuerySet of all the books
        obj = Exam.objects.filter(exam_name=self.kwargs['name'])
        #print(obj.first)
        context['Tester'] = "dsf"
        context['object'] = obj
        #print(context['files'])
        return context

def delete_img(request,*args,**kwargs):
    #print("D E L E T E")
    #url = reverse(url_name, args = args)
    fs=FileSystemStorage(location=r'.\functions\inputs\OMR_Files\MobileCameraBased\JE')
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

#download csv file on make
def download(request, path):
    file_path = os.path.join(settings.MEDIA_ROOT, path)
    if os.path.exists(file_path):
        with open(file_path, 'rb') as fh:
            response = HttpResponse(fh.read())
            response['Content-Disposition'] = 'inline; filename=' + os.path.basename(file_path)
            return response
    raise Http404


def ReportEval(request,*args,**kwargs):
        results_dir =FileSystemStorage(location=r'.\functions\outputs\Results')
        media_result_dir = FileSystemStorage(location=r'.\media\output')

        a=r".\functions\inputs\OMR_Files\MobileCameraBased\JE"
        #l = [a]

        main(a, directory = True)

        list_of_files = glob.glob(r".\functions\outputs\Results\*") # * means all if need specific format then *.csv
        #print("lof: ", list_of_files)
        latest_file = max(list_of_files, key=os.path.getctime)
        report=latest_file.split('\\')[-1]

        to_be_copied = results_dir.open(report,mode='rb')
        media_result_dir.save(report,to_be_copied)

        context ={
        'result': report

        }
        print(context['result'])
        return render(request,'eval_report.html',context)
