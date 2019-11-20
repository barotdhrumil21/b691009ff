from django import forms
from .models import omr_templates,Exam




class upload_form(forms.ModelForm):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['pdf_name'].widget.attrs.update({'class':"form-control",
                                                     'placeholder':"template name",
                                                     'aria-label':"template",
                                                     'aria-describedby':"basic-addon2"})
        self.fields['pdf'].widget.attrs.update({'class':"custom-file-input",'id':"inputGroupFile04"})

    class Meta:
        model = omr_templates
        fields =[
        'pdf_name',
        'pdf'
        ]

class exam_builder(forms.ModelForm):

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.fields['exam_name'].widget.attrs.update({'class':"form-control",
                                                     'placeholder':"exam name",
                                                     'aria-label':"exam",
                                                     'aria-describedby':"basic-addon2"})

        self.fields['ansKey'].widget.attrs.update({'class':"custom-file-input",
                                                    'id':"file-upload"})
        self.fields['ansKeyImg'].widget.attrs.update({'class':"custom-file-input",
                                                    'id':"file-upload2"})

        self.fields['template'].widget.attrs.update({'class':"custom-select",
                                                     'id':"inputGroupSelect02"})

    class Meta:
        model = Exam
        fields =[
        'exam_name',
        'ansKey',
        'ansKeyImg',
        'template'
        ]

class exam_editor(forms.ModelForm):

    class Meta:
        model = Exam
        fields =[
        'exam_name',
        'ansKey',
        'ansKeyImg',
        'template'
        ]


class multiple_files_input(forms.Form):
    inputs = forms.FileField(widget=forms.ClearableFileInput(attrs={'multiple': True}))
