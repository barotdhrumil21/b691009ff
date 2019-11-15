from django import forms
from django.contrib.auth.models import User


class user_signup_form(forms.ModelForm):

    username = forms.CharField(label='',
                               widget=forms.TextInput(attrs={
                                                        "type":"text" ,
                                                        "class":"input" ,
                                                        "placeholder":"Username"
                                                    }
                                                )
                                            )
    password = forms.CharField(label='',
                               widget=forms.TextInput(attrs={
                                                        "type":"password" ,
                                                        "class":"input" ,
                                                        "placeholder":"Password"
                                                    }
                                                )
                                            )
    email = forms.CharField(label='',
                               widget=forms.TextInput(attrs={
                                                        "type":"text" ,
                                                        "class":"input" ,
                                                        "placeholder":"Email"
                                                    }
                                                )
                                            )
    class Meta:
        model = User
        fields =[
        'username',
        'password',
        'email'
        ]

class user_login_form(forms.Form):
        username = forms.CharField(label='',
                                   widget=forms.TextInput(attrs={
                                                            "type":"text" ,
                                                            "class":"input" ,
                                                            "placeholder":"Username"
                                                        }
                                                    )
                                                )
        password = forms.CharField(label='',
                                   widget=forms.TextInput(attrs={
                                                            "type":"password" ,
                                                            "class":"input" ,
                                                            "placeholder":"Password"
                                                        }
                                                    )
                                                )
