from django.http import HttpResponse
from django.shortcuts import render,redirect
from .forms import user_signup_form,user_login_form
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login ,logout
from django.contrib.auth.decorators import login_required
# Create your views here.

@login_required
def home_view(request,*args,**kwargs):
    return render(request,'home.html',{})

def about_view(request,*args,**kwargs):
    return render(request,'about.html',{})


def register_view(request,*args,**kwargs):
    form = user_signup_form(request.POST or None)
    if form.is_valid():
        data=form.cleaned_data
        User.objects.create_user(data['username'],data['email'],data['password'])
        return redirect('home-page')
    context ={
     'form':form
    }
    return render(request,'register.html',context)

def landing_view(request,*args,**kwargs):
    form = user_login_form(request.POST or None)
    context ={
     'form':form
    }
    if form.is_valid():
        data = form.cleaned_data
        #print(data)
        user = authenticate(request, username=data['username'], password=data['password'])
        if user is not None:
            login(request,user)
            return redirect('home-page')
        else:
            context['errors']="username or password is incorrect! "
    return render(request,'landingPage.html',context)


def logout_view(request,*args,**kwargs):
    logout(request)
    return redirect('welcome-page')
