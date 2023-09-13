from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse, HttpResponseRedirect
import sys
from .forms import NameForm


class Entry(object):

    def __init__(self, key, val):
        self.key = key
        self.val = val

    def __unicode__(self):
        return u'%s:%s' % (self.key, self.val)

    def __str__(self):
        return u'%s:%s' % (self.key, self.val)


def index(request):
    context = {
        'entries': [Entry('v1', 'v1'), Entry('v2', 'v2')]
    }
    ret = render(request, 'my_app/index.html', context)
    return ret


def get_name(request):
    # if this is a POST request we need to process the form data
    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        form = NameForm(request.POST)
        # check whether it's valid:
        if form.is_valid():
            # process the data in form.cleaned_data as required
            # ...
            # redirect to a new URL:
            return HttpResponseRedirect('/thanks/')

    # if a GET (or any other method) we'll create a blank form
    else:
        form = NameForm(data={'your_name': 'unknown name'})

    return render(request, 'my_app/name.html', {'form': form})


def inherits(request):
    context = {}
    ret = render(request, 'my_app/inherits.html', context)
    return ret


def template_error(request):
    context = {
        'entries': [Entry('v1', 'v1'), Entry('v2', 'v2')]
    }

    ret = render(request, 'my_app/template_error.html', context)
    return ret


def template_error2(request):
    context = {}
    ret = render(request, 'my_app/template_error2.html', context)
    return ret


def no_var_error(request):
    context = {}
    ret = render(request, 'my_app/no_var_error.html', context)
    return ret
