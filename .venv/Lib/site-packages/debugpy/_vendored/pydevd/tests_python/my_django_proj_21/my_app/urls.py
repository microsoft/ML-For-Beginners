try:
    from django.conf.urls import url
except ImportError:
    from django.urls import re_path as url

from . import views

urlpatterns = [
    url(r'^$', views.index, name='index'),
    url(r'^name$', views.get_name, name='name'),
    url(r'^template_error2$', views.template_error2, name='template_error2'),
    url(r'^template_error$', views.template_error, name='template_error'),
    url(r'^inherits$', views.inherits, name='inherits'),
    url(r'^no_var_error$', views.no_var_error, name='no_var_error'),
]
