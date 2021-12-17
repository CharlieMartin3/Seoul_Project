from django.urls import path

from . import views

urlpatterns = [

path('accueil', views.accueil, name='accueil'),
path('visu', views.visu, name='visu'),
path('prediction', views.prediction, name='prediction'),
path('prediction2', views.prediction2, name='prediction2'),
path('predictionSemaine', views.predictionSemaine, name='predictionSemaine'),
path('prediction2Semaine', views.prediction2Semaine, name='prediction2Semaine'),


]