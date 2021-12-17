from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from random import *
from io import StringIO
from datetime import date
from datetime import datetime
import calendar
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler


#traitement des données :

sbdata = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00560/SeoulBikeData.csv', encoding= 'unicode_escape')

dates = pd.to_datetime(sbdata['Date'], dayfirst=True)
sbdata['jour'] = [dates[i].day_name() for i in range(len(dates))]
sbdata['week end'] = [ 'yes' if sbdata.iloc[i]['jour'] in ('Saturday','Sunday') else 'non'  for i in range(len(sbdata))]
sbdata['mois'] = [dates[i].month_name() for i in range(len(dates))]

sbdata_corr = sbdata.corr()
sbdata_temperature = pd.pivot_table(sbdata,values='Rented Bike Count',index='Temperature(°C)',aggfunc= np.mean)
sbdata_saison_hours = pd.pivot_table(sbdata,values='Rented Bike Count',index='Hour',columns='Seasons',aggfunc=sum)





# fonction qui va permettre de coloré le barplot en fonction de la valeur de la barre (plus lisible)
def colors_from_values(values, palette_name):
    # normalize the values to range [0, 1]
    normalized = (values - min(values)) / (max(values) - min(values))
    # convert to indices
    indices = np.round(normalized * (len(values) - 1)).astype(np.int32)
    # use the indices to get the colors
    palette = sns.color_palette(palette_name, len(values))
    return np.array(palette).take(indices, axis=0)

def filter_migration_pend(h):
    if(6<h<11) or (16<h<22):
        return 'Yes'
    else:
        return 'No'


def filter_4parties(h):
    if(h < 6):
        return 'nuit'
    elif (h < 12):
        return 'matin'
    elif (h < 18):
        return 'apres-midi'
    else :
      return 'soiree'




def return_graph(type_, data_,  width_, height_, x_=None, y_=None, hue_=None, colormap = None, style_ =None):

	fig, ax = plt.subplots(figsize=(width_,height_))

	if 'boxplot' == type_ :
		sns.boxplot(data=data_, x=x_, y=y_)

	elif type_ =='scatterplot':
		if hue_==None:
			sns.scatterplot(x=data_.index,y=y_, data=data_)
		else :
			sns.scatterplot(x=data_.index,y=y_, data=data_, hue=hue_)

	elif type_=='lineplot':
		sns.lineplot(data=data_,x=x_,y=y_,hue=hue_, ci=None)

	elif type_ == 'pie':
		ax.pie(x=y_,labels=x_,autopct='%1.1f%%')
		plt.legend(loc="lower left")

	elif type_=='barplot':
		sns.barplot(data=data_,x=x_,y=y_)


	else :
		sns.heatmap(data=data_,cmap=colormap) 

	imgdata = StringIO()
	fig.savefig(imgdata, format='svg')
	imgdata.seek(0)

	data = imgdata.getvalue()
	return data

def return_multiple_graph(type_, data_, x_, y_,nbplotsx_,nbplotsy_,sharex_, width_, height_, suptitle_):
	fig, axs = plt.subplots(nbplotsx_,nbplotsy_,figsize=(width_, height_),sharex=sharex_)
	fig.suptitle(suptitle_)
	index = 0

	if type_=="boxplot":
		for i in range(nbplotsx_):
			for j in range(nbplotsy_):
				sns.boxplot(ax=axs[i, j], data=sbdata, x=x_[index], y=y_[index])
				index = index+1

	elif type_ == "barplot":
		for i in range(nbplotsx_):
			for j in range(nbplotsy_):
				sns.barplot(ax=axs[i, j], data=data_, x=x_[index], y=y_[index], palette=colors_from_values(data_[y_[index]], "YlOrRd"))
				index = index+1


	imgdata = StringIO()
	fig.savefig(imgdata, format='svg')
	imgdata.seek(0)

	data = imgdata.getvalue()
	return data


# Create your views here.
def accueil(request):
	template = loader.get_template('appli1/accueil.html')
	contexte = {}
	return HttpResponse(template.render(contexte,request));


def visu(request):
	template = loader.get_template('appli1/visu.html')

	# Première partie : presentation du dataset
	head = sbdata.head().to_html(index_names=False, escape=False)
	describe = sbdata.describe().to_html(index_names=False, escape=False)
	correlation = return_graph('heatmap',sbdata_corr, width_=7,height_=7, colormap='YlGnBu')

	# deuxième partie : SEASONS
	seasonsRentedBike = return_graph('boxplot', sbdata,5,5,'Seasons','Rented Bike Count')

	xMeteoStatBySeason = ['Seasons','Seasons','Seasons','Seasons']
	yMeteoStatBySeason = ['Humidity(%)','Temperature(°C)','Wind speed (m/s)','Dew point temperature(°C)']
	meteoStatBySeason = return_multiple_graph('boxplot',sbdata,xMeteoStatBySeason,yMeteoStatBySeason,2,2,True,18,10,'Meteo stats by seasons')

	xhoursbyseason = [sbdata_saison_hours.index, sbdata_saison_hours.index, sbdata_saison_hours.index, sbdata_saison_hours.index]
	yhoursbyseason = ['Autumn','Spring','Summer','Winter']
	hoursbyseason = return_multiple_graph('barplot',sbdata_saison_hours,xhoursbyseason,yhoursbyseason,2,2,True,18,10,'Total hours stats by seasons')
	# Troisème partie : HOURS
	image = return_graph('boxplot', sbdata,10,5, 'Hour','Rented Bike Count')

	#hours1 :
	sbdata["Migration pendulaire"] = sbdata["Hour"].apply(lambda x:filter_migration_pend(int(x)))
	pre_hours1 = sbdata[['Temperature(°C)','Humidity(%)',
                      'Wind speed (m/s)','Dew point temperature(°C)','Solar Radiation (MJ/m2)',
                     'Rainfall(mm)','Snowfall (cm)','Migration pendulaire']]
	pre_hours2 = sbdata[['Rented Bike Count','Migration pendulaire']]
	hours_1 =  pre_hours1.groupby(['Migration pendulaire']).mean()
	hours_1['Rented Bike Count'] = pre_hours2.groupby(['Migration pendulaire']).sum()
	hours_1['RBC%'] = (hours_1['Rented Bike Count']/hours_1['Rented Bike Count'].sum())*100
	migrationpendulaire = hours_1.to_html(index_names=False, escape=False)

	#hours2 :
	sbdata["Partie_de_la_journee"] = sbdata["Hour"].apply(lambda x:filter_4parties(int(x)))
	pre_hours1 = sbdata[['Temperature(°C)','Humidity(%)',
                      'Wind speed (m/s)','Dew point temperature(°C)','Solar Radiation (MJ/m2)',
                     'Rainfall(mm)','Snowfall (cm)','Partie_de_la_journee']]
	pre_hours2 = sbdata[['Rented Bike Count','Partie_de_la_journee']]
	hours_2 =  pre_hours1.groupby(['Partie_de_la_journee']).mean()
	hours_2['Rented Bike Count'] = pre_hours2.groupby(['Partie_de_la_journee']).sum()
	hours_2['RBC%'] = (hours_2['Rented Bike Count']/hours_2['Rented Bike Count'].sum())*100
	hours_2.sort_values('RBC%',ascending=False)

	partiesjournees = return_graph('pie', hours_2, 7,7,x_=hours_2.index,y_=hours_2['RBC%'])


	# QUATRIEME partie : DAY
	jour = return_graph('boxplot', sbdata,7,7, 'jour','Rented Bike Count')
	barjour = return_graph('barplot',sbdata,7,7,'jour','Rented Bike Count')

	weekend = return_graph('boxplot', sbdata,5,5, 'week end','Rented Bike Count')

	weekdayline = return_graph('lineplot',sbdata,10,5,x_='Hour',y_='Rented Bike Count', hue_='jour')


	# 6eme partie : MOIS
	mois= return_graph('boxplot', sbdata,12,7, 'mois','Rented Bike Count')


	# 6eme partie : Donnees meteo
	temperature = return_graph('scatterplot', sbdata_temperature, 15,8, y_='Rented Bike Count')

	#temperature2 = return_graph('scatterplot', sbdata_temperature, 15,8, y_='Rented Bike Count')
	


	contexte = {'tab_head':head,'tab_desc':describe,'correlation' : correlation , 
	'image': image, 'meteoBySeason': meteoStatBySeason,'hoursbyseason':hoursbyseason, 'seasonsRentedBike':seasonsRentedBike, 'migrationpendulaire':migrationpendulaire, 'partiesjournees':partiesjournees,
	 'jour':jour,'barjour':barjour, 'weekend':weekend,'weekdayline':weekdayline,'mois':mois, 'temperature': temperature}
	return HttpResponse(template.render(contexte,request));





### PASSONS AUX PREDICTIONS :

sbdatapred = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00560/SeoulBikeData.csv', encoding= 'unicode_escape')

## Traitement des donnees :

df = pd.DataFrame(sbdatapred.copy())
del df['Dew point temperature(°C)']
#change le format de la date et la passe en index
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
df.set_index('Date', inplace = True)
df.rename(columns={'Rented Bike Count': 'Y'}, inplace=True)

#on commence la dataset au premier lundi
df = df[datetime(2017,12,4):datetime(2017,12,4)+timedelta(7*4*7)]
#on ajoute du lag à 1h, un jour, et une semaine
df["lag_1"] = df['Y'].shift(1)
df["lag_24"] = df['Y'].shift(24)
df["lag_168"] = df['Y'].shift(24*7)

df['Holiday'] = [1 if x=='Holiday' else 0 for x in df['Holiday'] ]
df['Functioning Day'] = [1 if x=='Yes' else 0 for x in df['Functioning Day'] ]
df['Weekday'] = pd.Categorical(pd.Series(df.index).dt.day_name(), categories=['Monday', 'Tuesday', 
             'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], ordered=True)
df['Weekend'] = (df['Weekday'].cat.codes >= 5).astype(int)
#df['Hour'] = pd.Series(pd.Series(df.index).dt.hour.values, index= df.index)

#les données temporelles sont cycliques, on peut donc les insérer avec leur sin et cos
df['Hour_sin'] = np.sin(df['Hour']*(2.*np.pi/24))
df['Hour_cos'] = np.cos(df['Hour']*(2.*np.pi/24))
#df[["Hour"]] = scaler.fit_transform(df[["Hour"]])
df['Weekday_sin'] = np.sin(df['Weekday'].cat.codes*(2.*np.pi/7))
df['Weekday_cos'] = np.cos(df['Weekday'].cat.codes*(2.*np.pi/7))
df.drop(["Hour", 'Seasons', 'Weekday', 'Humidity(%)','Visibility (10m)'], axis=1, inplace=True)

#on enlève la première semaine pour éviter les NaN des lags
df = df.iloc[24*7:]

## Creation du modèle : 
X = df.iloc[:, 1:].values
Y = df.iloc[:, [0]].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

# Fit regression model
regr = RandomForestRegressor(max_depth=12, random_state=0).fit(X_train, Y_train)



def prediction(request):
	template = loader.get_template('appli1/prediction.html')
	contexte = {}
	return HttpResponse(template.render(contexte,request));


def prediction2(request):
	template = loader.get_template('appli1/prediction2.html')
	

	# on prend les parametres pour effectuer la prediction :
	#date_ = ("10/12/2017")
	date_ = datetime.strptime(request.POST["date"], '%d/%m/%Y')
	weekday_ = date_.weekday()
	weekend_ = 0
	if weekday_>=5 :
		weekend_ = 1

	Weekday_sin = np.sin(weekday_*(2.*np.pi/7))
	Weekday_cos = np.cos(weekday_*(2.*np.pi/7))

	hour_ = int(request.POST["hour"])
	hour_sin = np.sin(hour_*(2.*np.pi/24))
	hour_cos = np.cos(hour_*(2.*np.pi/24))
	temperature_ = float(request.POST["temperature"])
	windspeed_ = float(request.POST["windspeed"])
	solarradiation_ = float(request.POST["solarradiation"])
	rainfall_ = float(request.POST["rainfall"])
	snowfall_ = float(request.POST["snowfall"])
	holiday_ = int(request.POST["holiday"])
	functioningday_ = 1
	lag1_ = int(request.POST["lag1"])
	lag10_ = int(request.POST["lag10"])
	lag100_ = int(request.POST["lag100"])


	#np.array([[0.000000, -2.5, 3.4, 0.0, 0, 1, 148.0, 326.0, 285.0, 0]])
	x_prediction = regr.predict(np.array([[temperature_,windspeed_,solarradiation_,rainfall_,snowfall_,
		holiday_, functioningday_,lag1_,lag10_,lag100_, weekend_, hour_sin,hour_cos,Weekday_sin,Weekday_cos]]))
	xpred = round(x_prediction[0])

	jour_ = ''
	if weekday_ == 0:
		jour_= 'lundi'
	elif weekday_==1:
		jour_ = 'mardi'
	elif weekday_==2:
		jour_ = 'mercredi'
	elif weekday_==3:
		jour_ = 'jeudi'
	elif weekday_==4:
		jour_ = 'vendredi'
	elif weekday_==1:
		jour_ = 'samedi'
	else :
		jour_='dimanche'

	vacances = 'oui'
	if holiday_ == 0:
		vacances='non'
	fonctionjour = 'oui'
	if functioningday_ ==0:
		fonctionjour = 'non'

	d_ = str(request.POST["date"]) + "/"+str(request.POST["hour"])
	datereturn = datetime.strptime(d_, '%d/%m/%Y/%H')




	contexte = {'prediction':xpred, 'hour':hour_,'date':datereturn,'jour':jour_,
	'lag1':lag1_,'lag10':lag10_,'lag100':lag100_, 'windspeed':windspeed_,
	'solarradiation':solarradiation_,'temperature':temperature_,
	'vacances':vacances,'fonctionjour':fonctionjour}

	return HttpResponse(template.render(contexte,request));



    








### PARTIE PREDICTION UNE SEMAINE EN AVANCE ###


sbdatapred = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00560/SeoulBikeData.csv', encoding= 'unicode_escape')

## Traitement des donnees :

df2 = pd.DataFrame(sbdatapred.copy())


#ajout d'une colonne avec la moyenne de la saison correspondante
TW = df2[df2['Seasons']=="Winter"]['Temperature(°C)'].mean()
TSP = df2[df2['Seasons']=="Spring"]['Temperature(°C)'].mean()
TS = df2[df2['Seasons']=="Summer"]['Temperature(°C)'].mean()
TA = df2[df2['Seasons']=="Autumn"]['Temperature(°C)'].mean()
df2.loc[df2['Seasons'] =='Winter', 'TemperatureM'] = TW 
df2.loc[df2['Seasons'] =='Spring', 'TemperatureM'] = TSP
df2.loc[df2['Seasons'] =='Summer', 'TemperatureM'] = TS
df2.loc[df2['Seasons'] =='Autumn', 'TemperatureM'] = TA

index = df2[df2['Functioning Day']==0].index
df2.drop(index, inplace=True)

del df2['Dew point temperature(°C)']
#change le format de la date et la passe en index
df2['Date'] = pd.to_datetime(df2['Date'], format='%d/%m/%Y')
df2.set_index('Date', inplace = True)
df2.rename(columns={'Rented Bike Count': 'Y'}, inplace=True)

#on commence la dataset au premier lundi
df2 = df2[datetime(2017,12,4):datetime(2017,12,4)+timedelta(7*4*7)]
#on ajoute du lag à une semaine
df2["lag_168"] = df2['Y'].shift(24*7)

df2['Holiday'] = [1 if x=='Holiday' else 0 for x in df2['Holiday'] ]
df2['Functioning Day'] = [1 if x=='Yes' else 0 for x in df2['Functioning Day'] ]
df2['Weekday'] = pd.Categorical(pd.Series(df2.index).dt.day_name(), categories=['Monday', 'Tuesday', 
             'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], ordered=True)
df2['Weekend'] = (df2['Weekday'].cat.codes >= 5).astype(int)
#df['Hour'] = pd.Series(pd.Series(df.index).dt.hour.values, index= df.index)

#les données temporelles sont cycliques, on peut donc les insérer avec leur sin et cos
df2['Hour_sin'] = np.sin(df2['Hour']*(2.*np.pi/24))
df2['Hour_cos'] = np.cos(df2['Hour']*(2.*np.pi/24))
#df[["Hour"]] = scaler.fit_transform(df[["Hour"]])
df2['Weekday_sin'] = np.sin(df2['Weekday'].cat.codes*(2.*np.pi/7))
df2['Weekday_cos'] = np.cos(df2['Weekday'].cat.codes*(2.*np.pi/7))
df2.drop([ 'Seasons', 'Weekday', 'Humidity(%)','Visibility (10m)'], axis=1, inplace=True)

#on enlève la première semaine pour éviter les NaN des lags
df2 = df2.iloc[24*7:]
df2.iloc[0]

## Creation du modèle : 
X = df2.iloc[:, 1:].values
Y = df2.iloc[:, [0]].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

# Fit regression model
regrSemaine = RandomForestRegressor(max_depth=12, random_state=0).fit(X_train, Y_train)







def predictionSemaine(request):
	template = loader.get_template('appli1/predictionSemaine.html')
	contexte = {}
	return HttpResponse(template.render(contexte,request));


def prediction2Semaine(request):
	template = loader.get_template('appli1/prediction2Semaine.html')
	

	# on prend les parametres pour effectuer la prediction :
	#date_ = ("10/12/2017")
	date_ = datetime.strptime(request.POST["date"], '%d/%m/%Y')
	weekday_ = date_.weekday()
	weekend_ = 0
	if weekday_>=5 :
		weekend_ = 1

	Weekday_sin = np.sin(weekday_*(2.*np.pi/7))
	Weekday_cos = np.cos(weekday_*(2.*np.pi/7))

	hour_ = int(request.POST["hour"])
	hour_sin = np.sin(hour_*(2.*np.pi/24))
	hour_cos = np.cos(hour_*(2.*np.pi/24))
	temperature_ = float(request.POST["temperature"])
	windspeed_ = float(request.POST["windspeed"])
	solarradiation_ = float(request.POST["solarradiation"])
	rainfall_ = float(request.POST["rainfall"])
	snowfall_ = float(request.POST["snowfall"])
	holiday_ = int(request.POST["holiday"])
	functioningday_ = 1
	lag100_ = int(request.POST["lag100"])

	temperatureM_ = TW

	doy = date_.timetuple().tm_yday

	# "day of year" ranges for the northern hemisphere
	spring = range(80, 172)
	summer = range(172, 264)
	autumn = range(264, 355)
	# winter = everything else

	if doy in spring:
  		temperatureM_ = TSP
	elif doy in summer:
  		temperatureM_ = TS
	elif doy in autumn:
  		temperatureM_ = TA

	

	#np.array([[0.000000, -2.5, 3.4, 0.0, 0, 1, 148.0, 326.0, 285.0, 0]])
	x_prediction = regrSemaine.predict(np.array([[hour_,temperature_,windspeed_,solarradiation_,rainfall_,snowfall_,
		holiday_, functioningday_,temperatureM_,lag100_, weekend_, hour_sin,hour_cos,Weekday_sin,Weekday_cos]]))
	xpred = round(x_prediction[0])

	jour_ = ''
	if weekday_ == 0:
		jour_= 'lundi'
	elif weekday_==1:
		jour_ = 'mardi'
	elif weekday_==2:
		jour_ = 'mercredi'
	elif weekday_==3:
		jour_ = 'jeudi'
	elif weekday_==4:
		jour_ = 'vendredi'
	elif weekday_==1:
		jour_ = 'samedi'
	else :
		jour_='dimanche'

	vacances = 'oui'
	if holiday_ == 0:
		vacances='non'
	fonctionjour = 'oui'
	if functioningday_ ==0:
		fonctionjour = 'non'

	d_ = str(request.POST["date"]) + "/"+str(request.POST["hour"])
	datereturn = datetime.strptime(d_, '%d/%m/%Y/%H')




	contexte = {'prediction':xpred, 'hour':hour_,'date':datereturn,'jour':jour_,
	'lag100':lag100_, 'windspeed':windspeed_,
	'solarradiation':solarradiation_,'temperature':temperature_,
	'vacances':vacances,'fonctionjour':fonctionjour}

	return HttpResponse(template.render(contexte,request));




