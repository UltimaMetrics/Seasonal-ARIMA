# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 12:16:58 2023

@author: sigma
"""

import matplotlib.pyplot as plt
from matplotlib import pyplot
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from unitroot import stationarity

#####HMA Nokia#####
df=pd.read_csv(r'D:\2022 Job interviews\GFK\Forecast HMA.csv')
df=pd.DataFrame(df)
df

df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
#sp1 = df["Total"]
sp2=df["Smart"]
sp3=df["FP"]
#Nokia total
#sp1

#result1 = seasonal_decompose(sp1, model='additive',extrapolate_trend='freq')
#result1.plot()
#plt.show()

#Convert to seasonal data of sp1
#s1=result1.seasonal

#For d
#stationarity(sp1)
#Not stationary

#For p
#from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
#plot_pacf(sp1, lags =3)
#plt.show()

#For q
#plot_acf(sp1, lags =3)
#plt.show()

#Seasonal
#For D
#stationarity(s1)
#Stationary

#For P
#from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
#plot_pacf(s1, lags =3)
#plt.show()

#For Q
#plot_acf(s1, lags =3)
#plt.show()

#For M
#Set M = 4 because 4 quarters in one year

#Forecasting
#model1= SARIMAX(sp1, order=(1, 0, 0), seasonal_order=(0, 2, 1, 4))
#model_fit1 = model1.fit(disp=False)
#yhat1 = model_fit1.predict(len(sp1), len(sp1))
#print(yhat1)
#####Since I will take the sum of sp2 and sp3, just ignore the model from above!!!#####

#Nokia Smartphone
sp2

result2 = seasonal_decompose(sp2, model='additive',extrapolate_trend='freq')
result2.plot()
plt.show()

#Convert to seasonal data of sp2
s2=result2.seasonal

#For d
stationarity(sp2)
#Not stationary

#For p
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_pacf(sp2, lags =3)
plt.show()

#For q
plot_acf(sp2, lags =3)
plt.show()


#Seasonal
#For D
stationarity(s2)
#stationary

#For P
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_pacf(s2, lags =3)
plt.show()

#For Q
plot_acf(s2, lags =3)
plt.show()


#For M
#Set M = 4 because 4 quarters in one year


#Forecasting
model2= SARIMAX(sp2, order=(1, 0, 0), seasonal_order=(0, 0, 0, 4))
model_fit2 = model2.fit(disp=False)
yhat2 = model_fit2.predict(len(sp2), len(sp2))
print(yhat2)

#Nokia Feature Phone

sp3

result3 = seasonal_decompose(sp3, model='additive',extrapolate_trend='freq')
result3.plot()
plt.show()

#Convert to seasonal data of sp3
s3=result3.seasonal


#For d
stationarity(sp3)
#Not stationary

#For p
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_pacf(sp3, lags =3)
plt.show()

#For q
plot_acf(sp3, lags =3)
plt.show()

#Seasonal
#For D
stationarity(s3)
#Stationary

#For P
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_pacf(s3, lags =3)
plt.show()

#For Q
plot_acf(s3, lags =3)
plt.show()


#For M
#Set M = 4 because 4 quarters in one year


#Forecasting
model3= SARIMAX(sp3, order=(1, 0, 0), seasonal_order=(0, 0, 0, 4))
model_fit3 = model3.fit(disp=False)
yhat3 = model_fit3.predict(len(sp3), len(sp3))
print(yhat3)



#####CMA Mediatek#####
df4=pd.read_csv(r'D:\2022 Job interviews\GFK\Forecast CMA.csv')

df4=pd.DataFrame(df4)
df4


df4['Date'] = pd.to_datetime(df4['Date'])
df4.set_index('Date', inplace=True)

sp4=df4["Shipments"]
sp4

result4= seasonal_decompose(sp4, model='additive',extrapolate_trend='freq')
result4.plot()
plt.show()

#Convert to seasonal data of sp3
s4=result4.seasonal

#For d
stationarity(sp4)
#Stationary

#For p
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_pacf(sp4, lags =3)
plt.show()

#For q
plot_acf(sp4, lags =3)
plt.show()

#Seaonal
#For D
stationarity(s4)
#Stationary

#For P
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_pacf(s4, lags =3)
plt.show()

#For Q
plot_acf(s4, lags =3)
plt.show()

#For M
#Set M = 4 because 4 quarters in one year

#Forecasting
model4= SARIMAX(sp4, order=(0, 0, 0), seasonal_order=(0, 2, 1, 4))
model_fit4 = model4.fit(disp=False)
yhat4 = model_fit4.predict(len(sp4), len(sp4))
print(yhat4)


