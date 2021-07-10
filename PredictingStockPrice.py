import os
#from itertools import Predicate
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dropout, Dense, LSTM 
#local
import CryptoTrainingModel as ctm

moneda  = "USD"
metric = "Close"


'''Data Prep'''
cripto = []

#Parseamos el repertorio por csv de cripto
data_directory = f"Ficheros Originales"+os.sep




for fichero in os.listdir(data_directory):
    if fichero.endswith(f"{moneda}.csv"):
        cripto.append( fichero.split('-')[0])
    else:
        continue




print(f"Tabla de Cripto: {cripto}")


#Load Data
company = 'BTC'

start = dt.datetime(2012,1,1)
end = dt.datetime(2020,1,1)

'''Tabla de modelo '''
modelosCrypto = []

for ticker in cripto:
    modelosCrypto.append({"ticker":ticker, "model": ctm.TFMmodeler(data_directory,ticker,moneda) })

''' Test the Model Accuracy  on Existing Data '''
for instance  in modelosCrypto:    
    instance["model"].testModel()




