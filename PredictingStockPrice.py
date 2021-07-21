import os
import multiprocessing as mp
import time
import math
#from itertools import Predicate
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dropout, Dense, LSTM 
#local
import CryptoTrainingModel as ctm

moneda  = ctm.TFMmodeler.moneda
metric = ctm.TFMmodeler.metric


'''Data Prep'''
cripto = []
'''Tabla de modelo '''
modelosCrypto = []

# Variable de la carrpeta de datos de crypto
data_directory = f"Ficheros Originales"+os.sep


def build_test_models( ticker: str):
    '''Build Model'''
    current_model = ctm.TFMmodeler(data_directory,ticker,moneda)
    ''' Test the Model Accuracy  on Existing Data '''
    current_model.testModel()

    modelosCrypto.append({"ticker":ticker, "model": current_model })

if __name__ == '__main__':
    
    poolSize = os.cpu_count()
    if (poolSize != None):
        poolSize =poolSize//2
    else:
        poolSize=1 # es màs efectivo coger solo la mitad de los procesadores disponible
    
    # Creamos un pool de tarea para cargar por lote a los modelos
    #with mp.Pool(processes=None) as pool:
    with mp.Pool(processes=poolSize) as pool:
        #Load Data     
        #Parseamos el repertorio por csv de cripto

        for fichero in os.listdir(data_directory):
            if fichero.endswith(f"{moneda}.csv"):
                cripto.append( fichero.split('-')[0])
            else:
                continue

        print(f"Tabla de Cripto: {cripto}")


        # Creamos una tabla de modelo, con un modelo propio a cada cripto
        #for ticker in cripto:
        pool.map(build_test_models,cripto)
        #process=mp.Process( build_test_models(moneda, data_directory, modelosCrypto, ticker) )
        #process.start()
        #process.join()
        pool.close()
        pool.join()

