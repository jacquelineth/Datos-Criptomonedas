from math import e
import os
#from itertools import Predicate
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import datetime as dt

from sklearn.preprocessing import MinMaxScaler
import sklearn.model_selection as model_selection
from tensorflow.keras.models import Sequential 
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dropout, Dense, LSTM 


"""Default Values and constant"""
prediction_days = 60
class TFMmodeler:
    """Una clase  para llevar todo el modelo de una crimtomoneda, datos y kit de entranamiento """

    # Propriedades estatica 
    model_directory=f"modelos"+os.sep
    test_directory = f"Datos_Pruebas"+os.sep
    moneda = "USD"
    metric = "Close"

    def __init__(self, directory: str , pticker: str, moneda: str, store=True, loadOnly=False):
        self.ticker = pticker
        self.scaler = MinMaxScaler(feature_range=(0,1))

        # Solo carga un modelo calculado
        if loadOnly:
            if self.loadModel() :
                return 


        self.moneda = moneda
        self.directory = directory

       
        rawdata = pd.read_csv(directory+f"{pticker}-{moneda}.csv") 
        #Drop Null and NAN row
        rawdata = rawdata.dropna(axis=0 )
        #Drop  unused columns
        rawdata = rawdata.drop (['Open','High','Low', 'Volume','Adj Close'], axis=1)
        #Convertir fecha en tipo datetime
        rawdata['Date'] =  pd.to_datetime(rawdata['Date'], format='%Y-%m-%d')

        #Partimos la table de datos y guardamos 25% para probar luego
        self.data, self.test_data = model_selection.train_test_split(rawdata, test_size=0.25, stratify=None, shuffle=False)
        # Prepare Data
        
        # De la columna Close, reducimos a -1,1 
        scaled_data = self.scaler.fit_transform (self.data['Close'].values.reshape(-1,1))

        # Carga el modelo si ya existe
        if self.loadModel():
            #print(f"Loading stored Model for {self.ticker} with data={self.data.info()}")
            print(f"Loading stored Model for {self.ticker}")
            store=False
            return 
        else :
            print(f"Computing  Model for {self.ticker}")

        """Datos  de  entrenamiento"""

        x_train = []
        y_train = []

        # Hacemos una lista un dia (y ) y otra de los 60 dias anteriores
        for x in range(prediction_days,len(scaled_data)):
            x_train.append(scaled_data[x-prediction_days:x,0])
            y_train.append(scaled_data[x,0])

        #transformamos en array de numpay
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1 ))    

        # Build The Model
        self.model = Sequential()

        self.model.add(LSTM(units=64, return_sequences=True, input_shape=(x_train.shape[1],1)))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=64, return_sequences=False, input_shape=(x_train.shape[1],1)) )
        self.model.add(Dropout(0.2))
        #self.model.add(LSTM(units=50))
        self.model.add(Dropout(0.2))

        self.model.add(Dense(32,kernel_initializer="uniform",activation='relu'))  

        self.model.add(Dense(units=1)) # Prediciton of the next closing

        #self.model.compile(optimizer='adam',loss='mean_squared_error')
        self.model.compile(optimizer='adam',loss='mae')
        history=self.model.fit(x_train,y_train, epochs=20, batch_size=32)

        if store:
            self.saveModel()
            self.saveHistory(history)
    # return self.model



    """Serialization"""
    
    def saveModel(self):
        self.model.save(self.model_directory+f"{self.ticker}-{self.moneda}.h5")

    def loadModel(self) -> bool:
        if os.path.exists(self.model_directory+f"{self.ticker}-{self.moneda}.h5"):
            self.model= load_model(self.model_directory+f"{self.ticker}-{self.moneda}.h5")
            return True
        else:
            return False


    def saveHistory(self,history):
        # Dibujamos nuestra gráfica de apredizaje
 
        epochs_plot = range(1, len(history.history['loss'])+1, 1)

        plt.figure(); 
        plt.plot(epochs_plot, history.history['loss'], 'r--', label = 'Evolución del loss entrenamiento')
        plt.title('Performance de mi red neuronal')
        plt.ylabel('loss')
        plt.xlabel('Época')
        plt.legend()
        plt.savefig(self.model_directory+f"{self.ticker}-{self.moneda}_train.png")
 

    """ Test the Model """            
    
    def testModel(self ): 


        ###TODO :Get better  Test data
        exceptData = self.test_data # Guardamos una referncia por si peta
        try: # Intenta recuperar un fichero de prueba propio 
            self.test_data = pd.read_csv(self.test_directory+f"{self.ticker}-{self.moneda}.csv") 
            self.test_data = self.test_data.drop (['Open','High','Low', 'Volume','Adj Close'], axis=1)
            self.test_data = self.test_data.dropna(axis=0 )
            self.test_data['Date'] =  pd.to_datetime(self.test_data['Date'], format='%Y-%m-%d')
            exceptData = [] # limpiamos la memoria
        except FileNotFoundError : # o usamos los datos apartados antes
            print(f"Using split Test Data for {self.ticker}")
            self.test_data = exceptData # recuperamos los antiguos data 
            print(f"{self.ticker}===>\n"+"\n\n" )
            self.test_data.info()
        actual_prices =  self.test_data['Close'].values
        # Vamos a guardar solo las columnas de cierre , y añadir los datos anteriores
        total_dataset = pd.concat((self.data['Close'], self.test_data['Close']), axis=0)
        # asi podemos empezar a contar los 60 dias tambien para los primeros datos de prueba
        model_inputs = total_dataset[len(total_dataset)-len(self.test_data)- prediction_days:].values

        #limitamos los valores 
        model_inputs =  model_inputs.reshape(-1,1)
        model_inputs = self.scaler.transform(model_inputs)

        # Make Predictions on  Test Data

        x_test = []

        for x in range(prediction_days,len(model_inputs)):
            x_test.append(model_inputs[x-prediction_days:x, 0])

        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        predicted_prices = self.model.predict(x_test)
        predicted_prices = self.scaler.inverse_transform(predicted_prices)
        

        # Plot The Test Predicitons
        plt.figure()
        plt.plot(actual_prices, color= "black", label = f"Actual {self.ticker} price" )
        plt.plot(predicted_prices, color = "red", label = f"Predicted {self.ticker} price")
        plt.title(f"{self.ticker} Share Price")
        plt.xlabel('Time')
        plt.ylabel(f'{self.ticker} Share Price' )
        plt.legend()
        plt.savefig(self.model_directory+f"{self.ticker}-{self.moneda}.png")
        
        print(f"Generated plot for {self.ticker}")

        # Predict Next Day 
        real_data = [model_inputs[len(model_inputs)  - prediction_days:len(model_inputs), 0]] 
        real_data = np.array(real_data)
        real_data = np.reshape(real_data, (real_data.shape[0],real_data.shape[1],1))


        
        prediction = self.model.predict(real_data)
        prediction = self.scaler.inverse_transform(prediction)
        print (f"Prediction: {prediction}")
        



        return predicted_prices

    def predictNext(self, data):
        #limpiamos los valores null
        data = data.dropna(axis=0 )
        #Quitamos las fechas
        data = data.values
        #Ajustamos el fit
        self.scaler.fit(data)
        data =  data.reshape(-1,1)
        data = self.scaler.transform(data)


        real_data = [data[len(data)  - prediction_days:len(data), 0]] 
        real_data =  np.array(real_data)
        real_data = np.reshape(real_data, (real_data.shape[0],real_data.shape[1],1))


        
        prediction = self.model.predict(real_data)
        prediction = self.scaler.inverse_transform(prediction)
        print (f"Prediction: {prediction}")  
        return prediction
