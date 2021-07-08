import os
from itertools import Predicate
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dropout, Dense, LSTM 


'''Default Values and constant'''
prediction_day = 60
scaler = MinMaxScaler(feature_range=(0,1))

class TFMmodeler:
    def __init__(self,str directory, str pticker, str moneda):
        self.ticker = pticker
        self.moneda = moneda
        self.directory = directory
        data = pd.read_csv(directory+f"{pticker}-f{moneda}.csv") 

        # Prepare Data
        
        scaled_data = scaler.fit_transform (data['Close'].values.reshape(-1,1))

        x_train = []
        y_train = []

        for x in range(prediction_day,len(scaled_data)):
            x_train.append(scaled_data[x-prediction_day:x,0])
            y_train.append(scaled_data[x,0])

        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train,(x_train.shape[0],x_train.shape[1],1) ))    

        # Build The Model
        self.model = Sequential()

        self.model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(unit=50, return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(unit=50))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(units=1)) # Prediciton of the next closing

        self.model.compile(optimizer='adam',loss='mean_squared_error')
        self.model.fit(x_train,y_train, epochs=25, batch_size=32)
        return self.model

    def saveModel(self):
        ###TODO persist to file
    
    def loadModel(self):
        ###TODO retrieve from file
    
    def testModel(dt.datetime pStart, dt.datetime pEnd):
        # Load Test Data
        test_start= pstart
        test_end= pEnd

        ###TODO :Get Test data
        test_data = web.DataReader(company, 'yahoo', test_start, test_end)
        actual_prices =  test_data['Close'].values

        total_dataset = pd.concat((data['Close'], test_data['Close']))

        model_inputs = total_dataset[len(total_dataset)-len(test_data)- prediction_day:].values
        model_inputs =  model_inputs.reshape(-1,1)
        model_inputs = scaler.transform(model_inputs)

        # Make Predictions on  Test Data

        x_test = []

        for x in range(prediction_day,len (model_inputs)):
            x_test.append(model_inputs[x-prediction_day:x, 0])

        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        predicted_prices = model.predict(x_test)
        predicted_prices = scaler.inverse_transform(predicted_prices)

        