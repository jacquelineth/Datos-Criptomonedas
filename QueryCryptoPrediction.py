import CryptoTrainingModel as ctm
import yfinance as yf
import pandas as pd
import os
import sys

carpeta_transition ="Datos_eval"+os.sep


''''Load List of available models'''
cripto = []
for fichero in os.listdir(ctm.TFMmodeler.model_directory):
    if fichero.endswith(f"{ctm.TFMmodeler.moneda}.h5"):
        cripto.append( fichero.split('-')[0])
    else:
        continue


'''Display and Choose Crypto'''
text =""
while not ( text in cripto) :
    text = input(f" Teclea una de las monedas :\n{cripto}\n=>")
    text = text.upper()


'''Retrieve latest data from Yahoo'''
ticker = yf.Ticker(f"{text}-{ctm.TFMmodeler.moneda}")
data = ticker.history(period="3mo", interval="1d",actions=False)
fullname = ticker.info

'''Cleaning Data'''
data = data.drop (['Open','High','Low', 'Volume'], axis=1)
     
#print(data.to_json())
data.tail(2)
#  TODO Date Format in Epoch (datetime.datetime.fromtimestamp(X/1000)).strftime("%Y-%m-%d")
#data.to_csv(carpeta_transition+f"{fullname}.json")


''' Run Crypto Model'''

model = ctm.TFMmodeler(directory="", pticker=text, moneda=ctm.TFMmodeler.moneda, loadOnly=True  )
model.predictNext(data)


'''Display Result'''