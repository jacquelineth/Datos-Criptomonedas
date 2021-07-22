
import os 

import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
import datetime as dt

from seaborn.matrix import heatmap

moneda  = "USD"
metric = "Close"


'''Data Prep'''
cripto = []

#Parseamos el repertorio por csv de cripto
directory = f"Ficheros Originales"+os.sep
for fichero in os.listdir(directory):
    if fichero.endswith(f"{moneda}.csv"):
        cripto.append( fichero.split('-')[0])
    else:
        continue




print(f"Tabla de Cripto {cripto}")

colnames = []

primero  = True 

for ticker in cripto :

    data = pd.read_csv(directory+f"{ticker}-{moneda}.csv")
    data = data.dropna(axis=0 )
    if primero: 
        combined =  data[[metric]].copy ()
        colnames.append(ticker)
        combined.columns =  colnames
        primero = False
    else:
        combined  = combined.join(data[metric])
        colnames.append(ticker)
        combined.columns = colnames  


plt.yscale('log')

mitad = len(cripto)//2


plt.subplot(2,1,1)
for ticker in cripto[mitad:]:
    plt.plot(combined[ticker], label=ticker)
    plt.legend(loc="upper left")
plt.yscale('log')
#plt.show()

plt.subplot(2,1,2 )
for ticker in cripto[:mitad]:
    plt.plot(combined[ticker], label=ticker)
    plt.legend(loc="upper left")
plt.yscale('log')

plt.show()


'''Calculate Correlation'''

combined =  combined.pct_change().corr(method="pearson") #linear correlation 
## Referencia  :  https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.corr.html Comprobar otros algoritmo 
''' Probando algoritmo de correlaciones
combined =  combined.pct_change().corr(method="kendall") # ordinal/rank correlation 
        ##Referencia  : https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient
combined =  combined.pct_change().corr(method="spearman") # strictly monotone
        ##Referncia  : https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient
'''

sns.heatmap(combined , annot=True, cmap="coolwarm", annot_kws={"size":6})
## El heatmap enseña la correlación con colores entre las monedas, predomina el azul entonces sin correlaciones notable
plt.show()
