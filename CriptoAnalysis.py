import os 

import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import seaborn as sns
import datetime as dt

from seaborn.matrix import heatmap

moneda  = "USD"
metric = "Close"


'''Data Prep'''
cripto = []

#Parseamos el repertorio por csv de cripto
directory = f"Ficheros Originales\\"
for fichero in os.listdir(directory):
    if fichero.endswith(f"{moneda}.csv"):
        cripto.append( fichero.split('-')[0])
    else:
        continue


start = dt.datetime(2018, 1,1 )
end =  dt.datetime.now()


print(f"Tabla de Cripto {cripto}")

colnames = []

primero  = True 

for ticker in cripto :

    data = pd.read_csv(f"Ficheros Originales\{ticker}-{moneda}.csv")
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

for ticker in cripto:
    plt.plot(combined[ticker], label=ticker)

plt.legend(loc="upper right")
plt.show()

'''Calculate Correlation'''
combined =  combined.pct_change().corr(method="pearson")
sns.heatmap(combined , annot=False, cmap="coolwarm")
plt.show()