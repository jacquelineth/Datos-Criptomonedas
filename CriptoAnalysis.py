import os 

import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import seaborn as sns
import datetime as dt

from seaborn.matrix import heatmap

currency  = "USD"
metric = "Close"


'''Data Prep'''
crypto = []

directory = f"Ficheros Originales\\"
for filename in os.listdir(directory):
    if filename.endswith(f"{currency}.csv"):
        crypto.append( filename.split('-')[0])
    else:
        continue


start = dt.datetime(2018, 1,1 )
end =  dt.datetime.now()

# crypto = ['BTC','ADA','BCH','BNB' 'ETH', 'LTC', 'XRP', 'DOGE', 'MATIC']
print(crypto)

colnames = []

first  = True 

for ticker in crypto :

    data = pd.read_csv(f"Ficheros Originales\{ticker}-{currency}.csv")
    if first: 
        combined =  data[[metric]].copy ()
        colnames.append(ticker)
        combined.columns =  colnames
        first = False
    else:
        combined  = combined.join(data[metric])
        colnames.append(ticker)
        combined.columns = colnames  


plt.yscale('log')

for ticker in crypto:
    plt.plot(combined[ticker], label=ticker)

plt.legend(loc="upper right")
plt.show()

'''Calculate Correlation'''
combined =  combined.pct_change().corr(method="pearson")
sns.heatmap(combined , annot=False, cmap="coolwarm")
plt.show()