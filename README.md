# ML-s-algorithm-on-stock-
This is a machine learning algorithm on Linear regression. It is used to determine the price of stock.
import quandl,datetime
import sklearn
from sklearn import preprocessing, svm, linear_model
from sklearn.model_selection import cross_validate
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use(['ggplot'])



df = quandl.get('WIKI/GOOGL')
print(df.head)
print(df.describe())

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
print(df.head(9))

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['PCT_change', 'Adj. Close', 'HL_PCT', 'Adj. Volume']]
print(df.head())

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.1 * len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)



x = np.array(df.drop(['label', 'Adj. Close'], 1))
x = preprocessing.scale(x)
x_lately =x[-forecast_out:]
x = x[:-forecast_out]

df.dropna(inplace=True)
y = np.array(df['label'])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)
# linear = linear_model.LinearRegression(n_jobs=-1)
# linear.fit(x_train,y_train)
# with open ('linearregression.pickle','wb') as f:
#     pickle.dump(linear,f)

pickle_in= open('linearregression.pickle','rb')
linear = pickle.load(pickle_in)

accuracy = linear.score(x_test,y_test)
print(accuracy)

forecast_set = linear.predict(x_lately)
print(forecast_set,accuracy, forecast_out)

df['forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

print(df.head())
print(df.tail())

df['Adj. Close'].plot()
df['forecast'].plot()

plt.legend(loc=4)
plt.xlabel('date')
plt.ylabel('price')
plt.show()



df['Adj. Close'].plot()
df['Adj. Volume'].plot()
plt.legend(loc=1)
plt.show()
