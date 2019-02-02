import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import pickle
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

boston_dataset = load_boston()

boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)

print("\n--- Boston head \n")
print(boston.head())

print("\n--- Boston describe \n")
print(boston.describe())

boston['PRICE'] = boston_dataset.target

boston.isnull().sum()

sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.distplot(boston['PRICE'], bins=30)
plt.savefig('rozklad_zmiennej_PRICE.png', bbox_inches='tight')
plt.show()


correlation_matrix = boston.corr().round(2)
sns.heatmap(data=correlation_matrix, annot=True)
plt.savefig('macierz_korelacji.png', bbox_inches='tight')

plt.figure(figsize=(20, 5))

features = ['LSTAT', 'RM']
target = boston['PRICE']

for i, col in enumerate(features):
    plt.subplot(1, len(features) , i+1)
    x = boston[col]
    y = target
    plt.scatter(x, y, marker='o')
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('PRICE')

plt.savefig('wykres_punktowy_RM_LSTAT.png', bbox_inches='tight')
plt.show()

X = pd.DataFrame(np.c_[boston['LSTAT'], boston['RM']], columns=['LSTAT', 'RM'])
Y = boston['PRICE']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)

print("\n--- X train \n")

print(X_train.shape)

print("\n--- X test  \n")

print(X_test.shape)

print("\n--- Y train  \n")

print(Y_train.shape)

print("\n--- Y test  \n")

print(Y_test.shape)

# --- Utworzenei i zapis modelu
lin_model = LinearRegression()
lin_model.fit(X_train, Y_train)
pickle.dump(lin_model, open('lin_model.model', 'wb'));

# --- Odczyt modelu z pliku
# lin_model = pickle.load(open('lin_model.model', 'rb'))


y_train_predict = lin_model.predict(X_train)
rmse = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))
r2 = r2_score(Y_train, y_train_predict)



print ("\n--- Wydajność modelu dla zestawu trenującego\n")
print('RMSE : {}'.format(rmse))
print('R2 : {}'.format(r2))

plt.scatter(Y_train, y_train_predict)
plt.xlabel("Prices train: $Y_i$")
plt.ylabel("Predicted prices train: $\hat{Y}_i$")
plt.title("Prices train vs Predicted prices train: $Y_i$ vs $\hat{Y}_i$")
plt.savefig('wykres_punktowy_Ptrain_PPtrain.png', bbox_inches='tight')

plt.show()

y_test_predict = lin_model.predict(X_test)
rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))
r2 = r2_score(Y_test, y_test_predict)

print ("\n--- Wydajność modelu dla zestawu testowego\n")
print('RMSE : {}'.format(rmse))
print('R2 : {}'.format(r2))


plt.scatter(Y_test, y_test_predict)
plt.xlabel("Prices Test: $Y_i$")
plt.ylabel("Predicted prices Test: $\hat{Y}_i$")
plt.title("Prices Test vs Predicted prices Test: $Y_i$ vs $\hat{Y}_i$")
plt.savefig('wykres_punktowy_Ptest_PPtest.png', bbox_inches='tight')
plt.show()

plt.scatter(lin_model.predict(X_train), lin_model.predict(X_train) - Y_train, c ='b', s = 40, alpha=0.5)
plt.scatter(lin_model.predict(X_test), lin_model.predict(X_test) - Y_test, c ='g', s = 40)
plt.hlines(y = 0, xmin = 0, xmax = 50)
plt.title('Wykres z wykorzystaniem danych treningowych ( niebieski )  i testowych ( zielony )')
plt.ylabel('')
plt.savefig('wykres_punktowy_wykorzystanie_danych.png', bbox_inches='tight')
plt.show()