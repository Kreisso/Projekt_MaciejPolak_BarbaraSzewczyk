import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import pickle
from sklearn.model_selection import cross_val_score
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

boston_dataset = load_boston()

boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
boston['PRICE'] = boston_dataset.target

boston.isnull().sum()

X = pd.DataFrame(np.c_[boston['LSTAT'], boston['RM'], boston['PTRATIO'], boston['INDUS']], columns=['LSTAT', 'RM', 'PTRATIO', 'INDUS'])
Y = boston['PRICE']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)

print("\n--- LSTAT i RM i PTRATIO i INDUS \n")

# --- Odczyt modelu z pliku
lin_model = pickle.load(open('LSTAT_RM_PTRATIO_INDUS/lin_model.model', 'rb'))



scores = cross_val_score(lin_model, X_test, Y_test, cv=5)
print("-- Cross Validation\n")
print(scores)


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
plt.show()



print("\n--- LSTAT i RM i PTRATIO \n")


X = pd.DataFrame(np.c_[boston['LSTAT'], boston['RM'], boston['PTRATIO']], columns=['LSTAT', 'RM', 'PTRATIO'])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)

# --- Odczyt modelu z pliku
lin_model = pickle.load(open('LSTAT_RM_PTRATIO/lin_model.model', 'rb'))

scores = cross_val_score(lin_model, X_test, Y_test, cv=5)
print("-- Cross Validation\n")
print(scores)

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
plt.show()


print("\n--- LSTAT i RM  \n")

X = pd.DataFrame(np.c_[boston['LSTAT'], boston['RM']], columns=['LSTAT', 'RM'])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)
# --- Odczyt modelu z pliku
lin_model = pickle.load(open('LSTAT_RM/lin_model.model', 'rb'))

scores = cross_val_score(lin_model, X_test, Y_test, cv=5)
print("-- Cross Validation\n")
print(scores)

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
plt.show()




