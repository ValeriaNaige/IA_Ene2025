# imports necesarios
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

#cargamos los datos de entrada
data = pd.read_csv("articulos_ml.csv")


#veamos cuantas dimensiones y registros contiene
print(data.shape)
print(data.head())
print(data.describe())

# gráficas de barras del contenido del archivo csv
data.drop(['Title','url', 'Elapsed days'], axis=1).hist()
plt.show()

#
filtered_data=data[(data['Word count']<=3500)&(data['# Shares']<=80000)]
colores=['orange','blue']
tamanios=[30,60]

f1=filtered_data['Word count'].values
f2=filtered_data['# Shares'].values


asignar=[]
for index, row in filtered_data.iterrows():
    if(row['Word count']>1808):
        asignar.append(colores[0])
    else:
        asignar.append(colores[1])

plt.scatter(f1,f2,c=asignar,s=tamanios[0])
#plt.show()


#AsignamosnuestravariabledeentradaXpara entrenamientoylasetiquetasY.
dataX=filtered_data[["Word count"]]
X_train=np.array(dataX)
y_train=filtered_data['# Shares'].values

#Creamos el objeto de Regresión Linear
regr=linear_model.LinearRegression()

#Entrenamos nuestro modelo
regr.fit(X_train,y_train)

#Hacemos las predicciones que en definitiva una línea (en este caso, al ser 2D)
y_pred=regr.predict(X_train)

#Veamos los coeficienetes obtenidos, En nuestro caso, serán la Tangente
print('Coefficients: ', regr.coef_)
#Este es el valor donde corta el ejeY (enX=0)
print('Independentterm: ', regr.intercept_)
#Error Cuadrado Medio
print("Mean squared error: %.2f" %mean_squared_error (y_train,y_pred))
#Puntaje de Varianza. El mejor puntaje es un 1.0
print('Variance score: %.2f' %r2_score (y_train,y_pred))


#Vamos a comprobar:
# Quiero predecir cuántos "Shares" voy a obtener por un artículo con 2.000 palabras,
# según nuestro modelo, hacemos:
y_Dosmil = regr.predict([[2000]])
print('Predicción de Shres en un artículo de 2000 palabras: ', int(y_Dosmil))