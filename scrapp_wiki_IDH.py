# En este proyecto extrairemos datos del siguiente enlace de wikipedia_
# https://es.wikipedia.org/wiki/Anexo:Países_de_América_Latina_por_índice_de_desarrollo_humano
# Información sobre el índice de desarrollo humano histórico (5ta tabla), como podemos observar, hay varias tablas en esta página.
# Este proyecto es el último de los ejercicios del curso de DA de LinkedIn

# Importamos librerias y métoods
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from sklearn.model_selection import KFold, train_test_split
from sklearn import linear_model
from sklearn.metrics import r2_score


#------------------------------ [ Primera Parte ] ---------------------------- #
#   En esta primera parte haremos el scrapping de la página, obteniendo        #
#   información y limpiandola a su vez, para después armar nuestra trama       #
#   de datos y emplear pandas para su manejo.                                  #
#------------------------------...................---------------------------- #

    # En una variable ponemos la URL, en otra accedemos a la página mediante requesrts con la función get, si todo sale bien tendremos un código de respuesta.
    # Seguido, obtenemos la sopa, se pasa mediante texto para eleminar otras características (como jsons etc) y se utiliza un "parser" que es html, es el cómo viene
my_url = 'https://es.wikipedia.org/wiki/Anexo:Países_de_América_Latina_por_índice_de_desarrollo_humano'
page = requests.get(my_url)
my_soup = BeautifulSoup(page.text, 'html.parser')
    # Buscaremos todas las etiquetas que contengan la tabla de interés, asimismo, en la inspección web vimos que las tablas
    # tienen un alias: wikitable sortable
my_soup.find_all('table')[4]
my_soup.find('table', class_ = 'wikitable sortable')
    # BS4 tiene la ventaja de indexar lo que encuentra, por ello asignamos a la variable table la posición [4] de la lista.
    # La cual es obviamente la lista del IDH que nos interesa. En la página web de wiki es la quinta tabla que aparece.
table = my_soup.find_all('table')[4]
    # Una vez identificada nuestra tabla, ahora buscaremos los headers. En la inspección de wikipedia estos headers estan con la etiqueta 'th'
table.find_all('th')
years_header_raw = table.find_all('th')
print(years_header_raw)
    # Pasamos la lista de todas las 'th' encontradas a una variable ~cruda y aqui viene lo trucoso.
    # En una nueva variable, enlistaremos todos estos headers, el loop es como viene; la variable interna del ciclo 'title' se pondra en 
    # la lista en forma de texto, por eso tenemos primero el title.text for title in, en donde estará ciclado en la lista years_headers_raw, dando [title.text for title in years_header_raw]
years_header = [title.text for title in years_header_raw]
print(years_header)
    # Limpiamos con strip() y en una iteración todos los elementos de la lista.
years_header = [title.text.strip() for title in years_header_raw]
years_header = years_header[1:19]
print(years_header)
    # Definimos nuestra trama de datos
df = pd.DataFrame(columns = years_header)
    # Ya que tenemos iniciado el dataframe con los headers, ahora tenemos que rellenar la tabla.
    # Las etiquetas de wikipedia que hacen referencia a las filas (rows) son 'tr'
table.find_all('tr')
    # Ahora bien, los datos en cada fila vienen por la etiqueta de 'td'. Para ello creamos un ciclo que haga la 
    # iteración en cada columna que se encontró en table y así guardamos en cada ciclo en una nueva variable para las 
    # filas, los datos de cada celda.
data_columns = table.find_all('tr')
for row in data_columns:
    data_row = row.find_all('td')

    # Limpiamos de una vez cada dato de cada celda 
for row in data_columns:
    data_row = row.find_all('td')
    individual_data_row = [data.text.strip() for data in data_row]
print(individual_data_row)

    # De nuevo, hacemos el mismo ciclo, reemplazamos de una vez la coma por el punto (para poder ser leído posteriormente)
    # como int, y lo ponemos de una vez en el dataframe, modificando el indíce para ir colocando en cada fila, 
    # así cada que aumente el frame, se ajustará el tamaño y se añadirá al final.
for row in data_columns[1:]:
    data_row = row.find_all('td')
    individual_data_row = [data.text.strip().replace(",",".") for data in data_row]
    
    length = len(df)
    df.loc[length] = individual_data_row
print(df)

    # Como podemos observar, la columna de países quedó vacía debido a que los datos de entrada venían vacios, en la página de la tabla, 
    # se agrega un ícono y una extensión a la bandera de cada país, por lo tanto, sacaremos la información a partir de los tags (dentro del tag).
table.find_all('a', title=True)
    # Buscando dentro del tag < a > vemos que en los títulos vienen los nombres de los países. Extrairemos esta información.
for flag in table.find_all('a', title=True):
    individual_flag = flag['title'][11:]
    print(individual_flag)
    # Ya sea a partir del titulo "title" o del enlace "href", podemos obtener esta información. Algo nuevo que se encontró aquí es que dentro de los 
    # tags podemos indexar esta info; cuando la variable "flag" recorre la lista de table.find_all('a', href=True) (todas las tags a que tengan titulo) 
    # podemos acceder a los atributos de la siguiente manera: flag['class'], flag['href'] o flag['title'].
    # Cada una nos dará su debida info. Por ejemplo, class nos dará siempre "mw-file-description", title nos dará la info que buscamos y href otra descripción
    # Ahora, usamos el indexado en cada dato para limpiar el dato: flag['title'][11:]
individual_flag = list()
for flag in table.find_all('a', title=True):
    individual_flag.append(flag['title'][11:])
    
df_flags = pd.DataFrame({'País': individual_flag})
print(df_flags)
    # unimos nuestro previa trama de datos con esta nueva columna:
df_final = df.join(df_flags, how='left', lsuffix='País', rsuffix='')
print(df_final)
    # Pequeño bug suscitado al momento de hacer el join, quitamos esa columna
df_final = df_final.drop(columns = 'PaísPaís')

#------------------------------ [ Segunda Parte ] ---------------------------- #
#   En esta nueva parte, haremos la analítica mediante ML para el uso          #
#   e interpretación de la información                                         #
#------------------------------...................---------------------------- #

    # Hacemos un subset de datos, excluyendo las columnas con los países y generando
    # el año 2020 como la variable respuesta (esta debe ser incluida en el primer subset).
X = df_final.iloc[:,:-1].astype(float)
Y = X.iloc[:,-1].astype(float)
print(X)
    # Mediante Kmean generamos 3 subsets para entremiento y testeo
kf = KFold(n_splits=3)
kf.get_n_splits(X)
    # inicializamos nuestro modelo de regresión
reg = linear_model.LinearRegression()
res = []
    # En esta iteración, en cada subset generado (3), repartiremos los datos
    # aleatoriamente en las variables de entrenamiento y testeo, para después
    # ajustar al modelo de regresión y finalmente hacer una predicción con el subset de testeo
    # para así generar un valor puntaje de R cuadrada en cadda caso (3 subsets).
for train_index, test_index in kf.split(X):
    X_train, X_test = X.loc[train_index], X.loc[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    reg.fit(X_train, Y_train)
    pred = reg.predict(X_test)
    print("R2 score: ", r2_score(Y_test, pred))
    res.append(r2_score(Y_test, pred))

    # Vemos los coeficientes de salida
print("R2 mean: ", np.mean(res))
print(reg.coef_, reg.intercept_)
    # Ahora, para ver el comportamiento de los datos, vamos a transponer la trama de datos y graficamos
df2 = df_final.T.iloc[:-1,:].astype(float)
df2.plot(grid=True)
plt.show()
