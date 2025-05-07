#------------1.Importamos las bibliotecas necesarias
# Importamos las bibliotecas necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# De scikit-learn para procesamiento de datos y evaluación
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Usamos MLPRegressor como Red Neuronal
from sklearn.neural_network import MLPRegressor

#------------2. Leemos y preparamos los datos
# Cargar el archivo CSV en un DataFrame
df = pd.read_csv('student_habits_performance.csv')

# Mostrar las primeras filas del dataframe para entender la estructura
#df.head() #Si esta al final de la celda asi se pinta
print(df.head())

# Ver información sobre el dataframe (tipos de datos, valores nulos, etc.)
df.info()

# Rellenamos los valores faltantes en la columna 'parental_education_level' con el valor más frecuente
# Hacemos esto porque el dataset no puede tener columnas vacias a la hora de procesarse
df['parental_education_level'] = df['parental_education_level'].fillna(df['parental_education_level'].mode()[0])

# Aseguramos que no haya más valores nulos
print(df.isna().sum())

#------------3. Preprocesamiento de variables categóricas
# Mapeamos las variables categóricas que puede tener un orden a valores numéricos (Le das un numero mayor cuanto mas importante sea el parametro)
diet_quality = {'Poor': 0, 'Fair': 1, 'Good': 2}
parental_education_level = {'High School': 0, 'Bachelor': 1, 'Master': 2}
internet_quality = {'Poor': 0, 'Average': 1, 'Good': 2}

# Creamos las nuevas columnas codificadas numéricamente
df['dq_e'] = df['diet_quality'].map(diet_quality)
df['pel_e'] = df['parental_education_level'].map(parental_education_level)
df['iq_e'] = df['internet_quality'].map(internet_quality)

# Convertimos las variables categóricas restantes usando get_dummies
#Como son variables categoricas que no se pueden ordenar, las transformamos en variables numericas con One hot-encoding
dummies = pd.get_dummies(df[['gender', 'part_time_job', 'extracurricular_participation']], drop_first=True) #One hot encoding

# Concatenamos las variables categóricas procesadas con el dataframe original
df2 = pd.concat([df, dummies], axis=1)

# Eliminamos las columnas originales de texto ya procesadas
df2 = df2.drop(['gender', 'part_time_job', 'diet_quality', 'parental_education_level', 'internet_quality', 'extracurricular_participation', 'student_id'], axis=1) 

# Verificamos las primeras filas del dataframe procesado
print(df2.head())

#------------4.Correlacion de las variables numericas
# Calculamos la matriz de correlación
corr = df2.corr()

# Visualizamos la matriz de correlación con un mapa de calor
sns.heatmap(corr, annot=True)
plt.show()

#------------5.Division de los datos en conjuntos de entrenamiento y prueba
# Definimos X (características) y y (etiqueta)
X = df2.drop('exam_score', axis=1)
y = df2['exam_score']

# Dividimos los datos en entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Verificamos las dimensiones de los conjuntos
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

#------------6. Estandarización de los datos
# Estandarizamos los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Verificamos las dimensiones después de la estandarización
print(X_train_scaled.shape, X_test_scaled.shape)


#------------7.Creamos la Red Neuronal con MLPRegressor
# Creamos el modelo de red neuronal con MLPRegressor de scikit-learn
model = MLPRegressor(hidden_layer_sizes=(256, 128, 64, 32), activation='relu', solver='adam',learning_rate_init=0.01, max_iter=1000,early_stopping=True, random_state=42)

# Entrenamos el modelo
model.fit(X_train_scaled, y_train)

#------------8.Evaluamos el modelo
# Realizamos predicciones
y_pred = model.predict(X_test_scaled)

# Calculamos R² y RMSE
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Mostramos los resultados
print(f"Red Neuronal → R²: {r2:.3f}, RMSE: {rmse:.2f}")

#------------9.Opcional: visualizar la evolución de la pérdida durante el entrenamiento
plt.plot(model.loss_curve_)
plt.title('Progreso de la pérdida durante el entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.show()
