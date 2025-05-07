#------------1. Importamos bibliotecas necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA

#------------2. Cargamos y preparamos los datos
df = pd.read_csv('student_habits_performance.csv')
df['parental_education_level'] = df['parental_education_level'].fillna(df['parental_education_level'].mode()[0])

# Mapeamos variables ordinales
diet_quality = {'Poor': 0, 'Fair': 1, 'Good': 2}
parental_education_level = {'High School': 0, 'Bachelor': 1, 'Master': 2}
internet_quality = {'Poor': 0, 'Average': 1, 'Good': 2}

df['dq_e'] = df['diet_quality'].map(diet_quality)
df['pel_e'] = df['parental_education_level'].map(parental_education_level)
df['iq_e'] = df['internet_quality'].map(internet_quality)

# One-hot encoding para variables sin orden
dummies = pd.get_dummies(df[['gender', 'part_time_job', 'extracurricular_participation']], drop_first=True)
df2 = pd.concat([df, dummies], axis=1)

# Eliminamos columnas ya transformadas
df2 = df2.drop(['gender', 'part_time_job', 'diet_quality', 'parental_education_level', 
                'internet_quality', 'extracurricular_participation', 'student_id'], axis=1)

#------------3. División de datos
X = df2.drop('exam_score', axis=1)
y = df2['exam_score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#------------4. Estandarización
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#------------5. Correlación con variable objetivo
df_corr = df2.copy()
df_corr['exam_score'] = y
sns.heatmap(df_corr.corr()[['exam_score']].sort_values(by='exam_score', ascending=False), annot=True)
plt.title("Correlación con 'exam_score'")
plt.show()

#------------6. Selección de características
selector = SelectKBest(score_func=f_regression, k='all')
selector.fit(X_train_scaled, y_train)

print("\nPuntuación de características según SelectKBest:")
for feature, score in zip(X.columns, selector.scores_):
    print(f"{feature}: {score:.2f}")

#------------7. Modelo KNN sin PCA
knn_model = KNeighborsRegressor(n_neighbors=7)
knn_model.fit(X_train_scaled, y_train)
y_pred_knn = knn_model.predict(X_test_scaled)

r2_knn = r2_score(y_test, y_pred_knn)
rmse_knn = np.sqrt(mean_squared_error(y_test, y_pred_knn))
print(f"\nKNN (original) → R²: {r2_knn:.3f}, RMSE: {rmse_knn:.2f}")

#------------8. Aplicamos PCA
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"\nComponentes retenidos con PCA (95% varianza): {pca.n_components_}")

# Modelo KNN con PCA
knn_pca = KNeighborsRegressor(n_neighbors=7)
knn_pca.fit(X_train_pca, y_train)
y_pred_pca = knn_pca.predict(X_test_pca)

r2_pca = r2_score(y_test, y_pred_pca)
rmse_pca = np.sqrt(mean_squared_error(y_test, y_pred_pca))
print(f"KNN con PCA → R²: {r2_pca:.3f}, RMSE: {rmse_pca:.2f}")
