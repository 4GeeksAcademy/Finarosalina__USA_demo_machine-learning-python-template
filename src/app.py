import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import requests
from sklearn.preprocessing import StandardScaler


url = 'https://raw.githubusercontent.com/4GeeksAcademy/regularized-linear-regression-project-tutorial/main/demographic_health_data.csv'
response = requests.get(url)

if response.status_code == 200:
    with open('/workspaces/Finarosalina__USA_demo_machine-learning-python-template/data/raw/demographic_health_data.csv', 'wb') as file:
        file.write(response.content)
    print("Archivo descargado correctamente!")
else:
    print(f"Hubo un problema al descargar el archivo: {response.status_code}")


ds=pd.read_csv('/workspaces/Finarosalina__USA_demo_machine-learning-python-template/data/raw/demographic_health_data.csv')


pd.set_option('display.max_columns', None)
ds.head(10)

ds.columns

ds.shape

ds.info()

# Verificar si hay filas duplicadas
duplicados = ds.duplicated()
print(ds[duplicados])

ds.isnull().sum().sort_values(ascending=False).head(20)


for col in ds.select_dtypes(include='object').columns:
    print(f"{col}:\n", ds[col].value_counts(), "\n")


# categóricas : 'COUNTY_NAME', 'STATE_NAME'
sns.countplot(data=ds, x='COUNTY_NAME')
plt.xticks(rotation=45)
plt.show()
sns.countplot(data=ds, x='STATE_NAME')
plt.xticks(rotation=45)
plt.show()

corr = ds.corr(numeric_only=True)

plt.figure(figsize=(20,30))
sns.heatmap(corr, cmap='coolwarm', center=0)
plt.title("Mapa de calor de correlaciones")
plt.show()


corr = ds.corr(numeric_only=True)


target_corr = corr["Heart disease_number"].sort_values(ascending=False)
print(target_corr.head(45))  # Las más correlacionadas positivamente


target_corr = corr["Heart disease_number"].sort_values(ascending=False)
print(target_corr.tail(5))  # Las más correlacionadas negativamente

top_positive = target_corr.head(45)
top_negative = target_corr.tail(5)

selected_features = top_positive.index.tolist() + top_negative.index.tolist()

ds_filtrado = ds[selected_features + ["Heart disease_number"]]
ds_filtrado.shape
ds_filtrado.dtypes


# Verificar las columnas en ds_filtrado
print(ds_filtrado.columns)


# Eliminar las columnas duplicadas
ds_filtrado = ds_filtrado.loc[:, ~ds_filtrado.columns.duplicated()]


ds_filtrado.columns

ds_filtrado.head()


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = ds_filtrado.drop("Heart disease_number", axis=1)
y = ds_filtrado["Heart disease_number"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
columnas = X.columns  

X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=columnas)
X_test = pd.DataFrame(scaler.transform(X_test), columns=columnas)


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

# Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("R²:", r2)



from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

alphas = np.logspace(-3, 3, 20)  # De 0.001 a 1000

ridge_r2 = []
lasso_r2 = []
ridge_mse = []
lasso_mse = []

for alpha in alphas:
    # Ridge
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)
    ridge_r2.append(r2_score(y_test, y_pred_ridge))
    ridge_mse.append(mean_squared_error(y_test, y_pred_ridge))

    # Lasso
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_train, y_train)
    y_pred_lasso = lasso.predict(X_test)
    lasso_r2.append(r2_score(y_test, y_pred_lasso))
    lasso_mse.append(mean_squared_error(y_test, y_pred_lasso))

# Gráficas
plt.figure(figsize=(14, 5))

# R²
plt.subplot(1, 2, 1)
plt.plot(alphas, ridge_r2, label='Ridge R²', marker='o')
plt.plot(alphas, lasso_r2, label='Lasso R²', marker='s')
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('R² Score')
plt.title('R² vs Alpha')
plt.legend()
plt.grid(True)

# MSE
plt.subplot(1, 2, 2)
plt.plot(alphas, ridge_mse, label='Ridge MSE', marker='o')
plt.plot(alphas, lasso_mse, label='Lasso MSE', marker='s')
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('Mean Squared Error')
plt.title('MSE vs Alpha')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


from sklearn.linear_model import Ridge

ridge_final = Ridge(alpha=0.1)
ridge_final.fit(X_train, y_train)

y_pred_ridge = ridge_final.predict(X_test)

print("R² Ridge:", r2_score(y_test, y_pred_ridge))
print("MSE Ridge:", mean_squared_error(y_test, y_pred_ridge))


print(X_train.shape)  # Verifica las dimensiones de X_train
print(ridge_final.coef_.shape)  # Verifica las dimensiones de los coeficientes


ridge_coef = pd.Series(ridge_final.coef_, index=X_train.columns)

print("Coeficientes Ridge:")
print(ridge_coef)


print("Intercepto del modelo Ridge:", ridge_final.intercept_)


ds_filtrado.to_csv('/workspaces/Finarosalina__USA_demo_machine-learning-python-template/data/processed/ds_filtrado.csv', index=False)

# Guardar conjuntos de entrenamiento y prueba
X_train.to_csv('/workspaces/Finarosalina__USA_demo_machine-learning-python-template/data/processed/X_train.csv', index=False)
X_test.to_csv('/workspaces/Finarosalina__USA_demo_machine-learning-python-template/data/processed/X_test.csv', index=False)
y_train.to_csv('/workspaces/Finarosalina__USA_demo_machine-learning-python-template/data/processed/y_train.csv', index=False)
y_test.to_csv('/workspaces/Finarosalina__USA_demo_machine-learning-python-template/data/processed/y_test.csv', index=False)


import json

notebook_path = '/workspaces/Finarosalina__USA_demo_machine-learning-python-template/src/explore.ipynb'
with open(notebook_path, 'r') as notebook_file:
    notebook_content = json.load(notebook_file)

python_script_path = '/workspaces/Finarosalina__USA_demo_machine-learning-python-template/src/app.py'
with open(python_script_path, 'w') as app_file:
    
    for cell in notebook_content['cells']:
        
        if cell['cell_type'] == 'code':
            
            code = ''.join(cell['source'])
            app_file.write(code + '\n\n') 

print(f"El código ha sido extraído del notebook y guardado en {python_script_path}")


