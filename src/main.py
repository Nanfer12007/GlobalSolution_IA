import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

def carregar_dados(caminho_arquivo):
    """Carrega o arquivo CSV."""
    return pd.read_csv(caminho_arquivo)

def treinar_modelo(X_train, y_train):
    """Treina o RandomForest com GridSearch."""
    rfc = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
    }
    grid_search = GridSearchCV(rfc, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def avaliar_modelo(modelo, X_test, y_test):
    """Avalia o modelo e imprime métricas."""
    y_pred = modelo.predict(X_test)
    print("Relatório de Classificação:\n", classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusão')
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.show()

def salvar_modelo(modelo, caminho):
    """Salva o modelo treinado."""
    joblib.dump(modelo, caminho)
    print(f"Modelo salvo em: {caminho}")

def mostrar_importancia_variaveis(modelo, features):
    """Mostra a importância das variáveis."""
    importances = modelo.feature_importances_
    feat_importances = pd.Series(importances, index=features)
    feat_importances.nlargest(5).plot(kind='barh')
    plt.title('Importância das Variáveis')
    plt.show()

if __name__ == "__main__":
    # Caminhos
    caminho_dados = os.path.join("..", "data", "dados_clima.csv")
    caminho_modelo = os.path.join("..", "models", "modelo_alerta.pkl")
    
    # Carregar dados
    data = carregar_dados(caminho_dados)
    data = data.dropna()
    
    # Features e target
    X = data[['temp_max', 'temp_min', 'umidade', 'vento', 'pressao']]
    y = data['alerta_calor']
    
    # Dividir dados
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Treinar modelo
    modelo = treinar_modelo(X_train, y_train)
    
    # Avaliar
    avaliar_modelo(modelo, X_test, y_test)
    
    # Importância das variáveis
    mostrar_importancia_variaveis(modelo, X.columns)
    
    # Salvar modelo
    salvar_modelo(modelo, caminho_modelo)
