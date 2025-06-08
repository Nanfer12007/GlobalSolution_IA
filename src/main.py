import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle 
import os

# Caminho do CSV (já salvo localmente)
caminho_arquivo = 'data/clima_sp.csv'

# Lê o arquivo CSV, separador = ';'
df = pd.read_csv(caminho_arquivo, sep=';', encoding='utf-8')

# Renomeia as colunas para facilitar o uso
df = df.rename(columns={
    'TEMPERATURA DO AR - BULBO SECO, HORARIA(°C)': 'TEMPERATURA',
    'UMIDADE RELATIVA DO AR, HORARIA(%)': 'UMIDADE',
    'VENTO, VELOCIDADE HORARIA(m/s)': 'VENTO'
})

# Remove linhas com dados ausentes
df = df.dropna(subset=['TEMPERATURA', 'UMIDADE', 'VENTO'])

# Variáveis independentes (entradas)
X = df[['TEMPERATURA', 'UMIDADE', 'VENTO']]

# Variável alvo (1 = alerta de calor extremo, 0 = normal)
y = (df['TEMPERATURA'] > 33).astype(int)

# Divide treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Cria e treina o modelo
modelo = RandomForestClassifier(random_state=42)
modelo.fit(X_train, y_train)

# Avalia o modelo
y_pred = modelo.predict(X_test)
print("Relatório de Classificação:")
print(classification_report(y_test, y_pred))

# Salva o modelo treinado
os.makedirs('models', exist_ok=True)
with open('models/modelo_alerta_calor.pkl', 'wb') as f:
    pickle.dump(modelo, f)
print("✅ Modelo salvo em: models/modelo_alerta_calor.joblib")
