import pickle
import numpy as np

# Carrega o modelo salvo com pickle
with open('models/modelo_alerta_calor.pkl', 'rb') as f:
    modelo = pickle.load(f)

# Entrada: temperatura, umidade, vento
entrada = np.array([[28.0, 60.0, 3.0]])

# Faz a previsão
predicao = modelo.predict(entrada)

# Resultado interpretado
if predicao[0] == 1:
    print("⚠️ ALERTA DE CALOR EXTREMO")
else:
    print("✅ Temperatura dentro dos padrões")
