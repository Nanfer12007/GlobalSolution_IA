# SmartMotoIA

Kaio Cumpian Silva - 99816 <br/>
Gabriel Yuji Suzuki - RM556588 <br/>
Lucas Felix Vassiliades - RM97677 <br/>

## Problema
Com o avanço das mudanças climáticas, a frequência de ondas de calor aumentou, afetando a saúde pública, especialmente idosos, crianças e pessoas com doenças crônicas. 

## Objetivo da Análise
O objetivo é criar um Sistema de Alerta de Calor Extremo, que através de machine learning consiga prever dias de risco baseado em dados históricos de temperatura e umidade, enviando alertas preventivos.

## Modelo dscolhido: Random Forest Classifier
O Random Forest Classifier foi escolhido por oferecer boa performance em problemas de classificação binária com múltiplas variáveis numéricas e categóricas, como é o caso da previsão de alertas de calor extremo.

## Metodologia Utilizada

1 - Coleta de dados
* Utilizou-se o dataset público do clima da cidade de São Paulo, disponível no Kaggle, contendo variáveis meteorológicas como:

- Temperatura do ar
- Umidade relativa
- Velocidade do vento
- Precipitação, radiação solar, entre outras

2 - Pré-processamento

- Conversão do CSV com separador ;
- Seleção das colunas relevantes: temperatura, umidade, vento
- Remoção de valores nulos
- Renomeação de colunas para facilitar o uso no código

3 - Definição do Alvo (Target)

* O sistema considera como alerta de calor extremo todas as amostras com temperatura > 33 °C, gerando um rótulo binário:

- 1 = risco

- 0 = normal

4 - Divisão dos Dados

- Separação entre dados de treino (80%) e teste (20%)

5 - Modelo de Machine Learning

- Algoritmo utilizado: Random Forest Classifier

- Justificativa: algoritmo robusto, bom desempenho com variáveis contínuas, resistente a overfitting

6 - Treinamento e Validação

- O modelo foi treinado com os dados de treino

- Avaliação com métricas como accuracy, precision, recall, f1-score

## Tecnologias Utilizadas 

Componente                    Função <br/>

 pandas                       | Manipulação de dados, leitura do CSV e estruturação em DataFrame <br/>
 numpy                        | Operações matemáticas de base e manipulação de arrays <br/>
 scikit-learn                 | Separação dos dados, treinamento do modelo e otimização de hiperparâmetros <br/>
 matplotlib                   | Geração de gráficos, nesse caso o gráfico de importância das variáveis <br/>
 seaborn                      | Visualização de dados <br/>

## Instalar Pacotes

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

# Resultados Obtidos
O modelo foi capaz de identificar condições de calor extremo com precisão, baseado nos dados históricos. As métricas indicaram:

- Boa acurácia

- Bom equilíbrio entre precision e recall
