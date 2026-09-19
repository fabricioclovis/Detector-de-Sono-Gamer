import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# ==========================================
# PASSO 1: Preparação dos Dados
# ==========================================
# Criando o DataFrame original do seu projeto
gamer = pd.DataFrame({
    'horas_jogo': [1, 2, 4, 6, 8, 10],
    'cansaco': [1, 2, 3, 5, 8, 10]
})

# Separando as features (entradas, X) e o target (saída esperada, y)
# Convertendo para arrays do NumPy, o formato que o TensorFlow prefere
X = gamer['horas_jogo'].values
y = gamer['cansaco'].values


# ==========================================
# PASSO 2: Criação do Modelo
# ==========================================
# Iniciamos um modelo Sequencial (uma pilha linear de camadas)
modelo = Sequential()

# Adicionamos uma camada Densa (Totalmente Conectada).
# units=1: Queremos prever apenas 1 valor (o cansaço).
# input_shape=[1]: Nossa entrada tem apenas 1 característica (horas de jogo).
modelo.add(Dense(units=1, input_shape=[1]))


# ==========================================
# PASSO 3: Compilação do Modelo
# ==========================================
# Precisamos dizer ao modelo como ele vai aprender e como medirá seus erros.
# loss='mean_squared_error': Calcula o Erro Quadrático Médio (distância entre a previsão e o real).
# optimizer=Adam: O "ajustador" que vai tentar diminuir o erro a cada tentativa. 
# A taxa de aprendizado (learning_rate) em 0.1 ajuda o modelo a aprender mais rápido neste exemplo simples.
modelo.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), 
    loss='mean_squared_error'
)


# ==========================================
# PASSO 4: Treinamento (Ajuste/Fit)
# ==========================================
print("Iniciando o treinamento do Detector de Sono Gamer...")

# O modelo vai observar os dados 500 vezes (epochs=500).
# verbose=0 esconde os logs de cada época para deixar o terminal mais limpo.
modelo.fit(X, y, epochs=500, verbose=0)

print("Treinamento concluído com sucesso!\n")


# ==========================================
# PASSO 5: Predição (Teste Prático)
# ==========================================
# Vamos prever o cansaço de um jogador que passou 7 horas jogando
horas_teste = np.array([7.0])

# O método predict recebe nossos dados e retorna a previsão com base no que o modelo aprendeu
previsao = modelo.predict(horas_teste)

# Extraímos o número de dentro da estrutura de array que o TensorFlow retorna
nivel_cansaco = previsao[0][0]

print(f"--- RESULTADO DA PREDIÇÃO ---")
print(f"Horas jogando: {horas_teste[0]}h")
print(f"Nível de cansaço previsto: {nivel_cansaco:.1f} / 10")