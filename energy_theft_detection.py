import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import warnings

# Ignorar avisos para manter a saida limpa
warnings.filterwarnings("ignore")

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def generate_dummy_data(n_samples=3000):
    """
    Gera dados ficticios para demonstracao de deteccao de furto de energia.
    """
    np.random.seed(42)
    
    # Gerando dados normais
    consumo_medio = np.random.normal(200, 50, n_samples)
    variacao_consumo = np.random.normal(20, 10, n_samples)
    latitude = np.random.uniform(-23.6, -23.4, n_samples)
    longitude = np.random.uniform(-46.7, -46.5, n_samples)
    tipo_consumidor = np.random.randint(0, 2, n_samples)
    
    alvo = np.zeros(n_samples)
    
    # Criando padroes de furto mais agressivos para o modelo aprender
    for i in range(n_samples):
        # Regra 1: Consumo muito baixo com variacao alta (perfil tipico de desvio)
        if consumo_medio[i] < 60 and variacao_consumo[i] > 30:
            alvo[i] = 1
        # Regra 2: Queda drastica (simulada por consumo baixissimo independente da variacao)
        elif consumo_medio[i] < 30:
            alvo[i] = 1
        # Regra 3: Localizacao especifica com consumo abaixo da media
        elif latitude[i] > -23.42 and consumo_medio[i] < 100:
            alvo[i] = 1
            
    data = pd.DataFrame({
        'consumo_medio': consumo_medio,
        'variacao_consumo': variacao_consumo,
        'latitude': latitude,
        'longitude': longitude,
        'tipo_consumidor': tipo_consumidor,
        'furto': alvo
    })
    return data

def main():
    clear_screen()
    print("================================================")
    print("   SISTEMA DE DETECCAO DE FURTOS DE ENERGIA     ")
    print("================================================")
    
    # 1. Preparacao dos dados
    print("\n[1] Gerando base de dados sintetica...")
    df = generate_dummy_data()
    print(f"Total de registros: {len(df)}")
    print(f"Casos de furto identificados: {df['furto'].sum()}")
    
    X = df.drop('furto', axis=1)
    y = df['furto']
    
    # 2. Divisao em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Normalizacao
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Treinamento da Rede Neural
    print("\n[2] Treinando Rede Neural (Multi-layer Perceptron)...")
    # Ajustando parametros para melhor convergencia em dados desbalanceados
    mlp = MLPClassifier(
        hidden_layer_sizes=(32, 16), 
        activation='relu',
        solver='adam',
        max_iter=1000, 
        random_state=42
    )
    mlp.fit(X_train_scaled, y_train)
    
    # 5. Avaliacao
    print("\n[3] Avaliacao do Modelo:")
    predictions = mlp.predict(X_test_scaled)
    
    print("\nMatriz de Confusao:")
    print(confusion_matrix(y_test, predictions))
    
    print("\nRelatorio de Metricas:")
    print(classification_report(y_test, predictions))
    
    # 6. Testes Praticos
    print("\n[4] Simulacao de Casos Reais:")
    
    casos = [
        # Consumo, Variacao, Lat, Long, Tipo
        {"nome": "Consumidor Padrao", "dados": [250, 15, -23.5, -46.6, 0]},
        {"nome": "Suspeita por Baixo Consumo", "dados": [20, 40, -23.41, -46.51, 0]},
        {"nome": "Comercio Suspeito", "dados": [45, 35, -23.55, -46.62, 1]}
    ]
    
    for caso in casos:
        entrada = np.array([caso["dados"]])
        entrada_scaled = scaler.transform(entrada)
        pred = mlp.predict(entrada_scaled)[0]
        prob = mlp.predict_proba(entrada_scaled)[0][1]
        
        status = "ALERTA DE FURTO" if pred == 1 else "NORMAL"
        print(f"- {caso['nome']}: {status} (Probabilidade: {prob:.2%})")

if __name__ == "__main__":
    main()
