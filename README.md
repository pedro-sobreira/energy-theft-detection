# Detecção de Furtos de Energia com Redes Neurais

Este projeto implementa um sistema simplificado de detecção de furtos de energia elétrica em unidades consumidoras utilizando Redes Neurais Artificiais (MLP - Multi-layer Perceptron).

## 🚀 Objetivo

O objetivo é identificar padrões de consumo que indiquem possíveis irregularidades ou furtos, levando em consideração:
- **Histórico de Consumo:** Média mensal de consumo em kWh.
- **Variabilidade:** Desvio padrão do consumo, identificando quedas bruscas ou instabilidades.
- **Localização Geográfica:** Coordenadas (Latitude e Longitude) para análise regional.
- **Tipo de Consumidor:** Diferenciação entre perfis residenciais e comerciais.

## 🛠️ Tecnologias Utilizadas

- **Python 3**
- **Scikit-learn:** Para a implementação da rede neural e pré-processamento.
- **Pandas & NumPy:** Para manipulação de dados e cálculos matemáticos.

## 📋 Como Funciona

1. **Geração de Dados:** O script gera uma base de dados sintética simulando milhares de consumidores.
2. **Pré-processamento:** Os dados são normalizados (StandardScaler) para otimizar o treinamento da rede neural.
3. **Treinamento:** Uma rede neural MLP é treinada para classificar os perfis entre "Normal" e "Suspeito".
4. **Predição:** O sistema avalia novos casos e fornece a probabilidade de ocorrência de furto.

## 🔧 Como Executar

1. Certifique-se de ter o Python instalado.
2. Instale as dependências necessárias:
   ```bash
   pip install scikit-learn pandas numpy
   ```
3. Execute o programa:
   ```bash
   python energy_theft_detection.py
   ```

## 📊 Exemplo de Saída

O programa exibirá uma matriz de confusão e um relatório de métricas (precisão, recall, f1-score), além de realizar testes práticos com perfis de consumidores simulados.

---
*Nota: Este é um modelo educacional e simplificado para demonstração do uso de redes neurais em problemas de infraestrutura.*
