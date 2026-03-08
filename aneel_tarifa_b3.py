import os
import pandas as pd
import matplotlib.pyplot as plt
import requests
import io

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def load_local_data(filepath):
    print(f"Carregando dados locais de: {filepath}")
    # Usando ISO-8859-1 pois os dados da ANEEL costumam vir nesta codificacao
    return pd.read_csv(filepath, sep=';', encoding='ISO-8859-1', low_memory=False)

def process_data(df):
    print("Processando dados para o Grupo B3...")
    
    # Filtrar pelo Subgrupo B3
    df_b3 = df[df['DscSubGrupo'] == 'B3'].copy()
    
    # Converter datas
    df_b3['DatInicioVigencia'] = pd.to_datetime(df_b3['DatInicioVigencia'])
    
    # Extrair o ano
    df_b3['Ano'] = df_b3['DatInicioVigencia'].dt.year
    
    # Limpar e converter valores numericos (VlrTUSD e VlrTE)
    # Os valores vem com virgula como separador decimal
    for col in ['VlrTUSD', 'VlrTE']:
        df_b3[col] = df_b3[col].astype(str).str.replace(',', '.').astype(float)
    
    # Calcular a Tarifa Total (TUSD + TE)
    df_b3['TarifaTotal'] = df_b3['VlrTUSD'] + df_b3['VlrTE']
    
    # Agrupar por ano e calcular a media
    serie_historica = df_b3.groupby('Ano')['TarifaTotal'].mean().reset_index()
    
    return serie_historica

def plot_data(serie_historica):
    print("Gerando grafico da serie historica...")
    plt.figure(figsize=(12, 6))
    plt.plot(serie_historica['Ano'], serie_historica['TarifaTotal'], marker='o', linestyle='-', color='b')
    
    plt.title('Serie Historica da Tarifa Media de Energia - Grupo B3 (ANEEL)')
    plt.xlabel('Ano')
    plt.ylabel('Tarifa Media (R$/kWh)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Ajustar os anos no eixo X
    plt.xticks(serie_historica['Ano'], rotation=45)
    
    plt.tight_layout()
    plt.savefig('tarifa_b3_serie_historica.png')
    print("Grafico salvo como 'tarifa_b3_serie_historica.png'")

def main():
    clear_screen()
    print("--- Analise de Tarifas de Energia ANEEL (Grupo B3) ---")
    
    filepath = "/home/ubuntu/tarifas_completas.csv"
    
    try:
        if not os.path.exists(filepath):
            print("Arquivo local nao encontrado. Baixando...")
            url = "https://dadosabertos.aneel.gov.br/dataset/5a583f3e-1646-4f67-bf0f-69db4203e89e/resource/fcf2906c-7c32-4b9b-a637-054e7a5234f4/download/tarifas-homologadas-distribuidoras-energia-eletrica.csv"
            response = requests.get(url)
            with open(filepath, 'wb') as f:
                f.write(response.content)
        
        df = load_local_data(filepath)
        serie = process_data(df)
        
        if not serie.empty:
            print("\nSerie Historica Anual (Media):")
            print(serie.to_string(index=False))
            plot_data(serie)
        else:
            print("Nenhum dado encontrado para o Grupo B3.")
            
    except Exception as e:
        print(f"Ocorreu um erro: {e}")

if __name__ == "__main__":
    main()
