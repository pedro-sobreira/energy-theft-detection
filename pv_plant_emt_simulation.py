import os
import numpy as np
import matplotlib.pyplot as plt

def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")

def main():
    clear_screen()
    print("--- Simulacao Simplificada de Usina Fotovoltaica 5 MW ---")

    # --- Parametros do Sistema ---
    P_pv_rated = 5e6  # Potencia nominal da usina fotovoltaica (5 MW)
    V_grid_rms = 13.8e3  # Tensao de linha RMS da rede (13.8 kV)
    f_grid = 60  # Frequencia da rede (Hz)
    omega = 2 * np.pi * f_grid  # Frequencia angular

    # Impedancia da rede (simplificada para media tensao)
    # Assumindo uma impedancia base para 13.8kV, 5MVA
    S_base = P_pv_rated
    Z_base = (V_grid_rms**2) / S_base
    R_grid = 0.01 * Z_base  # Resistencia da rede (pu)
    X_grid = 0.1 * Z_base  # Reatancia da rede (pu)
    Z_grid = R_grid + 1j * X_grid

    # --- Parametros de Simulacao ---
    t_sim = 5  # Tempo total de simulacao (s)
    dt = 1e-4  # Passo de tempo (s) - para uma simulacao "quase-EMT"
    t = np.arange(0, t_sim, dt)
    num_steps = len(t)

    # --- Ponto de Curto-Circuito ---
    t_fault_start = 1.0  # Tempo de inicio do curto-circuito (s)
    t_fault_end = 1.5    # Tempo de fim do curto-circuito (s) - para recuperacao
    fault_impedance_factor = 0.01  # Reducao da impedancia da rede durante o curto (0.01 = 1% da impedancia normal)

    # --- Variaveis de Saida ---
    V_pcc = np.zeros(num_steps, dtype=complex)
    I_pcc = np.zeros(num_steps, dtype=complex)
    P_pcc = np.zeros(num_steps)
    Q_pcc = np.zeros(num_steps)

    # --- Simulacao --- 
    print(f"Iniciando simulacao por {t_sim} segundos com passo de {dt}s...")
    for i in range(num_steps):
        current_time = t[i]

        # Tensao da rede (assumida como referencia de fase)
        V_grid_phase = (V_grid_rms / np.sqrt(3)) * np.exp(1j * (omega * current_time))

        # Potencia ativa e reativa injetada pela PV (simplificado)
        # Assumimos que a usina tenta injetar potencia ativa nominal e Q=0
        P_pv = P_pv_rated
        Q_pv = 0

        # --- Curto-Circuito ---
        current_Z_grid = Z_grid
        if t_fault_start <= current_time < t_fault_end:
            # Simula um curto-circuito trifasico reduzindo drasticamente a impedancia da rede
            current_Z_grid = Z_grid * fault_impedance_factor
            # Durante o curto, a usina pode ter sua injecao de potencia limitada
            # Simplificacao: Reduzir a potencia injetada para evitar correntes infinitas no modelo simplificado
            P_pv = P_pv_rated * 0.1 # 10% da potencia nominal durante o curto
            Q_pv = 0 # Pode haver injecao de Q para suporte de tensao, mas simplificamos para 0

        # --- Modelo Simplificado da Usina e PCC ---
        # A usina tenta injetar P_pv e Q_pv no PCC
        # No modelo simplificado, a tensao no PCC eh determinada pela injecao de corrente da usina
        # e pela impedancia da rede.
        # S_pcc = V_pcc * conj(I_pcc)
        # P_pcc = real(S_pcc), Q_pcc = imag(S_pcc)

        # Para um inversor GFL, ele controla a corrente injetada.
        # Assumimos que a tensao no PCC eh proxima da tensao da rede
        # e calculamos a corrente necessaria para injetar P_pv e Q_pv.
        # S_pv = P_pv + 1j*Q_pv
        # I_pv_conj = S_pv / V_grid_phase
        # I_pv = np.conj(I_pv_conj)

        # Uma abordagem mais robusta para GFL: o inversor controla a corrente para manter a tensao e injetar potencia
        # No entanto, para simplificar e mostrar o efeito do curto, vamos considerar a injecao de potencia
        # e a impedancia da rede para calcular a tensao e corrente no PCC.
        
        # Tensao no PCC (simplificado: V_pcc = V_grid - I_pcc * Z_grid)
        # Corrente no PCC (simplificado: I_pcc = (P_pv - 1j*Q_pv) / conj(V_pcc))
        # Isso forma um sistema de equacoes nao-linear. Para simplicidade, vamos iterar ou usar uma aproximacao.

        # Aproximacao: Assumimos que a tensao no PCC eh proxima da tensao da rede, e a corrente eh determinada pela potencia
        # e pela impedancia da rede.
        # S_inj = P_pv + 1j*Q_pv
        # I_inj = np.conj(S_inj / V_grid_phase) # Corrente que a usina tenta injetar

        # Considerando a impedancia da rede, a tensao no PCC sera afetada pela corrente injetada.
        # V_pcc = V_grid_phase - I_inj * current_Z_grid # Isso eh uma simplificacao, pois I_inj depende de V_pcc
        # Para uma simulacao mais direta do impacto do curto:
        # A tensao no PCC cai drasticamente durante o curto.
        # A corrente eh limitada pela impedancia da fonte e pela impedancia de curto.

        # Modelagem mais direta do curto-circuito na tensao e corrente no PCC
        if t_fault_start <= current_time < t_fault_end:
            # Durante o curto, a tensao no PCC cai para um valor proximo de zero
            V_pcc[i] = V_grid_phase * 0.1 # 10% da tensao nominal durante o curto
            # A corrente eh limitada pela impedancia da rede e pela capacidade do inversor
            I_pcc[i] = V_pcc[i] / current_Z_grid # Corrente de curto
            # A potencia tambem cai
            P_pcc[i] = np.real(V_pcc[i] * np.conj(I_pcc[i]))
            Q_pcc[i] = np.imag(V_pcc[i] * np.conj(I_pcc[i]))
        else:
            # Operacao normal: Inversor injeta potencia ativa nominal e Q=0
            # Assumimos que o inversor consegue manter a tensao no PCC proxima da tensao da rede
            V_pcc[i] = V_grid_phase
            # Corrente para injetar P_pv e Q_pv
            S_pcc_target = P_pv + 1j * Q_pv
            I_pcc[i] = np.conj(S_pcc_target / V_pcc[i])
            P_pcc[i] = P_pv
            Q_pcc[i] = Q_pv

    # --- Plotagem dos Resultados ---
    print("Gerando graficos dos resultados...")
    plt.figure(figsize=(14, 10))

    # Tensao RMS na saida da usina (PCC)
    plt.subplot(4, 1, 1)
    plt.plot(t, np.abs(V_pcc) * np.sqrt(3), label="Tensao de Linha RMS (V)") # Tensao de linha
    plt.title("Tensao de Linha RMS na Saida da Usina (PCC)")
    plt.ylabel("Tensao (V)")
    plt.grid(True)
    plt.axvline(t_fault_start, color="r", linestyle="--", label="Inicio do Curto")
    plt.axvline(t_fault_end, color="g", linestyle="--", label="Fim do Curto")
    plt.legend()

    # Corrente RMS na saida da usina (PCC)
    plt.subplot(4, 1, 2)
    plt.plot(t, np.abs(I_pcc) * np.sqrt(3), label="Corrente de Linha RMS (A)") # Corrente de linha
    plt.title("Corrente de Linha RMS na Saida da Usina (PCC)")
    plt.ylabel("Corrente (A)")
    plt.grid(True)
    plt.axvline(t_fault_start, color="r", linestyle="--")
    plt.axvline(t_fault_end, color="g", linestyle="--")
    plt.legend()

    # Potencia Ativa na saida da usina (PCC)
    plt.subplot(4, 1, 3)
    plt.plot(t, P_pcc / 1e6, label="Potencia Ativa (MW)", color="orange")
    plt.title("Potencia Ativa na Saida da Usina (PCC)")
    plt.ylabel("Potencia Ativa (MW)")
    plt.grid(True)
    plt.axvline(t_fault_start, color="r", linestyle="--")
    plt.axvline(t_fault_end, color="g", linestyle="--")
    plt.legend()

    # Potencia Reativa na saida da usina (PCC)
    plt.subplot(4, 1, 4)
    plt.plot(t, Q_pcc / 1e6, label="Potencia Reativa (MVAR)", color="purple")
    plt.title("Potencia Reativa na Saida da Usina (PCC)")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Potencia Reativa (MVAR)")
    plt.grid(True)
    plt.axvline(t_fault_start, color="r", linestyle="--")
    plt.axvline(t_fault_end, color="g", linestyle="--")
    plt.legend()

    plt.tight_layout()
    plt.savefig("pv_plant_emt_simulation_results.png")
    print("Graficos salvos como \"pv_plant_emt_simulation_results.png\"")
    # plt.show() # Nao usar plt.show() em ambientes nao interativos

if __name__ == "__main__":
    main()
