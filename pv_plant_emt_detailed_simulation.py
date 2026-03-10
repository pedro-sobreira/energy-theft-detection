import os
import numpy as np
import matplotlib.pyplot as plt

def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")

# --- Funcoes de Controle ---
class PIController:
    def __init__(self, kp, ki, Ts, upper_limit=None, lower_limit=None):
        self.kp = kp
        self.ki = ki
        self.Ts = Ts
        self.integrator = 0.0
        self.error_prev = 0.0
        self.output = 0.0
        self.upper_limit = upper_limit
        self.lower_limit = lower_limit

    def update(self, error):
        # Termo Proporcional
        P_term = self.kp * error

        # Termo Integral (com anti-windup)
        self.integrator += self.ki * error * self.Ts
        if self.upper_limit is not None and self.integrator > self.upper_limit:
            self.integrator = self.upper_limit
        elif self.lower_limit is not None and self.integrator < self.lower_limit:
            self.integrator = self.lower_limit

        I_term = self.integrator

        self.output = P_term + I_term

        # Limitar a saida do controlador
        if self.upper_limit is not None and self.output > self.upper_limit:
            self.output = self.upper_limit
        elif self.lower_limit is not None and self.output < self.lower_limit:
            self.output = self.lower_limit

        self.error_prev = error
        return self.output

class PLL:
    def __init__(self, kp, ki, Ts, f_grid):
        self.kp = kp
        self.ki = ki
        self.Ts = Ts
        self.integrator = 0.0
        self.theta = 0.0  # Angulo estimado da rede
        self.omega_0 = 2 * np.pi * f_grid # Frequencia nominal da rede
        self.pi_controller = PIController(kp, ki, Ts, upper_limit=2*np.pi*70, lower_limit=2*np.pi*50) # Limitar a saida do PI para a frequencia

    def update(self, v_q_grid):
        # v_q_grid eh o erro de fase (idealmente 0)
        omega_out = self.omega_0 + self.pi_controller.update(v_q_grid)
        self.theta += omega_out * self.Ts
        return self.theta

# --- Modelos de Componentes ---
class PVArray:
    def __init__(self, P_rated, V_mpp, I_mpp):
        self.P_rated = P_rated
        self.V_mpp = V_mpp
        self.I_mpp = I_mpp

    def get_power(self, irradiance_factor=1.0, temperature_factor=1.0):
        # Modelo simplificado: potencia proporcional a irradiacao
        return self.P_rated * irradiance_factor

    def get_voltage(self, power_output):
        # Modelo simplificado: tensao no MPP
        return self.V_mpp

class BoostConverter:
    def __init__(self, L, C_dc, R_load, V_in_rated, I_in_rated, Ts):
        self.L = L
        self.C_dc = C_dc
        self.R_load = R_load # Carga equivalente do inversor
        self.V_in_rated = V_in_rated
        self.I_in_rated = I_in_rated
        self.Ts = Ts

        self.i_L = 0.0 # Corrente no indutor
        self.v_C_dc = V_in_rated # Tensao no capacitor DC-Link

    def update(self, V_in, I_in, duty_cycle):
        # Equacoes de estado do conversor Boost (aproximacao media)
        # di_L/dt = (V_in - (1-d)V_out) / L
        # dV_out/dt = ((1-d)I_L - I_out) / C_dc

        # Corrente de saida do boost (entrada do inversor) - simplificacao
        I_out_boost = self.v_C_dc / self.R_load # Assumindo carga resistiva equivalente

        # Atualizacao do indutor
        di_L = (V_in - (1 - duty_cycle) * self.v_C_dc) / self.L
        self.i_L += di_L * self.Ts

        # Atualizacao do capacitor DC-Link
        dv_C_dc = ((1 - duty_cycle) * self.i_L - I_out_boost) / self.C_dc
        self.v_C_dc += dv_C_dc * self.Ts

        return self.v_C_dc, self.i_L

class InverterGFL:
    def __init__(self, L_filter, C_filter, R_filter, V_dc_rated, P_rated, V_grid_rms, f_grid, Ts):
        self.L_filter = L_filter
        self.C_filter = C_filter
        self.R_filter = R_filter
        self.V_dc_rated = V_dc_rated
        self.P_rated = P_rated
        self.V_grid_rms = V_grid_rms
        self.f_grid = f_grid
        self.Ts = Ts
        self.omega = 2 * np.pi * f_grid

        # Variaveis de estado do filtro LCL (simplificado para L filter)
        self.i_L_inv_alpha = 0.0
        self.i_L_inv_beta = 0.0
        self.v_inv_alpha = 0.0
        self.v_inv_beta = 0.0

        # Controladores PI
        self.pi_id = PIController(kp=0.5, ki=50, Ts=Ts, upper_limit=1.0, lower_limit=-1.0) # Corrente d
        self.pi_iq = PIController(kp=0.5, ki=50, Ts=Ts, upper_limit=1.0, lower_limit=-1.0) # Corrente q
        self.pi_vdc = PIController(kp=0.1, ki=10, Ts=Ts, upper_limit=self.P_rated/self.V_dc_rated, lower_limit=0) # Tensao DC-Link

        self.pll = PLL(kp=100, ki=1000, Ts=Ts, f_grid=f_grid)

    def abc_to_dq(self, v_a, v_b, v_c, theta):
        # Transformada Park
        v_alpha = (2/3) * (v_a - 0.5 * v_b - 0.5 * v_c)
        v_beta = (2/3) * (np.sqrt(3)/2 * v_b - np.sqrt(3)/2 * v_c)
        v_d = v_alpha * np.cos(theta) + v_beta * np.sin(theta)
        v_q = -v_alpha * np.sin(theta) + v_beta * np.cos(theta)
        return v_d, v_q

    def dq_to_abc(self, v_d, v_q, theta):
        # Transformada Park Inversa
        v_alpha = v_d * np.cos(theta) - v_q * np.sin(theta)
        v_beta = v_d * np.sin(theta) + v_q * np.cos(theta)
        v_a = v_alpha
        v_b = -0.5 * v_alpha + np.sqrt(3)/2 * v_beta
        v_c = -0.5 * v_alpha - np.sqrt(3)/2 * v_beta
        return v_a, v_b, v_c

    def update(self, V_dc, V_grid_abc, I_grid_abc, P_ref, Q_ref):
        # 1. PLL para sincronizacao com a rede
        v_d_grid, v_q_grid = self.abc_to_dq(V_grid_abc[0], V_grid_abc[1], V_grid_abc[2], self.pll.theta)
        self.pll.update(v_q_grid) # Atualiza o angulo da rede
        theta_grid = self.pll.theta

        # 2. Controle de Tensao DC-Link (gera referencia de corrente ativa Id_ref)
        V_dc_error = self.V_dc_rated - V_dc
        Id_ref = self.pi_vdc.update(V_dc_error)
        
        # 3. Controle de Corrente (eixos d-q)
        # Transformar correntes da rede para o referencial d-q
        i_d_grid, i_q_grid = self.abc_to_dq(I_grid_abc[0], I_grid_abc[1], I_grid_abc[2], theta_grid)

        # Referencia de corrente reativa (para Q_ref)
        # Q_ref = 1.5 * V_d_grid * I_q_ref (aproximado)
        # I_q_ref = Q_ref / (1.5 * V_d_grid) se V_d_grid nao for zero
        Iq_ref = 0.0 # Para operacao com fator de potencia unitario

        # Erros de corrente
        id_error = Id_ref - i_d_grid
        iq_error = Iq_ref - i_q_grid

        # Saida dos controladores PI de corrente (referencias de tensao d-q para o inversor)
        v_d_inv_ref = self.pi_id.update(id_error) - self.omega * self.L_filter * i_q_grid + v_d_grid # Feedforward de tensao da rede e termo desacoplamento
        v_q_inv_ref = self.pi_iq.update(iq_error) + self.omega * self.L_filter * i_d_grid + v_q_grid # Feedforward de tensao da rede e termo desacoplamento

        # 4. Modulacao PWM (simplificado: assume-se que o inversor gera as tensoes de referencia)
        v_a_inv, v_b_inv, v_c_inv = self.dq_to_abc(v_d_inv_ref, v_q_inv_ref, theta_grid)

        # 5. Dinamica do Filtro L (simplificado, sem capacitor)
        # di_L/dt = (V_inv - V_grid - R_filter * i_L) / L_filter
        # Usamos as correntes da rede como as correntes do indutor do inversor para esta simplificacao
        # Atualizacao das correntes do filtro (inversor)
        di_L_inv_alpha = (v_a_inv - V_grid_abc[0] - self.R_filter * self.i_L_inv_alpha) / self.L_filter
        di_L_inv_beta = (v_b_inv - V_grid_abc[1] - self.R_filter * self.i_L_inv_beta) / self.L_filter

        self.i_L_inv_alpha += di_L_inv_alpha * self.Ts
        self.i_L_inv_beta += di_L_inv_beta * self.Ts

        # Correntes de saida do inversor (para a rede)
        I_out_inv_abc = np.array([self.i_L_inv_alpha, self.i_L_inv_beta, -(self.i_L_inv_alpha + self.i_L_inv_beta)])

        # Potencias no PCC
        P_pcc = (V_grid_abc[0] * I_out_inv_abc[0] + V_grid_abc[1] * I_out_inv_abc[1] + V_grid_abc[2] * I_out_inv_abc[2])
        Q_pcc = 0 # Simplificado, controle de Q eh 0

        return I_out_inv_abc, P_pcc, Q_pcc, theta_grid


def main():
    clear_screen()
    print("--- Simulacao EMT Detalhada de Usina Fotovoltaica 5 MW ---")

    # --- Parametros do Sistema ---
    P_pv_rated = 5e6  # Potencia nominal da usina fotovoltaica (5 MW)
    V_grid_line_rms = 13.8e3  # Tensao de linha RMS da rede (13.8 kV)
    V_grid_phase_rms = V_grid_line_rms / np.sqrt(3) # Tensao de fase RMS
    f_grid = 60  # Frequencia da rede (Hz)
    omega = 2 * np.pi * f_grid  # Frequencia angular

    V_dc_rated = 800.0 # Tensao nominal do barramento DC (V)
    V_pv_mpp = 600.0 # Tensao no MPP do arranjo PV (V)
    I_pv_mpp = P_pv_rated / V_pv_mpp # Corrente no MPP do arranjo PV (A)

    # Parametros do Boost Converter
    L_boost = 1e-3 # Indutancia do boost (H)
    C_dc_boost = 1000e-6 # Capacitancia do DC-Link do boost (F)
    R_load_boost = V_dc_rated / (P_pv_rated / V_dc_rated) # Carga equivalente do inversor

    # Parametros do Inversor GFL (filtro L)
    L_filter_inv = 0.5e-3 # Indutancia do filtro do inversor (H)
    R_filter_inv = 0.01 # Resistencia do filtro do inversor (Ohm)

    # Impedancia da rede (simplificada para media tensao)
    S_base = P_pv_rated
    Z_base = (V_grid_line_rms**2) / S_base
    R_grid = 0.01 * Z_base  # Resistencia da rede (pu)
    X_grid = 0.1 * Z_base  # Reatancia da rede (pu)
    Z_grid = R_grid + 1j * X_grid

    # --- Parametros de Simulacao ---
    t_sim = 5.0  # Tempo total de simulacao (s)
    Ts = 50e-6  # Passo de tempo (s) - para uma simulacao EMT
    t = np.arange(0, t_sim, Ts)
    num_steps = len(t)

    # --- Ponto de Curto-Circuito ---
    t_fault_start = 1.0  # Tempo de inicio do curto-circuito (s)
    t_fault_end = 1.5    # Tempo de fim do curto-circuito (s) - para recuperacao
    fault_impedance_factor = 0.01  # Reducao da impedancia da rede durante o curto (0.01 = 1% da impedancia normal)

    # --- Inicializacao dos Modelos ---
    pv_array = PVArray(P_pv_rated, V_pv_mpp, I_pv_mpp)
    boost_converter = BoostConverter(L_boost, C_dc_boost, R_load_boost, V_pv_mpp, I_pv_mpp, Ts)
    inverter_gfl = InverterGFL(L_filter_inv, 0, R_filter_inv, V_dc_rated, P_pv_rated, V_grid_line_rms, f_grid, Ts)

    # --- Variaveis de Saida ---
    V_pcc_line_rms_out = np.zeros(num_steps)
    I_pcc_line_rms_out = np.zeros(num_steps)
    P_pcc_out = np.zeros(num_steps)
    Q_pcc_out = np.zeros(num_steps)
    V_dc_out = np.zeros(num_steps)
    theta_grid_out = np.zeros(num_steps)

    # --- Simulacao --- 
    print(f"Iniciando simulacao por {t_sim} segundos com passo de {Ts}s...")
    for i in range(num_steps):
        current_time = t[i]

        # Tensao da rede trifasica (assumida como fonte ideal para simplificacao)
        V_grid_a = V_grid_phase_rms * np.sqrt(2) * np.sin(omega * current_time)
        V_grid_b = V_grid_phase_rms * np.sqrt(2) * np.sin(omega * current_time - 2*np.pi/3)
        V_grid_c = V_grid_phase_rms * np.sqrt(2) * np.sin(omega * current_time + 2*np.pi/3)
        V_grid_abc = np.array([V_grid_a, V_grid_b, V_grid_c])

        # Correntes da rede (feedback para o inversor, inicialmente zero)
        I_grid_abc = np.array([0.0, 0.0, 0.0]) # Sera atualizado pelo inversor

        # --- Curto-Circuito ---
        current_Z_grid = Z_grid
        if t_fault_start <= current_time < t_fault_end:
            # Simula um curto-circuito trifasico reduzindo drasticamente a impedancia da rede
            current_Z_grid = Z_grid * fault_impedance_factor
            # Durante o curto, a tensao da rede no PCC cai
            V_grid_a *= 0.1 # 10% da tensao nominal
            V_grid_b *= 0.1
            V_grid_c *= 0.1
            V_grid_abc = np.array([V_grid_a, V_grid_b, V_grid_c])

        # 1. Modelo PV
        P_pv_output = pv_array.get_power()
        V_pv_output = pv_array.get_voltage(P_pv_output)

        # 2. Controle MPPT (simplificado: duty cycle fixo ou P&O basico)
        # Para esta simulacao, vamos assumir um MPPT ideal que mantem a tensao de entrada do boost no V_pv_mpp
        # e ajusta o duty cycle para transferir a potencia maxima.
        # A saida do boost eh o V_dc_rated
        duty_cycle_mppt = 1 - (V_pv_output / V_dc_rated) # Aproximacao para boost ideal
        if duty_cycle_mppt < 0: duty_cycle_mppt = 0
        if duty_cycle_mppt > 0.9: duty_cycle_mppt = 0.9 # Limite para duty cycle

        # 3. Boost Converter
        V_dc_boost, I_L_boost = boost_converter.update(V_pv_output, I_pv_mpp, duty_cycle_mppt)
        V_dc_out[i] = V_dc_boost

        # 4. Inversor GFL
        I_out_inv_abc, P_pcc, Q_pcc, theta_grid = inverter_gfl.update(V_dc_boost, V_grid_abc, I_grid_abc, P_pv_output, 0)

        # Atualizar I_grid_abc para o proximo passo (feedback)
        I_grid_abc = I_out_inv_abc

        # Salvar resultados
        V_pcc_line_rms_out[i] = np.sqrt(np.mean(V_grid_abc**2)) * np.sqrt(3) # A tensao no PCC eh a tensao da rede para este modelo
        I_pcc_line_rms_out[i] = np.sqrt(np.mean(I_out_inv_abc**2)) * np.sqrt(3) # Corrente de linha RMS
        P_pcc_out[i] = P_pcc
        Q_pcc_out[i] = Q_pcc
        theta_grid_out[i] = theta_grid

    # --- Plotagem dos Resultados ---
    print("Gerando graficos dos resultados...")
    plt.figure(figsize=(14, 12))

    # Tensao de Linha RMS na saida da usina (PCC)
    plt.subplot(5, 1, 1)
    plt.plot(t, V_pcc_line_rms_out, label="Tensao de Linha RMS (V)")
    plt.title("Tensao de Linha RMS na Saida da Usina (PCC)")
    plt.ylabel("Tensao (V)")
    plt.grid(True)
    plt.axvline(t_fault_start, color='r', linestyle='--', label='Inicio do Curto')
    plt.axvline(t_fault_end, color='g', linestyle='--', label='Fim do Curto')
    plt.legend()

    # Corrente de Linha RMS na saida da usina (PCC)
    plt.subplot(5, 1, 2)
    plt.plot(t, I_pcc_line_rms_out, label="Corrente de Linha RMS (A)")
    plt.title("Corrente de Linha RMS na Saida da Usina (PCC)")
    plt.ylabel("Corrente (A)")
    plt.grid(True)
    plt.axvline(t_fault_start, color='r', linestyle='--')
    plt.axvline(t_fault_end, color='g', linestyle='--')
    plt.legend()

    # Potencia Ativa na saida da usina (PCC)
    plt.subplot(5, 1, 3)
    plt.plot(t, P_pcc_out / 1e6, label="Potencia Ativa (MW)", color='orange')
    plt.title("Potencia Ativa na Saida da Usina (PCC)")
    plt.ylabel("Potencia Ativa (MW)")
    plt.grid(True)
    plt.axvline(t_fault_start, color='r', linestyle='--')
    plt.axvline(t_fault_end, color='g', linestyle='--')
    plt.legend()

    # Potencia Reativa na saida da usina (PCC)
    plt.subplot(5, 1, 4)
    plt.plot(t, Q_pcc_out / 1e6, label="Potencia Reativa (MVAR)", color='purple')
    plt.title("Potencia Reativa na Saida da Usina (PCC)")
    plt.ylabel("Potencia Reativa (MVAR)")
    plt.grid(True)
    plt.axvline(t_fault_start, color='r', linestyle='--')
    plt.axvline(t_fault_end, color='g', linestyle='--')
    plt.legend()

    # Tensao DC-Link
    plt.subplot(5, 1, 5)
    plt.plot(t, V_dc_out, label="Tensao DC-Link (V)", color='brown')
    plt.title("Tensao no Barramento DC (DC-Link)")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Tensao (V)")
    plt.grid(True)
    plt.axvline(t_fault_start, color='r', linestyle='--')
    plt.axvline(t_fault_end, color='g', linestyle='--')
    plt.legend()

    plt.tight_layout()
    plt.savefig("pv_plant_emt_detailed_simulation_results.png")
    print("Graficos salvos como \"pv_plant_emt_detailed_simulation_results.png\"")

if __name__ == "__main__":
    main()
