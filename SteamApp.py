import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import torch
import torch.nn as nn
import joblib

st.set_page_config(page_title="Hydrazine Tank Digital Twin", layout="wide")
st.title("🚀 Hydrazine Tank Pressure Digital Twin")

scaler_X = joblib.load("C:\python\hydrazine tank pressure due to RTA and decomposition\scaler_X.save")
scaler_y = joblib.load("C:\python\hydrazine tank pressure due to RTA and decomposition\scaler_y.save")

class PressureLSTM(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.hidden_size = 128
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=3,
            batch_first=True
        )
        
        self.fc = nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 3)
        )

    def forward(self, t):
        return self.net(t)

lstm_model = PressureLSTM(input_size=5)
lstm_model.load_state_dict(torch.load("C:\python\hydrazine tank pressure due to RTA and decomposition\lstm_model.pth", map_location="cpu"))
lstm_model.eval()

pinn_model = PINN()
pinn_model.load_state_dict(torch.load("hydrazine tank pressure due to RTA and decomposition/pinn_model.pth", map_location="cpu"))
pinn_model.eval()

st.sidebar.header("⚙️ Input Parameters")

C0 = st.sidebar.slider("Initial Concentration", 500.0, 2000.0, 1200.0)
T0 = st.sidebar.slider("Initial Temperature (K)", 280.0, 450.0, 320.0)
n_g0 = st.sidebar.slider("Initial Gas Moles", 0.0, 0.1, 0.02)

A = st.sidebar.number_input("Pre-exponential Factor (A)", value=1.2e7)
Ea = st.sidebar.number_input("Activation Energy (Ea)", value=85000.0)
UA = st.sidebar.number_input("Heat Transfer (UA)", value=15.0)
tau = st.sidebar.slider("Sensor Delay τ", 0.1, 3.0, 1.0)

simulate_btn = st.sidebar.button("▶ Run Simulation")
R = 8.314
V = 0.02
rho = 1020.0
Cp = 3000.0
dH = -5.0e5
nu_g = 1.5
T_env = 300.0

def hydrazine_ode(t, y):
    C, T, n_g, Pm = y
    
    k = A * np.exp(-Ea / (R * T))
    r = k * C

    dCdt = -r
    dTdt = (-dH * r)/(rho*Cp) - (UA/(rho*Cp*V))*(T - T_env)
    dn_gdt = nu_g * r * V

    P_true = (n_g * R * T) / V
    dPmdt = (P_true - Pm) / tau

    return [dCdt, dTdt, dn_gdt, dPmdt]

if simulate_btn:

    # Initial Conditions
    y0 = [C0, T0, n_g0, (n_g0 * R * T0) / V]
    t_eval = np.linspace(0, 50, 400)

    # Solve ODE
    sol = solve_ivp(hydrazine_ode, (0, 50), y0, t_eval=t_eval, method="BDF")

    t = sol.t
    C = sol.y[0]
    T = sol.y[1]
    n_g = sol.y[2]
    P_true = (n_g * R * T) / V
    Pm = sol.y[3]

    # Sensor Noise
    noise = np.random.normal(0, 0.02 * 1e5, size=P_true.shape)
    P_sensor = np.clip(Pm + noise, 0, None)

    # LSTM Prediction
    features = np.column_stack([
        t,
        T,
        P_sensor,
        C,
        n_g
    ]).astype(np.float32)

    features_scaled = scaler_X.transform(features)

    SEQ_LEN = 25
    X_ml = []

    for i in range(len(features_scaled) - SEQ_LEN):
        X_ml.append(features_scaled[i:i+SEQ_LEN])

    X_ml = torch.tensor(np.array(X_ml), dtype=torch.float32)

    with torch.no_grad():
        y_ml_scaled = lstm_model(X_ml).numpy()

    y_ml_log = scaler_y.inverse_transform(y_ml_scaled.reshape(-1, 1))
    P_ml = np.expm1(y_ml_log)

    P_ml = np.pad(P_ml.flatten(), (SEQ_LEN, 0), mode='edge')

    # PINN Prediction
    t_tensor = torch.tensor(t.reshape(-1, 1), dtype=torch.float32)

    with torch.no_grad():
        pinn_pred = pinn_model(t_tensor).numpy()

    C_pinn = pinn_pred[:, 0]
    T_pinn = pinn_pred[:, 1]
    n_g_pinn = pinn_pred[:, 2]

    P_pinn = (n_g_pinn * R * T_pinn) / V

    # PLOTS
    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots()
        ax1.plot(t, T)
        ax1.set_title("🌡 Temperature Evolution")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Temperature (K)")
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots()
        ax2.plot(t, P_true/1e5, label="True")
        ax2.plot(t, P_sensor/1e5, '--', label="Sensor")
        ax2.plot(t, P_ml/1e5, ':', label="LSTM")
        ax2.plot(t, P_pinn/1e5, '-.', label="PINN")
        ax2.legend()
        ax2.set_title("💨 Pressure Comparison")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Pressure (bar)")
        st.pyplot(fig2)

    # SAFETY WARNINGS
    if np.any(P_true > 50e5):
        st.error("🚨 Overpressure Failure!")

    elif np.any(P_true > 0.85 * 50e5):
        st.warning("⚠ Approaching Pressure Limit")

    if np.any(T > 800):
        st.error("🔥 Thermal Runaway!")