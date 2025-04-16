
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import itertools

# === EXPLICAÇÃO DO APP ===
st.markdown("""
### ℹ️ Sobre o modelo

Este aplicativo utiliza uma **Rede Neural Artificial (RNA)** treinada para estimar a **deformação vertical no topo do subleito** de pavimentos aeroportuários, considerando diferentes configurações de estrutura e tráfego.

---

### 🌡️ Correção de Temperatura no Revestimento

A **temperatura padrão considerada para os módulos de resiliência (MR)** dos revestimentos asfálticos é de **25°C**.

Para ajustar o MR ao clima local, é utilizada a seguinte **equação do método Austroads (2013)**:

> \( E_T = E_{25} \cdot e^{-0.08 \cdot (T - 25)} \)

Onde:
- \( E_T \) = módulo ajustado para temperatura média anual \( T \)
- \( E_{25} \) = módulo a 25°C

---

### 🎯 Coeficientes de Variação (COV)

O **COV** (Coeficiente de Variação) representa o **grau de incerteza** de uma variável de entrada, expresso em percentual (%).

Ele indica o quanto determinado parâmetro (como espessura ou módulo) pode variar em torno de seu valor médio.

> 🔒 **Deseja uma análise determinística?** Basta **definir todos os COVs como zero**.  
> Isso desabilita a análise probabilística, considerando apenas os valores médios.

Os COVs sugeridos neste aplicativo foram definidos com base na literatura técnica.

---

### ✈️ Desvio-padrão do Wander

O **wander** é a variação lateral da posição de pouso/decolagem das aeronaves na pista. Essa dispersão influencia a distribuição de carga ao longo da largura do pavimento.

O parâmetro usado para representar essa dispersão é o **desvio-padrão lateral do wander**, medido em centímetros.

> 📌 O **valor padrão adotado é 77,3 cm**, conforme recomendação da **FAA (Federal Aviation Administration)**.
""")

# === CARREGAR MODELO ===
modelo = joblib.load('modelo_4_camadas_ESRS_ESRD_COMPLETO.pkl')
weights = modelo['weights']
scaler = modelo['scaler']
aeronaves_dict = modelo['aeronaves']

# === Funções ===
def tanh(x): return np.tanh(x)
def neural_network(weights, X):
    W1 = weights[:9 * 10].reshape((9, 10))
    b1 = weights[90:100]
    W2 = weights[100:180].reshape((10, 8))
    b2 = weights[180:188]
    W3 = weights[-9:-1].reshape((8, 1))
    b3 = weights[-1]
    hidden1 = tanh(np.dot(X, W1) + b1)
    hidden2 = tanh(np.dot(hidden1, W2) + b2)
    output = np.dot(hidden2, W3) + b3
    return output.flatten()

def prever_deformacao_customizada(valores):
    entrada = np.array([valores])
    entrada_norm = scaler.transform(entrada)
    return neural_network(weights, entrada_norm)[0]

def corrigir_mr_para_Tc(E25_mpa, Tc):
    return E25_mpa * np.exp(-0.08 * (Tc - 25))

def calcular_confiabilidade_rosenblueth(media_inputs, covs, confianca):
    indices = {
        'revest_MR': 2, 'revest_h': 3,
        'base_MR': 4, 'base_h': 5,
        'subbase_MR': 6, 'subbase_h': 7,
        'subleito_MR': 8
    }
    variaveis = list(indices.items())
    mu_vector = media_inputs.copy()
    cov_values = [covs[k] / 100 for k, _ in variaveis]

    if all(c == 0 for c in cov_values):
        st.warning("Todos os COVs estão zerados. A análise será determinística.")

    combinacoes = list(itertools.product([-1, 1], repeat=7))
    resultados = []
    for sinais in combinacoes:
        v = mu_vector.copy()
        for i, (_, idx) in enumerate(variaveis):
            v[idx] = mu_vector[idx] * (1 + sinais[i] * cov_values[i])
        f = prever_deformacao_customizada(v)
        resultados.append(f)

    media = np.mean(resultados)
    desvio = np.std(resultados)
    z = stats.norm.ppf(confianca / 100)
    confiavel = media + z * desvio
    return media, confiavel

def calcular_C(ev):
    if ev >= 1.765e-3:
        return (0.00414131 / ev) ** 8.1
    else:
        log_C = (-0.1638 + 185.19 * ev) ** (-0.60586)
        return 10 ** log_C

def calcular_N(a, b, L):
    return (1 + (b * L / 200)) * a * L

def calcular_cp(faixa_centro, weq, pos_pneus, wander_std):
    return sum([
        (stats.norm.cdf(faixa_centro + weq/2, pneu_pos, wander_std) -
         stats.norm.cdf(faixa_centro - weq/2, pneu_pos, wander_std))
        for pneu_pos in pos_pneus
    ])

def obter_pc_por_faixa(aeronave_nome, h_total, wander_std):
    faixa_largura = 25.4
    faixas = np.linspace(-40 * faixa_largura, 40 * faixa_largura, 81)
    aero = aeronaves_dict[aeronave_nome]
    if aero['Ne'] == 1 or h_total >= aero['t'] - aero['w']:
        weq = aero['w'] + aero['t'] + h_total if aero['Ne'] > 1 else aero['w'] + h_total
        posicoes = [aero['xk_centro']]
    else:
        weq = aero['w'] + h_total
        desloc = aero['t'] / 2
        posicoes = [aero['xk_centro'] - desloc, aero['xk_centro'] + desloc]
    posicoes += [-p for p in posicoes]
    cp_vals = np.array([calcular_cp(f, weq, posicoes, wander_std) for f in faixas])
    return faixas, 1 / cp_vals

# === INTERFACE ===
st.title("Predição de Deformações em Pavimentos Aeroportuários com RNA")

st.header("Dados do Pavimento")
Tc = st.number_input("Temperatura média anual (°C)", value=27.0)
revest_MR = st.number_input("MR 25°C Revestimento (MPa)", value=2500.0)
revest_h = st.number_input("Espessura do Revestimento (m)", value=0.3)
base_MR = st.number_input("MR Base (MPa)", value=300.0)
base_h = st.number_input("Espessura da Base (m)", value=0.3)
subbase_MR = st.number_input("MR Subbase (MPa)", value=150.0)
subbase_h = st.number_input("Espessura da Subbase (m)", value=0.4)
subleito_MR = st.number_input("MR Subleito (MPa)", value=60.0)
vida_util = st.number_input("Vida útil (anos)", value=20.0)
wander_std = st.number_input("Wander std (cm)", value=77.3)

st.markdown("---")
st.header("Configurações de Confiabilidade")
cov_padrao = {
    'revest_MR': 15, 'base_MR': 20, 'subbase_MR': 20, 'subleito_MR': 20,
    'revest_h': 7, 'base_h': 12, 'subbase_h': 15
}
covs = {}
for key, val in cov_padrao.items():
    covs[key] = st.number_input(f'COV {key} (%)', value=float(val))
confianca = st.number_input("Nível de Confiabilidade (%)", value=95.0)

st.markdown("---")
st.header("Dados das Aeronaves")
cdf_total = np.zeros(81)
faixas_ref = np.linspace(-40 * 25.4, 40 * 25.4, 81)

for nome in sorted(aeronaves_dict.keys()):
    props = aeronaves_dict[nome]
    col1, col2 = st.columns(2)
    with col1:
        a = st.number_input(f"Decolagens anuais {nome}", key=f"{nome}_a", value=0)
    with col2:
        b = st.number_input(f"Crescimento b (%) {nome}", key=f"{nome}_b", value=0.0)

    if a > 0:
        entrada = [
            props['Pressão dos Pneus(MPa)'],
            props['dist_rodas(m)'],
            revest_MR, revest_h,
            base_MR, base_h,
            subbase_MR, subbase_h,
            subleito_MR
        ]
        entrada_corrigida = entrada.copy()
        entrada_corrigida[2] = corrigir_mr_para_Tc(revest_MR, Tc)
        ev_med, ev_conf = calcular_confiabilidade_rosenblueth(entrada_corrigida, covs, confianca)
        C = calcular_C(ev_conf)
        N = calcular_N(a, b, vida_util)
        faixas, pc = obter_pc_por_faixa(nome, (revest_h + base_h + subbase_h) * 100, wander_std)
        cdf = N / (pc * C)
        cdf_total += cdf
        st.success(f"{nome} - Deformação confiável: {ev_conf:.6e} mm/mm")

st.markdown("---")
st.subheader("Distribuição Acumulada de Carga (CDF)")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(faixas_ref, cdf_total, label="CDF Total", color="black", linewidth=2)
ax.axhline(1, color='red', linestyle='--', label="Limite CDF = 1")
ax.set_xlabel("Posição lateral (cm)")
ax.set_ylabel("CDF")
ax.grid(True)
ax.legend()
st.pyplot(fig)
