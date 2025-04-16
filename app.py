
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import itertools

st.set_page_config(layout="wide")

# === EXPLICAÇÃO DO APP ===
st.markdown("""
### ℹ️ Sobre o modelo

Este aplicativo utiliza uma **Rede Neural Artificial (RNA)** treinada para estimar a **deformação vertical no topo do subleito** de pavimentos aeroportuários, considerando diferentes configurações de estrutura e tráfego.

---

### 🌡️ Correção de Temperatura no Revestimento

A **temperatura padrão considerada para os módulos de resiliência (MR)** dos revestimentos asfálticos é de **25°C**.

Para ajustar o MR ao clima local, é utilizada a seguinte **equação do método Austroads (2013)**

---

### 🎯 Coeficientes de Variação (COV)

O **COV** (Coeficiente de Variação) representa o **grau de incerteza** de uma variável de entrada, expresso em percentual (%).

> 🔒 **Deseja uma análise determinística?** Basta **definir todos os COVs como zero**.

---

### ✈️ Desvio-padrão do Wander

O **wander** é a variação lateral da posição de pouso/decolagem das aeronaves na pista.

> 📌 O **valor padrão adotado é 77,3 cm**, conforme recomendação da **FAA (Federal Aviation Administration)**.
""")

modelo = joblib.load('modelo_4_camadas_ESRS_ESRD_COMPLETO.pkl')
weights = modelo['weights']
scaler = modelo['scaler']
aeronaves_dict = modelo['aeronaves']

def tanh(x): return np.tanh(x)
def neural_network(weights, X):
    W1 = weights[:90].reshape((9, 10))
    b1 = weights[90:100]
    W2 = weights[100:180].reshape((10, 8))
    b2 = weights[180:188]
    W3 = weights[-9:-1].reshape((8, 1))
    b3 = weights[-1]
    return np.dot(tanh(np.dot(tanh(np.dot(X, W1) + b1), W2) + b2), W3) + b3

def prever_deformacao_customizada(valores):
    entrada = np.array([valores])
    entrada_norm = scaler.transform(entrada)
    return neural_network(weights, entrada_norm)[0, 0]

def corrigir_mr_para_Tc(E25, Tc): return E25 * np.exp(-0.08 * (Tc - 25))

def calcular_confiabilidade_rosenblueth(media_inputs, covs, confianca):
    idx = {'revest_MR': 2, 'revest_h': 3, 'base_MR': 4, 'base_h': 5, 'subbase_MR': 6, 'subbase_h': 7, 'subleito_MR': 8}
    cov_values = [covs[k] / 100 for k in idx]
    sinais = list(itertools.product([-1, 1], repeat=7))
    resultados = []
    for s in sinais:
        v = media_inputs.copy()
        for i, key in enumerate(idx):
            v[idx[key]] *= (1 + s[i] * cov_values[i])
        resultados.append(prever_deformacao_customizada(v))
    media, desvio = np.mean(resultados), np.std(resultados)
    z = stats.norm.ppf(confianca / 100)
    return media, media + z * desvio

def calcular_C(ev):
    return (0.00414131 / ev) ** 8.1 if ev >= 1.765e-3 else 10 ** ((-0.1638 + 185.19 * ev) ** (-0.60586))

def calcular_N(a, b, L): return (1 + (b * L / 200)) * a * L

def calcular_cp(c, w, pos, std):
    return sum(stats.norm.cdf(c + w/2, p, std) - stats.norm.cdf(c - w/2, p, std) for p in pos)

def obter_pc(xk, w, t, Ne, h_total, std):
    f = np.linspace(-40 * 25.4, 40 * 25.4, 81)
    if Ne == 1 or h_total >= t - w:
        weq = w + t + h_total if Ne > 1 else w + h_total
        pos = [xk]
    else:
        weq = w + h_total
        d = t / 2
        pos = [xk - d, xk + d]
    pos += [-p for p in pos]
    cp = np.array([calcular_cp(fi, weq, pos, std) for fi in f])
    return f, 1 / cp

st.title("Predição de Deformações - Pavimentos Aeroportuários")

Tc = st.number_input("Temperatura média anual (°C)", value=27.0)
revest_MR = st.number_input("MR 25°C (MPa)", value=2500.0)
revest_h = st.number_input("h revest. (m)", value=0.3)
base_MR = st.number_input("MR base (MPa)", value=300.0)
base_h = st.number_input("h base (m)", value=0.3)
subbase_MR = st.number_input("MR subbase (MPa)", value=150.0)
subbase_h = st.number_input("h subbase (m)", value=0.4)
subleito_MR = st.number_input("MR subleito (MPa)", value=60.0)
vida_util = st.number_input("Vida útil (anos)", value=20.0)
wander_std = st.number_input("Wander std (cm)", value=77.3)

st.subheader("Coeficientes de Variação")
cov_padrao = {'revest_MR': 15, 'base_MR': 20, 'subbase_MR': 20, 'subleito_MR': 20, 'revest_h': 7, 'base_h': 12, 'subbase_h': 15}
covs = {k: st.number_input(f"COV {k} (%)", value=v) for k, v in cov_padrao.items()}
confianca = st.number_input("Confiabilidade (%)", value=95.0)

st.subheader("Aeronaves e tráfego")
aeronaves_ativas = []
for nome, props in aeronaves_dict.items():
    col1, col2 = st.columns(2)
    with col1:
        a = st.number_input(f"{nome} - decolagens anuais", value=0, key=f"{nome}_a")
    with col2:
        b = st.number_input(f"{nome} - crescimento b (%)", value=0.0, key=f"{nome}_b")
    if a > 0:
        aeronaves_ativas.append((nome, props, a, b))

executar = st.button("Gerar gráfico CDF acumulado")

if executar:
    cdf_total = np.zeros(81)
    fx = np.linspace(-40 * 25.4, 40 * 25.4, 81)
    fig, ax = plt.subplots(figsize=(12, 6))

    for nome, props, a, b in aeronaves_ativas:
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
        ev_original = prever_deformacao_customizada(entrada)
        ev_corrigida = prever_deformacao_customizada(entrada_corrigida)
        _, ev_conf = calcular_confiabilidade_rosenblueth(entrada_corrigida, covs, confianca)
        C = calcular_C(ev_conf)
        N = calcular_N(a, b, vida_util)
        fxi, pci = obter_pc(300, 30, 90, 2 if nome != "CESSNA" else 1, (revest_h + base_h + subbase_h) * 100, wander_std)
        cdf = N / (pci * C)
        cdf_total += cdf

        st.markdown(f"""**Aeronave: {nome}**  
- MR original: {revest_MR:.2f} MPa | Deformação: {ev_original:.6f} mm/mm  
- MR corrigido: {entrada_corrigida[2]:.2f} MPa | Deformação: {ev_corrigida:.6f} mm/mm  
- Deformação confiável ({confianca:.0f}%): {ev_conf:.6f} mm/mm""")

        ax.plot(fxi, cdf, label=nome)

    ax.plot(fx, cdf_total, label="CDF Total (soma)", color="black", linestyle="--", linewidth=3)
    ax.axhline(1, color="red", linestyle="--", label="Limite CDF = 1")
    ax.set_title("Distribuição do CDF ao longo da largura da pista")
    ax.set_xlabel("Posição lateral na pista (cm)")
    ax.set_ylabel("CDF")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)
