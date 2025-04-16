import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import itertools

# === CARREGAR MODELO TREINADO ===
try:
    modelo = joblib.load('modelo_4_camadas_ESRS_ESRD_COMPLETO.pkl')
except FileNotFoundError:
    st.error("Arquivo 'modelo_4_camadas_ESRS_ESRD_COMPLETO.pkl' não encontrado. Certifique-se de que ele está no diretório do projeto.")
    st.stop()

weights = modelo['weights']
scaler = modelo['scaler']
aeronaves_dict = modelo['aeronaves']

# === FUNÇÕES RNA ===
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

# === AUSTROADS: Temperatura média anual ===
def calcular_temperaturas(temp_media_anual):
    return None, temp_media_anual

def corrigir_mr_para_Tc(E25_mpa, temp_media_anual):
    return E25_mpa * np.exp(-0.08 * (temp_media_anual - 25))

# === CÁLCULO DE COBERTURAS ===
def calcular_C(ev):
    # Definir um valor mínimo para ev para evitar problemas numéricos
    ev_min = 1e-6
    ev = max(ev, ev_min)
    
    if ev >= 1.765e-3:
        C = (0.00414131 / ev) ** 8.1
    else:
        # Restaurar a fórmula original, com proteção contra valores inválidos
        termo = max(-0.1638 + 185.19 * ev, 1e-6)
        log_C = termo ** (-0.60586)
        C = 10 ** log_C
    
    # Definir limites para C
    C_max = 1e10
    C_min = 1e-2
    return min(max(C, C_min), C_max)

def calcular_N(a, b, L):
    return (1 + (b * L / 200)) * a * L

def calcular_cp(faixa_centro, weq, pos_pneus, wander_std):
    return sum([
        (stats.norm.cdf(faixa_centro + weq/2, pneu_pos, wander_std) -
         stats.norm.cdf(faixa_centro - weq/2, pneu_pos, wander_std))
        for pneu_pos in pos_pneus
    ])

def obter_pc_por_faixa(aeronave_nome, h_total, dados_aeronaves, wander_std=77.3):
    faixa_largura = 25.4
    num_faixas = 81
    faixas = np.linspace(-40 * faixa_largura, 40 * faixa_largura, num_faixas)
    for aero in dados_aeronaves:
        if aero['nome'] == aeronave_nome:
            if aero['Ne'] == 1 or h_total >= aero['t'] - aero['w']:
                weq = aero['w'] + aero['t'] + h_total if aero['Ne'] > 1 else aero['w'] + h_total
                posicoes_pneus = [aero['xk_centro']]
            else:
                weq = aero['w'] + h_total
                deslocamento = aero['t'] / 2
                posicoes_pneus = [aero['xk_centro'] - deslocamento, aero['xk_centro'] + deslocamento]
            posicoes_pneus_simetrico = posicoes_pneus + [-pos for pos in posicoes_pneus]
            cp_values = np.array([calcular_cp(f, weq, posicoes_pneus_simetrico, wander_std) for f in faixas])
            pc_values = 1 / cp_values
            return faixas, pc_values
    raise ValueError(f"Aeronave {aeronave_nome} não encontrada.")

# === DADOS DAS AERONAVES ===
dados_aeronaves_completos = [
    {'nome': 'A-320',   'w': 31.5,   't': 92.69, 'xk_centro': 333.15, 'Ne': 2},
    {'nome': 'A-321',   'w': 34.7, 't': 92.71, 'xk_centro': 379.5,  'Ne': 2},
    {'nome': 'B-737',   'w': 32.3, 't': 86.36, 'xk_centro': 285.75, 'Ne': 2},
    {'nome': 'EMB-195', 'w': 29.2, 't': 86.36, 'xk_centro': 297.18, 'Ne': 2},
    {'nome': 'ATR-72',  'w': 22.6, 't': 43.60, 'xk_centro': 205.0,  'Ne': 2},
    {'nome': 'CESSNA',  'w': 16.9, 't': 0, 'xk_centro': 208.28,  'Ne': 1},
]

# === INTERFACE DO STREAMLIT ===
st.title("Análise de Pavimento Aeroportuário")

st.markdown("""
Bem-vindo ao nosso programa! 🚀 Este aplicativo utiliza uma **Rede Neural Artificial (RNA)** para o dimensionamento de pavimentos aeroportuários. 🛫

- **Análise Probabilística**: Os **Coeficientes de Variação (COV)** permitem considerar incertezas nas propriedades do pavimento. Insira valores de COV para realizar uma análise probabilística. 📊
- **Análise Determinística**: Se preferir uma análise sem incertezas, basta definir todos os COVs como **0**. ✅
- **Correção de Temperatura**: Ajustamos o módulo de rigidez (MR) com base na temperatura média anual, seguindo as diretrizes do **Austroads (2013)**. 🌡️

Preencha os campos abaixo e clique em "Gerar gráfico CDF acumulado" para ver os resultados! 😄
""")

st.header("Características do Pavimento e Temperatura")
col1, col2 = st.columns(2)
with col1:
    temp_anual = st.number_input("Temp. média anual (°C)", value=25.0, step=0.1)
    revest_E = st.number_input("MR 25°C (MPa) Revestimento", value=2500.0, step=1.0)
    base_E = st.number_input("MR base (MPa)", value=300.0, step=1.0)
    subbase_E = st.number_input("MR subbase (MPa)", value=150.0, step=1.0)
with col2:
    revest_h = st.number_input("h revest. (m)", value=0.3, step=0.01)
    base_h = st.number_input("h base (m)", value=0.3, step=0.01)
    subbase_h = st.number_input("h subbase (m)", value=0.4, step=0.01)

subleito_E = st.number_input("MR subleito (MPa)", value=60.0, step=1.0)
col3, col4 = st.columns(2)
with col3:
    vida_util = st.number_input("Vida útil (anos)", value=20, step=1)
with col4:
    wander_std = st.number_input("Wander std (cm)", value=77.3, step=0.1)

st.markdown(
    "*O valor padrão do desvio-padrão do wander é 77,3 cm, conforme recomendação da FAA. "
    "Altere se desejar simular outros cenários de distribuição lateral.*"
)

# === CONFIABILIDADE ===
st.header("Parâmetros de Confiabilidade")
st.markdown(
    "*Os Coeficientes de Variação (COV) representam o grau de incerteza associado a cada variável de entrada. "
    "Eles são expressos em porcentagem e indicam o quanto um parâmetro pode variar em torno de seu valor médio. "
    "Se desejar uma análise determinística (sem incertezas), defina todos os COVs como zero. Os COVs dispostos foram de acordo com a literatura.*"
)

covs_padrao = {
    'revest_MR': 15, 'base_MR': 20, 'subbase_MR': 20, 'subleito_MR': 20,
    'revest_h': 7, 'base_h': 12, 'subbase_h': 15
}
cov_values = {}
col5, col6 = st.columns(2)
for i, (key, default) in enumerate(covs_padrao.items()):
    with col5 if i % 2 == 0 else col6:
        cov_values[key] = st.number_input(f"COV {key} (%)", value=float(default), step=0.1)

nivel_confianca = st.number_input("Confiabilidade (%)", value=95.0, step=0.1)

def calcular_confiabilidade_rosenblueth(aeronave, media_inputs):
    indices = {
        'revest_MR': 2, 'revest_h': 3,
        'base_MR': 4, 'base_h': 5,
        'subbase_MR': 6, 'subbase_h': 7,
        'subleito_MR': 8
    }
    variaveis = list(indices.items())
    mu_vector = media_inputs.copy()
    covs = [cov_values[k] / 100 for k, _ in variaveis]

    if all(cov == 0 for cov in covs):
        st.warning("Todos os COVs estão zerados. A análise será determinística (sem incertezas).")

    combinacoes = list(itertools.product([-1, 1], repeat=7))
    resultados = []
    for sinais in combinacoes:
        v = mu_vector.copy()
        for i, (_, idx) in enumerate(variaveis):
            v[idx] = mu_vector[idx] * (1 + sinais[i] * covs[i])
        f = prever_deformacao_customizada(v)
        resultados.append(f)

    media = np.mean(resultados)
    desvio = np.std(resultados)
    z = stats.norm.ppf(nivel_confianca / 100)
    confiavel = media + z * desvio
    return media, confiavel

# === AERONAVES ===
st.header("Dados de Aeronaves (Decolagens e Crescimento)")
aeronave_inputs = {}
for nome in sorted(aeronaves_dict.keys()):
    col7, col8 = st.columns(2)
    with col7:
        dec = st.number_input(f"{nome} - a (decolagens)", value=0, step=1, key=f"dec_{nome}")
    with col8:
        cres = st.number_input(f"{nome} - b (%)", value=0.0, step=0.1, key=f"cres_{nome}")
    aeronave_inputs[nome] = (dec, cres)

# === EXECUÇÃO ===
if st.button("Ger