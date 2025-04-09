# app.py - versão para Streamlit

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import itertools

# === CARREGAR MODELO TREINADO ===
modelo = joblib.load('modelo_4_camadas_ESRD_COMPLETO.pkl')
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

# === TEMPERATURA ===
def corrigir_mr_para_Tc(E25_mpa, temp_media_anual):
    return E25_mpa * np.exp(-0.08 * (temp_media_anual - 25))

# === CONFIABILIDADE ===
def calcular_confiabilidade_rosenblueth(aeronave, media_inputs, cov_dict, confianca):
    indices = {
        'revest_E': 2, 'revest_h': 3,
        'base_E': 4, 'base_h': 5,
        'subbase_E': 6, 'subbase_h': 7,
        'subleito_E': 8
    }
    variaveis = list(indices.items())
    mu_vector = media_inputs.copy()
    cov_values = [cov_dict[k] / 100 for k, _ in variaveis]
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

# === COBERTURA ===
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

# === APP STREAMLIT ===
st.title("Análise de CDF Multiaeronaves com Confiabilidade")

# Entradas do usuário
temp_anual = st.number_input('Temperatura média anual (°C)', value=27.0)
revest_E = st.number_input('MR 25°C (MPa)', value=2500.0)
revest_h = st.number_input('h revest. (m)', value=0.3)
base_E = st.number_input('E base (MPa)', value=300.0)
base_h = st.number_input('h base (m)', value=0.3)
subbase_E = st.number_input('E subbase (MPa)', value=150.0)
subbase_h = st.number_input('h subbase (m)', value=0.4)
subleito_E = st.number_input('E subleito (MPa)', value=60.0)
vida_util = st.number_input('Vida útil (anos)', value=20)

st.markdown("### Coeficientes de Variação (COV)")
covs = {
    'revest_E': st.number_input('COV revest_E (%)', value=15.0),
    'revest_h': st.number_input('COV revest_h (%)', value=7.0),
    'base_E': st.number_input('COV base_E (%)', value=20.0),
    'base_h': st.number_input('COV base_h (%)', value=12.0),
    'subbase_E': st.number_input('COV subbase_E (%)', value=20.0),
    'subbase_h': st.number_input('COV subbase_h (%)', value=15.0),
    'subleito_E': st.number_input('COV subleito_E (%)', value=20.0),
}
nivel_confianca = st.number_input('Nível de confiança (%)', value=95.0)

# AERONAVES
aeronave_selecionadas = {}
st.markdown("### Dados de Aeronaves")
for nome in sorted(aeronaves_dict.keys()):
    col1, col2 = st.columns(2)
    with col1:
        dec = st.number_input(f'{nome} - a (decolagens)', min_value=0, value=0, key=f"a_{nome}")
    with col2:
        cres = st.number_input(f'{nome} - b (%) crescimento)', value=0.0, key=f"b_{nome}")
    if dec > 0:
        aeronave_selecionadas[nome] = (dec, cres)

# BOTÃO EXECUTAR
if st.button("Gerar gráfico CDF acumulado"):
    faixas = np.linspace(-40 * 25.4, 40 * 25.4, 81)
    cdf_total = np.zeros_like(faixas)
    fig, ax = plt.subplots(figsize=(12, 6))

    revest_E_corrigido = corrigir_mr_para_Tc(revest_E, temp_anual)
    h_total_cm = (revest_h + base_h + subbase_h) * 100
    fx_ref = None

    for nome, (dec, cres) in aeronave_selecionadas.items():
        entrada_original = [
            aeronaves_dict[nome]['Pressão dos Pneus(MPa)'],
            aeronaves_dict[nome]['dist_rodas(m)'],
            revest_E, revest_h, base_E, base_h, subbase_E, subbase_h, subleito_E
        ]
        entrada_corrigida = entrada_original.copy()
        entrada_corrigida[2] = revest_E_corrigido

        ev_media, ev_confiavel = calcular_confiabilidade_rosenblueth(nome, entrada_corrigida, covs, nivel_confianca)

        st.write(f"#### {nome}")
        st.write(f"MR corrigido: {revest_E_corrigido:.2f} MPa")
        st.write(f"Deformação confiável ({nivel_confianca:.0f}%): {ev_confiavel:.6f} mm/mm")

        C = calcular_C(ev_confiavel)
        N = calcular_N(dec, cres, vida_util)
        fx, pc = obter_pc_por_faixa(nome, h_total_cm, dados_aeronaves_completos)
        if fx_ref is None:
            fx_ref = fx
        cdf = N / (pc * C)
        cdf_total += cdf
        ax.plot(fx, cdf, label=f"{nome}")

    ax.plot(fx_ref, cdf_total, color='black', linestyle='--', linewidth=3, label='CDF Total')
    ax.axhline(1, color='red', linestyle='--', label='Limite CDF = 1')
    ax.set_title('Distribuição do CDF ao longo da largura da pista')
    ax.set_xlabel('Posição lateral na pista (cm)')
    ax.set_ylabel('CDF')
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)
