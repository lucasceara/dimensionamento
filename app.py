# Imports iniciais
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import ipywidgets as widgets
from IPython.display import display, clear_output
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

# === AUSTROADS: Temperatura média anual ===
def calcular_temperaturas(temp_media_anual):
    return None, temp_media_anual

def corrigir_mr_para_Tc(E25_mpa, temp_media_anual):
    return E25_mpa * np.exp(-0.08 * (temp_media_anual - 25))

# === INTERFACE PARA CONFIABILIDADE ===
covs_padrao = {
    'revest_E': 15, 'base_E': 20, 'subbase_E': 20, 'subleito_E': 20,
    'revest_h': 7, 'base_h': 12, 'subbase_h': 15
}

cov_widgets = {
    key: widgets.FloatText(value=val, description=f'COV {key} (%)')
    for key, val in covs_padrao.items()
}

nivel_confianca = widgets.FloatText(value=95.0, description='Confiabilidade (%)')

confiab_section = widgets.VBox([
    widgets.HTML('<b>Coeficientes de Variação (COV):</b>')
] + list(cov_widgets.values()) + [nivel_confianca])

def calcular_confiabilidade_rosenblueth(aeronave, media_inputs):
    indices = {
        'revest_E': 2, 'revest_h': 3,
        'base_E': 4, 'base_h': 5,
        'subbase_E': 6, 'subbase_h': 7,
        'subleito_E': 8
    }
    variaveis = list(indices.items())
    mu_vector = media_inputs.copy()
    cov_values = [cov_widgets[k].value / 100 for k, _ in variaveis]
    
    # Geração de todas as combinações +1/-1 (total 2^7 = 128)
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
    z = stats.norm.ppf(nivel_confianca.value / 100)
    confiavel = media + z * desvio
    return media, confiavel

# === CÁLCULO DE COBERTURAS ===
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

# === DADOS DAS AERONAVES ===
dados_aeronaves_completos = [
    {'nome': 'A-320',   'w': 31.5,   't': 92.69, 'xk_centro': 333.15, 'Ne': 2},
    {'nome': 'A-321',   'w': 32.240, 't': 92.71, 'xk_centro': 379.5,  'Ne': 2},
    {'nome': 'B-737',   'w': 30.062, 't': 86.36, 'xk_centro': 285.75, 'Ne': 2},
    {'nome': 'EMB-195', 'w': 27.194, 't': 86.36, 'xk_centro': 297.18, 'Ne': 2},
    {'nome': 'ATR-72',  'w': 21.035, 't': 43.60, 'xk_centro': 205.0,  'Ne': 2},
]
