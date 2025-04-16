
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import itertools

# === EXPLICAÇÃO DO APP ===
st.markdown("""### ℹ️ Sobre o modelo
Este aplicativo utiliza uma **Rede Neural Artificial (RNA)** treinada para estimar a **deformação vertical no topo do subleito** de pavimentos aeroportuários.

---

### 🌡️ Correção de Temperatura no Revestimento
A temperatura padrão para o MR é **25°C**. A correção térmica é feita pelo modelo **Austroads (2013)**:
\( E_T = E_{25} \cdot e^{-0.08 \cdot (T - 25)} \)

---

### 🎯 Coeficientes de Variação (COV)
O **COV** indica a incerteza percentual de cada variável. Para análise determinística, use **COV = 0** em todas as variáveis.

---

### ✈️ Desvio-padrão do Wander
Desvio-padrão lateral da posição de pouso/decolagem da aeronave. Valor padrão: **77,3 cm (FAA)**.
""")

# === CARREGAR MODELO ===
modelo = joblib.load('modelo_4_camadas_ESRS_ESRD_COMPLETO.pkl')
weights = modelo['weights']
scaler = modelo['scaler']
aeronaves_dict = modelo['aeronaves']

# Funções omitidas aqui por brevidade (iguais às anteriores)...

# === EXIBIÇÃO DO GRÁFICO COM BOTÃO ===
executar = st.button("Gerar gráfico CDF")

if executar and entradas_aeronaves:
    for nome, props, a, b in entradas_aeronaves:
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
        ev_med, ev_conf = calcular_confiabilidade_rosenblueth(entrada_corrigida, covs, confianca)
        C = calcular_C(ev_conf)
        N = calcular_N(a, b, vida_util)
        xk = 300
        Ne = 2 if nome != 'CESSNA' else 1
        t = 90
        w = 30
        fx, pc = obter_pc_por_faixa(xk, w, t, Ne, (revest_h + base_h + subbase_h) * 100, wander_std)
        cdf = N / (pc * C)
        cdf_total += cdf

        st.markdown(f"""
**Aeronave: {nome}**  
- MR original: {revest_MR:.2f} MPa | Deformação: {ev_original:.6f} mm/mm  
- MR corrigido: {entrada_corrigida[2]:.2f} MPa | Deformação: {ev_corrigida:.6f} mm/mm  
- Deformação confiável ({confianca:.0f}%): {ev_conf:.6f} mm/mm
""")

    st.markdown("---")
    st.subheader("Distribuição do Dano Acumulado (CDF)")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(fx, cdf_total, label="CDF Total", color="black", linewidth=2)
    ax.axhline(1, color='red', linestyle='--', label="Limite CDF = 1")
    ax.set_xlabel("Posição lateral (cm)")
    ax.set_ylabel("CDF")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)
