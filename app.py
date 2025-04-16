import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import itertools

st.set_page_config(layout="wide")

# === EXPLICAÇÃO DO APP ===
st.markdown("""### ℹ️ Sobre o modelo

Este aplicativo utiliza uma **Rede Neural Artificial (RNA)** treinada para estimar a **deformação vertical no topo do subleito** de pavimentos aeroportuários, considerando diferentes configurações de estrutura e tráfego.

---

### 🌡️ Correção de Temperatura no Revestimento

A **temperatura padrão considerada para os módulos de resiliência (MR)** dos revestimentos asfálticos é de **25°C**.

Para ajustar o MR ao clima local, é utilizada a seguinte **equação do método Austroads (2013)**:

\[
E_T = E_{25} \cdot e^{-0.08 \cdot (T - 25)}
\]

---

### 🎯 Coeficientes de Variação (COV)

O **COV** (Coeficiente de Variação) representa o **grau de incerteza** de uma variável de entrada, expresso em percentual (%).

> 🔒 **Deseja uma análise determinística?** Basta **definir todos os COVs como zero**.

---

### ✈️ Desvio-padrão do Wander

O **wander** é a variação lateral da posição de pouso/decolagem das aeronaves na pista.

> 📌 O **valor padrão adotado é 77,3 cm**, conforme recomendação da **FAA (Federal Aviation Administration)**.
""")

# === AQUI ENTRARIA TODO O RESTANTE DO CÓDIGO FUNCIONAL ===
# Para manter legível neste ambiente, recomendo baixar o arquivo abaixo:
