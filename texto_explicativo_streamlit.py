import streamlit as st
st.markdown("""
### ℹ️ Sobre o modelo

Este aplicativo utiliza uma **Rede Neural Artificial (RNA)** treinada para estimar a **deformação vertical no topo do subleito** de pavimentos aeroportuários, considerando diferentes configurações de estrutura e tráfego.

---

### 🌡️ Correção de Temperatura no Revestimento

A **temperatura padrão considerada para os módulos de resiliência (MR)** dos revestimentos asfálticos é de **25°C**.

Para ajustar o MR ao clima local, é utilizada a seguinte **equação do método Austroads (2013)**:

Essa correção permite avaliar o comportamento do revestimento em diferentes regiões climáticas.

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

Esse valor pode ser ajustado para simular diferentes níveis de variação lateral.

---

""")
