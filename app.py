
# (...) [código anterior permanece o mesmo até a parte do botão "executar"]

    st.subheader("Distribuição do Dano Acumulado (CDF)")
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
        _, ev_conf = calcular_confiabilidade_rosenblueth(entrada_corrigida, covs, confianca)
        C = calcular_C(ev_conf)
        N = calcular_N(a, b, vida_util)
        fx, pc = obter_pc(300, 30, 90, 2 if nome != "CESSNA" else 1, (revest_h + base_h + subbase_h) * 100, wander_std)
        cdf = N / (pc * C)
        ax.plot(fx, cdf, label=nome)
        cdf_total += cdf

    ax.plot(fx, cdf_total, label="CDF Total (soma)", color="black", linestyle="--", linewidth=3)
    ax.axhline(1, color="red", linestyle="--", label="Limite CDF = 1")
    ax.set_title("Distribuição do CDF ao longo da largura da pista")
    ax.set_xlabel("Posição lateral na pista (cm)")
    ax.set_ylabel("CDF")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)
