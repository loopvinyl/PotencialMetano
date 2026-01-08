# =============================================================================
# NOVAS FUNÇÕES PARA ENTRADA CONTÍNUA (INTEGRANDO CÁLCULOS DO SCRIPT 2)
# =============================================================================

def calcular_emissoes_aterro_completo_continuo(residuos_kg_dia, umidade, temperatura, doc_val, 
                                               massa_exposta_kg, h_exposta, dias_simulacao):
    """
    Calcula CH₄ + N₂O do aterro para entrada contínua
    Baseado no Script 2 (Zziwa et al. adaptado)
    """
    # Parâmetros fixos do aterro
    MCF = 1.0
    F = 0.5
    OX = 0.1
    Ri = 0.0
    k_ano = 0.06
    
    # 1. CÁLCULO DE CH₄ (METANO)
    DOCf = 0.0147 * temperatura + 0.28
    potencial_CH4_por_kg = doc_val * DOCf * MCF * F * (16/12) * (1 - Ri) * (1 - OX)
    potencial_CH4_lote_diario = residuos_kg_dia * potencial_CH4_por_kg
    
    # Perfil temporal de decaimento
    t = np.arange(1, dias_simulacao + 1, dtype=float)
    kernel_ch4 = np.exp(-k_ano * (t - 1) / 365.0) - np.exp(-k_ano * t / 365.0)
    kernel_ch4 = kernel_ch4 / kernel_ch4.sum()  # Normalizar
    
    entradas_diarias = np.ones(dias_simulacao, dtype=float)
    emissoes_CH4 = np.convolve(entradas_diarias, kernel_ch4, mode='full')[:dias_simulacao]
    emissoes_CH4 *= potencial_CH4_lote_diario
    
    # 2. CÁLCULO DE N₂O (ÓXIDO NITROSO)
    fator_umid = (1 - umidade) / (1 - 0.55)
    f_aberto = np.clip((massa_exposta_kg / residuos_kg_dia) * (h_exposta / 24), 0.0, 1.0)
    
    E_aberto = 1.91  # g N₂O-N/ton
    E_fechado = 2.15  # g N₂O-N/ton
    E_medio = f_aberto * E_aberto + (1 - f_aberto) * E_fechado
    E_medio_ajust = E_medio * fator_umid
    
    # Emissão diária de N₂O (kg/dia)
    emissao_diaria_N2O = (E_medio_ajust * (44/28) / 1_000_000) * residuos_kg_dia
    
    # Perfil temporal de N₂O (5 dias - Wang et al. 2017)
    kernel_n2o = np.array([0.10, 0.30, 0.40, 0.15, 0.05], dtype=float)
    emissoes_N2O = np.convolve(np.full(dias_simulacao, emissao_diaria_N2O), kernel_n2o, mode='full')[:dias_simulacao]
    
    # 3. EMISSÕES PRÉ-DESCARTE (Feng et al. 2020)
    CH4_pre_descarte_ugC_por_kg_h_media = 2.78
    fator_conversao_C_para_CH4 = 16/12
    CH4_pre_descarte_ugCH4_por_kg_h_media = CH4_pre_descarte_ugC_por_kg_h_media * fator_conversao_C_para_CH4
    CH4_pre_descarte_g_por_kg_dia = CH4_pre_descarte_ugCH4_por_kg_h_media * 24 / 1_000_000
    
    N2O_pre_descarte_mgN_por_kg = 20.26
    N2O_pre_descarte_mgN_por_kg_dia = N2O_pre_descarte_mgN_por_kg / 3
    N2O_pre_descarte_g_por_kg_dia = N2O_pre_descarte_mgN_por_kg_dia * (44/28) / 1000
    
    emissoes_CH4_pre_descarte_kg = np.full(dias_simulacao, residuos_kg_dia * CH4_pre_descarte_g_por_kg_dia / 1000)
    emissoes_N2O_pre_descarte_kg = np.zeros(dias_simulacao)
    
    # Perfil N₂O pré-descarte (3 dias)
    PERFIL_N2O_PRE_DESCARTE = {1: 0.8623, 2: 0.10, 3: 0.0377}
    
    for dia_entrada in range(dias_simulacao):
        for dias_apos_descarte, fracao in PERFIL_N2O_PRE_DESCARTE.items():
            dia_emissao = dia_entrada + dias_apos_descarte - 1
            if dia_emissao < dias_simulacao:
                emissoes_N2O_pre_descarte_kg[dia_emissao] += (
                    residuos_kg_dia * N2O_pre_descarte_g_por_kg_dia * fracao / 1000
                )
    
    # 4. TOTAL DE EMISSÕES DO ATERRO
    total_ch4_aterro_kg = emissoes_CH4 + emissoes_CH4_pre_descarte_kg
    total_n2o_aterro_kg = emissoes_N2O + emissoes_N2O_pre_descarte_kg
    
    return total_ch4_aterro_kg, total_n2o_aterro_kg, DOCf

def calcular_emissoes_vermi_completo_continuo(residuos_kg_dia, umidade, dias_simulacao):
    """
    Calcula CH₄ + N₂O da vermicompostagem para entrada contínua
    Baseado em Yang et al. (2017)
    """
    # Parâmetros fixos
    TOC_YANG = 0.436  # Fração de carbono orgânico total
    TN_YANG = 14.2 / 1000  # Fração de nitrogênio total
    CH4_C_FRAC_YANG = 0.13 / 100  # 0.13%
    N2O_N_FRAC_YANG = 0.92 / 100  # 0.92%
    
    fracao_ms = 1 - umidade
    
    # Metano total por lote diário
    ch4_total_por_lote_diario = residuos_kg_dia * (TOC_YANG * CH4_C_FRAC_YANG * (16/12) * fracao_ms)
    
    # Óxido nitroso total por lote diário
    n2o_total_por_lote_diario = residuos_kg_dia * (TN_YANG * N2O_N_FRAC_YANG * (44/28) * fracao_ms)
    
    # Perfis temporais (50 dias)
    PERFIL_CH4_VERMI = np.array([
        0.02, 0.02, 0.02, 0.03, 0.03, 0.04, 0.04, 0.05, 0.05, 0.06,
        0.07, 0.08, 0.09, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04,
        0.03, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
        0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005,
        0.002, 0.002, 0.002, 0.002, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001
    ])
    PERFIL_CH4_VERMI /= PERFIL_CH4_VERMI.sum()
    
    PERFIL_N2O_VERMI = np.array([
        0.15, 0.10, 0.20, 0.05, 0.03, 0.03, 0.03, 0.04, 0.05, 0.06,
        0.08, 0.09, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02,
        0.01, 0.01, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005,
        0.002, 0.002, 0.002, 0.002, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001,
        0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001
    ])
    PERFIL_N2O_VERMI /= PERFIL_N2O_VERMI.sum()
    
    # Inicializar arrays de emissões
    emissoes_CH4 = np.zeros(dias_simulacao)
    emissoes_N2O = np.zeros(dias_simulacao)
    
    # Convolução para entrada contínua
    for dia_entrada in range(dias_simulacao):
        for dia_compostagem in range(len(PERFIL_CH4_VERMI)):
            dia_emissao = dia_entrada + dia_compostagem
            if dia_emissao < dias_simulacao:
                emissoes_CH4[dia_emissao] += ch4_total_por_lote_diario * PERFIL_CH4_VERMI[dia_compostagem]
                emissoes_N2O[dia_emissao] += n2o_total_por_lote_diario * PERFIL_N2O_VERMI[dia_compostagem]
    
    return emissoes_CH4, emissoes_N2O

def calcular_emissoes_compostagem_completo_continuo(residuos_kg_dia, umidade, dias_simulacao):
    """
    Calcula CH₄ + N₂O da compostagem termofílica para entrada contínua
    Baseado em Yang et al. (2017)
    """
    # Parâmetros fixos
    TOC_YANG = 0.436
    TN_YANG = 14.2 / 1000
    CH4_C_FRAC_THERMO = 0.006  # 0.6%
    N2O_N_FRAC_THERMO = 0.0196  # 1.96%
    
    fracao_ms = 1 - umidade
    
    # Totais por lote diário
    ch4_total_por_lote_diario = residuos_kg_dia * (TOC_YANG * CH4_C_FRAC_THERMO * (16/12) * fracao_ms)
    n2o_total_por_lote_diario = residuos_kg_dia * (TN_YANG * N2O_N_FRAC_THERMO * (44/28) * fracao_ms)
    
    # Perfis temporais (50 dias)
    PERFIL_CH4_THERMO = np.array([
        0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.15, 0.18, 0.20, 0.18,
        0.15, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.02,
        0.01, 0.01, 0.01, 0.01, 0.01, 0.005, 0.005, 0.005, 0.005, 0.005,
        0.002, 0.002, 0.002, 0.002, 0.002, 0.001, 0.001, 0.001, 0.001, 0.001,
        0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001
    ])
    PERFIL_CH4_THERMO /= PERFIL_CH4_THERMO.sum()
    
    PERFIL_N2O_THERMO = np.array([
        0.10, 0.08, 0.15, 0.05, 0.03, 0.04, 0.05, 0.07, 0.10, 0.12,
        0.15, 0.18, 0.20, 0.18, 0.15, 0.12, 0.10, 0.08, 0.06, 0.05,
        0.04, 0.03, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
        0.005, 0.005, 0.005, 0.005, 0.005, 0.002, 0.002, 0.002, 0.002, 0.002,
        0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001
    ])
    PERFIL_N2O_THERMO /= PERFIL_N2O_THERMO.sum()
    
    # Inicializar arrays
    emissoes_CH4 = np.zeros(dias_simulacao)
    emissoes_N2O = np.zeros(dias_simulacao)
    
    # Convolução
    for dia_entrada in range(dias_simulacao):
        for dia_compostagem in range(len(PERFIL_CH4_THERMO)):
            dia_emissao = dia_entrada + dia_compostagem
            if dia_emissao < dias_simulacao:
                emissoes_CH4[dia_emissao] += ch4_total_por_lote_diario * PERFIL_CH4_THERMO[dia_compostagem]
                emissoes_N2O[dia_emissao] += n2o_total_por_lote_diario * PERFIL_N2O_THERMO[dia_compostagem]
    
    return emissoes_CH4, emissoes_N2O

# =============================================================================
# MODIFICAÇÃO DA INTERFACE PARA DUAS ABAS
# =============================================================================

# No início do script, após o título:
st.title("🔬 Estimação do Potencial de Emissões - Comparação Completa")

# Criar abas
tab1, tab2 = st.tabs(["📦 Análise por Lote (100 kg)", "📈 Entrada Contínua (kg/dia)"])

with tab1:
    # Manter o código atual do Script 1 (lote único)
    st.header("Análise por Lote Único de 100 kg")
    # ... (código atual do Script 1)

with tab2:
    st.header("Análise para Entrada Contínua (kg/dia)")
    
    # =============================================================================
    # PAINEL LATERAL ESPECÍFICO PARA ENTRADA CONTÍNUA
    # =============================================================================
    with st.sidebar:
        st.header("⚙️ Parâmetros Entrada Contínua")
        
        # Entrada principal em kg/dia
        residuos_kg_dia = st.number_input(
            "Resíduos orgânicos (kg/dia)", 
            min_value=10, 
            max_value=5000, 
            value=100, 
            step=10,
            help="Quantidade diária de resíduos para processamento contínuo"
        )
        
        st.subheader("📊 Parâmetros Ambientais")
        
        umidade_valor = st.slider(
            "Umidade do resíduo (%) - Contínuo", 
            50, 95, 85, 1,
            help="Percentual de umidade dos resíduos orgânicos"
        )
        umidade = umidade_valor / 100.0
        
        temperatura = st.slider(
            "Temperatura média (°C) - Contínuo", 
            15, 35, 25, 1,
            help="Temperatura média ambiente"
        )
        
        # DOC (Carbono Orgânico Degradável)
        doc_val = st.slider(
            "DOC - Carbono Orgânico Degradável (fração)", 
            0.10, 0.50, 0.15, 0.01,
            help="Fração de carbono orgânico degradável nos resíduos"
        )
        
        st.subheader("🏭 Parâmetros Operacionais do Aterro")
        
        massa_exposta_kg = st.slider(
            "Massa exposta na frente de trabalho (kg)", 
            50, 500, 100, 10,
            help="Massa de resíduos exposta diariamente no aterro"
        )
        
        h_exposta = st.slider(
            "Horas expostas por dia", 
            4, 24, 8, 1,
            help="Horas diárias de exposição dos resíduos no aterro"
        )
        
        st.subheader("⏰ Período de Análise")
        anos_simulacao = st.slider(
            "Anos de simulação - Contínuo", 
            1, 50, 20, 1,
            help="Período total da simulação em anos"
        )
        
        dias_simulacao = anos_simulacao * 365
        
        if st.button("🚀 Calcular Emissões Contínuas", type="primary", key="btn_continuo"):
            st.session_state.run_simulacao_continuo = True

    # =============================================================================
    # EXECUÇÃO DA SIMULAÇÃO CONTÍNUA
    # =============================================================================
    if st.session_state.get('run_simulacao_continuo', False):
        with st.spinner(f'Calculando emissões para {residuos_kg_dia} kg/dia durante {anos_simulacao} anos...'):
            
            # 1. CÁLCULO DAS EMISSÕES COMPLETAS
            # Aterro
            ch4_aterro, n2o_aterro, DOCf = calcular_emissoes_aterro_completo_continuo(
                residuos_kg_dia, umidade, temperatura, doc_val,
                massa_exposta_kg, h_exposta, dias_simulacao
            )
            
            # Vermicompostagem
            ch4_vermi, n2o_vermi = calcular_emissoes_vermi_completo_continuo(
                residuos_kg_dia, umidade, dias_simulacao
            )
            
            # Compostagem
            ch4_compost, n2o_compost = calcular_emissoes_compostagem_completo_continuo(
                residuos_kg_dia, umidade, dias_simulacao
            )
            
            # 2. CRIAR DATAFRAME COM RESULTADOS
            datas = pd.date_range(start=datetime.now(), periods=dias_simulacao, freq='D')
            
            df_continuo = pd.DataFrame({
                'Data': datas,
                'CH4_Aterro_kg_dia': ch4_aterro,
                'N2O_Aterro_kg_dia': n2o_aterro,
                'CH4_Vermi_kg_dia': ch4_vermi,
                'N2O_Vermi_kg_dia': n2o_vermi,
                'CH4_Compost_kg_dia': ch4_compost,
                'N2O_Compost_kg_dia': n2o_compost
            })
            
            # 3. CONVERTER PARA CO₂eq (GWP 20 anos - igual Script 2)
            GWP_CH4_20 = 79.7  # IPCC AR6 - 20 anos
            GWP_N2O_20 = 273   # IPCC AR6 - 20 anos
            
            # Cálculo diário de tCO₂eq
            for gas, gwp in [('CH4', GWP_CH4_20), ('N2O', GWP_N2O_20)]:
                for cenario in ['Aterro', 'Vermi', 'Compost']:
                    col_kg = f'{gas}_{cenario}_kg_dia'
                    col_tco2eq = f'{gas}_{cenario}_tCO2eq_dia'
                    df_continuo[col_tco2eq] = df_continuo[col_kg] * gwp / 1000
            
            # Totais por cenário
            df_continuo['Total_Aterro_tCO2eq_dia'] = (
                df_continuo['CH4_Aterro_tCO2eq_dia'] + df_continuo['N2O_Aterro_tCO2eq_dia']
            )
            df_continuo['Total_Vermi_tCO2eq_dia'] = (
                df_continuo['CH4_Vermi_tCO2eq_dia'] + df_continuo['N2O_Vermi_tCO2eq_dia']
            )
            df_continuo['Total_Compost_tCO2eq_dia'] = (
                df_continuo['CH4_Compost_tCO2eq_dia'] + df_continuo['N2O_Compost_tCO2eq_dia']
            )
            
            # Acumulados
            for cenario in ['Aterro', 'Vermi', 'Compost']:
                col_dia = f'Total_{cenario}_tCO2eq_dia'
                col_acum = f'Total_{cenario}_tCO2eq_acum'
                df_continuo[col_acum] = df_continuo[col_dia].cumsum()
            
            # Reduções (emissões evitadas)
            df_continuo['Reducao_Vermi_tCO2eq_acum'] = (
                df_continuo['Total_Aterro_tCO2eq_acum'] - df_continuo['Total_Vermi_tCO2eq_acum']
            )
            df_continuo['Reducao_Compost_tCO2eq_acum'] = (
                df_continuo['Total_Aterro_tCO2eq_acum'] - df_continuo['Total_Compost_tCO2eq_acum']
            )
            
            # 4. RESULTADOS ANUAIS (agrupamento)
            df_continuo['Ano'] = df_continuo['Data'].dt.year
            df_anual = df_continuo.groupby('Ano').agg({
                'Total_Aterro_tCO2eq_dia': 'sum',
                'Total_Vermi_tCO2eq_dia': 'sum',
                'Total_Compost_tCO2eq_dia': 'sum'
            }).reset_index()
            
            df_anual.rename(columns={
                'Total_Aterro_tCO2eq_dia': 'Aterro_Anual_tCO2eq',
                'Total_Vermi_tCO2eq_dia': 'Vermi_Anual_tCO2eq',
                'Total_Compost_tCO2eq_dia': 'Compost_Anual_tCO2eq'
            }, inplace=True)
            
            df_anual['Reducao_Vermi_Anual_tCO2eq'] = (
                df_anual['Aterro_Anual_tCO2eq'] - df_anual['Vermi_Anual_tCO2eq']
            )
            df_anual['Reducao_Compost_Anual_tCO2eq'] = (
                df_anual['Aterro_Anual_tCO2eq'] - df_anual['Compost_Anual_tCO2eq']
            )
            
            # 5. EXIBIR RESULTADOS (igual ao Script 2)
            st.header("📊 Resultados - Entrada Contínua")
            
            # Totais acumulados
            total_evitado_vermi = df_continuo['Reducao_Vermi_tCO2eq_acum'].iloc[-1]
            total_evitado_compost = df_continuo['Reducao_Compost_tCO2eq_acum'].iloc[-1]
            
            # Médias anuais
            media_anual_vermi = total_evitado_vermi / anos_simulacao
            media_anual_compost = total_evitado_compost / anos_simulacao
            
            # Exibir métricas
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🪱 Vermicompostagem")
                st.metric(
                    "Total de emissões evitadas",
                    f"{formatar_br(total_evitado_vermi)} tCO₂eq",
                    help=f"Acumulado em {anos_simulacao} anos"
                )
                st.metric(
                    "Média anual",
                    f"{formatar_br(media_anual_vermi)} tCO₂eq/ano",
                    help="Emissões evitadas por ano em média"
                )
            
            with col2:
                st.markdown("#### 🌡️ Compostagem Termofílica")
                st.metric(
                    "Total de emissões evitadas",
                    f"{formatar_br(total_evitado_compost)} tCO₂eq",
                    help=f"Acumulado em {anos_simulacao} anos"
                )
                st.metric(
                    "Média anual",
                    f"{formatar_br(media_anual_compost)} tCO₂eq/ano",
                    help="Emissões evitadas por ano em média"
                )
            
            # Diferença percentual
            dif_percentual = ((total_evitado_vermi - total_evitado_compost) / total_evitado_compost * 100) if total_evitado_compost > 0 else 0
            
            st.info(f"""
            **📈 Comparação:** A vermicompostagem evita **{dif_percentual:+.1f}%** mais emissões 
            que a compostagem termofílica ({formatar_br(total_evitado_vermi - total_evitado_compost)} tCO₂eq de diferença).
            """)
            
            # 6. GRÁFICO DE REDUÇÃO ACUMULADA
            st.subheader("📉 Redução de Emissões Acumulada")
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            ax.plot(df_continuo['Data'], df_continuo['Total_Aterro_tCO2eq_acum'], 
                   'r-', label='Cenário Base (Aterro)', linewidth=2, alpha=0.8)
            ax.plot(df_continuo['Data'], df_continuo['Total_Vermi_tCO2eq_acum'], 
                   'g-', label='Vermicompostagem', linewidth=2)
            ax.plot(df_continuo['Data'], df_continuo['Total_Compost_tCO2eq_acum'], 
                   'b-', label='Compostagem Termofílica', linewidth=2)
            
            # Área de redução
            ax.fill_between(df_continuo['Data'], 
                           df_continuo['Total_Vermi_tCO2eq_acum'], 
                           df_continuo['Total_Aterro_tCO2eq_acum'],
                           color='green', alpha=0.2, label='Redução Vermicompostagem')
            ax.fill_between(df_continuo['Data'], 
                           df_continuo['Total_Compost_tCO2eq_acum'], 
                           df_continuo['Total_Aterro_tCO2eq_acum'],
                           color='blue', alpha=0.1, label='Redução Compostagem')
            
            ax.set_title(f'Emissões Acumuladas - {residuos_kg_dia} kg/dia × {anos_simulacao} anos', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Data')
            ax.set_ylabel('tCO₂eq Acumulado')
            ax.legend(title='Cenário de Gestão', loc='upper left')
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.yaxis.set_major_formatter(FuncFormatter(br_format))
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            
            # 7. GRÁFICO ANUAL COMPARATIVO
            st.subheader("📊 Comparação Anual das Emissões Evitadas")
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            x = np.arange(len(df_anual['Ano']))
            bar_width = 0.35
            
            ax.bar(x - bar_width/2, df_anual['Reducao_Vermi_Anual_tCO2eq'], bar_width,
                   label='Vermicompostagem', color='green', alpha=0.8)
            ax.bar(x + bar_width/2, df_anual['Reducao_Compost_Anual_tCO2eq'], bar_width,
                   label='Compostagem Termofílica', color='blue', alpha=0.8)
            
            # Adicionar valores
            for i, (v1, v2) in enumerate(zip(df_anual['Reducao_Vermi_Anual_tCO2eq'], 
                                            df_anual['Reducao_Compost_Anual_tCO2eq'])):
                if v1 > 0:
                    ax.text(i - bar_width/2, v1 + max(v1, v2)*0.02, 
                           formatar_br(v1), ha='center', fontsize=8, fontweight='bold')
                if v2 > 0:
                    ax.text(i + bar_width/2, v2 + max(v1, v2)*0.02, 
                           formatar_br(v2), ha='center', fontsize=8, fontweight='bold')
            
            ax.set_xlabel('Ano')
            ax.set_ylabel('Emissões Evitadas (tCO₂eq/ano)')
            ax.set_title('Redução Anual de Emissões por Cenário')
            ax.set_xticks(x)
            ax.set_xticklabels(df_anual['Ano'], rotation=45)
            ax.legend()
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            ax.yaxis.set_major_formatter(FuncFormatter(br_format))
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # 8. RESUMO DETALHADO
            with st.expander("📋 Detalhamento dos Cálculos - Entrada Contínua"):
                st.markdown(f"""
                ### **Metodologia de Cálculo - Entrada Contínua**
                
                **Parâmetros Utilizados:**
                - Resíduos: {residuos_kg_dia} kg/dia ({residuos_kg_dia*365/1000:.1f} ton/ano)
                - Umidade: {umidade_valor}% ({umidade:.2f} fração)
                - Temperatura: {temperatura}°C
                - DOC: {doc_val:.3f}
                - Período: {anos_simulacao} anos ({dias_simulacao} dias)
                - Massa exposta: {massa_exposta_kg} kg
                - Horas expostas: {h_exposta}h/dia
                
                **Fatores GWP (IPCC AR6 - 20 anos):**
                - CH₄: {GWP_CH4_20} kg CO₂eq/kg CH₄
                - N₂O: {GWP_N2O_20} kg CO₂eq/kg N₂O
                
                **Resultados Acumulados ({anos_simulacao} anos):**
                - **Vermicompostagem:** {formatar_br(total_evitado_vermi)} tCO₂eq evitadas
                - **Compostagem Termofílica:** {formatar_br(total_evitado_compost)} tCO₂eq evitadas
                - **Diferença:** {formatar_br(total_evitado_vermi - total_evitado_compost)} tCO₂eq ({dif_percentual:+.1f}%)
                
                **Médias Anuais:**
                - Vermicompostagem: {formatar_br(media_anual_vermi)} tCO₂eq/ano
                - Compostagem: {formatar_br(media_anual_compost)} tCO₂eq/ano
                
                **Equivalências:**
                - Vermicompostagem evita o equivalente a **{media_anual_vermi/1.5:.0f} carros** fora de circulação/ano
                - Compostagem evita o equivalente a **{media_anual_compost/1.5:.0f} carros** fora de circulação/ano
                (Considerando 1,5 tCO₂eq/ano por carro médio)
                """)
            
            # 9. COMPARAÇÃO COM SCRIPT 2 (Tabela 18)
            st.subheader("🔗 Comparação com Metodologia da Tese (Tabela 18)")
            
            # Calcular usando os mesmos parâmetros do Script 2 para comparação
            # Para 100 kg/dia × 20 anos, o Script 2 mostra 1.405,87 tCO₂eq para vermicompostagem
            
            # Fator de escala para 100 kg/dia
            if residuos_kg_dia == 100 and anos_simulacao == 20:
                st.success(f"""
                **✅ Resultado Comparável ao Script 2 (Tabela 18):**
                
                Sua simulação ({residuos_kg_dia} kg/dia × {anos_simulacao} anos) é diretamente comparável 
                aos resultados do Script 2 que mostram **1.405,87 tCO₂eq** para vermicompostagem.
                
                **Seu resultado:** {formatar_br(total_evitado_vermi)} tCO₂eq
                **Diferença:** {formatar_br(total_evitado_vermi - 1405.87)} tCO₂eq ({((total_evitado_vermi - 1405.87)/1405.87*100):+.1f}%)
                
                *Nota: Pequenas diferenças são esperadas devido a variações nos parâmetros ambientais.*
                """)
            else:
                st.info(f"""
                **📊 Para comparação com o Script 2 (Tabela 18):**
                
                O Script 2 mostra **1.405,87 tCO₂eq** para 100 kg/dia × 20 anos com vermicompostagem.
                
                **Sua simulação atual:** {formatar_br(total_evitado_vermi)} tCO₂eq
                **Escala:** {residuos_kg_dia} kg/dia × {anos_simulacao} anos
                
                *Para comparar diretamente, configure: 100 kg/dia × 20 anos*
                """)
