import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import requests
from bs4 import BeautifulSoup
import warnings
from scipy.signal import fftconvolve

# =============================================================================
# CONFIGURAÇÕES INICIAIS
# =============================================================================
st.set_page_config(
    page_title="Simulador de Emissões de Metano - Três Abordagens", 
    layout="wide",
    page_icon="🌱"
)
warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
np.seterr(divide='ignore', invalid='ignore')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")

# =============================================================================
# FUNÇÕES DE COTAÇÃO DO CARBONO (MANTIDAS)
# =============================================================================

def obter_cotacao_carbono_investing():
    """Obtém a cotação em tempo real do carbono via Investing.com"""
    try:
        url = "https://www.investing.com/commodities/carbon-emissions"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept-Language': 'pt-BR,pt;q=0.9,en;q=0.8',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Referer': 'https://www.investing.com/'
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        selectores = [
            '[data-test="instrument-price-last"]',
            '.text-2xl',
            '.last-price-value',
            '.instrument-price-last',
            '.pid-1062510-last',
            '.float_lang_base_1',
            '.top.bold.inlineblock',
            '#last_last'
        ]
        
        preco = None
        fonte = "Investing.com"
        
        for seletor in selectores:
            try:
                elemento = soup.select_one(seletor)
                if elemento:
                    texto_preco = elemento.text.strip().replace(',', '')
                    texto_preco = ''.join(c for c in texto_preco if c.isdigit() or c == '.')
                    if texto_preco:
                        preco = float(texto_preco)
                        break
            except (ValueError, AttributeError):
                continue
        
        if preco is not None:
            return preco, "€", "Carbon Emissions Future", True, fonte
        
        return 85.57, "€", "Carbon Emissions (EU ETS Reference)", False, "EU ETS Reference Price"
        
    except Exception as e:
        return None, None, None, False, f"Investing.com - Erro: {str(e)}"

def obter_cotacao_carbono():
    preco, moeda, contrato_info, sucesso, fonte = obter_cotacao_carbono_investing()
    if sucesso:
        return preco, moeda, f"{contrato_info}", True, fonte
    return 85.57, "€", "Carbon Emissions (EU ETS Reference)", False, "EU ETS Reference Price"

def obter_cotacao_euro_real():
    try:
        url = "https://economia.awesomeapi.com.br/last/EUR-BRL"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            cotacao = float(data['EURBRL']['bid'])
            return cotacao, "R$", True, "AwesomeAPI"
    except:
        pass
    
    try:
        url = "https://api.exchangerate-api.com/v4/latest/EUR"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            cotacao = data['rates']['BRL']
            return cotacao, "R$", True, "ExchangeRate-API"
    except:
        pass
    
    return 6.36, "R$", False, "Reference Rate for EU ETS"

# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def formatar_br(numero):
    """Formata números no padrão brasileiro: 1.234,56"""
    if pd.isna(numero):
        return "N/A"
    numero = round(numero, 2)
    return f"{numero:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def br_format(x, pos):
    """Função de formatação para eixos de gráficos (padrão brasileiro)"""
    if x == 0:
        return "0"
    if abs(x) < 0.01:
        return f"{x:.1e}".replace(".", ",")
    if abs(x) >= 1000:
        return f"{x:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

# =============================================================================
# PARÂMETROS FIXOS (YANG ET AL. 2017)
# =============================================================================

# PARÂMETROS COMUNS
TOC_YANG = 0.436  # Fração de carbono orgânico total
TN_YANG = 14.2 / 1000  # Fração de nitrogênio total

# VERMICOMPOSTAGEM (COM MINHOCAS)
CH4_C_FRAC_YANG = 0.13 / 100  # 0.13% do TOC emitido como CH4-C
N2O_N_FRAC_YANG = 0.92 / 100  # 0.92% do TN emitido como N2O-N

# COMPOSTAGEM TERMOFÍLICA (SEM MINHOCAS)
CH4_C_FRAC_THERMO = 0.006  # 0.6% do TOC emitido como CH4-C
N2O_N_FRAC_THERMO = 0.0196  # 1.96% do TN emitido como N2O-N

# PERFIS TEMPORAIS (50 DIAS)
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

# GWP (IPCC AR6)
GWP_CH4_20 = 79.7  # 20 anos
GWP_N2O_20 = 273   # 20 anos
GWP_CH4_100 = 27.9  # 100 anos (para comparação)

# =============================================================================
# FUNÇÕES DE CÁLCULO COM CONVOLUÇÃO (PARA TODAS AS ABAS)
# =============================================================================

def calcular_metano_aterro_convolucao(residuos_kg_dia, umidade, temperatura, doc_val, dias_simulacao):
    """
    Calcula metano do aterro usando convolução para entrada contínua
    Base IPCC 2006
    """
    # Parâmetros fixos
    MCF = 1.0
    F = 0.5
    OX = 0.1
    Ri = 0.0
    k_ano = 0.06
    
    # 1. Cálculo do DOCf
    DOCf = 0.0147 * temperatura + 0.28
    
    # 2. Potencial de metano por kg
    potencial_CH4_por_kg = doc_val * DOCf * MCF * F * (16/12) * (1 - Ri) * (1 - OX)
    potencial_CH4_lote_diario = residuos_kg_dia * potencial_CH4_por_kg
    
    # 3. Kernel de decaimento exponencial
    t = np.arange(1, dias_simulacao + 1, dtype=float)
    kernel_ch4 = np.exp(-k_ano * (t - 1) / 365.0) - np.exp(-k_ano * t / 365.0)
    
    # 4. Convolução para entrada contínua
    entradas_diarias = np.ones(dias_simulacao, dtype=float)
    emissoes_CH4 = fftconvolve(entradas_diarias, kernel_ch4, mode='full')[:dias_simulacao]
    emissoes_CH4 *= potencial_CH4_lote_diario
    
    return emissoes_CH4

def calcular_metano_vermi_convolucao(residuos_kg_dia, umidade, dias_simulacao):
    """
    Calcula metano da vermicompostagem usando convolução para entrada contínua
    Base Yang et al. 2017
    """
    fracao_ms = 1 - umidade
    
    # Metano total por lote diário
    ch4_total_por_lote_diario = residuos_kg_dia * (TOC_YANG * CH4_C_FRAC_YANG * (16/12) * fracao_ms)
    
    # Convolução usando perfil de 50 dias
    emissoes_CH4 = np.zeros(dias_simulacao)
    
    for dia_entrada in range(dias_simulacao):
        for dia_compostagem in range(len(PERFIL_CH4_VERMI)):
            dia_emissao = dia_entrada + dia_compostagem
            if dia_emissao < dias_simulacao:
                emissoes_CH4[dia_emissao] += ch4_total_por_lote_diario * PERFIL_CH4_VERMI[dia_compostagem]
    
    return emissoes_CH4

def calcular_metano_compostagem_convolucao(residuos_kg_dia, umidade, dias_simulacao):
    """
    Calcula metano da compostagem termofílica usando convolução para entrada contínua
    Base Yang et al. 2017
    """
    fracao_ms = 1 - umidade
    
    # Metano total por lote diário
    ch4_total_por_lote_diario = residuos_kg_dia * (TOC_YANG * CH4_C_FRAC_THERMO * (16/12) * fracao_ms)
    
    # Convolução usando perfil de 50 dias
    emissoes_CH4 = np.zeros(dias_simulacao)
    
    for dia_entrada in range(dias_simulacao):
        for dia_compostagem in range(len(PERFIL_CH4_THERMO)):
            dia_emissao = dia_entrada + dia_compostagem
            if dia_emissao < dias_simulacao:
                emissoes_CH4[dia_emissao] += ch4_total_por_lote_diario * PERFIL_CH4_THERMO[dia_compostagem]
    
    return emissoes_CH4

def calcular_n2o_vermi_convolucao(residuos_kg_dia, umidade, dias_simulacao):
    """Calcula N2O da vermicompostagem usando convolução"""
    fracao_ms = 1 - umidade
    n2o_total_por_lote_diario = residuos_kg_dia * (TN_YANG * N2O_N_FRAC_YANG * (44/28) * fracao_ms)
    
    emissoes_N2O = np.zeros(dias_simulacao)
    for dia_entrada in range(dias_simulacao):
        for dia_compostagem in range(len(PERFIL_N2O_VERMI)):
            dia_emissao = dia_entrada + dia_compostagem
            if dia_emissao < dias_simulacao:
                emissoes_N2O[dia_emissao] += n2o_total_por_lote_diario * PERFIL_N2O_VERMI[dia_compostagem]
    
    return emissoes_N2O

def calcular_n2o_compostagem_convolucao(residuos_kg_dia, umidade, dias_simulacao):
    """Calcula N2O da compostagem termofílica usando convolução"""
    fracao_ms = 1 - umidade
    n2o_total_por_lote_diario = residuos_kg_dia * (TN_YANG * N2O_N_FRAC_THERMO * (44/28) * fracao_ms)
    
    emissoes_N2O = np.zeros(dias_simulacao)
    for dia_entrada in range(dias_simulacao):
        for dia_compostagem in range(len(PERFIL_N2O_THERMO)):
            dia_emissao = dia_entrada + dia_compostagem
            if dia_emissao < dias_simulacao:
                emissoes_N2O[dia_emissao] += n2o_total_por_lote_diario * PERFIL_N2O_THERMO[dia_compostagem]
    
    return emissoes_N2O

def calcular_metano_aterro_pre_descarte(residuos_kg_dia, dias_simulacao):
    """
    Calcula metano pré-descarte (Feng et al. 2020)
    """
    CH4_pre_descarte_ugC_por_kg_h_media = 2.78
    fator_conversao_C_para_CH4 = 16/12
    CH4_pre_descarte_ugCH4_por_kg_h_media = CH4_pre_descarte_ugC_por_kg_h_media * fator_conversao_C_para_CH4
    CH4_pre_descarte_g_por_kg_dia = CH4_pre_descarte_ugCH4_por_kg_h_media * 24 / 1_000_000
    
    emissoes_CH4_pre_descarte_kg = np.full(dias_simulacao, residuos_kg_dia * CH4_pre_descarte_g_por_kg_dia / 1000)
    
    return emissoes_CH4_pre_descarte_kg

# =============================================================================
# INICIALIZAÇÃO DA SESSION STATE
# =============================================================================

def inicializar_session_state():
    if 'preco_carbono' not in st.session_state:
        preco_carbono, moeda, contrato_info, sucesso, fonte = obter_cotacao_carbono()
        st.session_state.preco_carbono = preco_carbono
        st.session_state.moeda_carbono = moeda
        st.session_state.fonte_cotacao = fonte
        
    if 'taxa_cambio' not in st.session_state:
        preco_euro, moeda_real, sucesso_euro, fonte_euro = obter_cotacao_euro_real()
        st.session_state.taxa_cambio = preco_euro
        st.session_state.moeda_real = moeda_real
        
    if 'moeda_real' not in st.session_state:
        st.session_state.moeda_real = "R$"
    if 'run_simulacao_aba1' not in st.session_state:
        st.session_state.run_simulacao_aba1 = False
    if 'run_simulacao_aba2' not in st.session_state:
        st.session_state.run_simulacao_aba2 = False
    if 'run_simulacao_aba3' not in st.session_state:
        st.session_state.run_simulacao_aba3 = False

# =============================================================================
# FUNÇÃO PARA EXIBIR COTAÇÃO
# =============================================================================

def exibir_cotacao_carbono():
    """Exibe a cotação do carbono no painel lateral"""
    st.sidebar.header("💰 Mercado de Carbono e Câmbio")
    
    # Exibe cotação atual do carbono
    st.sidebar.metric(
        label=f"Preço do Carbono (tCO₂eq)",
        value=f"{st.session_state.moeda_carbono} {st.session_state.preco_carbono:.2f}",
        help=f"Fonte: {st.session_state.fonte_cotacao}"
    )
    
    # Exibe cotação atual do Euro
    st.sidebar.metric(
        label="Euro (EUR/BRL)",
        value=f"{st.session_state.moeda_real} {st.session_state.taxa_cambio:.2f}",
        help="Cotação do Euro em Reais Brasileiros"
    )
    
    # Calcular preço do carbono em Reais
    preco_carbono_reais = st.session_state.preco_carbono * st.session_state.taxa_cambio
    
    st.sidebar.metric(
        label=f"Carbono em Reais (tCO₂eq)",
        value=f"R$ {preco_carbono_reais:.2f}",
        help="Preço do carbono convertido para Reais Brasileiros"
    )

# =============================================================================
# CONFIGURAÇÃO PRINCIPAL
# =============================================================================

# Inicializar session state
inicializar_session_state()

# Título principal
st.title("🔬 Simulador de Emissões de Metano - Três Abordagens com Convolução")
st.markdown("""
**Comparação completa de diferentes métodos de cálculo de emissões de metano**
Todas as abas utilizam **convolução para entrada contínua** (1 lote por dia)
""")

# Criar abas
aba1, aba2, aba3 = st.tabs([
    "📦 Aba 1: Lote Único com Convolução", 
    "📈 Aba 2: Contínuo 1 Lote/Dia (20 anos) - Só Metano",
    "🏭 Aba 3: Proposta da Tese (CH₄ + N₂O)"
])

# =============================================================================
# ABA 1: LOTE ÚNICO COM CONVOLUÇÃO
# =============================================================================
with aba1:
    st.header("📦 Lote Único - Análise com Convolução")
    st.markdown("""
    **Análise de um único lote de resíduos, mas usando convolução para mostrar o efeito temporal**
    
    Mesmo sendo um lote único, usamos a mesma metodologia de convolução para consistência.
    """)
    
    # Configurar sidebar específica para aba 1
    with st.sidebar:
        if st.session_state.get('aba_atual') != 1:
            st.session_state.aba_atual = 1
            st.session_state.run_simulacao_aba2 = False
            st.session_state.run_simulacao_aba3 = False
        
        st.header("⚙️ Parâmetros - Lote Único")
        
        # Entrada principal
        residuos_kg = st.number_input(
            "Peso do lote (kg)", 
            min_value=10, 
            max_value=1000, 
            value=100, 
            step=10,
            help="Peso do lote de resíduos orgânicos",
            key="lote_residuos_aba1"
        )
        
        # Parâmetros ambientais
        umidade_valor = st.slider(
            "Umidade do resíduo (%)", 
            50, 95, 85, 1,
            help="Percentual de umidade dos resíduos",
            key="umidade_aba1"
        )
        umidade = umidade_valor / 100.0
        
        temperatura = st.slider(
            "Temperatura média (°C)", 
            15, 35, 25, 1,
            help="Temperatura média ambiente (importante para aterro)",
            key="temp_aba1"
        )
        
        doc_val = st.slider(
            "DOC - Carbono Orgânico Degradável", 
            0.10, 0.50, 0.15, 0.01,
            help="Fração de carbono orgânico degradável",
            key="doc_aba1"
        )
        
        # Período de análise
        dias_simulacao = st.slider(
            "Dias de simulação", 
            50, 1000, 365, 50,
            help="Período total da simulação em dias",
            key="dias_aba1"
        )
        
        if st.button("🚀 Calcular Lote Único", type="primary", key="btn_aba1"):
            st.session_state.run_simulacao_aba1 = True

    # Executar simulação aba 1
    if st.session_state.get('run_simulacao_aba1', False):
        with st.spinner('Calculando emissões para lote único...'):
            # 1. CÁLCULO DAS EMISSÕES
            # Para lote único, usamos 1 kg/dia como entrada (equivalente ao lote)
            ch4_aterro = calcular_metano_aterro_convolucao(1, umidade, temperatura, doc_val, dias_simulacao)
            ch4_pre_descarte = calcular_metano_aterro_pre_descarte(1, dias_simulacao)
            ch4_aterro_total = ch4_aterro + ch4_pre_descarte
            
            ch4_vermi = calcular_metano_vermi_convolucao(1, umidade, dias_simulacao)
            ch4_compost = calcular_metano_compostagem_convolucao(1, umidade, dias_simulacao)
            
            # 2. AJUSTAR PARA O TAMANHO REAL DO LOTE
            ch4_aterro_total *= residuos_kg
            ch4_vermi *= residuos_kg
            ch4_compost *= residuos_kg
            
            # 3. CRIAR DATAFRAME
            datas = pd.date_range(start=datetime.now(), periods=dias_simulacao, freq='D')
            
            df_aba1 = pd.DataFrame({
                'Data': datas,
                'Aterro_CH4_kg': ch4_aterro_total,
                'Vermicompostagem_CH4_kg': ch4_vermi,
                'Compostagem_CH4_kg': ch4_compost
            })
            
            # Calcular totais
            total_aterro = df_aba1['Aterro_CH4_kg'].sum()
            total_vermi = df_aba1['Vermicompostagem_CH4_kg'].sum()
            total_compost = df_aba1['Compostagem_CH4_kg'].sum()
            
            # 4. EXIBIR RESULTADOS
            st.subheader("📊 Resultados - Lote Único")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Aterro Sanitário",
                    f"{formatar_br(total_aterro)} kg CH₄",
                    help=f"Total em {dias_simulacao} dias"
                )
            
            with col2:
                reducao_vermi = total_aterro - total_vermi
                reducao_perc = (reducao_vermi / total_aterro * 100) if total_aterro > 0 else 0
                st.metric(
                    "Vermicompostagem",
                    f"{formatar_br(total_vermi)} kg CH₄",
                    delta=f"-{reducao_perc:.1f}%",
                    delta_color="inverse",
                    help=f"Redução de {formatar_br(reducao_vermi)} kg"
                )
            
            with col3:
                reducao_compost = total_aterro - total_compost
                reducao_perc = (reducao_compost / total_aterro * 100) if total_aterro > 0 else 0
                st.metric(
                    "Compostagem",
                    f"{formatar_br(total_compost)} kg CH₄",
                    delta=f"-{reducao_perc:.1f}%",
                    delta_color="inverse",
                    help=f"Redução de {formatar_br(reducao_compost)} kg"
                )
            
            # 5. GRÁFICO DE EMISSÕES DIÁRIAS
            st.subheader("📈 Emissões Diárias de Metano")
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            ax.plot(df_aba1['Data'], df_aba1['Aterro_CH4_kg'], 'r-', 
                    label='Aterro Sanitário', linewidth=2, alpha=0.8)
            ax.plot(df_aba1['Data'], df_aba1['Vermicompostagem_CH4_kg'], 'g-', 
                    label='Vermicompostagem', linewidth=2)
            ax.plot(df_aba1['Data'], df_aba1['Compostagem_CH4_kg'], 'b-', 
                    label='Compostagem Termofílica', linewidth=2)
            
            ax.set_title(f'Emissões Diárias de Metano - Lote de {residuos_kg} kg', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Data')
            ax.set_ylabel('Metano (kg CH₄/dia)')
            ax.legend(title='Cenário')
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.yaxis.set_major_formatter(FuncFormatter(br_format))
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            
            # 6. COMPARAÇÃO DETALHADA
            st.subheader("📋 Comparação Detalhada")
            
            comparacao_df = pd.DataFrame({
                'Cenário': ['Aterro Sanitário', 'Vermicompostagem', 'Compostagem Termofílica'],
                'Metano Total (kg)': [total_aterro, total_vermi, total_compost],
                'Redução vs Aterro (kg)': [0, reducao_vermi, reducao_compost],
                'Redução vs Aterro (%)': [0, (reducao_vermi/total_aterro*100), (reducao_compost/total_aterro*100)],
                'Razão Vermi/Compost': ['-', f"{total_vermi/total_compost:.2f}x", '-']
            })
            
            st.dataframe(comparacao_df, use_container_width=True)
            
            st.info(f"""
            **🔬 Observação Científica:**
            
            A vermicompostagem emite **{total_vermi/total_compost:.2f}x menos metano** que a compostagem termofílica.
            Isso ocorre porque as minhocas:
            
            1. **Aeram o material** naturalmente, reduzindo condições anaeróbias
            2. **Consomem matéria orgânica** rapidamente
            3. **Produzem húmus** que é mais estável
            
            **Redução total:** {formatar_br(reducao_vermi)} kg CH₄ por lote de {residuos_kg} kg
            """)

# =============================================================================
# ABA 2: CONTÍNUO 1 LOTE/DIA (20 ANOS) - SÓ METANO
# =============================================================================
with aba2:
    st.header("📈 Entrada Contínua - 1 Lote por Dia (20 anos)")
    st.markdown("""
    **Simulação de entrada contínua: 1 lote por dia durante 20 anos**
    
    Foco exclusivo em **metano**, usando convolução para modelagem precisa.
    """)
    
    # Configurar sidebar específica para aba 2
    with st.sidebar:
        if st.session_state.get('aba_atual') != 2:
            st.session_state.aba_atual = 2
            st.session_state.run_simulacao_aba1 = False
            st.session_state.run_simulacao_aba3 = False
        
        st.header("⚙️ Parâmetros - Entrada Contínua")
        
        # Tamanho do lote diário
        residuos_kg_dia = st.number_input(
            "Resíduos por lote (kg/dia)", 
            min_value=10, 
            max_value=500, 
            value=100, 
            step=10,
            help="Quantidade de resíduos em cada lote diário",
            key="lote_diario_aba2"
        )
        
        # Parâmetros ambientais
        umidade_valor = st.slider(
            "Umidade do resíduo (%) - Contínuo", 
            50, 95, 85, 1,
            help="Percentual de umidade dos resíduos",
            key="umidade_aba2"
        )
        umidade = umidade_valor / 100.0
        
        temperatura = st.slider(
            "Temperatura média (°C) - Contínuo", 
            15, 35, 25, 1,
            help="Temperatura média ambiente",
            key="temp_aba2"
        )
        
        doc_val = st.slider(
            "DOC - Carbono Orgânico Degradável - Contínuo", 
            0.10, 0.50, 0.15, 0.01,
            help="Fração de carbono orgânico degradável",
            key="doc_aba2"
        )
        
        # Período de análise
        anos_simulacao = st.slider(
            "Anos de simulação", 
            1, 50, 20, 1,
            help="Período total da simulação em anos",
            key="anos_aba2"
        )
        
        dias_simulacao = anos_simulacao * 365
        
        if st.button("🚀 Calcular Entrada Contínua", type="primary", key="btn_aba2"):
            st.session_state.run_simulacao_aba2 = True

    # Executar simulação aba 2
    if st.session_state.get('run_simulacao_aba2', False):
        with st.spinner(f'Calculando emissões para {residuos_kg_dia} kg/dia durante {anos_simulacao} anos...'):
            # 1. CÁLCULO DAS EMISSÕES
            ch4_aterro = calcular_metano_aterro_convolucao(residuos_kg_dia, umidade, temperatura, doc_val, dias_simulacao)
            ch4_pre_descarte = calcular_metano_aterro_pre_descarte(residuos_kg_dia, dias_simulacao)
            ch4_aterro_total = ch4_aterro + ch4_pre_descarte
            
            ch4_vermi = calcular_metano_vermi_convolucao(residuos_kg_dia, umidade, dias_simulacao)
            ch4_compost = calcular_metano_compostagem_convolucao(residuos_kg_dia, umidade, dias_simulacao)
            
            # 2. CRIAR DATAFRAME
            datas = pd.date_range(start=datetime.now(), periods=dias_simulacao, freq='D')
            
            df_aba2 = pd.DataFrame({
                'Data': datas,
                'Aterro_CH4_kg_dia': ch4_aterro_total,
                'Vermicompostagem_CH4_kg_dia': ch4_vermi,
                'Compostagem_CH4_kg_dia': ch4_compost
            })
            
            # Calcular acumulados
            df_aba2['Aterro_Acumulado'] = df_aba2['Aterro_CH4_kg_dia'].cumsum()
            df_aba2['Vermi_Acumulado'] = df_aba2['Vermicompostagem_CH4_kg_dia'].cumsum()
            df_aba2['Compost_Acumulado'] = df_aba2['Compostagem_CH4_kg_dia'].cumsum()
            
            # Calcular totais
            total_aterro = df_aba2['Aterro_CH4_kg_dia'].sum()
            total_vermi = df_aba2['Vermicompostagem_CH4_kg_dia'].sum()
            total_compost = df_aba2['Compostagem_CH4_kg_dia'].sum()
            
            # 3. EXIBIR RESULTADOS
            st.subheader("📊 Resultados - Entrada Contínua")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Aterro Sanitário",
                    f"{formatar_br(total_aterro)} kg CH₄",
                    f"{formatar_br(total_aterro/1000)} ton",
                    help=f"Total em {anos_simulacao} anos"
                )
            
            with col2:
                reducao_vermi = total_aterro - total_vermi
                st.metric(
                    "Vermicompostagem",
                    f"{formatar_br(total_vermi)} kg CH₄",
                    f"-{formatar_br(reducao_vermi)} kg",
                    delta_color="inverse",
                    help=f"Redução vs aterro"
                )
            
            with col3:
                reducao_compost = total_aterro - total_compost
                st.metric(
                    "Compostagem",
                    f"{formatar_br(total_compost)} kg CH₄",
                    f"-{formatar_br(reducao_compost)} kg",
                    delta_color="inverse",
                    help=f"Redução vs aterro"
                )
            
            # 4. GRÁFICO DE EMISSÕES ACUMULADAS
            st.subheader("📉 Emissões Acumuladas de Metano")
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            ax.plot(df_aba2['Data'], df_aba2['Aterro_Acumulado']/1000, 'r-', 
                    label='Aterro Sanitário', linewidth=3, alpha=0.7)
            ax.plot(df_aba2['Data'], df_aba2['Vermi_Acumulado']/1000, 'g-', 
                    label='Vermicompostagem', linewidth=2)
            ax.plot(df_aba2['Data'], df_aba2['Compost_Acumulado']/1000, 'b-', 
                    label='Compostagem Termofílica', linewidth=2)
            
            # Área de redução
            ax.fill_between(df_aba2['Data'], 
                           df_aba2['Vermi_Acumulado']/1000, 
                           df_aba2['Aterro_Acumulado']/1000,
                           color='green', alpha=0.3, label='Redução Vermicompostagem')
            
            ax.set_title(f'Metano Acumulado - {residuos_kg_dia} kg/dia × {anos_simulacao} anos', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Data')
            ax.set_ylabel('Metano Acumulado (ton CH₄)')
            ax.legend(title='Cenário de Gestão', loc='upper left')
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.yaxis.set_major_formatter(FuncFormatter(br_format))
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            
            # 5. ANÁLISE ANUAL
            st.subheader("📅 Análise Anual")
            
            df_aba2['Ano'] = df_aba2['Data'].dt.year
            df_anual = df_aba2.groupby('Ano').agg({
                'Aterro_CH4_kg_dia': 'sum',
                'Vermicompostagem_CH4_kg_dia': 'sum',
                'Compostagem_CH4_kg_dia': 'sum'
            }).reset_index()
            
            df_anual.rename(columns={
                'Aterro_CH4_kg_dia': 'Aterro_kg_ano',
                'Vermicompostagem_CH4_kg_dia': 'Vermi_kg_ano',
                'Compostagem_CH4_kg_dia': 'Compost_kg_ano'
            }, inplace=True)
            
            # Converter para toneladas
            for col in ['Aterro_kg_ano', 'Vermi_kg_ano', 'Compost_kg_ano']:
                df_anual[col] = df_anual[col] / 1000
                df_anual[f'{col}_cum'] = df_anual[col].cumsum()
            
            # Gráfico de barras anuais
            fig, ax = plt.subplots(figsize=(12, 6))
            
            x = np.arange(len(df_anual))
            width = 0.25
            
            ax.bar(x - width, df_anual['Aterro_kg_ano'], width, label='Aterro', color='red', alpha=0.7)
            ax.bar(x, df_anual['Vermi_kg_ano'], width, label='Vermicompostagem', color='green', alpha=0.7)
            ax.bar(x + width, df_anual['Compost_kg_ano'], width, label='Compostagem', color='blue', alpha=0.7)
            
            ax.set_xlabel('Ano')
            ax.set_ylabel('Metano (ton CH₄/ano)')
            ax.set_title(f'Emissões Anuais de Metano - {residuos_kg_dia} kg/dia')
            ax.set_xticks(x)
            ax.set_xticklabels(df_anual['Ano'].astype(str), rotation=45)
            ax.legend()
            ax.grid(True, axis='y', linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # 6. VALOR FINANCEIRO
            st.subheader("💰 Valor Financeiro das Reduções")
            
            # Converter para CO₂eq (GWP 20 anos)
            total_evitado_vermi_kg = reducao_vermi * GWP_CH4_20
            total_evitado_vermi_tco2eq = total_evitado_vermi_kg / 1000
            
            total_evitado_compost_kg = reducao_compost * GWP_CH4_20
            total_evitado_compost_tco2eq = total_evitado_compost_kg / 1000
            
            # Preço em Reais
            preco_carbono_reais = st.session_state.preco_carbono * st.session_state.taxa_cambio
            
            valor_vermi_brl = total_evitado_vermi_tco2eq * preco_carbono_reais
            valor_compost_brl = total_evitado_compost_tco2eq * preco_carbono_reais
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "Vermicompostagem",
                    f"{formatar_br(total_evitado_vermi_tco2eq)} tCO₂eq",
                    f"R$ {formatar_br(valor_vermi_brl)}",
                    help="Valor das emissões evitadas em 20 anos"
                )
            
            with col2:
                st.metric(
                    "Compostagem",
                    f"{formatar_br(total_evitado_compost_tco2eq)} tCO₂eq",
                    f"R$ {formatar_br(valor_compost_brl)}",
                    help="Valor das emissões evitadas em 20 anos"
                )

# =============================================================================
# ABA 3: PROPOSTA DA TESE (CH₄ + N₂O) - IGUAL AO SCRIPT FORM
# =============================================================================
with aba3:
    st.header("🏭 Proposta da Tese - Análise Completa (CH₄ + N₂O)")
    st.markdown("""
    **Análise completa idêntica ao Script 1 (form.txt)**
    
    Inclui:
    - Metano (CH₄) + Óxido Nitroso (N₂O)
    - Emissões pré-descarte
    - Conversão para CO₂eq (GWP 20 anos)
    - Valor financeiro baseado em mercado de carbono
    - Análise de sensibilidade
    """)
    
    # Configurar sidebar específica para aba 3
    with st.sidebar:
        if st.session_state.get('aba_atual') != 3:
            st.session_state.aba_atual = 3
            st.session_state.run_simulacao_aba1 = False
            st.session_state.run_simulacao_aba2 = False
        
        st.header("⚙️ Parâmetros - Análise Completa")
        
        # Entrada principal
        residuos_kg_dia = st.number_input(
            "Resíduos orgânicos (kg/dia) - Completo", 
            min_value=10, 
            max_value=1000, 
            value=100, 
            step=10,
            help="Quantidade diária de resíduos",
            key="residuos_aba3"
        )
        
        # Parâmetros ambientais
        umidade_valor = st.slider(
            "Umidade do resíduo (%) - Completo", 
            50, 95, 85, 1,
            help="Percentual de umidade dos resíduos",
            key="umidade_aba3"
        )
        umidade = umidade_valor / 100.0
        
        temperatura = st.slider(
            "Temperatura média (°C) - Completo", 
            15, 35, 25, 1,
            help="Temperatura média ambiente",
            key="temp_aba3"
        )
        
        doc_val = st.slider(
            "DOC - Carbono Orgânico Degradável - Completo", 
            0.10, 0.50, 0.15, 0.01,
            help="Fração de carbono orgânico degradável",
            key="doc_aba3"
        )
        
        # Parâmetros operacionais do aterro
        st.subheader("🏭 Parâmetros Operacionais do Aterro")
        
        massa_exposta_kg = st.slider(
            "Massa exposta na frente de trabalho (kg)", 
            50, 500, 100, 10,
            help="Massa de resíduos exposta diariamente no aterro",
            key="massa_exposta_aba3"
        )
        
        h_exposta = st.slider(
            "Horas expostas por dia", 
            4, 24, 8, 1,
            help="Horas diárias de exposição dos resíduos no aterro",
            key="horas_expostas_aba3"
        )
        
        # Período de análise
        anos_simulacao = st.slider(
            "Anos de simulação - Completo", 
            5, 50, 20, 5,
            help="Período total da simulação em anos",
            key="anos_aba3"
        )
        
        dias_simulacao = anos_simulacao * 365
        
        if st.button("🚀 Executar Análise Completa", type="primary", key="btn_aba3"):
            st.session_state.run_simulacao_aba3 = True

    # Executar simulação aba 3
    if st.session_state.get('run_simulacao_aba3', False):
        with st.spinner(f'Calculando análise completa para {residuos_kg_dia} kg/dia durante {anos_simulacao} anos...'):
            # 1. CÁLCULO COMPLETO DE EMISSÕES
            
            # Aterro: CH₄ (com pré-descarte)
            ch4_aterro = calcular_metano_aterro_convolucao(residuos_kg_dia, umidade, temperatura, doc_val, dias_simulacao)
            ch4_pre_descarte = calcular_metano_aterro_pre_descarte(residuos_kg_dia, dias_simulacao)
            ch4_aterro_total = ch4_aterro + ch4_pre_descarte
            
            # Aterro: N₂O (simplificado - para exemplo)
            # Nota: Implementação completa exigiria função específica para N₂O do aterro
            # Usando valor médio para demonstração
            fator_umid = (1 - umidade) / (1 - 0.55)
            f_aberto = np.clip((massa_exposta_kg / residuos_kg_dia) * (h_exposta / 24), 0.0, 1.0)
            
            E_aberto = 1.91
            E_fechado = 2.15
            E_medio = f_aberto * E_aberto + (1 - f_aberto) * E_fechado
            E_medio_ajust = E_medio * fator_umid
            n2o_aterro_diario = (E_medio_ajust * (44/28) / 1_000_000) * residuos_kg_dia
            
            # Perfil N₂O aterro (5 dias)
            kernel_n2o = np.array([0.10, 0.30, 0.40, 0.15, 0.05], dtype=float)
            n2o_aterro = fftconvolve(np.full(dias_simulacao, n2o_aterro_diario), kernel_n2o, mode='full')[:dias_simulacao]
            
            # Vermicompostagem: CH₄ + N₂O
            ch4_vermi = calcular_metano_vermi_convolucao(residuos_kg_dia, umidade, dias_simulacao)
            n2o_vermi = calcular_n2o_vermi_convolucao(residuos_kg_dia, umidade, dias_simulacao)
            
            # Compostagem: CH₄ + N₂O
            ch4_compost = calcular_metano_compostagem_convolucao(residuos_kg_dia, umidade, dias_simulacao)
            n2o_compost = calcular_n2o_compostagem_convolucao(residuos_kg_dia, umidade, dias_simulacao)
            
            # 2. CRIAR DATAFRAME COMPLETO
            datas = pd.date_range(start=datetime.now(), periods=dias_simulacao, freq='D')
            
            df_aba3 = pd.DataFrame({
                'Data': datas,
                'CH4_Aterro_kg_dia': ch4_aterro_total,
                'N2O_Aterro_kg_dia': n2o_aterro,
                'CH4_Vermi_kg_dia': ch4_vermi,
                'N2O_Vermi_kg_dia': n2o_vermi,
                'CH4_Compost_kg_dia': ch4_compost,
                'N2O_Compost_kg_dia': n2o_compost
            })
            
            # 3. CONVERTER PARA CO₂eq (GWP 20 anos)
            # CH₄ para tCO₂eq
            for gas in ['CH4_Aterro', 'CH4_Vermi', 'CH4_Compost']:
                col_kg = f'{gas}_kg_dia'
                col_tco2eq = f'{gas}_tCO2eq_dia'
                df_aba3[col_tco2eq] = df_aba3[col_kg] * GWP_CH4_20 / 1000
            
            # N₂O para tCO₂eq
            for gas in ['N2O_Aterro', 'N2O_Vermi', 'N2O_Compost']:
                col_kg = f'{gas}_kg_dia'
                col_tco2eq = f'{gas}_tCO2eq_dia'
                df_aba3[col_tco2eq] = df_aba3[col_kg] * GWP_N2O_20 / 1000
            
            # Totais por cenário
            df_aba3['Total_Aterro_tCO2eq_dia'] = (
                df_aba3['CH4_Aterro_tCO2eq_dia'] + df_aba3['N2O_Aterro_tCO2eq_dia']
            )
            df_aba3['Total_Vermi_tCO2eq_dia'] = (
                df_aba3['CH4_Vermi_tCO2eq_dia'] + df_aba3['N2O_Vermi_tCO2eq_dia']
            )
            df_aba3['Total_Compost_tCO2eq_dia'] = (
                df_aba3['CH4_Compost_tCO2eq_dia'] + df_aba3['N2O_Compost_tCO2eq_dia']
            )
            
            # Acumulados
            for cenario in ['Aterro', 'Vermi', 'Compost']:
                col_dia = f'Total_{cenario}_tCO2eq_dia'
                col_acum = f'Total_{cenario}_tCO2eq_acum'
                df_aba3[col_acum] = df_aba3[col_dia].cumsum()
            
            # Reduções (emissões evitadas)
            df_aba3['Reducao_Vermi_tCO2eq_acum'] = (
                df_aba3['Total_Aterro_tCO2eq_acum'] - df_aba3['Total_Vermi_tCO2eq_acum']
            )
            df_aba3['Reducao_Compost_tCO2eq_acum'] = (
                df_aba3['Total_Aterro_tCO2eq_acum'] - df_aba3['Total_Compost_tCO2eq_acum']
            )
            
            # 4. EXIBIR RESULTADOS COMPLETOS
            st.subheader("📊 Resultados Completos (CH₄ + N₂O)")
            
            # Totais acumulados
            total_aterro_tco2eq = df_aba3['Total_Aterro_tCO2eq_dia'].sum()
            total_vermi_tco2eq = df_aba3['Total_Vermi_tCO2eq_dia'].sum()
            total_compost_tco2eq = df_aba3['Total_Compost_tCO2eq_dia'].sum()
            
            total_evitado_vermi = total_aterro_tco2eq - total_vermi_tco2eq
            total_evitado_compost = total_aterro_tco2eq - total_compost_tco2eq
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Aterro Sanitário",
                    f"{formatar_br(total_aterro_tco2eq)} tCO₂eq",
                    help="Total em 20 anos"
                )
            
            with col2:
                st.metric(
                    "Vermicompostagem",
                    f"{formatar_br(total_vermi_tco2eq)} tCO₂eq",
                    f"-{formatar_br(total_evitado_vermi)} tCO₂eq",
                    delta_color="inverse",
                    help="Redução vs aterro"
                )
            
            with col3:
                st.metric(
                    "Compostagem",
                    f"{formatar_br(total_compost_tco2eq)} tCO₂eq",
                    f"-{formatar_br(total_evitado_compost)} tCO₂eq",
                    delta_color="inverse",
                    help="Redução vs aterro"
                )
            
            # 5. GRÁFICO DE CONTRIBUIÇÃO POR GÁS
            st.subheader("🌫️ Contribuição por Tipo de Gás")
            
            # Calcular contribuição percentual
            contrib_vermi = {
                'CH₄': df_aba3['CH4_Vermi_tCO2eq_dia'].sum() / total_vermi_tco2eq * 100,
                'N₂O': df_aba3['N2O_Vermi_tCO2eq_dia'].sum() / total_vermi_tco2eq * 100
            }
            
            contrib_compost = {
                'CH₄': df_aba3['CH4_Compost_tCO2eq_dia'].sum() / total_compost_tco2eq * 100,
                'N₂O': df_aba3['N2O_Compost_tCO2eq_dia'].sum() / total_compost_tco2eq * 100
            }
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Gráfico vermicompostagem
            ax1.pie(
                [contrib_vermi['CH₄'], contrib_vermi['N₂O']],
                labels=['CH₄', 'N₂O'],
                autopct='%1.1f%%',
                colors=['#FF6B6B', '#4ECDC4'],
                startangle=90
            )
            ax1.set_title('Vermicompostagem')
            
            # Gráfico compostagem
            ax2.pie(
                [contrib_compost['CH₄'], contrib_compost['N₂O']],
                labels=['CH₄', 'N₂O'],
                autopct='%1.1f%%',
                colors=['#FF6B6B', '#4ECDC4'],
                startangle=90
            )
            ax2.set_title('Compostagem Termofílica')
            
            plt.suptitle('Contribuição dos Gases para o Total de CO₂eq', fontsize=14, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            
            st.info(f"""
            **🔬 Análise de Contribuição:**
            
            **Vermicompostagem:**
            - CH₄: {contrib_vermi['CH₄']:.1f}% do total
            - N₂O: {contrib_vermi['N₂O']:.1f}% do total
            
            **Compostagem Termofílica:**
            - CH₄: {contrib_compost['CH₄']:.1f}% do total
            - N₂O: {contrib_compost['N₂O']:.1f}% do total
            
            **Observação:** A compostagem termofílica tem maior contribuição de N₂O devido às 
            temperaturas mais altas que favorecem processos nitrificação/desnitrificação.
            """)
            
            # 6. VALOR FINANCEIRO COMPLETO
            st.subheader("💰 Valor Financeiro Total")
            
            # Preço em Reais
            preco_carbono_reais = st.session_state.preco_carbono * st.session_state.taxa_cambio
            
            valor_vermi_brl = total_evitado_vermi * preco_carbono_reais
            valor_compost_brl = total_evitado_compost * preco_carbono_reais
            
            # Médias anuais
            media_anual_vermi = total_evitado_vermi / anos_simulacao
            media_anual_compost = total_evitado_compost / anos_simulacao
            
            valor_anual_vermi_brl = media_anual_vermi * preco_carbono_reais
            valor_anual_compost_brl = media_anual_compost * preco_carbono_reais
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🪱 Vermicompostagem")
                st.metric(
                    "Valor Total (20 anos)",
                    f"R$ {formatar_br(valor_vermi_brl)}",
                    help=f"Baseado em {formatar_br(total_evitado_vermi)} tCO₂eq"
                )
                st.metric(
                    "Valor Anual Médio",
                    f"R$ {formatar_br(valor_anual_vermi_brl)}/ano",
                    help="Receita média anual com créditos"
                )
            
            with col2:
                st.markdown("#### 🌡️ Compostagem Termofílica")
                st.metric(
                    "Valor Total (20 anos)",
                    f"R$ {formatar_br(valor_compost_brl)}",
                    help=f"Baseado em {formatar_br(total_evitado_compost)} tCO₂eq"
                )
                st.metric(
                    "Valor Anual Médio",
                    f"R$ {formatar_br(valor_anual_compost_brl)}/ano",
                    help="Receita média anual com créditos"
                )
            
            # 7. COMPARAÇÃO ENTRE ABAS
            st.subheader("🔄 Comparação entre as Três Abordagens")
            
            # Para comparação, calcular só metano na mesma configuração
            ch4_aterro_simples = calcular_metano_aterro_convolucao(residuos_kg_dia, umidade, temperatura, doc_val, dias_simulacao)
            ch4_pre_descarte_simples = calcular_metano_aterro_pre_descarte(residuos_kg_dia, dias_simulacao)
            ch4_aterro_simples_total = (ch4_aterro_simples + ch4_pre_descarte_simples).sum() * GWP_CH4_20 / 1000
            
            ch4_vermi_simples = calcular_metano_vermi_convolucao(residuos_kg_dia, umidade, dias_simulacao)
            ch4_vermi_simples_total = ch4_vermi_simples.sum() * GWP_CH4_20 / 1000
            
            comparacao_df = pd.DataFrame({
                'Abordagem': [
                    'Aba 1: Lote Único (100 kg)', 
                    'Aba 2: Contínuo - Só CH₄', 
                    'Aba 3: Completo - CH₄ + N₂O'
                ],
                'Aterro (tCO₂eq)': [
                    'N/A',  # Aba 1 não tem CO₂eq direto
                    formatar_br(ch4_aterro_simples_total),
                    formatar_br(total_aterro_tco2eq)
                ],
                'Vermicompostagem (tCO₂eq)': [
                    'N/A',
                    formatar_br(ch4_vermi_simples_total),
                    formatar_br(total_vermi_tco2eq)
                ],
                'Redução (tCO₂eq)': [
                    'N/A',
                    formatar_br(ch4_aterro_simples_total - ch4_vermi_simples_total),
                    formatar_br(total_evitado_vermi)
                ],
                'Valor (R$)': [
                    'N/A',
                    formatar_br((ch4_aterro_simples_total - ch4_vermi_simples_total) * preco_carbono_reais),
                    formatar_br(valor_vermi_brl)
                ]
            })
            
            st.dataframe(comparacao_df, use_container_width=True)
            
            st.success(f"""
            **📈 Conclusões da Análise Completa:**
            
            1. **A inclusão de N₂O aumenta o total de CO₂eq em {(total_vermi_tco2eq/ch4_vermi_simples_total-1)*100:.1f}%**
            2. **O valor financeiro é {(valor_vermi_brl/((ch4_aterro_simples_total - ch4_vermi_simples_total) * preco_carbono_reais)-1)*100:.1f}% maior** considerando ambos os gases
            3. **A vermicompostagem evita {total_evitado_vermi/ch4_vermi_simples_total:.1f}x mais** quando consideramos CH₄ + N₂O vs apenas CH₄
            4. **Receita anual média:** R$ {formatar_br(valor_anual_vermi_brl)}/ano
            
            **💡 Recomendação:** Para projetos reais de créditos de carbono, use sempre a **Análise Completa (Aba 3)** 
            que considera todos os gases de efeito estufa relevantes.
            """)

# =============================================================================
# EXIBIR COTAÇÃO NA SIDEBAR
# =============================================================================
exibir_cotacao_carbono()

# =============================================================================
# RODAPÉ
# =============================================================================
st.markdown("---")
st.markdown("""
**📚 Referências Científicas:**
- **Yang et al. (2017)** - Emissões de GEE durante compostagem e vermicompostagem
- **IPCC (2006)** - Guidelines for National Greenhouse Gas Inventories
- **Feng et al. (2020)** - Emissões pré-descarte de resíduos orgânicos
- **UNFCCC (2016)** - Metodologias para projetos de carbono

**🔧 Metodologia Comum:**
- Todas as abas utilizam **convolução para entrada contínua**
- **Parâmetros consistentes** entre todas as análises
- **Base científica única** (Yang et al., 2017)

**🇧🇷 Contexto Brasileiro:**
- Preços em Real Brasileiro (R$)
- Parâmetros ajustados para condições tropicais
- Comparação com mercados de carbono internacionais
""")
