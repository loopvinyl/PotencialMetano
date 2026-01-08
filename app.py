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

# =============================================================================
# CONFIGURAÇÕES INICIAIS
# =============================================================================
st.set_page_config(
    page_title="Potencial de Metano - Simulação Aterro vs Compostagem", 
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
# PARÂMETROS FIXOS (CONSISTENTES COM SCRIPT PRINCIPAL)
# =============================================================================

# Parâmetros do Aterro (IPCC 2006)
DOC = 0.15  # Carbono orgânico degradável (fração)
MCF = 1.0   # Fator de correção de metano
F = 0.5     # Fração de metano no biogás
OX = 0.1    # Fator de oxidação
Ri = 0.0    # Metano recuperado
k_ano = 0.06  # Constante de decaimento anual

# Parâmetros para Vermicompostagem (Yang et al. 2017) - FIXOS
TOC_YANG = 0.436  # Fração de carbono orgânico total
TN_YANG = 14.2 / 1000  # Fração de nitrogênio total
CH4_C_FRAC_YANG = 0.13 / 100  # 0.13% FIXO
N2O_N_FRAC_YANG = 0.92 / 100  # 0.92% FIXO

# Parâmetros para Compostagem Termofílica (Yang et al. 2017) - FIXOS
CH4_C_FRAC_THERMO = 0.006    # 0.6% FIXO
N2O_N_FRAC_THERMO = 0.0196   # 1.96% FIXO

# GWP (IPCC AR6 20 anos)
GWP_CH4_20 = 79.7
GWP_N2O_20 = 273

# Perfis temporais (50 dias) - IGUAIS AO SCRIPT PRINCIPAL
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

# Emissões pré-descarte (Feng et al. 2020)
CH4_pre_descarte_ugC_por_kg_h_media = 2.78
fator_conversao_C_para_CH4 = 16/12
CH4_pre_descarte_ugCH4_por_kg_h_media = CH4_pre_descarte_ugC_por_kg_h_media * fator_conversao_C_para_CH4
CH4_pre_descarte_g_por_kg_dia = CH4_pre_descarte_ugCH4_por_kg_h_media * 24 / 1_000_000

N2O_pre_descarte_mgN_por_kg = 20.26
N2O_pre_descarte_mgN_por_kg_dia = N2O_pre_descarte_mgN_por_kg / 3
N2O_pre_descarte_g_por_kg_dia = N2O_pre_descarte_mgN_por_kg_dia * (44/28) / 1000

PERFIL_N2O_PRE_DESCARTE = {1: 0.8623, 2: 0.10, 3: 0.0377}

# =============================================================================
# FUNÇÕES DE COTAÇÃO DO CARBONO E CÂMBIO
# =============================================================================

def obter_cotacao_carbono_investing():
    """Obtém a cotação em tempo real do carbono via web scraping do Investing.com"""
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
    """Obtém a cotação em tempo real do carbono"""
    preco, moeda, contrato_info, sucesso, fonte = obter_cotacao_carbono_investing()
    
    if sucesso:
        return preco, moeda, f"{contrato_info}", True, fonte
    
    return 85.57, "€", "Carbon Emissions (EU ETS Reference)", False, "EU ETS Reference Price"

def obter_cotacao_euro_real():
    """Obtém a cotação em tempo real do Euro em relação ao Real Brasileiro"""
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
# FUNÇÕES PARA ANÁLISE POR LOTE (100 kg) - ABA 1
# =============================================================================

def calcular_potencial_metano_aterro(residuos_kg, umidade, temperatura, dias=365):
    """
    Calcula o potencial de geração de metano de um lote de resíduos no aterro
    Baseado na metodologia IPCC 2006 - CONSISTENTE COM SCRIPT PRINCIPAL
    """
    # DOCf calculado pela temperatura (DOCf = 0.0147 × T + 0.28)
    DOCf = 0.0147 * temperatura + 0.28
    
    # Cálculo do potencial de metano por kg de resíduo
    potencial_CH4_por_kg = DOC * DOCf * MCF * F * (16/12) * (1 - Ri) * (1 - OX)
    
    # Potencial total do lote
    potencial_CH4_total = residuos_kg * potencial_CH4_por_kg
    
    # Perfil temporal de decaimento (primeira ordem) - NÃO NORMALIZAR
    t = np.arange(1, dias + 1, dtype=float)
    kernel_ch4 = np.exp(-k_ano * (t - 1) / 365.0) - np.exp(-k_ano * t / 365.0)
    
    # Distribuir o potencial total ao longo do tempo
    emissoes_CH4 = potencial_CH4_total * kernel_ch4
    
    return emissoes_CH4, potencial_CH4_total, DOCf

def calcular_emissoes_vermicompostagem(residuos_kg, umidade, dias=50):
    """
    Calcula emissões de metano na vermicompostagem (Yang et al. 2017)
    """
    fracao_ms = 1 - umidade  # Fração de matéria seca
    
    # Metano total por lote - USANDO PARÂMETROS CORRETOS
    # CH4_C_FRAC_YANG = 0.13% (0.0013) - CORRIGIDO
    ch4_total_por_lote = residuos_kg * (TOC_YANG * (CH4_C_FRAC_YANG) * (16/12) * fracao_ms)
    
    # Distribuir emissões usando perfil normalizado
    perfil_ch4 = PERFIL_CH4_VERMI[:dias]
    perfil_ch4 = perfil_ch4 / perfil_ch4.sum() if perfil_ch4.sum() > 0 else perfil_ch4
    
    emissoes_CH4 = ch4_total_por_lote * perfil_ch4
    
    return emissoes_CH4, ch4_total_por_lote

def calcular_emissoes_compostagem(residuos_kg, umidade, dias=50):
    """
    Calcula emissões de metano na compostagem termofílica (Yang et al. 2017)
    """
    fracao_ms = 1 - umidade  # Fração de matéria seca
    
    # Metano total por lote - USANDO PARÂMETROS CORRETOS
    # CH4_C_FRAC_THERMO = 0.6% (0.006) - CORRIGIDO
    ch4_total_por_lote = residuos_kg * (TOC_YANG * CH4_C_FRAC_THERMO * (16/12) * fracao_ms)
    
    # Distribuir emissões usando perfil normalizado
    perfil_ch4 = PERFIL_CH4_THERMO[:dias]
    perfil_ch4 = perfil_ch4 / perfil_ch4.sum() if perfil_ch4.sum() > 0 else perfil_ch4
    
    emissoes_CH4 = ch4_total_por_lote * perfil_ch4
    
    return emissoes_CH4, ch4_total_por_lote

# =============================================================================
# FUNÇÕES PARA ENTRADA CONTÍNUA (kg/dia) - ABA 2
# =============================================================================

def calcular_emissoes_aterro_completo_continuo(residuos_kg_dia, umidade, temperatura, doc_val, 
                                               massa_exposta_kg, h_exposta, dias_simulacao):
    """
    Calcula CH₄ + N₂O do aterro para entrada contínua
    CONSISTENTE COM SCRIPT PRINCIPAL
    """
    # 1. CÁLCULO DE CH₄ (METANO)
    DOCf = 0.0147 * temperatura + 0.28
    potencial_CH4_por_kg = doc_val * DOCf * MCF * F * (16/12) * (1 - Ri) * (1 - OX)
    potencial_CH4_lote_diario = residuos_kg_dia * potencial_CH4_por_kg
    
    # Perfil temporal de decaimento - NÃO NORMALIZAR
    t = np.arange(1, dias_simulacao + 1, dtype=float)
    kernel_ch4 = np.exp(-k_ano * (t - 1) / 365.0) - np.exp(-k_ano * t / 365.0)
    
    # Convolução para entrada contínua
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
    emissoes_CH4_pre_descarte_kg = np.full(dias_simulacao, residuos_kg_dia * CH4_pre_descarte_g_por_kg_dia / 1000)
    emissoes_N2O_pre_descarte_kg = np.zeros(dias_simulacao)
    
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
    USANDO PARÂMETROS CORRETOS E CONSISTENTES
    """
    fracao_ms = 1 - umidade
    
    # Metano total por lote diário - CORRIGIDO: 0.13% (0.0013)
    ch4_total_por_lote_diario = residuos_kg_dia * (TOC_YANG * CH4_C_FRAC_YANG * (16/12) * fracao_ms)
    
    # Óxido nitroso total por lote diário - CORRIGIDO: 0.92% (0.0092)
    n2o_total_por_lote_diario = residuos_kg_dia * (TN_YANG * N2O_N_FRAC_YANG * (44/28) * fracao_ms)
    
    # Inicializar arrays de emissões
    emissoes_CH4 = np.zeros(dias_simulacao)
    emissoes_N2O = np.zeros(dias_simulacao)
    
    # Convolução para entrada contínua (50 dias de processo)
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
    USANDO PARÂMETROS CORRETOS E CONSISTENTES
    """
    fracao_ms = 1 - umidade
    
    # Totais por lote diário - CORRIGIDOS:
    # CH4: 0.6% (0.006), N2O: 1.96% (0.0196)
    ch4_total_por_lote_diario = residuos_kg_dia * (TOC_YANG * CH4_C_FRAC_THERMO * (16/12) * fracao_ms)
    n2o_total_por_lote_diario = residuos_kg_dia * (TN_YANG * N2O_N_FRAC_THERMO * (44/28) * fracao_ms)
    
    # Inicializar arrays
    emissoes_CH4 = np.zeros(dias_simulacao)
    emissoes_N2O = np.zeros(dias_simulacao)
    
    # Convolução (50 dias de processo)
    for dia_entrada in range(dias_simulacao):
        for dia_compostagem in range(len(PERFIL_CH4_THERMO)):
            dia_emissao = dia_entrada + dia_compostagem
            if dia_emissao < dias_simulacao:
                emissoes_CH4[dia_emissao] += ch4_total_por_lote_diario * PERFIL_CH4_THERMO[dia_compostagem]
                emissoes_N2O[dia_emissao] += n2o_total_por_lote_diario * PERFIL_N2O_THERMO[dia_compostagem]
    
    return emissoes_CH4, emissoes_N2O

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
# INICIALIZAÇÃO DA SESSION STATE
# =============================================================================

def inicializar_session_state():
    """Inicializa todas as variáveis de session state necessárias"""
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
    if 'run_simulation' not in st.session_state:
        st.session_state.run_simulation = False
    if 'run_simulacao_continuo' not in st.session_state:
        st.session_state.run_simulacao_continuo = False

# =============================================================================
# EXIBIR COTAÇÃO DO CARBONO NO PAINEL LATERAL
# =============================================================================

def exibir_cotacao_carbono():
    """Exibe a cotação do carbono com informações no painel lateral"""
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
# CONFIGURAÇÃO PRINCIPAL DO APLICATIVO
# =============================================================================

# Inicializar session state
inicializar_session_state()

# Título principal
st.title("🔬 Estimação do Potencial de Emissões - Comparação Completa")

# Criar abas
tab1, tab2 = st.tabs(["📦 Análise por Lote (100 kg)", "📈 Entrada Contínua (kg/dia)"])

# =============================================================================
# ABA 1: ANÁLISE POR LOTE (100 kg)
# =============================================================================
with tab1:
    st.header("Análise por Lote Único de 100 kg")
    st.markdown("""
    **Análise Comparativa: Aterro vs Vermicompostagem vs Compostagem**

    Este simulador calcula o potencial de geração de metano de um lote de 100 kg de resíduos orgânicos
    em três diferentes cenários de gestão, com análise financeira baseada no mercado de carbono.
    """)
    
    # Exibir cotação do carbono
    exibir_cotacao_carbono()
    
    # Parâmetros de entrada na sidebar (apenas para aba 1)
    with st.sidebar:
        if st.session_state.get('aba_atual') != 1:
            st.session_state.aba_atual = 1
            
        st.header("⚙️ Parâmetros de Entrada - Lote Único")
        
        # Entrada principal de resíduos (fixo em 100 kg para o lote)
        st.subheader("📦 Lote de Resíduos")
        residuos_kg = st.number_input(
            "Peso do lote (kg)", 
            min_value=10, 
            max_value=1000, 
            value=100, 
            step=10,
            help="Peso do lote de resíduos orgânicos para análise",
            key="lote_residuos"
        )
        
        st.subheader("📊 Parâmetros Ambientais")
        
        umidade_valor = st.slider(
            "Umidade do resíduo (%)", 
            50, 95, 85, 1,
            help="Percentual de umidade dos resíduos orgânicos",
            key="umidade_lote"
        )
        umidade = umidade_valor / 100.0
        
        temperatura = st.slider(
            "Temperatura média (°C)", 
            15, 35, 25, 1,
            help="Temperatura média ambiente (importante para cálculo do DOCf)",
            key="temp_lote"
        )
        
        st.subheader("⏰ Período de Análise")
        dias_simulacao = st.slider(
            "Dias de simulação", 
            50, 1000, 365, 50,
            help="Período total da simulação em dias",
            key="dias_lote"
        )
        
        if st.button("🚀 Calcular Potencial de Metano", type="primary", key="btn_lote"):
            st.session_state.run_simulation = True

    # Execução da simulação para aba 1
    if st.session_state.get('run_simulation', False):
        with st.spinner('Calculando potencial de metano para os três cenários...'):
            
            # 1. CÁLCULO DO POTENCIAL DE METANO PARA CADA CENÁRIO
            # Aterro Sanitário
            emissoes_aterro, total_aterro, DOCf = calcular_potencial_metano_aterro(
                residuos_kg, umidade, temperatura, dias_simulacao
            )
            
            # Vermicompostagem (50 dias de processo)
            dias_vermi = min(50, dias_simulacao)
            emissoes_vermi_temp, total_vermi = calcular_emissoes_vermicompostagem(
                residuos_kg, umidade, dias_vermi
            )
            emissoes_vermi = np.zeros(dias_simulacao)
            emissoes_vermi[:dias_vermi] = emissoes_vermi_temp
            
            # Compostagem Termofílica (50 dias de processo)
            dias_compost = min(50, dias_simulacao)
            emissoes_compost_temp, total_compost = calcular_emissoes_compostagem(
                residuos_kg, umidade, dias_compost
            )
            emissoes_compost = np.zeros(dias_simulacao)
            emissoes_compost[:dias_compost] = emissoes_compost_temp
            
            # 2. CRIAR DATAFRAME COM OS RESULTADOS
            datas = pd.date_range(start=datetime.now(), periods=dias_simulacao, freq='D')
            
            df = pd.DataFrame({
                'Data': datas,
                'Aterro_CH4_kg': emissoes_aterro,
                'Vermicompostagem_CH4_kg': emissoes_vermi,
                'Compostagem_CH4_kg': emissoes_compost
            })
            
            # Calcular valores acumulados
            df['Aterro_Acumulado'] = df['Aterro_CH4_kg'].cumsum()
            df['Vermi_Acumulado'] = df['Vermicompostagem_CH4_kg'].cumsum()
            df['Compost_Acumulado'] = df['Compostagem_CH4_kg'].cumsum()
            
            # Calcular reduções (evitadas) em relação ao aterro
            df['Reducao_Vermi'] = df['Aterro_Acumulado'] - df['Vermi_Acumulado']
            df['Reducao_Compost'] = df['Aterro_Acumulado'] - df['Compost_Acumulado']
            
            # 3. EXIBIR RESULTADOS PRINCIPAIS
            st.header("📊 Resultados - Potencial de Metano por Cenário")
            
            # Métricas principais
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Aterro Sanitário",
                    f"{formatar_br(total_aterro)} kg CH₄",
                    help=f"Total em {dias_simulacao} dias"
                )
            
            with col2:
                reducao_vermi_kg = total_aterro - total_vermi
                reducao_vermi_perc = (1 - total_vermi/total_aterro)*100 if total_aterro > 0 else 0
                st.metric(
                    "Vermicompostagem",
                    f"{formatar_br(total_vermi)} kg CH₄",
                    delta=f"-{formatar_br(reducao_vermi_perc)}%",
                    delta_color="inverse",
                    help=f"Redução de {formatar_br(reducao_vermi_kg)} kg vs aterro"
                )
            
            with col3:
                reducao_compost_kg = total_aterro - total_compost
                reducao_compost_perc = (1 - total_compost/total_aterro)*100 if total_aterro > 0 else 0
                st.metric(
                    "Compostagem Termofílica",
                    f"{formatar_br(total_compost)} kg CH₄",
                    delta=f"-{formatar_br(reducao_compost_perc)}%",
                    delta_color="inverse",
                    help=f"Redução de {formatar_br(reducao_compost_kg)} kg vs aterro"
                )
            
            # 4. GRÁFICO: EMISSÕES DIÁRIAS COMPARATIVAS (100 primeiros dias)
            st.subheader("📈 Emissões Diárias de Metano (Primeiros 100 dias)")
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            dias_exibir = min(100, dias_simulacao)
            
            ax.plot(df['Data'][:dias_exibir], df['Aterro_CH4_kg'][:dias_exibir], 
                   'r-', label='Aterro Sanitário', linewidth=2, alpha=0.8)
            ax.plot(df['Data'][:dias_exibir], df['Vermicompostagem_CH4_kg'][:dias_exibir], 
                   'g-', label='Vermicompostagem', linewidth=2)
            ax.plot(df['Data'][:dias_exibir], df['Compostagem_CH4_kg'][:dias_exibir], 
                   'b-', label='Compostagem Termofílica', linewidth=2)
            
            ax.set_xlabel('Data')
            ax.set_ylabel('Metano (kg CH₄/dia)')
            ax.set_title(f'Emissões Diárias de Metano - Lote de {residuos_kg} kg', 
                        fontsize=14, fontweight='bold')
            ax.legend(title='Cenário de Gestão')
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.yaxis.set_major_formatter(FuncFormatter(br_format))
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            
            # 5. GRÁFICO: REDUÇÃO DE EMISSÕES ACUMULADA
            st.subheader("📉 Redução de Emissões Acumulada (CH₄)")
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            ax.plot(df['Data'], df['Aterro_Acumulado'], 'r-', 
                    label='Aterro Sanitário', linewidth=3, alpha=0.7)
            ax.plot(df['Data'], df['Vermi_Acumulado'], 'g-', 
                    label='Vermicompostagem', linewidth=2)
            ax.plot(df['Data'], df['Compost_Acumulado'], 'b-', 
                    label='Compostagem Termofílica', linewidth=2)
            
            # Área de redução (evitadas)
            ax.fill_between(df['Data'], df['Vermi_Acumulado'], df['Aterro_Acumulado'],
                            color='green', alpha=0.3, label='Redução Vermicompostagem')
            ax.fill_between(df['Data'], df['Compost_Acumulado'], df['Aterro_Acumulado'],
                            color='blue', alpha=0.2, label='Redução Compostagem')
            
            ax.set_title(f'Acumulado de Metano em {dias_simulacao} Dias - Lote de {residuos_kg} kg', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Data')
            ax.set_ylabel('Metano Acumulado (kg CH₄)')
            ax.legend(title='Cenário de Gestão', loc='upper left')
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.yaxis.set_major_formatter(FuncFormatter(br_format))
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            
            # 6. CÁLCULO DE CO₂eq E VALOR FINANCEIRO
            st.header("💰 Valor Financeiro das Emissões Evitadas")
            
            # Converter metano para CO₂eq (GWP CH₄ 20 anos = 79.7)
            total_evitado_vermi_kg = (total_aterro - total_vermi) * GWP_CH4_20
            total_evitado_vermi_tco2eq = total_evitado_vermi_kg / 1000
            
            total_evitado_compost_kg = (total_aterro - total_compost) * GWP_CH4_20
            total_evitado_compost_tco2eq = total_evitado_compost_kg / 1000
            
            # Calcular valor em Reais
            preco_carbono_reais = st.session_state.preco_carbono * st.session_state.taxa_cambio
            
            valor_vermi_brl = total_evitado_vermi_tco2eq * preco_carbono_reais
            valor_compost_brl = total_evitado_compost_tco2eq * preco_carbono_reais
            
            # Exibir métricas
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "Vermicompostagem",
                    f"{formatar_br(total_evitado_vermi_tco2eq)} tCO₂eq",
                    f"R$ {formatar_br(valor_vermi_brl)}",
                    delta_color="off"
                )
            
            with col2:
                st.metric(
                    "Compostagem",
                    f"{formatar_br(total_evitado_compost_tco2eq)} tCO₂eq",
                    f"R$ {formatar_br(valor_compost_brl)}",
                    delta_color="off"
                )

# =============================================================================
# ABA 2: ENTRADA CONTÍNUA (kg/dia)
# =============================================================================
with tab2:
    st.header("Análise para Entrada Contínua (kg/dia)")
    st.markdown("""
    **Análise Comparativa Completa: CH₄ + N₂O com GWP de 20 anos**
    
    Este simulador calcula as emissões totais (metano + óxido nitroso) para operação contínua,
    utilizando a mesma metodologia do Script 2 para comparação direta.
    """)
    
    # Configurar sidebar para aba 2
    with st.sidebar:
        # Limpar flags da aba 1
        if st.session_state.get('aba_atual') != 2:
            st.session_state.aba_atual = 2
            st.session_state.run_simulation = False
        
        st.header("⚙️ Parâmetros Entrada Contínua")
        
        # Entrada principal em kg/dia
        residuos_kg_dia = st.number_input(
            "Resíduos orgânicos (kg/dia)", 
            min_value=10, 
            max_value=5000, 
            value=100, 
            step=10,
            help="Quantidade diária de resíduos para processamento contínuo",
            key="continuo_residuos"
        )
        
        st.subheader("📊 Parâmetros Ambientais")
        
        umidade_valor_cont = st.slider(
            "Umidade do resíduo (%) - Contínuo", 
            50, 95, 85, 1,
            help="Percentual de umidade dos resíduos orgânicos",
            key="umidade_cont"
        )
        umidade_cont = umidade_valor_cont / 100.0
        
        temperatura_cont = st.slider(
            "Temperatura média (°C) - Contínuo", 
            15, 35, 25, 1,
            help="Temperatura média ambiente",
            key="temp_cont"
        )
        
        # DOC (Carbono Orgânico Degradável)
        doc_val = st.slider(
            "DOC - Carbono Orgânico Degradável (fração)", 
            0.10, 0.50, 0.15, 0.01,
            help="Fração de carbono orgânico degradável nos resíduos",
            key="doc_cont"
        )
        
        st.subheader("🏭 Parâmetros Operacionais do Aterro")
        
        massa_exposta_kg = st.slider(
            "Massa exposta na frente de trabalho (kg)", 
            50, 500, 100, 10,
            help="Massa de resíduos exposta diariamente no aterro",
            key="massa_exposta"
        )
        
        h_exposta = st.slider(
            "Horas expostas por dia", 
            4, 24, 8, 1,
            help="Horas diárias de exposição dos resíduos no aterro",
            key="horas_expostas"
        )
        
        st.subheader("⏰ Período de Análise")
        anos_simulacao_cont = st.slider(
            "Anos de simulação - Contínuo", 
            1, 50, 20, 1,
            help="Período total da simulação em anos",
            key="anos_cont"
        )
        
        dias_simulacao_cont = anos_simulacao_cont * 365
        
        if st.button("🚀 Calcular Emissões Contínuas", type="primary", key="btn_continuo"):
            st.session_state.run_simulacao_continuo = True

    # Execução da simulação para aba 2
    if st.session_state.get('run_simulacao_continuo', False):
        with st.spinner(f'Calculando emissões para {residuos_kg_dia} kg/dia durante {anos_simulacao_cont} anos...'):
            
            # 1. CÁLCULO DAS EMISSÕES COMPLETAS
            # Aterro
            ch4_aterro, n2o_aterro, DOCf = calcular_emissoes_aterro_completo_continuo(
                residuos_kg_dia, umidade_cont, temperatura_cont, doc_val,
                massa_exposta_kg, h_exposta, dias_simulacao_cont
            )
            
            # Vermicompostagem
            ch4_vermi, n2o_vermi = calcular_emissoes_vermi_completo_continuo(
                residuos_kg_dia, umidade_cont, dias_simulacao_cont
            )
            
            # Compostagem
            ch4_compost, n2o_compost = calcular_emissoes_compostagem_completo_continuo(
                residuos_kg_dia, umidade_cont, dias_simulacao_cont
            )
            
            # 2. CRIAR DATAFRAME COM RESULTADOS
            datas = pd.date_range(start=datetime.now(), periods=dias_simulacao_cont, freq='D')
            
            df_continuo = pd.DataFrame({
                'Data': datas,
                'CH4_Aterro_kg_dia': ch4_aterro,
                'N2O_Aterro_kg_dia': n2o_aterro,
                'CH4_Vermi_kg_dia': ch4_vermi,
                'N2O_Vermi_kg_dia': n2o_vermi,
                'CH4_Compost_kg_dia': ch4_compost,
                'N2O_Compost_kg_dia': n2o_compost
            })
            
            # 3. CONVERTER PARA CO₂eq (GWP 20 anos)
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
            
            # 5. EXIBIR RESULTADOS
            st.header("📊 Resultados - Entrada Contínua")
            
            # Totais acumulados
            total_evitado_vermi = df_continuo['Reducao_Vermi_tCO2eq_acum'].iloc[-1]
            total_evitado_compost = df_continuo['Reducao_Compost_tCO2eq_acum'].iloc[-1]
            
            # Médias anuais
            media_anual_vermi = total_evitado_vermi / anos_simulacao_cont
            media_anual_compost = total_evitado_compost / anos_simulacao_cont
            
            # Exibir métricas
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🪱 Vermicompostagem")
                st.metric(
                    "Total de emissões evitadas",
                    f"{formatar_br(total_evitado_vermi)} tCO₂eq",
                    help=f"Acumulado em {anos_simulacao_cont} anos"
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
                    help=f"Acumulado em {anos_simulacao_cont} anos"
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
            
            # 6. GRÁFICO DE EMISSÕES DIÁRIAS (primeiros 2 anos)
            st.subheader("📈 Emissões Diárias de CO₂eq (Primeiros 2 anos)")
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            dias_exibir = min(730, dias_simulacao_cont)  # 2 anos
            
            ax.plot(df_continuo['Data'][:dias_exibir], df_continuo['Total_Aterro_tCO2eq_dia'][:dias_exibir], 
                   'r-', label='Aterro Sanitário', linewidth=1, alpha=0.7)
            ax.plot(df_continuo['Data'][:dias_exibir], df_continuo['Total_Vermi_tCO2eq_dia'][:dias_exibir], 
                   'g-', label='Vermicompostagem', linewidth=1, alpha=0.7)
            ax.plot(df_continuo['Data'][:dias_exibir], df_continuo['Total_Compost_tCO2eq_dia'][:dias_exibir], 
                   'b-', label='Compostagem Termofílica', linewidth=1, alpha=0.7)
            
            ax.set_title(f'Emissões Diárias de CO₂eq - {residuos_kg_dia} kg/dia (Primeiros 2 anos)', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Data')
            ax.set_ylabel('tCO₂eq/dia')
            ax.legend(title='Cenário de Gestão')
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.yaxis.set_major_formatter(FuncFormatter(br_format))
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            
            # 7. GRÁFICO DE REDUÇÃO ACUMULADA
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
            
            ax.set_title(f'Emissões Acumuladas - {residuos_kg_dia} kg/dia × {anos_simulacao_cont} anos', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Data')
            ax.set_ylabel('tCO₂eq Acumulado')
            ax.legend(title='Cenário de Gestão', loc='upper left')
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.yaxis.set_major_formatter(FuncFormatter(br_format))
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            
            # 8. COMPARAÇÃO COM SCRIPT 2
            st.subheader("🔗 Comparação com Metodologia da Tese (Tabela 18)")
            
            # Para 100 kg/dia × 20 anos, o Script 2 mostra 1.405,87 tCO₂eq para vermicompostagem
            if residuos_kg_dia == 100 and anos_simulacao_cont == 20:
                st.success(f"""
                **✅ Resultado Comparável ao Script 2 (Tabela 18):**
                
                Sua simulação ({residuos_kg_dia} kg/dia × {anos_simulacao_cont} anos) é diretamente comparável 
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
                **Escala:** {residuos_kg_dia} kg/dia × {anos_simulacao_cont} anos
                
                *Para comparar diretamente, configure: 100 kg/dia × 20 anos*
                """)

# =============================================================================
# RODAPÉ
# =============================================================================
st.markdown("---")
st.markdown("""
**📚 Referências:**
- IPCC (2006). Guidelines for National Greenhouse Gas Inventories
- Yang et al. (2017). Greenhouse gas emissions during MSW landfilling in China
- UNFCCC (2016). Clean Development Mechanism - Methodology AMS-III.F
- EU ETS Market Data (2024). European Carbon Futures

**🔧 Desenvolvido para análise comparativa de potenciais de metano em diferentes cenários de gestão de resíduos.**
""")
