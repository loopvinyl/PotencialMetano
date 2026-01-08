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

# Configurações iniciais
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
        
        # Várias estratégias para encontrar o preço
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
        
        # Fallback para valor padrão
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
# FUNÇÕES DE CÁLCULO DE EMISSÕES - FOCADAS NO POTENCIAL DE METANO
# =============================================================================

def calcular_potencial_metano_aterro(residuos_kg, umidade, temperatura, dias=365):
    """
    Calcula o potencial de geração de metano de um lote de resíduos no aterro
    Baseado na metodologia IPCC 2006
    
    Fórmula: CH4 = Resíduos × DOC × DOCf × MCF × F × (16/12) × (1 - OX) × (1 - Ri)
    """
    # Parâmetros fixos (IPCC 2006)
    DOC = 0.15  # Carbono orgânico degradável (fração)
    MCF = 1.0   # Fator de correção de metano (para aterros sanitários)
    F = 0.5     # Fração de metano no biogás
    OX = 0.1    # Fator de oxidação
    Ri = 0.0    # Metano recuperado
    
    # DOCf calculado pela temperatura (DOCf = 0.0147 × T + 0.28)
    DOCf = 0.0147 * temperatura + 0.28
    
    # Cálculo do potencial de metano por kg de resíduo
    potencial_CH4_por_kg = DOC * DOCf * MCF * F * (16/12) * (1 - Ri) * (1 - OX)
    
    # Potencial total do lote
    potencial_CH4_total = residuos_kg * potencial_CH4_por_kg
    
    # Perfil temporal de decaimento (primeira ordem)
    k_ano = 0.06  # Constante de decaimento anual
    
    # Gerar emissões ao longo do tempo
    t = np.arange(1, dias + 1, dtype=float)
    kernel_ch4 = np.exp(-k_ano * (t - 1) / 365.0) - np.exp(-k_ano * t / 365.0)
    
    # Normalizar o kernel para que a soma seja 1
    kernel_ch4 = kernel_ch4 / kernel_ch4.sum()
    
    # Distribuir o potencial total ao longo do tempo
    emissoes_CH4 = potencial_CH4_total * kernel_ch4
    
    return emissoes_CH4, potencial_CH4_total, DOCf

def calcular_emissoes_vermicompostagem(residuos_kg, umidade, dias=50):
    """
    Calcula emissões de metano na vermicompostagem (Yang et al. 2017)
    """
    # Parâmetros fixos para vermicompostagem
    TOC = 0.436  # Fração de carbono orgânico total
    CH4_C_FRAC = 0.13 / 100  # Fração do TOC emitida como CH4-C (0.13%)
    fracao_ms = 1 - umidade  # Fração de matéria seca
    
    # Metano total por lote
    ch4_total_por_lote = residuos_kg * (TOC * CH4_C_FRAC * (16/12) * fracao_ms)
    
    # Perfil temporal baseado em Yang et al. (2017)
    perfil_ch4 = np.array([
        0.02, 0.02, 0.02, 0.03, 0.03,  # Dias 1-5
        0.04, 0.04, 0.05, 0.05, 0.06,  # Dias 6-10
        0.07, 0.08, 0.09, 0.10, 0.09,  # Dias 11-15
        0.08, 0.07, 0.06, 0.05, 0.04,  # Dias 16-20
        0.03, 0.02, 0.02, 0.01, 0.01,  # Dias 21-25
        0.01, 0.01, 0.01, 0.01, 0.01,  # Dias 26-30
        0.005, 0.005, 0.005, 0.005, 0.005,  # Dias 31-35
        0.005, 0.005, 0.005, 0.005, 0.005,  # Dias 36-40
        0.002, 0.002, 0.002, 0.002, 0.002,  # Dias 41-45
        0.001, 0.001, 0.001, 0.001, 0.001   # Dias 46-50
    ])
    
    # Normalizar perfil
    perfil_ch4 = perfil_ch4 / perfil_ch4.sum()
    
    # Distribuir emissões
    emissoes_CH4 = ch4_total_por_lote * perfil_ch4
    
    return emissoes_CH4, ch4_total_por_lote

def calcular_emissoes_compostagem(residuos_kg, umidade, dias=50):
    """
    Calcula emissões de metano na compostagem termofílica (Yang et al. 2017)
    """
    # Parâmetros fixos para compostagem termofílica
    TOC = 0.436  # Fração de carbono orgânico total
    CH4_C_FRAC = 0.006  # Fração do TOC emitida como CH4-C (0.6%)
    fracao_ms = 1 - umidade  # Fração de matéria seca
    
    # Metano total por lote
    ch4_total_por_lote = residuos_kg * (TOC * CH4_C_FRAC * (16/12) * fracao_ms)
    
    # Perfil temporal para compostagem termofílica
    perfil_ch4 = np.array([
        0.01, 0.02, 0.03, 0.05, 0.08,  # Dias 1-5
        0.12, 0.15, 0.18, 0.20, 0.18,  # Dias 6-10 (pico termofílico)
        0.15, 0.12, 0.10, 0.08, 0.06,  # Dias 11-15
        0.05, 0.04, 0.03, 0.02, 0.02,  # Dias 16-20
        0.01, 0.01, 0.01, 0.01, 0.01,  # Dias 21-25
        0.005, 0.005, 0.005, 0.005, 0.005,  # Dias 26-30
        0.002, 0.002, 0.002, 0.002, 0.002,  # Dias 31-35
        0.001, 0.001, 0.001, 0.001, 0.001,  # Dias 36-40
        0.001, 0.001, 0.001, 0.001, 0.001,  # Dias 41-45
        0.001, 0.001, 0.001, 0.001, 0.001   # Dias 46-50
    ])
    
    # Normalizar perfil
    perfil_ch4 = perfil_ch4 / perfil_ch4.sum()
    
    # Distribuir emissões
    emissoes_CH4 = ch4_total_por_lote * perfil_ch4
    
    return emissoes_CH4, ch4_total_por_lote

# =============================================================================
# NOVAS FUNÇÕES PARA SIMULAÇÃO CONTÍNUA (1 LOTE POR DIA POR 20 ANOS)
# =============================================================================

def calcular_emissoes_aterro_continuo(residuos_kg_dia, umidade, temperatura, anos=20):
    """
    Calcula emissões de metano do aterro com entrada contínua de 1 lote por dia
    Baseado no script v2n_noAr - simulação de 20 anos
    """
    dias = anos * 365
    
    # Parâmetros fixos (IPCC 2006)
    DOC = 0.15  # Carbono orgânico degradável (fração)
    MCF = 1.0   # Fator de correção de metano (para aterros sanitários)
    F = 0.5     # Fração de metano no biogás
    OX = 0.1    # Fator de oxidação
    Ri = 0.0    # Metano recuperado
    
    # DOCf calculado pela temperatura (DOCf = 0.0147 × T + 0.28)
    DOCf = 0.0147 * temperatura + 0.28
    
    # Cálculo do potencial de metano por kg de resíduo
    potencial_CH4_por_kg = DOC * DOCf * MCF * F * (16/12) * (1 - Ri) * (1 - OX)
    
    # Potencial diário
    potencial_CH4_diario = residuos_kg_dia * potencial_CH4_por_kg
    
    # Constante de decaimento anual
    k_ano = 0.06
    
    # Kernel de decaimento (primeira ordem)
    t = np.arange(1, dias + 1, dtype=float)
    kernel_ch4 = np.exp(-k_ano * (t - 1) / 365.0) - np.exp(-k_ano * t / 365.0)
    
    # Convolução: entrada diária constante com o kernel
    entradas_diarias = np.ones(dias, dtype=float)
    emissoes_CH4 = fftconvolve(entradas_diarias, kernel_ch4, mode='full')[:dias]
    emissoes_CH4 *= potencial_CH4_diario
    
    # Potencial total acumulado em 20 anos
    potencial_CH4_total = np.sum(emissoes_CH4)
    
    return emissoes_CH4, potencial_CH4_total, DOCf

def calcular_emissoes_vermicompostagem_continuo(residuos_kg_dia, umidade, anos=20):
    """
    Calcula emissões de metano na vermicompostagem com entrada contínua
    """
    dias = anos * 365
    
    # Parâmetros fixos para vermicompostagem
    TOC = 0.436  # Fração de carbono orgânico total
    CH4_C_FRAC = 0.13 / 100  # Fração do TOC emitida como CH4-C (0.13%)
    fracao_ms = 1 - umidade  # Fração de matéria seca
    
    # Metano total por lote diário
    ch4_total_por_lote = residuos_kg_dia * (TOC * CH4_C_FRAC * (16/12) * fracao_ms)
    
    # Perfil temporal (50 dias)
    perfil_ch4 = np.array([
        0.02, 0.02, 0.02, 0.03, 0.03,  # Dias 1-5
        0.04, 0.04, 0.05, 0.05, 0.06,  # Dias 6-10
        0.07, 0.08, 0.09, 0.10, 0.09,  # Dias 11-15
        0.08, 0.07, 0.06, 0.05, 0.04,  # Dias 16-20
        0.03, 0.02, 0.02, 0.01, 0.01,  # Dias 21-25
        0.01, 0.01, 0.01, 0.01, 0.01,  # Dias 26-30
        0.005, 0.005, 0.005, 0.005, 0.005,  # Dias 31-35
        0.005, 0.005, 0.005, 0.005, 0.005,  # Dias 36-40
        0.002, 0.002, 0.002, 0.002, 0.002,  # Dias 41-45
        0.001, 0.001, 0.001, 0.001, 0.001   # Dias 46-50
    ])
    
    # Normalizar perfil
    perfil_ch4 = perfil_ch4 / perfil_ch4.sum()
    
    # Convolução: entrada diária constante com o perfil de 50 dias
    entradas_diarias = np.ones(dias, dtype=float)
    emissoes_CH4 = fftconvolve(entradas_diarias, perfil_ch4, mode='full')[:dias]
    emissoes_CH4 *= ch4_total_por_lote
    
    # Potencial total acumulado em 20 anos
    potencial_CH4_total = np.sum(emissoes_CH4)
    
    return emissoes_CH4, potencial_CH4_total

def calcular_emissoes_compostagem_continuo(residuos_kg_dia, umidade, anos=20):
    """
    Calcula emissões de metano na compostagem termofílica com entrada contínua
    """
    dias = anos * 365
    
    # Parâmetros fixos para compostagem termofílica
    TOC = 0.436  # Fração de carbono orgânico total
    CH4_C_FRAC = 0.006  # Fração do TOC emitida como CH4-C (0.6%)
    fracao_ms = 1 - umidade  # Fração de matéria seca
    
    # Metano total por lote diário
    ch4_total_por_lote = residuos_kg_dia * (TOC * CH4_C_FRAC * (16/12) * fracao_ms)
    
    # Perfil temporal para compostagem termofílica (50 dias)
    perfil_ch4 = np.array([
        0.01, 0.02, 0.03, 0.05, 0.08,  # Dias 1-5
        0.12, 0.15, 0.18, 0.20, 0.18,  # Dias 6-10 (pico termofílico)
        0.15, 0.12, 0.10, 0.08, 0.06,  # Dias 11-15
        0.05, 0.04, 0.03, 0.02, 0.02,  # Dias 16-20
        0.01, 0.01, 0.01, 0.01, 0.01,  # Dias 21-25
        0.005, 0.005, 0.005, 0.005, 0.005,  # Dias 26-30
        0.002, 0.002, 0.002, 0.002, 0.002,  # Dias 31-35
        0.001, 0.001, 0.001, 0.001, 0.001,  # Dias 36-40
        0.001, 0.001, 0.001, 0.001, 0.001,  # Dias 41-45
        0.001, 0.001, 0.001, 0.001, 0.001   # Dias 46-50
    ])
    
    # Normalizar perfil
    perfil_ch4 = perfil_ch4 / perfil_ch4.sum()
    
    # Convolução: entrada diária constante com o perfil de 50 dias
    entradas_diarias = np.ones(dias, dtype=float)
    emissoes_CH4 = fftconvolve(entradas_diarias, perfil_ch4, mode='full')[:dias]
    emissoes_CH4 *= ch4_total_por_lote
    
    # Potencial total acumulado em 20 anos
    potencial_CH4_total = np.sum(emissoes_CH4)
    
    return emissoes_CH4, potencial_CH4_total

# =============================================================================
# FUNÇÃO PARA SIMULAR OS TRÊS CENÁRIOS ECONÔMICOS
# =============================================================================

def simular_cenarios_financeiros(total_evitado_tco2eq, preco_carbono_eur, taxa_cambio):
    """
    Simula três cenários econômicos:
    1. Otimista: Mercado regulado (preço atual do painel)
    2. Base: Mercado voluntário (USD 7.48 ≈ R$ 37.40)
    3. Pessimista: Sem venda de créditos
    """
    # Conversão de EUR para BRL
    preco_carbono_brl = preco_carbono_eur * taxa_cambio
    
    # Preço do mercado voluntário (USD 7.48 convertido para BRL)
    taxa_usd_brl = 5.0  # Taxa estimada USD/BRL
    preco_voluntario_usd = 7.48
    preco_voluntario_brl = preco_voluntario_usd * taxa_usd_brl
    
    # Cenários
    cenarios = {
        'Otimista (Mercado Regulado)': {
            'preco': preco_carbono_brl,
            'descricao': f'Preço atual: €{preco_carbono_eur:.2f} = R${preco_carbono_brl:.2f}/tCO₂eq',
            'valor_total': total_evitado_tco2eq * preco_carbono_brl
        },
        'Base (Mercado Voluntário)': {
            'preco': preco_voluntario_brl,
            'descricao': f'Preço voluntário: USD {preco_voluntario_usd:.2f} = R${preco_voluntario_brl:.2f}/tCO₂eq',
            'valor_total': total_evitado_tco2eq * preco_voluntario_brl
        },
        'Pessimista (Sem Créditos)': {
            'preco': 0.0,
            'descricao': 'Não consegue vender créditos de carbono',
            'valor_total': 0.0
        }
    }
    
    return cenarios

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
    if 'run_continuous_simulation' not in st.session_state:
        st.session_state.run_continuous_simulation = False

# =============================================================================
# FUNÇÃO PARA EXIBIR COTAÇÃO DO CARBONO NO PAINEL LATERAL
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
    
    # Informações adicionais
    with st.sidebar.expander("ℹ️ Informações do Mercado de Carbono"):
        st.markdown(f"""
        **📊 Cotações Atuais:**
        - **Fonte do Carbono:** {st.session_state.fonte_cotacao}
        - **Preço Atual:** {st.session_state.moeda_carbono} {st.session_state.preco_carbono:.2f}/tCO₂eq
        - **Câmbio EUR/BRL:** 1 Euro = R$ {st.session_state.taxa_cambio:.2f}
        - **Carbono em Reais:** R$ {preco_carbono_reais:.2f}/tCO₂eq
        
        **🌍 Comparação de Mercados:**
        - **Mercado Voluntário:** ~USD 7.48 ≈ R$ 37.40/tCO₂eq
        - **Mercado Regulado (EU ETS):** ~€85.57 ≈ R$ 544.23/tCO₂eq
        
        **💡 Importante:**
        - Os preços são baseados no mercado regulado da UE
        - Valores em tempo real sujeitos a variações de mercado
        - Conversão para Real utilizando câmbio comercial
        """)

# =============================================================================
# FUNÇÃO PARA FORMATAR NÚMEROS NO PADRÃO BRASILEIRO
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
# CONFIGURAÇÃO PRINCIPAL DO APLICATIVO
# =============================================================================

# Título do aplicativo
st.title("🔬 Estimação do Potencial de Metano - Lote de 100 kg")
st.markdown("""
**Análise Comparativa: Aterro vs Vermicompostagem vs Compostagem**

Este simulador calcula o potencial de geração de metano de um lote de 100 kg de resíduos orgânicos
em três diferentes cenários de gestão, com análise financeira baseada no mercado de carbono.
""")

# Inicializar session state
inicializar_session_state()

# =============================================================================
# PAINEL LATERAL COM PARÂMETROS
# =============================================================================

# Exibir cotação do carbono
exibir_cotacao_carbono()

# Parâmetros de entrada
with st.sidebar:
    st.header("⚙️ Parâmetros de Entrada - Brasil")
    
    # Seletor de tipo de simulação
    tipo_simulacao = st.radio(
        "Tipo de Simulação",
        ["Lote Único", "Entrada Contínua (1 lote/dia por 20 anos)"],
        help="Escolha entre analisar um lote único ou simular entrada contínua"
    )
    
    if tipo_simulacao == "Lote Único":
        # Entrada principal de resíduos (fixo em 100 kg para o lote)
        st.subheader("📦 Lote de Resíduos")
        residuos_kg = st.number_input(
            "Peso do lote (kg)", 
            min_value=10, 
            max_value=1000, 
            value=100, 
            step=10,
            help="Peso do lote de resíduos orgânicos para análise"
        )
        
        st.subheader("📊 Parâmetros Ambientais")
        
        umidade_valor = st.slider(
            "Umidade do resíduo (%)", 
            50, 95, 85, 1,
            help="Percentual de umidade dos resíduos orgânicos"
        )
        umidade = umidade_valor / 100.0
        
        temperatura = st.slider(
            "Temperatura média (°C)", 
            15, 35, 25, 1,
            help="Temperatura média ambiente (importante para cálculo do DOCf)"
        )
        
        st.subheader("⏰ Período de Análise")
        dias_simulacao = st.slider(
            "Dias de simulação", 
            50, 1000, 365, 50,
            help="Período total da simulação em dias"
        )
        
        if st.button("🚀 Calcular Potencial de Metano (Lote Único)", type="primary"):
            st.session_state.run_simulation = True
            st.session_state.run_continuous_simulation = False
            
    else:  # Entrada Contínua
        st.subheader("📦 Sistema de Entrada Contínua")
        
        residuos_kg_dia = st.number_input(
            "Peso do lote diário (kg/dia)", 
            min_value=10, 
            max_value=1000, 
            value=100, 
            step=10,
            help="Peso de cada lote diário de resíduos orgânicos"
        )
        
        st.subheader("📊 Parámetros Ambientais")
        
        umidade_valor = st.slider(
            "Umidade do resíduo (%)", 
            50, 95, 85, 1,
            help="Percentual de umidade dos resíduos orgânicos"
        )
        umidade = umidade_valor / 100.0
        
        temperatura = st.slider(
            "Temperatura média (°C)", 
            15, 35, 25, 1,
            help="Temperatura média ambiente (importante para cálculo do DOCf)"
        )
        
        st.subheader("⏰ Período de Análise")
        anos_simulacao = st.slider(
            "Anos de simulação", 
            5, 50, 20, 5,
            help="Período total da simulação em anos"
        )
        
        if st.button("🚀 Simular Entrada Contínua (20 anos)", type="primary"):
            st.session_state.run_continuous_simulation = True
            st.session_state.run_simulation = False

# =============================================================================
# EXECUÇÃO DA SIMULAÇÃO PARA LOTE ÚNICO (ORIGINAL)
# =============================================================================

if st.session_state.get('run_simulation', False) and tipo_simulacao == "Lote Único":
    with st.spinner('Calculando potencial de metano para os três cenários...'):
        
        # =====================================================================
        # 1. CÁLCULO DO POTENCIAL DE METANO PARA CADA CENÁRIO
        # =====================================================================
        
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
        
        # =====================================================================
        # 2. CRIAR DATAFRAME COM OS RESULTADOS
        # =====================================================================
        
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
        
        # =====================================================================
        # 3. EXIBIR RESULTADOS PRINCIPAIS
        # =====================================================================
        
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
        
        # Exibir parâmetros intermediários de cálculo
        with st.expander("🧮 Detalhes dos Cálculos Intermediários", expanded=False):
            st.markdown(f"""
            **Parâmetros Intermediários para Aterro:**
            - **DOC (Carbono Orgânico Degradável):** 0.15 (fração)
            - **DOCf (fração degradável):** {DOCf:.4f} (calculado: 0.0147 × {temperatura}°C + 0.28)
            - **MCF (Fator de Correção de Metano):** 1.0
            - **F (Fração de Metano no Biogás):** 0.5
            - **OX (Fator de Oxidação):** 0.1
            - **Ri (Metano Recuperado):** 0.0
            - **Potencial CH₄ por kg:** {formatar_br(total_aterro/residuos_kg if residuos_kg > 0 else 0)} kg CH₄/kg resíduo
            
            **Parâmetros para Vermicompostagem:**
            - **TOC (Carbono Orgânico Total):** 0.436
            - **CH₄_C_FRAC (Fração emitida):** 0.13%
            - **Matéria Seca:** {formatar_br((1-umidade)*100)}%
            
            **Parâmetros para Compostagem:**
            - **TOC (Carbono Orgânico Total):** 0.436
            - **CH₄_C_FRAC (Fração emitida):** 0.6%
            - **Matéria Seca:** {formatar_br((1-umidade)*100)}%
            """)
        
        # =====================================================================
        # 4. GRÁFICO: REDUÇÃO DE EMISSÕES ACUMULADA
        # =====================================================================
        
        st.subheader("📉 Redução de Emissões Acumulada (CH₄)")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Configurar formatação
        br_formatter = FuncFormatter(br_format)
        
        # Plotar linhas de acumulado
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
        
        # Configurar gráfico
        ax.set_title(f'Acumulado de Metano em {dias_simulacao} Dias - Lote de {residuos_kg} kg', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Data')
        ax.set_ylabel('Metano Acumulado (kg CH₄)')
        ax.legend(title='Cenário de Gestão', loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.yaxis.set_major_formatter(br_formatter)
        
        # Rotacionar labels do eixo x
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        st.pyplot(fig)
        
        # =====================================================================
        # 5. GRÁFICO: EMISSÕES DIÁRIAS COMPARATIVAS
        # =====================================================================
        
        st.subheader("📈 Emissões Diárias de Metano")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plotar emissões diárias (apenas primeiros 100 dias para melhor visualização)
        dias_exibir = min(100, dias_simulacao)
        
        # Criar gráfico com barras para visualizar melhor as diferenças
        x_pos = np.arange(dias_exibir)
        bar_width = 0.25
        
        # Usar barras para visualização mais clara
        ax.bar(x_pos - bar_width, df['Aterro_CH4_kg'][:dias_exibir], bar_width, 
                label='Aterro', color='red', alpha=0.7)
        ax.bar(x_pos, df['Vermicompostagem_CH4_kg'][:dias_exibir], bar_width, 
                label='Vermicompostagem', color='green', alpha=0.7)
        ax.bar(x_pos + bar_width, df['Compostagem_CH4_kg'][:dias_exibir], bar_width, 
                label='Compostagem', color='blue', alpha=0.7)
        
        ax.set_xlabel('Dias')
        ax.set_ylabel('Metano (kg CH₄/dia)')
        ax.set_title(f'Emissões Diárias de Metano (Primeiros {dias_exibir} Dias)', 
                    fontsize=14, fontweight='bold')
        ax.legend(title='Cenário')
        ax.grid(True, linestyle='--', alpha=0.5, axis='y')
        ax.yaxis.set_major_formatter(br_formatter)
        
        # Ajustar ticks do eixo x
        ax.set_xticks(x_pos[::10])
        ax.set_xticklabels([f'Dia {i+1}' for i in x_pos[::10]])
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # =====================================================================
        # 6. EXIBIR FÓRMULAS UTILIZADAS
        # =====================================================================
        
        with st.expander("🧮 Fórmulas Utilizadas nos Cálculos", expanded=False):
            st.markdown(f"""
            ### **Aterro Sanitário (IPCC 2006)**
            
            **Potencial de Metano por kg de resíduo:**
            ```
            CH₄_por_kg = DOC × DOCf × MCF × F × (16/12) × (1 - OX) × (1 - Ri)
            ```
            
            **Onde:**
            - **DOC** = Carbono Orgânico Degradável = 0.15 (fração)
            - **DOCf** = 0.0147 × T + 0.28 (T = {temperatura}°C) = {DOCf:.4f}
            - **MCF** = Fator de Correção de Metano = 1.0 (aterros sanitários)
            - **F** = Fração de Metano no Biogás = 0.5
            - **OX** = Fator de Oxidação = 0.1
            - **Ri** = Metano Recuperado = 0.0
            
            **Decaimento temporal (primeira ordem):**
            ```
            k_ano = 0.06 (constante de decaimento anual)
            Emissão(t) = Potencial Total × [exp(-k_ano×(t-1)/365) - exp(-k_ano×t/365)]
            ```
            
            ### **Vermicompostagem (Yang et al. 2017)**
            
            **Metano total por lote:**
            ```
            CH₄_total = Resíduos × TOC × CH₄_C_FRAC × (16/12) × (1 - Umidade)
            CH₄_total = {residuos_kg} × 0.436 × 0.0013 × (16/12) × (1 - {umidade:.3f})
            CH₄_total = {formatar_br(total_vermi)} kg CH₄
            ```
            
            **Onde:**
            - **TOC** = Carbono Orgânico Total = 0.436 (fração)
            - **CH₄_C_FRAC** = Fração do TOC emitida como CH₄-C = 0.0013 (0.13%)
            - **Perfil temporal**: Distribuição baseada em Yang et al. (2017) - {dias_vermi} dias
            
            ### **Compostagem Termofílica (Yang et al. 2017)**
            
            **Metano total por lote:**
            ```
            CH₄_total = Resíduos × TOC × CH₄_C_FRAC × (16/12) × (1 - Umidade)
            CH₄_total = {residuos_kg} × 0.436 × 0.006 × (16/12) × (1 - {umidade:.3f})
            CH₄_total = {formatar_br(total_compost)} kg CH₄
            ```
            
            **Onde:**
            - **TOC** = Carbono Orgânico Total = 0.436 (fração)
            - **CH₄_C_FRAC** = Fração do TOC emitida como CH₄-C = 0.006 (0.6%)
            - **Perfil temporal**: Distribuição específica para compostagem termofílica - {dias_compost} dias
            """)
        
        # =====================================================================
        # 7. SIMULAÇÃO DOS TRÊS CENÁRIOS FINANCEIROS
        # =====================================================================
        
        st.header("💰 Simulação de Cenários Financeiros - Mercado de Carbono")
        
        # Converter metano para CO₂eq (GWP CH₄ = 27.9 para 100 anos - IPCC AR6)
        GWP_CH4 = 27.9  # kg CO₂eq por kg CH₄
        
        total_evitado_vermi_kg = (total_aterro - total_vermi) * GWP_CH4
        total_evitado_vermi_tco2eq = total_evitado_vermi_kg / 1000
        
        total_evitado_compost_kg = (total_aterro - total_compost) * GWP_CH4
        total_evitado_compost_tco2eq = total_evitado_compost_kg / 1000
        
        # Simular cenários financeiros
        cenarios_vermi = simular_cenarios_financeiros(
            total_evitado_vermi_tco2eq, 
            st.session_state.preco_carbono,
            st.session_state.taxa_cambio
        )
        
        cenarios_compost = simular_cenarios_financeiros(
            total_evitado_compost_tco2eq,
            st.session_state.preco_carbono,
            st.session_state.taxa_cambio
        )
        
        # Exibir métricas de CO₂eq
        st.subheader("🌍 Impacto em CO₂eq (Potencial de Aquecimento Global)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Formatar com 6 casas decimais para mostrar diferenças sutis
            valor_vermi_formatado = f"{total_evitado_vermi_tco2eq:,.6f}".replace(",", "X").replace(".", ",").replace("X", ".")
            st.metric(
                "Vermicompostagem",
                f"{valor_vermi_formatado} tCO₂eq",
                help=f"Equivalente a {formatar_br(total_evitado_vermi_tco2eq * 1000)} kg CO₂eq"
            )
        
        with col2:
            # Formatar com 6 casas decimais para mostrar diferenças sutis
            valor_compost_formatado = f"{total_evitado_compost_tco2eq:,.6f}".replace(",", "X").replace(".", ",").replace("X", ".")
            st.metric(
                "Compostagem",
                f"{valor_compost_formatado} tCO₂eq",
                help=f"Equivalente a {formatar_br(total_evitado_compost_tco2eq * 1000)} kg CO₂eq"
            )
        
        # Calcular e mostrar a diferença percentual
        diferenca_percentual = ((total_evitado_vermi_tco2eq - total_evitado_compost_tco2eq) / total_evitado_compost_tco2eq * 100) if total_evitado_compost_tco2eq > 0 else 0
        
        st.caption(f"📊 **Diferença:** A vermicompostagem evita {diferenca_percentual:+.4f}% mais emissões que a compostagem termofílica")
        
        # Exibir tabela comparativa com mais casas decimais
        st.subheader("📊 Comparação de Cenários Financeiros")
        
        dados_comparativos = []
        for cenario in ['Otimista (Mercado Regulado)', 'Base (Mercado Voluntário)', 'Pessimista (Sem Créditos)']:
            dados_comparativos.append({
                'Cenário': cenario,
                'Descrição': cenarios_vermi[cenario]['descricao'],
                'Vermicompostagem (R$)': formatar_br(cenarios_vermi[cenario]['valor_total']),
                'Compostagem (R$)': formatar_br(cenarios_compost[cenario]['valor_total']),
                'Diferença (R$)': formatar_br(cenarios_vermi[cenario]['valor_total'] - cenarios_compost[cenario]['valor_total'])
            })
        
        df_comparativo = pd.DataFrame(dados_comparativos)
        st.dataframe(df_comparativo, use_container_width=True)
        
        # Gráfico de barras comparativo
        fig, ax = plt.subplots(figsize=(10, 6))
        
        cenarios_nomes = list(cenarios_vermi.keys())
        valores_vermi = [cenarios_vermi[c]['valor_total'] for c in cenarios_nomes]
        valores_compost = [cenarios_compost[c]['valor_total'] for c in cenarios_nomes]
        
        x = np.arange(len(cenarios_nomes))
        width = 0.35
        
        ax.bar(x - width/2, valores_vermi, width, label='Vermicompostagem', color='green', alpha=0.8)
        ax.bar(x + width/2, valores_compost, width, label='Compostagem', color='blue', alpha=0.8)
        
        ax.set_xlabel('Cenário Financeiro')
        ax.set_ylabel('Valor Financeiro (R$)')
        ax.set_title('Valor dos Créditos de Carbono por Cenário')
        ax.set_xticks(x)
        ax.set_xticklabels([c.split('(')[0].strip() for c in cenarios_nomes])
        ax.legend()
        ax.yaxis.set_major_formatter(br_formatter)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Adicionar valores nas barras
        for i, (v1, v2) in enumerate(zip(valores_vermi, valores_compost)):
            if v1 > 0:
                ax.text(i - width/2, v1 + max(v1, v2)*0.01, f'R${v1:,.0f}', 
                       ha='center', fontsize=9, fontweight='bold')
            if v2 > 0:
                ax.text(i + width/2, v2 + max(v1, v2)*0.01, f'R${v2:,.0f}', 
                       ha='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # =====================================================================
        # 8. RESUMO DETALHADO
        # =====================================================================
        
        with st.expander("📋 Resumo Detalhado da Análise", expanded=False):
            st.markdown(f"""
            ### **Resumo da Análise - Lote de {residuos_kg} kg**
            
            **Parâmetros Utilizados:**
            - Umidade: {umidade_valor}% ({formatar_br(umidade)} fração)
            - Temperatura: {temperatura}°C
            - Período: {dias_simulacao} dias
            - GWP CH₄ (IPCC AR6): 27.9 kg CO₂eq/kg CH₄
            
            **Resultados de Metano:**
            - **Aterro Sanitário:** {formatar_br(total_aterro)} kg CH₄
            - **Vermicompostagem:** {formatar_br(total_vermi)} kg CH₄
            - **Compostagem Termofílica:** {formatar_br(total_compost)} kg CH₄
            
            **Reduções em Relação ao Aterro:**
            - **Vermicompostagem:** {formatar_br(total_aterro - total_vermi)} kg CH₄ ({formatar_br(reducao_vermi_perc)}%)
            - **Compostagem:** {formatar_br(total_aterro - total_compost)} kg CH₄ ({formatar_br(reducao_compost_perc)}%)
            
            **Em CO₂eq Evitadas (com alta precisão):**
            - **Vermicompostagem:** {valor_vermi_formatado} tCO₂eq
            - **Compostagem:** {valor_compost_formatado} tCO₂eq
            - **Diferença:** {diferenca_percentual:+.4f}%
            
            **Cenário Financeiro Mais Favorável (Regulado):**
            - **Vermicompostagem:** R$ {formatar_br(cenarios_vermi['Otimista (Mercado Regulado)']['valor_total'])}
            - **Compostagem:** R$ {formatar_br(cenarios_compost['Otimista (Mercado Regulado)']['valor_total'])}
            
            **💡 Conclusão:**
            A vermicompostagem apresenta maior potencial de redução de emissões de metano 
            ({formatar_br(reducao_vermi_perc)}% vs {formatar_br(reducao_compost_perc)}% da compostagem),
            resultando em maior valor financeiro potencial no mercado de carbono.
            
            **⚖️ Viabilidade Financeira:**
            - **Mercado Regulado:** Projeto altamente viável para ambas tecnologias
            - **Mercado Voluntário:** Viabilidade moderada, depende de outros benefícios
            - **Sem Créditos:** Necessidade de subsídios ou outras fontes de receita
            """)

# =============================================================================
# EXECUÇÃO DA SIMULAÇÃO PARA ENTRADA CONTÍNUA (NOVA FUNCIONALIDADE)
# =============================================================================

elif st.session_state.get('run_continuous_simulation', False) and tipo_simulacao == "Entrada Contínua (1 lote/dia por 20 anos)":
    with st.spinner('Calculando potencial de metano para entrada contínua de 1 lote por dia durante 20 anos...'):
        
        # =====================================================================
        # 1. CÁLCULO DO POTENCIAL DE METANO PARA ENTRADA CONTÍNUA
        # =====================================================================
        
        # Aterro Sanitário - entrada contínua
        emissoes_aterro_cont, total_aterro_cont, DOCf = calcular_emissoes_aterro_continuo(
            residuos_kg_dia, umidade, temperatura, anos_simulacao
        )
        
        # Vermicompostagem - entrada contínua
        emissoes_vermi_cont, total_vermi_cont = calcular_emissoes_vermicompostagem_continuo(
            residuos_kg_dia, umidade, anos_simulacao
        )
        
        # Compostagem Termofílica - entrada contínua
        emissoes_compost_cont, total_compost_cont = calcular_emissoes_compostagem_continuo(
            residuos_kg_dia, umidade, anos_simulacao
        )
        
        # =====================================================================
        # 2. CRIAR DATAFRAME COM OS RESULTADOS
        # =====================================================================
        
        dias_total = anos_simulacao * 365
        datas = pd.date_range(start=datetime.now(), periods=dias_total, freq='D')
        
        df_cont = pd.DataFrame({
            'Data': datas,
            'Aterro_CH4_kg_dia': emissoes_aterro_cont,
            'Vermicompostagem_CH4_kg_dia': emissoes_vermi_cont,
            'Compostagem_CH4_kg_dia': emissoes_compost_cont
        })
        
        # Calcular valores acumulados
        df_cont['Aterro_Acumulado'] = df_cont['Aterro_CH4_kg_dia'].cumsum()
        df_cont['Vermi_Acumulado'] = df_cont['Vermicompostagem_CH4_kg_dia'].cumsum()
        df_cont['Compost_Acumulado'] = df_cont['Compostagem_CH4_kg_dia'].cumsum()
        
        # Calcular reduções (evitadas) em relação ao aterro
        df_cont['Reducao_Vermi'] = df_cont['Aterro_Acumulado'] - df_cont['Vermi_Acumulado']
        df_cont['Reducao_Compost'] = df_cont['Aterro_Acumulado'] - df_cont['Compost_Acumulado']
        
        # Agrupar por ano para análise anual
        df_cont['Ano'] = df_cont['Data'].dt.year
        df_anual = df_cont.groupby('Ano').agg({
            'Aterro_CH4_kg_dia': 'sum',
            'Vermicompostagem_CH4_kg_dia': 'sum',
            'Compostagem_CH4_kg_dia': 'sum',
            'Reducao_Vermi': 'last',
            'Reducao_Compost': 'last'
        }).reset_index()
        
        # =====================================================================
        # 3. EXIBIR RESULTADOS PRINCIPAIS - ENTRADA CONTÍNUA
        # =====================================================================
        
        st.header(f"📊 Resultados - Entrada Contínua ({anos_simulacao} anos)")
        
        # Métricas principais
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Aterro Sanitário",
                f"{formatar_br(total_aterro_cont)} kg CH₄",
                help=f"Total acumulado em {anos_simulacao} anos"
            )
        
        with col2:
            reducao_vermi_kg = total_aterro_cont - total_vermi_cont
            reducao_vermi_perc = (1 - total_vermi_cont/total_aterro_cont)*100 if total_aterro_cont > 0 else 0
            st.metric(
                "Vermicompostagem",
                f"{formatar_br(total_vermi_cont)} kg CH₄",
                delta=f"-{formatar_br(reducao_vermi_perc)}%",
                delta_color="inverse",
                help=f"Redução de {formatar_br(reducao_vermi_kg)} kg vs aterro"
            )
        
        with col3:
            reducao_compost_kg = total_aterro_cont - total_compost_cont
            reducao_compost_perc = (1 - total_compost_cont/total_aterro_cont)*100 if total_aterro_cont > 0 else 0
            st.metric(
                "Compostagem Termofílica",
                f"{formatar_br(total_compost_cont)} kg CH₄",
                delta=f"-{formatar_br(reducao_compost_perc)}%",
                delta_color="inverse",
                help=f"Redução de {formatar_br(reducao_compost_kg)} kg vs aterro"
            )
        
        # Métricas anuais
        st.subheader("📈 Métricas Anuais Médias")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            media_anual_aterro = total_aterro_cont / anos_simulacao
            st.metric(
                "Aterro (média anual)",
                f"{formatar_br(media_anual_aterro)} kg CH₄/ano",
                help="Produção média anual de metano no aterro"
            )
        
        with col2:
            media_anual_vermi = total_vermi_cont / anos_simulacao
            st.metric(
                "Vermicompostagem (média anual)",
                f"{formatar_br(media_anual_vermi)} kg CH₄/ano",
                help="Produção média anual de metano na vermicompostagem"
            )
        
        with col3:
            media_anual_compost = total_compost_cont / anos_simulacao
            st.metric(
                "Compostagem (média anual)",
                f"{formatar_br(media_anual_compost)} kg CH₄/ano",
                help="Produção média anual de metano na compostagem"
            )
        
        # =====================================================================
        # 4. GRÁFICO: REDUÇÃO DE EMISSÕES ACUMULADA (20 ANOS) - NOVA FUNCIONALIDADE
        # =====================================================================
        
        st.subheader(f"📉 Redução de Emissões Acumulada ({anos_simulacao} anos) - 1 Lote/dia")
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Configurar formatação
        br_formatter = FuncFormatter(br_format)
        
        # Plotar linhas de acumulado
        ax.plot(df_cont['Data'], df_cont['Aterro_Acumulado'], 'r-', 
                label='Aterro Sanitário', linewidth=3, alpha=0.7)
        ax.plot(df_cont['Data'], df_cont['Vermi_Acumulado'], 'g-', 
                label='Vermicompostagem', linewidth=2)
        ax.plot(df_cont['Data'], df_cont['Compost_Acumulado'], 'b-', 
                label='Compostagem Termofílica', linewidth=2)
        
        # Área de redução (evitadas)
        ax.fill_between(df_cont['Data'], df_cont['Vermi_Acumulado'], df_cont['Aterro_Acumulado'],
                        color='green', alpha=0.3, label='Redução Vermicompostagem')
        ax.fill_between(df_cont['Data'], df_cont['Compost_Acumulado'], df_cont['Aterro_Acumulado'],
                        color='blue', alpha=0.2, label='Redução Compostagem')
        
        # Configurar gráfico
        ax.set_title(f'Acumulado de Metano em {anos_simulacao} Anos - Entrada de {residuos_kg_dia} kg/dia', 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('Ano')
        ax.set_ylabel('Metano Acumulado (kg CH₄)')
        ax.legend(title='Cenário de Gestão', loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.yaxis.set_major_formatter(br_formatter)
        
        # Ajustar ticks do eixo x para mostrar anos
        anos = df_cont['Data'].dt.year.unique()
        ax.set_xticks([df_cont['Data'].iloc[0] + pd.DateOffset(years=i) for i in range(0, anos_simulacao + 1, max(1, anos_simulacao//10))])
        ax.set_xticklabels([f'Ano {i}' for i in range(0, anos_simulacao + 1, max(1, anos_simulacao//10))])
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        st.pyplot(fig)
        
        # =====================================================================
        # 5. GRÁFICO: EMISSÕES DIÁRIAS (PRIMEIROS 2 ANOS)
        # =====================================================================
        
        st.subheader("📊 Emissões Diárias de Metano (Primeiros 2 Anos)")
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Plotar apenas primeiros 2 anos (730 dias) para melhor visualização
        dias_exibir = min(730, dias_total)
        
        # Criar gráfico com linhas para visualizar tendências
        ax.plot(df_cont['Data'][:dias_exibir], df_cont['Aterro_CH4_kg_dia'][:dias_exibir], 
                'r-', label='Aterro', linewidth=1.5, alpha=0.7)
        ax.plot(df_cont['Data'][:dias_exibir], df_cont['Vermicompostagem_CH4_kg_dia'][:dias_exibir], 
                'g-', label='Vermicompostagem', linewidth=1.5, alpha=0.7)
        ax.plot(df_cont['Data'][:dias_exibir], df_cont['Compostagem_CH4_kg_dia'][:dias_exibir], 
                'b-', label='Compostagem', linewidth=1.5, alpha=0.7)
        
        ax.set_xlabel('Data')
        ax.set_ylabel('Metano (kg CH₄/dia)')
        ax.set_title(f'Emissões Diárias de Metano - Primeiros {dias_exibir//365} Anos', 
                    fontsize=14, fontweight='bold')
        ax.legend(title='Cenário')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.yaxis.set_major_formatter(br_formatter)
        
        # Ajustar ticks do eixo x
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        st.pyplot(fig)
        
        # =====================================================================
        # 6. GRÁFICO: COMPARAÇÃO ANUAL
        # =====================================================================
        
        st.subheader("📈 Comparação Anual das Emissões")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Gráfico 1: Emissões anuais totais
        bar_width = 0.25
        x_pos = np.arange(len(df_anual['Ano']))
        
        ax1.bar(x_pos - bar_width, df_anual['Aterro_CH4_kg_dia'], bar_width, 
                label='Aterro', color='red', alpha=0.7)
        ax1.bar(x_pos, df_anual['Vermicompostagem_CH4_kg_dia'], bar_width, 
                label='Vermicompostagem', color='green', alpha=0.7)
        ax1.bar(x_pos + bar_width, df_anual['Compostagem_CH4_kg_dia'], bar_width, 
                label='Compostagem', color='blue', alpha=0.7)
        
        ax1.set_xlabel('Ano')
        ax1.set_ylabel('Metano Anual (kg CH₄)')
        ax1.set_title('Emissões Anuais de Metano por Cenário')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(df_anual['Ano'])
        ax1.legend()
        ax1.yaxis.set_major_formatter(br_formatter)
        ax1.grid(axis='y', linestyle='--', alpha=0.5)
        
        # Gráfico 2: Redução acumulada anual
        ax2.plot(df_anual['Ano'], df_anual['Reducao_Vermi'], 'g-', 
                label='Redução Vermicompostagem', linewidth=2, marker='o')
        ax2.plot(df_anual['Ano'], df_anual['Reducao_Compost'], 'b-', 
                label='Redução Compostagem', linewidth=2, marker='s')
        
        ax2.set_xlabel('Ano')
        ax2.set_ylabel('Metano Evitado Acumulado (kg CH₄)')
        ax2.set_title('Redução Acumulada de Metano vs Aterro')
        ax2.legend()
        ax2.yaxis.set_major_formatter(br_formatter)
        ax2.grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # =====================================================================
        # 7. RESUMO DAS EMISSÕES EVITADAS - NOVA SEÇÃO (BASEADO EM v2n_noAr)
        # =====================================================================
        
        st.markdown("---")
        st.header("📊 Resumo das Emissões Evitadas")
        
        # Calcular emissões evitadas para ambas metodologias em tCO₂eq
        GWP_CH4 = 27.9  # kg CO₂eq por kg CH₄ (IPCC AR6)
        
        # Metodologia da Tese (Vermicompostagem)
        total_evitado_tese_kg = (total_aterro_cont - total_vermi_cont) * GWP_CH4
        total_evitado_tese_tco2eq = total_evitado_tese_kg / 1000
        media_anual_tese = total_evitado_tese_tco2eq / anos_simulacao
        
        # Metodologia UNFCCC (Compostagem Termofílica)
        total_evitado_unfccc_kg = (total_aterro_cont - total_compost_cont) * GWP_CH4
        total_evitado_unfccc_tco2eq = total_evitado_unfccc_kg / 1000
        media_anual_unfccc = total_evitado_unfccc_tco2eq / anos_simulacao
        
        # Layout com duas colunas
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📋 Metodologia da Tese (Vermicompostagem)")
            st.metric(
                "Total de emissões evitadas", 
                f"{formatar_br(total_evitado_tese_tco2eq)} tCO₂eq",
                help=f"Total acumulado em {anos_simulacao} anos"
            )
            st.metric(
                "Média anual", 
                f"{formatar_br(media_anual_tese)} tCO₂eq/ano",
                help=f"Emissões evitadas por ano em média"
            )
        
        with col2:
            st.markdown("#### 📋 Metodologia UNFCCC (Compostagem Termofílica)")
            st.metric(
                "Total de emissões evitadas", 
                f"{formatar_br(total_evitado_unfccc_tco2eq)} tCO₂eq",
                help=f"Total acumulado em {anos_simulacao} anos"
            )
            st.metric(
                "Média anual", 
                f"{formatar_br(media_anual_unfccc)} tCO₂eq/ano",
                help=f"Emissões evitadas por ano em média"
            )
        
        # Comparação percentual
        diferenca_absoluta = total_evitado_tese_tco2eq - total_evitado_unfccc_tco2eq
        diferenca_percentual = (diferenca_absoluta / total_evitado_unfccc_tco2eq * 100) if total_evitado_unfccc_tco2eq > 0 else 0
        
        st.caption(f"""
        📈 **Comparação:** A metodologia da Tese (Vermicompostagem) evita **{formatar_br(diferenca_absoluta)} tCO₂eq** 
        ({diferenca_percentual:+.2f}%) a mais que a metodologia UNFCCC em {anos_simulacao} anos.
        """)
        
        # Adicionar explicação sobre as metodologias
        with st.expander("🔍 Entenda as Metodologias", expanded=False):
            st.markdown(f"""
            **📋 Metodologia da Tese (Vermicompostagem em Reatores):**
            
            **Base Científica:**
            - **Fonte:** Yang et al. (2017) - Greenhouse gas emissions during MSW landfilling in China
            - **CH₄_C_FRAC:** 0.13% do Carbono Orgânico Total (TOC) emitido como CH₄-C
            - **Perfil Temporal:** 50 dias com distribuição específica para vermicompostagem
            - **Processo:** Compostagem acelerada com minhocas (Eisenia fetida) em reatores controlados
            
            **Cálculo do Metano:**
            ```
            CH₄_total = Resíduos × TOC × CH₄_C_FRAC × (16/12) × (1 - Umidade)
            CH₄_total = {residuos_kg_dia} kg/dia × 0.436 × 0.0013 × 1.333 × (1 - {umidade:.3f})
            CH₄_total = {formatar_br(media_anual_vermi)} kg CH₄/ano (média)
            ```
            
            **📋 Metodologia UNFCCC (Compostagem Termofílica a Céu Aberto):**
            
            **Base Científica:**
            - **Fonte:** UNFCCC (2016) - Clean Development Mechanism - Methodology AMS-III.F
            - **CH₄_C_FRAC:** 0.6% do Carbono Orgânico Total (TOC) emitido como CH₄-C
            - **Perfil Temporal:** 50 dias com pico termofílico
            - **Processo:** Compostagem tradicional sem minhocas, em leiras a céu aberto
            
            **Cálculo do Metano:**
            ```
            CH₄_total = Resíduos × TOC × CH₄_C_FRAC × (16/12) × (1 - Umidade)
            CH₄_total = {residuos_kg_dia} kg/dia × 0.436 × 0.006 × 1.333 × (1 - {umidade:.3f})
            CH₄_total = {formatar_br(media_anual_compost)} kg CH₄/ano (média)
            ```
            
            **🌍 Conversão para CO₂eq:**
            ```
            CO₂eq = CH₄ (kg) × GWP_CH₄ (27.9) ÷ 1000
            GWP_CH₄ = 27.9 kg CO₂eq/kg CH₄ (IPCC AR6, 100 anos)
            ```
            
            **⚖️ Por que a diferença?**
            - **Vermicompostagem:** Processo mais controlado, menor produção de metano (0.13% vs 0.6%)
            - **Compostagem tradicional:** Maior temperatura, condições mais favoráveis à metanogênese
            - **Eficiência:** As minhocas aceleram a decomposição aeróbica, reduzindo condições anaeróbicas
            
            **📊 Resumo dos Parâmetros:**
            - **Resíduos processados:** {residuos_kg_dia} kg/dia = {formatar_br(residuos_kg_dia * 365 / 1000)} ton/ano
            - **Período:** {anos_simulacao} anos ({dias_total} dias)
            - **Umidade:** {umidade_valor}%
            - **Temperatura:** {temperatura}°C
            - **GWP CH₄:** 27.9 kg CO₂eq/kg CH₄
            """)
        
        # =====================================================================
        # 8. GRÁFICO: EMISSÕES EVITADAS ANUAIS (tCO₂eq) - NOVO GRÁFICO
        # =====================================================================
        
        st.subheader("📈 Emissões Evitadas Anuais (tCO₂eq)")
        
        # Calcular emissões evitadas anuais em tCO₂eq
        df_anual['Evitado_Tese_tCO2eq'] = (df_anual['Aterro_CH4_kg_dia'] - df_anual['Vermicompostagem_CH4_kg_dia']) * GWP_CH4 / 1000
        df_anual['Evitado_UNFCCC_tCO2eq'] = (df_anual['Aterro_CH4_kg_dia'] - df_anual['Compostagem_CH4_kg_dia']) * GWP_CH4 / 1000
        
        # Calcular acumulado
        df_anual['Acumulado_Tese'] = df_anual['Evitado_Tese_tCO2eq'].cumsum()
        df_anual['Acumulado_UNFCCC'] = df_anual['Evitado_UNFCCC_tCO2eq'].cumsum()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Gráfico 1: Emissões evitadas anuais
        x_pos = np.arange(len(df_anual['Ano']))
        bar_width = 0.35
        
        bars1 = ax1.bar(x_pos - bar_width/2, df_anual['Evitado_Tese_tCO2eq'], bar_width,
                        label='Metodologia da Tese', color='green', alpha=0.8, edgecolor='black')
        bars2 = ax1.bar(x_pos + bar_width/2, df_anual['Evitado_UNFCCC_tCO2eq'], bar_width,
                        label='Metodologia UNFCCC', color='blue', alpha=0.8, edgecolor='black', hatch='//')
        
        ax1.set_xlabel('Ano')
        ax1.set_ylabel('Emissões Evitadas (tCO₂eq/ano)')
        ax1.set_title(f'Emissões Evitadas Anuais - Comparação entre Metodologias ({anos_simulacao} anos)')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(df_anual['Ano'])
        ax1.legend(title='Metodologia')
        ax1.yaxis.set_major_formatter(br_formatter)
        ax1.grid(axis='y', linestyle='--', alpha=0.5)
        
        # Adicionar valores nas barras
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax1.text(bar.get_x() + bar.get_width()/2., height + max(df_anual['Evitado_Tese_tCO2eq'].max(), 
                             df_anual['Evitado_UNFCCC_tCO2eq'].max())*0.01,
                             f'{height:,.1f}'.replace(',', 'X').replace('.', ',').replace('X', '.'),
                             ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # Gráfico 2: Emissões evitadas acumuladas
        ax2.plot(df_anual['Ano'], df_anual['Acumulado_Tese'], 'g-', 
                 label='Metodologia da Tese (Acumulado)', linewidth=2.5, marker='o', markersize=6)
        ax2.plot(df_anual['Ano'], df_anual['Acumulado_UNFCCC'], 'b-', 
                 label='Metodologia UNFCCC (Acumulado)', linewidth=2.5, marker='s', markersize=6)
        
        # Área entre as curvas (diferença)
        ax2.fill_between(df_anual['Ano'], df_anual['Acumulado_UNFCCC'], df_anual['Acumulado_Tese'],
                         color='green', alpha=0.2, label='Diferença a favor da Tese')
        
        ax2.set_xlabel('Ano')
        ax2.set_ylabel('Emissões Evitadas Acumuladas (tCO₂eq)')
        ax2.set_title('Acumulado de Emissões Evitadas - Comparação entre Metodologias')
        ax2.set_xticks(df_anual['Ano'])
        ax2.legend(title='Metodologia', loc='upper left')
        ax2.yaxis.set_major_formatter(br_formatter)
        ax2.grid(True, linestyle='--', alpha=0.5)
        
        # Adicionar valores nos pontos do acumulado
        for i, (tese, unfccc) in enumerate(zip(df_anual['Acumulado_Tese'], df_anual['Acumulado_UNFCCC'])):
            ax2.text(df_anual['Ano'].iloc[i], tese + max(df_anual['Acumulado_Tese'].max(), 
                     df_anual['Acumulado_UNFCCC'].max())*0.02,
                     f'{tese:,.1f}'.replace(',', 'X').replace('.', ',').replace('X', '.'),
                     ha='center', fontsize=8, fontweight='bold', color='green')
            ax2.text(df_anual['Ano'].iloc[i], unfccc - max(df_anual['Acumulado_Tese'].max(), 
                     df_anual['Acumulado_UNFCCC'].max())*0.02,
                     f'{unfccc:,.1f}'.replace(',', 'X').replace('.', ',').replace('X', '.'),
                     ha='center', fontsize=8, fontweight='bold', color='blue')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # =====================================================================
        # 9. TABELA DETALHADA DAS EMISSÕES EVITADAS
        # =====================================================================
        
        with st.expander("📋 Tabela Detalhada das Emissões Evitadas por Ano", expanded=False):
            # Criar DataFrame com todos os dados
            tabela_detalhada = pd.DataFrame({
                'Ano': df_anual['Ano'],
                'Aterro_CH4_kg': df_anual['Aterro_CH4_kg_dia'],
                'Vermicompostagem_CH4_kg': df_anual['Vermicompostagem_CH4_kg_dia'],
                'Compostagem_CH4_kg': df_anual['Compostagem_CH4_kg_dia'],
                'Redução_Vermi_kg': df_anual['Aterro_CH4_kg_dia'] - df_anual['Vermicompostagem_CH4_kg_dia'],
                'Redução_Compost_kg': df_anual['Aterro_CH4_kg_dia'] - df_anual['Compostagem_CH4_kg_dia'],
                'Redução_Vermi_tCO2eq': df_anual['Evitado_Tese_tCO2eq'],
                'Redução_Compost_tCO2eq': df_anual['Evitado_UNFCCC_tCO2eq'],
                'Acumulado_Tese_tCO2eq': df_anual['Acumulado_Tese'],
                'Acumulado_UNFCCC_tCO2eq': df_anual['Acumulado_UNFCCC']
            })
            
            # Formatar os números
            for col in tabela_detalhada.columns:
                if col != 'Ano':
                    tabela_detalhada[col] = tabela_detalhada[col].apply(lambda x: formatar_br(x) if not pd.isna(x) else "N/A")
            
            st.dataframe(tabela_detalhada, use_container_width=True)
            
            # Botão para download
            csv = tabela_detalhada.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download da Tabela (CSV)",
                data=csv,
                file_name=f"emissoes_evitadas_{residuos_kg_dia}kg_{anos_simulacao}anos.csv",
                mime="text/csv",
            )
        
        # =====================================================================
        # 10. ANÁLISE FINANCEIRA PARA ENTRADA CONTÍNUA
        # =====================================================================
        
        st.header("💰 Análise Financeira - Entrada Contínua")
        
        # Converter metano para CO₂eq (GWP CH₄ = 27.9 para 100 anos - IPCC AR6)
        # Nota: Já temos essas variáveis da seção anterior
        # total_evitado_tese_tco2eq e total_evitado_unfccc_tco2eq
        
        # Simular cenários financeiros para ambas metodologias
        cenarios_tese = simular_cenarios_financeiros(
            total_evitado_tese_tco2eq, 
            st.session_state.preco_carbono,
            st.session_state.taxa_cambio
        )
        
        cenarios_unfccc = simular_cenarios_financeiros(
            total_evitado_unfccc_tco2eq,
            st.session_state.preco_carbono,
            st.session_state.taxa_cambio
        )
        
        # Exibir métricas financeiras
        st.subheader("🌍 Valor Financeiro das Emissões Evitadas")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Preço do Carbono",
                f"€{st.session_state.preco_carbono:.2f}/tCO₂eq",
                help=f"Fonte: {st.session_state.fonte_cotacao}"
            )
        
        with col2:
            valor_tese_eur = total_evitado_tese_tco2eq * st.session_state.preco_carbono
            valor_tese_brl = valor_tese_eur * st.session_state.taxa_cambio
            st.metric(
                "Valor Tese (20 anos)",
                f"R$ {formatar_br(valor_tese_brl)}",
                help=f"€{formatar_br(valor_tese_eur)} em Euros"
            )
        
        with col3:
            valor_unfccc_eur = total_evitado_unfccc_tco2eq * st.session_state.preco_carbono
            valor_unfccc_brl = valor_unfccc_eur * st.session_state.taxa_cambio
            st.metric(
                "Valor UNFCCC (20 anos)",
                f"R$ {formatar_br(valor_unfccc_brl)}",
                help=f"€{formatar_br(valor_unfccc_eur)} em Euros"
            )
        
        # Tabela comparativa de cenários financeiros
        st.subheader("📊 Comparação de Cenários Financeiros (20 anos)")
        
        dados_comparativos = []
        for cenario in ['Otimista (Mercado Regulado)', 'Base (Mercado Voluntário)', 'Pessimista (Sem Créditos)']:
            dados_comparativos.append({
                'Cenário': cenario,
                'Descrição': cenarios_tese[cenario]['descricao'],
                'Metodologia Tese (R$)': formatar_br(cenarios_tese[cenario]['valor_total']),
                'Metodologia UNFCCC (R$)': formatar_br(cenarios_unfccc[cenario]['valor_total']),
                'Diferença (R$)': formatar_br(cenarios_tese[cenario]['valor_total'] - cenarios_unfccc[cenario]['valor_total']),
                'Valor Anual Tese (R$/ano)': formatar_br(cenarios_tese[cenario]['valor_total'] / anos_simulacao)
            })
        
        df_comparativo = pd.DataFrame(dados_comparativos)
        st.dataframe(df_comparativo, use_container_width=True)
        
        # Gráfico de barras comparativo
        fig, ax = plt.subplots(figsize=(12, 7))
        
        cenarios_nomes = list(cenarios_tese.keys())
        valores_tese = [cenarios_tese[c]['valor_total'] for c in cenarios_nomes]
        valores_unfccc = [cenarios_unfccc[c]['valor_total'] for c in cenarios_nomes]
        
        x = np.arange(len(cenarios_nomes))
        width = 0.35
        
        ax.bar(x - width/2, valores_tese, width, label='Metodologia da Tese', color='green', alpha=0.8)
        ax.bar(x + width/2, valores_unfccc, width, label='Metodologia UNFCCC', color='blue', alpha=0.8)
        
        ax.set_xlabel('Cenário Financeiro')
        ax.set_ylabel('Valor Financeiro (R$)')
        ax.set_title(f'Valor dos Créditos de Carbono por Cenário ({anos_simulacao} anos)')
        ax.set_xticks(x)
        ax.set_xticklabels([c.split('(')[0].strip() for c in cenarios_nomes])
        ax.legend()
        ax.yaxis.set_major_formatter(br_formatter)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Adicionar valores nas barras
        for i, (v1, v2) in enumerate(zip(valores_tese, valores_unfccc)):
            if v1 > 0:
                ax.text(i - width/2, v1 + max(v1, v2)*0.01, f'R${v1:,.0f}', 
                       ha='center', fontsize=9, fontweight='bold')
            if v2 > 0:
                ax.text(i + width/2, v2 + max(v1, v2)*0.01, f'R${v2:,.0f}', 
                       ha='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # =====================================================================
        # 11. RESUMO DETALHADO - ENTRADA CONTÍNUA
        # =====================================================================
        
        with st.expander("📋 Resumo Detalhado da Análise - Entrada Contínua", expanded=False):
            st.markdown(f"""
            ### **Resumo da Análise - Entrada Contínua ({anos_simulacao} anos)**
            
            **Parâmetros Utilizados:**
            - Lote diário: {residuos_kg_dia} kg/dia
            - Umidade: {umidade_valor}% ({formatar_br(umidade)} fração)
            - Temperatura: {temperatura}°C
            - Período: {anos_simulacao} anos ({dias_total} dias)
            - GWP CH₄ (IPCC AR6): 27.9 kg CO₂eq/kg CH₄
            
            **Resultados de Metano Acumulado ({anos_simulacao} anos):**
            - **Aterro Sanitário:** {formatar_br(total_aterro_cont)} kg CH₄
            - **Vermicompostagem:** {formatar_br(total_vermi_cont)} kg CH₄
            - **Compostagem Termofílica:** {formatar_br(total_compost_cont)} kg CH₄
            
            **Reduções em Relação ao Aterro:**
            - **Vermicompostagem:** {formatar_br(total_aterro_cont - total_vermi_cont)} kg CH₄ ({formatar_br(reducao_vermi_perc)}%)
            - **Compostagem:** {formatar_br(total_aterro_cont - total_compost_cont)} kg CH₄ ({formatar_br(reducao_compost_perc)}%)
            
            **Em CO₂eq Evitadas ({anos_simulacao} anos):**
            - **Metodologia da Tese:** {formatar_br(total_evitado_tese_tco2eq)} tCO₂eq
            - **Metodologia UNFCCC:** {formatar_br(total_evitado_unfccc_tco2eq)} tCO₂eq
            - **Diferença:** {diferenca_percentual:+.2f}%
            
            **Métricas Anuais Médias:**
            - **Aterro:** {formatar_br(media_anual_aterro)} kg CH₄/ano
            - **Vermicompostagem:** {formatar_br(media_anual_vermi)} kg CH₄/ano
            - **Compostagem:** {formatar_br(media_anual_compost)} kg CH₄/ano
            - **Redução Tese:** {formatar_br(media_anual_tese)} tCO₂eq/ano
            - **Redução UNFCCC:** {formatar_br(media_anual_unfccc)} tCO₂eq/ano
            
            **Cenário Financeiro Mais Favorável (Regulado - {anos_simulacao} anos):**
            - **Metodologia da Tese:** R$ {formatar_br(cenarios_tese['Otimista (Mercado Regulado)']['valor_total'])}
            - **Metodologia UNFCCC:** R$ {formatar_br(cenarios_unfccc['Otimista (Mercado Regulado)']['valor_total'])}
            
            **Valor Anual Médio (Regulado):**
            - **Metodologia da Tese:** R$ {formatar_br(cenarios_tese['Otimista (Mercado Regulado)']['valor_total'] / anos_simulacao)}/ano
            - **Metodologia UNFCCC:** R$ {formatar_br(cenarios_unfccc['Otimista (Mercado Regulado)']['valor_total'] / anos_simulacao)}/ano
            
            **💡 Conclusão:**
            A simulação de entrada contínua mostra que, ao longo de {anos_simulacao} anos, a vermicompostagem 
            apresenta uma redução significativa de {formatar_br(reducao_vermi_perc)}% nas emissões de metano 
            em comparação com o aterro, enquanto a compostagem reduz {formatar_br(reducao_compost_perc)}%.
            A metodologia da Tese (vermicompostagem) é {diferenca_percentual:+.2f}% mais eficiente que a 
            metodologia UNFCCC em termos de redução de emissões.
            
            **⚖️ Viabilidade Financeira em Larga Escala:**
            - **Mercado Regulado:** Projeto altamente atrativo, com retorno financeiro significativo
            - **Mercado Voluntário:** Viabilidade moderada, pode ser complementado com outras receitas
            - **Sem Créditos:** Necessidade de políticas públicas ou incentivos para viabilizar
            
            **📊 Recomendação:**
            A vermicompostagem em reatores apresenta melhor desempenho ambiental e maior potencial 
            financeiro no mercado de carbono, especialmente no cenário regulado da UE.
            """)

else:
    st.info("💡 Configure os parâmetros no painel lateral e clique no botão correspondente para iniciar a simulação.")

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
**🔄 Nova Funcionalidade: Simulação de entrada contínua (1 lote/dia por 20 anos) baseada no script v2n_noAr.**
**📊 Nova Seção: Resumo das Emissões Evitadas com comparação entre Metodologia da Tese e UNFCCC.**
""")
