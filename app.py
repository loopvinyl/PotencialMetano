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
    
    if st.button("🚀 Calcular Potencial de Metano", type="primary"):
        st.session_state.run_simulation = True

# =============================================================================
# EXECUÇÃO DA SIMULAÇÃO
# =============================================================================

if st.session_state.get('run_simulation', False):
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
            st.metric(
                "Vermicompostagem",
                f"{formatar_br(total_evitado_vermi_tco2eq)} tCO₂eq",
                help=f"Equivalente a {formatar_br(total_evitado_vermi_tco2eq * 1000)} kg CO₂eq"
            )
        
        with col2:
            st.metric(
                "Compostagem",
                f"{formatar_br(total_evitado_compost_tco2eq)} tCO₂eq",
                help=f"Equivalente a {formatar_br(total_evitado_compost_tco2eq * 1000)} kg CO₂eq"
            )
        
        # Exibir tabela comparativa
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
            
            **Em CO₂eq Evitadas:**
            - **Vermicompostagem:** {formatar_br(total_evitado_vermi_tco2eq)} tCO₂eq
            - **Compostagem:** {formatar_br(total_evitado_compost_tco2eq)} tCO₂eq
            
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

else:
    st.info("💡 Configure os parâmetros no painel lateral e clique em 'Calcular Potencial de Metano' para iniciar a simulação.")

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
