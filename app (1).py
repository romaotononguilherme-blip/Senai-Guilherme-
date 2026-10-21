# Gravando as atualizacoes completas do painel interativo diretamente no arquivo app.py
import streamlit as st
# Importando o framework do Streamlit para controle total da nossa interface web
import pandas as pd
# Importando pandas para carregar e realizar os tratamentos necessarios nos dados
import os
# Importando a biblioteca os para verificacao de arquivos de dados no sistema
import numpy as np
# Importando numpy para operacoes de selecao condicional e logica de faixas
import plotly.express as px
# Importando o modulo Express do Plotly para construcao dos graficos de dispersao e barras
import plotly.graph_objects as go
# Importando os objetos graficos do Plotly para criar o indicador de probabilidade
from sklearn.model_selection import train_test_split
# Importando o particionador de dados para replicar a divisao de treino e teste
from sklearn.preprocessing import StandardScaler
# Importando o padronizador para tratar a escala de valores das caracteristicas
from sklearn.linear_model import LogisticRegression
# Importando a regressao logistica para calculo das probabilidades de risco

# Definindo a configuracao de visualizacao em tela cheia do Streamlit
st.set_page_config(page_title="Previsão de Inadimplência", layout="wide")
# Adicionando o titulo principal no topo da pagina do painel
st.title("Análise de Inadimplência de Clientes")

# Definindo o local fixo onde se encontra nossa planilha excel de clientes
caminho_dados = "/content/inadimplencia-clientes.xlsx"

# Interrompendo o processo com erro caso a planilha necessaria nao exista
if not os.path.exists(caminho_dados):
    # Apresentando alerta visual vermelho para notificacao de arquivo ausente
    st.error("Erro: A planilha 'inadimplencia-clientes.xlsx' não foi encontrada. O app não pode continuar.")
    # Parando a execucao de forma segura
    st.stop()

# Configurando cache para evitar treinar o modelo de machine learning a cada interacao do usuario
@st.cache_resource
def inicializar_e_treinar():
    # Lendo a planilha excel original para obter os dados historicos
    dados_brutos = pd.read_excel(caminho_dados)
    # Removendo duplicatas e linhas incompletas mantendo a integridade
    dados_limpos = dados_brutos.drop_duplicates().dropna()
    # Guardando uma copia dos dados mapeados textualmente para utilizacao nos graficos
    dados_plot = dados_limpos.copy()
    # Mapeando os niveis de escolaridade para variaveis numericas ordinais de 0 a 3
    dados_limpos['Escolaridade'] = dados_limpos['Escolaridade'].map({'Fundamental': 0, 'Médio': 1, 'Superior': 2, 'Pós-graduação': 3})
    # Mapeando os tipos de vinculo empregaticio para numeros equivalentes
    dados_limpos['Vínculo Empregatício'] = dados_limpos['Vínculo Empregatício'].map({'Informal': 0, 'Autônomo': 1, 'CLT': 2, 'Servidor Público': 3})
    # Mapeando a posse de imovel proprio para valores binarios de 0 ou 1
    dados_limpos['Imóvel Próprio'] = dados_limpos['Imóvel Próprio'].map({'Não': 0, 'Sim': 1})
    # Mapeando a situacao real de adimplencia para valores de 0 ou 1
    dados_limpos['Situação'] = dados_limpos['Situação'].map({'Adimplente': 0, 'Inadimplente': 1})
    # Separando a matriz de entrada removendo os identificadores
    X = dados_limpos.drop(columns=['ID do Cliente', 'Situação'])
    # Separando a variavel resposta correspondente a inadimplencia
    y = dados_limpos['Situação']
    # Dividindo os dados para replicar as dimensoes de treino anteriores
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # Instanciando o padronizador de dados com base nas caracteristicas
    scaler_app = StandardScaler()
    # Ajustando o padronizador apenas com os dados de treino para evitar vazamento
    X_train_scaled = scaler_app.fit_transform(X_train)
    # Inicializando o classificador com numero de iteracoes configurado
    modelo_app = LogisticRegression(max_iter=1000, random_state=42)
    # Treinando o classificador utilizando a matriz padronizada obtida
    modelo_app.fit(X_train_scaled, y_train)
    # Retornando os objetos necessários e os dados de plot originais para os graficos
    return modelo_app, scaler_app, X, dados_plot

# Executando a carga cacheada dos modelos e dos dados para exibicao
modelo_treinado, padronizador, dados_limites, df_plot = inicializar_e_treinar()

# Dividindo a tela principal em duas colunas equivalentes para controle e resultados
col_esquerda, col_direita = st.columns(2)

# Iniciando a criacao de todos os elementos interativos na coluna esquerda
with col_esquerda:
    # Cabecalho da area de informacoes do novo cliente
    st.subheader("Dados de Entrada do Cliente")
    # Selecao numerica para definir a idade do cliente simulado
    idade = st.number_input("Idade", min_value=18, max_value=120, value=30, step=1)
    # Selecao textual da escolaridade do cliente simulado
    escolaridade_texto = st.selectbox("Escolaridade", ['Fundamental', 'Médio', 'Superior', 'Pós-graduação'])
    # Selecao textual do vinculo profissional do cliente simulado
    vinculo_texto = st.selectbox("Vínculo Empregatício", ['Informal', 'Autônomo', 'CLT', 'Servidor Público'])
    # Entrada de valor monetario para definir a renda mensal do cliente simulado
    renda = st.number_input("Renda Mensal (R$)", min_value=0.0, max_value=100000.0, value=5000.0, step=100.0)
    # Entrada de valor monetario do emprestimo solicitado para analise
    emprestimo = st.number_input("Valor do Empréstimo (R$)", min_value=0.0, max_value=500000.0, value=15000.0, step=500.0)
    # Entrada numerica do prazo de quitacao em quantidade de meses
    prazo = st.number_input("Prazo (meses)", min_value=1, max_value=360, value=36, step=1)
    # Entrada do valor individual de cada parcela mensal do plano de pagamento
    parcela = st.number_input("Valor da Parcela (R$)", min_value=0.0, max_value=50000.0, value=500.0, step=50.0)
    # Entrada do numero de inadimplencias ou atrasos ja cometidos pelo cliente no passado
    atrasos = st.number_input("Atrasos Anteriores", min_value=0, max_value=50, value=0, step=1)
    # Tempo total em anos do cliente como parceiro ativo da instituicao financeira
    tempo_relacao = st.number_input("Tempo de Relacionamento (anos)", min_value=0, max_value=50, value=2, step=1)
    # Entrada para indicar se o cliente e proprietario de imovel proprio
    imovel_texto = st.selectbox("Imóvel Próprio", ['Não', 'Sim'])
    
    # Adicionando uma divisao e cabecalho especifico para os filtros interativos de dados historicos
    st.markdown("--- ")
    # Titulo para a secao de filtros dos graficos de comparacao
    st.subheader("Filtros para o Histórico (Gráficos 2 e 3)")
    # Criando o filtro multiselecao de vinculos profissionais pre-marcado com todas as opcoes
    filtro_vinculo = st.multiselect("Filtrar por Vínculo Empregatício", options=list(df_plot['Vínculo Empregatício'].unique()), default=list(df_plot['Vínculo Empregatício'].unique()))
    # Criando o filtro multiselecao de escolaridade pre-marcado com todas as opcoes disponiveis
    filtro_escolaridade = st.multiselect("Filtrar por Escolaridade", options=list(df_plot['Escolaridade'].unique()), default=list(df_plot['Escolaridade'].unique()))

# Mapeando os inputs de texto selecionados pelo usuario para o formato numerico do classificador
escolaridade_num = {'Fundamental': 0, 'Médio': 1, 'Superior': 2, 'Pós-graduação': 3}[escolaridade_texto]
# Convertendo o vinculo empregaticio em codigo ordinal valido
vinculo_num = {'Informal': 0, 'Autônomo': 1, 'CLT': 2, 'Servidor Público': 3}[vinculo_texto]
# Convertendo a posse de imovel em binario compreensivel
imovel_num = {'Não': 0, 'Sim': 1}[imovel_texto]

# Criando um novo DataFrame para representar as caracteristicas do cliente simulado
novos_dados = pd.DataFrame([[idade, escolaridade_num, vinculo_num, renda, emprestimo, prazo, parcela, atrasos, tempo_relacao, imovel_num]],
                           columns=['Idade', 'Escolaridade', 'Vínculo Empregatício', 'Renda Mensal (R$)', 'Valor do Empréstimo (R$)', 'Prazo (meses)', 'Valor da Parcela (R$)', 'Atrasos Anteriores', 'Tempo de Relacionamento (anos)', 'Imóvel Próprio'])

# Aplicando o padronizador treinado sobre os dados coletados na interface
novos_dados_scaled = padronizador.transform(novos_dados)

# Obtendo a classificacao discreta final de inadimplencia
previsao_classe = modelo_treinado.predict(novos_dados_scaled)[0]
# Obtendo a probabilidade para as duas classes possiveis do modelo
probabilidades = modelo_treinado.predict_proba(novos_dados_scaled)[0]
# Extraindo especificamente a probabilidade de risco em termos percentuais
probabilidade_inadimplencia = probabilidades[1] * 100

# Iniciando a montagem e exibicao dos resultados na coluna direita
with col_direita:
    # Titulo do bloco de previsao em tempo real
    st.subheader("Resultado da Análise de Risco")
    # Exibindo mensagem em formato verde para previsoes seguras de adimplencia
    if previsao_classe == 0:
        # Notificando o sucesso e o baixo risco calculado
        st.success("Situação Prevista: **ADIMPLENTE** (Baixo Risco de Crédito)")
    else:
        # Notificando o alto risco calculado de inadimplencia em vermelho
        st.error("Situação Prevista: **INADIMPLENTE** (Alto Risco de Crédito)")
    
    # Mostrando a probabilidade calculada em formato destacado de metrica
    st.metric(label="Probabilidade de Inadimplência", value=f"{probabilidade_inadimplencia:.2f}%")
    
    # Criando um medidor circular para ilustrar graficamente a probabilidade de inadimplencia do cliente
    fig_medidor = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probabilidade_inadimplencia,
        title={'text': "Probabilidade de Inadimplência (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "salmon"}
            ]
        }
    ))
    # Ajustando tamanho e margens do indicador de risco para melhor exibicao
    fig_medidor.update_layout(height=220, margin=dict(l=20, r=20, t=50, b=20))
    # Exibindo o medidor de risco circular gerado diretamente na tela
    st.plotly_chart(fig_medidor, use_container_width=True)
    
    # Aplicando os filtros selecionados pelo usuario na planilha historica para visualizacao
    df_filtrado = df_plot[df_plot['Vínculo Empregatício'].isin(filtro_vinculo) & df_plot['Escolaridade'].isin(filtro_escolaridade)].copy()
    
    # Apresentando um aviso amigavel caso o filtro selecionado exclua todos os registros historicos
    if df_filtrado.empty:
        # Exibindo aviso amarelo instruindo a selecao de ao menos um item
        st.warning("Por favor, selecione pelo menos uma opção nos filtros para visualizar os gráficos.")
    else:
        # Criando o grafico de dispersao interativo relacionando renda por parcela
        fig_dispersao = px.scatter(
            df_filtrado, 
            x="Renda Mensal (R$)", 
            y="Valor da Parcela (R$)", 
            color="Situação", 
            color_discrete_map={"Adimplente": "blue", "Inadimplente": "red"},
            title="Dispersão de Renda Mensal vs Valor da Parcela",
            labels={"Situação": "Situação Real"}
        )
        # Adicionando a estrela amarela que destaca o cliente simulado interativamente no grafico
        fig_dispersao.add_trace(go.Scatter(
            x=[renda], 
            y=[parcela], 
            mode='markers', 
            marker=dict(color='yellow', size=18, symbol='star', line=dict(color='black', width=2)),
            name='Cliente Simulado'
        ))
        # Ajustando o layout e tamanho geral do grafico de dispersao
        fig_dispersao.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
        # Exibindo o grafico de dispersao integrado de forma responsiva na coluna direita
        st.plotly_chart(fig_dispersao, use_container_width=True)
        
        # Definindo as faixas de atrasos anteriores com base no historico de clientes
        bins_atrasos = [-1, 0, 1, 2, 5, 100]
        # Definindo as etiquetas textuais legiveis de cada faixa de atraso anterior
        labels_atrasos = ['Nenhum', '1 Atraso', '2 Atrasos', '3 a 5 Atrasos', 'Mais de 5']
        # Criando a nova coluna de faixa de atrasos no dataframe historico filtrado
        df_filtrado['Faixa de Atrasos'] = pd.cut(df_filtrado['Atrasos Anteriores'], bins=bins_atrasos, labels=labels_atrasos)
        
        # Agrupando os dados historicos por faixa para calcular a taxa real de inadimplencia
        taxa_inad = df_filtrado.groupby('Faixa de Atrasos', observed=False).apply(
            lambda x: (x['Situação'] == 'Inadimplente').sum() / len(x) * 100 if len(x) > 0 else 0
        ).reset_index(name='Taxa de Inadimplência (%)')
        
        # Determinando em qual faixa de atraso o cliente simulado atualmente se enquadra
        faixa_simulado = pd.cut([atrasos], bins=bins_atrasos, labels=labels_atrasos)[0]
        # Mapeando uma cor diferente (laranja) para a faixa em que o cliente simulado esta, e azul para as demais
        cores_barras = ['#FF4B4B' if faixa == faixa_simulado else '#1C83E1' for faixa in taxa_inad['Faixa de Atrasos']]
        
        # Criando o grafico de barras com plotly express para demonstrar as taxas reais de inadimplencia
        fig_barras = px.bar(
            taxa_inad, 
            x='Faixa de Atrasos', 
            y='Taxa de Inadimplência (%)', 
            title='Taxa de Inadimplência Real por Faixa de Atrasos',
            labels={'Taxa de Inadimplência (%)': 'Inadimplência (%)', 'Faixa de Atrasos': 'Atrasos Anteriores'}
        )
        # Aplicando as cores mapeadas dinamicamente nas barras para destacar o cliente simulado
        fig_barras.update_traces(marker_color=cores_barras)
        # Configurando a escala e limites do eixo vertical de percentual de 0 a 100
        fig_barras.update_yaxes(range=[0, 100])
        # Ajustando o layout e margens do grafico de barras para melhor encaixe
        fig_barras.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
        # Exibindo o novo grafico de barras de taxas na interface de forma responsiva
        st.plotly_chart(fig_barras, use_container_width=True)
