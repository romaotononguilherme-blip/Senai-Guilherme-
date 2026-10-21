
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


st.set_page_config(
    page_title="Previsão de Inadimplência",
    layout="wide"
)

st.title("Análise de Inadimplência de Clientes")

# Caminho da planilha
PASTA_APP = os.path.dirname(os.path.abspath(__file__))
CAMINHO_DADOS = os.path.join(
    PASTA_APP, "inadimplencia-clientes.xlsx"
)

COLUNAS_NUMERICAS = [
    "Idade",
    "Renda Mensal (R$)",
    "Valor do Empréstimo (R$)",
    "Prazo (meses)",
    "Valor da Parcela (R$)",
    "Atrasos Anteriores",
    "Tempo de Relacionamento (anos)"
]

COLUNAS_MODELO = [
    "Idade",
    "Escolaridade",
    "Vínculo Empregatício",
    "Renda Mensal (R$)",
    "Valor do Empréstimo (R$)",
    "Prazo (meses)",
    "Valor da Parcela (R$)",
    "Atrasos Anteriores",
    "Tempo de Relacionamento (anos)",
    "Imóvel Próprio"
]

MAPA_ESCOLARIDADE = {
    "Fundamental": 0,
    "Médio": 1,
    "Superior": 2,
    "Pós-graduação": 3
}

MAPA_VINCULO = {
    "Informal": 0,
    "Autônomo": 1,
    "CLT": 2,
    "Servidor Público": 3
}

MAPA_IMOVEL = {"Não": 0, "Sim": 1}
MAPA_SITUACAO = {"Adimplente": 0, "Inadimplente": 1}

if not os.path.isfile(CAMINHO_DADOS):
    st.error(
        "Planilha inadimplencia-clientes.xlsx não encontrada. "
        "Coloque-a na mesma pasta do app.py."
    )
    st.stop()

try:
    df_original = pd.read_excel(CAMINHO_DADOS)
except Exception as erro:
    st.error(f"Erro ao ler a planilha: {erro}")
    st.stop()

COLUNAS_OBRIGATORIAS = [
    "ID do Cliente"
] + COLUNAS_MODELO + ["Situação"]

faltantes = [
    c for c in COLUNAS_OBRIGATORIAS
    if c not in df_original.columns
]

if faltantes:
    st.error(f"Colunas ausentes na planilha: {faltantes}")
    st.write("Colunas encontradas:", list(df_original.columns))
    st.stop()


@st.cache_resource
def inicializar_e_treinar(caminho, modificacao):
    dados = pd.read_excel(caminho).drop_duplicates()
    dados_plot = dados.copy()

    categorias = [
        "Escolaridade",
        "Vínculo Empregatício",
        "Imóvel Próprio",
        "Situação"
    ]

    for coluna in categorias:
        dados[coluna] = (
            dados[coluna].astype("string").str.strip()
        )

    for coluna in COLUNAS_NUMERICAS:
        dados[coluna] = pd.to_numeric(
            dados[coluna], errors="coerce"
        )

    dados["Escolaridade"] = dados["Escolaridade"].map(
        MAPA_ESCOLARIDADE
    )
    dados["Vínculo Empregatício"] = (
        dados["Vínculo Empregatício"].map(MAPA_VINCULO)
    )
    dados["Imóvel Próprio"] = dados["Imóvel Próprio"].map(
        MAPA_IMOVEL
    )
    dados["Situação"] = dados["Situação"].map(
        MAPA_SITUACAO
    )

    dados = dados.dropna(
        subset=COLUNAS_MODELO + ["Situação"]
    ).copy()

    if dados.empty:
        raise ValueError("Não há registros válidos para treinamento.")

    X = dados[COLUNAS_MODELO].astype(float)
    y = dados["Situação"].astype(int)

    if y.nunique() != 2:
        raise ValueError(
            "A coluna Situação precisa conter Adimplente "
            "e Inadimplente."
        )

    if len(dados) < 5 or y.value_counts().min() < 2:
        raise ValueError(
            "Existem poucos exemplos de uma das classes. "
            "Adicione mais registros à planilha."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    modelo = LogisticRegression(
        max_iter=1000,
        random_state=42
    )
    modelo.fit(X_train_scaled, y_train)

    return modelo, scaler, X, dados_plot


try:
    modelo, scaler, dados_limites, df_plot = (
        inicializar_e_treinar(
            CAMINHO_DADOS,
            os.path.getmtime(CAMINHO_DADOS)
        )
    )
except Exception as erro:
    st.error(f"Erro ao treinar o modelo: {erro}")
    st.stop()


# Interface
col_esquerda, col_direita = st.columns(2)

with col_esquerda:
    st.subheader("Dados de Entrada do Cliente")

    idade = st.number_input(
        "Idade", 18, 120, 30
    )

    escolaridade = st.selectbox(
        "Escolaridade", list(MAPA_ESCOLARIDADE.keys())
    )

    vinculo = st.selectbox(
        "Vínculo Empregatício", list(MAPA_VINCULO.keys())
    )

    renda = st.number_input(
        "Renda Mensal (R$)", 0.0, 100000.0, 5000.0,
        step=100.0
    )

    emprestimo = st.number_input(
        "Valor do Empréstimo (R$)", 0.0, 500000.0,
        15000.0, step=500.0
    )

    prazo = st.number_input(
        "Prazo (meses)", 1, 360, 36
    )

    parcela = st.number_input(
        "Valor da Parcela (R$)", 0.0, 50000.0,
        500.0, step=50.0
    )

    atrasos = st.number_input(
        "Atrasos Anteriores", 0, 50, 0
    )

    tempo_relacao = st.number_input(
        "Tempo de Relacionamento (anos)", 0, 50, 2
    )

    imovel = st.selectbox(
        "Imóvel Próprio", list(MAPA_IMOVEL.keys())
    )

    st.markdown("---")
    st.subheader("Filtros para o Histórico")

    opcoes_vinculo = sorted(
        df_plot["Vínculo Empregatício"]
        .dropna().astype(str).unique().tolist()
    )

    opcoes_escolaridade = sorted(
        df_plot["Escolaridade"]
        .dropna().astype(str).unique().tolist()
    )

    filtro_vinculo = st.multiselect(
        "Filtrar por Vínculo Empregatício",
        opcoes_vinculo,
        default=opcoes_vinculo
    )

    filtro_escolaridade = st.multiselect(
        "Filtrar por Escolaridade",
        opcoes_escolaridade,
        default=opcoes_escolaridade
    )


# Previsão
novos_dados = pd.DataFrame(
    [[
        idade,
        MAPA_ESCOLARIDADE[escolaridade],
        MAPA_VINCULO[vinculo],
        renda,
        emprestimo,
        prazo,
        parcela,
        atrasos,
        tempo_relacao,
        MAPA_IMOVEL[imovel]
    ]],
    columns=COLUNAS_MODELO
)

novos_dados = novos_dados[dados_limites.columns]
novos_scaled = scaler.transform(novos_dados)

classe = int(modelo.predict(novos_scaled)[0])
probabilidades = modelo.predict_proba(novos_scaled)[0]
indice = list(modelo.classes_).index(1)
probabilidade = float(probabilidades[indice] * 100)


# Resultados
with col_direita:
    st.subheader("Resultado da Análise de Risco")

    if classe == 0:
        st.success(
            "Situação Prevista: ADIMPLENTE "
            "(menor risco estimado pelo modelo)"
        )
    else:
        st.error(
            "Situação Prevista: INADIMPLENTE "
            "(maior risco estimado pelo modelo)"
        )

    st.metric(
        "Probabilidade Estimada de Inadimplência",
        f"{probabilidade:.2f}%"
    )

    medidor = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probabilidade,
            number={"suffix": "%"},
            title={"text": "Probabilidade de Inadimplência"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, 30], "color": "lightgreen"},
                    {"range": [30, 70], "color": "yellow"},
                    {"range": [70, 100], "color": "salmon"}
                ]
            }
        )
    )

    medidor.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    st.plotly_chart(medidor, use_container_width=True)

    st.caption(
        "A previsão é uma estimativa estatística e não garante "
        "o comportamento futuro do cliente."
    )

    df_filtrado = df_plot[
        df_plot["Vínculo Empregatício"].astype("string").isin(
            filtro_vinculo
        )
        &
        df_plot["Escolaridade"].astype("string").isin(
            filtro_escolaridade
        )
    ].copy()

    if df_filtrado.empty:
        st.warning(
            "Nenhum registro corresponde aos filtros selecionados."
        )
    else:
        # Gráfico de dispersão
        fig_dispersao = px.scatter(
            df_filtrado,
            x="Renda Mensal (R$)",
            y="Valor da Parcela (R$)",
            color="Situação",
            color_discrete_map={
                "Adimplente": "blue",
                "Inadimplente": "red"
            },
            title="Renda Mensal vs. Valor da Parcela",
            labels={"Situação": "Situação Real"}
        )

        fig_dispersao.add_trace(
            go.Scatter(
                x=[renda],
                y=[parcela],
                mode="markers",
                marker=dict(
                    color="yellow",
                    size=18,
                    symbol="star",
                    line=dict(color="black", width=2)
                ),
                name="Cliente Simulado"
            )
        )

        fig_dispersao.update_layout(height=350)
        st.plotly_chart(
            fig_dispersao,
            use_container_width=True
        )

        # Gráfico da taxa de inadimplência por faixa de atrasos
        df_filtrado["Atrasos Anteriores"] = pd.to_numeric(
            df_filtrado["Atrasos Anteriores"],
            errors="coerce"
        )

        df_filtrado = df_filtrado.dropna(
            subset=["Atrasos Anteriores", "Situação"]
        ).copy()

        bins = [-1, 0, 1, 2, 5, float("inf")]
        labels = [
            "Nenhum",
            "1 Atraso",
            "2 Atrasos",
            "3 a 5 Atrasos",
            "Mais de 5"
        ]

        df_filtrado["Faixa de Atrasos"] = pd.cut(
            df_filtrado["Atrasos Anteriores"],
            bins=bins,
            labels=labels
        )

        taxa = (
            df_filtrado.groupby(
                "Faixa de Atrasos", observed=False
            )
            .agg(
                total=("Situação", "size"),
                inadimplentes=(
                    "Situação",
                    lambda s: (
                        s.astype(str).str.strip() == "Inadimplente"
                    ).sum()
                )
            )
            .reset_index()
        )

        taxa["Taxa de Inadimplência (%)"] = np.where(
            taxa["total"] > 0,
            taxa["inadimplentes"] / taxa["total"] * 100,
            0
        )

        faixa_cliente = pd.cut(
            [atrasos], bins=bins, labels=labels
        )[0]

        cores = [
            "#FF4B4B" if faixa == faixa_cliente else "#1C83E1"
            for faixa in taxa["Faixa de Atrasos"]
        ]

        fig_barras = px.bar(
            taxa,
            x="Faixa de Atrasos",
            y="Taxa de Inadimplência (%)",
            title="Inadimplência Real por Faixa de Atrasos",
            category_orders={"Faixa de Atrasos": labels},
            custom_data=["total"]
        )

        fig_barras.update_traces(
            marker_color=cores,
            hovertemplate=(
                "Faixa: %{x}<br>"
                "Inadimplência: %{y:.2f}%<br>"
                "Registros: %{customdata[0]}<extra></extra>"
            )
        )

        fig_barras.update_yaxes(range=[0, 100])
        fig_barras.update_layout(height=350)

        st.plotly_chart(
            fig_barras,
            use_container_width=True
        )
