import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import classification_report, confusion_matrix, RocCurveDisplay, r2_score, mean_absolute_error, root_mean_squared_error, roc_auc_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from kmodes.kprototypes import KPrototypes

random_state = 9999

st.set_page_config(page_title="Projeto Final - Machine Learning", layout="wide")

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background-color: #0f172a;
        color: #e2e8f0;
    }
    
    .stApp {
        background-color: #0f172a;
    }
    
    .header-container {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 1.2rem 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        border: 1px solid #334155;
    }
    
    .header-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f1f5f9;
        margin: 0;
    }
    
    .header-subtitle {
        font-size: 0.85rem;
        color: #94a3b8;
        margin: 0.3rem 0 0 0;
    }
    
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 0.9rem;
        border-radius: 8px;
        border: 1px solid #334155;
    }
    
    div[data-testid="metric-container"] label {
        font-size: 0.7rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #94a3b8 !important;
    }
    
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-size: 1.4rem !important;
        font-weight: 700 !important;
        color: #f1f5f9 !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: #1e293b;
        padding: 0.3rem;
        border-radius: 8px;
        border: 1px solid #334155;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 2.3rem;
        padding: 0 1.3rem;
        background-color: transparent;
        border-radius: 6px;
        font-weight: 500;
        font-size: 0.85rem;
        color: #94a3b8;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
    }
    
    .streamlit-expanderHeader {
        background-color: #1e293b;
        border-radius: 6px;
        border: 1px solid #334155;
        font-weight: 600;
        font-size: 0.85rem;
        padding: 0.7rem 1rem !important;
        color: #e2e8f0 !important;
    }
    
    .streamlit-expanderHeader:hover {
        border-color: #667eea;
        background-color: #334155;
    }
    
    details[open] .streamlit-expanderHeader {
        border-color: #667eea;
    }
    
    .streamlit-expanderContent {
        border: 1px solid #334155;
        border-top: none;
        border-bottom-left-radius: 6px;
        border-bottom-right-radius: 6px;
        padding: 0.8rem;
        background: #1e293b;
    }
    
    [data-testid="stFileUploader"] {
        background-color: #1e293b;
        border: 2px dashed #475569;
        border-radius: 8px;
        padding: 1.3rem;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #667eea;
    }
    
    .stCodeBlock {
        background: #0f172a !important;
        border: 1px solid #334155;
        border-radius: 6px;
        font-size: 0.8rem;
    }
    
    code {
        color: #e2e8f0 !important;
    }
    
    h1, h2, h3 {
        color: #f1f5f9 !important;
        font-weight: 700;
    }
    
    .dataframe {
        font-size: 0.8rem;
        background-color: #1e293b;
        color: #e2e8f0;
    }
    
    .stAlert {
        border-radius: 6px;
        background-color: #1e293b;
        color: #e2e8f0;
    }
</style>
""",
    unsafe_allow_html=True,
)

plt.style.use("dark_background")

st.markdown(
    """
<div class="header-container">
    <div class="header-title">Sistema de Análise de Vendas</div>
    <div class="header-subtitle">Análise Exploratória • Detecção de Fraudes • Previsão de Receita • Perfilamento</div>
</div>
""",
    unsafe_allow_html=True,
)

uploaded = st.file_uploader("Envie o dataset (.xlsx)", type=["xlsx"])

if uploaded:
    dataset = pd.read_excel(uploaded, sheet_name=None)
    st.success("Arquivo carregado com sucesso!")

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📈 Análise Exploratória",
            "🛡️ Detecção de Fraudes",
            "💰 Previsão de Receita",
            "👥 Perfilamento de Clientes",
        ]
    )

    # ------------------------------------------------------------------
    # TAB 1 – EDA
    # ------------------------------------------------------------------
    with tab1:
        vendas = dataset["vendas"].copy()
        vendas["fraude_suspeita"] = vendas["fraude_suspeita"].map({0: "Não", 1: "Sim"})
        st.subheader("Informações do Dataset de Vendas")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total de dados", f"{len(vendas):,}")
        col2.metric("Linhas com faltas", f"{len(vendas[vendas.isnull().any(axis=1)]):,}")
        col3.metric("Colunas", len(vendas.columns))

        with st.expander("Visualizar Colunas"):
            st.write(vendas.columns.tolist())

        vendas_dados = vendas.drop(columns=["venda_id", "cliente_id", "produto_id"]).dropna()
        vendas_num = vendas_dados.select_dtypes(include=["int64", "float64"])
        corr_mat = vendas_num.corr()

        # MATRIZ DE CORRELAÇÃO – VENDAS (OCUPANDO MAIS ESPAÇO NA TELA)
        with st.expander("Matriz de Correlação"):
            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_facecolor("#1e293b")
            ax.set_facecolor("#1e293b")
            sns.heatmap(
                corr_mat,
                annot=True,
                cmap="viridis",
                fmt=".2f",
                linewidths=0.5,
                ax=ax,
                annot_kws={"size": 8},
            )
            ax.set_title("Matriz de Correlação - Vendas", fontsize=11, pad=10)
            plt.xticks(rotation=45, ha="right", fontsize=8)
            plt.yticks(fontsize=8)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

        with st.expander("Distribuições Numéricas"):
            cols_list = vendas_num.columns.tolist()
            for i in range(0, len(cols_list), 4):
                cols = st.columns(4)
                for j, col in enumerate(cols_list[i : i + 4]):
                    with cols[j]:
                        fig, ax = plt.subplots(figsize=(3, 2.5))
                        fig.patch.set_facecolor("#1e293b")
                        ax.set_facecolor("#1e293b")
                        sns.histplot(vendas_num[col], kde=True, ax=ax)
                        ax.set_title(col, fontsize=9)
                        ax.set_xlabel("")
                        ax.set_ylabel("")
                        plt.tight_layout()
                        st.pyplot(fig)

        with st.expander("Distribuições Categóricas"):
            vendas_cat = vendas_dados.select_dtypes(include=["object"])
            cat_cols_list = vendas_cat.columns.tolist()
            for i in range(0, len(cat_cols_list), 3):
                cols = st.columns(3)
                for j, col in enumerate(cat_cols_list[i : i + 3]):
                    with cols[j]:
                        fig, ax = plt.subplots(figsize=(4, 2.8))
                        fig.patch.set_facecolor("#1e293b")
                        ax.set_facecolor("#1e293b")
                        sns.countplot(x=vendas_cat[col], ax=ax)
                        ax.set_title(col, fontsize=8.5)
                        ax.set_xlabel("")
                        ax.set_ylabel("")
                        ax.tick_params(labelsize=7)
                        if vendas_cat[col].nunique() > 10:
                            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
                        elif vendas_cat[col].nunique() > 5:
                            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
                        plt.tight_layout()
                        st.pyplot(fig)

        clientes = dataset["clientes"].copy()

        st.subheader("Informações do Dataset de Clientes")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total de dados", f"{len(clientes):,}")
        col2.metric(
            "Linhas com valores faltando",
            f"{len(clientes[clientes.isnull().any(axis=1)]):,}",
        )
        col3.metric("Colunas", len(clientes.columns))

        with st.expander("Visualizar Colunas"):
            st.write(clientes.columns.tolist())

        clientes_dados = clientes.drop(columns=["cliente_id"]).dropna()
        clientes_num = clientes_dados.select_dtypes(include=["int64", "float64"])
        corr_mat_clientes = clientes_num.corr()

        # MATRIZ DE CORRELAÇÃO – CLIENTES (OCUPANDO MAIS ESPAÇO NA TELA)
        with st.expander("Matriz de Correlação"):
            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_facecolor("#1e293b")
            ax.set_facecolor("#1e293b")
            sns.heatmap(
                corr_mat_clientes,
                annot=True,
                cmap="viridis",
                fmt=".2f",
                linewidths=0.5,
                ax=ax,
                annot_kws={"size": 8},
            )
            ax.set_title("Matriz de Correlação - Clientes", fontsize=11, pad=10)
            plt.xticks(rotation=45, ha="right", fontsize=8)
            plt.yticks(fontsize=8)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

        with st.expander("Distribuições Numéricas"):
            cols_list = clientes_num.columns.tolist()
            for i in range(0, len(cols_list), 4):
                cols = st.columns(4)
                for j, col in enumerate(cols_list[i : i + 4]):
                    with cols[j]:
                        fig, ax = plt.subplots(figsize=(3, 2.5))
                        fig.patch.set_facecolor("#1e293b")
                        ax.set_facecolor("#1e293b")
                        sns.histplot(clientes_num[col], kde=True, ax=ax)
                        ax.set_title(col, fontsize=9)
                        ax.set_xlabel("")
                        ax.set_ylabel("")
                        plt.tight_layout()
                        st.pyplot(fig)

        with st.expander("Distribuições Categóricas"):
            clientes_cat = clientes_dados.select_dtypes(include=["object"])
            cat_cols_list = clientes_cat.columns.tolist()
            for i in range(0, len(cat_cols_list), 3):
                cols = st.columns(3)
                for j, col in enumerate(cat_cols_list[i : i + 3]):
                    with cols[j]:
                        fig, ax = plt.subplots(figsize=(4, 2.8))
                        fig.patch.set_facecolor("#1e293b")
                        ax.set_facecolor("#1e293b")
                        sns.countplot(x=clientes_cat[col], ax=ax)
                        ax.set_title(col, fontsize=8.5)
                        ax.set_xlabel("")
                        ax.set_ylabel("")
                        ax.tick_params(labelsize=7)
                        if clientes_cat[col].nunique() > 10:
                            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
                        elif clientes_cat[col].nunique() > 5:
                            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
                        plt.tight_layout()
                        st.pyplot(fig)

    # ------------------------------------------------------------------
    # TAB 2 – FRAUDE
    # ------------------------------------------------------------------
    with tab2:
        st.subheader("Modelo de Detecção de Fraude")
        vendas = dataset["vendas"].copy()
        vendas_dados = vendas.drop(
            columns=["venda_id", "cliente_id", "produto_id", "faixa_renda", "data"]
        ).dropna()
        vendas_dados[
            vendas_dados.select_dtypes(include="object").columns
        ] = vendas_dados.select_dtypes(include="object").astype("category")

        vendas_labels = vendas_dados["fraude_suspeita"]
        vendas_features = vendas_dados.drop(
            columns=["fraude_suspeita", "margem", "receita", "satisfacao", "devolucao"]
        )

        v_feat_treino, v_feat_test, v_labels_treino, v_labels_teste = train_test_split(
            vendas_features,
            vendas_labels,
            test_size=0.4,
            stratify=vendas_labels,
            random_state=random_state,
        )

        sample_weight = compute_sample_weight("balanced", y=v_labels_treino)

        classifier = xgb.XGBClassifier(enable_categorical=True, random_state=random_state)
        classifier.fit(v_feat_treino, v_labels_treino, sample_weight=sample_weight)

        pred = classifier.predict(v_feat_test)
        pred_proba = classifier.predict_proba(v_feat_test)[:, 1]

        st.write("**Relatório de Classificação**")
        st.code(classification_report(v_labels_teste, pred, target_names=["Normal", "Fraude"]))

        with st.expander("Matriz de Confusão"):
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                fig, ax = plt.subplots(figsize=(5, 4))
                fig.patch.set_facecolor("#1e293b")
                ax.set_facecolor("#1e293b")
                sns.heatmap(
                    confusion_matrix(v_labels_teste, pred),
                    annot=True,
                    cmap="Blues",
                    fmt="d",
                    annot_kws={"size": 10},
                    ax=ax,
                )
                ax.set_title("Matriz de Confusão", fontsize=10, pad=10)
                plt.tight_layout()
                st.pyplot(fig)

        with st.expander("Curva ROC"):
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                fig, ax = plt.subplots(figsize=(5, 4))
                fig.patch.set_facecolor("#1e293b")
                ax.set_facecolor("#1e293b")
                RocCurveDisplay.from_predictions(
                    v_labels_teste, pred_proba, name="XGBoost", ax=ax
                )
                ax.set_title("Curva ROC", fontsize=10, pad=10)
                ax.grid(alpha=0.2)
                plt.tight_layout()
                st.pyplot(fig)

        st.write(f"AUC Score: **{roc_auc_score(v_labels_teste, pred_proba):.4f}**")

    # ------------------------------------------------------------------
    # TAB 3 – REGRESSÃO
    # ------------------------------------------------------------------
    with tab3:
        st.subheader("Modelo de Previsão de Receita")
        vendas = dataset["vendas"].copy()
        vendas_dados = vendas.drop(
            columns=["venda_id", "cliente_id", "produto_id", "faixa_renda", "data"]
        ).dropna()
        vendas_dados = vendas_dados.drop(
            columns=["margem", "fraude_suspeita", "satisfacao", "devolucao"]
        )
        vendas_dados = pd.get_dummies(vendas_dados, drop_first=True)

        vendas_valor = vendas_dados["receita"]
        vendas_features = vendas_dados.drop(columns=["receita"])

        v_feat_treino, v_feat_test, v_valor_treino, v_valor_test = train_test_split(
            vendas_features, vendas_valor, test_size=0.4
        )

        regressor = RandomForestRegressor(
            min_samples_split=5, min_samples_leaf=20, random_state=random_state
        )
        regressor.fit(v_feat_treino, v_valor_treino)

        valor_pred = regressor.predict(v_feat_test)

        col1, col2, col3 = st.columns(3)
        col1.metric("RMSE", f"{root_mean_squared_error(v_valor_test, valor_pred):.2f}")
        col2.metric("MAE", f"{mean_absolute_error(v_valor_test, valor_pred):.2f}")
        col3.metric("R² Score", f"{r2_score(v_valor_test, valor_pred):.2f}")

        with st.expander("📊 Comparação: Valores Reais x Preditos"):
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                fig, ax = plt.subplots(figsize=(5, 4))
                fig.patch.set_facecolor("#1e293b")
                ax.set_facecolor("#1e293b")
                ax.scatter(v_valor_test, valor_pred, alpha=0.6, s=20)
                ax.plot(
                    [v_valor_test.min(), v_valor_test.max()],
                    [v_valor_test.min(), v_valor_test.max()],
                    "r--",
                )
                ax.set_xlabel("Valores Reais", fontsize=9)
                ax.set_ylabel("Valores Preditos", fontsize=9)
                ax.set_title("Comparação", fontsize=10, pad=10)
                ax.grid(alpha=0.2)
                plt.tight_layout()
                st.pyplot(fig)

    # ------------------------------------------------------------------
    # TAB 4 – CLUSTERIZAÇÃO
    # ------------------------------------------------------------------
    with tab4:
        st.subheader("Clusterização de Clientes")
        cliente_dados = (
            dataset["clientes"].drop(columns=["cliente_id", "faixa_renda", "regiao"]).dropna()
        )

        scaler = StandardScaler()
        numeric_cols = cliente_dados.select_dtypes(include=["int64", "float64"]).columns
        cliente_dados[numeric_cols] = scaler.fit_transform(cliente_dados[numeric_cols])

        cat_cols = cliente_dados.select_dtypes(include=["object"]).columns
        cat_cols_index = [cliente_dados.columns.get_loc(col) for col in cat_cols]

        model = KPrototypes(n_clusters=5, n_init=10, random_state=random_state)
        model.fit(cliente_dados, categorical=cat_cols_index)

        centroids = pd.DataFrame(model.cluster_centroids_, columns=cliente_dados.columns)
        num_cols = ["idade", "renda_mensal", "score_fidelidade"]
        centroids.columns = list(num_cols) + list(cat_cols)
        centroids[numeric_cols] = scaler.inverse_transform(centroids[numeric_cols])

        with st.expander("📌 Centroides dos Clusters"):
            st.dataframe(centroids)

else:
    st.warning("Envie um arquivo para continuar.")
