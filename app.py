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

random_state=9999

st.set_page_config(page_title="Projeto Final - Machine Learning", layout="wide")
st.title("Sistema de Análise de Vendas")

uploaded = st.file_uploader("Envie o dataset (.xlsx)", type=["xlsx"])

if uploaded:
    dataset = pd.read_excel(uploaded, sheet_name=None)
    st.success("Arquivo carregado com sucesso!")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Análise Exploratória",
        "🛑 Detecção de Fraudes",
        "💰 Previsão de Receita",
        "👥 Perfilamento de Clientes"
    ])

    with tab1:
        vendas = dataset['vendas'].copy()
        vendas['fraude_suspeita'] = vendas['fraude_suspeita'].map({0: 'Não', 1: 'Sim'})
        st.subheader("Informações do Dataset de Vendas")
        st.write(f"**Total de dados:** {len(vendas)}")
        st.write(f"**Linhas com faltas:** {len(vendas[vendas.isnull().any(axis=1)])}")
        st.write("**Colunas:**", vendas.columns.tolist())

        vendas_dados = vendas.drop(columns=['venda_id', 'cliente_id', 'produto_id']).dropna()

        vendas_num = vendas_dados.select_dtypes(include=['int64', 'float64'])
        corr_mat = vendas_num.corr()
        with st.expander("Matriz de Correlação"):
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr_mat, annot=True, cmap='viridis', fmt=".2f", linewidths=.5, ax=ax)
            st.pyplot(fig, width='content')

        with st.expander("Distribuições Numéricas"):
            for col in vendas_num.columns:
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.histplot(vendas_num[col], kde=True, ax=ax)
                ax.set_title(col)
                st.pyplot(fig, width='content')

        with st.expander("Distribuições Categóricas"):
            vendas_cat = vendas_dados.select_dtypes(include=['object'])
            for col in vendas_cat.columns:
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.countplot(x=vendas_cat[col], ax=ax)
                ax.set_title(col)
                if vendas_cat[col].nunique() > 10:
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
                st.pyplot(fig, width='content')
                
        clientes = dataset['clientes'].copy()

        st.subheader("Informações do Dataset de Clientes")
        st.write(f"**Total de dados:** {len(clientes)}")
        st.write(f"**Linhas com valores faltando:** {len(clientes[clientes.isnull().any(axis=1)])}")
        st.write("**Colunas:**", clientes.columns.tolist())

        clientes_dados = clientes.drop(columns=['cliente_id']).dropna()

        clientes_num = clientes_dados.select_dtypes(include=['int64', 'float64'])
        corr_mat = clientes_num.corr()

        with st.expander("Matriz de Correlação"):
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr_mat, annot=True, cmap='viridis', fmt=".2f", linewidths=.5, ax=ax)
            ax.set_title('Matriz de Correlação')
            st.pyplot(fig, width='content')

        # Distribuições Numéricas
        with st.expander("Distribuições Numéricas"):
            for col in clientes_num.columns:
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.histplot(clientes_num[col], kde=True, ax=ax)
                ax.set_title(col)
                st.pyplot(fig, width='content')

        # Distribuições Categóricas
        with st.expander("Distribuições Categóricas"):
            clientes_cat = clientes_dados.select_dtypes(include=['object'])
            for col in clientes_cat.columns:
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.countplot(x=clientes_cat[col], ax=ax)
                ax.set_title(col)
                if clientes_cat[col].nunique() > 10:
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
                st.pyplot(fig, width='content')

    with tab2:
        st.subheader("Modelo de Detecção de Fraude")
        vendas = dataset['vendas'].copy()
        vendas_dados = vendas.drop(columns=['venda_id', 'cliente_id', 'produto_id', 'faixa_renda', 'data']).dropna()
        vendas_dados[vendas_dados.select_dtypes(include='object').columns] = vendas_dados.select_dtypes(include='object').astype("category")

        vendas_labels = vendas_dados['fraude_suspeita']
        vendas_features = vendas_dados.drop(columns=['fraude_suspeita', 'margem','receita','satisfacao','devolucao'])

        v_feat_treino, v_feat_test, v_labels_treino, v_labels_teste = train_test_split(
            vendas_features, vendas_labels, test_size=0.4, stratify=vendas_labels, random_state=random_state
        )

        sample_weight = compute_sample_weight("balanced", y=v_labels_treino)

        classifier = xgb.XGBClassifier(enable_categorical=True, random_state=random_state)
        classifier.fit(v_feat_treino, v_labels_treino, sample_weight=sample_weight)

        pred = classifier.predict(v_feat_test)
        pred_proba = classifier.predict_proba(v_feat_test)[:,1]

        st.write("**Relatório de Classificação**")
        st.code(classification_report(v_labels_teste, pred, target_names=['Normal','Fraude']))

        with st.expander(" Matriz de Confusão"):
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(confusion_matrix(v_labels_teste, pred), annot=True, cmap="Blues", fmt='d')
            st.pyplot(fig, width='content')

        with st.expander("Curva ROC"):
            fig, ax = plt.subplots(figsize=(8, 6))
            RocCurveDisplay.from_predictions(v_labels_teste, pred_proba, name="XGBoost", ax=ax)
            st.pyplot(fig, width='content')

        st.write(f"AUC Score: **{roc_auc_score(v_labels_teste, pred_proba):.4f}**")

    with tab3:
        st.subheader("Modelo de Previsão de Receita")
        vendas = dataset['vendas'].copy()
        vendas_dados = vendas.drop(columns=['venda_id', 'cliente_id', 'produto_id', 'faixa_renda', 'data']).dropna()
        vendas_dados = vendas_dados.drop(columns=['margem','fraude_suspeita','satisfacao','devolucao'])
        vendas_dados = pd.get_dummies(vendas_dados, drop_first=True)

        vendas_valor = vendas_dados['receita']
        vendas_features = vendas_dados.drop(columns=['receita'])

        v_feat_treino, v_feat_test, v_valor_treino, v_valor_test = train_test_split(vendas_features, vendas_valor, test_size=0.4)

        regressor = RandomForestRegressor(min_samples_split=5, min_samples_leaf=20, random_state=random_state)
        regressor.fit(v_feat_treino, v_valor_treino)

        valor_pred = regressor.predict(v_feat_test)

        st.metric("RMSE", f"{root_mean_squared_error(v_valor_test, valor_pred):.2f}")
        st.metric("MAE", f"{mean_absolute_error(v_valor_test, valor_pred):.2f}")
        st.metric("R² Score", f"{r2_score(v_valor_test, valor_pred):.2f}")

        with st.expander("📊 Comparação: Valores Reais x Preditos"):
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(v_valor_test, valor_pred, alpha=0.6)
            ax.plot([v_valor_test.min(), v_valor_test.max()],
                    [v_valor_test.min(), v_valor_test.max()], 'r--')
            ax.set_xlabel("Valores Reais")
            ax.set_ylabel("Valores Preditos")
            st.pyplot(fig, width='content')

    with tab4:
        st.subheader("Clusterização de Clientes")
        cliente_dados = dataset['clientes'].drop(columns=['cliente_id', 'faixa_renda', 'regiao']).dropna()

        scaler = StandardScaler()
        numeric_cols = cliente_dados.select_dtypes(include=['int64','float64']).columns
        cliente_dados[numeric_cols] = scaler.fit_transform(cliente_dados[numeric_cols])

        cat_cols = cliente_dados.select_dtypes(include=['object']).columns
        cat_cols_index = [cliente_dados.columns.get_loc(col) for col in cat_cols]

        model = KPrototypes(n_clusters=5, n_init=10, random_state=random_state)
        model.fit(cliente_dados, categorical=cat_cols_index)

        centroids = pd.DataFrame(model.cluster_centroids_, columns=cliente_dados.columns)
        num_cols = ['idade','renda_mensal','score_fidelidade']
        centroids.columns = list(num_cols)+list(cat_cols)
        centroids[numeric_cols] = scaler.inverse_transform(centroids[numeric_cols])

        with st.expander("📌 Centroides dos Clusters"):
            st.dataframe(centroids)

else:
    st.warning("Envie um arquivo para continuar.")