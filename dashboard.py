import pandas as pd
import streamlit as st
import plotly.express as px

st.title("Resultados análisis")

with st.container():
    st.markdown("## Zonas con mayor cantidad de pedidos")
    st.markdown("Aunque esta variable no se tomó para realizar las predicciones puede mostrar información útil")
    df_num = pd.read_pickle("data/df_all.pkl")
    fig = px.histogram(df_num,
                       x="seller_address_state_name")

    fig.update_xaxes(tickangle=30)
    st.plotly_chart(fig, use_container_width=True)

with st.container():
    st.markdown("## Matriz de correlación")
    st.markdown("Esta matriz ayudó a determinar las variables clave para el análisis")
    df_corr_matrix = pd.read_pickle("data/df_corr_matrix.pkl")
    for i in df_corr_matrix.columns:
        df_corr_matrix[i] = round(df_corr_matrix[i], 3)

    fig = px.imshow(df_corr_matrix,
                    text_auto=True,
                    aspect="auto")

    fig.update_xaxes(tickangle=30)
    st.plotly_chart(fig, use_container_width=True)

df_results_text = pd.read_csv("data/results_text_class.csv", index_col=0)
df_results_num = pd.read_csv("data/results_num_class.csv", index_col=0)

df_results_text = df_results_text.set_index("model")
df_results_num = df_results_num.set_index("model")

st.dataframe(df_results_text.style.highlight_max(axis=0))
st.dataframe(df_results_num.style.highlight_max(axis=0))

st.dataframe(df_num.head(20))
