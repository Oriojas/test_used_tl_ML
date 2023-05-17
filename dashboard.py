import pandas as pd
import streamlit as st
import plotly.express as px

st.title("Resultados análisis")

df_num = pd.read_pickle("data/df_all.pkl")

fig = px.histogram(df_num,
                   x="seller_address_state_name")

fig.update_xaxes(tickangle=30)
st.plotly_chart(fig, use_container_width=True)


df_corr_matrix = pd.read_pickle("data/df_corr_matrix.pkl")
for i in df_corr_matrix.columns:
    df_corr_matrix[i] = round(df_corr_matrix[i], 3)


fig = px.imshow(df_corr_matrix,
                text_auto=True,
                aspect="auto")

fig.update_xaxes(tickangle=30)
st.plotly_chart(fig, use_container_width=True)

st.dataframe(df_num.head(20).style.highlight_max(axis=0))
