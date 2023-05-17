import pandas as pd
import streamlit as st
import plotly.express as px
import streamlit.components.v1 as components

st.set_page_config(page_title="Dashboard",
                   page_icon="🛒",
                   layout="wide")

st.title("Resultados análisis")
st.markdown("""En este dashboard se condensan los resultados de los análisis relizados, las concluciones se encuentran 
                al final de este tablero, para cualquier duda consultar [oscarriojas@gmail.com](oscarriojas@gmail.com),
                el código se encuentra en el siguiente repositorio: 
                https://github.com/Oriojas/test_used_tl_ML.git
            """)

df_num = pd.read_pickle("data/df_all.pkl")

with st.container():
    column_1, column_2 = st.columns([1, 1])
    with column_1:
        st.markdown("### Estados con mayor cantidad de pedidos")
        st.markdown("Aunque esta variable no se tomó para realizar las predicciones puede mostrar información útil")
        fig = px.histogram(df_num,
                           x="seller_address_state_name")

        fig.update_xaxes(tickangle=30)
        st.plotly_chart(fig, use_container_width=True)

    with column_2:
        st.markdown("### Ciudades con mayor cantidad de pedidos")
        st.markdown("Aunque esta variable no se tomó para realizar las predicciones puede mostrar información útil")
        fig = px.histogram(df_num,
                           x="seller_address_city_name")

        fig.update_xaxes(tickangle=30)
        st.plotly_chart(fig, use_container_width=True)

with st.container():
    st.markdown("### Resultados EDA")
    st.markdown("Para ver cada análisis desplegar la pestaña correspondiente: ")
    with st.expander("Análisis inicial"):
        with open("output_initial.html", "r") as f:
            html_data = f.read()
            components.html(html_data, height=1200, scrolling=True)
    with st.expander("Análisis limpieza"):
        with open("output_clean.html", "r") as f:
            html_data = f.read()
            components.html(html_data, height=1200, scrolling=True)
    with st.expander("Análisis final"):
        with open("output_final.html", "r") as f:
            html_data = f.read()
            components.html(html_data, height=1200, scrolling=True)

with st.container():
    st.markdown("### Matriz de correlación")
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

with st.container():
    column_1, column_2 = st.columns([1, 1])
    with column_1:
        st.markdown("#### Resultados variable **title**")
        st.markdown("Para este caso el mejor desempeño lo tuvo la regresión lógistica con un accuracy de 0.76 y un f1 de 0.75 muestra un desbalance en la clasificación y con precision score se ve este desbalanceo hacia los falsos positivos")
        st.dataframe(df_results_text.style.highlight_max(axis=0))
    with column_2:
        st.markdown("#### Resultados variables **numéricas**")
        st.markdown("Para este caso el mejor desempeño lo tuvo en KNN con un accuracy de 0.76 y un f1 de 0.75 muestra un desbalance en la clasificación y con precision score se ve este desbalanceo hacia los falsos positivos")
        st.dataframe(df_results_num.style.highlight_max(axis=0))

st.markdown("### Ejemplo de los datos sin normalizar utilizados para el análisis")
st.dataframe(df_num.head(20))

st.markdown("### Concluciones:")
st.markdown("""
* Para mejorar el desempeño se sugiere crear un modelo mixto con texto y numeros, por temas de tiempo no se puedo desarrollar, pero la idea es que el modelo de texto exprese en terminos de probabilidad la clasificación y este sea el valor de entrada a en módelo numerico

* En resumen, en este análisis de datos se realizaron diversas etapas y técnicas para explorar y analizar un conjunto de datos. Se extrajeron los campos relevantes y se llevó a cabo un análisis exploratorio detallado utilizando la biblioteca "ydata_profiling". Se realizó una limpieza de datos y se corrigieron los valores atípicos en la variable "price". Además, se aplicó un análisis de componentes principales (PCA) para comprender la estructura de las variables numéricas y el texto en el campo "title".

* Se intentó construir un clasificador utilizando una LSTM de PyTorch, pero se encontraron problemas de sobreajuste y limitaciones de tiempo y recursos de GPU. Se exploró la posibilidad de hacer un ajuste fino de un modelo preentrenado basado en Roberta de HuggingFace, aunque debido a las limitaciones de memoria de la GPU, este proceso se realizó en la plataforma Google Colab.

* Se entrenaron varios modelos utilizando diferentes algoritmos para las variables de texto y las variables numéricas. Se evaluaron métricas de rendimiento como la precisión (accuracy) y la puntuación F1 (f1_score). Los mejores resultados se obtuvieron con modelos logísticos y SVM con kernel lineal para las variables de texto, mientras que para las variables numéricas, el KNN mostró el mejor rendimiento.

* En general, se ha realizado un análisis exhaustivo utilizando diferentes técnicas y modelos para comprender y predecir los datos. Sin embargo, se recomienda realizar un análisis más detallado y considerar otras técnicas o enfoques para mejorar aún más los resultados obtenidos.
""")
