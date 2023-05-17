# test_used_tl_ML

Para este análisis se procedió de la siguiente manera:

1. Descarga de los datos suministrados según lo sugerido y la extracción de algunos campos que se encontraban dentro de filas en el archivo <extract.py> donde se crearos las clases y los métodos para esa tarea, le prueba de este script se realizó en el archivo <test_methods>, vale la pena mencionar que se descargaron los datos en un solo archivo llamado train debido a que se realizó un análisis exploratorio para todo el conjunto de datos posteriormente se crearon los grupos de entrenamiento y test

2. Con los datos resultantes se procedió a realizar el análisis exploratorio que se encuentra en el notebook <EDA_and_clean_text.ipynb> donde están las notas especificas para cada iteración de la limpieza de variables, este paso consistió se desarrollo en tres etapas para poder conocer el comportamiento de los datos y las variables realmente importantes para el análisis:
* En la primera etapa <output_initial.html> se analizaron todas las variables en conjunto con la ayuda de la librería ydata_profiling de la cual se trataron primero las variables con la alarma de *Unsupported*, donde se requería mayor limpieza o definitivamente la cardinalidad era muy alta, en algunos otros casos
* En la segunda etapa se realizó la limpieza de las variables mencionadas anteriormente y se encontraron varios valores fuera de rango “outliers” en la variable price los cuales se corrigieron, el resumen de esta etapa donde se muestra mas a detalle cada paso esta en el notebook <EDA_and_clean_text.ipynb>
* En la tercera etapa se construyo una matriz de correlación con las variables normalizadas y se procedió a limpiar el texto de la variable “title”, en esta etapa se separaron los datos en variables numéricas y la variable, para realizar análisis separados en un principio

3. Para esta etapa se hace un análisis de componentes principales de cada dataset el que tiene variables numéricas y el que tiene el campo “title” los resultados de este análisis están en el notebook <PCA.ipynb>

4. En paralelo y con la teória de que el texto de la variable “title” era suficiente para resolver el problema, por esta razón se construyo un clasificador con una LSTM con la librería pytorch, pero la construcción de este clasificador tardo mas de los debido y los resultados preliminares mostraron gran cantidad de over fitting, además el tiempo empleado para el entrenamiento y la construcción fue muy alto, así mismo se planteó hace un fine tuning de una modelo entrenado con hugginface  en español para clasificación basado en Roberta <https://arxiv.org/abs/1907.11692> congelando la última capa y reentrenando con los datos, este modelo no se puedo entrenar en local, razón por la cual se debió hacer en la plataforma Google Collab

5. Para el texto y las variables numéricas se entrenaron cinco modelos diferentes para determinar el mejor desempeño y sobre ese modelo con alto desempeño hacer una optimización de parámetros:
   * **Variables de numericas:**
     * Modelo Gaussiano: 0.57 de accuracy y 0.67 de f1_score
     * Modelos Logístico: 0.72 de accuracy y 0.59 de f1_score
     * KNN: 0.76 de accuracy y 0.75 de f1_score
     * SVM kernel lineal: 0.72 de accuracy y 0.59 de f1_score
     * SVM kernel sigmoide: 0.50 de accuracy y 0.44 de f1_score

   * **Variables de numericas:**
     * Modelo Gaussiano: 0.57 de accuracy y 0.67 de f1_score
     * Modelos Logístico: 0.72 de accuracy y 0.59 de f1_score
     * KNN: 0.76 de accuracy y 0.75 de f1_score
     * SVM kernel lineal: 0.72 de accuracy y 0.59 de f1_score
     * SVM kernel sigmoide: 0.50 de accuracy y 0.44 de f1_sco

7. Finalmente, para el modelo Roberta que se intentó hacer el fine tuning  por el tamaño del modelo 12 Gigas de memoria de GPU solo se pudieron hacer 2 epoch pero el accuracy solo llegó al 0.694, el notebook se llama <fine_tuning_huggingface.ipynb>