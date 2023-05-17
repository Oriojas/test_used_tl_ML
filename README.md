# test_used_tl_ML

Para este análisis se procedió de la siguiente manera:

1. Descarga de los datos suministrados según lo sugerido y la extracción de algunos campos que se encontraban dentro de filas en el archivo <extract.py> donde se crearos las clases y los métodos para esa tarea, le prueba de este script se realizó en el archivo <test_methods>, vale la pena mencionar que se descargaron los datos en un solo archivo llamado train debido a que se realizó un análisis exploratorio para todo el conjunto de datos posteriormente se crearon los grupos de entrenamiento y test

2. Con los datos resultantes se procedió a realizar el análisis exploratorio que consistió en tres etapas para poder conocer el comportamiento de los datos y las variables realmente importantes para el análisis:
	* En la primera etapa <output_initial.html> se analizaron todas las variables en conjunto con la ayuda de la librería ydata_profiling de la cual se trataron primero las variables con la alarma de *Unsupported*