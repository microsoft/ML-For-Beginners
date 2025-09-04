<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "a683e1fe430bb0d4a10b68f6ca15e0a6",
  "translation_date": "2025-09-03T22:38:39+00:00",
  "source_file": "2-Regression/2-Data/README.md",
  "language_code": "es"
}
-->
# Construir un modelo de regresión usando Scikit-learn: preparar y visualizar datos

![Infografía de visualización de datos](../../../../translated_images/data-visualization.54e56dded7c1a804d00d027543f2881cb32da73aeadda2d4a4f10f3497526114.es.png)

Infografía por [Dasani Madipalli](https://twitter.com/dasani_decoded)

## [Cuestionario previo a la lección](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/11/)

> ### [¡Esta lección está disponible en R!](../../../../2-Regression/2-Data/solution/R/lesson_2.html)

## Introducción

Ahora que tienes las herramientas necesarias para comenzar a construir modelos de aprendizaje automático con Scikit-learn, estás listo para empezar a formular preguntas sobre tus datos. Al trabajar con datos y aplicar soluciones de aprendizaje automático, es muy importante saber cómo formular la pregunta correcta para desbloquear adecuadamente el potencial de tu conjunto de datos.

En esta lección, aprenderás:

- Cómo preparar tus datos para construir modelos.
- Cómo usar Matplotlib para la visualización de datos.

## Formular la pregunta correcta sobre tus datos

La pregunta que necesitas responder determinará qué tipo de algoritmos de aprendizaje automático utilizarás. Y la calidad de la respuesta que obtengas dependerá en gran medida de la naturaleza de tus datos.

Echa un vistazo a los [datos](https://github.com/microsoft/ML-For-Beginners/blob/main/2-Regression/data/US-pumpkins.csv) proporcionados para esta lección. Puedes abrir este archivo .csv en VS Code. Una revisión rápida muestra inmediatamente que hay espacios en blanco y una mezcla de datos de tipo cadena y numéricos. También hay una columna extraña llamada 'Package' donde los datos son una mezcla entre 'sacks', 'bins' y otros valores. De hecho, los datos están un poco desordenados.

[![ML para principiantes - Cómo analizar y limpiar un conjunto de datos](https://img.youtube.com/vi/5qGjczWTrDQ/0.jpg)](https://youtu.be/5qGjczWTrDQ "ML para principiantes - Cómo analizar y limpiar un conjunto de datos")

> 🎥 Haz clic en la imagen de arriba para ver un breve video sobre cómo preparar los datos para esta lección.

De hecho, no es muy común recibir un conjunto de datos completamente listo para usar y crear un modelo de aprendizaje automático directamente. En esta lección, aprenderás cómo preparar un conjunto de datos sin procesar utilizando bibliotecas estándar de Python. También aprenderás varias técnicas para visualizar los datos.

## Caso de estudio: 'el mercado de calabazas'

En esta carpeta encontrarás un archivo .csv en la carpeta raíz `data` llamado [US-pumpkins.csv](https://github.com/microsoft/ML-For-Beginners/blob/main/2-Regression/data/US-pumpkins.csv), que incluye 1757 líneas de datos sobre el mercado de calabazas, organizados en agrupaciones por ciudad. Estos son datos sin procesar extraídos de los [Informes estándar de mercados terminales de cultivos especiales](https://www.marketnews.usda.gov/mnp/fv-report-config-step1?type=termPrice) distribuidos por el Departamento de Agricultura de los Estados Unidos.

### Preparar los datos

Estos datos son de dominio público. Se pueden descargar en muchos archivos separados, por ciudad, desde el sitio web del USDA. Para evitar demasiados archivos separados, hemos concatenado todos los datos de las ciudades en una sola hoja de cálculo, por lo que ya hemos _preparado_ un poco los datos. Ahora, echemos un vistazo más de cerca a los datos.

### Los datos de calabazas - primeras conclusiones

¿Qué notas sobre estos datos? Ya viste que hay una mezcla de cadenas, números, espacios en blanco y valores extraños que necesitas interpretar.

¿Qué pregunta puedes formular sobre estos datos utilizando una técnica de regresión? ¿Qué tal "Predecir el precio de una calabaza en venta durante un mes determinado"? Al observar nuevamente los datos, hay algunos cambios que necesitas hacer para crear la estructura de datos necesaria para esta tarea.

## Ejercicio - analizar los datos de calabazas

Utilicemos [Pandas](https://pandas.pydata.org/) (el nombre significa `Python Data Analysis`), una herramienta muy útil para dar forma a los datos, para analizar y preparar estos datos de calabazas.

### Primero, verifica si faltan fechas

Primero necesitarás tomar medidas para verificar si faltan fechas:

1. Convierte las fechas al formato de mes (estas son fechas de EE. UU., por lo que el formato es `MM/DD/YYYY`).
2. Extrae el mes a una nueva columna.

Abre el archivo _notebook.ipynb_ en Visual Studio Code e importa la hoja de cálculo a un nuevo dataframe de Pandas.

1. Usa la función `head()` para ver las primeras cinco filas.

    ```python
    import pandas as pd
    pumpkins = pd.read_csv('../data/US-pumpkins.csv')
    pumpkins.head()
    ```

    ✅ ¿Qué función usarías para ver las últimas cinco filas?

1. Verifica si hay datos faltantes en el dataframe actual:

    ```python
    pumpkins.isnull().sum()
    ```

    Hay datos faltantes, pero tal vez no importen para la tarea en cuestión.

1. Para que tu dataframe sea más fácil de trabajar, selecciona solo las columnas que necesitas, utilizando la función `loc`, que extrae del dataframe original un grupo de filas (pasadas como primer parámetro) y columnas (pasadas como segundo parámetro). La expresión `:` en el caso siguiente significa "todas las filas".

    ```python
    columns_to_select = ['Package', 'Low Price', 'High Price', 'Date']
    pumpkins = pumpkins.loc[:, columns_to_select]
    ```

### Segundo, determina el precio promedio de las calabazas

Piensa en cómo determinar el precio promedio de una calabaza en un mes determinado. ¿Qué columnas elegirías para esta tarea? Pista: necesitarás 3 columnas.

Solución: toma el promedio de las columnas `Low Price` y `High Price` para llenar la nueva columna Price, y convierte la columna Date para que solo muestre el mes. Afortunadamente, según la verificación anterior, no hay datos faltantes para fechas o precios.

1. Para calcular el promedio, agrega el siguiente código:

    ```python
    price = (pumpkins['Low Price'] + pumpkins['High Price']) / 2

    month = pd.DatetimeIndex(pumpkins['Date']).month

    ```

   ✅ Siéntete libre de imprimir cualquier dato que desees verificar usando `print(month)`.

2. Ahora, copia tus datos convertidos en un nuevo dataframe de Pandas:

    ```python
    new_pumpkins = pd.DataFrame({'Month': month, 'Package': pumpkins['Package'], 'Low Price': pumpkins['Low Price'],'High Price': pumpkins['High Price'], 'Price': price})
    ```

    Al imprimir tu dataframe, verás un conjunto de datos limpio y ordenado sobre el cual puedes construir tu nuevo modelo de regresión.

### Pero espera, ¡hay algo extraño aquí!

Si miras la columna `Package`, las calabazas se venden en muchas configuraciones diferentes. Algunas se venden en medidas de '1 1/9 bushel', otras en '1/2 bushel', algunas por calabaza, otras por libra, y algunas en grandes cajas con anchos variables.

> Parece que las calabazas son muy difíciles de pesar de manera consistente.

Al profundizar en los datos originales, es interesante notar que cualquier cosa con `Unit of Sale` igual a 'EACH' o 'PER BIN' también tiene el tipo de `Package` por pulgada, por bin, o 'each'. Parece que las calabazas son muy difíciles de pesar de manera consistente, así que filtremos seleccionando solo las calabazas con la cadena 'bushel' en su columna `Package`.

1. Agrega un filtro en la parte superior del archivo, debajo de la importación inicial del .csv:

    ```python
    pumpkins = pumpkins[pumpkins['Package'].str.contains('bushel', case=True, regex=True)]
    ```

    Si imprimes los datos ahora, puedes ver que solo estás obteniendo las aproximadamente 415 filas de datos que contienen calabazas por bushel.

### Pero espera, ¡hay una cosa más por hacer!

¿Notaste que la cantidad de bushel varía por fila? Necesitas normalizar los precios para mostrar el precio por bushel, así que haz algunos cálculos para estandarizarlo.

1. Agrega estas líneas después del bloque que crea el dataframe new_pumpkins:

    ```python
    new_pumpkins.loc[new_pumpkins['Package'].str.contains('1 1/9'), 'Price'] = price/(1 + 1/9)

    new_pumpkins.loc[new_pumpkins['Package'].str.contains('1/2'), 'Price'] = price/(1/2)
    ```

✅ Según [The Spruce Eats](https://www.thespruceeats.com/how-much-is-a-bushel-1389308), el peso de un bushel depende del tipo de producto, ya que es una medida de volumen. "Un bushel de tomates, por ejemplo, se supone que pesa 56 libras... Las hojas y los vegetales ocupan más espacio con menos peso, por lo que un bushel de espinacas pesa solo 20 libras". ¡Es todo bastante complicado! No nos molestemos en hacer una conversión de bushel a libra, y en su lugar fijemos el precio por bushel. Todo este estudio sobre bushels de calabazas, sin embargo, demuestra lo importante que es entender la naturaleza de tus datos.

Ahora puedes analizar el precio por unidad basado en su medida de bushel. Si imprimes los datos una vez más, puedes ver cómo se han estandarizado.

✅ ¿Notaste que las calabazas vendidas por medio bushel son muy caras? ¿Puedes averiguar por qué? Pista: las calabazas pequeñas son mucho más caras que las grandes, probablemente porque hay muchas más por bushel, dado el espacio no utilizado que ocupa una calabaza grande y hueca para pastel.

## Estrategias de visualización

Parte del rol del científico de datos es demostrar la calidad y naturaleza de los datos con los que están trabajando. Para hacerlo, a menudo crean visualizaciones interesantes, como gráficos, diagramas y tablas, que muestran diferentes aspectos de los datos. De esta manera, pueden mostrar visualmente relaciones y vacíos que de otro modo serían difíciles de descubrir.

[![ML para principiantes - Cómo visualizar datos con Matplotlib](https://img.youtube.com/vi/SbUkxH6IJo0/0.jpg)](https://youtu.be/SbUkxH6IJo0 "ML para principiantes - Cómo visualizar datos con Matplotlib")

> 🎥 Haz clic en la imagen de arriba para ver un breve video sobre cómo visualizar los datos para esta lección.

Las visualizaciones también pueden ayudar a determinar la técnica de aprendizaje automático más adecuada para los datos. Un gráfico de dispersión que parece seguir una línea, por ejemplo, indica que los datos son buenos candidatos para un ejercicio de regresión lineal.

Una biblioteca de visualización de datos que funciona bien en Jupyter notebooks es [Matplotlib](https://matplotlib.org/) (que también viste en la lección anterior).

> Obtén más experiencia con la visualización de datos en [estos tutoriales](https://docs.microsoft.com/learn/modules/explore-analyze-data-with-python?WT.mc_id=academic-77952-leestott).

## Ejercicio - experimentar con Matplotlib

Intenta crear algunos gráficos básicos para mostrar el nuevo dataframe que acabas de crear. ¿Qué mostraría un gráfico de líneas básico?

1. Importa Matplotlib en la parte superior del archivo, debajo de la importación de Pandas:

    ```python
    import matplotlib.pyplot as plt
    ```

1. Vuelve a ejecutar todo el notebook para actualizar.
1. En la parte inferior del notebook, agrega una celda para graficar los datos como un cuadro:

    ```python
    price = new_pumpkins.Price
    month = new_pumpkins.Month
    plt.scatter(price, month)
    plt.show()
    ```

    ![Un gráfico de dispersión que muestra la relación entre precio y mes](../../../../translated_images/scatterplot.b6868f44cbd2051c6680ccdbb1510697d06a3ff6cd4abda656f5009c0ed4e3fc.es.png)

    ¿Es este un gráfico útil? ¿Hay algo que te sorprenda?

    No es particularmente útil, ya que solo muestra tus datos como una dispersión de puntos en un mes determinado.

### Hazlo útil

Para que los gráficos muestren datos útiles, generalmente necesitas agrupar los datos de alguna manera. Intentemos crear un gráfico donde el eje y muestre los meses y los datos demuestren la distribución de los mismos.

1. Agrega una celda para crear un gráfico de barras agrupado:

    ```python
    new_pumpkins.groupby(['Month'])['Price'].mean().plot(kind='bar')
    plt.ylabel("Pumpkin Price")
    ```

    ![Un gráfico de barras que muestra la relación entre precio y mes](../../../../translated_images/barchart.a833ea9194346d769c77a3a870f7d8aee51574cd1138ca902e5500830a41cbce.es.png)

    ¡Este es un gráfico de datos más útil! Parece indicar que el precio más alto de las calabazas ocurre en septiembre y octubre. ¿Cumple con tus expectativas? ¿Por qué sí o por qué no?

---

## 🚀Desafío

Explora los diferentes tipos de visualización que ofrece Matplotlib. ¿Qué tipos son más apropiados para problemas de regresión?

## [Cuestionario posterior a la lección](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/12/)

## Repaso y autoestudio

Echa un vistazo a las muchas formas de visualizar datos. Haz una lista de las diversas bibliotecas disponibles y anota cuáles son mejores para ciertos tipos de tareas, por ejemplo, visualizaciones en 2D frente a visualizaciones en 3D. ¿Qué descubres?

## Tarea

[Explorar la visualización](assignment.md)

---

**Descargo de responsabilidad**:  
Este documento ha sido traducido utilizando el servicio de traducción automática [Co-op Translator](https://github.com/Azure/co-op-translator). Aunque nos esforzamos por garantizar la precisión, tenga en cuenta que las traducciones automatizadas pueden contener errores o imprecisiones. El documento original en su idioma nativo debe considerarse como la fuente autorizada. Para información crítica, se recomienda una traducción profesional realizada por humanos. No nos hacemos responsables de malentendidos o interpretaciones erróneas que puedan surgir del uso de esta traducción.