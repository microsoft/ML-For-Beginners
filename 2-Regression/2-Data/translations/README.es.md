# Construye un modelo de regresión usando Scikit-learn: prepara y visualiza los datos

![Infografía de visualización de datos](../images/data-visualization.png)

Infografía por [Dasani Madipalli](https://twitter.com/dasani_decoded)

## [Examen previo a la lección](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/11?loc=es)

> ### [Esta lección se encuentra disponible en R!](../solution/R/lesson_2-R.ipynb)

## Introducción

Ahora que has configurado las herramientas necesarias para iniciar el trabajo con la construcción de modelo de aprendizaje automático con Scikit-learn, estás listo para comenzar a realizar preguntas a tus datos. Mientras trabajas con los datos y aplicas soluciones de ML, es muy importante entender cómo realizar las preguntas correctas para desbloquear el potencial de tu conunto de datos.

En esta lección, aprenderás:

- Cómo preparar tus datos para la construcción de modelos.
- Cómo usar Matplotlib para visualización de datos.

[![Preparación y visualización de datos](https://img.youtube.com/vi/11AnOn_OAcE/0.jpg)](https://youtu.be/11AnOn_OAcE "Video de preparación y visualizción de datos - ¡Clic para ver!")
> 🎥 Da clic en la imagen superior para ver un video de los aspectos clave de esta lección


## Realizando la pregunta correcta a tus datos

La pregunta para la cual necesitas respuesta determinará qué tipo de algoritmos de ML requerirás. Y la calidad de la respuesta que obtendas será altamente dependiente de la naturaleza de tus datos.

Echa un vistazo a los [datos](../../data/US-pumpkins.csv) provistos para esta lección. Puedes abrir este archivo .csv en VS Code. Un vistazo rápido muestra inmediatamente que existen campos en blanco y una mezcla de datos numéricos y de cadena. También hay una columna extraña llamada 'Package' donde los datos están mezclados entre los valores 'sacks', 'bins' y otros. Los datos de hecho, son un pequeño desastre.

De hecho, no es muy común obtener un conjunto de datos que esté totalmente listo para su uso en un modelo de ML. En esta lección, aprenderás cómo preparar un conjunto de datos en crudo usando librerías estándares de Python. También aprenderás varias técnicas para visualizar los datos.

## Caso de estudio: 'El mercado de calabazas'

En este directorio encontrarás un archivo .cvs in la raíz del directorio `data` llamado [US-pumpkins.csv](../../data/US-pumpkins.csv), el cual incluye  1757 líneas de datos acerca del mercado de calabazas, ordenados en agrupaciones por ciudad. Estos son loas datos extraídos de [Reportes estándar de mercados terminales de cultivos especializados](https://www.marketnews.usda.gov/mnp/fv-report-config-step1?type=termPrice) distribuido por el Departamento de Agricultura de los Estados Unidos.

### Preparando los datos

Estos datos son de dominio público. Puede ser descargado en varios archivos por separado, por ciudad, desde el sitio web de USDA. para evitar demasiados archivos por separado, hemos concatenado todos los datos de ciudad en una hoja de cálculo, así ya hemos _preparado_ los datos un poco. Lo siguiente es dar un vistazo más a fondo a los datos.

### Los datos de las calabazas - conclusiones iniciales

¿Qué notas acerca de los datos? Ya has visto que hay una mezcla de cadenas, números, blancos y valores extraños a los cuales debes encontrarle sentido.

¿Qué preguntas puedes hacerle a los datos usando una técnica de regresión? Qué tal el "predecir el precio de la venta de calabaza durante un mes dado". Viendo nuevamente los datos, hay algunos cambios que necesitas hacer para crear las estructuras de datos necesarias para la tarea.
## Ejercicio - Analiza los datos de la calabaza

Usemos [Pandas](https://pandas.pydata.org/), (el nombre es un acrónimo de `Python Data Analysis`) a tool very useful for shaping data, to analyze and prepare this pumpkin data.

### Primero, revisa las fechas faltantes

Necesitarás realizar algunos pasos para revisar las fechas faltantes:

1. Convertir las fechas a formato de mes (las fechas están en formato de EE.UU., por lo que el formato es `MM/DD/YYYY`).
2. Extrae el mes en una nueva columna.

Abre el archivo _notebook.ipynb_ en Visual Studio Code e importa la hoja de cálculo en un nuevo dataframe de Pandas.

1. Usa la función `head()` para visualizar las primeras cinco filas.

    ```python
    import pandas as pd
    pumpkins = pd.read_csv('../data/US-pumpkins.csv')
    pumpkins.head()
    ```

    ✅ ¿Qué función usarías para ver las últimas cinco filas?

1. Revisa si existen datos faltantes en el dataframe actual:

    ```python
    pumpkins.isnull().sum()
    ```

    Hay datos faltabtes, pero quizá no importen para la tarea en cuestión.

1. Para facilitar el trabajo con tu dataframe, elimina varias de sus columnas usando `drop()`, manteniendo sólo las columnas que necesitas:

    ```python
    new_columns = ['Package', 'Month', 'Low Price', 'High Price', 'Date']
    pumpkins = pumpkins.drop([c for c in pumpkins.columns if c not in new_columns], axis=1)
    ```

### Segundo, determina el precio promedio de la calabaza

Piensa en cómo determinar el precio promedio de la calabaza en un mes dado. ¿Qué columnas eligirás para esa tarea? Pista: necesitarás 3 columnas.

Solución: toma el promedio de las columnas `Low Price` y `High Price` para poblar la nueva columna `Price` y convierte la columna `Date` para mostrar únicamente el mes, Afortunadamente, de acuerdo a la revisión  de arriba, no hay datos faltantes para las fechas o precios.

1. Para calcular el promedio, agrega el siguiente código:

    ```python
    price = (pumpkins['Low Price'] + pumpkins['High Price']) / 2

    month = pd.DatetimeIndex(pumpkins['Date']).month

    ```

   ✅ Siéntete libre de imprimir cualquier dato que desees verificar, usando `print(month)`.

2. Ahora, copia tus datos convertidos en un nuevo dataframe de Pandas:

    ```python
    new_pumpkins = pd.DataFrame({'Month': month, 'Package': pumpkins['Package'], 'Low Price': pumpkins['Low Price'],'High Price': pumpkins['High Price'], 'Price': price})
    ```

    Imprimir tu dataframe te mostrará un conjunto de datos limpio y ordenado, en el cual puedes construir tu nuevo modelo de regresión. 

### ¡Pero espera!, Hay algo raro aquí

Si observas la columna `Package`, las calabazas se venden en distintas configuraciones. Algunas son vendidas en medidas de '1 1/9 bushel', y otras en '1/2 bushel', algunas por pieza, algunas por libra y otras en grandes cajas de ancho variable.

> Las calabazas parecen muy difíciles de pesar consistentemente.

Indagando en los datos originales, es interesante que cualquiera con el valor `Unit of Sale` igualado a 'EACH' o 'PER BIN' también tiene el tipo de `Package` por pulgada, por cesto, o 'each'. Las calabazas parecen muy difíciles de pesar consistentemente, por lo que las filtraremos seleccionando solo aquellas calabazas con el string 'bushel' en su columna `Package`.

1. Agrega un filtro al inicio del archivo, debajo de la importación inicial del .csv:

    ```python
    pumpkins = pumpkins[pumpkins['Package'].str.contains('bushel', case=True, regex=True)]
    ```

    Si imprimes los datos ahora, puedes ver que solo estás obteniendo alrededor de 415 filas de datos que contienen calabazas por fanegas.

### ¡Pero espera! Aún hay algo más que hacer

¿Notaste que la cantidad de fanegas varían por fila? Necesitas normalizar el precio para así mostrar el precio por fanega, así que haz los cálculos para estandarizarlo.

1. Agrega estas líneas después del bloque para así crear el dataframe new_pumpkins:

    ```python
    new_pumpkins.loc[new_pumpkins['Package'].str.contains('1 1/9'), 'Price'] = price/(1 + 1/9)

    new_pumpkins.loc[new_pumpkins['Package'].str.contains('1/2'), 'Price'] = price/(1/2)
    ```

✅ De acuerdo a [The Spruce Eats](https://www.thespruceeats.com/how-much-is-a-bushel-1389308), el peso de una fanega depende del tipo de producto, ya que es una medida de volumen. "Una fanega de tomates, por ejemplo, se supone pese 56 libras... Las hojas y verduras usan más espacio con menos peso, por lo que una fanega de espinaca es de sólo 20 libras." ¡Todo es tan complicado! No nos molestemos en realizar una conversión fanega-a-libra, y en su lugar hagámosla por precio de fanega. ¡Todo este estudio de las fanegas de calabazas nos mostrará cuán importante es comprender la naturaleza de tus datos!

Ahora, puedes analizar el precio por unidad basándote en su medida de fanega. Si imprimes los datos una vez más, verás que ya están estandarizados.

✅ ¿Notaste que las calabazas vendidas por media fanega son más caras? ¿Puedes descubrir la razón? Ayuda: Las calabazas pequeñas son mucho más caras que las grandes, probablemente porque hay muchas más de ellas por fanega, dado el espacio sin usar dejado por una calabaza grande.

## Estrategias de visualización

parte del rol de un científico de datos es el demostrar la calidad y naturaleza de los dato con los que está trabajando. Para hacerlo, usualmente crean visualizaciones interesantes, o gráficos, grafos, y gráficas, mostrando distintos aspectos de los datos. De esta forma, son capaces de mostrar visualmente las relaciones y brechas de que otra forma son difíciles de descubrir.

Las visualizaciones también ayudan a determinar la técnica de aprendizaje automático más apropiada para los datos. Por ejemplo, un gráfico de dispersión que parece seguir una línea, indica que los datos son un buen candidato para un ejercicio de regresión lineal.

Una librería de visualización de datos que funciona bien en los notebooks de Jupyter es [Matplotlib](https://matplotlib.org/) (la cual también viste en la lección anterior).

> Obtén más experiencia con la visualización de datos en [estos tutoriales](https://docs.microsoft.com/learn/modules/explore-analyze-data-with-python?WT.mc_id=academic-77952-leestott).

## Ejercicio - experimenta con Matplotlib

Intenta crear algunas gráficas básicas para mostrar el nuevo dataframe que acabas de crear. ¿Qué mostraría una gráfica de línea básica?

1. Importa Matplotlib al inicio del archivo, debajo de la importación de Pandas:

    ```python
    import matplotlib.pyplot as plt
    ```

1. Vuelve a correr todo el notebook para refrescarlo.
1. Al final del notebook, agrega una celda para graficar los datos como una caja:

    ```python
    price = new_pumpkins.Price
    month = new_pumpkins.Month
    plt.scatter(price, month)
    plt.show()
    ```

    ![Una gráfica de dispersión mostrando la relación precio a mes](../images/scatterplot.png)

    ¿La gráfica es útil? ¿Hay algo acerca de ésta que te sorprenda?

    No es particularmente útil ya que todo lo que hace es mostrar tus datos como puntos dispersos en un mes dado.

### Hacerlo útil

Para obtener gráficas para mostrar datos útiles, necesitas agrupar los datos de alguna forma. Probemos creando un gráfico donde el eje y muestre los meses y los datos demuestren la distribución de los datos.

1. Agrega una celda para crear una gráfica de barras agrupadas:

    ```python
    new_pumpkins.groupby(['Month'])['Price'].mean().plot(kind='bar')
    plt.ylabel("Pumpkin Price")
    ```

    ![Una gráfica de bsarras mostrando la relación precio a mes](../images/barchart.png)

    ¡Esta es una visualización de datos más útil! Parece indicar que el precio más alto para las calabazas ocurre en Septiembre y Octubre. ¿Cumple esto con tus expectativas? ¿por qué sí o por qué no?

---

## 🚀Desafío

Explora los distintos tipos de visualización que ofrece Matplotlib. ¿Qué tipos son los más apropiados para problemas de regresión?

## [Examen posterior a la lección](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/12?loc=es)

## Revisión y autoestudio

Dale un vistazo a las distintas forma de visualizar los datos. Haz un lista de las distintas librerías disponibles y nota cuales son mejores para cierto tipo de tareas, por ejemplo visualizaciones 2D vs visualizaciones 3D. ¿Qué descubriste?

## Asignación

[Explorando la visualización](assignment.es.md)
