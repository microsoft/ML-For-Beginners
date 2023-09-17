# Introducción a la clasificación

En estas cuatro lecciones, explorarás un enfoque fundamental de aprendizaje automático clásico - _classification_. Ensayaremos usando varios algoritmos de clasificación con un conjunto de datos acerca de todas las cocinas brillantes de Asia e India. ¡Espero estés hambriento!

![Solo una pizca!](../images/pinch.png)

> ¡Celebra las cocinas de toda Asia en estas lecciones! Imagen de [Jen Looper](https://twitter.com/jenlooper)

La clasificación es una form de [aprendizaje supervisado](https://wikipedia.org/wiki/Supervised_learning) que conlleva mucho en común con técnicas de regresión. Si el aprendizaje automático trata todo acerca de la predicción de valores o nombres para las cosas usando conjuntos de datos, entonces la clasificación generalmente recae en dos grupos: _clasificación binaria_ y _clasificación multiclase_.

[![Introducción a la clasificación](https://img.youtube.com/vi/eg8DJYwdMyg/0.jpg)](https://youtu.be/eg8DJYwdMyg "Introducción a la clasificación")

> 🎥 Da clic en la imagen de arriba para ver el video: John Guttag del MIT presenta la clasificación

Recuerda:

- **La regresión lineal** te ayudó a predecir las relaciones entre las variables y hacer predicciones precisas donde un nuevo punto de datos podría reacer en una relación a esa línea. Por lo que puedes predecir _qué precio tendrá una calabaza en Septiembre vs Diciembre_, por ejemplo.
- **La regresión logística** te ayudó a descubrir "categorías binarias": en este punto de precio, ¿_la calabaza pertenece a la categoría orange or not-orange_?

La clasificación utiliza varios algorítmos para determinar otras formas de determinar la clase o etiqueta de un punto de datos. Trabajemos con estos datos de cocina para ver si, al observar un grupo de ingredientes, podemos determinar su cocina u origen.

## [Examen previo a la lección](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/19?loc=es)

> ### [¡Esta lección está disponible en R!](./solution/R/lesson_10-R.ipynb)

### Introducción

La clasificación es una de las actividades fundamentales del investigador de aprendizaje automático y el científico de datos. Desde la clasificación básica de un valor binario ("¿este correo electrónico es o no spam?"), hasta complejas clasificaciones de imágenes y segmentación utilizando la visión por computadora, simpre es útil ser capaz de ordenar los datos en clases y hacerle preguntas.

Para expresar el proceso de una forma más científica, nuestro método de clasificación crea un modelo predictivo que te habilita asignar la relación entre las variables de entrada a las variables de salida.

![Clasificación binaria vs multiclase](../images/binary-multiclass.png)

> Problemas binarios vs multiclase para que los algoritmos de clasificación los manejen. Infografía de [Jen Looper](https://twitter.com/jenlooper)

Antes de empezar el proceso de limpieza de nuestros datos, visualizarlos, y prepararlos para nuestras tareas de aprendizaje automático, aprendamos un poco acerca de las diversas formas en que el aprendizaje automático puede ser aprovechado para clasificar los datos.

Derivado de las clasificaciones [estadísticas](https://wikipedia.org/wiki/Statistical_classification), usando cracterística de uso del aprendizaje automático clásico, como `smoker`, `weight`, y `age` para determinar la _probabilidad de desarrollar X enfermedad_. Como una técnica de aprendizaje supervisada similar para los ejercicios de regresión que desempeñaste anteriormente, tus datos son etiquetados y los algoritmos de aprendizaje automático use esas etiquetas para clasificar y predecir clases (o 'características') de un conjunto de datos y asignarlos a un grupo o resultado.

✅ Date un momento para imaginar un conjunto de datos acerca de las cocinas. ¿Qué sería capaz de responder un modelo multiclase? ¿Qué sería capaz de responder un modelo binario? ¿Qué si quisieras determinar si una cocina dada fuera probable que usara fenogreco? ¿Qué si quisieras ver si, dado un regalo de una bolsa del supermercado llena de anís estrella, alcachofa, coliflor y rábano picante, pudieras crear un platillo Indio típico?

[![Canastas locas y misteriosas](https://img.youtube.com/vi/GuTeDbaNoEU/0.jpg)](https://youtu.be/GuTeDbaNoEU "Canastas locas y misteriosas")

> 🎥 Da clic en la imagen superior para ver un video. Toda la premisa del programa 'Chopped' es el 'cesto misterioso' donde los chefs tienen que hacer algunos platillos a partir de la elección al azar de ingredientes. ¡Seguramente un modelo de aprendizaje automático habría ayudado!

## Hola 'clasificador'

La pregutna que queremos hacer a este conjunto de datos de cocina es realmente una **pregunta multiclase**, así como tenemos muchas cocinas nacionales potenciales para trabajar, Dado un lote de ingredientes, ¿en cuáles de estas muchas clases encajarán los datos?

Scikit-learn ofrece diversos algoritmos distintos para usar en la clasificación de datos, dependiente en la naturaleza del problema que quieres resolver. En las siguientes dos lecciones, aprenderás acerca de varios de estos algoritmos.

## Ejercicio - limpia y equilibra tus datos

La primer tarea a la mano, antes de iniciar este proyecto, es limpiar y **equilibrar** tus datos para obtener mejores resultados. Comienza con el archivo en blanco _notebook.ipynb_ en la raíz de este directorio.

Lo primero a instalar es [imblearn](https://imbalanced-learn.org/stable/). Este es un paquete de Scikit-learn que te permitirá equilibrar mejor los datos (aprenderás más acerca de esta tarea en un minuto).

1. Para instalar `imblearn`, ejecuta `pip install`, así:

    ```python
    pip install imblearn
    ```

1. Importa los paquetes que necesitas para importar tus datos y visualizarlos, también importa `SMOTE` de `imblearn`.

    ```python
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np
    from imblearn.over_sampling import SMOTE
    ```

    Ahora está configurado para leer importart los datos a continuación.

1. La siguiente tarea será importar los datos:

    ```python
    df  = pd.read_csv('../data/cuisines.csv')
    ```

   Usando `read_csv()` leerá el contenido del archivo csv _cusines.csv_ y colocarlo en la variable `df`.

1. Comprueba la forma de los datos:

    ```python
    df.head()
    ```

   Las primeras cinco filas lucen así:

    ```output
    |     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
    | --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
    | 0   | 65         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 1   | 66         | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 2   | 67         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 3   | 68         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
    | 4   | 69         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
    ```

1. Obtén información acerca de estos datos llamando a `info()`:

    ```python
    df.info()
    ```

    Tu salida se parece a:

    ```output
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2448 entries, 0 to 2447
    Columns: 385 entries, Unnamed: 0 to zucchini
    dtypes: int64(384), object(1)
    memory usage: 7.2+ MB
    ```

## Ejercicio - aprendiendo acerca de cocinas

Ahora el trabajo comienza a hacerse más interesante. Descubramos la distribución de los datos, por cocina

1. Grafica los datos como barras llamando `barh()`:

    ```python
    df.cuisine.value_counts().plot.barh()
    ```

    ![Distribución de datos de cocina](../images/cuisine-dist.png)

    Existe un número finito de cocinas, pero la distribución de los datos es irregular. ¡Puedes corregirlo! Anter de hacerlo, explora un poco más.

1. Descubre cuántos datos están disponibles por cocina e imprímelos:

    ```python
    thai_df = df[(df.cuisine == "thai")]
    japanese_df = df[(df.cuisine == "japanese")]
    chinese_df = df[(df.cuisine == "chinese")]
    indian_df = df[(df.cuisine == "indian")]
    korean_df = df[(df.cuisine == "korean")]
    
    print(f'thai df: {thai_df.shape}')
    print(f'japanese df: {japanese_df.shape}')
    print(f'chinese df: {chinese_df.shape}')
    print(f'indian df: {indian_df.shape}')
    print(f'korean df: {korean_df.shape}')
    ```

    la salida luce así:

    ```output
    thai df: (289, 385)
    japanese df: (320, 385)
    chinese df: (442, 385)
    indian df: (598, 385)
    korean df: (799, 385)
    ```

## Descubriendo ingredientes

Ahora puedes profundizar en los datos y aprender cuáles son los ingredientes típicos por cocina. Deberías limpiar los datos recurrentes que crean confusión entre cocinas, así que aprendamos acerca de este problema.

1. Crea una función `create_ingredient()` en Python para crear un dataframe ingrediente. Esta función comenzará eliminando una columna inútil y ordenando los ingredientes por su conteo:

    ```python
    def create_ingredient_df(df):
        ingredient_df = df.T.drop(['cuisine','Unnamed: 0']).sum(axis=1).to_frame('value')
        ingredient_df = ingredient_df[(ingredient_df.T != 0).any()]
        ingredient_df = ingredient_df.sort_values(by='value', ascending=False,
        inplace=False)
        return ingredient_df
    ```

   Ahora puedes usar esa función para tener una idea de los 10 ingredientes más populares por cocina.

1. Llama `create_ingredient()` y graficalo llamando `barh()`:

    ```python
    thai_ingredient_df = create_ingredient_df(thai_df)
    thai_ingredient_df.head(10).plot.barh()
    ```

    ![Tailandeses](../images/thai.png)

1. Haz lo mismo para los datos de ingredientes japoneses:

    ```python
    japanese_ingredient_df = create_ingredient_df(japanese_df)
    japanese_ingredient_df.head(10).plot.barh()
    ```

    ![Japoneses](../images/japanese.png)

1. Ahora para los ingredientes chinos:

    ```python
    chinese_ingredient_df = create_ingredient_df(chinese_df)
    chinese_ingredient_df.head(10).plot.barh()
    ```

    ![Chinos](../images/chinese.png)

1. Grafica los ingredientes indios:

    ```python
    indian_ingredient_df = create_ingredient_df(indian_df)
    indian_ingredient_df.head(10).plot.barh()
    ```

    ![indios](../images/indian.png)

1. Finalmente, grafica los ingredientes coreanos:

    ```python
    korean_ingredient_df = create_ingredient_df(korean_df)
    korean_ingredient_df.head(10).plot.barh()
    ```

    ![coreanos](../images/korean.png)

1. Ahora, eliminar los ingredientes más comunes que crean confusión entre las distintas cocinas, llamando `drop()`:

   ¡Todos aman el arroz, el ajo y el gengibre!

    ```python
    feature_df= df.drop(['cuisine','Unnamed: 0','rice','garlic','ginger'], axis=1)
    labels_df = df.cuisine #.unique()
    feature_df.head()
    ```

## Equilibra el conjunto de datos

Ahora que has limpiado los datos, usa [SMOTE](https://imbalanced-learn.org/dev/references/generated/imblearn.over_sampling.SMOTE.html) - "Técnica de sobremuestreo de minoritario sintético" - para equilibrarlo.

1. Llama `fit_resample()`, esta estrategia genera nuevas muestras por interpolación.

    ```python
    oversample = SMOTE()
    transformed_feature_df, transformed_label_df = oversample.fit_resample(feature_df, labels_df)
    ```

    Al equilibrar tus datos, tendrás mejores resultados cuando los clasifiques. Piensa en una clasificación binaria. Si la mayoría de tus datos es una clase, un modelo de aprendizaje automático va a predecir esa clase más frecuentemente, solo porque hay más datos para ello. Equilibrar los datos toma cualquier dato sesgado y ayuda a remover este desequilibrio.

1. Ahora puedes comprobar los números de etiquetas por ingredientes:

    ```python
    print(f'new label count: {transformed_label_df.value_counts()}')
    print(f'old label count: {df.cuisine.value_counts()}')
    ```

    Tu salida luce así:

    ```output
    new label count: korean      799
    chinese     799
    indian      799
    japanese    799
    thai        799
    Name: cuisine, dtype: int64
    old label count: korean      799
    indian      598
    chinese     442
    japanese    320
    thai        289
    Name: cuisine, dtype: int64
    ```

    Los datos están bien y limpios, equilibrados ¡y muy deliciosos!

1. El último paso es guardar tus datos equilibrados, incluyendo etiquetas y características, en un nuevo dataframe que pueda ser exportado a un archivo:

    ```python
    transformed_df = pd.concat([transformed_label_df,transformed_feature_df],axis=1, join='outer')
    ```

1. Puedes dar un vistazo más a los datos usando `transformed_df.head()` y `transformed_df.info()`. Guarda una copia de estos datos para un uso en futuras lecciones:

    ```python
    transformed_df.head()
    transformed_df.info()
    transformed_df.to_csv("../data/cleaned_cuisines.csv")
    ```

    Este nuevo CSV ahora puede ser encontrado en la directorio raíz data.

---

## 🚀Desafío

Este plan de estudios contiene varios conjuntos de datos interesantes. Profundiza en los directorios `data` y ve si alguno contiene conjuntos de datos que serían apropiados para clasificación binaria o multiclase. ¿Qué preguntas harías a este conunto de datos?

## [Examen posterior a la lección](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/20?loc=es)

## Revisión y autoestudio

Explora la API de SMOTE. ¿Para qué casos de uso es se usa mejor? ¿Qué problemas resuelve?

## Asignación

[Explora métodos de clasificación](assignment.es.md)
