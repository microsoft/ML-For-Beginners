# Introducción a la predicción de series de tiempo

![Resumen de series de tiempo en un boceto](../../../sketchnotes/ml-timeseries.png)

> Boceto de [Tomomi Imura](https://www.twitter.com/girlie_mac)

En esta lección y la siguiente, aprenderás un poco acerca de la predicción de series de tiempo, una parte interesante y valiosa del repertorio de de un científico de ML, la cual es un poco menos conocida que otros temas. La predicción de series de tiempo es una especie de 'bola de cristal': basada en el rendimiento pasado de una variable como el precio, puedes predecir su valor potencial futuro.

[![Introducción a la predicción de series de tiempo](https://img.youtube.com/vi/cBojo1hsHiI/0.jpg)](https://youtu.be/cBojo1hsHiI "Introducción a la predicción de series de tiempo")

> 🎥 Da clic en la imagen de arriba para ver un video acerca de la predicción de series de tiempo

## [Examen previo a la lección](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/41?loc=es)

Es un campo útil e interesante con valor real para el negocio, dada su aplicación directa a problemas de precio, inventario e incidentes de cadenas de suministro. Mientras que las técnicas de aprendizaje profundo han comenzado a usarse para ganar más conocimiento para mejorar el rendimiento de futuras predicciones, la predicción de series de tiempo sigue siendo un campo muy informado por técnicas de aprendizaje automático clásico.

> El útil plan de estudios de series de tiempo de Penn State puede ser encontrado [aquí](https://online.stat.psu.edu/stat510/lesson/1)

## Introducción

Supón que mantienes un arreglo de parquímetros inteligentes que proveen datos acerca de que tan seguido son usados y con qué duración de tiempo.

> ¿Qué pasaría si pudieras predecir, basado en el rendimiento pasado del medidor, su valor futuro de acuerdo a las leyes de suministro y demanda?

Predecir de forma precisa cuándo actuar para así lograr tu objetivo es una desafío que podría ser abordado con la predicción de series de tiempo.No haría feliz a la gente que le cobraran más en hora pico cuando están buscando un lugar para estacionarse, ¡pero sería una forma segura de generar ingresos para limpiar las calles!

Exploremos algunos de los tipos de algoritmos de series de tiempo e iniciemos un notebook para limpiar y preparar algunos datos. Los datos que analizarás son tomados de la competencia de predicción de GEFCom2014. Esta consiste de 3 años de carga eléctrica por hora y los valores de temperatura entre el 2012 y 2014. Dados los patrones históricos de carga eléctrica y temperatura, puedes predecir valores futuros de carga eléctrica.

En este ejemplo, aprenderás cómo predecir un paso de tiempo adelante, usando sólo la carga histórica. Antes de iniciar, sin embargo, es útil entender qué está pasando detrás de escena.

## Algunas definiciones

Al encontrar el término 'series de tiempo' necesitas entender su uso en varios contextos diferentes.

🎓 **Series de tiempo**

En matemáticas, "una serie de tiempo es una serie de puntos de datos indexados (o listados o graficados) en orden de tiempo. Más comúnmente, una serie de tiempo es una secuencia tomada en puntos sucesivos igualmente espaciados en el tiempo." Un ejemplo de una serie de tiempo es el valor diario de cierre de el [Promedio Industrial Down Jones](https://wikipedia.org/wiki/Time_series). El uso de gráficos de series de tiempo y modelado estadístico se encuentra frecuentemente en el procesamiento de señales, predicción del clima, predicción de sismos, y otros campos donde ocurren eventos y los puntos de datos pueden ser graficados en el tiempo.

🎓 **Análisis de series de tiempo**

El análisis de series de tiempo, es el análisis de los datos de las series de tiempo previamente mencionadas. Los datos de las series de tiempo pueden tomar distintas formas, incluyendo 'series de tiempo interrumpidas' las cuales detectan patrones en la evolución de las series de tiempo antes y después de un evento de interrupción. El tipo de análisis necesario para las series de tiempo depende de la naturaleza de los datos. Los datos de series de tiempo en sí mismos pueden tomar la forma de series de números o caracteres.

El análisis a realizar, usa una variedad de métodos, incluyendo dominio de frecuencia y dominio de tiempo, lineal y no lineal y más. [Aprende más](https://www.itl.nist.gov/div898/handbook/pmc/section4/pmc4.htm) acerca de varias formas de analizar este tipo de datos.

🎓 **Predicción de series de tiempo**

La predicción de series de tiempo es el uso de un modelo para predecir valores futuros basándose en patrones mostrados por datos previamente recopilados como ocurrieron en el pasado. Mientras es posible usar modelos de regresión para explorar los datos de las series de tiempo, con índices de tiempo como variables x en un plano, dichos datos se analizan mejor usando tipos especiales de modelos.

Los datos de series de timpo son una lista de observaciones ordenadas, a diferencia de los datos que pueden ser analizados por regresión lineal. El más común es ARIMA, el cual es un acrónimo que significa "Autoregressive Integrated Moving Average".

Los [modelos ARIMA](https://online.stat.psu.edu/stat510/lesson/1/1.1) "relacionan el valor presente de una serie de valores pasados y errores de predicción anteriores". Estos son más apropiados para el análisis de datos en el dominio de tiempo, donde los datos están se ordenan en el tiempo.

> Existen varios tipos de modelos ARIMA, los cuales puedes aprender [aquí](https://people.duke.edu/~rnau/411arim.htm) y que conocerás más tarde.

En la siguiente lección, construirás un modelo ARIMA usando [series de tiempo univariante](https://itl.nist.gov/div898/handbook/pmc/section4/pmc44.htm), las cual se enfoca en una variable que cambia su valor en el tiempo. Un ejemplo de este tipo de datos es [este conjunto de datos](https://itl.nist.gov/div898/handbook/pmc/section4/pmc4411.htm) que registra la concentración mensual de CO2 en el Observatorio Mauna Loa:

|  CO2   | YearMonth | Year  | Month |
| :----: | :-------: | :---: | :---: |
| 330.62 |  1975.04  | 1975  |   1   |
| 331.40 |  1975.13  | 1975  |   2   |
| 331.87 |  1975.21  | 1975  |   3   |
| 333.18 |  1975.29  | 1975  |   4   |
| 333.92 |  1975.38  | 1975  |   5   |
| 333.43 |  1975.46  | 1975  |   6   |
| 331.85 |  1975.54  | 1975  |   7   |
| 330.01 |  1975.63  | 1975  |   8   |
| 328.51 |  1975.71  | 1975  |   9   |
| 328.41 |  1975.79  | 1975  |  10   |
| 329.25 |  1975.88  | 1975  |  11   |
| 330.97 |  1975.96  | 1975  |  12   |

✅ Identifica la variable que cambia en el tiempo en este conjunto de datos

## [Características de datos](https://online.stat.psu.edu/stat510/lesson/1/1.1) de series de tiempo a considerar

Al mirar datos de series de tiempo, puedes notar que tienen ciertas características que necesitas tomar ne consideración y mitigar para entender mejor sus patrones. Si consideras los datos de las series de tiempo como proporcionando potencialmente una 'señal' que quieres analizar, estas características pueden ser interpretadas como 'ruido'. Frecuentemente necesitarás reducir este 'ruido' al compensar algunas de estas características usando ciertas técnicas estadísticas.

Aquí hay algunos conceptos que deberías saber para ser capaz de trabajar con las series de tiempo:

🎓 **Tendencias**

Las tendencias se definen como incrementos y decrementos medibles en el tiempo. [Lee más](https://machinelearningmastery.com/time-series-trends-in-python). En el contexto de las series de tiempo, se trata de cómo usar las tendencias y, si es necesario, eliminarlas.

🎓 **[Estacionalidad](https://machinelearningmastery.com/time-series-seasonality-with-python/)**

La estacionalidad se define como fluctuaciones periódicas, tales como prisas de vacaciones que pueden afectar las ventas, por ejemplo. [Da un vistazo](https://itl.nist.gov/div898/handbook/pmc/section4/pmc443.htm) a cómo los distintos tipos de gráficos muestran la estacionalidad en los datos.

🎓 **Valores atípicos**

Los valores atípicos están muy lejos de la varianza de datos estándar.

🎓 **Ciclos de largo plazo**

Independiente de la estacionalidad, los datos pueden mostrar un ciclo de largo plazo como un declive que dura más de un año.

🎓 **Varianza constante**

En el tiempo, algunos datos muestran fluctuaciones constantes, tales como el uso de energía por día y noche.

🎓 **Cambios abruptos**

Los datos pueden mostrar un cambio abrupto que puede necesitar mayor análisis. El cierre abrupto de negocios debido al COVID, por ejemplo, causó cambios en los datos.

✅ Aquí hay una [muestra de gráfico de series de tiempo](https://www.kaggle.com/kashnitsky/topic-9-part-1-time-series-analysis-in-python) mostrando la moneda diaria en juego gastada en algunos años. ¿Puedes identificar alguna de las características listadas arriba en estos datos?

![Gasto de moneda en el juego](../images/currency.png)

## Ejercicio - comenzando con los datos de uso de energía

Comencemos creando un modelo de series de tiempo para predecir el uso futuro de energía dato su uso pasado.

> Los datos en este ejemplo se tomaron de la competencia de predicción GEFCom2014. Consta de 3 años de valores de carga eléctrica y de temperatura medidos por hora entre 2012 y 2014.

>
> Tao Hong, Pierre Pinson, Shu Fan, Hamidreza Zareipour, Alberto Troccoli y Rob J. Hyndman, "Probabilistic energy forecasting: Global Energy Forecasting Competition 2014 and beyond", International Journal of Forecasting, vol.32, no.3, pp 896-913, July-September, 2016.

1. En el directorio `working` de esta lección, abre el archivo _notebook.ipynb_. Empieza agregando las bibliotecas que te ayudarán a cargar y visualizar datos

    ```python
    import os
    import matplotlib.pyplot as plt
    from common.utils import load_data
    %matplotlib inline
    ```

    Nota, estás usando los archivos del direcorio `common` incluido el cual configura tu ambiente y maneja la descarga de los datos.

2. Ahora, examina los datos como un dataframe llamando `load_data()` y `head()`:

    ```python
    data_dir = './data'
    energy = load_data(data_dir)[['load']]
    energy.head()
    ```

    Puedes ver que hay dos columnas representando la fecha y la carga:

    |                     |  load  |
    | :-----------------: | :----: |
    | 2012-01-01 00:00:00 | 2698.0 |
    | 2012-01-01 01:00:00 | 2558.0 |
    | 2012-01-01 02:00:00 | 2444.0 |
    | 2012-01-01 03:00:00 | 2402.0 |
    | 2012-01-01 04:00:00 | 2403.0 |

3. Ahora, grafica los datos llamando `plot()`:

    ```python
    energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![Gráfico de energía](../images/energy-plot.png)

4. Ahora, grafica la primer semana de Julio de 2014, al proveerla como entrada a `energy` en el patrón `[from date]: [to date]`:

    ```python
    energy['2014-07-01':'2014-07-07'].plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![julio](../images/july-2014.png)

    ¡Un hermoso gráfico! Da un vistazo a estos gráficos y ve si puedes determinar alguna de las características listadas arriba. ¿Que podemos suponer al visualizar los datos?

En la siguiente lección, crearás un modelo ARIMA para realizar algunas predicciones.

---

## 🚀Desafío

Haz una lista de todas las industrias y áreas de consulta en las que puedes pensar que se beneficiarían de la predicción de series de tiempo. ¿Puedes pensar en una aplicación de estas técnicas en las artes, en la econometría, ecología, venta al menudeo, la industria, finanzas? ¿Dónde más?

## [Examen posterior a la lección](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/42?loc=es)

## Revisión y autoestudio

Aunque no las cubriremos aquí, las redes neuronales son usadas algunas veces para mejorar los métodos clásicos de predicción de series de tiempo. Lee más acerca de ellas [en este artículo](https://medium.com/microsoftazure/neural-networks-for-forecasting-financial-and-economic-time-series-6aca370ff412)

## Asignación

[Visualiza algunas series de tiempo más](assignment.es.md)
