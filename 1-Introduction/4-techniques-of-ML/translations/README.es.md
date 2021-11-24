# Técnicas de Machine Learning

El proceso de creación, uso y mantenimiento de modelos de machine learning, y los datos que se utilizan, es un proceso muy diferente de muchos otros flujos de trabajo de desarrollo. En esta lección, demistificaremos el proceso y describiremos las principales técnicas que necesita saber. Vas a:  

- Comprender los procesos que sustentan el machine learning a un alto nivel.
- Explorar conceptos básicos como 'modelos', 'predicciones', y 'datos de entrenamiento'

 
## [Cuestionario previo a la conferencia](https://white-water-09ec41f0f.azurestaticapps.net/quiz/7/)
## Introducción

A un alto nivel, el arte de crear procesos de machine learning (ML) se compone de una serie de pasos:

1. **Decidir sobre la pregunta**. La mayoría de los procesos de ML, comienzan por hacer una pregunta que no puede ser respondida por un simple programa condicional o un motor basado en reglas. Esas preguntas a menudo giran en torno a predicciones basadas en una recopilación de datos.
2. **Recopile y prepare datos**. Para poder responder a su pregunta, necesita datos. La calidad y, a veces, cantidad de sus datos determinarán que tan bien puede responder a su pregunta inicial. La visualización de datos es un aspecto importante de esta fase. Esta fase también incluye dividir los datos en un grupo de entrenamiento y pruebas para construir un modelo.
3. **Elige un método de entrenamiento**. Dependiendo de su pregunta y la naturaleza de sus datos, debe elegir cómo desea entrenar un modelo para reflejar mejor sus datos y hacer predicciones precisas contra ellos. Esta es la parte de su proceso de ML que requiere experiencia específica y, a menudo, una cantidad considerable de experimentación.
4. **Entrena el modelo**. Usando sus datos de entrenamiento, usará varios algoritmos para entrenar un modelo para reconocer patrones en los datos. El modelo puede aprovechar las ponderaciones internas que se pueden ajustar para privilegiar ciertas partes de los datos sobre otras para construir un modelo mejor.
5. **Evaluar el modelo**. Utiliza datos nunca antes vistos (sus datos de prueba) de su conjunto recopilado para ver cómo se está desempeñando el modelo.
6. **Ajuste de parámetros**. Según el rendimiento de su modelo, puede rehacer el proceso utilizando diferentes parámetros, o variables, que controlan el comportamiento de los algoritmos utilizados para entrenar el modelo.
7. **Predecir**. Utilice nuevas entradas para probar la precisión de su modelo.

## Qué preguntas hacer

Las computadoras son particularmente hábiles para descubrir patrones ocultos en los datos. Esta utlidad es muy útil para los investigadores que tienen preguntas sobre un dominio determinado que no pueden responderse fácilmente mediante la creación de un motor de reglas basadas en condicionales. Dada una tarea actuarial, por ejemplo, un científico de datos podría construir reglas creadas manualmente sobre la mortalidad de los fumadores frente a los no fumadores.

Sin embargo, cuando se incorporan muchas otras variables a la ecuación, un modelo de ML podría resultar más eficiente para predecir las tasas de mortalidad futuras en función de los antecedentes de salud. Un ejemplo más alegre podría hacer predicciones meteorológicas para el mes de abril en una ubicación determinada que incluya latitud, longitud, cambio climático, proximidad al océano, patrones de la corriente en chorro, y más. 

✅ Esta [presentación de diapositivas](https://www2.cisl.ucar.edu/sites/default/files/0900%20June%2024%20Haupt_0.pdf) sobre modelos meteorológicos ofrece una perspectiva histórica del uso de ML en el análisis meteorológico.

## Tarea previas a la construcción

Antes de comenzar a construir su modelo, hay varias tareas que debe completar. Para examinar su pregunta y formar una hipótesis basada en las predicciones de su modelo, debe identificar y configurar varios elementos.

### Datos

Para poder responder su pregunta con algún tipo de certeza, necesita una buena cantidad de datos del tipo correcto.
Hay dos cosas que debe hacer en este punto:

- **Recolectar datos**. Teniendo en cuenta la lección anterior sobre la equidad en el análisis de datos, recopile sus datos con cuidado. Tenga en cuenta la fuente de estos datos, cualquier sesgo inherente que pueda tener y documente su origen.
- **Preparar datos**. Hay varios pasos en el proceso de preparación de datos. Podría necesitar recopilar datos y normalizarlos si provienen de diversas fuentes. Puede mejorar la calidad y cantidad de los datos mediante varios métodos, como convertir strings en números (como hacemos en [Clustering](../../5-Clustering/1-Visualize/README.md)). También puede generar nuevos datos, basados en los originales (como hacemos en [Clasificación](../../4-Classification/1-Introduction/README.md)). Puede limpiar y editar los datos (como lo haremos antes de la lección [Web App](../../3-Web-App/README.md)). Por último, es posible que también deba aleatorizarlo y mezclarlo, según sus técnicas de entrenamiento.

✅ Despúes de recopilar y procesar sus datos, tómese un momento para ver si su forma le permitirá responder a su pregunta. ¡Puede ser que los datos no funcionen bien en su tarea dada, como descubriremos en nuestras lecciones de[Clustering](../../5-Clustering/1-Visualize/README.md)!

### Características y destino

Una característica es una propiedad medible de los datos. En muchos conjuntos de datos se expresa como un encabezado de columna como 'date' 'size' o 'color'. La variable de entidad, normalmente representada como `X` en el código, representa la variable de entrada que se utilizará para entrenar el modelo.

Un objetivo es una cosa que está tratando de predecir. Target generalmente representado como `y` en el código, representa la respuesta a la pregunta que está tratando de hacer de sus datos: en diciembre, ¿qué color de calabazas serán más baratas?; en San Francisco, ¿qué barrios tendrán el mejor precio de bienes raíces? A veces, target también se conoce como atributo label.

### Seleccionando su variable característica

🎓 **Selección y extracción de características** ¿Cómo sabe que variable elegir al construir un modelo? Probablemente pasará por un proceso de selección o extracción de características para elegir las variables correctas para un mayor rendimiento del modelo. Sin embargo, no son lo mismo: "La extracción de características crea nuevas características a partir de funciones de las características originales, mientras que la selección de características devuelve un subconjunto de las características." ([fuente](https://wikipedia.org/wiki/Feature_selection))

### Visualiza tus datos

Un aspecto importante del conjunto de herramientas del científico de datos es el poder de visualizar datos utilizando varias bibliotecas excelentes como Seaborn o MatPlotLib. Representar sus datos visualmente puede permitirle descubrir correlaciones ocultas que puede aprovechar. Sus visualizaciones también pueden ayudarlo a descubrir sesgos o datos desequilibrados. (como descubrimos en [Clasificación](../../4-Classification/2-Classifiers-1/README.md)).

### Divide tu conjunto de datos

Antes del entrenamiento, debe dividir su conjunto de datos en dos o más partes de tamaño desigual pero que representen bien los datos.

- **Entrenamiento**. Esta parte del conjunto de datos se ajusta a su modelo para entrenarlo. Este conjunto constituye la mayor parte del conjunto de datos original.
- **Pruebas**. Un conjunto de datos de pruebas es un grupo independiente de datos, a menudo recopilado a partir de los datos originales, que se utiliza para confirmar el rendimiento del modelo construido.
- **Validación**. Un conjunto de validación es un pequeño grupo independiente de ejemplos que se usa para ajustar los hiperparámetros o la arquitectura del modelo para mejorar el modelo. Dependiendo del tamaño de su conjunto de datos y de la pregunta que se está haciendo, es posible que no necesite crear este tercer conjunto (como notamos en [Pronóstico se series de tiempo](../../7-TimeSeries/1-Introduction/README.md)).

## Contruye un modelo

Usando sus datos de entrenamiento, su objetivo es construir un modelo, o una representación estadística de sus datos, utilizando varios algoritmos para **entrenarlo**. El entrenamiento de un modelo lo expone a los datos y le permite hacer suposiciones sobre los patrones percibidos que descubre, valida y rechaza.

### Decide un método de entrenamiento

Dependiendo de su pregunta y la naturaleza de sus datos, elegirá un método para entrenarlos. Echando un vistazo a la [documentación de Scikit-learn ](https://scikit-learn.org/stable/user_guide.html) - que usamos en este curso - puede explorar muchas formas de entrenar un modelo. Dependiendo de su experiencia, es posible que deba probar varios métodos diferentes para construir el mejor modelo. Es probable que pase por un proceso en el que los científicos de datos evalúan el rendimiento de un modelo alimentándolo con datos no vistos anteriormente por el modelo, verificando la precisión, el sesgo, y otros problemas que degradan la calidad, y seleccionando el método de entrenamieto más apropiado para la tarea en cuestión.
### Entrena un modelo

Armado con sus datos de entrenamiento, está listo para "ajustarlo" para crear un modelo. Notará que en muchas bibliotecas de ML encontrará un método de la forma 'model.fit' - es en este momento que envía su variable de característica como una matriz de valores (generalmente `X`) y una variable de destino (generalmente `y`).

### Evaluar el modelo

Una vez que se completa el proceso de entrenamiento (puede tomar muchas iteraciones, o 'épocas', entrenar un modelo de gran tamaño), podrá evaluar la calidad del modelo utilizando datos de prueba para medir su rendimiento. Estos datos son un subconjunto de los datos originales que el modelo no ha analizado previamente. Puede imprimir una tabla de métricas sobre la calidad de su modelo.

🎓 **Ajuste del modelo (Model fitting)**

En el contexto del machine learning, el ajuste del modelo se refiere a la precisión de la función subyacente del modelo cuando intenta analizar datos con los que no está familiarizado.

🎓 **Ajuste insuficiente (Underfitting)** y **sobreajuste (overfitting)** son problemas comunes que degradan la calidad del modelo, ya que el modelo no encaja suficientemente bien, o encaja demasiado bien. Esto hace que el modelo haga predicciones demasiado estrechamente alineadas o demasiado poco alineadas con sus datos de entrenamiento. Un modelo sobreajustado (overfitting) predice demasiado bien los datos de entrenamiento porque ha aprendido demasiado bien los detalles de los datos y el ruido. Un modelo insuficientemente ajustado (Underfitting) es impreciso, ya que ni puede analizar con precisión sus datos de entrenamiento ni los datos que aún no ha 'visto'.

![Sobreajuste de un modelo](images/overfitting.png)
> Infografía de  [Jen Looper](https://twitter.com/jenlooper)

## Ajuste de parámetros

Una vez que haya completado su entrenamiento inicial, observe la calidad del modelo y considere mejorarlo ajustando sus 'hiperparámetros'. Lea más sobre el proceso [en la documentación](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters?WT.mc_id=academic-15963-cxa).

## Predicción

Este es el momento en el que puede usar datos completamente nuevos para probar la precisión de su modelo. En una configuración de ML aplicado, donde está creando activos web para usar el modelo en producción, este proceso puede implicar la recopilación de la entrada del usuario (presionar un botón, por ejemplo) para establecer una variable y enviarla al modelo para la inferencia o evaluación.
En estas lecciones, descubrirá cómo utilizar estos pasos para preparar, construir, probar, evaluar, y predecir - todos los gestos de un científico de datos y más, a medida que avanza en su viaje para convertirse en un ingeniero de machine learning 'full stack'.
---

## 🚀Desafío

Dibuje un diagrama de flujos que refleje los pasos de practicante de ML. ¿Dónde te ves ahora mismo en el proceso? ¿Dónde predice que encontrará dificultades? ¿Qué te parece fácil? 

## [Cuestionario posterior a la conferencia](https://white-water-09ec41f0f.azurestaticapps.net/quiz/8/)

## Revisión & Autoestudio

Busque entrevistas en línea con científicos de datos que analicen su trabajo diario. Aquí está [uno](https://www.youtube.com/watch?v=Z3IjgbbCEfs).

## Asignación

[Entrevistar a un científico de datos](assignment.md)
