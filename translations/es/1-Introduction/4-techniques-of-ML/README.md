<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "dc4575225da159f2b06706e103ddba2a",
  "translation_date": "2025-09-03T23:33:19+00:00",
  "source_file": "1-Introduction/4-techniques-of-ML/README.md",
  "language_code": "es"
}
-->
# Técnicas de Aprendizaje Automático

El proceso de construir, usar y mantener modelos de aprendizaje automático y los datos que utilizan es muy diferente de muchos otros flujos de trabajo de desarrollo. En esta lección, desmitificaremos el proceso y describiremos las principales técnicas que necesitas conocer. Tú:

- Comprenderás los procesos que sustentan el aprendizaje automático a un nivel general.
- Explorarás conceptos básicos como 'modelos', 'predicciones' y 'datos de entrenamiento'.

## [Cuestionario previo a la lección](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/7/)

[![ML para principiantes - Técnicas de Aprendizaje Automático](https://img.youtube.com/vi/4NGM0U2ZSHU/0.jpg)](https://youtu.be/4NGM0U2ZSHU "ML para principiantes - Técnicas de Aprendizaje Automático")

> 🎥 Haz clic en la imagen de arriba para ver un breve video sobre esta lección.

## Introducción

A un nivel general, el arte de crear procesos de aprendizaje automático (ML) se compone de varios pasos:

1. **Decidir la pregunta**. La mayoría de los procesos de ML comienzan formulando una pregunta que no puede ser respondida mediante un programa condicional simple o un motor basado en reglas. Estas preguntas suelen girar en torno a predicciones basadas en una colección de datos.
2. **Recopilar y preparar datos**. Para poder responder a tu pregunta, necesitas datos. La calidad y, a veces, la cantidad de tus datos determinarán qué tan bien puedes responder a tu pregunta inicial. Visualizar los datos es un aspecto importante de esta fase. Esta fase también incluye dividir los datos en un grupo de entrenamiento y otro de prueba para construir un modelo.
3. **Elegir un método de entrenamiento**. Dependiendo de tu pregunta y la naturaleza de tus datos, necesitas elegir cómo quieres entrenar un modelo para que refleje mejor tus datos y haga predicciones precisas. Esta es la parte de tu proceso de ML que requiere experiencia específica y, a menudo, una cantidad considerable de experimentación.
4. **Entrenar el modelo**. Usando tus datos de entrenamiento, emplearás varios algoritmos para entrenar un modelo que reconozca patrones en los datos. El modelo puede utilizar pesos internos que se ajustan para privilegiar ciertas partes de los datos sobre otras y así construir un mejor modelo.
5. **Evaluar el modelo**. Utilizas datos nunca antes vistos (tus datos de prueba) de tu conjunto recopilado para ver cómo está funcionando el modelo.
6. **Ajuste de parámetros**. Basándote en el rendimiento de tu modelo, puedes repetir el proceso utilizando diferentes parámetros o variables que controlan el comportamiento de los algoritmos utilizados para entrenar el modelo.
7. **Predecir**. Utiliza nuevas entradas para probar la precisión de tu modelo.

## Qué pregunta hacer

Las computadoras son particularmente hábiles para descubrir patrones ocultos en los datos. Esta utilidad es muy útil para los investigadores que tienen preguntas sobre un dominio dado que no pueden ser respondidas fácilmente mediante la creación de un motor basado en reglas condicionales. Dado un trabajo actuarial, por ejemplo, un científico de datos podría construir reglas manuales sobre la mortalidad de fumadores frente a no fumadores.

Sin embargo, cuando se introducen muchas otras variables en la ecuación, un modelo de ML podría resultar más eficiente para predecir tasas de mortalidad futuras basándose en historiales de salud pasados. Un ejemplo más alegre podría ser hacer predicciones meteorológicas para el mes de abril en una ubicación dada basándose en datos que incluyen latitud, longitud, cambio climático, proximidad al océano, patrones de la corriente en chorro y más.

✅ Este [conjunto de diapositivas](https://www2.cisl.ucar.edu/sites/default/files/2021-10/0900%20June%2024%20Haupt_0.pdf) sobre modelos meteorológicos ofrece una perspectiva histórica sobre el uso de ML en el análisis del clima.  

## Tareas previas a la construcción

Antes de comenzar a construir tu modelo, hay varias tareas que necesitas completar. Para probar tu pregunta y formular una hipótesis basada en las predicciones de un modelo, necesitas identificar y configurar varios elementos.

### Datos

Para poder responder a tu pregunta con algún grado de certeza, necesitas una buena cantidad de datos del tipo adecuado. Hay dos cosas que necesitas hacer en este punto:

- **Recopilar datos**. Teniendo en cuenta la lección anterior sobre equidad en el análisis de datos, recopila tus datos con cuidado. Sé consciente de las fuentes de estos datos, cualquier sesgo inherente que puedan tener y documenta su origen.
- **Preparar datos**. Hay varios pasos en el proceso de preparación de datos. Podrías necesitar compilar datos y normalizarlos si provienen de fuentes diversas. Puedes mejorar la calidad y cantidad de los datos mediante varios métodos, como convertir cadenas en números (como hacemos en [Clustering](../../5-Clustering/1-Visualize/README.md)). También podrías generar nuevos datos basados en los originales (como hacemos en [Clasificación](../../4-Classification/1-Introduction/README.md)). Puedes limpiar y editar los datos (como haremos antes de la lección de [Aplicación Web](../../3-Web-App/README.md)). Finalmente, podrías necesitar aleatorizarlos y mezclarlos, dependiendo de tus técnicas de entrenamiento.

✅ Después de recopilar y procesar tus datos, tómate un momento para ver si su forma te permitirá abordar tu pregunta. Puede ser que los datos no funcionen bien en tu tarea, como descubrimos en nuestras lecciones de [Clustering](../../5-Clustering/1-Visualize/README.md).

### Características y Objetivo

Una [característica](https://www.datasciencecentral.com/profiles/blogs/an-introduction-to-variable-and-feature-selection) es una propiedad medible de tus datos. En muchos conjuntos de datos, se expresa como un encabezado de columna como 'fecha', 'tamaño' o 'color'. Tu variable de característica, usualmente representada como `X` en el código, representa la variable de entrada que se usará para entrenar el modelo.

Un objetivo es aquello que estás tratando de predecir. El objetivo, usualmente representado como `y` en el código, representa la respuesta a la pregunta que estás tratando de hacer a tus datos: en diciembre, ¿qué **color** de calabazas será el más barato? En San Francisco, ¿qué vecindarios tendrán el mejor **precio** de bienes raíces? A veces, el objetivo también se denomina atributo de etiqueta.

### Selección de tu variable de característica

🎓 **Selección de características y extracción de características** ¿Cómo sabes qué variable elegir al construir un modelo? Probablemente pasarás por un proceso de selección de características o extracción de características para elegir las variables correctas para el modelo más eficiente. Sin embargo, no son lo mismo: "La extracción de características crea nuevas características a partir de funciones de las características originales, mientras que la selección de características devuelve un subconjunto de las características." ([fuente](https://wikipedia.org/wiki/Feature_selection))

### Visualiza tus datos

Un aspecto importante del conjunto de herramientas de un científico de datos es el poder de visualizar datos utilizando varias bibliotecas excelentes como Seaborn o MatPlotLib. Representar tus datos visualmente podría permitirte descubrir correlaciones ocultas que puedes aprovechar. Tus visualizaciones también podrían ayudarte a descubrir sesgos o datos desequilibrados (como descubrimos en [Clasificación](../../4-Classification/2-Classifiers-1/README.md)).

### Divide tu conjunto de datos

Antes de entrenar, necesitas dividir tu conjunto de datos en dos o más partes de tamaño desigual que aún representen bien los datos.

- **Entrenamiento**. Esta parte del conjunto de datos se ajusta a tu modelo para entrenarlo. Este conjunto constituye la mayoría del conjunto de datos original.
- **Prueba**. Un conjunto de prueba es un grupo independiente de datos, a menudo extraído de los datos originales, que utilizas para confirmar el rendimiento del modelo construido.
- **Validación**. Un conjunto de validación es un grupo más pequeño e independiente de ejemplos que utilizas para ajustar los hiperparámetros o la arquitectura del modelo para mejorarlo. Dependiendo del tamaño de tus datos y la pregunta que estás haciendo, podrías no necesitar construir este tercer conjunto (como señalamos en [Pronóstico de Series Temporales](../../7-TimeSeries/1-Introduction/README.md)).

## Construcción de un modelo

Usando tus datos de entrenamiento, tu objetivo es construir un modelo, o una representación estadística de tus datos, utilizando varios algoritmos para **entrenarlo**. Entrenar un modelo lo expone a los datos y le permite hacer suposiciones sobre patrones percibidos que descubre, valida y acepta o rechaza.

### Decide un método de entrenamiento

Dependiendo de tu pregunta y la naturaleza de tus datos, elegirás un método para entrenarlo. Al recorrer la [documentación de Scikit-learn](https://scikit-learn.org/stable/user_guide.html) - que usamos en este curso - puedes explorar muchas formas de entrenar un modelo. Dependiendo de tu experiencia, podrías tener que probar varios métodos diferentes para construir el mejor modelo. Es probable que pases por un proceso en el que los científicos de datos evalúan el rendimiento de un modelo alimentándolo con datos no vistos, verificando su precisión, sesgo y otros problemas que degradan la calidad, y seleccionando el método de entrenamiento más apropiado para la tarea en cuestión.

### Entrena un modelo

Con tus datos de entrenamiento, estás listo para 'ajustarlo' y crear un modelo. Notarás que en muchas bibliotecas de ML encontrarás el código 'model.fit' - es en este momento que envías tu variable de característica como un arreglo de valores (usualmente 'X') y una variable objetivo (usualmente 'y').

### Evalúa el modelo

Una vez que el proceso de entrenamiento esté completo (puede tomar muchas iteraciones, o 'épocas', para entrenar un modelo grande), podrás evaluar la calidad del modelo utilizando datos de prueba para medir su rendimiento. Estos datos son un subconjunto de los datos originales que el modelo no ha analizado previamente. Puedes imprimir una tabla de métricas sobre la calidad de tu modelo.

🎓 **Ajuste del modelo**

En el contexto del aprendizaje automático, el ajuste del modelo se refiere a la precisión de la función subyacente del modelo mientras intenta analizar datos con los que no está familiarizado.

🎓 **Subajuste** y **sobreajuste** son problemas comunes que degradan la calidad del modelo, ya que el modelo se ajusta demasiado poco o demasiado bien. Esto hace que el modelo haga predicciones demasiado alineadas o demasiado poco alineadas con sus datos de entrenamiento. Un modelo sobreajustado predice los datos de entrenamiento demasiado bien porque ha aprendido demasiado bien los detalles y el ruido de los datos. Un modelo subajustado no es preciso ya que no puede analizar con precisión ni sus datos de entrenamiento ni los datos que aún no ha 'visto'.

![modelo sobreajustado](../../../../translated_images/overfitting.1c132d92bfd93cb63240baf63ebdf82c30e30a0a44e1ad49861b82ff600c2b5c.es.png)
> Infografía por [Jen Looper](https://twitter.com/jenlooper)

## Ajuste de parámetros

Una vez que tu entrenamiento inicial esté completo, observa la calidad del modelo y considera mejorarlo ajustando sus 'hiperparámetros'. Lee más sobre el proceso [en la documentación](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters?WT.mc_id=academic-77952-leestott).

## Predicción

Este es el momento en el que puedes usar datos completamente nuevos para probar la precisión de tu modelo. En un entorno de ML 'aplicado', donde estás construyendo activos web para usar el modelo en producción, este proceso podría implicar recopilar entradas de usuario (por ejemplo, presionar un botón) para establecer una variable y enviarla al modelo para inferencia o evaluación.

En estas lecciones, descubrirás cómo usar estos pasos para preparar, construir, probar, evaluar y predecir - todos los gestos de un científico de datos y más, mientras avanzas en tu camino para convertirte en un ingeniero de ML 'full stack'.

---

## 🚀Desafío

Dibuja un diagrama de flujo que refleje los pasos de un practicante de ML. ¿Dónde te ves ahora en el proceso? ¿Dónde predices que encontrarás dificultades? ¿Qué te parece fácil?

## [Cuestionario posterior a la lección](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/8/)

## Revisión y Autoestudio

Busca en línea entrevistas con científicos de datos que discutan su trabajo diario. Aquí tienes [una](https://www.youtube.com/watch?v=Z3IjgbbCEfs).

## Tarea

[Entrevista a un científico de datos](assignment.md)

---

**Descargo de responsabilidad**:  
Este documento ha sido traducido utilizando el servicio de traducción automática [Co-op Translator](https://github.com/Azure/co-op-translator). Si bien nos esforzamos por garantizar la precisión, tenga en cuenta que las traducciones automáticas pueden contener errores o imprecisiones. El documento original en su idioma nativo debe considerarse como la fuente autorizada. Para información crítica, se recomienda una traducción profesional realizada por humanos. No nos hacemos responsables de malentendidos o interpretaciones erróneas que puedan surgir del uso de esta traducción.