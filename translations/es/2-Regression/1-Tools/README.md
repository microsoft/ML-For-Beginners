<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "6b1cb0e46d4c5b747eff6e3607642760",
  "translation_date": "2025-09-03T22:34:01+00:00",
  "source_file": "2-Regression/1-Tools/README.md",
  "language_code": "es"
}
-->
# Comienza con Python y Scikit-learn para modelos de regresión

![Resumen de regresiones en un sketchnote](../../../../translated_images/ml-regression.4e4f70e3b3ed446e3ace348dec973e133fa5d3680fbc8412b61879507369b98d.es.png)

> Sketchnote por [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Cuestionario previo a la lección](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/9/)

> ### [¡Esta lección está disponible en R!](../../../../2-Regression/1-Tools/solution/R/lesson_1.html)

## Introducción

En estas cuatro lecciones, descubrirás cómo construir modelos de regresión. Hablaremos sobre para qué sirven en breve. Pero antes de hacer nada, ¡asegúrate de tener las herramientas adecuadas para comenzar el proceso!

En esta lección, aprenderás a:

- Configurar tu computadora para tareas locales de aprendizaje automático.
- Trabajar con Jupyter notebooks.
- Usar Scikit-learn, incluyendo su instalación.
- Explorar la regresión lineal con un ejercicio práctico.

## Instalaciones y configuraciones

[![ML para principiantes - Configura tus herramientas para construir modelos de aprendizaje automático](https://img.youtube.com/vi/-DfeD2k2Kj0/0.jpg)](https://youtu.be/-DfeD2k2Kj0 "ML para principiantes - Configura tus herramientas para construir modelos de aprendizaje automático")

> 🎥 Haz clic en la imagen de arriba para ver un breve video sobre cómo configurar tu computadora para ML.

1. **Instalar Python**. Asegúrate de que [Python](https://www.python.org/downloads/) esté instalado en tu computadora. Usarás Python para muchas tareas de ciencia de datos y aprendizaje automático. La mayoría de los sistemas informáticos ya incluyen una instalación de Python. También hay disponibles [Paquetes de Codificación de Python](https://code.visualstudio.com/learn/educators/installers?WT.mc_id=academic-77952-leestott) útiles para facilitar la configuración a algunos usuarios.

   Sin embargo, algunos usos de Python requieren una versión específica del software, mientras que otros requieren una versión diferente. Por esta razón, es útil trabajar dentro de un [entorno virtual](https://docs.python.org/3/library/venv.html).

2. **Instalar Visual Studio Code**. Asegúrate de tener Visual Studio Code instalado en tu computadora. Sigue estas instrucciones para [instalar Visual Studio Code](https://code.visualstudio.com/) para la instalación básica. Vas a usar Python en Visual Studio Code en este curso, por lo que podrías querer repasar cómo [configurar Visual Studio Code](https://docs.microsoft.com/learn/modules/python-install-vscode?WT.mc_id=academic-77952-leestott) para el desarrollo en Python.

   > Familiarízate con Python trabajando en esta colección de [módulos de aprendizaje](https://docs.microsoft.com/users/jenlooper-2911/collections/mp1pagggd5qrq7?WT.mc_id=academic-77952-leestott)
   >
   > [![Configura Python con Visual Studio Code](https://img.youtube.com/vi/yyQM70vi7V8/0.jpg)](https://youtu.be/yyQM70vi7V8 "Configura Python con Visual Studio Code")
   >
   > 🎥 Haz clic en la imagen de arriba para ver un video: usando Python dentro de VS Code.

3. **Instalar Scikit-learn**, siguiendo [estas instrucciones](https://scikit-learn.org/stable/install.html). Dado que necesitas asegurarte de usar Python 3, se recomienda que utilices un entorno virtual. Nota: si estás instalando esta biblioteca en una Mac M1, hay instrucciones especiales en la página enlazada arriba.

4. **Instalar Jupyter Notebook**. Necesitarás [instalar el paquete Jupyter](https://pypi.org/project/jupyter/).

## Tu entorno de autoría de ML

Vas a usar **notebooks** para desarrollar tu código en Python y crear modelos de aprendizaje automático. Este tipo de archivo es una herramienta común para los científicos de datos y se identifican por su sufijo o extensión `.ipynb`.

Los notebooks son un entorno interactivo que permite al desarrollador tanto codificar como agregar notas y escribir documentación alrededor del código, lo cual es bastante útil para proyectos experimentales o de investigación.

[![ML para principiantes - Configura Jupyter Notebooks para comenzar a construir modelos de regresión](https://img.youtube.com/vi/7E-jC8FLA2E/0.jpg)](https://youtu.be/7E-jC8FLA2E "ML para principiantes - Configura Jupyter Notebooks para comenzar a construir modelos de regresión")

> 🎥 Haz clic en la imagen de arriba para ver un breve video sobre este ejercicio.

### Ejercicio - trabajar con un notebook

En esta carpeta, encontrarás el archivo _notebook.ipynb_.

1. Abre _notebook.ipynb_ en Visual Studio Code.

   Se iniciará un servidor Jupyter con Python 3+. Encontrarás áreas del notebook que pueden ser `ejecutadas`, piezas de código. Puedes ejecutar un bloque de código seleccionando el ícono que parece un botón de reproducción.

2. Selecciona el ícono `md` y agrega un poco de markdown, y el siguiente texto **# Bienvenido a tu notebook**.

   Luego, agrega algo de código en Python.

3. Escribe **print('hello notebook')** en el bloque de código.
4. Selecciona la flecha para ejecutar el código.

   Deberías ver la declaración impresa:

    ```output
    hello notebook
    ```

![VS Code con un notebook abierto](../../../../translated_images/notebook.4a3ee31f396b88325607afda33cadcc6368de98040ff33942424260aa84d75f2.es.jpg)

Puedes intercalar tu código con comentarios para auto-documentar el notebook.

✅ Piensa por un momento en cómo es diferente el entorno de trabajo de un desarrollador web frente al de un científico de datos.

## Puesta en marcha con Scikit-learn

Ahora que Python está configurado en tu entorno local y te sientes cómodo con los notebooks de Jupyter, vamos a familiarizarnos con Scikit-learn (se pronuncia `sci` como en `science`). Scikit-learn proporciona una [API extensa](https://scikit-learn.org/stable/modules/classes.html#api-ref) para ayudarte a realizar tareas de ML.

Según su [sitio web](https://scikit-learn.org/stable/getting_started.html), "Scikit-learn es una biblioteca de aprendizaje automático de código abierto que admite aprendizaje supervisado y no supervisado. También proporciona varias herramientas para ajuste de modelos, preprocesamiento de datos, selección y evaluación de modelos, y muchas otras utilidades."

En este curso, usarás Scikit-learn y otras herramientas para construir modelos de aprendizaje automático para realizar lo que llamamos tareas de 'aprendizaje automático tradicional'. Hemos evitado deliberadamente redes neuronales y aprendizaje profundo, ya que están mejor cubiertos en nuestro próximo currículo 'AI para Principiantes'.

Scikit-learn hace que sea sencillo construir modelos y evaluarlos para su uso. Se centra principalmente en el uso de datos numéricos y contiene varios conjuntos de datos listos para usar como herramientas de aprendizaje. También incluye modelos preconstruidos para que los estudiantes los prueben. Vamos a explorar el proceso de cargar datos preempaquetados y usar un estimador para el primer modelo de ML con Scikit-learn con algunos datos básicos.

## Ejercicio - tu primer notebook con Scikit-learn

> Este tutorial fue inspirado por el [ejemplo de regresión lineal](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py) en el sitio web de Scikit-learn.

[![ML para principiantes - Tu primer proyecto de regresión lineal en Python](https://img.youtube.com/vi/2xkXL5EUpS0/0.jpg)](https://youtu.be/2xkXL5EUpS0 "ML para principiantes - Tu primer proyecto de regresión lineal en Python")

> 🎥 Haz clic en la imagen de arriba para ver un breve video sobre este ejercicio.

En el archivo _notebook.ipynb_ asociado a esta lección, elimina todas las celdas presionando el ícono de 'papelera'.

En esta sección, trabajarás con un pequeño conjunto de datos sobre diabetes que está integrado en Scikit-learn para fines de aprendizaje. Imagina que quisieras probar un tratamiento para pacientes diabéticos. Los modelos de aprendizaje automático podrían ayudarte a determinar qué pacientes responderían mejor al tratamiento, basándote en combinaciones de variables. Incluso un modelo de regresión muy básico, cuando se visualiza, podría mostrar información sobre variables que te ayudarían a organizar tus ensayos clínicos teóricos.

✅ Hay muchos tipos de métodos de regresión, y cuál elijas depende de la respuesta que estés buscando. Si quieres predecir la altura probable de una persona dada su edad, usarías regresión lineal, ya que estás buscando un **valor numérico**. Si estás interesado en descubrir si un tipo de cocina debería considerarse vegana o no, estás buscando una **asignación de categoría**, por lo que usarías regresión logística. Aprenderás más sobre regresión logística más adelante. Piensa un poco en algunas preguntas que puedes hacer a los datos y cuál de estos métodos sería más apropiado.

Vamos a comenzar con esta tarea.

### Importar bibliotecas

Para esta tarea, importaremos algunas bibliotecas:

- **matplotlib**. Es una herramienta útil para [gráficos](https://matplotlib.org/) y la usaremos para crear un gráfico de líneas.
- **numpy**. [numpy](https://numpy.org/doc/stable/user/whatisnumpy.html) es una biblioteca útil para manejar datos numéricos en Python.
- **sklearn**. Esta es la biblioteca [Scikit-learn](https://scikit-learn.org/stable/user_guide.html).

Importa algunas bibliotecas para ayudarte con tus tareas.

1. Agrega las importaciones escribiendo el siguiente código:

   ```python
   import matplotlib.pyplot as plt
   import numpy as np
   from sklearn import datasets, linear_model, model_selection
   ```

   Arriba estás importando `matplotlib`, `numpy` y estás importando `datasets`, `linear_model` y `model_selection` de `sklearn`. `model_selection` se usa para dividir datos en conjuntos de entrenamiento y prueba.

### El conjunto de datos de diabetes

El [conjunto de datos de diabetes](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) integrado incluye 442 muestras de datos sobre diabetes, con 10 variables de características, algunas de las cuales incluyen:

- age: edad en años
- bmi: índice de masa corporal
- bp: presión arterial promedio
- s1 tc: células T (un tipo de glóbulos blancos)

✅ Este conjunto de datos incluye el concepto de 'sexo' como una variable de característica importante para la investigación sobre diabetes. Muchos conjuntos de datos médicos incluyen este tipo de clasificación binaria. Piensa un poco en cómo categorizaciones como esta podrían excluir a ciertas partes de la población de los tratamientos.

Ahora, carga los datos X e y.

> 🎓 Recuerda, esto es aprendizaje supervisado, y necesitamos un objetivo 'y' nombrado.

En una nueva celda de código, carga el conjunto de datos de diabetes llamando a `load_diabetes()`. El parámetro `return_X_y=True` indica que `X` será una matriz de datos y `y` será el objetivo de regresión.

1. Agrega algunos comandos de impresión para mostrar la forma de la matriz de datos y su primer elemento:

    ```python
    X, y = datasets.load_diabetes(return_X_y=True)
    print(X.shape)
    print(X[0])
    ```

    Lo que estás obteniendo como respuesta es una tupla. Lo que estás haciendo es asignar los dos primeros valores de la tupla a `X` e `y` respectivamente. Aprende más [sobre tuplas](https://wikipedia.org/wiki/Tuple).

    Puedes ver que estos datos tienen 442 elementos organizados en matrices de 10 elementos:

    ```text
    (442, 10)
    [ 0.03807591  0.05068012  0.06169621  0.02187235 -0.0442235  -0.03482076
    -0.04340085 -0.00259226  0.01990842 -0.01764613]
    ```

    ✅ Piensa un poco sobre la relación entre los datos y el objetivo de regresión. La regresión lineal predice relaciones entre la característica X y la variable objetivo y. ¿Puedes encontrar el [objetivo](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) para el conjunto de datos de diabetes en la documentación? ¿Qué está demostrando este conjunto de datos, dado ese objetivo?

2. A continuación, selecciona una porción de este conjunto de datos para graficar seleccionando la tercera columna del conjunto de datos. Puedes hacerlo usando el operador `:` para seleccionar todas las filas y luego seleccionando la tercera columna usando el índice (2). También puedes cambiar la forma de los datos para que sean una matriz 2D, como se requiere para graficar, usando `reshape(n_rows, n_columns)`. Si uno de los parámetros es -1, la dimensión correspondiente se calcula automáticamente.

   ```python
   X = X[:, 2]
   X = X.reshape((-1,1))
   ```

   ✅ En cualquier momento, imprime los datos para verificar su forma.

3. Ahora que tienes los datos listos para ser graficados, puedes ver si una máquina puede ayudar a determinar una división lógica entre los números en este conjunto de datos. Para hacer esto, necesitas dividir tanto los datos (X) como el objetivo (y) en conjuntos de prueba y entrenamiento. Scikit-learn tiene una forma sencilla de hacer esto; puedes dividir tus datos de prueba en un punto dado.

   ```python
   X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33)
   ```

4. Ahora estás listo para entrenar tu modelo. Carga el modelo de regresión lineal y entrénalo con tus conjuntos de entrenamiento X e y usando `model.fit()`:

    ```python
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    ```

    ✅ `model.fit()` es una función que verás en muchas bibliotecas de ML como TensorFlow.

5. Luego, crea una predicción usando datos de prueba, utilizando la función `predict()`. Esto se usará para dibujar la línea entre los grupos de datos.

    ```python
    y_pred = model.predict(X_test)
    ```

6. Ahora es momento de mostrar los datos en un gráfico. Matplotlib es una herramienta muy útil para esta tarea. Crea un gráfico de dispersión de todos los datos de prueba X e y, y usa la predicción para dibujar una línea en el lugar más apropiado entre los grupos de datos del modelo.

    ```python
    plt.scatter(X_test, y_test,  color='black')
    plt.plot(X_test, y_pred, color='blue', linewidth=3)
    plt.xlabel('Scaled BMIs')
    plt.ylabel('Disease Progression')
    plt.title('A Graph Plot Showing Diabetes Progression Against BMI')
    plt.show()
    ```

   ![un gráfico de dispersión mostrando puntos de datos sobre diabetes](../../../../translated_images/scatterplot.ad8b356bcbb33be68d54050e09b9b7bfc03e94fde7371f2609ae43f4c563b2d7.es.png)
✅ Piensa un poco en lo que está sucediendo aquí. Una línea recta está atravesando muchos pequeños puntos de datos, pero ¿qué está haciendo exactamente? ¿Puedes ver cómo deberías poder usar esta línea para predecir dónde debería encajar un nuevo punto de datos no visto en relación con el eje y del gráfico? Intenta poner en palabras el uso práctico de este modelo.

¡Felicidades, construiste tu primer modelo de regresión lineal, creaste una predicción con él y la mostraste en un gráfico!

---
## 🚀Desafío

Grafica una variable diferente de este conjunto de datos. Pista: edita esta línea: `X = X[:,2]`. Dado el objetivo de este conjunto de datos, ¿qué puedes descubrir sobre la progresión de la diabetes como enfermedad?

## [Cuestionario posterior a la lección](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/10/)

## Revisión y Autoestudio

En este tutorial, trabajaste con regresión lineal simple, en lugar de regresión univariada o múltiple. Lee un poco sobre las diferencias entre estos métodos, o echa un vistazo a [este video](https://www.coursera.org/lecture/quantifying-relationships-regression-models/linear-vs-nonlinear-categorical-variables-ai2Ef).

Lee más sobre el concepto de regresión y piensa en qué tipo de preguntas pueden responderse con esta técnica. Toma este [tutorial](https://docs.microsoft.com/learn/modules/train-evaluate-regression-models?WT.mc_id=academic-77952-leestott) para profundizar tu comprensión.

## Tarea

[Un conjunto de datos diferente](assignment.md)

---

**Descargo de responsabilidad**:  
Este documento ha sido traducido utilizando el servicio de traducción automática [Co-op Translator](https://github.com/Azure/co-op-translator). Aunque nos esforzamos por garantizar la precisión, tenga en cuenta que las traducciones automatizadas pueden contener errores o imprecisiones. El documento original en su idioma nativo debe considerarse como la fuente autorizada. Para información crítica, se recomienda una traducción profesional realizada por humanos. No nos hacemos responsables de malentendidos o interpretaciones erróneas que puedan surgir del uso de esta traducción.