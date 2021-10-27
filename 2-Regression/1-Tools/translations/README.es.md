# Comience con Python y Scikit-learn para modelos de regresión

![Resumen de regresiones en un boceto](../../sketchnotes/ml-regression.png)

> Boceto de [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Cuestionario previo](https://jolly-sea-0a877260f.azurestaticapps.net/quiz/9/)
## Introducción

En estas cuatro lecciones, descubrirá como crear modelos de regresión. Discutiremos para que sirven estos en breve. Pero antes de hacer cualquier cosa, asegúrese de tener las herramientas adecuadas para comenzar el proceso!

En esta lección, aprenderá a:

- Configurar su computadora para tares locales de machine learning.
- Trabajar con cuadernos Jupyter.
- Usar Scikit-learn, incluida la instalación.
- Explorar la regressión lineal con un ejercicio práctico.

## Instalaciones y configuraciones.

[![Uso de Python con Visual Studio Code](https://img.youtube.com/vi/yyQM70vi7V8/0.jpg)](https://youtu.be/yyQM70vi7V8 "Uso de Python con Visual Studio Code")

> 🎥 Haga click en la imagen de arriba para ver un video: usando Python dentro de VS Code.

1. **Instale Python**. Asegúrese de que [Python](https://www.python.org/downloads/) esté instalado en su computadora. Utilizará Python para muchas tareas de ciencia de datos y machine learning. La mayoría de los sistemas informáticos ya incluyen una instalación de Python. También hay disponibles [paquetes de código de Python](https://code.visualstudio.com/learn/educators/installers?WT.mc_id=academic-15963-cxa) útiles para facilitar la configuración a algunos usuarios.

   Sin embargo algunos usos de Python requieren una versión del software, mientras otros requieren una versión diferente. Por esta razón, es útil trabajar dentro de un [entorno virtual](https://docs.python.org/3/library/venv.html).

2. **Instale Visual Studio Code**. Asegúrese de tener Visual Studio Code instalado en su computadora. Siga estas instrucciones para [instalar Visual Studio Code](https://code.visualstudio.com/) para la instalación básica. Va a utilizar Python en Visual Studio Code en este curso, por lo que es posible que desee repasar cómo [configurar Visual Studio Code](https://docs.microsoft.com/learn/modules/python-install-vscode?WT.mc_id=academic-15963-cxa) para el desarrollo en Python.

   > Siéntase cómodo con Python trabajando con esta colección de [módulos de aprendizaje](https://docs.microsoft.com/users/jenlooper-2911/collections/mp1pagggd5qrq7?WT.mc_id=academic-15963-cxa)

3. **Instale Scikit-learn**, siguiendo [estas instrucciones](https://scikit-learn.org/stable/install.html). Dado que debe asegurarse de usar Python3, se recomienda que use un entorno virtual. Tenga en cuenta que si está instalando esta biblioteca en una Mac M1, hay instrucciones especiales en la página vinculada arriba.

1. **Instale Jupyter Notebook**. Deberá [instalar el paquete de Jupyter](https://pypi.org/project/jupyter/). 

## El entorno de creación de ML

Utilizará **cuadernos** para desarrollar su código en Python y crear modelos de machine learning. Este tipo de archivos es una herramienta común para científicos de datos, y se pueden identificar por su sufijo o extensión `.ipynb`.

Los cuadernos son un entorno interactivo que permiten al desarrollador codificar y agregar notas y escribir documentación sobre el código lo cual es bastante útil para proyectos experimentales u orientados a la investigación.
### Ejercicio - trabajar con un cuaderno

En esta carpeta, encontrará el archivo _notebook.ipynb_. 

1. Abra _notebook.ipynb_ en Visual Studio Code.

Un servidor de Jupyter comenzará con Python 3+ iniciado. Encontrará áreas del cuaderno que se pueden ejecutar, fragmentos de código. Puede ejecutar un bloque de código seleccionando el icono que parece un botón de reproducción.

1. Seleccione el icono `md` y agregue un poco de _markdown_, y el siguiente texto **# Welcome to your notebook**.

   A continuación, agregue algo de código Python. 

1. Escriba **print('hello notebook')** en el bloque de código.
1. Seleccione la flecha para ejecutar el código.

   Debería ver impresa la declaración:

    ```output
    hello notebook
    ```

![VS Code con un cuaderno abierto](../images/notebook.jpg)

Puede intercalar su código con comentarios para autodocumentar el cuaderno.

✅ Piense por un minuto en cuán diferente es el entorno de trabajo de un desarrollador web en comparación con el de un científico de datos.

## En funcionamiento con Scikit-learn

Ahora que Python está configurado en un entorno local, y se siente cómo con los cuadernos de Jupyter, vamos a sentirnos igualmente cómodos con Scikit-learn (pronuncie `sci` como en `science`). Scikit-learn proporciona una [API extensa](https://scikit-learn.org/stable/modules/classes.html#api-ref) para ayudarlo a realizar tares de ML.

Según su [sitio web](https://scikit-learn.org/stable/getting_started.html), "Scikit-learn es una biblioteca de machine learning de código abierto que admite el aprendizaje supervisado y no supervisado. También proporciona varias herramientas para el ajuste de modelos, preprocesamiento de datos, selección y evaluación de modelos, y muchas otras utilidades."

En este curso, utilizará Scikit-learn y otras herramientas para crear modelos de machine learning para realizar lo que llamamos tareas de 'machine learning tradicional'. Hemos evitado deliberadamente las redes neuronales y el _deep learning_, ya que se tratarán mejor en nuestro próximo plan de estudios 'IA para principiantes'. 

Scikit-learn hace que sea sencillo construir modelos y evaluarlos para su uso. Se centra principalmente en el uso de datos numéricos y contiene varios conjuntos de datos listos para usar como herramientas de aprendizaje. También incluye modelos prediseñados para que los estudiantes lo prueben. Exploremos el proceso de cargar datos preempaquetados y el uso de un primer modelo de estimador integrado con Scikit-learn con algunos datos básicos.

## Ejercicio - su primer cuaderno de Scikit-learn

> Este tutorial se insipiró en el [ejemplo de regresión lineal](https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py) en el sitio web de Scikit-learn's.

En el archivo _notebook.ipynb_ asociado a esta lección, borré todas las celdas presionando el icono 'papelera'.

En esta sección, trabajará con un pequeño conjunto de datos sobre la diabetes que está integrado con Scikit-learn con fines de aprendizaje. Imagínese que quisiera probar un tratamiento para pacientes diabéticos. Los modelos de Machine Learning, pueden ayudarlo a determinar que pacientes responderían mejor al tratamiento, en función  de combinaciones de variables. Incluso un modelo de regresión muy básico, cuando se visualiza, puede mostrar información sobre variables que le ayudarían en sus ensayos clínicos teóricos.

✅ Hay muchos tipos de métodos de regresión y el que elija dependerá de las respuestas que esté buscando. Si desea predecir la altura probable de una persona de una edad determinada, utlizaría la regresión lineal, ya que busca un **valor numérico**. Si está interesado en descubrir si un tipo de cocina puede considerarse vegano o no, está buscando una **asignación de categoría**, por lo que utilizaría la regresión logística. Más adelante aprenderá más sobre la regresión logística. Piense un poco en algunas preguntas que puede hacer a los datos y cuáles de estos métodos sería más apropiado.

Comencemos con esta tarea.

### Importar bibliotecas

Para esta tarea importaremos algunas librerías:

- **matplotlib**. Es una [herramienta gráfica](https://matplotlib.org/) útil y la usaremos para crear un diagrama de líneas.
- **numpy**. [numpy](https://numpy.org/doc/stable/user/whatisnumpy.html) es una librería útil para manejar datos numéricos en Python.
- **sklearn**. Esta es la librería Scikit-learn.

Importar algunas librerías para ayudarte con tus tareas.

1. Agrege importaciones escribiendo el siguiente código:

   ```python
   import matplotlib.pyplot as plt
   import numpy as np
   from sklearn import datasets, linear_model, model_selection
   ```

Arriba estás importando `matplottlib`, `numpy` y estás importando `datasets`, `linear_model` y `model_selection` de `sklearn`. `model_selection` se usa para dividir datos en conjuntos de entrenamiento y de prueba.

### El conjunto de datos de diabetes

El [conjunto de datos de diabetes](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) incluye 442 muestras de datos sobre la diabetes, con 10 variables de características, algunas de las cuales incluyen:

edad: edad en años.
bmi: índice de masa corporal.
bp: presión arterial promedio.
s1 tc: Células-T (un tipo de glóbulos blancos).

✅ Este conjunto de datos incluye el concepto de sexo como una variable característica importante para la investigación sobre la diabetes. Piense un poco en cómo categorizaciones como ésta podrían excluir a ciertas partes de una población de los tratamientos.

Ahora cargue los datos X e y.

> 🎓 Recuerde, esto es aprendizeje supervisado, y necesitamos un objetivo llamado 'y'.

En una nueva celda de código, cargue el conjunto de datos de diabetes llamando `load_diabetes()`. La entrada `return_X_y=True` indica que `X` será una matriz de datos, y `y` será el objetivo de regresión. 

1. Agregue algunos comandos de impresión para mostrar la forma de la matriz de datos y su primer elemento:

    ```python
    X, y = datasets.load_diabetes(return_X_y=True)
    print(X.shape)
    print(X[0])
    ```

    Lo que recibe como respuesta es una tupla. Lo que está haciendo es asignar los dos primeros valores de la tupla a `X` y `y` respectivamente. Más información [sobre tuplas](https://wikipedia.org/wiki/Tuple).

    Puede ver que estos datos tienen 442 elementos en forma de matrices de 10 elementos:
    

    ```text
    (442, 10)
    [ 0.03807591  0.05068012  0.06169621  0.02187235 -0.0442235  -0.03482076
    -0.04340085 -0.00259226  0.01990842 -0.01764613]
    ```

    ✅ Piense un poco en la relación entre los datos y el objetivo de la regresión. La regresión lineal predice relaciones entre la característica X y la variable objetivo y. ¿Puede encontrar el [objetivo](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset) para el cojunto de datos de diabetes en la documentación? ¿Qué está demostrando este conjunto de datos dado ese objetivo?

2. A continuación, seleccione una parte de este conjunto de datos para graficarlos colocándolos en una nueva matriz utilizando la función `newaxis` de _numpy_. Vamos a utilizar una regresión lineal para generar una línea entre los valores de estos datos, según un patrón que determine.

   ```python
   X = X[:, np.newaxis, 2]
   ```

   ✅ En cualquier momento, imprima los datos para comprobar su forma.

3. Ahora que tiene los datos listos para graficarlos, puede ver si una máquina puede ayudar a determinar una división lógica entre los númnero en este conjunto de datos. Para hacer esto, necesita dividir los datos (X) y el objetivo (y) en conjunto de datos de prueba y entrenamiento. Scikit-learn tiene una forma sencilla de hacer esto; puede dividir sus datos de prueba en un punto determinado.

   ```python
   X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33)
   ```

4. Ahora está listo para entrenar su modelo! Cargue el modelo de regresión lineal y entrénelo con sus datos de entrenamiento X e y usando `model.fit()`:

    ```python
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    ```

    ✅ `model.fit()` es una función que verá en muchas bibliotecas de ML como TensorFlow

5. Luego, cree una predicción usando datos de prueba, usando la función `predict()`. Esto se utilizará para trazar la línea entre los grupos de datos.

    ```python
    y_pred = model.predict(X_test)
    ```

6. Ahora es el momento de mostrar los datos en una gráfica. Matplotlib es una herramienta muy útil para esta tarea. Cree una gráfica de dispersión de todos los datos de prueba X e y, y use la predicción para dibujar una línea en el lugar más apropiado, entre las agrupaciones de datos del modelo.

    ```python
    plt.scatter(X_test, y_test,  color='black')
    plt.plot(X_test, y_pred, color='blue', linewidth=3)
    plt.show()
    ```

   ![un diagrama de dispersión que muestra puntos de datos sobre la diabetes](./images/scatterplot.png)

   ✅ Piense un poco sobre lo que está pasando aquí. Una línea recta atraviesa muchos pequeños puntos de datos, pero ¿qué está haciendo excactamente? ¿Puede ver cómo debería poder usar esta línea para predecir dónde debe encajar un punto de datos nuevo y no visto en relación con el eje y del gráfico? Intente poner en palabras el uso práctico de este modelo.

Felicitaciones, construiste tu primer modelo de regresión lineal, creaste una predicción con él y lo mostraste en una gráfica!

---
## Desafío

Grafique una variable diferente de este conjunto de datos. Sugerencia: edite esta linea: `X = X[:, np.newaxis, 2]`. Dado el objetivo de este conjunto de datos,¿qué puede descubrir sobre la progresión de la diabetes?
## [Cuestionario posterior a la conferencia](https://jolly-sea-0a877260f.azurestaticapps.net/quiz/10/)

## Revisión y autoestudio

En este tutorial, trabajó con regresión lineal simple, en lugar de regresión lineal univariante o múltiple. Lea un poco sobre las diferencias entre estos métodos o eche un vistazo a [este video](https://www.coursera.org/lecture/quantifying-relationships-regression-models/linear-vs-nonlinear-categorical-variables-ai2Ef)

Lea más sobre el concepto de regresión lineal y piense que tipo de preguntas se pueden responder con esta técnica.Tome este [tutorial](https://docs.microsoft.com/learn/modules/train-evaluate-regression-models?WT.mc_id=academic-15963-cxa) para profundizar su comprensión.

## Asignación 

[Un conjunto de datos diferentes](assignment.md)
