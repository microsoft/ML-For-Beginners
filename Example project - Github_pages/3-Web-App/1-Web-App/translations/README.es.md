# Construye una aplicación web usando un modelo de aprendizaje automático

En esta lección, entrenarás un modelo de aprendizaje automático sobre un conjunto de datos que está fuera de este mundo: _avistamiento de OVNIs durante el siglo pasado_, proporcionados por la base de datos de NUFORC.

Aprenderás:

- Cómo hacer 'pickle' a un modelo entrenado
- Cómo usar ese modelo en una aplicación Flask

Continuaremos nuestro uso de notebooks para limpiar los datos y entrenar nuestro modelo, pero puedes llevar el proceso un paso más allá explorando el uso de un modelo 'en la naturaleza', por así decirlo: en una aplicación web.

Para hacer esto, necesitas construir una aplicación web usando Flask.

## [Examen previo a la lección](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/17?loc=es)

## Construyendo una aplicación

Existen muchas formas de construir aplicaciones web para consumir modelos de aprendizaje automático. Tu arquitectura web podría influir en la forma que tu modelo es entrenado. Imagina que estás trabajando en un negocio donde el grupo de ciencia de datos ha entrenado un modelo que quieren uses en una aplicación.

### Consideraciones

Hay muchas preguntas que necesitas realizar:

- **¿Es una aplicación web o móvil?** Si estás construyendo una aplicación móvil o necesitas uar el modelo en un contexto de IoT, podrías usar [TensorFlow Lite](https://www.tensorflow.org/lite/) y usar el modelo en una applicación Android o iOS.
- **¿Dónde residirá el modelo?** ¿En la nube o de forma local?
- **Soporte fuera de línea.** ¿La aplicación trabaja en modo fuera de línea?
- **¿Qué tecnología se usó para entrenar al modelo?** La tecnología elegida puede influir en las herramientas que necesitas utilizar.
    - **Uso de TensorFlow.** Si estás entrenando un modelo usando TensorFlow, por ejemplo, ese ecosistema proporciona la capacidad de convertir un modelo de TensorFlow para su uso en una aplicación web usando [TensorFlow.js](https://www.tensorflow.org/js/).
    - **Uso de PyTorch.** Si estás construyendo un modelo usando una librería como [PyTorch](https://pytorch.org/), tienes la opción de exportarlo en formato [ONNX](https://onnx.ai/) (Open Neural Network Exchange) para usarlo en aplicaciones web de javascript que puedan usar el entorno de ejecución [Onnx Runtime](https://www.onnxruntime.ai/). Esta opción será explorada en una futura lección para un modelo entrenado Scikit-learn.
    - **Uso de Lobe.ai o Azure Custom Vision.** Si estás usando un sistema de aprendizaje automático SaaS (Software as a Service) como lo es [Lobe.ai](https://lobe.ai/) o [Azure Custom Vision](https://azure.microsoft.com/services/cognitive-services/custom-vision-service/?WT.mc_id=academic-77952-leestott) para entrenar un modelo, este tipo de software proporciona formas de exportar el modelo a diversas plataformas, incluyendo el construir una API a medida para que esta sea consultada en la nube por tu aplicación en línea.

También tienes la oportunidad de construir una plicación web completamente en Flask que será capaz de entrenar el propio modelo en un navegador web. Esto también puede ser realizado usando TensorFlow.js en un contexto JavaScript.

Para nuestros propósitos, ya que hemos estado trabajando con notebooks basados en Python, exploremos los pasos que necesitas realizar para exportar un modelo entrenado desde un notebook a un formato legible para una aplicación web construida en Python.

## Herramientas

Para esta tarea, necesitas dos herramientas: Flask y Pickle, ambos corren en Python.

✅ ¿Qué es [Flask](https://palletsprojects.com/p/flask/)? Definido como un 'micro-framework' por sus creadores, Flask proporciona las características básicas de los frameworks web usando Python y un motor de plantillas para construir páginas web. Da un vistazo a [este módulo de aprendizaje](https://docs.microsoft.com/learn/modules/python-flask-build-ai-web-app?WT.mc_id=academic-77952-leestott) para practicar construir con Flask.

✅ ¿Qué es [Pickle](https://docs.python.org/3/library/pickle.html)? Pickle 🥒 es un módulo de Python que serializa y deserializa estructura de objetos Python. Cuando conviertes un modelo en 'pickle', serializas o aplanas su estructura para su uso en la web. Sé cuidadoso: Pickle no es intrínsecamente seguro, por lo que debes ser cuidadoso si solicitaste hacer 'un-pickle' en un archivo. Un archivo hecho pickle tiene el sufijo `.pkl`.

## Ejercicio - limpia tus datos

En esta lección usarás datos de 80,000 avistamientos de OVNIs, recopilados por [NUFORC](https://nuforc.org) (El centro nacional de informes OVNI). Estos datos tienen algunas descripciones interesantes de avistamientos OVNI, por ejemplo:


- **Descripción larga del ejemplo.** "Un hombre emerge de un haz de luz que brilla en un campo de hierba por la noche y corre hacia el estacionamiento de Texas Instruments".
- **Descripción corta del ejemplo.** "las luces nos persiguieron".

La hoja de cálculo [ufos.csv](../data/ufos.csv) incluye columnas acerca de los campos `city`, `state` y `country` donde ocurrió el avistamiento, la forma (`shape`) y su latitud (`latitude`) y ubicación (`latitude` y `longitude`).

En el [notebook](../notebook.ipynb) en blanco incluído en esta lección:

1. Importa `pandas`, `matplotlib`, y `numpy` como lo hiciste en lecciones anteriores e importa la hoja de cálculo ufos. Puedes dar un vistazo al conjunto de datos de ejemplo:

    ```python
    import pandas as pd
    import numpy as np
    
    ufos = pd.read_csv('./data/ufos.csv')
    ufos.head()
    ```

1. Convierte los datos de OVNIs en un pequeño dataframe con nuevos títulos. Revisa los valores únicos en el campo `Country`.

    ```python
    ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'],'Latitude': ufos['latitude'],'Longitude': ufos['longitude']})
    
    ufos.Country.unique()
    ```

1. Ahora, puedes reducir la cantidad de datos que necesitamos manejar eliminando cualquier valor nulo e importando únicamente los avistamientos entre 1 y 60 segundos:

    ```python
    ufos.dropna(inplace=True)
    
    ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]
    
    ufos.info()
    ```

1. Importa la librería `LabelEncoder` de Scikit-learn para convertir los valores de texto de los países a número:

    ✅ LabelEncoder codifica los datos alfabéticamente

    ```python
    from sklearn.preprocessing import LabelEncoder
    
    ufos['Country'] = LabelEncoder().fit_transform(ufos['Country'])
    
    ufos.head()
    ```

    Tus datos deberían verse así:

    ```output
    	Seconds	Country	Latitude	Longitude
    2	20.0	3		53.200000	-2.916667
    3	20.0	4		28.978333	-96.645833
    14	30.0	4		35.823889	-80.253611
    23	60.0	4		45.582778	-122.352222
    24	3.0		3		51.783333	-0.783333
    ```

## Ejercicio - construye tu modelo

Ahora puedes prepararte para entrenar un modelo dividiendo los datos entre los grupos de entrenamiento y pruebas.

1. Selecciona las tres características que quieres entrenar en tu vector X, y el vector Y será `Country`. Quieres ser capaz de introducir `Seconds`, `Latitude` y `Longitude` y obtener un id de país de regreso.

    ```python
    from sklearn.model_selection import train_test_split
    
    Selected_features = ['Seconds','Latitude','Longitude']
    
    X = ufos[Selected_features]
    y = ufos['Country']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    ```

1. Entrena tu modelo usando regresión logística:

    ```python
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    print(classification_report(y_test, predictions))
    print('Predicted labels: ', predictions)
    print('Accuracy: ', accuracy_score(y_test, predictions))
    ```

La precisión no es mala **(alrededor del 95%)**, como era de esperar, ya que `Country` y `Latitude/Longitude` se correlacionan.

El modelo que creaste no es muy revolucionario como deberías ser capaz de inferir un país (`Country`) por su latitud y longitud (`Latitude`, `Longitude`), pero es un buen ejercicio intentar entrenar desde datos en crudo que ya limpiaste, exportaste y luego usa este modelo en una aplicación web.

## Ejercicio - Haz 'pickle' a tu modelo

Ahora, ¡es momento de hacer _pickle_ a tu modelo! Puedes hacer eso con pocas líneas de código. Una vez la hiciste _pickle_, carga tu modelo serializado (pickled) y pruébalo constra un arreglo de datos de muestra que contenga los valores para segundos, latitud y longitud.

```python
import pickle
model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model = pickle.load(open('ufo-model.pkl','rb'))
print(model.predict([[50,44,-12]]))
```

El modelo regresa **'3'**, lo cual es el código de país para el Reino Unido (UK). ¡Sorprendente! 👽

## Ejercicio - Construye una aplicación Flask

Ahora puedes construir una aplicación Flask para llamara tu modelo y regresar resultados similares, pero de una forma visualmente más agradable.

1. Comienza por crear un directorio llamado **web-app** junto al archivo _notebook.ipynb_  donde reside el archivo _ufo-model.pkl_.

1. En ese directorio crea 3 directorios más: **static**, con un directorio **css** dentro de el, y **templates**.  Ahora tienes la siguiente estructura de directorios:

    ```output
    web-app/
      static/
        css/
      templates/
    notebook.ipynb
    ufo-model.pkl
    ```

    ✅ Consulta el directorio de la solución para una vista de la aplicación terminada.

1. El primer archivo a crear en el directorio _web-app_ es el archivo **requirements.txt**. Así como _package.json_ en una aplicación JavaScript, este archivo lista las dependencias requeridas por la aplicación. En **requirements.txt** agrega las líneas:

    ```text
    scikit-learn
    pandas
    numpy
    flask
    ```

1. Ahora, ejecuta este archivo navegando a _web-app_:

    ```bash
    cd web-app
    ```

1. Escribe en tu terminal `pip install`, para instalar las librerías listadas en _requirements.txt_:

    ```bash
    pip install -r requirements.txt
    ```

1. Ahora, estás listo para crear 3 archivos más y así terminar la aplicación:

    1. Crea el archivo **app.py** en la raíz.
    2. Crea el archivo **index.html** dentro del directorio _templates_.
    3. Crea el archivo **styles.css** dentro del directorio _static/css_.

1. Construye el archivo _styles.css_ file con algunos estilos:

    ```css
    body {
    	width: 100%;
    	height: 100%;
    	font-family: 'Helvetica';
    	background: black;
    	color: #fff;
    	text-align: center;
    	letter-spacing: 1.4px;
    	font-size: 30px;
    }
    
    input {
    	min-width: 150px;
    }
    
    .grid {
    	width: 300px;
    	border: 1px solid #2d2d2d;
    	display: grid;
    	justify-content: center;
    	margin: 20px auto;
    }
    
    .box {
    	color: #fff;
    	background: #2d2d2d;
    	padding: 12px;
    	display: inline-block;
    }
    ```

1. Lo siguiente es constuir el archivo _index.html_:

    ```html
    <!DOCTYPE html>
    <html>
      <head>
        <meta charset="UTF-8">
        <title>🛸 UFO Appearance Prediction! 👽</title>
        <link rel="stylesheet" href="{{ url_for('static', filename='css/styles.css') }}">
      </head>
    
      <body>
        <div class="grid">
    
          <div class="box">
    
            <p>According to the number of seconds, latitude and longitude, which country is likely to have reported seeing a UFO?</p>
    
            <form action="{{ url_for('predict')}}" method="post">
              <input type="number" name="seconds" placeholder="Seconds" required="required" min="0" max="60" />
              <input type="text" name="latitude" placeholder="Latitude" required="required" />
              <input type="text" name="longitude" placeholder="Longitude" required="required" />
              <button type="submit" class="btn">Predict country where the UFO is seen</button>
            </form>
    
            <p>{{ prediction_text }}</p>
    
          </div>
    
        </div>
    
      </body>
    </html>
    ```

    Echa un vistazo a la plantilla en este archivo. Nota la sitaxis 'mustache' alrededor de las variables que serán proporcionadas por la aplicación, como el texto de predicción `{{}}`. También hay un formulario que publica una predicción a la ruta `/predict`.

    Finalmente, estás listo para construir el archivo python que maneja el consumo de el modelo y la pantalla de predicciones:

1. En `app.py` agrega:

    ```python
    import numpy as np
    from flask import Flask, request, render_template
    import pickle
    
    app = Flask(__name__)
    
    model = pickle.load(open("./ufo-model.pkl", "rb"))
    
    
    @app.route("/")
    def home():
        return render_template("index.html")
    
    
    @app.route("/predict", methods=["POST"])
    def predict():
    
        int_features = [int(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        prediction = model.predict(final_features)
    
        output = prediction[0]
    
        countries = ["Australia", "Canada", "Germany", "UK", "US"]
    
        return render_template(
            "index.html", prediction_text="Likely country: {}".format(countries[output])
        )
    
    
    if __name__ == "__main__":
        app.run(debug=True)
    ```

    > 💡 Tip: Cuando agregas [`debug=True`](https://www.askpython.com/python-modules/flask/flask-debug-mode) mientras ejecutas la aplicación web usando Flask, cualquier cambio que realices a tu aplicación será reflejado inmediatamente sin la necesidad de reiniciar el servidor. ¡Ten cuidado! No actives este modo en una aplicación en producción.

Si ejecutas `python app.py` o `python3 app.py` - tu servidor web inicia, localmente, y puedes llenar un pequeño formulario para obtener una respuesta a tu pregunta en cuestión acerca de ¡dónde han avistado OVNIs!

Antes de hacerlo, echa un vistazo a las partes de `app.py`:

1. Primero, las dependencias son cargadas y la aplicación inicia.
2. Luego, el modelo es importado.
3. Lo siguiente, el archivo index.html es renderizado en la ruta principal.

En la ruta `/predict`, pasan muchas cosas cuando el formulario se publica:

1. Las variables del formulario son reunidas y convertidas a un arreglo de numpy. Luego estas son enviadas al modelo y se regresa una predicción.
2. Los países que queremos se muestren son re-renderizados como texto legible de su código de país previsto, y ese valor es enviado de vuelta a index.html para ser renderizado en la plantilla.

Usando un modelo de esta forma, con Flask y un modelo hecho pickled, es relativamente sencillo. La cosa más difícil es entender qué forma tienen los datos que deben ser enviados al modelo para obtener una predicción. Todo eso depende en cómo fue entrenado el modelo. Este tiene 3 puntos de datos como entrada para así obtener una predicción.

En un entorno profesional, puedes ver cómo la buena comunicación es necesaria entre las personas las cuales entrenan el modelo y aquellas que lo consumen en una aplicación web o móvil. En nuestro caso, es una sola persona, ¡tú!

---

## 🚀 Desafío

En lugar de trabajar en un notebook e importar el modelo a una aplicación Flask, ¡podrías entrenar el modelo directo en la aplicación Flask! Intenta convertir tu código Python en el notebook, quizá después que tus datos sean limpiados, para entrenar el modelo desde la aplicación en una ruta llamada `train`. ¿Cuáles son los pros y contras de seguir este método?

## [Examen posterior a la lección](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/18?loc=es)

## Revisión y autoestudio

Hay muchas formas de construir una aplicación web para consumir modelos de aprendizaje automático. Haz una lista de las formas en que podrías usar JavaScript o Python para construir una aplicación web para aprovechar el apredizaje automático. Considera la arquitectura: ¿El modelo debería estar en la aplicación o vivir en la nube? Si es lo segundo, ¿Cómo lo accederías? Dibuja un modelo de arquitectura para una solución web de aprendizaje automático aplicada.

## Asignación

[Prueba un modelo diferente](assignment.es.md)
