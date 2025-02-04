# Crear una Aplicación Web para Usar un Modelo de ML

En esta lección, entrenarás un modelo de ML con un conjunto de datos fuera de este mundo: _Avistamientos de OVNIs durante el último siglo_, obtenidos de la base de datos de NUFORC.

Aprenderás:

- Cómo 'pickle' un modelo entrenado
- Cómo usar ese modelo en una aplicación Flask

Continuaremos usando notebooks para limpiar datos y entrenar nuestro modelo, pero puedes llevar el proceso un paso más allá explorando el uso de un modelo 'en el mundo real', por así decirlo: en una aplicación web.

Para hacer esto, necesitas construir una aplicación web usando Flask.

## [Cuestionario Previo a la Lección](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/17/)

## Construyendo una Aplicación

Hay varias formas de construir aplicaciones web para consumir modelos de machine learning. Tu arquitectura web puede influir en la forma en que tu modelo está entrenado. Imagina que estás trabajando en una empresa donde el grupo de ciencia de datos ha entrenado un modelo que quieren que uses en una aplicación.

### Consideraciones

Hay muchas preguntas que necesitas hacer:

- **¿Es una aplicación web o una aplicación móvil?** Si estás construyendo una aplicación móvil o necesitas usar el modelo en un contexto de IoT, podrías usar [TensorFlow Lite](https://www.tensorflow.org/lite/) y usar el modelo en una aplicación Android o iOS.
- **¿Dónde residirá el modelo?** ¿En la nube o localmente?
- **Soporte offline.** ¿La aplicación tiene que funcionar sin conexión?
- **¿Qué tecnología se utilizó para entrenar el modelo?** La tecnología elegida puede influir en las herramientas que necesitas usar.
    - **Usando TensorFlow.** Si estás entrenando un modelo usando TensorFlow, por ejemplo, ese ecosistema proporciona la capacidad de convertir un modelo de TensorFlow para su uso en una aplicación web usando [TensorFlow.js](https://www.tensorflow.org/js/).
    - **Usando PyTorch.** Si estás construyendo un modelo usando una biblioteca como [PyTorch](https://pytorch.org/), tienes la opción de exportarlo en formato [ONNX](https://onnx.ai/) (Open Neural Network Exchange) para su uso en aplicaciones web JavaScript que pueden usar el [Onnx Runtime](https://www.onnxruntime.ai/). Esta opción será explorada en una lección futura para un modelo entrenado con Scikit-learn.
    - **Usando Lobe.ai o Azure Custom Vision.** Si estás usando un sistema ML SaaS (Software como Servicio) como [Lobe.ai](https://lobe.ai/) o [Azure Custom Vision](https://azure.microsoft.com/services/cognitive-services/custom-vision-service/?WT.mc_id=academic-77952-leestott) para entrenar un modelo, este tipo de software proporciona formas de exportar el modelo para muchas plataformas, incluyendo la construcción de una API personalizada para ser consultada en la nube por tu aplicación en línea.

También tienes la oportunidad de construir una aplicación web completa con Flask que sería capaz de entrenar el modelo por sí misma en un navegador web. Esto también se puede hacer usando TensorFlow.js en un contexto JavaScript.

Para nuestros propósitos, ya que hemos estado trabajando con notebooks basados en Python, exploremos los pasos que necesitas seguir para exportar un modelo entrenado desde dicho notebook a un formato legible por una aplicación web construida en Python.

## Herramienta

Para esta tarea, necesitas dos herramientas: Flask y Pickle, ambas funcionan en Python.

✅ ¿Qué es [Flask](https://palletsprojects.com/p/flask/)? Definido como un 'micro-framework' por sus creadores, Flask proporciona las características básicas de los frameworks web usando Python y un motor de plantillas para construir páginas web. Echa un vistazo a [este módulo de aprendizaje](https://docs.microsoft.com/learn/modules/python-flask-build-ai-web-app?WT.mc_id=academic-77952-leestott) para practicar la construcción con Flask.

✅ ¿Qué es [Pickle](https://docs.python.org/3/library/pickle.html)? Pickle 🥒 es un módulo de Python que serializa y deserializa una estructura de objetos de Python. Cuando 'pickleas' un modelo, serializas o aplastas su estructura para su uso en la web. Ten cuidado: pickle no es intrínsecamente seguro, así que ten cuidado si te piden 'des-picklear' un archivo. Un archivo pickled tiene el sufijo `.pkl`.

## Ejercicio - limpiar tus datos

En esta lección usarás datos de 80,000 avistamientos de OVNIs, recopilados por [NUFORC](https://nuforc.org) (El Centro Nacional de Informes de OVNIs). Estos datos tienen algunas descripciones interesantes de avistamientos de OVNIs, por ejemplo:

- **Descripción larga de ejemplo.** "Un hombre emerge de un rayo de luz que brilla en un campo de hierba por la noche y corre hacia el estacionamiento de Texas Instruments".
- **Descripción corta de ejemplo.** "las luces nos persiguieron".

La hoja de cálculo [ufos.csv](../../../../3-Web-App/1-Web-App/data/ufos.csv) incluye columnas sobre el `city`, `state` y `country` donde ocurrió el avistamiento, el `shape` del objeto y su `latitude` y `longitude`.

En el [notebook](../../../../3-Web-App/1-Web-App/notebook.ipynb) en blanco incluido en esta lección:

1. importa `pandas`, `matplotlib`, y `numpy` como hiciste en lecciones anteriores e importa la hoja de cálculo de ufos. Puedes echar un vistazo a un conjunto de datos de muestra:

    ```python
    import pandas as pd
    import numpy as np
    
    ufos = pd.read_csv('./data/ufos.csv')
    ufos.head()
    ```

1. Convierte los datos de ufos a un pequeño dataframe con títulos nuevos. Revisa los valores únicos en el campo `Country`.

    ```python
    ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'],'Latitude': ufos['latitude'],'Longitude': ufos['longitude']})
    
    ufos.Country.unique()
    ```

1. Ahora, puedes reducir la cantidad de datos que necesitamos manejar eliminando cualquier valor nulo e importando solo avistamientos entre 1-60 segundos:

    ```python
    ufos.dropna(inplace=True)
    
    ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]
    
    ufos.info()
    ```

1. Importa la biblioteca `LabelEncoder` de Scikit-learn para convertir los valores de texto de los países a un número:

    ✅ LabelEncoder codifica datos alfabéticamente

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

## Ejercicio - construir tu modelo

Ahora puedes prepararte para entrenar un modelo dividiendo los datos en el grupo de entrenamiento y prueba.

1. Selecciona las tres características que quieres entrenar como tu vector X, y el vector y será el `Country`. You want to be able to input `Seconds`, `Latitude` and `Longitude` y obtén un id de país para devolver.

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

La precisión no está mal **(alrededor del 95%)**, no es sorprendente, ya que `Country` and `Latitude/Longitude` correlate.

The model you created isn't very revolutionary as you should be able to infer a `Country` from its `Latitude` and `Longitude`, pero es un buen ejercicio intentar entrenar desde datos en bruto que limpiaste, exportaste y luego usar este modelo en una aplicación web.

## Ejercicio - 'pickle' tu modelo

¡Ahora, es momento de _pickle_ tu modelo! Puedes hacerlo en unas pocas líneas de código. Una vez que esté _pickled_, carga tu modelo pickled y pruébalo contra un array de datos de muestra que contenga valores para segundos, latitud y longitud,

```python
import pickle
model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model = pickle.load(open('ufo-model.pkl','rb'))
print(model.predict([[50,44,-12]]))
```

El modelo devuelve **'3'**, que es el código de país para el Reino Unido. ¡Increíble! 👽

## Ejercicio - construir una aplicación Flask

Ahora puedes construir una aplicación Flask para llamar a tu modelo y devolver resultados similares, pero de una manera más visualmente agradable.

1. Comienza creando una carpeta llamada **web-app** junto al archivo _notebook.ipynb_ donde reside tu archivo _ufo-model.pkl_.

1. En esa carpeta crea tres carpetas más: **static**, con una carpeta **css** dentro, y **templates**. Ahora deberías tener los siguientes archivos y directorios:

    ```output
    web-app/
      static/
        css/
      templates/
    notebook.ipynb
    ufo-model.pkl
    ```

    ✅ Consulta la carpeta de solución para ver la aplicación terminada

1. El primer archivo que debes crear en la carpeta _web-app_ es el archivo **requirements.txt**. Al igual que _package.json_ en una aplicación JavaScript, este archivo lista las dependencias requeridas por la aplicación. En **requirements.txt** agrega las líneas:

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

1. En tu terminal escribe `pip install`, para instalar las bibliotecas listadas en _requirements.txt_:

    ```bash
    pip install -r requirements.txt
    ```

1. Ahora, estás listo para crear tres archivos más para terminar la aplicación:

    1. Crea **app.py** en la raíz.
    2. Crea **index.html** en el directorio _templates_.
    3. Crea **styles.css** en el directorio _static/css_.

1. Construye el archivo _styles.css_ con algunos estilos:

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

1. Luego, construye el archivo _index.html_:

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

    Echa un vistazo a la plantilla en este archivo. Nota la sintaxis 'mustache' alrededor de las variables que serán proporcionadas por la aplicación, como el texto de predicción: `{{}}`. There's also a form that posts a prediction to the `/predict` route.

    Finally, you're ready to build the python file that drives the consumption of the model and the display of predictions:

1. In `app.py` agrega:

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

    > 💡 Consejo: cuando agregas [`debug=True`](https://www.askpython.com/python-modules/flask/flask-debug-mode) while running the web app using Flask, any changes you make to your application will be reflected immediately without the need to restart the server. Beware! Don't enable this mode in a production app.

If you run `python app.py` or `python3 app.py` - your web server starts up, locally, and you can fill out a short form to get an answer to your burning question about where UFOs have been sighted!

Before doing that, take a look at the parts of `app.py`:

1. First, dependencies are loaded and the app starts.
1. Then, the model is imported.
1. Then, index.html is rendered on the home route.

On the `/predict` route, several things happen when the form is posted:

1. The form variables are gathered and converted to a numpy array. They are then sent to the model and a prediction is returned.
2. The Countries that we want displayed are re-rendered as readable text from their predicted country code, and that value is sent back to index.html to be rendered in the template.

Using a model this way, with Flask and a pickled model, is relatively straightforward. The hardest thing is to understand what shape the data is that must be sent to the model to get a prediction. That all depends on how the model was trained. This one has three data points to be input in order to get a prediction.

In a professional setting, you can see how good communication is necessary between the folks who train the model and those who consume it in a web or mobile app. In our case, it's only one person, you!

---

## 🚀 Challenge

Instead of working in a notebook and importing the model to the Flask app, you could train the model right within the Flask app! Try converting your Python code in the notebook, perhaps after your data is cleaned, to train the model from within the app on a route called `train`. ¿Cuáles son los pros y los contras de seguir este método?

## [Cuestionario Posterior a la Lección](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/18/)

## Revisión y Autoestudio

Hay muchas formas de construir una aplicación web para consumir modelos de ML. Haz una lista de las formas en que podrías usar JavaScript o Python para construir una aplicación web que aproveche el machine learning. Considera la arquitectura: ¿debería el modelo permanecer en la aplicación o vivir en la nube? Si es lo último, ¿cómo lo accederías? Dibuja un modelo arquitectónico para una solución web aplicada de ML.

## Tarea

[Prueba un modelo diferente](assignment.md)

        **Descargo de responsabilidad**: 
        Este documento ha sido traducido utilizando servicios de traducción automática basados en IA. Aunque nos esforzamos por lograr precisión, tenga en cuenta que las traducciones automáticas pueden contener errores o inexactitudes. El documento original en su idioma nativo debe considerarse la fuente autorizada. Para información crítica, se recomienda una traducción humana profesional. No somos responsables de ningún malentendido o interpretación errónea que surja del uso de esta traducción.