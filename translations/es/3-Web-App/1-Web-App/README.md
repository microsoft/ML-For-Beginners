<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "2680c691fbdb6367f350761a275e2508",
  "translation_date": "2025-09-03T23:45:27+00:00",
  "source_file": "3-Web-App/1-Web-App/README.md",
  "language_code": "es"
}
-->
# Construir una aplicación web para usar un modelo de ML

En esta lección, entrenarás un modelo de ML con un conjunto de datos que es de otro mundo: _avistamientos de OVNIs durante el último siglo_, obtenidos de la base de datos de NUFORC.

Aprenderás:

- Cómo 'pickle' un modelo entrenado
- Cómo usar ese modelo en una aplicación Flask

Continuaremos utilizando notebooks para limpiar datos y entrenar nuestro modelo, pero puedes llevar el proceso un paso más allá explorando cómo usar un modelo 'en el mundo real', por así decirlo: en una aplicación web.

Para hacer esto, necesitas construir una aplicación web usando Flask.

## [Cuestionario previo a la lección](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/17/)

## Construyendo una aplicación

Existen varias formas de construir aplicaciones web para consumir modelos de aprendizaje automático. La arquitectura de tu aplicación web puede influir en la forma en que se entrena tu modelo. Imagina que estás trabajando en una empresa donde el grupo de ciencia de datos ha entrenado un modelo que quieren que utilices en una aplicación.

### Consideraciones

Hay muchas preguntas que necesitas hacerte:

- **¿Es una aplicación web o una aplicación móvil?** Si estás construyendo una aplicación móvil o necesitas usar el modelo en un contexto de IoT, podrías usar [TensorFlow Lite](https://www.tensorflow.org/lite/) y utilizar el modelo en una aplicación Android o iOS.
- **¿Dónde residirá el modelo?** ¿En la nube o localmente?
- **Soporte sin conexión.** ¿La aplicación necesita funcionar sin conexión?
- **¿Qué tecnología se utilizó para entrenar el modelo?** La tecnología elegida puede influir en las herramientas que necesitas usar.
    - **Usando TensorFlow.** Si estás entrenando un modelo con TensorFlow, por ejemplo, ese ecosistema proporciona la capacidad de convertir un modelo de TensorFlow para usarlo en una aplicación web mediante [TensorFlow.js](https://www.tensorflow.org/js/).
    - **Usando PyTorch.** Si estás construyendo un modelo con una biblioteca como [PyTorch](https://pytorch.org/), tienes la opción de exportarlo en formato [ONNX](https://onnx.ai/) (Open Neural Network Exchange) para usarlo en aplicaciones web JavaScript que pueden utilizar el [Onnx Runtime](https://www.onnxruntime.ai/). Esta opción será explorada en una lección futura para un modelo entrenado con Scikit-learn.
    - **Usando Lobe.ai o Azure Custom Vision.** Si estás utilizando un sistema SaaS (Software como Servicio) de ML como [Lobe.ai](https://lobe.ai/) o [Azure Custom Vision](https://azure.microsoft.com/services/cognitive-services/custom-vision-service/?WT.mc_id=academic-77952-leestott) para entrenar un modelo, este tipo de software proporciona formas de exportar el modelo para muchas plataformas, incluyendo la construcción de una API personalizada para ser consultada en la nube por tu aplicación en línea.

También tienes la oportunidad de construir una aplicación web completa en Flask que pueda entrenar el modelo directamente en un navegador web. Esto también se puede hacer usando TensorFlow.js en un contexto de JavaScript.

Para nuestros propósitos, dado que hemos estado trabajando con notebooks basados en Python, exploremos los pasos que necesitas seguir para exportar un modelo entrenado desde dicho notebook a un formato legible por una aplicación web construida en Python.

## Herramienta

Para esta tarea, necesitas dos herramientas: Flask y Pickle, ambas ejecutándose en Python.

✅ ¿Qué es [Flask](https://palletsprojects.com/p/flask/)? Definido como un 'micro-framework' por sus creadores, Flask proporciona las características básicas de los frameworks web usando Python y un motor de plantillas para construir páginas web. Echa un vistazo a [este módulo de aprendizaje](https://docs.microsoft.com/learn/modules/python-flask-build-ai-web-app?WT.mc_id=academic-77952-leestott) para practicar la construcción con Flask.

✅ ¿Qué es [Pickle](https://docs.python.org/3/library/pickle.html)? Pickle 🥒 es un módulo de Python que serializa y deserializa una estructura de objetos de Python. Cuando 'pickleas' un modelo, serializas o aplanas su estructura para usarlo en la web. Ten cuidado: pickle no es intrínsecamente seguro, así que ten cuidado si te piden 'despicklear' un archivo. Un archivo pickleado tiene el sufijo `.pkl`.

## Ejercicio - limpia tus datos

En esta lección usarás datos de 80,000 avistamientos de OVNIs, recopilados por [NUFORC](https://nuforc.org) (El Centro Nacional de Reportes de OVNIs). Estos datos tienen algunas descripciones interesantes de avistamientos de OVNIs, por ejemplo:

- **Descripción larga de ejemplo.** "Un hombre emerge de un rayo de luz que brilla en un campo de hierba por la noche y corre hacia el estacionamiento de Texas Instruments".
- **Descripción corta de ejemplo.** "las luces nos persiguieron".

La hoja de cálculo [ufos.csv](../../../../3-Web-App/1-Web-App/data/ufos.csv) incluye columnas sobre la `ciudad`, `estado` y `país` donde ocurrió el avistamiento, la `forma` del objeto y su `latitud` y `longitud`.

En el [notebook](notebook.ipynb) en blanco incluido en esta lección:

1. Importa `pandas`, `matplotlib` y `numpy` como lo hiciste en lecciones anteriores e importa la hoja de cálculo de los OVNIs. Puedes echar un vistazo a un conjunto de datos de muestra:

    ```python
    import pandas as pd
    import numpy as np
    
    ufos = pd.read_csv('./data/ufos.csv')
    ufos.head()
    ```

1. Convierte los datos de los OVNIs a un pequeño dataframe con títulos nuevos. Revisa los valores únicos en el campo `Country`.

    ```python
    ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'],'Latitude': ufos['latitude'],'Longitude': ufos['longitude']})
    
    ufos.Country.unique()
    ```

1. Ahora, puedes reducir la cantidad de datos con los que necesitamos trabajar eliminando cualquier valor nulo e importando solo avistamientos entre 1-60 segundos:

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

## Ejercicio - construye tu modelo

Ahora puedes prepararte para entrenar un modelo dividiendo los datos en grupos de entrenamiento y prueba.

1. Selecciona las tres características que deseas entrenar como tu vector X, y el vector y será el `Country`. Quieres poder ingresar `Seconds`, `Latitude` y `Longitude` y obtener un id de país como resultado.

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

La precisión no está mal **(alrededor del 95%)**, lo cual no es sorprendente, ya que `Country` y `Latitude/Longitude` están correlacionados.

El modelo que creaste no es muy revolucionario, ya que deberías poder inferir un `Country` a partir de su `Latitude` y `Longitude`, pero es un buen ejercicio para intentar entrenar desde datos crudos que limpiaste, exportaste y luego usar este modelo en una aplicación web.

## Ejercicio - 'picklea' tu modelo

¡Ahora es momento de _picklear_ tu modelo! Puedes hacerlo en unas pocas líneas de código. Una vez que esté _pickleado_, carga tu modelo pickleado y pruébalo contra un arreglo de datos de muestra que contenga valores para segundos, latitud y longitud.

```python
import pickle
model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model = pickle.load(open('ufo-model.pkl','rb'))
print(model.predict([[50,44,-12]]))
```

El modelo devuelve **'3'**, que es el código de país para el Reino Unido. ¡Increíble! 👽

## Ejercicio - construye una aplicación Flask

Ahora puedes construir una aplicación Flask para llamar a tu modelo y devolver resultados similares, pero de una manera más visualmente atractiva.

1. Comienza creando una carpeta llamada **web-app** junto al archivo _notebook.ipynb_ donde reside tu archivo _ufo-model.pkl_.

1. En esa carpeta crea tres carpetas más: **static**, con una carpeta **css** dentro de ella, y **templates**. Ahora deberías tener los siguientes archivos y directorios:

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

    Observa la plantilla en este archivo. Nota la sintaxis 'mustache' alrededor de las variables que serán proporcionadas por la aplicación, como el texto de predicción: `{{}}`. También hay un formulario que envía una predicción a la ruta `/predict`.

    Finalmente, estás listo para construir el archivo Python que impulsa el consumo del modelo y la visualización de las predicciones:

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

    > 💡 Consejo: cuando agregas [`debug=True`](https://www.askpython.com/python-modules/flask/flask-debug-mode) mientras ejecutas la aplicación web usando Flask, cualquier cambio que hagas en tu aplicación se reflejará inmediatamente sin necesidad de reiniciar el servidor. ¡Cuidado! No habilites este modo en una aplicación de producción.

Si ejecutas `python app.py` o `python3 app.py`, tu servidor web se iniciará localmente y podrás completar un formulario corto para obtener una respuesta a tu pregunta sobre dónde se han avistado OVNIs.

Antes de hacer eso, observa las partes de `app.py`:

1. Primero, se cargan las dependencias y se inicia la aplicación.
1. Luego, se importa el modelo.
1. Después, se renderiza index.html en la ruta principal.

En la ruta `/predict`, suceden varias cosas cuando se envía el formulario:

1. Las variables del formulario se recopilan y se convierten en un arreglo numpy. Luego se envían al modelo y se devuelve una predicción.
2. Los países que queremos mostrar se vuelven a renderizar como texto legible a partir de su código de país predicho, y ese valor se envía de vuelta a index.html para ser renderizado en la plantilla.

Usar un modelo de esta manera, con Flask y un modelo pickleado, es relativamente sencillo. Lo más difícil es entender qué forma deben tener los datos que se deben enviar al modelo para obtener una predicción. Todo depende de cómo se entrenó el modelo. Este tiene tres puntos de datos que deben ingresarse para obtener una predicción.

En un entorno profesional, puedes ver cómo es necesaria una buena comunicación entre las personas que entrenan el modelo y aquellas que lo consumen en una aplicación web o móvil. En nuestro caso, ¡es solo una persona, tú!

---

## 🚀 Desafío

En lugar de trabajar en un notebook e importar el modelo a la aplicación Flask, podrías entrenar el modelo directamente dentro de la aplicación Flask. Intenta convertir tu código Python en el notebook, tal vez después de que tus datos estén limpios, para entrenar el modelo desde dentro de la aplicación en una ruta llamada `train`. ¿Cuáles son las ventajas y desventajas de seguir este método?

## [Cuestionario posterior a la lección](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/18/)

## Repaso y autoestudio

Existen muchas formas de construir una aplicación web para consumir modelos de ML. Haz una lista de las formas en que podrías usar JavaScript o Python para construir una aplicación web que aproveche el aprendizaje automático. Considera la arquitectura: ¿debería el modelo permanecer en la aplicación o vivir en la nube? Si es lo último, ¿cómo lo accederías? Dibuja un modelo arquitectónico para una solución web de ML aplicada.

## Tarea

[Prueba un modelo diferente](assignment.md)

---

**Descargo de responsabilidad**:  
Este documento ha sido traducido utilizando el servicio de traducción automática [Co-op Translator](https://github.com/Azure/co-op-translator). Si bien nos esforzamos por lograr precisión, tenga en cuenta que las traducciones automáticas pueden contener errores o imprecisiones. El documento original en su idioma nativo debe considerarse como la fuente autorizada. Para información crítica, se recomienda una traducción profesional realizada por humanos. No nos hacemos responsables de malentendidos o interpretaciones erróneas que puedan surgir del uso de esta traducción.