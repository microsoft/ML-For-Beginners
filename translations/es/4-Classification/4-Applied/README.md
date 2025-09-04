<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "ad2cf19d7490247558d20a6a59650d13",
  "translation_date": "2025-09-03T23:56:00+00:00",
  "source_file": "4-Classification/4-Applied/README.md",
  "language_code": "es"
}
-->
# Construir una Aplicación Web de Recomendación de Cocina

En esta lección, construirás un modelo de clasificación utilizando algunas de las técnicas que has aprendido en lecciones anteriores y con el delicioso conjunto de datos de cocina utilizado a lo largo de esta serie. Además, crearás una pequeña aplicación web para usar un modelo guardado, aprovechando el runtime web de Onnx.

Uno de los usos prácticos más útiles del aprendizaje automático es construir sistemas de recomendación, ¡y hoy puedes dar el primer paso en esa dirección!

[![Presentando esta aplicación web](https://img.youtube.com/vi/17wdM9AHMfg/0.jpg)](https://youtu.be/17wdM9AHMfg "ML Aplicado")

> 🎥 Haz clic en la imagen de arriba para ver un video: Jen Looper construye una aplicación web utilizando datos clasificados de cocina.

## [Cuestionario previo a la lección](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/25/)

En esta lección aprenderás:

- Cómo construir un modelo y guardarlo como un modelo Onnx.
- Cómo usar Netron para inspeccionar el modelo.
- Cómo usar tu modelo en una aplicación web para inferencia.

## Construye tu modelo

Construir sistemas de ML aplicados es una parte importante para aprovechar estas tecnologías en tus sistemas empresariales. Puedes usar modelos dentro de tus aplicaciones web (y así utilizarlos en un contexto offline si es necesario) usando Onnx.

En una [lección anterior](../../3-Web-App/1-Web-App/README.md), construiste un modelo de regresión sobre avistamientos de OVNIs, lo "pickleaste" y lo usaste en una aplicación Flask. Aunque esta arquitectura es muy útil de conocer, es una aplicación Python de pila completa, y tus requisitos pueden incluir el uso de una aplicación JavaScript.

En esta lección, puedes construir un sistema básico basado en JavaScript para inferencia. Sin embargo, primero necesitas entrenar un modelo y convertirlo para usarlo con Onnx.

## Ejercicio - entrenar un modelo de clasificación

Primero, entrena un modelo de clasificación utilizando el conjunto de datos de cocina limpio que usamos.

1. Comienza importando bibliotecas útiles:

    ```python
    !pip install skl2onnx
    import pandas as pd 
    ```

    Necesitas '[skl2onnx](https://onnx.ai/sklearn-onnx/)' para ayudar a convertir tu modelo de Scikit-learn al formato Onnx.

1. Luego, trabaja con tus datos de la misma manera que lo hiciste en lecciones anteriores, leyendo un archivo CSV usando `read_csv()`:

    ```python
    data = pd.read_csv('../data/cleaned_cuisines.csv')
    data.head()
    ```

1. Elimina las dos primeras columnas innecesarias y guarda los datos restantes como 'X':

    ```python
    X = data.iloc[:,2:]
    X.head()
    ```

1. Guarda las etiquetas como 'y':

    ```python
    y = data[['cuisine']]
    y.head()
    
    ```

### Comienza la rutina de entrenamiento

Usaremos la biblioteca 'SVC', que tiene buena precisión.

1. Importa las bibliotecas apropiadas de Scikit-learn:

    ```python
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report
    ```

1. Separa los conjuntos de entrenamiento y prueba:

    ```python
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
    ```

1. Construye un modelo de clasificación SVC como lo hiciste en la lección anterior:

    ```python
    model = SVC(kernel='linear', C=10, probability=True,random_state=0)
    model.fit(X_train,y_train.values.ravel())
    ```

1. Ahora, prueba tu modelo llamando a `predict()`:

    ```python
    y_pred = model.predict(X_test)
    ```

1. Imprime un informe de clasificación para verificar la calidad del modelo:

    ```python
    print(classification_report(y_test,y_pred))
    ```

    Como vimos antes, la precisión es buena:

    ```output
                    precision    recall  f1-score   support
    
         chinese       0.72      0.69      0.70       257
          indian       0.91      0.87      0.89       243
        japanese       0.79      0.77      0.78       239
          korean       0.83      0.79      0.81       236
            thai       0.72      0.84      0.78       224
    
        accuracy                           0.79      1199
       macro avg       0.79      0.79      0.79      1199
    weighted avg       0.79      0.79      0.79      1199
    ```

### Convierte tu modelo a Onnx

Asegúrate de hacer la conversión con el número adecuado de tensores. Este conjunto de datos tiene 380 ingredientes listados, por lo que necesitas anotar ese número en `FloatTensorType`:

1. Convierte usando un número de tensor de 380.

    ```python
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    
    initial_type = [('float_input', FloatTensorType([None, 380]))]
    options = {id(model): {'nocl': True, 'zipmap': False}}
    ```

1. Crea el archivo onx y guárdalo como **model.onnx**:

    ```python
    onx = convert_sklearn(model, initial_types=initial_type, options=options)
    with open("./model.onnx", "wb") as f:
        f.write(onx.SerializeToString())
    ```

    > Nota, puedes pasar [opciones](https://onnx.ai/sklearn-onnx/parameterized.html) en tu script de conversión. En este caso, pasamos 'nocl' como True y 'zipmap' como False. Dado que este es un modelo de clasificación, tienes la opción de eliminar ZipMap, que produce una lista de diccionarios (no es necesario). `nocl` se refiere a la información de clase incluida en el modelo. Reduce el tamaño de tu modelo configurando `nocl` como 'True'.

Ejecutar todo el notebook ahora construirá un modelo Onnx y lo guardará en esta carpeta.

## Visualiza tu modelo

Los modelos Onnx no son muy visibles en Visual Studio Code, pero hay un software gratuito muy bueno que muchos investigadores usan para visualizar el modelo y asegurarse de que esté correctamente construido. Descarga [Netron](https://github.com/lutzroeder/Netron) y abre tu archivo model.onnx. Puedes ver tu modelo simple visualizado, con sus 380 entradas y clasificador listados:

![Visualización de Netron](../../../../translated_images/netron.a05f39410211915e0f95e2c0e8b88f41e7d13d725faf660188f3802ba5c9e831.es.png)

Netron es una herramienta útil para visualizar tus modelos.

Ahora estás listo para usar este modelo en una aplicación web. Construyamos una aplicación que será útil cuando mires en tu refrigerador y trates de averiguar qué combinación de tus ingredientes sobrantes puedes usar para cocinar un plato determinado, según lo determine tu modelo.

## Construye una aplicación web de recomendación

Puedes usar tu modelo directamente en una aplicación web. Esta arquitectura también te permite ejecutarla localmente e incluso offline si es necesario. Comienza creando un archivo `index.html` en la misma carpeta donde guardaste tu archivo `model.onnx`.

1. En este archivo _index.html_, agrega el siguiente marcado:

    ```html
    <!DOCTYPE html>
    <html>
        <header>
            <title>Cuisine Matcher</title>
        </header>
        <body>
            ...
        </body>
    </html>
    ```

1. Ahora, trabajando dentro de las etiquetas `body`, agrega un poco de marcado para mostrar una lista de casillas de verificación que reflejen algunos ingredientes:

    ```html
    <h1>Check your refrigerator. What can you create?</h1>
            <div id="wrapper">
                <div class="boxCont">
                    <input type="checkbox" value="4" class="checkbox">
                    <label>apple</label>
                </div>
            
                <div class="boxCont">
                    <input type="checkbox" value="247" class="checkbox">
                    <label>pear</label>
                </div>
            
                <div class="boxCont">
                    <input type="checkbox" value="77" class="checkbox">
                    <label>cherry</label>
                </div>
    
                <div class="boxCont">
                    <input type="checkbox" value="126" class="checkbox">
                    <label>fenugreek</label>
                </div>
    
                <div class="boxCont">
                    <input type="checkbox" value="302" class="checkbox">
                    <label>sake</label>
                </div>
    
                <div class="boxCont">
                    <input type="checkbox" value="327" class="checkbox">
                    <label>soy sauce</label>
                </div>
    
                <div class="boxCont">
                    <input type="checkbox" value="112" class="checkbox">
                    <label>cumin</label>
                </div>
            </div>
            <div style="padding-top:10px">
                <button onClick="startInference()">What kind of cuisine can you make?</button>
            </div> 
    ```

    Nota que a cada casilla de verificación se le asigna un valor. Esto refleja el índice donde se encuentra el ingrediente según el conjunto de datos. Por ejemplo, la manzana, en esta lista alfabética, ocupa la quinta columna, por lo que su valor es '4', ya que comenzamos a contar desde 0. Puedes consultar la [hoja de cálculo de ingredientes](../../../../4-Classification/data/ingredient_indexes.csv) para descubrir el índice de un ingrediente dado.

    Continuando tu trabajo en el archivo index.html, agrega un bloque de script donde se llame al modelo después del cierre final `</div>`.

1. Primero, importa el [Runtime de Onnx](https://www.onnxruntime.ai/):

    ```html
    <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.9.0/dist/ort.min.js"></script> 
    ```

    > El Runtime de Onnx se utiliza para habilitar la ejecución de tus modelos Onnx en una amplia gama de plataformas de hardware, incluyendo optimizaciones y una API para usar.

1. Una vez que el Runtime esté en su lugar, puedes llamarlo:

    ```html
    <script>
        const ingredients = Array(380).fill(0);
        
        const checks = [...document.querySelectorAll('.checkbox')];
        
        checks.forEach(check => {
            check.addEventListener('change', function() {
                // toggle the state of the ingredient
                // based on the checkbox's value (1 or 0)
                ingredients[check.value] = check.checked ? 1 : 0;
            });
        });

        function testCheckboxes() {
            // validate if at least one checkbox is checked
            return checks.some(check => check.checked);
        }

        async function startInference() {

            let atLeastOneChecked = testCheckboxes()

            if (!atLeastOneChecked) {
                alert('Please select at least one ingredient.');
                return;
            }
            try {
                // create a new session and load the model.
                
                const session = await ort.InferenceSession.create('./model.onnx');

                const input = new ort.Tensor(new Float32Array(ingredients), [1, 380]);
                const feeds = { float_input: input };

                // feed inputs and run
                const results = await session.run(feeds);

                // read from results
                alert('You can enjoy ' + results.label.data[0] + ' cuisine today!')

            } catch (e) {
                console.log(`failed to inference ONNX model`);
                console.error(e);
            }
        }
               
    </script>
    ```

En este código, están ocurriendo varias cosas:

1. Creaste un array de 380 posibles valores (1 o 0) que se configurarán y enviarán al modelo para inferencia, dependiendo de si se marca una casilla de verificación de ingrediente.
2. Creaste un array de casillas de verificación y una forma de determinar si fueron marcadas en una función `init` que se llama cuando la aplicación comienza. Cuando se marca una casilla, el array `ingredients` se altera para reflejar el ingrediente elegido.
3. Creaste una función `testCheckboxes` que verifica si alguna casilla fue marcada.
4. Usas la función `startInference` cuando se presiona el botón y, si alguna casilla está marcada, comienzas la inferencia.
5. La rutina de inferencia incluye:
   1. Configurar una carga asincrónica del modelo.
   2. Crear una estructura de Tensor para enviar al modelo.
   3. Crear 'feeds' que reflejan la entrada `float_input` que creaste al entrenar tu modelo (puedes usar Netron para verificar ese nombre).
   4. Enviar estos 'feeds' al modelo y esperar una respuesta.

## Prueba tu aplicación

Abre una sesión de terminal en Visual Studio Code en la carpeta donde reside tu archivo index.html. Asegúrate de tener [http-server](https://www.npmjs.com/package/http-server) instalado globalmente y escribe `http-server` en el prompt. Debería abrirse un localhost y podrás ver tu aplicación web. Verifica qué cocina se recomienda según varios ingredientes:

![Aplicación web de ingredientes](../../../../translated_images/web-app.4c76450cabe20036f8ec6d5e05ccc0c1c064f0d8f2fe3304d3bcc0198f7dc139.es.png)

¡Felicidades, has creado una aplicación web de 'recomendación' con algunos campos! Tómate un tiempo para desarrollar este sistema.

## 🚀Desafío

Tu aplicación web es muy minimalista, así que continúa desarrollándola utilizando ingredientes y sus índices del archivo de datos [ingredient_indexes](../../../../4-Classification/data/ingredient_indexes.csv). ¿Qué combinaciones de sabores funcionan para crear un plato nacional determinado?

## [Cuestionario posterior a la lección](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/26/)

## Revisión y Autoestudio

Aunque esta lección solo tocó la utilidad de crear un sistema de recomendación para ingredientes alimenticios, esta área de aplicaciones de ML es muy rica en ejemplos. Lee más sobre cómo se construyen estos sistemas:

- https://www.sciencedirect.com/topics/computer-science/recommendation-engine
- https://www.technologyreview.com/2014/08/25/171547/the-ultimate-challenge-for-recommendation-engines/
- https://www.technologyreview.com/2015/03/23/168831/everything-is-a-recommendation/

## Tarea 

[Construye un nuevo recomendador](assignment.md)

---

**Descargo de responsabilidad**:  
Este documento ha sido traducido utilizando el servicio de traducción automática [Co-op Translator](https://github.com/Azure/co-op-translator). Si bien nos esforzamos por lograr precisión, tenga en cuenta que las traducciones automáticas pueden contener errores o imprecisiones. El documento original en su idioma nativo debe considerarse como la fuente autorizada. Para información crítica, se recomienda una traducción profesional realizada por humanos. No nos hacemos responsables de malentendidos o interpretaciones erróneas que puedan surgir del uso de esta traducción.