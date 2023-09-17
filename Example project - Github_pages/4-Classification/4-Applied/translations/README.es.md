# Construye una aplicación web de recomendación de cocina

En esta lección, construirás un modelo de clasificación usando algunas de las técnicas que aprendiste en las lecciones anteriores y con el conjunto de datos de la cocina deliciosa usada a través de este serie de lecciones. Además, construirás una pequeña aplicación web para usar un modelo guardado, aprovechando el runtime web de Onnx.

Uno de los usos prácticos más útiles del aprendizaje automático es construir sistemas de recomendación, y ¡hoy puedes tomar el primer en esa dirección!

[![Presentando esta aplicación web](https://img.youtube.com/vi/17wdM9AHMfg/0.jpg)](https://youtu.be/17wdM9AHMfg "ML aplicado")

> 🎥 Haz clic en la imagen de arriba para ver el video: Jen Looper construye una aplicación web usando los datos clasificados de cocina.

## [Examen previo a la lección](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/25?loc=es)

En esta lección aprenderás:

- Cómo construir un modelo y guardarlo como un modelo Onnx
- Cómo usar Netron para inspeccionar el modelo
- Cómo usar tu modelo en una aplicación web por inferencia

## Construye tu modelo

Construir sistemas de aprendizaje automático aplicado es un parte importante de aprovechar estas tecnologías para tus sistemas de negocio. Puedes usar modelos dentro de tus aplicaciones web (y así usarlos de sin conexión en caso de ser necesario) al usar Onnx.

En la [lección anterior](../../../3-Web-App/1-Web-App/translations/README.es.md), construiste un modelo de regresión acerca de los avistamientos OVNI, le hiciste "pickle", y los usaste en una aplicación Flask. Aunque esta arquitectura es muy útil conocerla, es una aplicación Python full-stack, y tus requerimientos pueden incluir el uso de una aplicación JavaScript.

En esta lección, puedes construir un sistema JavaScript básico por inferencia. Pero primero, necesitas entrenar tu modelo y convertirlo para usarlo con Onnx.

## Ejercicio - entrena tu modelo de clasificación

Primero, entrena un modelo de clasificación usando el conjunto limpio de datos de cocina que ya usamos.

1. Comienza importando bibliotecas útiles:

    ```python
    !pip install skl2onnx
    import pandas as pd 
    ```

    Necesitas '[skl2onnx](https://onnx.ai/sklearn-onnx/)' para ayudar a convertir tu modelo Scikit-learn al formato Onnx.

1. Luego, trabaja con tus datos de la misma forma que lo hiciste en las lecciones anteriores, al leer el archivo CSV usando `read_csv()`:

    ```python
    data = pd.read_csv('../data/cleaned_cuisines.csv')
    data.head()
    ```

1. Elimina las primeras dos columnas y guarda los datos restantes como 'X':

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

Usaremos la biblioteca 'SVC' la cual tiene buena precisión.

1. Importa las bibliotecas correspondientes de Scikit-learn:

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

1. Construye un model de clasificación SVC como lo hiciste en la lección anterior:

    ```python
    model = SVC(kernel='linear', C=10, probability=True,random_state=0)
    model.fit(X_train,y_train.values.ravel())
    ```

1. Ahora, prueba tu modelo al llamar a `predict()`:

    ```python
    y_pred = model.predict(X_test)
    ```

1. Imprime un reporte de clasificación para revisar la calidad del modelo:

    ```python
    print(classification_report(y_test,y_pred))
    ```

    Como vimos anteriormente, la precisión es buena:

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

Asegúrate que haces la conversión con el número adecuado de Tensor. Este conjunto de datos lista 380 ingredientes, por lo que necesitas anotar ese número en `FloatTensorType`:

1. Conviértelo usando un número de tensor de 380.

    ```python
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    
    initial_type = [('float_input', FloatTensorType([None, 380]))]
    options = {id(model): {'nocl': True, 'zipmap': False}}
    ```

1. Crea el onx y guárdalo como un archivo **model.onnx**:

    ```python
    onx = convert_sklearn(model, initial_types=initial_type, options=options)
    with open("./model.onnx", "wb") as f:
        f.write(onx.SerializeToString())
    ```

    > Nota, puedes pasar [opciones](https://onnx.ai/sklearn-onnx/parameterized.html) a tu script de conversión. En este caso, pasamos 'nocl' como True y 'zipap' como False. Ya que este es un modelo de clasificación, tienes la opción de eliminar ZipMap el cual produce una lista de diccionarios (no necesarios). `nocl` se refiere a la información de clase que se incluye en el modelo. Reduce el tamaño de tu modelo al configurar `nocl` a 'True'.

Al ejecutar todo el notebook construirás un modelo Onnx y lo guardará en su directorio.

## Observa tu modelo

Los modelos Onnx no se aprecian bien en Visual Studio Code, pero muchos investigadores usan buen software libre para visualizar el modelo y asegurar que se construyó de forma adecuada. Descarga [Netron](https://github.com/lutzroeder/Netron) y abre tu archivo `model.onnx`. Puedes visualizar de forma simple tu modelo, con sus 380 entradas y el clasificador listado:


![Netron visual](../images/netron.png)

Netron es una herramienta útil para ver tus modelos.

Ahora estás listo para usar este modelo limpio en una aplicación web. Construyamos una aplicación que nos será útil cuando veas tu refrigerador e intentes descubrir qué combinación de tus ingredientes sobrantes puedes usar para realizar un platillo de cierta cocina, de acuerdo a lo que determina tu modelo.

## Construye una aplicación web de recomendación

Puedes usar tu modelo directamente en la aplicación web. Esta arquitectura también te permite ejecutarlo de forma local e incluso sin conexión en caso de ser necesario. Empieza creando un archivo `index.html` en el mismo directorio donde guardaste tu archivo `model.onnx`.

1. En este archivo _index.html_, agrega el siguiente código:

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

1. Ahora, trabaja dentro de las etiquetas `body`, agrega algo de código para mostrar una lista de checkboxes reflejando algunos ingredientes:

    ```html
    <h1>Revisa tu refrigerador. ¿Qué puedes crear?</h1>
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
                <button onClick="startInference()">¿Qué clase de cocina puedes preparar?</button>
            </div> 
    ```

    Nota que a cada checkbox se le asigna un valor. Esto refleja el índice donde se encuentran los ingredientes de acuerdo al conjunto de datos. La manzana (Apple), por ejemplo, en este listado alfabético, ocupa la quinta columna, por lo que su valor es '4' ya que empezamos a contar a partir del 0. Puedes consultar la [hoja de cálculo de los ingredientes](../../data/ingredient_indexes.csv) para descubrir el índice asignado a cierto ingrediente.

    Continúa tu trabajo en el archivo index.html, agrega un bloque script donde se llame al modelo, después de el `</div>` final de cierre.

1. Primero, importa el [Runtime de Onnx](https://www.onnxruntime.ai/):

    ```html
    <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.9.0/dist/ort.min.js"></script> 
    ```

    > El Runtime de Onnx se usa para permitir ejecutar tus modelos Onnx a través de una gran variedad de plataformas hardware, incluyendo optimizaciones y el uso de una API.

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

En este código, suceden varias cosas:

1. Creaste un arreglo de 380 valores posibles(1 o 0) para ser configurados y enviados al modelo por inferencia, dependiendo si checkbox de ingrediente está seleccionado.
2. Creaste un arreglo de checkboxes y una forma de determinar si fueron seleccionados en una función `init` que es llamada cuando inicia la aplicación. Cuando se selecciona un checkbox, se modifica el arreglo `ingredients` para reflejar al ingrediente seleccionado.
3. Creaste una función `testCheckboxes` que verifica si algún checkbox se seleccionó.
4. Usa la función `startInference` cuando se presione el botón y, si algún checkbox fue seleccionado, comienza la inferencia.
5. La rutina de inferencia incluye:
   1. Configuración de una carga asíncrona del modelo
   2. Creación de una estructura Tensor para enviar al modelo
   3. Creación de 'feeds' que refleja la entrada `float_input` que creaste cuando entrenaste tu modelo (puedes usar Netron para verificar el nombre)
   4. Envía estos 'feeds' al modelo y espera la respuesta

## Prueba tu aplicación

Abre una sesión de terminal en Visual Studio Code en el directorio donde reside tu archivo index.html. Asegúrate que tienes instalado [http-server](https://www.npmjs.com/package/http-server) de forma global, y escribe `http-server` en la terminal. Se debería abrir una ventana del navegador web para ver tu aplicación en localhost. Revisa qué cocina es recomendada basada en varios ingredientes:

![Aplicación web de ingredientes](../images/web-app.png)

Felicidades, has creado una aplicación de 'recomendación' con pocos campos. ¡Lleva algo de tiempo el construir este sistema!

## 🚀Desafío

Tu aplicación web es mínima, así que continua construyéndola usando los ingredientes y sus índices de los datos [ingredient_indexes](../../data/ingredient_indexes.csv). ¿Qué combinaciones de sabor funcionan para crear un determinado platillo nacional?

## [Examen posterior a la lección](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/26?loc=es)

## Revisión y autoestudio

Mientras esta lección sólo se refirió a la utilidad de crear un sistema de recomendación para ingredientes alimenticios, esta área de aplicaciones del aprendizaje automático es muy rica en ejemplos. Lee más acerca de cómo se construyen estos sistemas:

- https://www.sciencedirect.com/topics/computer-science/recommendation-engine
- https://www.technologyreview.com/2014/08/25/171547/the-ultimate-challenge-for-recommendation-engines/
- https://www.technologyreview.com/2015/03/23/168831/everything-is-a-recommendation/

## Asignación

[Construye una nueva aplicación de recomendación](assignment.es.md)
