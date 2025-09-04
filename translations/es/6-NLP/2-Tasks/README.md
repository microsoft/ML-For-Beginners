<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "6534e145d52a3890590d27be75386e5d",
  "translation_date": "2025-09-04T00:35:26+00:00",
  "source_file": "6-NLP/2-Tasks/README.md",
  "language_code": "es"
}
-->
# Tareas y técnicas comunes de procesamiento de lenguaje natural

Para la mayoría de las tareas de *procesamiento de lenguaje natural*, el texto que se va a procesar debe descomponerse, examinarse y los resultados almacenarse o cruzarse con reglas y conjuntos de datos. Estas tareas permiten al programador derivar el _significado_, la _intención_ o solo la _frecuencia_ de términos y palabras en un texto.

## [Cuestionario previo a la lección](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/33/)

Descubramos técnicas comunes utilizadas en el procesamiento de texto. Combinadas con aprendizaje automático, estas técnicas te ayudan a analizar grandes cantidades de texto de manera eficiente. Sin embargo, antes de aplicar ML a estas tareas, entendamos los problemas que enfrenta un especialista en NLP.

## Tareas comunes en NLP

Existen diferentes formas de analizar un texto en el que estás trabajando. Hay tareas que puedes realizar y, a través de ellas, puedes comprender el texto y sacar conclusiones. Por lo general, llevas a cabo estas tareas en una secuencia.

### Tokenización

Probablemente lo primero que la mayoría de los algoritmos de NLP tienen que hacer es dividir el texto en tokens o palabras. Aunque esto suena simple, tener en cuenta la puntuación y los delimitadores de palabras y oraciones en diferentes idiomas puede complicarlo. Es posible que tengas que usar varios métodos para determinar las demarcaciones.

![tokenización](../../../../translated_images/tokenization.1641a160c66cd2d93d4524e8114e93158a9ce0eba3ecf117bae318e8a6ad3487.es.png)
> Tokenizando una oración de **Orgullo y Prejuicio**. Infografía por [Jen Looper](https://twitter.com/jenlooper)

### Embeddings

[Word embeddings](https://wikipedia.org/wiki/Word_embedding) son una forma de convertir tus datos de texto en valores numéricos. Los embeddings se realizan de manera que las palabras con un significado similar o palabras que se usan juntas se agrupen.

![word embeddings](../../../../translated_images/embedding.2cf8953c4b3101d188c2f61a5de5b6f53caaa5ad4ed99236d42bc3b6bd6a1fe2.es.png)
> "Tengo el mayor respeto por tus nervios, son mis viejos amigos." - Word embeddings para una oración en **Orgullo y Prejuicio**. Infografía por [Jen Looper](https://twitter.com/jenlooper)

✅ Prueba [esta herramienta interesante](https://projector.tensorflow.org/) para experimentar con word embeddings. Al hacer clic en una palabra, se muestran grupos de palabras similares: 'juguete' se agrupa con 'disney', 'lego', 'playstation' y 'consola'.

### Parsing y etiquetado de partes del discurso

Cada palabra que ha sido tokenizada puede etiquetarse como una parte del discurso: un sustantivo, verbo o adjetivo. La oración `el rápido zorro rojo saltó sobre el perro marrón perezoso` podría etiquetarse como POS con zorro = sustantivo, saltó = verbo.

![parsing](../../../../translated_images/parse.d0c5bbe1106eae8fe7d60a183cd1736c8b6cec907f38000366535f84f3036101.es.png)

> Parseando una oración de **Orgullo y Prejuicio**. Infografía por [Jen Looper](https://twitter.com/jenlooper)

El parsing consiste en reconocer qué palabras están relacionadas entre sí en una oración; por ejemplo, `el rápido zorro rojo saltó` es una secuencia de adjetivo-sustantivo-verbo que está separada de la secuencia `perro marrón perezoso`.

### Frecuencia de palabras y frases

Un procedimiento útil al analizar un gran cuerpo de texto es construir un diccionario de cada palabra o frase de interés y cuántas veces aparece. La frase `el rápido zorro rojo saltó sobre el perro marrón perezoso` tiene una frecuencia de palabras de 2 para "el".

Veamos un texto de ejemplo donde contamos la frecuencia de palabras. El poema Los Ganadores de Rudyard Kipling contiene el siguiente verso:

```output
What the moral? Who rides may read.
When the night is thick and the tracks are blind
A friend at a pinch is a friend, indeed,
But a fool to wait for the laggard behind.
Down to Gehenna or up to the Throne,
He travels the fastest who travels alone.
```

Como las frecuencias de frases pueden ser sensibles o insensibles a mayúsculas según se requiera, la frase `un amigo` tiene una frecuencia de 2, `el` tiene una frecuencia de 6 y `viajes` tiene una frecuencia de 2.

### N-grams

Un texto puede dividirse en secuencias de palabras de una longitud establecida: una sola palabra (unigramas), dos palabras (bigramas), tres palabras (trigramas) o cualquier número de palabras (n-grams).

Por ejemplo, `el rápido zorro rojo saltó sobre el perro marrón perezoso` con un puntaje de n-gram de 2 produce los siguientes n-grams:

1. el rápido  
2. rápido zorro  
3. zorro rojo  
4. rojo saltó  
5. saltó sobre  
6. sobre el  
7. el perro  
8. perro marrón  
9. marrón perezoso  

Podría ser más fácil visualizarlo como una ventana deslizante sobre la oración. Aquí está para n-grams de 3 palabras, el n-gram está en negrita en cada oración:

1.   <u>**el rápido zorro**</u> rojo saltó sobre el perro marrón perezoso  
2.   el **<u>rápido zorro rojo</u>** saltó sobre el perro marrón perezoso  
3.   el rápido **<u>zorro rojo saltó</u>** sobre el perro marrón perezoso  
4.   el rápido zorro **<u>rojo saltó sobre</u>** el perro marrón perezoso  
5.   el rápido zorro rojo **<u>saltó sobre el</u>** perro marrón perezoso  
6.   el rápido zorro rojo saltó **<u>sobre el perro</u>** marrón perezoso  
7.   el rápido zorro rojo saltó sobre <u>**el perro marrón**</u> perezoso  
8.   el rápido zorro rojo saltó sobre el **<u>perro marrón perezoso</u>**

![ventana deslizante de n-grams](../../../../6-NLP/2-Tasks/images/n-grams.gif)

> Valor de n-gram de 3: Infografía por [Jen Looper](https://twitter.com/jenlooper)

### Extracción de frases nominales

En la mayoría de las oraciones, hay un sustantivo que es el sujeto u objeto de la oración. En inglés, a menudo se identifica porque tiene 'a', 'an' o 'the' antes de él. Identificar el sujeto u objeto de una oración mediante la 'extracción de la frase nominal' es una tarea común en NLP al intentar comprender el significado de una oración.

✅ En la oración "No puedo fijar la hora, ni el lugar, ni la mirada ni las palabras, que sentaron las bases. Hace demasiado tiempo. Estaba en medio antes de darme cuenta de que había comenzado.", ¿puedes identificar las frases nominales?

En la oración `el rápido zorro rojo saltó sobre el perro marrón perezoso` hay 2 frases nominales: **rápido zorro rojo** y **perro marrón perezoso**.

### Análisis de sentimiento

Una oración o texto puede analizarse para determinar su sentimiento, o cuán *positivo* o *negativo* es. El sentimiento se mide en *polaridad* y *objetividad/subjetividad*. La polaridad se mide de -1.0 a 1.0 (negativo a positivo) y de 0.0 a 1.0 (más objetivo a más subjetivo).

✅ Más adelante aprenderás que hay diferentes formas de determinar el sentimiento utilizando aprendizaje automático, pero una forma es tener una lista de palabras y frases que han sido categorizadas como positivas o negativas por un experto humano y aplicar ese modelo al texto para calcular un puntaje de polaridad. ¿Puedes ver cómo esto funcionaría en algunas circunstancias y menos en otras?

### Inflexión

La inflexión te permite tomar una palabra y obtener el singular o plural de la misma.

### Lematización

Un *lema* es la raíz o palabra principal de un conjunto de palabras; por ejemplo, *voló*, *vuela*, *volando* tienen como lema el verbo *volar*.

También hay bases de datos útiles disponibles para el investigador de NLP, en particular:

### WordNet

[WordNet](https://wordnet.princeton.edu/) es una base de datos de palabras, sinónimos, antónimos y muchos otros detalles para cada palabra en muchos idiomas diferentes. Es increíblemente útil al intentar construir traducciones, correctores ortográficos o herramientas de lenguaje de cualquier tipo.

## Bibliotecas de NLP

Afortunadamente, no tienes que construir todas estas técnicas tú mismo, ya que hay excelentes bibliotecas de Python disponibles que hacen que sea mucho más accesible para desarrolladores que no están especializados en procesamiento de lenguaje natural o aprendizaje automático. Las próximas lecciones incluyen más ejemplos de estas, pero aquí aprenderás algunos ejemplos útiles para ayudarte con la próxima tarea.

### Ejercicio - usando la biblioteca `TextBlob`

Usemos una biblioteca llamada TextBlob, ya que contiene APIs útiles para abordar este tipo de tareas. TextBlob "se basa en los hombros gigantes de [NLTK](https://nltk.org) y [pattern](https://github.com/clips/pattern), y funciona bien con ambos." Tiene una cantidad considerable de ML integrado en su API.

> Nota: Una útil [Guía de inicio rápido](https://textblob.readthedocs.io/en/dev/quickstart.html#quickstart) está disponible para TextBlob y se recomienda para desarrolladores experimentados en Python.

Al intentar identificar *frases nominales*, TextBlob ofrece varias opciones de extractores para encontrarlas.

1. Echa un vistazo a `ConllExtractor`.

    ```python
    from textblob import TextBlob
    from textblob.np_extractors import ConllExtractor
    # import and create a Conll extractor to use later 
    extractor = ConllExtractor()
    
    # later when you need a noun phrase extractor:
    user_input = input("> ")
    user_input_blob = TextBlob(user_input, np_extractor=extractor)  # note non-default extractor specified
    np = user_input_blob.noun_phrases                                    
    ```

    > ¿Qué está pasando aquí? [ConllExtractor](https://textblob.readthedocs.io/en/dev/api_reference.html?highlight=Conll#textblob.en.np_extractors.ConllExtractor) es "Un extractor de frases nominales que utiliza chunk parsing entrenado con el corpus de entrenamiento ConLL-2000." ConLL-2000 se refiere a la Conferencia de Aprendizaje Computacional de Lenguaje Natural de 2000. Cada año, la conferencia organizaba un taller para abordar un problema difícil de NLP, y en 2000 fue el chunking de frases nominales. Se entrenó un modelo en el Wall Street Journal, con "las secciones 15-18 como datos de entrenamiento (211727 tokens) y la sección 20 como datos de prueba (47377 tokens)". Puedes ver los procedimientos utilizados [aquí](https://www.clips.uantwerpen.be/conll2000/chunking/) y los [resultados](https://ifarm.nl/erikt/research/np-chunking.html).

### Desafío - mejorando tu bot con NLP

En la lección anterior construiste un bot de preguntas y respuestas muy simple. Ahora, harás que Marvin sea un poco más empático analizando tu entrada para determinar el sentimiento y mostrando una respuesta que coincida con el sentimiento. También necesitarás identificar una `frase nominal` y preguntar sobre ella.

Tus pasos al construir un bot conversacional mejorado:

1. Imprime instrucciones aconsejando al usuario cómo interactuar con el bot.  
2. Inicia un bucle:  
   1. Acepta la entrada del usuario.  
   2. Si el usuario ha pedido salir, entonces sal.  
   3. Procesa la entrada del usuario y determina una respuesta de sentimiento adecuada.  
   4. Si se detecta una frase nominal en el sentimiento, pluralízala y pide más información sobre ese tema.  
   5. Imprime la respuesta.  
3. Vuelve al paso 2.  

Aquí está el fragmento de código para determinar el sentimiento usando TextBlob. Nota que solo hay cuatro *gradientes* de respuesta de sentimiento (puedes tener más si lo deseas):

```python
if user_input_blob.polarity <= -0.5:
  response = "Oh dear, that sounds bad. "
elif user_input_blob.polarity <= 0:
  response = "Hmm, that's not great. "
elif user_input_blob.polarity <= 0.5:
  response = "Well, that sounds positive. "
elif user_input_blob.polarity <= 1:
  response = "Wow, that sounds great. "
```

Aquí hay un ejemplo de salida para guiarte (la entrada del usuario está en las líneas que comienzan con >):

```output
Hello, I am Marvin, the friendly robot.
You can end this conversation at any time by typing 'bye'
After typing each answer, press 'enter'
How are you today?
> I am ok
Well, that sounds positive. Can you tell me more?
> I went for a walk and saw a lovely cat
Well, that sounds positive. Can you tell me more about lovely cats?
> cats are the best. But I also have a cool dog
Wow, that sounds great. Can you tell me more about cool dogs?
> I have an old hounddog but he is sick
Hmm, that's not great. Can you tell me more about old hounddogs?
> bye
It was nice talking to you, goodbye!
```

Una posible solución a la tarea está [aquí](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/2-Tasks/solution/bot.py)

✅ Verificación de conocimiento

1. ¿Crees que las respuestas empáticas podrían 'engañar' a alguien para que piense que el bot realmente los entiende?  
2. ¿Hace que el bot sea más 'creíble' identificar la frase nominal?  
3. ¿Por qué sería útil extraer una 'frase nominal' de una oración?  

---

Implementa el bot en la verificación de conocimiento anterior y pruébalo con un amigo. ¿Puede engañarlos? ¿Puedes hacer que tu bot sea más 'creíble'?

## 🚀Desafío

Toma una tarea de la verificación de conocimiento anterior e intenta implementarla. Prueba el bot con un amigo. ¿Puede engañarlos? ¿Puedes hacer que tu bot sea más 'creíble'?

## [Cuestionario posterior a la lección](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/34/)

## Revisión y autoestudio

En las próximas lecciones aprenderás más sobre análisis de sentimiento. Investiga esta técnica interesante en artículos como estos en [KDNuggets](https://www.kdnuggets.com/tag/nlp)

## Tarea 

[Haz que un bot responda](assignment.md)

---

**Descargo de responsabilidad**:  
Este documento ha sido traducido utilizando el servicio de traducción automática [Co-op Translator](https://github.com/Azure/co-op-translator). Si bien nos esforzamos por garantizar la precisión, tenga en cuenta que las traducciones automatizadas pueden contener errores o imprecisiones. El documento original en su idioma nativo debe considerarse como la fuente autorizada. Para información crítica, se recomienda una traducción profesional realizada por humanos. No nos hacemos responsables de malentendidos o interpretaciones erróneas que puedan surgir del uso de esta traducción.