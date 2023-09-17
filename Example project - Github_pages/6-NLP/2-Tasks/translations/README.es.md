# Tareas y técnicas comunes del procesamiento del lenguaje natural

Para la mayoría de tareas de *procesamiento del lenguaje natural*, el texto a ser procesado debe ser partido en bloques, examinado y los resultados almacenados y tener referencias cruzadas con reglas y conjuntos de datos. Esta tareas, le permiten al programador obtener el _significado_, _intención_ o sólo la _frecuencia_ de los términos y palabras en un texto.

## [Examen previo a la lección](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/33?loc=es)

Descubramos técnicas comunes usadas en el procesamiento de texto. Combinadas con el aprendizaje automático, estas técnicas te ayudan a analizar grandes cantidades de texto de forma eficiente, Antes de aplicar aprendizaje automático a estas tareas, primero entendamos los problemas encontrados por un especialista del procesamiento del lenguaje natural.

## Tareas comunes al procesamiento del lenguaje natural

Existen distintas forma de analizar un texto en el cual trabajas. Hay tareas que puedes realizar y a través de estas tareas eres capaz de estimar la comprensión del texto y sacar conclusiones. Usualmente llevas a cabo estas tareas en secuencia.

### Tokenización

Probablemente la primer cosa que la mayoría de los algoritmos tiene que hacer es dividir el texto en `tokens`, o palabras. Aunque esto suena simple, teniendo en cuenta la puntuación y distintas palabras y delimitadoras de oraciones en los diferentes idiomas puede hacerlo difícil. Puede que tengas que usar varios métodos para determinar la separación.

![Tokenización](../images/tokenization.png)
> Tokenizando una oración de **Orgullo y Prejuicio**. Infografía de [Jen Looper](https://twitter.com/jenlooper)

### Incrustaciones

[Las incrustaciones de palabras](https://wikipedia.org/wiki/Word_embedding) son una forma de convertir numéricamente tus datos de texto. Las incrustaciones se realizan de tal forma que las palabras con significado similar o palabras que se usan juntas son agrupadas.

![Incrustaciones de palabras](../images/embedding.png)
> "I have the highest respect for your nerves, they are my old friends." - Palabras incrustadas para una oración en **Orgullo y Prejuicio**. Infografía de [Jen Looper](https://twitter.com/jenlooper)

✅ Prueba [esta herramienta interesante](https://projector.tensorflow.org/) par aexperimentar con palabras embebidas. Dando cic en una palabra se muestran grupos de palabras similares: 'toy' se agrupa con 'disney', 'lego', 'playstation', y 'console'.

### Parseo y etiquetado de parte del discurso

Cada palabra que ha sido tokenizada puede ser etiquetada como parte de un discurso - un sustantivo, verbo o adjetivo. La oración `the quick red fox jumped over the lazy brown dog` puede ser etiquetada como parte del discurso como fox = noun, jumped = verb.

![Parseo](../images/parse.png)

> Analizando una oración de **Orgullo y Prejuicio**. Infografía de [Jen Looper](https://twitter.com/jenlooper)

El parseo es reconocer qué palabras están relacionadas con otras en una oración - por ejemplo `the quick red fox jumped` es una secuencia adjetivo-sustantivo-verbo que está separada de la secuencia `lazy brown dog`.

### Frecuencias de palabras y frases

Un procedimiento útil cuando analizas un gran bloque de texto es construir un diccionario de cada palabra o frase de interés y qué tan frecuente aparece. La frase `the quick red fox jumped over the lazy brown dog` tiene una frecuencia de palabra de 2 para `the`.

Veamos un texto de ejemplo donde contamos las frecuencias de las palabras. El poema The Winners de Rudyard Kipling contiene el siguiente verso:

```output
What the moral? Who rides may read.
When the night is thick and the tracks are blind
A friend at a pinch is a friend, indeed,
But a fool to wait for the laggard behind.
Down to Gehenna or up to the Throne,
He travels the fastest who travels alone.
```

Como las frecuencias de frases pueden distinguir entre mayúsculas y minúsculas según se requiera, la frase `a friend` tiene una frecuencia de 2, `the` tiene una frecuencia de 6, y `travels` es 2.

### N-gramas

Un texto puede ser dividido en secuencias de palabras de una longitud definida, una palabra simple (unigrama), dos palabras (bigramas), tres palabras (trigramas) o cualquier número de palabras (n-gramas).

Por ejemplo `the quick red fox jumped over the lazy brown dog` con un n-grama de puntaje 2 produce los siguientes n-gramas:

1. the quick
2. quick red
3. red fox
4. fox jumped
5. jumped over
6. over the
7. the lazy
8. lazy brown
9. brown dog

Podría ser más fácil visualizarlo como una caja deslizante sobre la oración. Aquí se presenta para n-gramas de 3 palabras, el n-grama está en negritas en cada oración:

1.   <u>**the quick red**</u> fox jumped over the lazy brown dog
2.   the **<u>quick red fox</u>** jumped over the lazy brown dog
3.   the quick **<u>red fox jumped</u>** over the lazy brown dog
4.   the quick red **<u>fox jumped over</u>** the lazy brown dog
5.   the quick red fox **<u>jumped over the</u>** lazy brown dog
6.   the quick red fox jumped **<u>over the lazy</u>** brown dog
7.   the quick red fox jumped over <u>**the lazy brown**</u> dog
8.   the quick red fox jumped over the **<u>lazy brown dog</u>**

![Ventana deslizante de n-gramas](../images/n-grams.gif)

> N-grama de valor 3: Infografía de [Jen Looper](https://twitter.com/jenlooper)

### Extracción de frases nominales

En la mayoría de las oraciones, hay un sustantivo que es el sujeto u objeto de la oración. En Inglés, se suele identificar como al tener 'a', 'an' o 'the' precediéndole. Identificar el sujeto u objeto de una oración al 'extraer la frase nominal' es una tarea común del procesamiento del lenguaje natural al intentar comprender el significado de una oración.

✅ En la oración "I cannot fix on the hour, or the spot, or the look or the words, which laid the foundation. It is too long ago. I was in the middle before I knew that I had begun.", ¿puedes identificar las frases nominales?

En la oración `the quick red fox jumped over the lazy brown dog` hay 2 frases nominales: **quick red fox** y **lazy brown dog**.

### Análisis de sentimiento

Una oración o texto puede ser analizado por sentimiento, o que tan *positivo* o *negativo* es. El sentimiento se mide en *polaridad* y *objetividad/subjetividad*. La polaridad se mide de -1.0 a 1.0 (negativo a positivo) y 0.0 a 1.0 (de más objetivo a más subjetivo).

✅ Más adelante aprenderás que hay distintas formas de determinar el sentimiento usando aprendizaje automático, pero una forma es tener una lista de palabras  y frase que son categorizadas como positivas o negativas por un humano experto y aplica ese modelo al texto para calcular un puntaje de polaridad. ¿Puedes ver cómo esto podría funcionar mejor en ciertas circunstancias y peor en otras?

### Inflexión

La inflexión te permite tomar una palabra y obtener el singular o plural de la misma.

### Lematización

Un *lema* es la raíz o palabra principal para un conjunto de palabras, por ejemplo *flew*, *flies*, *flying* tiene como lema el verbo *fly*.

También hay bases de datos útiles para el investigador del procesamiento del lenguaje natural, notablemente:

### WordNet

[WordNet](https://wordnet.princeton.edu/) es una base de datos de palabras, sinónimos antónimos y muchos otros detalles para cada palabra en distintos idiomas. Es increíblemente útil al intentar construir traducciones, correctores ortográficos, o herramientas de idioma de cualquier tipo.

## Bibliotecas NLP

Afortunadamente, no tienes que construir todas estas técnicas por ti mismo, ya que existen excelentes bibliotecas Python disponibles que hacen mucho más accesible a los desarrolladores que no están especializados en el procesamiento de lenguaje natural o aprendizaje automático. Las siguientes lecciones incluyen más ejemplos de estos, pero aquí aprenderás algunos ejemplos útiles para ayudarte con las siguientes tareas.

### Ejercicio - usando la biblioteca `TextBlob`

Usemos una biblioteca llamada TextBlob ya que contiene APIs útiles para abordar este tipo de tareas. TextBlob "se para sobre hombros de gigantes como [NLTK](https://nltk.org) y [pattern](https://github.com/clips/pattern), y se integran bien con ambos." Tiene una cantidad considerable de aprendizaje automático embebido en su API.

> Nota: Hay una guía de [Inicio rápido](https://textblob.readthedocs.io/en/dev/quickstart.html#quickstart) útil disponible para TextBlob que es recomendada por desarrolladores Python experimentados.

Al intentar identificar *frases nominales*, TextBlob ofrece varias opciones de extractores para encontrar frases nominales.

1. Da un vistazo a `ConllExtractor`.

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

    > ¿Qué pasa aquí? [ConllExtractor](https://textblob.readthedocs.io/en/dev/api_reference.html?highlight=Conll#textblob.en.np_extractors.ConllExtractor) es "un extractor de frases nominales que usa el análisis de fragmentos entrenado con el corpus de entrenamiento ConLL-2000." ConLL-2000 se refiere a la conferencia en aprendizaje de lenguaje natural computacional del 2000. Cada año la conferencia organiza un talle para abordar un problema difícil del procesamiento del lenguaje natural, y en el 2000 fue de fragmentación de sustantivos. Se entrenó un modelo en el Wall Street Journal, con las "secciones 15 a 18 como datos de entrenamiento (211727 tokens) y la sección 20 como datos de prueba (47377 tokens)". Puedes revisar los procedimientos usados [aquí](https://www.clips.uantwerpen.be/conll2000/chunking/) y los [resultados](https://ifarm.nl/erikt/research/np-chunking.html).

### Desafío - Mejora tu bot con procesamiento del lenguaje natural

En la lección anterior construiste un bot de preguntas y respuestas muy simple. Ahora, harás a Marvin un poco más comprensivo al analizar tu entrada para sentimiento e imprimir una respuesta para emparejar el sentimiento. También necesitarás identificar `noun_phrase` y preguntar al respecto.

Tus pasos al construir un mejor bot conversacional:

1. Imprime las instrucciones avisando al usuario cómo interactuar con el bot
2. Inicia el ciclo
   1. Acepta la entrada del usuario
   2. Si el usuario pidió salir, entonces sal del programa
   3. Procesa la entrada del usuario y determina la respuesta de sentimiento apropiada
   4. Si se detectó una frase nominal en el sentimiento, pluralízalo y pide más entradas de ese tema
   5. Imprime la respuesta
3. Regresa al paso 2

Aquí tienes el fragmento de código para determinar el sentimiento usando TextBlob. Nota que sólo hay 4 *gradientes* de respuesta de sentimiento (podrías tener más si quisieras):

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

Aquí está una salida de muestra para guiarte (la entrada del usuario está en las líneas que comienzan con >):

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

Una posible solución a esta tarea esta [aquí](../solution/bot.py)

✅ Revisión de conocimiento

1. ¿Piensas que las respuestas comprensivas 'engañarían' a alguien a creer que el bot en realidad les entendió?
2. ¿El identificar la frane nominal hace al bot más 'creíble'?
3. ¿Por qué extraer una 'frase nominal' de una oración es algo útil?

---

Implementa el bot con la revisión de conocimiento anterior y pruébalo con un amigo. ¿Pudo engañarlo? ¿Puedes hacer a tu bot más 'creíble'?

## 🚀Desafío

Toma una tarea de la revisión de conocimiento previo y trata de implementarla. Prueba el bot con un amigo. ¿Pudo engañarlo? ¿Puedes hacer a tu bot más 'creíble'?

## [Examen posterior a la lectura](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/34?loc=es)

## Revisión y autoestudio

En las siguientes lecciones aprenderás más acerca del análisis de sentimiento. Investiga esta técnica interesante en artículos como estos en [KDNuggets](https://www.kdnuggets.com/tag/nlp)

## Asignación

[Haz que un bot responda](assignment.es.md)
