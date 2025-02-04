# Traducción y análisis de sentimientos con ML

En las lecciones anteriores aprendiste a construir un bot básico usando `TextBlob`, una biblioteca que incorpora ML detrás de escena para realizar tareas básicas de NLP como la extracción de frases nominales. Otro desafío importante en la lingüística computacional es la _traducción_ precisa de una oración de un idioma hablado o escrito a otro.

## [Cuestionario previo a la clase](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/35/)

La traducción es un problema muy difícil debido al hecho de que hay miles de idiomas y cada uno puede tener reglas gramaticales muy diferentes. Un enfoque es convertir las reglas gramaticales formales de un idioma, como el inglés, en una estructura no dependiente del idioma, y luego traducirla volviendo a convertirla a otro idioma. Este enfoque significa que seguirías los siguientes pasos:

1. **Identificación**. Identificar o etiquetar las palabras en el idioma de entrada como sustantivos, verbos, etc.
2. **Crear traducción**. Producir una traducción directa de cada palabra en el formato del idioma objetivo.

### Oración de ejemplo, inglés a irlandés

En 'inglés', la oración _I feel happy_ tiene tres palabras en el orden:

- **sujeto** (I)
- **verbo** (feel)
- **adjetivo** (happy)

Sin embargo, en el idioma 'irlandés', la misma oración tiene una estructura gramatical muy diferente: las emociones como "*happy*" o "*sad*" se expresan como estando *sobre* ti.

La frase en inglés `I feel happy` en irlandés sería `Tá athas orm`. Una traducción *literal* sería `Happy is upon me`.

Un hablante de irlandés que traduce al inglés diría `I feel happy`, no `Happy is upon me`, porque entiende el significado de la oración, aunque las palabras y la estructura de la oración sean diferentes.

El orden formal para la oración en irlandés es:

- **verbo** (Tá o is)
- **adjetivo** (athas, o happy)
- **sujeto** (orm, o upon me)

## Traducción

Un programa de traducción ingenuo podría traducir solo palabras, ignorando la estructura de la oración.

✅ Si has aprendido un segundo (o tercer o más) idioma como adulto, es posible que hayas comenzado pensando en tu idioma nativo, traduciendo un concepto palabra por palabra en tu cabeza al segundo idioma, y luego hablando tu traducción. Esto es similar a lo que hacen los programas de traducción ingenuos. ¡Es importante superar esta fase para alcanzar la fluidez!

La traducción ingenua lleva a malas (y a veces hilarantes) malas traducciones: `I feel happy` se traduce literalmente como `Mise bhraitheann athas` en irlandés. Eso significa (literalmente) `me feel happy` y no es una oración válida en irlandés. Aunque el inglés y el irlandés son idiomas hablados en dos islas vecinas muy cercanas, son idiomas muy diferentes con estructuras gramaticales diferentes.

> Puedes ver algunos videos sobre las tradiciones lingüísticas irlandesas como [este](https://www.youtube.com/watch?v=mRIaLSdRMMs)

### Enfoques de aprendizaje automático

Hasta ahora, has aprendido sobre el enfoque de reglas formales para el procesamiento del lenguaje natural. Otro enfoque es ignorar el significado de las palabras y _en su lugar usar el aprendizaje automático para detectar patrones_. Esto puede funcionar en la traducción si tienes muchos textos (un *corpus*) o textos (*corpora*) en ambos idiomas de origen y objetivo.

Por ejemplo, considera el caso de *Orgullo y Prejuicio*, una novela inglesa bien conocida escrita por Jane Austen en 1813. Si consultas el libro en inglés y una traducción humana del libro en *francés*, podrías detectar frases en uno que se traducen _idiomáticamente_ en el otro. Lo harás en un momento.

Por ejemplo, cuando una frase en inglés como `I have no money` se traduce literalmente al francés, podría convertirse en `Je n'ai pas de monnaie`. "Monnaie" es un falso amigo francés complicado, ya que 'money' y 'monnaie' no son sinónimos. Una mejor traducción que un humano podría hacer sería `Je n'ai pas d'argent`, porque transmite mejor el significado de que no tienes dinero (en lugar de 'cambio suelto' que es el significado de 'monnaie').

![monnaie](../../../../translated_images/monnaie.606c5fa8369d5c3b3031ef0713e2069485c87985dd475cd9056bdf4c76c1f4b8.es.png)

> Imagen de [Jen Looper](https://twitter.com/jenlooper)

Si un modelo de ML tiene suficientes traducciones humanas para construir un modelo, puede mejorar la precisión de las traducciones identificando patrones comunes en textos que han sido previamente traducidos por hablantes humanos expertos de ambos idiomas.

### Ejercicio - traducción

Puedes usar `TextBlob` para traducir oraciones. Prueba la famosa primera línea de **Orgullo y Prejuicio**:

```python
from textblob import TextBlob

blob = TextBlob(
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife!"
)
print(blob.translate(to="fr"))

```

`TextBlob` hace un buen trabajo con la traducción: "C'est une vérité universellement reconnue, qu'un homme célibataire en possession d'une bonne fortune doit avoir besoin d'une femme!".

Se puede argumentar que la traducción de TextBlob es mucho más exacta, de hecho, que la traducción francesa de 1932 del libro por V. Leconte y Ch. Pressoir:

"C'est une vérité universelle qu'un célibataire pourvu d'une belle fortune doit avoir envie de se marier, et, si peu que l'on sache de son sentiment à cet egard, lorsqu'il arrive dans une nouvelle résidence, cette idée est si bien fixée dans l'esprit de ses voisins qu'ils le considèrent sur-le-champ comme la propriété légitime de l'une ou l'autre de leurs filles."

En este caso, la traducción informada por ML hace un mejor trabajo que el traductor humano que está poniendo palabras innecesarias en la boca del autor original para 'claridad'.

> ¿Qué está pasando aquí? y ¿por qué TextBlob es tan bueno en la traducción? Bueno, detrás de escena, está usando Google Translate, una IA sofisticada capaz de analizar millones de frases para predecir las mejores cadenas para la tarea en cuestión. No hay nada manual en esto y necesitas una conexión a internet para usar `blob.translate`.

✅ Try some more sentences. Which is better, ML or human translation? In which cases?

## Sentiment analysis

Another area where machine learning can work very well is sentiment analysis. A non-ML approach to sentiment is to identify words and phrases which are 'positive' and 'negative'. Then, given a new piece of text, calculate the total value of the positive, negative and neutral words to identify the overall sentiment. 

This approach is easily tricked as you may have seen in the Marvin task - the sentence `Great, that was a wonderful waste of time, I'm glad we are lost on this dark road` es una oración sarcástica y de sentimiento negativo, pero el algoritmo simple detecta 'great', 'wonderful', 'glad' como positivas y 'waste', 'lost' y 'dark' como negativas. El sentimiento general se ve influenciado por estas palabras contradictorias.

✅ Detente un segundo y piensa en cómo transmitimos sarcasmo como hablantes humanos. La inflexión del tono juega un papel importante. Intenta decir la frase "Well, that film was awesome" de diferentes maneras para descubrir cómo tu voz transmite significado.

### Enfoques de ML

El enfoque de ML sería recopilar manualmente cuerpos de texto negativos y positivos: tweets, reseñas de películas, o cualquier cosa donde el humano haya dado una puntuación *y* una opinión escrita. Luego, se pueden aplicar técnicas de NLP a las opiniones y puntuaciones, de modo que surjan patrones (por ejemplo, las reseñas de películas positivas tienden a tener la frase 'Oscar worthy' más que las reseñas de películas negativas, o las reseñas de restaurantes positivos dicen 'gourmet' mucho más que 'disgusting').

> ⚖️ **Ejemplo**: Si trabajas en la oficina de un político y se está debatiendo una nueva ley, los constituyentes podrían escribir a la oficina con correos electrónicos a favor o en contra de la nueva ley en particular. Supongamos que te encargan leer los correos electrónicos y clasificarlos en 2 montones, *a favor* y *en contra*. Si hubiera muchos correos electrónicos, podrías sentirte abrumado intentando leerlos todos. ¿No sería genial si un bot pudiera leerlos todos por ti, entenderlos y decirte en qué montón pertenece cada correo electrónico? 
> 
> Una forma de lograr eso es usar Machine Learning. Entrenarías el modelo con una porción de los correos electrónicos *en contra* y una porción de los correos electrónicos *a favor*. El modelo tendería a asociar frases y palabras con el lado en contra y el lado a favor, *pero no entendería ninguno de los contenidos*, solo que ciertas palabras y patrones eran más probables de aparecer en un correo electrónico *en contra* o *a favor*. Podrías probarlo con algunos correos electrónicos que no usaste para entrenar el modelo y ver si llegaba a la misma conclusión que tú. Luego, una vez que estuvieras satisfecho con la precisión del modelo, podrías procesar futuros correos electrónicos sin tener que leer cada uno.

✅ ¿Este proceso te suena similar a los procesos que has utilizado en lecciones anteriores?

## Ejercicio - oraciones sentimentales

El sentimiento se mide con una *polaridad* de -1 a 1, lo que significa que -1 es el sentimiento más negativo y 1 es el más positivo. El sentimiento también se mide con una puntuación de 0 a 1 para objetividad (0) y subjetividad (1).

Echa otro vistazo a *Orgullo y Prejuicio* de Jane Austen. El texto está disponible aquí en [Project Gutenberg](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm). El siguiente ejemplo muestra un programa corto que analiza el sentimiento de las primeras y últimas oraciones del libro y muestra su polaridad de sentimiento y puntuación de subjetividad/objetividad.

Deberías usar la biblioteca `TextBlob` (descrita arriba) para determinar `sentimiento` (no tienes que escribir tu propio calculador de sentimientos) en la siguiente tarea.

```python
from textblob import TextBlob

quote1 = """It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife."""

quote2 = """Darcy, as well as Elizabeth, really loved them; and they were both ever sensible of the warmest gratitude towards the persons who, by bringing her into Derbyshire, had been the means of uniting them."""

sentiment1 = TextBlob(quote1).sentiment
sentiment2 = TextBlob(quote2).sentiment

print(quote1 + " has a sentiment of " + str(sentiment1))
print(quote2 + " has a sentiment of " + str(sentiment2))
```

Ves la siguiente salida:

```output
It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want # of a wife. has a sentiment of Sentiment(polarity=0.20952380952380953, subjectivity=0.27142857142857146)

Darcy, as well as Elizabeth, really loved them; and they were
     both ever sensible of the warmest gratitude towards the persons
      who, by bringing her into Derbyshire, had been the means of
      uniting them. has a sentiment of Sentiment(polarity=0.7, subjectivity=0.8)
```

## Desafío - comprobar la polaridad del sentimiento

Tu tarea es determinar, utilizando la polaridad del sentimiento, si *Orgullo y Prejuicio* tiene más oraciones absolutamente positivas que absolutamente negativas. Para esta tarea, puedes asumir que una puntuación de polaridad de 1 o -1 es absolutamente positiva o negativa respectivamente.

**Pasos:**

1. Descarga una [copia de Orgullo y Prejuicio](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm) de Project Gutenberg como un archivo .txt. Elimina los metadatos al principio y al final del archivo, dejando solo el texto original.
2. Abre el archivo en Python y extrae el contenido como una cadena.
3. Crea un TextBlob usando la cadena del libro.
4. Analiza cada oración en el libro en un bucle.
   1. Si la polaridad es 1 o -1, almacena la oración en una matriz o lista de mensajes positivos o negativos.
5. Al final, imprime todas las oraciones positivas y negativas (por separado) y el número de cada una.

Aquí hay una [solución de ejemplo](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/3-Translation-Sentiment/solution/notebook.ipynb).

✅ Comprobación de Conocimientos

1. El sentimiento se basa en las palabras utilizadas en la oración, pero ¿entiende el código *las palabras*?
2. ¿Crees que la polaridad del sentimiento es precisa o, en otras palabras, estás *de acuerdo* con las puntuaciones?
   1. En particular, ¿estás de acuerdo o en desacuerdo con la polaridad **positiva** absoluta de las siguientes oraciones?
      * “What an excellent father you have, girls!” said she, when the door was shut.
      * “Your examination of Mr. Darcy is over, I presume,” said Miss Bingley; “and pray what is the result?” “I am perfectly convinced by it that Mr. Darcy has no defect.
      * How wonderfully these sort of things occur!
      * I have the greatest dislike in the world to that sort of thing.
      * Charlotte is an excellent manager, I dare say.
      * “This is delightful indeed!
      * I am so happy!
      * Your idea of the ponies is delightful.
   2. Las siguientes 3 oraciones fueron puntuadas con un sentimiento absolutamente positivo, pero al leerlas detenidamente, no son oraciones positivas. ¿Por qué el análisis de sentimiento pensó que eran oraciones positivas?
      * Happy shall I be, when his stay at Netherfield is over!” “I wish I could say anything to comfort you,” replied Elizabeth; “but it is wholly out of my power.
      * If I could but see you as happy!
      * Our distress, my dear Lizzy, is very great.
   3. ¿Estás de acuerdo o en desacuerdo con la polaridad **negativa** absoluta de las siguientes oraciones?
      - Everybody is disgusted with his pride.
      - “I should like to know how he behaves among strangers.” “You shall hear then—but prepare yourself for something very dreadful.
      - The pause was to Elizabeth’s feelings dreadful.
      - It would be dreadful!

✅ Cualquier aficionado a Jane Austen entenderá que ella a menudo usa sus libros para criticar los aspectos más ridículos de la sociedad de la Regencia inglesa. Elizabeth Bennett, el personaje principal en *Orgullo y Prejuicio*, es una observadora social perspicaz (como la autora) y su lenguaje a menudo está muy matizado. Incluso Mr. Darcy (el interés amoroso en la historia) nota el uso juguetón y burlón del lenguaje de Elizabeth: "He tenido el placer de conocerte lo suficiente como para saber que disfrutas mucho ocasionalmente profesando opiniones que de hecho no son tuyas."

---

## 🚀Desafío

¿Puedes mejorar a Marvin aún más extrayendo otras características de la entrada del usuario?

## [Cuestionario posterior a la clase](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/36/)

## Revisión y Autoestudio

Hay muchas maneras de extraer sentimientos de un texto. Piensa en las aplicaciones comerciales que podrían hacer uso de esta técnica. Piensa en cómo puede salir mal. Lee más sobre sistemas empresariales sofisticados que analizan sentimientos como [Azure Text Analysis](https://docs.microsoft.com/azure/cognitive-services/Text-Analytics/how-tos/text-analytics-how-to-sentiment-analysis?tabs=version-3-1?WT.mc_id=academic-77952-leestott). Prueba algunas de las oraciones de Orgullo y Prejuicio anteriores y ve si puede detectar matices.

## Asignación

[Licencia poética](assignment.md)

        **Descargo de responsabilidad**: 
        Este documento ha sido traducido utilizando servicios de traducción automática basados en inteligencia artificial. Aunque nos esforzamos por lograr precisión, tenga en cuenta que las traducciones automáticas pueden contener errores o inexactitudes. El documento original en su idioma nativo debe considerarse la fuente autorizada. Para información crítica, se recomienda una traducción humana profesional. No somos responsables de ningún malentendido o interpretación errónea que surja del uso de esta traducción.