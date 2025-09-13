<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1c2ec40cf55c98a028a359c27ef7e45a",
  "translation_date": "2025-09-04T22:28:35+00:00",
  "source_file": "6-NLP/1-Introduction-to-NLP/README.md",
  "language_code": "es"
}
-->
# Introducción al procesamiento de lenguaje natural

Esta lección cubre una breve historia y conceptos importantes del *procesamiento de lenguaje natural*, un subcampo de la *lingüística computacional*.

## [Cuestionario previo a la lección](https://ff-quizzes.netlify.app/en/ml/)

## Introducción

El procesamiento de lenguaje natural, conocido como NLP por sus siglas en inglés, es una de las áreas más conocidas donde se ha aplicado el aprendizaje automático y se utiliza en software de producción.

✅ ¿Puedes pensar en algún software que uses todos los días que probablemente tenga algo de NLP integrado? ¿Qué hay de tus programas de procesamiento de texto o aplicaciones móviles que usas regularmente?

Aprenderás sobre:

- **La idea de los idiomas**. Cómo se desarrollaron los idiomas y cuáles han sido las principales áreas de estudio.
- **Definición y conceptos**. También aprenderás definiciones y conceptos sobre cómo las computadoras procesan texto, incluyendo análisis sintáctico, gramática e identificación de sustantivos y verbos. Hay algunas tareas de codificación en esta lección, y se introducen varios conceptos importantes que aprenderás a programar más adelante en las próximas lecciones.

## Lingüística computacional

La lingüística computacional es un área de investigación y desarrollo que, durante muchas décadas, ha estudiado cómo las computadoras pueden trabajar con los idiomas, e incluso entenderlos, traducirlos y comunicarse con ellos. El procesamiento de lenguaje natural (NLP) es un campo relacionado que se centra en cómo las computadoras pueden procesar idiomas 'naturales', es decir, humanos.

### Ejemplo - dictado en el teléfono

Si alguna vez has dictado a tu teléfono en lugar de escribir o le has hecho una pregunta a un asistente virtual, tu voz fue convertida en texto y luego procesada o *analizada* desde el idioma que hablaste. Las palabras clave detectadas se procesaron en un formato que el teléfono o asistente pudo entender y actuar en consecuencia.

![comprensión](../../../../6-NLP/1-Introduction-to-NLP/images/comprehension.png)
> ¡La comprensión lingüística real es difícil! Imagen por [Jen Looper](https://twitter.com/jenlooper)

### ¿Cómo es posible esta tecnología?

Esto es posible porque alguien escribió un programa de computadora para hacerlo. Hace unas décadas, algunos escritores de ciencia ficción predijeron que las personas hablarían principalmente con sus computadoras y que estas siempre entenderían exactamente lo que querían decir. Lamentablemente, resultó ser un problema más difícil de lo que muchos imaginaron, y aunque hoy en día es un problema mucho mejor entendido, existen desafíos significativos para lograr un procesamiento de lenguaje natural 'perfecto' cuando se trata de entender el significado de una oración. Este es un problema particularmente difícil cuando se trata de entender el humor o detectar emociones como el sarcasmo en una oración.

En este punto, podrías estar recordando las clases escolares donde el profesor cubría las partes de la gramática en una oración. En algunos países, se enseña gramática y lingüística como una materia dedicada, pero en muchos, estos temas se incluyen como parte del aprendizaje de un idioma: ya sea tu primer idioma en la escuela primaria (aprendiendo a leer y escribir) y quizás un segundo idioma en la escuela secundaria. ¡No te preocupes si no eres un experto en diferenciar sustantivos de verbos o adverbios de adjetivos!

Si tienes dificultades con la diferencia entre el *presente simple* y el *presente progresivo*, no estás solo. Esto es algo desafiante para muchas personas, incluso hablantes nativos de un idioma. La buena noticia es que las computadoras son muy buenas aplicando reglas formales, y aprenderás a escribir código que pueda *analizar* una oración tan bien como un humano. El mayor desafío que examinarás más adelante es entender el *significado* y el *sentimiento* de una oración.

## Prerrequisitos

Para esta lección, el principal prerrequisito es poder leer y entender el idioma de esta lección. No hay problemas matemáticos ni ecuaciones que resolver. Aunque el autor original escribió esta lección en inglés, también está traducida a otros idiomas, por lo que podrías estar leyendo una traducción. Hay ejemplos donde se utilizan varios idiomas diferentes (para comparar las diferentes reglas gramaticales de distintos idiomas). Estos *no* están traducidos, pero el texto explicativo sí lo está, por lo que el significado debería ser claro.

Para las tareas de codificación, usarás Python y los ejemplos están en Python 3.8.

En esta sección, necesitarás y usarás:

- **Comprensión de Python 3**. Comprensión del lenguaje de programación en Python 3, esta lección utiliza entrada, bucles, lectura de archivos, arreglos.
- **Visual Studio Code + extensión**. Usaremos Visual Studio Code y su extensión de Python. También puedes usar un IDE de Python de tu elección.
- **TextBlob**. [TextBlob](https://github.com/sloria/TextBlob) es una biblioteca simplificada de procesamiento de texto para Python. Sigue las instrucciones en el sitio de TextBlob para instalarlo en tu sistema (instala también los corpora, como se muestra a continuación):

   ```bash
   pip install -U textblob
   python -m textblob.download_corpora
   ```

> 💡 Consejo: Puedes ejecutar Python directamente en entornos de VS Code. Consulta los [documentos](https://code.visualstudio.com/docs/languages/python?WT.mc_id=academic-77952-leestott) para más información.

## Hablando con máquinas

La historia de intentar que las computadoras entiendan el lenguaje humano se remonta a décadas atrás, y uno de los primeros científicos en considerar el procesamiento de lenguaje natural fue *Alan Turing*.

### La 'prueba de Turing'

Cuando Turing investigaba la *inteligencia artificial* en la década de 1950, consideró si se podría realizar una prueba conversacional entre un humano y una computadora (a través de correspondencia escrita) donde el humano en la conversación no estuviera seguro de si estaba conversando con otro humano o con una computadora.

Si, después de cierto tiempo de conversación, el humano no podía determinar si las respuestas provenían de una computadora o no, ¿podría decirse que la computadora estaba *pensando*?

### La inspiración - 'el juego de imitación'

La idea para esto provino de un juego de fiesta llamado *El juego de imitación*, donde un interrogador está solo en una habitación y tiene la tarea de determinar cuál de dos personas (en otra habitación) es hombre y cuál es mujer. El interrogador puede enviar notas y debe tratar de pensar en preguntas cuyas respuestas escritas revelen el género de la persona misteriosa. Por supuesto, los jugadores en la otra habitación intentan engañar al interrogador respondiendo preguntas de manera que lo confundan o lo engañen, mientras dan la apariencia de responder honestamente.

### Desarrollando Eliza

En la década de 1960, un científico del MIT llamado *Joseph Weizenbaum* desarrolló [*Eliza*](https://wikipedia.org/wiki/ELIZA), una 'terapeuta' computarizada que hacía preguntas al humano y daba la apariencia de entender sus respuestas. Sin embargo, aunque Eliza podía analizar una oración e identificar ciertos constructos gramaticales y palabras clave para dar una respuesta razonable, no podía decirse que *entendiera* la oración. Si a Eliza se le presentaba una oración con el formato "**Yo estoy** <u>triste</u>", podría reorganizar y sustituir palabras en la oración para formar la respuesta "¿Cuánto tiempo has **estado** <u>triste</u>?".

Esto daba la impresión de que Eliza entendía la declaración y estaba haciendo una pregunta de seguimiento, mientras que en realidad estaba cambiando el tiempo verbal y agregando algunas palabras. Si Eliza no podía identificar una palabra clave para la que tuviera una respuesta, en su lugar daba una respuesta aleatoria que podría aplicarse a muchas declaraciones diferentes. Eliza podía ser fácilmente engañada, por ejemplo, si un usuario escribía "**Tú eres** una <u>bicicleta</u>", podría responder con "¿Cuánto tiempo he **sido** una <u>bicicleta</u>?", en lugar de una respuesta más razonada.

[![Conversando con Eliza](https://img.youtube.com/vi/RMK9AphfLco/0.jpg)](https://youtu.be/RMK9AphfLco "Conversando con Eliza")

> 🎥 Haz clic en la imagen de arriba para ver un video sobre el programa original de ELIZA

> Nota: Puedes leer la descripción original de [Eliza](https://cacm.acm.org/magazines/1966/1/13317-elizaa-computer-program-for-the-study-of-natural-language-communication-between-man-and-machine/abstract) publicada en 1966 si tienes una cuenta de ACM. Alternativamente, lee sobre Eliza en [Wikipedia](https://wikipedia.org/wiki/ELIZA).

## Ejercicio - programando un bot conversacional básico

Un bot conversacional, como Eliza, es un programa que solicita la entrada del usuario y parece entender y responder de manera inteligente. A diferencia de Eliza, nuestro bot no tendrá varias reglas que le den la apariencia de tener una conversación inteligente. En cambio, nuestro bot tendrá una sola habilidad: mantener la conversación con respuestas aleatorias que podrían funcionar en casi cualquier conversación trivial.

### El plan

Tus pasos al construir un bot conversacional:

1. Imprimir instrucciones que aconsejen al usuario cómo interactuar con el bot.
2. Iniciar un bucle:
   1. Aceptar la entrada del usuario.
   2. Si el usuario ha pedido salir, entonces salir.
   3. Procesar la entrada del usuario y determinar la respuesta (en este caso, la respuesta es una elección aleatoria de una lista de posibles respuestas genéricas).
   4. Imprimir la respuesta.
3. Volver al paso 2.

### Construyendo el bot

Vamos a crear el bot a continuación. Comenzaremos definiendo algunas frases.

1. Crea este bot tú mismo en Python con las siguientes respuestas aleatorias:

    ```python
    random_responses = ["That is quite interesting, please tell me more.",
                        "I see. Do go on.",
                        "Why do you say that?",
                        "Funny weather we've been having, isn't it?",
                        "Let's change the subject.",
                        "Did you catch the game last night?"]
    ```

    Aquí hay un ejemplo de salida para guiarte (la entrada del usuario está en las líneas que comienzan con `>`):

    ```output
    Hello, I am Marvin, the simple robot.
    You can end this conversation at any time by typing 'bye'
    After typing each answer, press 'enter'
    How are you today?
    > I am good thanks
    That is quite interesting, please tell me more.
    > today I went for a walk     
    Did you catch the game last night?
    > I did, but my team lost
    Funny weather we've been having, isn't it?
    > yes but I hope next week is better
    Let's change the subject.
    > ok, lets talk about music
    Why do you say that?
    > because I like music!
    Why do you say that?
    > bye
    It was nice talking to you, goodbye!
    ```

    Una posible solución a la tarea está [aquí](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/1-Introduction-to-NLP/solution/bot.py).

    ✅ Detente y reflexiona

    1. ¿Crees que las respuestas aleatorias podrían 'engañar' a alguien para que piense que el bot realmente las entendió?
    2. ¿Qué características necesitaría el bot para ser más efectivo?
    3. Si un bot realmente pudiera 'entender' el significado de una oración, ¿necesitaría 'recordar' el significado de oraciones anteriores en una conversación también?

---

## 🚀Desafío

Elige uno de los elementos de "detente y reflexiona" anteriores y trata de implementarlo en código o escribe una solución en papel usando pseudocódigo.

En la próxima lección, aprenderás sobre una serie de otros enfoques para analizar el lenguaje natural y el aprendizaje automático.

## [Cuestionario posterior a la lección](https://ff-quizzes.netlify.app/en/ml/)

## Revisión y autoestudio

Echa un vistazo a las referencias a continuación como oportunidades de lectura adicional.

### Referencias

1. Schubert, Lenhart, "Computational Linguistics", *The Stanford Encyclopedia of Philosophy* (Spring 2020 Edition), Edward N. Zalta (ed.), URL = <https://plato.stanford.edu/archives/spr2020/entries/computational-linguistics/>.
2. Princeton University "About WordNet." [WordNet](https://wordnet.princeton.edu/). Princeton University. 2010.

## Tarea

[Busca un bot](assignment.md)

---

**Descargo de responsabilidad**:  
Este documento ha sido traducido utilizando el servicio de traducción automática [Co-op Translator](https://github.com/Azure/co-op-translator). Si bien nos esforzamos por garantizar la precisión, tenga en cuenta que las traducciones automatizadas pueden contener errores o imprecisiones. El documento original en su idioma nativo debe considerarse la fuente autorizada. Para información crítica, se recomienda una traducción profesional realizada por humanos. No nos hacemos responsables de malentendidos o interpretaciones erróneas que puedan surgir del uso de esta traducción.