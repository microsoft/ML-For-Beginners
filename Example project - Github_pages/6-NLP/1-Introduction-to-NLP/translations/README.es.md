# Introducción al procesamiento del lenguaje natural

Esta lección cubre una breve historia y conceptos importante del *procesamiento del lenguaje natural*, un subcampo de la *ligüística computacional*.

## [Examen previo a la lección](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/31?loc=es)

## Introducción

NLP, como se conoce comúnmente, es una de las áreas más conocidas donde se ha aplicado y usado el aprendizaje automático en software de producción.

✅ ¿Puedes pensar en software que usas cada d´ia que probablemente tenga algo de NLP embebido? ¿Qué hay de tus programas de procesamiento de palabras o aplicaciones web que usas regularmente?

Aprenderás acerca de:

- **La idea de los idiomas**. Cómo se desarrollaron los idiomas y cuáles han sido las mayores áreas de estudio.
- **Definiciones y conceptos**. También aprenderás definiciones y conceptos acerca de cómo procesan texto las computadoras, incluyendo análisis, gramática e identificación de sustantivos y verbos. Hay algunas tareas de codificación en esta lección, y  se presentan varios conceptos importantes que aprenderás a codificar posteriormente en las próximas lecciones.

## Lingüística computacional

La lingüística computacional es una área de la investigación y desarrollo que por varias décadas ha estudiado cómo pueden trabajar las computadoras e incluso entender, traducir y comunicar con idiomas. El procesamiento del lenguaje natural es un campo relacionado que se enfoca en cómo las computadoras pueden procesar el lenguaje 'natural' o humano.

### Ejemplo - dictado telefónico

Si alguna vez has dictado a tu teléfono en lugar de escribir o hacerle una pregunta al asistente virtual, tu voz se convirtió a texto y luego fue procesada o *parseada* desde el idioma que hablaste. Las palabras clave detectadas fueron procesadas en un formato que el teléfono o asistente pueda entender y actuar.

![Comprensión](../images/comprehension.png)
> ¡La comprensión lingüística real es difícil! Imagen de [Jen Looper](https://twitter.com/jenlooper)

### ¿Cómo es posible esta tecnología?

Es posible porque alguien escribió un programa de computadora que lo hace. Hace algunas décadas, algunos escritores de ciencia ficción predijeron que la gente hablaría regularmente con sus computadoras, y las computadoras siempre entenderían exactamente lo que éstas quieren. Tristemente, resultó ser un problema más complejo del que se imaginó, y aunque es un problema mejor comprendido ahora, hay desafíos significativos en lograr un 'perfecto' procesamiento del lenguaje natural cuando se trata de entender el significado de una oración. Este es un problema particularmente difícil cuando se trata de entender el humor o detectar las emociones tal como el sarcasmo en una oración.

En este punto, recordarás las clases escolares donde el profesor cubría las partes de la gramática en una oración. En algunos países, los estudiantes aprenden gramática y lingüística como una materia dedicada, pero en varios casos, estos temas se incluyen como parte del aprendizaje de un idioma: ya sea tu primer idioma en la escuela primaria (aprendiendo a leer y escribir) y quizá como un segundo idioma en la escuela secundaria o la preparatoria. ¡No te preocupes si no eres un experto diferenciando sustantivos de verbos o adverbios de adjetivos!

Si tienes problemas diferenciando entre *presente simple* y *presente continuo*, no estás solo. Esto es un algo desafiante para mucha gente, incluso hablantes nativos de un idioma. La buena noticia es que las computadoras son muy buenas aplicando reglas formales, y aprenderás a escribir código que puede *parsear* una oración tan bien como un humano. El mayor desafío que examinarás más adelante es el entender el *significado* y *sentimiento* de una oración.

## Prerrequisitos

Para esta lección, el prerrequisito principal es ser capaz de leer y comprender el idioma de esta lección. No hay problemas matemáticos ni ecuaciones a resolver. Aunque el actor original escribió esta lección en Inglés, también está traducida a otros idiomas, por lo que podrías leer la traducción. Hay ejemplos donde se usan un número distinto de idiomas (para comparar las distintas reglas gramaticales de los distintos idiomas). Estas *no* son traducidas, pero su texto explicativo sí, así que el significado debería ser claro.

Para las tareas de programación, usarás Python y los ejemplos usan Python 3.8.

En esta sección, necesitarás y usarás:

- **Comprensión de Python 3**. Comprensión del lenguaje de programación Python 3, esta lección usa entradas, ciclos, lectura de archivos, arreglos.
- **Visual Studio Code + extensión**. Usaremos Visual Studio Code y su extensión para Python. También puedes usar algún IDE para Python de tu elección.
- **TextBlob**. [TextBlob](https://github.com/sloria/TextBlob) es una biblioteca de procesamiento de texto simplificada para Python. Sigue las instrucciones en el sitio de TextBlob para instalarla en tu sistema (también instala corpora, como se muestra abajo):

   ```bash
   pip install -U textblob
   python -m textblob.download_corpora
   ```

> 💡 Consejo: Puedes ejecutar Python directamente en los ambientes de VS Code. Revisa la [documentación](https://code.visualstudio.com/docs/languages/python?WT.mc_id=academic-77952-leestott) para mayor información.

## Hablando con las máquinas

La historia de intentar hacer que las computadoras comprendan el lenguaje humano se remota a décadas atrás, y uno de los primeros científicos en considerar el procesamiento del lenguaje natural fue *Alan Turing*.

### La 'prueba de Turing'

Cuando Turing estaba investigando *la inteligencia artificial* en los años 1950, considero que si una prueba conversacional pudiera ser proporcionada a un humano y una computadora (a través de correspondencia mecanografiada) donde el humano en la conversación no estuviese seguro si estuviesen conversando con otro humano o una computadora.

Si, después de cierto tiempo de conversación, el humano no pudiese determinar si las respuestas provinieron de una computadora o no, entonces pudiese decirse que la computadora estaba *pensando*?

### La inspiración - 'el juego de imitación'

La idea para este juego provino de una juego de fiesta llamado *El juego de imitación* donde un interrogador está solo en una habitación y tiene como objetivo determinar cuál de las dos personas (en otra habitación) son hombres y mujeres, respectivamente. El interrogador puede enviar notas, y debe intentar pensar en preguntas donde las respuestas escritas revelen el género de la persona misteriosa. Desde luego, los jugadores en la otra habitación intentan engañar al interrogador al responder a sus preguntas de tal forma que engañen o confundan al interrogador, pero dando la apariencia de responder honestamente.

### Desarrollando a Eliza

En los años 1960s un científico del MIT llamado *Joseph Weizenbaum* desarrolló a [*Eliza*](https://wikipedia.org/wiki/ELIZA), un 'terapeuta' de computadora que realiza preguntas a los humanos y da la sensación de entender sus respuestas. Sin embargo, mientras Eliza podía analizar una oración e identificar ciertas construcciones gramaticales y palabras clase para así darles respuestas razonables, no debería decirse  *entender* la oración. Si a Eliza le fuera presentada una oración con el siguiente formato "**I am** <u>sad</u>" podría reorganizar y sustituir palabras en la oración para formar la respuesta "How long have **you been** <u>sad</u>".

Esto daba la impresión que Eliza entendió la oración y le fue hecha una pregunta de seguimiento, aunque en realidad, cambió el tiempo verbal y agregó algunas palabras. Si Eliza no podía identificar una palabra clave para la cual tenía una respuesta, en su lugar daría una respuesta aleatoria que pudiese ser aplicable a distintas oraciones. Eliza podía ser engañada fácilmente, por ejemplo si un usuario escribió "**You are** a <u>bicycle</u>" podría responder con "How long have **I been** a <u>bicycle</u>?", en lugar de una respuesta más elaborada.

[![Chateando con Eliza](https://img.youtube.com/vi/RMK9AphfLco/0.jpg)](https://youtu.be/RMK9AphfLco "Chateando con Eliza")

> 🎥 Da clic en la imagen de arriba para ver el video del programa original ELIZA

> Nota: Puedes leer la descripción original de [Eliza](https://cacm.acm.org/magazines/1966/1/13317-elizaa-computer-program-for-the-study-of-natural-language-communication-between-man-and-machine/abstract) publicado en 1966 si tienes una cuenta ACM, lee acerca de Eliza en [wikipedia](https://wikipedia.org/wiki/ELIZA)

## Ejercicio - Programar un bot conversacional básico

Un bot conversacional, como Eliza, es una programa que obtiene entradas del usuario y parece entender y responder inteligentemente. A diferencia de Eliza, nuestro bot no tendrá varias reglas dándole la apariencia de tener una conversación inteligente. En su lugar, nuestro bot tendrá sólo una habilidad, mantener la conversación con respuestas aleatorias que funcionen en casi cualquier conversación trivial.

### El plan


Tus pasos para construir un bot conversacional:

1. Imprime instrucciones asesorando al usuario cómo interactuar con el bot
2. Empieza un ciclo
   1. Acepta la entrada del usuario
   2. Si el usuario pidió salir, entonces sal
   3. Procesa la entrada del usuario y determina la respuesta (en este caso, la respuesta es una elección aleatoria de una lista de posibles respuestas genéricas)
   4. Imprime la respuesta
3. Vuelve al paso 2

### Construye el bot

Ahora creemos el bot. Iniciaremos definiendo algunas frases.

1. Crea este bot tú mismo en Python con las siguientes respuestas aleatorias:

    ```python
    random_responses = ["That is quite interesting, please tell me more.",
                        "I see. Do go on.",
                        "Why do you say that?",
                        "Funny weather we've been having, isn't it?",
                        "Let's change the subject.",
                        "Did you catch the game last night?"]
    ```

    Aquí tienes unas salidas de ejemplo para guiarte (la entrada del usuario está en las líneas que empiezan con `>`):

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

    [Aquí](../solution/bot.py) tienes una posible solución a la tarea.

    ✅ Detente y considera

    1. ¿Crees que las respuestas aleatorias podrías 'engañar' a alguien haciéndole pensar que el bot realmente los entendió?
    2. ¿Qué características necesitaría el bot para ser más efectivo?
    3. Si el bot pudiera 'entender' realmente el significado de una oración, ¿también necesitaría 'recordar' el significado de oraciones anteriores en una conversación?

---

## 🚀Desafío

Elige uno de los elementos "Detente y considera" de arriba y trata de implementarlos en código o escribe una solución en papel usando pseudo-código.

En la siguiente lección, aprenderás acerca de otros enfoques de cómo analizar el lenguaje natural y aprendizaje automático.

## [Examen posterior a la lección](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/32?loc=es)

## Revisión y autoestudio

Da un vistazo a las referencias abajo para más oportunidades de lectura.

### Referencias

1. Schubert, Lenhart, "Computational Linguistics", *The Stanford Encyclopedia of Philosophy* (Spring 2020 Edition), Edward N. Zalta (ed.), URL = <https://plato.stanford.edu/archives/spr2020/entries/computational-linguistics/>.
2. Princeton University "About WordNet." [WordNet](https://wordnet.princeton.edu/). Princeton University. 2010.

## Asignación

[Buscar un bot](assignment.es.md)
