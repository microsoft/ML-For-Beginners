# Introducción al procesamiento de lenguaje natural

Esta lección cubre una breve historia y conceptos importantes del *procesamiento de lenguaje natural*, un subcampo de la *lingüística computacional*.

## [Cuestionario previo a la lección](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/31/)

## Introducción

El procesamiento de lenguaje natural (NLP, por sus siglas en inglés), como se le conoce comúnmente, es una de las áreas más conocidas donde se ha aplicado el aprendizaje automático y se utiliza en software de producción.

✅ ¿Puedes pensar en algún software que uses todos los días que probablemente tenga algo de NLP integrado? ¿Qué hay de tus programas de procesamiento de texto o aplicaciones móviles que usas regularmente?

Aprenderás sobre:

- **La idea de los lenguajes**. Cómo se desarrollaron los lenguajes y cuáles han sido las principales áreas de estudio.
- **Definición y conceptos**. También aprenderás definiciones y conceptos sobre cómo las computadoras procesan texto, incluyendo análisis sintáctico, gramática e identificación de sustantivos y verbos. Hay algunas tareas de codificación en esta lección, y se introducen varios conceptos importantes que aprenderás a codificar más adelante en las próximas lecciones.

## Lingüística computacional

La lingüística computacional es un área de investigación y desarrollo de muchas décadas que estudia cómo las computadoras pueden trabajar con, e incluso entender, traducir y comunicarse con los lenguajes. El procesamiento de lenguaje natural (NLP) es un campo relacionado enfocado en cómo las computadoras pueden procesar lenguajes 'naturales', o humanos.

### Ejemplo - dictado por teléfono

Si alguna vez has dictado a tu teléfono en lugar de escribir o has hecho una pregunta a un asistente virtual, tu discurso se convirtió en una forma de texto y luego se procesó o *analizó* desde el idioma que hablaste. Las palabras clave detectadas se procesaron luego en un formato que el teléfono o asistente pudiera entender y actuar en consecuencia.

![comprensión](../../../../translated_images/comprehension.619708fc5959b0f6a24ebffba2ad7b0625391a476141df65b43b59de24e45c6f.es.png)
> ¡La comprensión lingüística real es difícil! Imagen de [Jen Looper](https://twitter.com/jenlooper)

### ¿Cómo es posible esta tecnología?

Esto es posible porque alguien escribió un programa de computadora para hacerlo. Hace unas décadas, algunos escritores de ciencia ficción predijeron que la gente hablaría principalmente con sus computadoras, y las computadoras siempre entenderían exactamente lo que querían decir. Lamentablemente, resultó ser un problema más difícil de lo que muchos imaginaron, y aunque hoy en día es un problema mucho mejor comprendido, existen desafíos significativos para lograr un procesamiento de lenguaje natural 'perfecto' cuando se trata de entender el significado de una oración. Esto es particularmente difícil cuando se trata de entender el humor o detectar emociones como el sarcasmo en una oración.

En este punto, puede que estés recordando las clases escolares donde el maestro cubría las partes de la gramática en una oración. En algunos países, se enseña gramática y lingüística como una asignatura dedicada, pero en muchos, estos temas se incluyen como parte del aprendizaje de un idioma: ya sea tu primer idioma en la escuela primaria (aprender a leer y escribir) y tal vez un segundo idioma en la escuela secundaria. ¡No te preocupes si no eres un experto en diferenciar sustantivos de verbos o adverbios de adjetivos!

Si te cuesta la diferencia entre el *presente simple* y el *presente progresivo*, no estás solo. Esto es un desafío para muchas personas, incluso para hablantes nativos de un idioma. La buena noticia es que las computadoras son realmente buenas para aplicar reglas formales, y aprenderás a escribir código que pueda *analizar* una oración tan bien como un humano. El mayor desafío que examinarás más adelante es entender el *significado* y el *sentimiento* de una oración.

## Requisitos previos

Para esta lección, el requisito principal es poder leer y entender el idioma de esta lección. No hay problemas matemáticos ni ecuaciones que resolver. Aunque el autor original escribió esta lección en inglés, también está traducida a otros idiomas, por lo que podrías estar leyendo una traducción. Hay ejemplos donde se usan varios idiomas diferentes (para comparar las diferentes reglas gramaticales de diferentes idiomas). Estos *no* están traducidos, pero el texto explicativo sí, por lo que el significado debería ser claro.

Para las tareas de codificación, usarás Python y los ejemplos están usando Python 3.8.

En esta sección, necesitarás y usarás:

- **Comprensión de Python 3**. Comprensión del lenguaje de programación en Python 3, esta lección usa entrada, bucles, lectura de archivos, matrices.
- **Visual Studio Code + extensión**. Usaremos Visual Studio Code y su extensión de Python. También puedes usar un IDE de Python de tu elección.
- **TextBlob**. [TextBlob](https://github.com/sloria/TextBlob) es una biblioteca simplificada de procesamiento de texto para Python. Sigue las instrucciones en el sitio de TextBlob para instalarlo en tu sistema (instala también los corpora, como se muestra a continuación):

   ```bash
   pip install -U textblob
   python -m textblob.download_corpora
   ```

> 💡 Consejo: Puedes ejecutar Python directamente en entornos de VS Code. Consulta los [documentos](https://code.visualstudio.com/docs/languages/python?WT.mc_id=academic-77952-leestott) para obtener más información.

## Hablando con máquinas

La historia de intentar que las computadoras entiendan el lenguaje humano se remonta a décadas, y uno de los primeros científicos en considerar el procesamiento de lenguaje natural fue *Alan Turing*.

### La 'prueba de Turing'

Cuando Turing estaba investigando la *inteligencia artificial* en la década de 1950, consideró si se podría dar una prueba conversacional a un humano y una computadora (a través de correspondencia escrita) donde el humano en la conversación no estuviera seguro si estaba conversando con otro humano o una computadora.

Si, después de una cierta duración de la conversación, el humano no podía determinar si las respuestas provenían de una computadora o no, ¿se podría decir que la computadora estaba *pensando*?

### La inspiración - 'el juego de imitación'

La idea para esto vino de un juego de fiesta llamado *El Juego de Imitación* donde un interrogador está solo en una habitación y tiene la tarea de determinar cuál de dos personas (en otra habitación) es hombre y mujer respectivamente. El interrogador puede enviar notas y debe tratar de pensar en preguntas cuyas respuestas escritas revelen el género de la persona misteriosa. Por supuesto, los jugadores en la otra habitación están tratando de engañar al interrogador respondiendo preguntas de manera que confundan o confundan al interrogador, mientras que también dan la apariencia de responder honestamente.

### Desarrollando Eliza

En la década de 1960, un científico del MIT llamado *Joseph Weizenbaum* desarrolló [*Eliza*](https://wikipedia.org/wiki/ELIZA), un 'terapeuta' de computadora que haría preguntas al humano y daría la apariencia de entender sus respuestas. Sin embargo, aunque Eliza podía analizar una oración e identificar ciertos constructos gramaticales y palabras clave para dar una respuesta razonable, no se podía decir que *entendiera* la oración. Si a Eliza se le presentaba una oración siguiendo el formato "**Yo estoy** <u>triste</u>", podría reorganizar y sustituir palabras en la oración para formar la respuesta "¿Cuánto tiempo has **estado** <u>triste</u>?".

Esto daba la impresión de que Eliza entendía la declaración y estaba haciendo una pregunta de seguimiento, mientras que en realidad, estaba cambiando el tiempo verbal y agregando algunas palabras. Si Eliza no podía identificar una palabra clave para la cual tenía una respuesta, en su lugar daría una respuesta aleatoria que podría ser aplicable a muchas declaraciones diferentes. Eliza podría ser fácilmente engañada, por ejemplo, si un usuario escribía "**Tú eres** una <u>bicicleta</u>", podría responder "¿Cuánto tiempo he **sido** una <u>bicicleta</u>?", en lugar de una respuesta más razonada.

[![Chateando con Eliza](https://img.youtube.com/vi/RMK9AphfLco/0.jpg)](https://youtu.be/RMK9AphfLco "Chateando con Eliza")

> 🎥 Haz clic en la imagen de arriba para ver un video sobre el programa original ELIZA

> Nota: Puedes leer la descripción original de [Eliza](https://cacm.acm.org/magazines/1966/1/13317-elizaa-computer-program-for-the-study-of-natural-language-communication-between-man-and-machine/abstract) publicada en 1966 si tienes una cuenta de ACM. Alternativamente, lee sobre Eliza en [wikipedia](https://wikipedia.org/wiki/ELIZA)

## Ejercicio - codificando un bot conversacional básico

Un bot conversacional, como Eliza, es un programa que obtiene la entrada del usuario y parece entender y responder inteligentemente. A diferencia de Eliza, nuestro bot no tendrá varias reglas que le den la apariencia de tener una conversación inteligente. En su lugar, nuestro bot tendrá una única habilidad: mantener la conversación con respuestas aleatorias que podrían funcionar en casi cualquier conversación trivial.

### El plan

Tus pasos al construir un bot conversacional:

1. Imprime instrucciones que aconsejen al usuario cómo interactuar con el bot
2. Inicia un bucle
   1. Acepta la entrada del usuario
   2. Si el usuario ha pedido salir, entonces salir
   3. Procesa la entrada del usuario y determina la respuesta (en este caso, la respuesta es una elección aleatoria de una lista de posibles respuestas genéricas)
   4. Imprime la respuesta
3. Vuelve al paso 2

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

    Aquí tienes una salida de muestra para guiarte (la entrada del usuario está en las líneas que comienzan con `>`):

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

    Una posible solución a la tarea está [aquí](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/1-Introduction-to-NLP/solution/bot.py)

    ✅ Detente y considera

    1. ¿Crees que las respuestas aleatorias 'engañarían' a alguien para que piense que el bot realmente los entendió?
    2. ¿Qué características necesitaría el bot para ser más efectivo?
    3. Si un bot realmente pudiera 'entender' el significado de una oración, ¿necesitaría 'recordar' el significado de oraciones anteriores en una conversación también?

---

## 🚀Desafío

Elige uno de los elementos de "detente y considera" anteriores e intenta implementarlo en código o escribe una solución en papel usando pseudocódigo.

En la próxima lección, aprenderás sobre varias otras aproximaciones para analizar el lenguaje natural y el aprendizaje automático.

## [Cuestionario posterior a la lección](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/32/)

## Revisión y autoestudio

Echa un vistazo a las referencias a continuación como oportunidades de lectura adicional.

### Referencias

1. Schubert, Lenhart, "Lingüística Computacional", *The Stanford Encyclopedia of Philosophy* (Edición de Primavera 2020), Edward N. Zalta (ed.), URL = <https://plato.stanford.edu/archives/spr2020/entries/computational-linguistics/>.
2. Princeton University "Acerca de WordNet." [WordNet](https://wordnet.princeton.edu/). Princeton University. 2010. 

## Asignación 

[Busca un bot](assignment.md)

        **Descargo de responsabilidad**:
        Este documento ha sido traducido utilizando servicios de traducción automática basados en inteligencia artificial. Si bien nos esforzamos por lograr precisión, tenga en cuenta que las traducciones automáticas pueden contener errores o inexactitudes. El documento original en su idioma nativo debe considerarse la fuente autorizada. Para información crítica, se recomienda una traducción humana profesional. No nos hacemos responsables de ningún malentendido o interpretación errónea que surja del uso de esta traducción.