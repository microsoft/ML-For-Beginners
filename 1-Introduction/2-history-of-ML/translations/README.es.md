# Historia del machine learning

![Resumen de la historia del machine learning en un boceto](../../sketchnotes/ml-history.png)
> Boceto por [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Cuestionario previo a la conferencia](https://white-water-09ec41f0f.azurestaticapps.net/quiz/3/)

En esta lección, analizaremos los principales hitos en la historia del machine learning y la inteligencia artificial.

La historia de la inteligencia artificial (AI) como campo está entrelazada con la historia del machine learning, ya que los algoritmos y avances computacionales que sustentan el ML ayudaron al desarrollo de la inteligencia artificial. Es útil recordar que, si bien estos campos comenzaron a cristalizar en la década de 1950 como áreas distintas de investigación, importantes [descubrimientos algorítmicos, estadísticos, matemáticos, computacionales y técnicos](https://wikipedia.org/wiki/Timeline_of_machine_learning) fueron predecesores y contemporáneos a esta era. De hecho, las personas han estado pensando en estas preguntas durante [cientos de años](https://wikipedia.org/wiki/History_of_artificial_intelligence): este artículo analiza los fundamentos intelectuales históricos de la idea de una 'máquina pensante.'

## Descubrimientos notables

- 1763, 1812 [Teorema de Bayes](https://es.wikipedia.org/wiki/Teorema_de_Bayes) y sus predecesores. Este teorema y sus aplicaciones son la base de la inferencia, describiendo la probabilidad de que ocurra un evento basado en el conocimiento previo.
- 1805 [Teoría de mínimos cuadrados](https://es.wikipedia.org/wiki/M%C3%ADnimos_cuadrados) por el matemático francés Adrien-Marie Legendre. Esta teoría, sobre la que aprenderemos en nuestra unidad de Regresión, ayuda al ajustar los modelos a los datos.    
- 1913 [Cadenas de Markov](https://es.wikipedia.org/wiki/Cadena_de_M%C3%A1rkov) el nombre del matemático ruso Andrey Markov es utilizado para describir una secuencia de posibles eventos basados en su estado anterior.
- 1957 [Perceptron](https://wikipedia.org/wiki/Perceptron) es un tipo de clasificador lineal inventado por el psicólogo Frank Rosenblatt que subyace a los avances en el deep learning.
- 1967 [Nearest Neighbor (Vecino más cercano)](https://es.wikipedia.org/wiki/K_vecinos_m%C3%A1s_pr%C3%B3ximos) es un algoritmo diseñado originalmente para trazar rutas. En un contexto de ML, se utiliza para detectar patrones.
- 1970 [Retropropagación](https://es.wikipedia.org/wiki/Propagaci%C3%B3n_hacia_atr%C3%A1s): es usada para entrenar [redes neuronales prealimentadas](https://es.wikipedia.org/wiki/Red_neuronal_prealimentada).
- 1982 [Redes neuronales recurrentes](https://es.wikipedia.org/wiki/Red_neuronal_recurrente) son redes neuronales artificiales derivadas de redes neuronales prealimentadas que crean grafos temporales.

✅ Investigue un poco. ¿Qué otras fechas se destacan como fundamentales en la historia del machine learning (ML) y la inteligencia artificial (AI)?
## 1950: Máquinas que piensan

Alan Turing, una persona verdaderamente notable que fue votada [por el público en 2019](https://wikipedia.org/wiki/Icons:_The_Greatest_Person_of_the_20th_Century) como el científico más grande del siglo XX, a quien se le atribuye haber ayudado a sentar las bases del concepto de una 'máquina que puede pensar.' Lidió con los detractores y con su propia necesidad de evidencia empírica de este concepto en parte mediante la creación de la [prueba de Turing](https://www.bbc.com/news/technology-18475646), que explorarás en nuestras lecciones de procesamiento de lenguaje natural (NLP, por sus siglas en inglés).

## 1956: Dartmouth Summer Research Project

"The Dartmouth Summer Research Project sobre inteligencia artificial fue un evento fundamental para la inteligencia artificial como campo" y fue aquí donde se acuñó el término 'inteligencia artificial' ([fuente](https://250.dartmouth.edu/highlights/artificial-intelligence-ai-coined-dartmouth)).


> Todos los aspectos del aprendizaje y cualquier otra característica de la inteligencia pueden, en principio, describirse con tanta precisión que se puede hacer una máquina para simularlos.

El investigador principal, el profesor de matemáticas John McCarthy, esperaba "proceder sobre las bases de la conjetura de que cada aspecto del aprendizaje o cualquier otra característica de la inteligencia pueden, en principio, describirse con tanta precisión que se puede hacer una máquina para simularlos." Los participantes, incluyeron otro gran experto en el campo, Marvin Minsky.

El taller tiene el mérito de haber iniciado y alentado varias discusiones que incluyen "el surgimiento de métodos simbólicos, sistemas en dominios limitados (primeros sistemas expertos), y sistemas deductivos contra sistemas inductivos."  ([fuente](https://es.wikipedia.org/wiki/Conferencia_de_Dartmouth)).

## 1956 - 1974: "Los años dorados"

Desde la década de 1950, hasta mediados de la de 1970, el optimismo se elevó con la esperanza de que la AI pudiera resolver muchos problemas. En 1967, Marvin Minsky declaró con seguridad que "dentro de una generación ... el problema de crear 'inteligencia artificial' estará resuelto en gran medida."  (Minsky, Marvin (1967), Computation: Finite and Infinite Machines, Englewood Cliffs, N.J.: Prentice-Hall)

La investigación del procesamiento del lenguaje natural floreció, la búsqueda se refinó y se hizo más poderosa, y el concepto de 'micro-mundos' fue creado, donde se completaban tareas simples utilizando instrucciones en lenguaje sencillo.

La investigación estuvo bien financiada por agencias gubernamentales, se realizaron avances en computación y algoritmos, y se construyeron prototipos de máquinas inteligentes. Algunas de esta máquinas incluyen:

* [Shakey la robot](https://wikipedia.org/wiki/Shakey_the_robot), que podría maniobrar y decidir cómo realizar las tareas de forma 'inteligente'.

    ![Shakey, un robot inteligente](images/shakey.jpg)
    > Shakey en 1972

* Eliza, unas de las primeras 'chatterbot', podía conversar con las personas y actuar como un 'terapeuta' primitivo. Aprenderá más sobre Eliza en las lecciones de NLP.

    ![Eliza, un bot](images/eliza.png)
    > Una versión de Eliza, un chatbot

* "Blocks world" era un ejemplo de micro-world donde los bloques se podían apilar y ordenar, y se podían probar experimentos en máquinas de enseñanza para tomar decisiones. Los avances creados con librerías como [SHRDLU](https://wikipedia.org/wiki/SHRDLU) ayudaron a inpulsar el procesamiento del lenguaje natural.

    [![blocks world con SHRDLU](https://img.youtube.com/vi/QAJz4YKUwqw/0.jpg)](https://www.youtube.com/watch?v=QAJz4YKUwqw "blocks world con SHRDLU")
    
    > 🎥 Haga clic en la imagen de arriba para ver un video: Blocks world con SHRDLU

## 1974 - 1980: "Invierno de la AI"

A mediados de la década de 1970, se hizo evidente que la complejidad de la fabricación de 'máquinas inteligentes' se había subestimado y que su promesa, dado la potencia computacional disponible, había sido exagerada. La financiación se agotó y la confianza en el campo se ralentizó. Algunos problemas que impactaron la confianza incluyeron:

- **Limitaciones**. La potencia computacional era demasiado limitada.
- **Explosión combinatoria**. La cantidad de parámetros necesitados para entrenar creció exponencialmente a medida que se pedía más a las computadoras sin una evolución paralela de la potencia y la capacidad de cómputo.
- **Escasez de datos**. Hubo una escasez de datos que obstaculizó el proceso de pruebas, desarrollo y refinamiento de algoritmos.
- **¿Estamos haciendo las preguntas correctas?**. Las mismas preguntas que se estaban formulando comenzaron a cuestionarse. Los investigadores comenzaron a criticar sus métodos:
  - Las pruebas de Turing se cuestionaron por medio, entre otras ideas, de la 'teoría de la habitación china' que postulaba que "programar una computadora digital puede hacerse que parezca que entiende el lenguaje, pero no puede producir una comprensión real" ([fuente](https://plato.stanford.edu/entries/chinese-room/))
  - Se cuestionó la ética de introducir inteligencias artificiales como la "terapeuta" Eliza en la sociedad.

Al mismo tiempo, comenzaron a formarse varias escuelas de pensamiento de AI. Se estableció una dicotomía entre las prácticas ["scruffy" vs. "neat AI"](https://wikipedia.org/wiki/Neats_and_scruffies). _Scruffy_ labs modificó los programas durante horas hasta que obtuvieron los objetivos deseados. _Neat_ labs "centrados en la lógica y la resolución de problemas formales". ELIZA y SHRDLU eran sistemas _scruffy_ muy conocidos. En la década de 1980, cuando surgió la demanda para hacer que los sistemas de aprendizaje fueran reproducibles, el enfoque _neat_ gradualmente tomó la vanguardia a medida que sus resultados eran más explicables.

## Systemas expertos de la década de 1980

A medida que el campo creció, su beneficio para las empresas se hizo más claro, y en la década de 1980 también lo hizo la proliferación de 'sistemas expertos'. "Los sistemas expertos estuvieron entre las primeras formas verdaderamente exitosas de software de inteligencia artificial (IA)." ([fuente](https://wikipedia.org/wiki/Expert_system)).

Este tipo de sistemas es en realidad _híbrido_, que consta parcialmente de un motor de reglas que define los requisitos comerciales, y un motor de inferencia que aprovechó el sistema de reglas para deducir nuevos hechos.

En esta era también se prestó mayor atención a las redes neuronales.

## 1987 - 1993: AI 'Chill'

La proliferación de hardware de sistemas expertos especializados tuvo el desafortunado efecto de volverse demasiado especializado. El auge de las computadoras personales también compitió con estos grandes sistemas centralizados especializados. La democratización de la informática había comenzado, y finalmente, allanó el camino para la explosión moderna del big data.

## 1993 - 2011

Esta época vió una nueva era para el ML y la IA para poder resolver problemas que anteriormente provenían de la falta de datos y de poder de cómputo. La cantidad de datos comenzó a aumentar rápidamente y a estar más disponible, para bien o para mal, especialmente con la llegada del smartphone alrededor del 2007. El poder computacional se expandió exponencialmente y los algoritmos evolucionaron al mismo tiempo. El campo comenzó a ganar madurez a medida que los días libres del pasado comenzaron a cristalizar en una verdadera disciplina.

## Ahora

Hoy en día, machine learning y la inteligencia artificial tocan casi todos los aspectos de nuestras vidas. Esta era requiere de una comprensión cuidadosa de los riesgos y los efectos potenciales de estos algoritmos en las vidas humanas. Como ha dicho Brad Smith de Microsoft, "La tecnología de la información plantea problemas que van al corazón de las protecciones fundamentales de los derechos humanos, como la privacidad y la libertad de expresión. Esos problemas aumentan las responsabilidades de las empresas de tecnología que crean estos productos. En nuestra opinión, también exige regulación gubernamental reflexiva y el desarrollo de normas sobre usos aceptables" ([fuente](https://www.technologyreview.com/2019/12/18/102365/the-future-of-ais-impact-on-society/)).

Queda por ver qué depara el futuro, pero es importante entender estos sistemas informáticos y el software y los algoritmos que ejecutan. Esperamos que este plan de estudios le ayude a comprender mejor para que pueda decidir por si mismo.

[![La historia del deep learning](https://img.youtube.com/vi/mTtDfKgLm54/0.jpg)](https://www.youtube.com/watch?v=mTtDfKgLm54 "The history of deep learning")
> 🎥 Haga clic en la imagen de arriba para ver un video: Yann LeCun analiza la historia del deep learning en esta conferencia

---
## 🚀Desafío

Sumérjase dentro de unos de estos momentos históricos y aprenda más sobre las personas detrás de ellos. Hay personajes fascinantes y nunca ocurrió ningún descubrimiento científico en un vacío cultural. ¿Qué descubres?

## [Cuestionario posterior a la lección](https://white-water-09ec41f0f.azurestaticapps.net/quiz/4/)

## Revisión y autoestudio

Aquí hay elementos para ver y escuchar:

[Este podcast donde Amy Boyd habla sobre la evolución de la IA](http://runasradio.com/Shows/Show/739)

[![La historia de la IA por Amy Boyd](https://img.youtube.com/vi/EJt3_bFYKss/0.jpg)](https://www.youtube.com/watch?v=EJt3_bFYKss "La historia de la IA por Amy Boyd")

## Tarea

[Crea un timeline](assignment.md)
