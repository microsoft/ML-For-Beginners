# Historia del machine learning

![Resumen de la historoia del machine learning en un boceto](../../sketchnotes/ml-history.png)
> Boceto por [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Cuestionario previo a la conferencia](https://jolly-sea-0a877260f.azurestaticapps.net/quiz/3/)

En esta lección, analizaremos los principales hitos en la historia del machine learning y la inteligencia artificial.

La historia de la inteligencia artificial, AI, como campo está entrelazada con la historia del machine learning, ya que los algoritmos y avances computacionales que sustentan el ML se incorporaron al desarrollo de la inteligencia artificial. Es útil recordar que, si bien, estos campos como áreas distintas de investigación comenzaron a cristalizar en la década de 1950, importantes [desubrimientos algorítmicos, estadísticos, matemáticos, computacionales y técnicos](https://wikipedia.org/wiki/Timeline_of_machine_learning) predecieronn y superpusieron a esta era. De hecho, las personas han estado pensando en estas preguntas durante [cientos de años](https://wikipedia.org/wiki/History_of_artificial_intelligence): este artículo analiza los fundamentos intelectuales históricos de la idea de una 'máquina pensante.'

## Descubrimientos notables

- 1763, 1812 [Teorema de Bayes](https://wikipedia.org/wiki/Bayes%27_theorem) y sus predecesores. Este teorema y sus aplicaciones son la base de la inferencia, describiendo la probabilidad de que ocurra un evento basado en el concimiento previo.
- 1805 [Teoría de mínimos cuadrados](https://wikipedia.org/wiki/Least_squares) por el matemático francés Adrien-Marie Legendre. Esta teoría, que aprenderá en nuestra unidad de Regresión, ayuda en el data fitting.
- 1913 [Cadenas de Markov](https://wikipedia.org/wiki/Markov_chain) el nombre del matemático ruso Andrey Markov es utilizado para describir una secuencia de eventos basados en su estado anterior.
- 1957 [Perceptron](https://wikipedia.org/wiki/Perceptron) es un tipo de clasificador lineal inventado por el psicólogo Frank Rosenblatt que subyace a los avances en el deep learning.
- 1967 [Nearest Neighbor (Vecino más cercano)](https://wikipedia.org/wiki/Nearest_neighbor) es un algoritmo diseñado originalmente para trazar rutas. En un contexto de ML, se utiliza para detectar patrones.
- 1970 [Backpropagation](https://wikipedia.org/wiki/Backpropagation) es usado para entrenar [feedforward neural networks](https://wikipedia.org/wiki/Feedforward_neural_network).
- 1982 [Recurrent Neural Networks](https://wikipedia.org/wiki/Recurrent_neural_network) son redes neuronales artificiales derivadas de redes neuronales feedforward que crean grafos temporales.

✅ Investigue un poco. ¿Qué otras fechas se destacan como fundamentales en la historia del machine learning (ML) y la inteligencia artificial (AI)?
## 1950: Máquinas que piensan

Alan Turing, una persona verdaderamente notable que fue votada [por el público en 2019](https://wikipedia.org/wiki/Icons:_The_Greatest_Person_of_the_20th_Century) como el científico más grande del siglo XX, se le atribuye haber ayudado a sentar las bases del concepto de una 'máquina que puede pensar.' Lidió con los detractores y su propia necesidad de evidencia empírica de este concepto en parte mediante la creación de la [prueba de Turing](https://www.bbc.com/news/technology-18475646, que explorarás en nuestras lecciones de NLP.

## 1956: Dartmouth Summer Research Project

"The Dartmouth Summer Research Project sobre inteligencia artificial fuer un evento fundamental para la inteligencia artificial como campo," y fue aquí donde el se acuñó el término 'inteligencia artificial' ([fuente](https://250.dartmouth.edu/highlights/artificial-intelligence-ai-coined-dartmouth)).


> Todos los aspectos del aprendizaje y cualquier otra característica de la inteligencia pueden, en principio, describirse con tanta precisión que se puede hacer una máquina para simularlos.

El investigador principal, el profesor de matemáticas John McCarthy, esperaba "proceder sobre las bases de la conjetura que cada aspecto del aprendizaje o cualquier otra característica de la inteligencia pueden, en principio, describirse con tanta precición que se se puede hacer una máquina para simularlos." Los participantes, incluyeron otra luminaria en el campo, Marvin Minsky.

El taller tiene el mérito de haber iniciado y alentado varias discusiones que incluyen "el surgimiento de métodos simbólicos, systemas en dominios limitados (primeros sistemas expertos), y sistemas deductivos versus sistemas inductivos."  ([fuente](https://wikipedia.org/wiki/Dartmouth_workshop)).

## 1956 - 1974: "Los años dorados"

Desde la década de 1950, hasta mediados de la de 1970, el optimismo se elevó con la esperanza de que la AI pudiera resolver muchos problemas. En 1967, Marvin Minsky declaró con seguridad que "dentro de una generación ... el problema de crear 'inteligencia artificial' se resolverá sustancialemte."  (Minsky, Marvin (1967), Computation: Finite and Infinite Machines, Englewood Cliffs, N.J.: Prentice-Hall)

La investigación del procesamiento del lenguaje natural floreció, la búsqueda se refinó y se hizo más poderosa, y el concepto de 'micro-worlds' fue creado, donde se completaban tareas simples utilizando instrucciones en lenguaje sencillo.

La investigación estuvo bien financiado por agencias gubernamentales, se realizaron avances en computación y algoritmos, y se construyeron prototipos de máquinas inteligentes.Algunas de esta máquinas incluyen:

* [Shakey la robot](https://wikipedia.org/wiki/Shakey_the_robot), que podría maniobrar y decidir cómo realizar las tares de forma 'inteligente'.

    ![Shakey, un robot inteligente](images/shakey.jpg)
    > Shakey en 1972

* Eliza, unas de las primeras 'chatterbot', podía conversar con las personas y actuar como un 'terapeuta' primitivo. Aprenderá más sobre ELiza en las lecciones de NLP.

    ![Eliza, un bot](images/eliza.png)
    > Una versión de Eliza, un chatbot

* "Blocks world" era un ejemplo de micro-world donde los bloques se podían apilar y ordenar, y se podían probar experimentos en máquinas de enseñanza para tomar decisiones. Los avances creados con librerías como [SHRDLU](https://wikipedia.org/wiki/SHRDLU) ayudaron a inpulsar el procesamiento del lenguaje natural.

    [![blocks world con SHRDLU](https://img.youtube.com/vi/QAJz4YKUwqw/0.jpg)](https://www.youtube.com/watch?v=QAJz4YKUwqw "blocks world con SHRDLU")
    
    > 🎥 Haga click en la imagen de arriba para ver un video: Blocks world con SHRDLU

## 1974 - 1980: "Invierno de la AI"

A mediados de la década de 1970, se hizo evidente que la complejidad de la fabricación de 'máquinas inteligentes' se había subestimado y que su promesa, dado la potencia computacional disponible, había sido exagerada. La financiación se agotó y la confianza en el campo se ralentizó. Algunos problemas que impactaron la confianza incluyeron:

- **Limitaciones**. La potencia computacional era demasiado limitada.
- **Explosión combinatoria**. La cantidad de parámetros necesitados para entrenar creció exponencialmente a medida que se pedía más a las computadoras sin una evolución paralela de la potencia y la capacidad de cómputo.
- **Escasez de datos**. Hubo una escasez de datos que obstaculizó el proceso de pruebas, desarrollo y refinamiento de algoritmos.
- **¿Estamos haciendo las preguntas correctas?**. Las mismas preguntas que se estaban formulando comenzaron a cuestionarse. Los investigadores comenzaron a criticar sus aproches:
  - Las pruebas de Turing se cuestionaron por medio, entre otras ideas, de la 'teoría de la habitación china' que postulaba que "progrmar una computadora digital puede hacerse que parezca que entiende el lenguaje, pero no puede producir una comprensión real" ([fuente](https://plato.stanford.edu/entries/chinese-room/))
  - Se cuestionó la ética de introducir inteligencias artificiales como la "terapeuta" Eliza en la sociedad.

Al mismo tiempo, comenzaron a formarse varia escuelas de pensamiento de AI. Se estableció una dicotomía entre las prácticas ["scruffy" vs. "neat AI"](https://wikipedia.org/wiki/Neats_and_scruffies). 
At the same time, various AI schools of thought began to form. A dichotomy was established between ["scruffy" vs. "neat AI"](https://wikipedia.org/wiki/Neats_and_scruffies) practices. _Scruffy_ labs tweaked programs for hours until they had the desired results. _Neat_ labs "focused on logic and formal problem solving". ELIZA and SHRDLU were well-known _scruffy_ systems. In the 1980s, as demand emerged to make ML systems reproducible, the _neat_ approach gradually took the forefront as its results are more explainable.

## 1980s Expert systems

As the field grew, its benefit to business became clearer, and in the 1980s so did the proliferation of 'expert systems'. "Expert systems were among the first truly successful forms of artificial intelligence (AI) software." ([source](https://wikipedia.org/wiki/Expert_system)).

This type of system is actually _hybrid_, consisting partially of a rules engine defining business requirements, and an inference engine that leveraged the rules system to deduce new facts.

This era also saw increasing attention paid to neural networks.

## 1987 - 1993: AI 'Chill'

The proliferation of specialized expert systems hardware had the unfortunate effect of becoming too specialized. The rise of personal computers also competed with these large, specialized, centralized systems. The democratization of computing had begun, and it eventually paved the way for the modern explosion of big data.

## 1993 - 2011

This epoch saw a new era for ML and AI to be able to solve some of the problems that had been caused earlier by the lack of data and compute power. The amount of data began to rapidly increase and become more widely available, for better and for worse, especially with the advent of the smartphone around 2007. Compute power expanded exponentially, and algorithms evolved alongside. The field began to gain maturity as the freewheeling days of the past began to crystallize into a true discipline.

## Now

Today, machine learning and AI touch almost every part of our lives. This era calls for careful understanding of the risks and potentials effects of these algorithms on human lives. As Microsoft's Brad Smith has stated, "Information technology raises issues that go to the heart of fundamental human-rights protections like privacy and freedom of expression. These issues heighten responsibility for tech companies that create these products. In our view, they also call for thoughtful government regulation and for the development of norms around acceptable uses" ([source](https://www.technologyreview.com/2019/12/18/102365/the-future-of-ais-impact-on-society/)).

It remains to be seen what the future holds, but it is important to understand these computer systems and the software and algorithms that they run. We hope that this curriculum will help you to gain a better understanding so that you can decide for yourself.

[![The history of deep learning](https://img.youtube.com/vi/mTtDfKgLm54/0.jpg)](https://www.youtube.com/watch?v=mTtDfKgLm54 "The history of deep learning")
> 🎥 Click the image above for a video: Yann LeCun discusses the history of deep learning in this lecture

---
## 🚀Challenge

Dig into one of these historical moments and learn more about the people behind them. There are fascinating characters, and no scientific discovery was ever created in a cultural vacuum. What do you discover?

## [Post-lecture quiz](https://jolly-sea-0a877260f.azurestaticapps.net/quiz/4/)

## Review & Self Study

Here are items to watch and listen to:

[This podcast where Amy Boyd discusses the evolution of AI](http://runasradio.com/Shows/Show/739)

[![The history of AI by Amy Boyd](https://img.youtube.com/vi/EJt3_bFYKss/0.jpg)](https://www.youtube.com/watch?v=EJt3_bFYKss "The history of AI by Amy Boyd")

## Assignment

[Create a timeline](assignment.md)
