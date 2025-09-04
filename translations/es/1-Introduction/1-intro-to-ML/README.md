<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "73e9a7245aa57f00cd413ffd22c0ccb6",
  "translation_date": "2025-09-03T23:37:03+00:00",
  "source_file": "1-Introduction/1-intro-to-ML/README.md",
  "language_code": "es"
}
-->
# Introducción al aprendizaje automático

## [Cuestionario previo a la clase](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/1/)

---

[![ML para principiantes - Introducción al aprendizaje automático para principiantes](https://img.youtube.com/vi/6mSx_KJxcHI/0.jpg)](https://youtu.be/6mSx_KJxcHI "ML para principiantes - Introducción al aprendizaje automático para principiantes")

> 🎥 Haz clic en la imagen de arriba para ver un video corto sobre esta lección.

¡Bienvenido a este curso sobre aprendizaje automático clásico para principiantes! Ya sea que seas completamente nuevo en este tema o un practicante experimentado de ML que busca repasar un área, ¡nos alegra que te unas a nosotros! Queremos crear un punto de partida amigable para tu estudio de ML y estaríamos encantados de evaluar, responder e incorporar tus [comentarios](https://github.com/microsoft/ML-For-Beginners/discussions).

[![Introducción al aprendizaje automático](https://img.youtube.com/vi/h0e2HAPTGF4/0.jpg)](https://youtu.be/h0e2HAPTGF4 "Introducción al aprendizaje automático")

> 🎥 Haz clic en la imagen de arriba para ver un video: John Guttag del MIT introduce el aprendizaje automático.

---
## Comenzando con el aprendizaje automático

Antes de comenzar con este plan de estudios, necesitas tener tu computadora configurada y lista para ejecutar notebooks localmente.

- **Configura tu máquina con estos videos**. Usa los siguientes enlaces para aprender [cómo instalar Python](https://youtu.be/CXZYvNRIAKM) en tu sistema y [configurar un editor de texto](https://youtu.be/EU8eayHWoZg) para desarrollo.
- **Aprende Python**. También se recomienda tener un entendimiento básico de [Python](https://docs.microsoft.com/learn/paths/python-language/?WT.mc_id=academic-77952-leestott), un lenguaje de programación útil para científicos de datos que utilizamos en este curso.
- **Aprende Node.js y JavaScript**. También usamos JavaScript algunas veces en este curso al construir aplicaciones web, por lo que necesitarás tener [node](https://nodejs.org) y [npm](https://www.npmjs.com/) instalados, así como [Visual Studio Code](https://code.visualstudio.com/) disponible para desarrollo tanto en Python como en JavaScript.
- **Crea una cuenta de GitHub**. Ya que nos encontraste aquí en [GitHub](https://github.com), es posible que ya tengas una cuenta, pero si no, crea una y luego haz un fork de este plan de estudios para usarlo por tu cuenta. (También puedes darnos una estrella 😊).
- **Explora Scikit-learn**. Familiarízate con [Scikit-learn](https://scikit-learn.org/stable/user_guide.html), un conjunto de bibliotecas de ML que referenciamos en estas lecciones.

---
## ¿Qué es el aprendizaje automático?

El término 'aprendizaje automático' es uno de los más populares y frecuentemente utilizados hoy en día. Existe una posibilidad no trivial de que hayas escuchado este término al menos una vez si tienes algún tipo de familiaridad con la tecnología, sin importar el área en la que trabajes. Sin embargo, la mecánica del aprendizaje automático es un misterio para la mayoría de las personas. Para un principiante en aprendizaje automático, el tema puede parecer abrumador a veces. Por lo tanto, es importante entender qué es realmente el aprendizaje automático y aprender sobre él paso a paso, a través de ejemplos prácticos.

---
## La curva de expectativas

![curva de expectativas de ML](../../../../translated_images/hype.07183d711a17aafe70915909a0e45aa286ede136ee9424d418026ab00fec344c.es.png)

> Google Trends muestra la reciente 'curva de expectativas' del término 'aprendizaje automático'.

---
## Un universo misterioso

Vivimos en un universo lleno de misterios fascinantes. Grandes científicos como Stephen Hawking, Albert Einstein y muchos más han dedicado sus vidas a buscar información significativa que revele los misterios del mundo que nos rodea. Esta es la condición humana de aprender: un niño humano aprende cosas nuevas y descubre la estructura de su mundo año tras año mientras crece hasta la adultez.

---
## El cerebro de un niño

El cerebro y los sentidos de un niño perciben los hechos de su entorno y gradualmente aprenden los patrones ocultos de la vida que ayudan al niño a crear reglas lógicas para identificar patrones aprendidos. El proceso de aprendizaje del cerebro humano hace que los humanos sean la criatura más sofisticada de este mundo. Aprender continuamente descubriendo patrones ocultos y luego innovando sobre esos patrones nos permite mejorar continuamente a lo largo de nuestra vida. Esta capacidad de aprendizaje y evolución está relacionada con un concepto llamado [plasticidad cerebral](https://www.simplypsychology.org/brain-plasticity.html). Superficialmente, podemos establecer algunas similitudes motivacionales entre el proceso de aprendizaje del cerebro humano y los conceptos del aprendizaje automático.

---
## El cerebro humano

El [cerebro humano](https://www.livescience.com/29365-human-brain.html) percibe cosas del mundo real, procesa la información percibida, toma decisiones racionales y realiza ciertas acciones según las circunstancias. Esto es lo que llamamos comportarse inteligentemente. Cuando programamos una réplica del proceso de comportamiento inteligente en una máquina, se llama inteligencia artificial (IA).

---
## Algunos términos

Aunque los términos pueden confundirse, el aprendizaje automático (ML) es un subconjunto importante de la inteligencia artificial. **ML se ocupa de usar algoritmos especializados para descubrir información significativa y encontrar patrones ocultos a partir de datos percibidos para corroborar el proceso de toma de decisiones racionales**.

---
## IA, ML, Aprendizaje profundo

![IA, ML, aprendizaje profundo, ciencia de datos](../../../../translated_images/ai-ml-ds.537ea441b124ebf69c144a52c0eb13a7af63c4355c2f92f440979380a2fb08b8.es.png)

> Un diagrama que muestra las relaciones entre IA, ML, aprendizaje profundo y ciencia de datos. Infografía por [Jen Looper](https://twitter.com/jenlooper) inspirada en [este gráfico](https://softwareengineering.stackexchange.com/questions/366996/distinction-between-ai-ml-neural-networks-deep-learning-and-data-mining).

---
## Conceptos a cubrir

En este plan de estudios, vamos a cubrir solo los conceptos básicos del aprendizaje automático que un principiante debe conocer. Cubrimos lo que llamamos 'aprendizaje automático clásico', principalmente utilizando Scikit-learn, una excelente biblioteca que muchos estudiantes usan para aprender los fundamentos. Para entender conceptos más amplios de inteligencia artificial o aprendizaje profundo, es indispensable un conocimiento fundamental sólido del aprendizaje automático, y queremos ofrecerlo aquí.

---
## En este curso aprenderás:

- conceptos básicos del aprendizaje automático
- la historia del ML
- ML y equidad
- técnicas de regresión en ML
- técnicas de clasificación en ML
- técnicas de agrupamiento en ML
- técnicas de procesamiento de lenguaje natural en ML
- técnicas de predicción de series temporales en ML
- aprendizaje por refuerzo
- aplicaciones reales del ML

---
## Lo que no cubriremos

- aprendizaje profundo
- redes neuronales
- IA

Para ofrecer una mejor experiencia de aprendizaje, evitaremos las complejidades de las redes neuronales, el 'aprendizaje profundo' - construcción de modelos con muchas capas utilizando redes neuronales - y la IA, que discutiremos en un plan de estudios diferente. También ofreceremos un próximo plan de estudios de ciencia de datos para centrarnos en ese aspecto de este campo más amplio.

---
## ¿Por qué estudiar aprendizaje automático?

El aprendizaje automático, desde una perspectiva de sistemas, se define como la creación de sistemas automatizados que pueden aprender patrones ocultos a partir de datos para ayudar en la toma de decisiones inteligentes.

Esta motivación está vagamente inspirada en cómo el cerebro humano aprende ciertas cosas basándose en los datos que percibe del mundo exterior.

✅ Piensa por un momento por qué una empresa querría intentar usar estrategias de aprendizaje automático en lugar de crear un motor basado en reglas codificadas.

---
## Aplicaciones del aprendizaje automático

Las aplicaciones del aprendizaje automático están ahora casi en todas partes y son tan ubicuas como los datos que fluyen en nuestras sociedades, generados por nuestros teléfonos inteligentes, dispositivos conectados y otros sistemas. Considerando el inmenso potencial de los algoritmos de aprendizaje automático de última generación, los investigadores han estado explorando su capacidad para resolver problemas reales multidimensionales y multidisciplinarios con grandes resultados positivos.

---
## Ejemplos de ML aplicado

**Puedes usar el aprendizaje automático de muchas maneras**:

- Para predecir la probabilidad de una enfermedad a partir del historial médico o informes de un paciente.
- Para aprovechar los datos meteorológicos y predecir eventos climáticos.
- Para entender el sentimiento de un texto.
- Para detectar noticias falsas y detener la propagación de propaganda.

Finanzas, economía, ciencias de la tierra, exploración espacial, ingeniería biomédica, ciencias cognitivas e incluso campos en las humanidades han adaptado el aprendizaje automático para resolver los arduos problemas de procesamiento de datos en sus áreas.

---
## Conclusión

El aprendizaje automático automatiza el proceso de descubrimiento de patrones al encontrar información significativa a partir de datos reales o generados. Ha demostrado ser altamente valioso en aplicaciones empresariales, de salud y financieras, entre otras.

En un futuro cercano, entender los fundamentos del aprendizaje automático será imprescindible para personas de cualquier área debido a su adopción generalizada.

---
# 🚀 Desafío

Dibuja, en papel o usando una aplicación en línea como [Excalidraw](https://excalidraw.com/), tu comprensión de las diferencias entre IA, ML, aprendizaje profundo y ciencia de datos. Agrega algunas ideas sobre los problemas que cada una de estas técnicas es buena resolviendo.

# [Cuestionario posterior a la clase](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/2/)

---
# Revisión y autoestudio

Para aprender más sobre cómo trabajar con algoritmos de ML en la nube, sigue este [Camino de Aprendizaje](https://docs.microsoft.com/learn/paths/create-no-code-predictive-models-azure-machine-learning/?WT.mc_id=academic-77952-leestott).

Toma un [Camino de Aprendizaje](https://docs.microsoft.com/learn/modules/introduction-to-machine-learning/?WT.mc_id=academic-77952-leestott) sobre los fundamentos del ML.

---
# Tarea

[Comienza a trabajar](assignment.md)

---

**Descargo de responsabilidad**:  
Este documento ha sido traducido utilizando el servicio de traducción automática [Co-op Translator](https://github.com/Azure/co-op-translator). Si bien nos esforzamos por garantizar la precisión, tenga en cuenta que las traducciones automatizadas pueden contener errores o imprecisiones. El documento original en su idioma nativo debe considerarse como la fuente autorizada. Para información crítica, se recomienda una traducción profesional realizada por humanos. No nos hacemos responsables de malentendidos o interpretaciones erróneas que puedan surgir del uso de esta traducción.