<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "8f819813b2ca08ec7b9f60a2c9336045",
  "translation_date": "2025-09-03T23:28:50+00:00",
  "source_file": "1-Introduction/3-fairness/README.md",
  "language_code": "es"
}
-->
# Construyendo soluciones de aprendizaje automático con IA responsable

![Resumen de IA responsable en aprendizaje automático en un sketchnote](../../../../translated_images/ml-fairness.ef296ebec6afc98a44566d7b6c1ed18dc2bf1115c13ec679bb626028e852fa1d.es.png)
> Sketchnote por [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Cuestionario previo a la lección](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/5/)

## Introducción

En este currículo, comenzarás a descubrir cómo el aprendizaje automático puede y está impactando nuestras vidas cotidianas. Incluso ahora, los sistemas y modelos están involucrados en tareas de toma de decisiones diarias, como diagnósticos médicos, aprobaciones de préstamos o detección de fraudes. Por lo tanto, es importante que estos modelos funcionen bien para proporcionar resultados confiables. Al igual que cualquier aplicación de software, los sistemas de IA pueden no cumplir con las expectativas o tener un resultado no deseado. Es por eso que es esencial poder entender y explicar el comportamiento de un modelo de IA.

Imagina lo que puede suceder cuando los datos que utilizas para construir estos modelos carecen de ciertos datos demográficos, como raza, género, visión política, religión, o representan desproporcionadamente dichos datos demográficos. ¿Qué pasa cuando la salida del modelo se interpreta para favorecer a algún grupo demográfico? ¿Cuál es la consecuencia para la aplicación? Además, ¿qué sucede cuando el modelo tiene un resultado adverso y es perjudicial para las personas? ¿Quién es responsable del comportamiento de los sistemas de IA? Estas son algunas preguntas que exploraremos en este currículo.

En esta lección, aprenderás a:

- Reconocer la importancia de la equidad en el aprendizaje automático y los daños relacionados con la falta de equidad.
- Familiarizarte con la práctica de explorar valores atípicos y escenarios inusuales para garantizar confiabilidad y seguridad.
- Comprender la necesidad de empoderar a todos mediante el diseño de sistemas inclusivos.
- Explorar lo vital que es proteger la privacidad y seguridad de los datos y las personas.
- Ver la importancia de tener un enfoque de caja de cristal para explicar el comportamiento de los modelos de IA.
- Ser consciente de cómo la responsabilidad es esencial para generar confianza en los sistemas de IA.

## Prerrequisito

Como prerrequisito, toma el "Camino de Aprendizaje de Principios de IA Responsable" y mira el video a continuación sobre el tema:

Aprende más sobre IA Responsable siguiendo este [Camino de Aprendizaje](https://docs.microsoft.com/learn/modules/responsible-ai-principles/?WT.mc_id=academic-77952-leestott)

[![Enfoque de Microsoft hacia la IA Responsable](https://img.youtube.com/vi/dnC8-uUZXSc/0.jpg)](https://youtu.be/dnC8-uUZXSc "Enfoque de Microsoft hacia la IA Responsable")

> 🎥 Haz clic en la imagen de arriba para ver un video: Enfoque de Microsoft hacia la IA Responsable

## Equidad

Los sistemas de IA deben tratar a todos de manera justa y evitar afectar a grupos similares de personas de diferentes maneras. Por ejemplo, cuando los sistemas de IA proporcionan orientación sobre tratamientos médicos, solicitudes de préstamos o empleo, deben hacer las mismas recomendaciones a todos con síntomas similares, circunstancias financieras o calificaciones profesionales. Cada uno de nosotros, como humanos, lleva consigo sesgos heredados que afectan nuestras decisiones y acciones. Estos sesgos pueden ser evidentes en los datos que usamos para entrenar sistemas de IA. A veces, esta manipulación ocurre de manera no intencional. A menudo es difícil saber conscientemente cuándo estás introduciendo sesgos en los datos.

**“Injusticia”** abarca impactos negativos, o “daños”, para un grupo de personas, como aquellos definidos en términos de raza, género, edad o estado de discapacidad. Los principales daños relacionados con la equidad pueden clasificarse como:

- **Asignación**, si un género o etnia, por ejemplo, es favorecido sobre otro.
- **Calidad del servicio**. Si entrenas los datos para un escenario específico pero la realidad es mucho más compleja, esto lleva a un servicio de bajo rendimiento. Por ejemplo, un dispensador de jabón que no parece ser capaz de detectar personas con piel oscura. [Referencia](https://gizmodo.com/why-cant-this-soap-dispenser-identify-dark-skin-1797931773)
- **Denigración**. Criticar y etiquetar injustamente algo o alguien. Por ejemplo, una tecnología de etiquetado de imágenes etiquetó erróneamente imágenes de personas de piel oscura como gorilas.
- **Sobre- o sub-representación**. La idea de que un cierto grupo no se ve en una determinada profesión, y cualquier servicio o función que siga promoviendo eso está contribuyendo al daño.
- **Estereotipos**. Asociar un grupo dado con atributos preasignados. Por ejemplo, un sistema de traducción entre inglés y turco puede tener inexactitudes debido a palabras con asociaciones estereotipadas de género.

![traducción al turco](../../../../translated_images/gender-bias-translate-en-tr.f185fd8822c2d4372912f2b690f6aaddd306ffbb49d795ad8d12a4bf141e7af0.es.png)
> traducción al turco

![traducción de regreso al inglés](../../../../translated_images/gender-bias-translate-tr-en.4eee7e3cecb8c70e13a8abbc379209bc8032714169e585bdeac75af09b1752aa.es.png)
> traducción de regreso al inglés

Al diseñar y probar sistemas de IA, debemos asegurarnos de que la IA sea justa y no esté programada para tomar decisiones sesgadas o discriminatorias, las cuales también están prohibidas para los seres humanos. Garantizar la equidad en la IA y el aprendizaje automático sigue siendo un desafío sociotécnico complejo.

### Confiabilidad y seguridad

Para generar confianza, los sistemas de IA deben ser confiables, seguros y consistentes bajo condiciones normales e inesperadas. Es importante saber cómo se comportarán los sistemas de IA en una variedad de situaciones, especialmente cuando son valores atípicos. Al construir soluciones de IA, se necesita un enfoque sustancial en cómo manejar una amplia variedad de circunstancias que las soluciones de IA podrían encontrar. Por ejemplo, un automóvil autónomo debe priorizar la seguridad de las personas. Como resultado, la IA que impulsa el automóvil necesita considerar todos los posibles escenarios que el automóvil podría enfrentar, como la noche, tormentas eléctricas o ventiscas, niños corriendo por la calle, mascotas, construcciones en la carretera, etc. Qué tan bien un sistema de IA puede manejar una amplia gama de condiciones de manera confiable y segura refleja el nivel de anticipación que el científico de datos o desarrollador de IA consideró durante el diseño o prueba del sistema.

> [🎥 Haz clic aquí para ver un video: ](https://www.microsoft.com/videoplayer/embed/RE4vvIl)

### Inclusión

Los sistemas de IA deben diseñarse para involucrar y empoderar a todos. Al diseñar e implementar sistemas de IA, los científicos de datos y desarrolladores de IA identifican y abordan posibles barreras en el sistema que podrían excluir a las personas de manera no intencional. Por ejemplo, hay 1,000 millones de personas con discapacidades en todo el mundo. Con el avance de la IA, pueden acceder a una amplia gama de información y oportunidades más fácilmente en su vida diaria. Al abordar las barreras, se crean oportunidades para innovar y desarrollar productos de IA con mejores experiencias que beneficien a todos.

> [🎥 Haz clic aquí para ver un video: inclusión en IA](https://www.microsoft.com/videoplayer/embed/RE4vl9v)

### Seguridad y privacidad

Los sistemas de IA deben ser seguros y respetar la privacidad de las personas. Las personas tienen menos confianza en sistemas que ponen en riesgo su privacidad, información o vidas. Al entrenar modelos de aprendizaje automático, dependemos de los datos para producir los mejores resultados. Al hacerlo, se debe considerar el origen de los datos y su integridad. Por ejemplo, ¿los datos fueron enviados por el usuario o estaban disponibles públicamente? Luego, al trabajar con los datos, es crucial desarrollar sistemas de IA que puedan proteger información confidencial y resistir ataques. A medida que la IA se vuelve más prevalente, proteger la privacidad y asegurar información personal y empresarial importante se está volviendo más crítico y complejo. Los problemas de privacidad y seguridad de datos requieren especial atención en la IA porque el acceso a los datos es esencial para que los sistemas de IA hagan predicciones y decisiones precisas e informadas sobre las personas.

> [🎥 Haz clic aquí para ver un video: seguridad en IA](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Como industria, hemos logrado avances significativos en privacidad y seguridad, impulsados significativamente por regulaciones como el GDPR (Reglamento General de Protección de Datos).
- Sin embargo, con los sistemas de IA debemos reconocer la tensión entre la necesidad de más datos personales para hacer los sistemas más efectivos y la privacidad.
- Al igual que con el nacimiento de las computadoras conectadas a internet, también estamos viendo un gran aumento en el número de problemas de seguridad relacionados con la IA.
- Al mismo tiempo, hemos visto que la IA se utiliza para mejorar la seguridad. Por ejemplo, la mayoría de los escáneres antivirus modernos están impulsados por heurísticas de IA.
- Necesitamos asegurarnos de que nuestros procesos de ciencia de datos se integren armoniosamente con las últimas prácticas de privacidad y seguridad.

### Transparencia

Los sistemas de IA deben ser comprensibles. Una parte crucial de la transparencia es explicar el comportamiento de los sistemas de IA y sus componentes. Mejorar la comprensión de los sistemas de IA requiere que las partes interesadas comprendan cómo y por qué funcionan para que puedan identificar posibles problemas de rendimiento, preocupaciones de seguridad y privacidad, sesgos, prácticas excluyentes o resultados no deseados. También creemos que quienes usan sistemas de IA deben ser honestos y transparentes sobre cuándo, por qué y cómo eligen implementarlos, así como las limitaciones de los sistemas que utilizan. Por ejemplo, si un banco utiliza un sistema de IA para apoyar sus decisiones de préstamos al consumidor, es importante examinar los resultados y entender qué datos influyen en las recomendaciones del sistema. Los gobiernos están comenzando a regular la IA en diversas industrias, por lo que los científicos de datos y las organizaciones deben explicar si un sistema de IA cumple con los requisitos regulatorios, especialmente cuando hay un resultado no deseado.

> [🎥 Haz clic aquí para ver un video: transparencia en IA](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Debido a que los sistemas de IA son tan complejos, es difícil entender cómo funcionan e interpretar los resultados.
- Esta falta de comprensión afecta la forma en que se gestionan, operacionalizan y documentan estos sistemas.
- Más importante aún, esta falta de comprensión afecta las decisiones tomadas utilizando los resultados que producen estos sistemas.

### Responsabilidad

Las personas que diseñan y despliegan sistemas de IA deben ser responsables de cómo operan sus sistemas. La necesidad de responsabilidad es particularmente crucial con tecnologías de uso sensible como el reconocimiento facial. Recientemente, ha habido una creciente demanda de tecnología de reconocimiento facial, especialmente por parte de organizaciones de aplicación de la ley que ven el potencial de la tecnología en usos como encontrar niños desaparecidos. Sin embargo, estas tecnologías podrían ser utilizadas por un gobierno para poner en riesgo las libertades fundamentales de sus ciudadanos al, por ejemplo, habilitar la vigilancia continua de individuos específicos. Por lo tanto, los científicos de datos y las organizaciones deben ser responsables de cómo su sistema de IA impacta a las personas o la sociedad.

[![Investigador líder en IA advierte sobre vigilancia masiva a través del reconocimiento facial](../../../../translated_images/accountability.41d8c0f4b85b6231301d97f17a450a805b7a07aaeb56b34015d71c757cad142e.es.png)](https://www.youtube.com/watch?v=Wldt8P5V6D0 "Enfoque de Microsoft hacia la IA Responsable")

> 🎥 Haz clic en la imagen de arriba para ver un video: Advertencias sobre vigilancia masiva a través del reconocimiento facial

En última instancia, una de las mayores preguntas para nuestra generación, como la primera generación que está llevando la IA a la sociedad, es cómo garantizar que las computadoras sigan siendo responsables ante las personas y cómo garantizar que las personas que diseñan computadoras sean responsables ante todos los demás.

## Evaluación de impacto

Antes de entrenar un modelo de aprendizaje automático, es importante realizar una evaluación de impacto para entender el propósito del sistema de IA; cuál es su uso previsto; dónde se desplegará; y quién interactuará con el sistema. Estas evaluaciones son útiles para los revisores o evaluadores del sistema para saber qué factores considerar al identificar riesgos potenciales y consecuencias esperadas.

Las siguientes son áreas de enfoque al realizar una evaluación de impacto:

* **Impacto adverso en individuos**. Ser consciente de cualquier restricción o requisito, uso no compatible o cualquier limitación conocida que obstaculice el rendimiento del sistema es vital para garantizar que el sistema no se utilice de manera que pueda causar daño a las personas.
* **Requisitos de datos**. Comprender cómo y dónde el sistema utilizará datos permite a los revisores explorar cualquier requisito de datos que debas tener en cuenta (por ejemplo, regulaciones de datos como GDPR o HIPAA). Además, examina si la fuente o cantidad de datos es suficiente para el entrenamiento.
* **Resumen del impacto**. Reúne una lista de posibles daños que podrían surgir del uso del sistema. A lo largo del ciclo de vida del aprendizaje automático, revisa si los problemas identificados se han mitigado o abordado.
* **Metas aplicables** para cada uno de los seis principios fundamentales. Evalúa si se cumplen las metas de cada principio y si hay alguna brecha.

## Depuración con IA responsable

Al igual que depurar una aplicación de software, depurar un sistema de IA es un proceso necesario para identificar y resolver problemas en el sistema. Hay muchos factores que pueden afectar que un modelo no funcione como se espera o de manera responsable. La mayoría de las métricas tradicionales de rendimiento de modelos son agregados cuantitativos del rendimiento de un modelo, lo cual no es suficiente para analizar cómo un modelo viola los principios de IA responsable. Además, un modelo de aprendizaje automático es una caja negra que dificulta entender qué impulsa su resultado o proporcionar explicaciones cuando comete un error. Más adelante en este curso, aprenderemos cómo usar el panel de IA Responsable para ayudar a depurar sistemas de IA. El panel proporciona una herramienta integral para que los científicos de datos y desarrolladores de IA realicen:

* **Análisis de errores**. Para identificar la distribución de errores del modelo que puede afectar la equidad o confiabilidad del sistema.
* **Visión general del modelo**. Para descubrir dónde hay disparidades en el rendimiento del modelo entre cohortes de datos.
* **Análisis de datos**. Para entender la distribución de datos e identificar cualquier posible sesgo en los datos que podría llevar a problemas de equidad, inclusión y confiabilidad.
* **Interpretabilidad del modelo**. Para entender qué afecta o influye en las predicciones del modelo. Esto ayuda a explicar el comportamiento del modelo, lo cual es importante para la transparencia y la responsabilidad.

## 🚀 Desafío

Para prevenir daños desde el principio, debemos:

- tener diversidad de antecedentes y perspectivas entre las personas que trabajan en los sistemas
- invertir en conjuntos de datos que reflejen la diversidad de nuestra sociedad
- desarrollar mejores métodos a lo largo del ciclo de vida del aprendizaje automático para detectar y corregir problemas de IA responsable cuando ocurran

Piensa en escenarios de la vida real donde la falta de confianza en un modelo sea evidente en su construcción y uso. ¿Qué más deberíamos considerar?

## [Cuestionario posterior a la lección](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/6/)
## Revisión y autoestudio
En esta lección, has aprendido algunos conceptos básicos sobre la equidad y la falta de equidad en el aprendizaje automático.  

Mira este taller para profundizar en los temas: 

- En busca de una IA responsable: Llevando los principios a la práctica por Besmira Nushi, Mehrnoosh Sameki y Amit Sharma

[![Responsible AI Toolbox: Un marco de código abierto para construir IA responsable](https://img.youtube.com/vi/tGgJCrA-MZU/0.jpg)](https://www.youtube.com/watch?v=tGgJCrA-MZU "RAI Toolbox: Un marco de código abierto para construir IA responsable")


> 🎥 Haz clic en la imagen de arriba para ver el video: RAI Toolbox: Un marco de código abierto para construir IA responsable por Besmira Nushi, Mehrnoosh Sameki y Amit Sharma

Además, lee: 

- Centro de recursos de IA responsable de Microsoft: [Responsible AI Resources – Microsoft AI](https://www.microsoft.com/ai/responsible-ai-resources?activetab=pivot1%3aprimaryr4) 

- Grupo de investigación FATE de Microsoft: [FATE: Fairness, Accountability, Transparency, and Ethics in AI - Microsoft Research](https://www.microsoft.com/research/theme/fate/) 

RAI Toolbox: 

- [Repositorio de GitHub de Responsible AI Toolbox](https://github.com/microsoft/responsible-ai-toolbox)

Lee sobre las herramientas de Azure Machine Learning para garantizar la equidad:

- [Azure Machine Learning](https://docs.microsoft.com/azure/machine-learning/concept-fairness-ml?WT.mc_id=academic-77952-leestott) 

## Tarea

[Explora RAI Toolbox](assignment.md)

---

**Descargo de responsabilidad**:  
Este documento ha sido traducido utilizando el servicio de traducción automática [Co-op Translator](https://github.com/Azure/co-op-translator). Si bien nos esforzamos por garantizar la precisión, tenga en cuenta que las traducciones automatizadas pueden contener errores o imprecisiones. El documento original en su idioma nativo debe considerarse como la fuente autorizada. Para información crítica, se recomienda una traducción profesional realizada por humanos. No nos hacemos responsables de malentendidos o interpretaciones erróneas que puedan surgir del uso de esta traducción.