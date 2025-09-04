<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "ba0f6e1019351351c8ee4c92867b6a0b",
  "translation_date": "2025-09-03T23:22:40+00:00",
  "source_file": "9-Real-World/2-Debugging-ML-Models/README.md",
  "language_code": "es"
}
-->
# Posdata: Depuración de modelos de aprendizaje automático utilizando componentes del panel de IA Responsable

## [Cuestionario previo a la clase](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/5/)

## Introducción

El aprendizaje automático impacta nuestra vida cotidiana. La IA está encontrando su lugar en algunos de los sistemas más importantes que nos afectan como individuos y como sociedad, desde la atención médica, las finanzas, la educación y el empleo. Por ejemplo, los sistemas y modelos están involucrados en tareas de toma de decisiones diarias, como diagnósticos médicos o detección de fraudes. En consecuencia, los avances en IA junto con su adopción acelerada están siendo recibidos con expectativas sociales en evolución y una creciente regulación en respuesta. Constantemente vemos áreas donde los sistemas de IA no cumplen con las expectativas; exponen nuevos desafíos; y los gobiernos están comenzando a regular las soluciones de IA. Por lo tanto, es importante que estos modelos sean analizados para proporcionar resultados justos, confiables, inclusivos, transparentes y responsables para todos.

En este currículo, exploraremos herramientas prácticas que pueden ser utilizadas para evaluar si un modelo tiene problemas relacionados con IA Responsable. Las técnicas tradicionales de depuración de aprendizaje automático tienden a basarse en cálculos cuantitativos como la precisión agregada o la pérdida promedio de error. Imagina lo que puede suceder cuando los datos que estás utilizando para construir estos modelos carecen de ciertos grupos demográficos, como raza, género, visión política, religión, o representan desproporcionadamente dichos grupos demográficos. ¿Qué sucede cuando la salida del modelo se interpreta como favorable para algunos grupos demográficos? Esto puede introducir una representación excesiva o insuficiente de estos grupos sensibles, resultando en problemas de equidad, inclusión o confiabilidad del modelo. Otro factor es que los modelos de aprendizaje automático son considerados cajas negras, lo que dificulta entender y explicar qué impulsa las predicciones de un modelo. Todos estos son desafíos que enfrentan los científicos de datos y desarrolladores de IA cuando no tienen herramientas adecuadas para depurar y evaluar la equidad o confiabilidad de un modelo.

En esta lección, aprenderás a depurar tus modelos utilizando:

- **Análisis de errores**: identificar dónde en la distribución de tus datos el modelo tiene altas tasas de error.
- **Visión general del modelo**: realizar análisis comparativos entre diferentes cohortes de datos para descubrir disparidades en las métricas de rendimiento de tu modelo.
- **Análisis de datos**: investigar dónde podría haber una representación excesiva o insuficiente de tus datos que pueda sesgar tu modelo para favorecer un grupo demográfico sobre otro.
- **Importancia de las características**: entender qué características están impulsando las predicciones de tu modelo a nivel global o local.

## Prerrequisito

Como prerrequisito, revisa [Herramientas de IA Responsable para desarrolladores](https://www.microsoft.com/ai/ai-lab-responsible-ai-dashboard)

> ![Gif sobre herramientas de IA Responsable](../../../../9-Real-World/2-Debugging-ML-Models/images/rai-overview.gif)

## Análisis de errores

Las métricas tradicionales de rendimiento de modelos utilizadas para medir la precisión son principalmente cálculos basados en predicciones correctas frente a incorrectas. Por ejemplo, determinar que un modelo es preciso el 89% del tiempo con una pérdida de error de 0.001 puede considerarse un buen rendimiento. Los errores a menudo no se distribuyen uniformemente en tu conjunto de datos subyacente. Puedes obtener una puntuación de precisión del modelo del 89%, pero descubrir que hay diferentes regiones de tus datos en las que el modelo falla el 42% del tiempo. Las consecuencias de estos patrones de falla con ciertos grupos de datos pueden llevar a problemas de equidad o confiabilidad. Es esencial entender las áreas donde el modelo está funcionando bien o no. Las regiones de datos donde hay un alto número de inexactitudes en tu modelo pueden resultar ser un grupo demográfico importante.

![Analizar y depurar errores del modelo](../../../../translated_images/ea-error-distribution.117452e1177c1dd84fab2369967a68bcde787c76c6ea7fdb92fcf15d1fce8206.es.png)

El componente de Análisis de Errores en el panel de IA Responsable ilustra cómo se distribuyen las fallas del modelo entre varios cohortes con una visualización en forma de árbol. Esto es útil para identificar características o áreas donde hay una alta tasa de error en tu conjunto de datos. Al ver de dónde provienen la mayoría de las inexactitudes del modelo, puedes comenzar a investigar la causa raíz. También puedes crear cohortes de datos para realizar análisis. Estos cohortes de datos ayudan en el proceso de depuración para determinar por qué el rendimiento del modelo es bueno en un cohorte, pero erróneo en otro.

![Análisis de errores](../../../../translated_images/ea-error-cohort.6886209ea5d438c4daa8bfbf5ce3a7042586364dd3eccda4a4e3d05623ac702a.es.png)

Los indicadores visuales en el mapa del árbol ayudan a localizar las áreas problemáticas más rápidamente. Por ejemplo, cuanto más oscuro sea el tono de rojo en un nodo del árbol, mayor será la tasa de error.

El mapa de calor es otra funcionalidad de visualización que los usuarios pueden utilizar para investigar la tasa de error utilizando una o dos características para encontrar un contribuyente a los errores del modelo en todo el conjunto de datos o cohortes.

![Mapa de calor de análisis de errores](../../../../translated_images/ea-heatmap.8d27185e28cee3830c85e1b2e9df9d2d5e5c8c940f41678efdb68753f2f7e56c.es.png)

Utiliza el análisis de errores cuando necesites:

* Obtener una comprensión profunda de cómo se distribuyen las fallas del modelo en un conjunto de datos y en varias dimensiones de entrada y características.
* Desglosar las métricas de rendimiento agregadas para descubrir automáticamente cohortes erróneos que informen tus pasos de mitigación específicos.

## Visión general del modelo

Evaluar el rendimiento de un modelo de aprendizaje automático requiere obtener una comprensión holística de su comportamiento. Esto se puede lograr revisando más de una métrica como tasa de error, precisión, recall, precisión o MAE (Error Absoluto Medio) para encontrar disparidades entre las métricas de rendimiento. Una métrica de rendimiento puede parecer excelente, pero las inexactitudes pueden exponerse en otra métrica. Además, comparar las métricas para encontrar disparidades en todo el conjunto de datos o cohortes ayuda a arrojar luz sobre dónde el modelo está funcionando bien o no. Esto es especialmente importante para observar el rendimiento del modelo entre características sensibles e insensibles (por ejemplo, raza, género o edad del paciente) para descubrir posibles problemas de equidad que pueda tener el modelo. Por ejemplo, descubrir que el modelo es más erróneo en un cohorte que tiene características sensibles puede revelar posibles problemas de equidad.

El componente de Visión General del Modelo en el panel de IA Responsable ayuda no solo a analizar las métricas de rendimiento de la representación de datos en un cohorte, sino que también brinda a los usuarios la capacidad de comparar el comportamiento del modelo entre diferentes cohortes.

![Cohortes de conjuntos de datos - visión general del modelo en el panel de IA Responsable](../../../../translated_images/model-overview-dataset-cohorts.dfa463fb527a35a0afc01b7b012fc87bf2cad756763f3652bbd810cac5d6cf33.es.png)

La funcionalidad de análisis basada en características del componente permite a los usuarios reducir subgrupos de datos dentro de una característica particular para identificar anomalías a nivel granular. Por ejemplo, el panel tiene inteligencia incorporada para generar automáticamente cohortes para una característica seleccionada por el usuario (por ejemplo, *"time_in_hospital < 3"* o *"time_in_hospital >= 7"*). Esto permite a un usuario aislar una característica particular de un grupo de datos más grande para ver si es un factor clave en los resultados erróneos del modelo.

![Cohortes de características - visión general del modelo en el panel de IA Responsable](../../../../translated_images/model-overview-feature-cohorts.c5104d575ffd0c80b7ad8ede7703fab6166bfc6f9125dd395dcc4ace2f522f70.es.png)

El componente de Visión General del Modelo admite dos clases de métricas de disparidad:

**Disparidad en el rendimiento del modelo**: Este conjunto de métricas calcula la disparidad (diferencia) en los valores de la métrica de rendimiento seleccionada entre subgrupos de datos. Aquí hay algunos ejemplos:

* Disparidad en la tasa de precisión
* Disparidad en la tasa de error
* Disparidad en la precisión
* Disparidad en el recall
* Disparidad en el error absoluto medio (MAE)

**Disparidad en la tasa de selección**: Esta métrica contiene la diferencia en la tasa de selección (predicción favorable) entre subgrupos. Un ejemplo de esto es la disparidad en las tasas de aprobación de préstamos. La tasa de selección significa la fracción de puntos de datos en cada clase clasificados como 1 (en clasificación binaria) o la distribución de valores de predicción (en regresión).

## Análisis de datos

> "Si torturas los datos lo suficiente, confesarán cualquier cosa" - Ronald Coase

Esta afirmación suena extrema, pero es cierto que los datos pueden ser manipulados para respaldar cualquier conclusión. Tal manipulación a veces puede ocurrir de manera no intencional. Como humanos, todos tenemos sesgos, y a menudo es difícil saber conscientemente cuándo estás introduciendo sesgos en los datos. Garantizar la equidad en la IA y el aprendizaje automático sigue siendo un desafío complejo.

Los datos son un gran punto ciego para las métricas tradicionales de rendimiento de modelos. Puedes tener puntuaciones de precisión altas, pero esto no siempre refleja el sesgo subyacente en los datos que podría estar en tu conjunto de datos. Por ejemplo, si un conjunto de datos de empleados tiene un 27% de mujeres en puestos ejecutivos en una empresa y un 73% de hombres en el mismo nivel, un modelo de IA de publicidad de empleo entrenado con estos datos puede dirigirse principalmente a una audiencia masculina para puestos de nivel superior. Tener este desequilibrio en los datos sesgó la predicción del modelo para favorecer un género. Esto revela un problema de equidad donde hay un sesgo de género en el modelo de IA.

El componente de Análisis de Datos en el panel de IA Responsable ayuda a identificar áreas donde hay una representación excesiva o insuficiente en el conjunto de datos. Ayuda a los usuarios a diagnosticar la causa raíz de errores y problemas de equidad introducidos por desequilibrios de datos o falta de representación de un grupo de datos particular. Esto brinda a los usuarios la capacidad de visualizar conjuntos de datos basados en resultados predichos y reales, grupos de errores y características específicas. A veces, descubrir un grupo de datos subrepresentado también puede revelar que el modelo no está aprendiendo bien, de ahí las altas inexactitudes. Tener un modelo con sesgo de datos no solo es un problema de equidad, sino que muestra que el modelo no es inclusivo ni confiable.

![Componente de análisis de datos en el panel de IA Responsable](../../../../translated_images/dataanalysis-cover.8d6d0683a70a5c1e274e5a94b27a71137e3d0a3b707761d7170eb340dd07f11d.es.png)

Utiliza el análisis de datos cuando necesites:

* Explorar las estadísticas de tu conjunto de datos seleccionando diferentes filtros para dividir tus datos en diferentes dimensiones (también conocidas como cohortes).
* Comprender la distribución de tu conjunto de datos entre diferentes cohortes y grupos de características.
* Determinar si tus hallazgos relacionados con equidad, análisis de errores y causalidad (derivados de otros componentes del panel) son resultado de la distribución de tu conjunto de datos.
* Decidir en qué áreas recolectar más datos para mitigar errores que provienen de problemas de representación, ruido en las etiquetas, ruido en las características, sesgo en las etiquetas y factores similares.

## Interpretabilidad del modelo

Los modelos de aprendizaje automático tienden a ser cajas negras. Entender qué características clave de los datos impulsan la predicción de un modelo puede ser un desafío. Es importante proporcionar transparencia sobre por qué un modelo hace una cierta predicción. Por ejemplo, si un sistema de IA predice que un paciente diabético está en riesgo de ser readmitido en un hospital en menos de 30 días, debería poder proporcionar datos de apoyo que llevaron a su predicción. Tener indicadores de datos de apoyo aporta transparencia para ayudar a los médicos o hospitales a tomar decisiones bien informadas. Además, poder explicar por qué un modelo hizo una predicción para un paciente individual permite responsabilidad con las regulaciones de salud. Cuando utilizas modelos de aprendizaje automático de maneras que afectan la vida de las personas, es crucial entender y explicar qué influye en el comportamiento de un modelo. La explicabilidad e interpretabilidad del modelo ayuda a responder preguntas en escenarios como:

* Depuración del modelo: ¿Por qué mi modelo cometió este error? ¿Cómo puedo mejorar mi modelo?
* Colaboración humano-IA: ¿Cómo puedo entender y confiar en las decisiones del modelo?
* Cumplimiento regulatorio: ¿Cumple mi modelo con los requisitos legales?

El componente de Importancia de las Características del panel de IA Responsable te ayuda a depurar y obtener una comprensión integral de cómo un modelo hace predicciones. También es una herramienta útil para profesionales de aprendizaje automático y tomadores de decisiones para explicar y mostrar evidencia de las características que influyen en el comportamiento de un modelo para el cumplimiento regulatorio. Los usuarios pueden explorar tanto explicaciones globales como locales para validar qué características impulsan la predicción de un modelo. Las explicaciones globales enumeran las principales características que afectaron la predicción general de un modelo. Las explicaciones locales muestran qué características llevaron a la predicción de un modelo para un caso individual. La capacidad de evaluar explicaciones locales también es útil para depurar o auditar un caso específico para comprender mejor y explicar por qué un modelo hizo una predicción precisa o inexacta.

![Componente de importancia de características del panel de IA Responsable](../../../../translated_images/9-feature-importance.cd3193b4bba3fd4bccd415f566c2437fb3298c4824a3dabbcab15270d783606e.es.png)

* Explicaciones globales: Por ejemplo, ¿qué características afectan el comportamiento general de un modelo de readmisión hospitalaria para pacientes diabéticos?
* Explicaciones locales: Por ejemplo, ¿por qué se predijo que un paciente diabético mayor de 60 años con hospitalizaciones previas sería readmitido o no readmitido dentro de los 30 días en un hospital?

En el proceso de depuración al examinar el rendimiento de un modelo entre diferentes cohortes, la Importancia de las Características muestra qué nivel de impacto tiene una característica entre los cohortes. Ayuda a revelar anomalías al comparar el nivel de influencia que tiene la característica en impulsar las predicciones erróneas de un modelo. El componente de Importancia de las Características puede mostrar qué valores en una característica influyeron positiva o negativamente en el resultado del modelo. Por ejemplo, si un modelo hizo una predicción inexacta, el componente te da la capacidad de profundizar y señalar qué características o valores de características impulsaron la predicción. Este nivel de detalle no solo ayuda en la depuración, sino que proporciona transparencia y responsabilidad en situaciones de auditoría. Finalmente, el componente puede ayudarte a identificar problemas de equidad. Para ilustrar, si una característica sensible como etnicidad o género tiene una alta influencia en impulsar la predicción de un modelo, esto podría ser un indicio de sesgo racial o de género en el modelo.

![Importancia de características](../../../../translated_images/9-features-influence.3ead3d3f68a84029f1e40d3eba82107445d3d3b6975d4682b23d8acc905da6d0.es.png)

Utiliza la interpretabilidad cuando necesites:

* Determinar cuán confiables son las predicciones de tu sistema de IA entendiendo qué características son más importantes para las predicciones.
* Abordar la depuración de tu modelo entendiéndolo primero e identificando si el modelo está utilizando características saludables o simplemente correlaciones falsas.
* Descubrir posibles fuentes de inequidad entendiendo si el modelo está basando sus predicciones en características sensibles o en características altamente correlacionadas con ellas.
* Generar confianza del usuario en las decisiones de tu modelo generando explicaciones locales para ilustrar sus resultados.
* Completar una auditoría regulatoria de un sistema de IA para validar modelos y monitorear el impacto de las decisiones del modelo en las personas.

## Conclusión

Todos los componentes del panel de IA Responsable son herramientas prácticas para ayudarte a construir modelos de aprendizaje automático que sean menos dañinos y más confiables para la sociedad. Mejoran la prevención de amenazas a los derechos humanos; la discriminación o exclusión de ciertos grupos de oportunidades de vida; y el riesgo de daño físico o psicológico. También ayudan a generar confianza en las decisiones de tu modelo al generar explicaciones locales para ilustrar sus resultados. Algunos de los posibles daños pueden clasificarse como:

- **Asignación**, si un género o etnicidad, por ejemplo, es favorecido sobre otro.
- **Calidad del servicio**. Si entrenas los datos para un escenario específico pero la realidad es mucho más compleja, esto lleva a un servicio de bajo rendimiento.
- **Estereotipos**. Asociar un grupo dado con atributos preasignados.
- **Denigración**. Criticar injustamente y etiquetar algo o alguien.
- **Representación excesiva o insuficiente**. La idea es que un cierto grupo no se vea representado en una determinada profesión, y cualquier servicio o función que siga promoviendo eso está contribuyendo al daño.

### Azure RAI dashboard

[Azure RAI dashboard](https://learn.microsoft.com/en-us/azure/machine-learning/concept-responsible-ai-dashboard?WT.mc_id=aiml-90525-ruyakubu) está construido sobre herramientas de código abierto desarrolladas por instituciones académicas y organizaciones líderes, incluyendo Microsoft. Estas herramientas son fundamentales para que los científicos de datos y desarrolladores de IA comprendan mejor el comportamiento de los modelos, descubran y mitiguen problemas indeseables en los modelos de IA.

- Aprende cómo usar los diferentes componentes consultando la [documentación del RAI dashboard.](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-responsible-ai-dashboard?WT.mc_id=aiml-90525-ruyakubu)

- Consulta algunos [notebooks de ejemplo del RAI dashboard](https://github.com/Azure/RAI-vNext-Preview/tree/main/examples/notebooks) para depurar escenarios de IA más responsables en Azure Machine Learning.

---
## 🚀 Desafío

Para evitar que se introduzcan sesgos estadísticos o de datos desde el principio, deberíamos:

- contar con una diversidad de antecedentes y perspectivas entre las personas que trabajan en los sistemas
- invertir en conjuntos de datos que reflejen la diversidad de nuestra sociedad
- desarrollar mejores métodos para detectar y corregir sesgos cuando ocurran

Piensa en escenarios de la vida real donde la injusticia sea evidente en la construcción y uso de modelos. ¿Qué más deberíamos considerar?

## [Cuestionario posterior a la clase](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/6/)
## Revisión y Autoestudio

En esta lección, has aprendido algunas herramientas prácticas para incorporar IA responsable en el aprendizaje automático.

Mira este taller para profundizar en los temas:

- Responsible AI Dashboard: Una solución integral para operacionalizar la IA responsable en la práctica por Besmira Nushi y Mehrnoosh Sameki

[![Responsible AI Dashboard: Una solución integral para operacionalizar la IA responsable en la práctica](https://img.youtube.com/vi/f1oaDNl3djg/0.jpg)](https://www.youtube.com/watch?v=f1oaDNl3djg "Responsible AI Dashboard: Una solución integral para operacionalizar la IA responsable en la práctica")


> 🎥 Haz clic en la imagen de arriba para ver el video: Responsible AI Dashboard: Una solución integral para operacionalizar la IA responsable en la práctica por Besmira Nushi y Mehrnoosh Sameki

Consulta los siguientes materiales para aprender más sobre IA responsable y cómo construir modelos más confiables:

- Herramientas del RAI dashboard de Microsoft para depurar modelos de aprendizaje automático: [Recursos de herramientas de IA responsable](https://aka.ms/rai-dashboard)

- Explora el kit de herramientas de IA responsable: [Github](https://github.com/microsoft/responsible-ai-toolbox)

- Centro de recursos de IA responsable de Microsoft: [Recursos de IA Responsable – Microsoft AI](https://www.microsoft.com/ai/responsible-ai-resources?activetab=pivot1%3aprimaryr4)

- Grupo de investigación FATE de Microsoft: [FATE: Equidad, Responsabilidad, Transparencia y Ética en IA - Microsoft Research](https://www.microsoft.com/research/theme/fate/)

## Tarea

[Explora el RAI dashboard](assignment.md)

---

**Descargo de responsabilidad**:  
Este documento ha sido traducido utilizando el servicio de traducción automática [Co-op Translator](https://github.com/Azure/co-op-translator). Si bien nos esforzamos por lograr precisión, tenga en cuenta que las traducciones automáticas pueden contener errores o imprecisiones. El documento original en su idioma nativo debe considerarse como la fuente autorizada. Para información crítica, se recomienda una traducción profesional realizada por humanos. No nos hacemos responsables de malentendidos o interpretaciones erróneas que puedan surgir del uso de esta traducción.