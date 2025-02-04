# Entrenar Mountain Car

[OpenAI Gym](http://gym.openai.com) ha sido diseñado de tal manera que todos los entornos proporcionan la misma API - es decir, los mismos métodos `reset`, `step` y `render`, y las mismas abstracciones de **espacio de acción** y **espacio de observación**. Por lo tanto, debería ser posible adaptar los mismos algoritmos de aprendizaje por refuerzo a diferentes entornos con mínimos cambios en el código.

## Un Entorno de Mountain Car

El [entorno de Mountain Car](https://gym.openai.com/envs/MountainCar-v0/) contiene un coche atrapado en un valle:
El objetivo es salir del valle y capturar la bandera, realizando en cada paso una de las siguientes acciones:

| Valor | Significado |
|---|---|
| 0 | Acelerar hacia la izquierda |
| 1 | No acelerar |
| 2 | Acelerar hacia la derecha |

El truco principal de este problema es, sin embargo, que el motor del coche no es lo suficientemente fuerte como para escalar la montaña en un solo intento. Por lo tanto, la única manera de tener éxito es conducir de un lado a otro para acumular impulso.

El espacio de observación consta de solo dos valores:

| Num | Observación  | Mín | Máx |
|-----|--------------|-----|-----|
|  0  | Posición del coche | -1.2| 0.6 |
|  1  | Velocidad del coche | -0.07 | 0.07 |

El sistema de recompensas para el coche de montaña es bastante complicado:

 * Se otorga una recompensa de 0 si el agente alcanza la bandera (posición = 0.5) en la cima de la montaña.
 * Se otorga una recompensa de -1 si la posición del agente es menor que 0.5.

El episodio termina si la posición del coche es mayor que 0.5, o si la duración del episodio es mayor que 200.
## Instrucciones

Adapta nuestro algoritmo de aprendizaje por refuerzo para resolver el problema del coche de montaña. Comienza con el código existente en [notebook.ipynb](../../../../8-Reinforcement/2-Gym/notebook.ipynb), sustituye el nuevo entorno, cambia las funciones de discretización del estado, e intenta hacer que el algoritmo existente entrene con mínimas modificaciones de código. Optimiza el resultado ajustando los hiperparámetros.

> **Nota**: Es probable que sea necesario ajustar los hiperparámetros para que el algoritmo converja.
## Rúbrica

| Criterios | Ejemplar | Adecuado | Necesita Mejorar |
| --------- | -------- | -------- | ---------------- |
|           | El algoritmo Q-Learning se adapta exitosamente desde el ejemplo de CartPole, con mínimas modificaciones de código, y es capaz de resolver el problema de capturar la bandera en menos de 200 pasos. | Se ha adoptado un nuevo algoritmo Q-Learning de Internet, pero está bien documentado; o se ha adoptado el algoritmo existente, pero no alcanza los resultados deseados. | El estudiante no pudo adoptar exitosamente ningún algoritmo, pero ha dado pasos sustanciales hacia la solución (implementó discretización del estado, estructura de datos Q-Table, etc.) |

**Descargo de responsabilidad**:
Este documento ha sido traducido utilizando servicios de traducción automática basados en inteligencia artificial. Aunque nos esforzamos por lograr precisión, tenga en cuenta que las traducciones automatizadas pueden contener errores o inexactitudes. El documento original en su idioma nativo debe considerarse la fuente autorizada. Para información crítica, se recomienda una traducción humana profesional. No nos hacemos responsables de cualquier malentendido o interpretación errónea que surja del uso de esta traducción.