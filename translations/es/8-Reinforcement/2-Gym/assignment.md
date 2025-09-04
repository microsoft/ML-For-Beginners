<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1f2b7441745eb52e25745423b247016b",
  "translation_date": "2025-09-04T00:31:37+00:00",
  "source_file": "8-Reinforcement/2-Gym/assignment.md",
  "language_code": "es"
}
-->
# Entrenar Mountain Car

[OpenAI Gym](http://gym.openai.com) ha sido diseñado de tal manera que todos los entornos proporcionan la misma API, es decir, los mismos métodos `reset`, `step` y `render`, y las mismas abstracciones de **espacio de acción** y **espacio de observación**. Por lo tanto, debería ser posible adaptar los mismos algoritmos de aprendizaje por refuerzo a diferentes entornos con cambios mínimos en el código.

## Un entorno de Mountain Car

El [entorno Mountain Car](https://gym.openai.com/envs/MountainCar-v0/) contiene un coche atrapado en un valle:

El objetivo es salir del valle y capturar la bandera, realizando en cada paso una de las siguientes acciones:

| Valor | Significado |
|---|---|
| 0 | Acelerar hacia la izquierda |
| 1 | No acelerar |
| 2 | Acelerar hacia la derecha |

El principal truco de este problema, sin embargo, es que el motor del coche no es lo suficientemente potente como para escalar la montaña en un solo intento. Por lo tanto, la única forma de tener éxito es conducir de un lado a otro para acumular impulso.

El espacio de observación consta de solo dos valores:

| Num | Observación   | Min  | Max  |
|-----|--------------|------|------|
|  0  | Posición del coche | -1.2 | 0.6  |
|  1  | Velocidad del coche | -0.07 | 0.07 |

El sistema de recompensas para el Mountain Car es bastante complicado:

 * Se otorga una recompensa de 0 si el agente alcanza la bandera (posición = 0.5) en la cima de la montaña.
 * Se otorga una recompensa de -1 si la posición del agente es menor a 0.5.

El episodio termina si la posición del coche es mayor a 0.5, o si la duración del episodio supera los 200 pasos.

## Instrucciones

Adapta nuestro algoritmo de aprendizaje por refuerzo para resolver el problema del Mountain Car. Comienza con el código existente en [notebook.ipynb](notebook.ipynb), sustituye el nuevo entorno, cambia las funciones de discretización de estados y trata de hacer que el algoritmo existente entrene con modificaciones mínimas de código. Optimiza el resultado ajustando los hiperparámetros.

> **Nota**: Es probable que se necesite ajustar los hiperparámetros para que el algoritmo converja.

## Rubrica

| Criterio | Ejemplar | Adecuado | Necesita Mejoras |
| -------- | --------- | -------- | ---------------- |
|          | El algoritmo de Q-Learning se adapta exitosamente desde el ejemplo de CartPole, con modificaciones mínimas de código, y es capaz de resolver el problema de capturar la bandera en menos de 200 pasos. | Se ha adoptado un nuevo algoritmo de Q-Learning desde Internet, pero está bien documentado; o se ha adoptado el algoritmo existente, pero no alcanza los resultados deseados. | El estudiante no pudo adoptar exitosamente ningún algoritmo, pero ha dado pasos sustanciales hacia la solución (implementó la discretización de estados, la estructura de datos de la Q-Table, etc.). |

---

**Descargo de responsabilidad**:  
Este documento ha sido traducido utilizando el servicio de traducción automática [Co-op Translator](https://github.com/Azure/co-op-translator). Si bien nos esforzamos por lograr precisión, tenga en cuenta que las traducciones automáticas pueden contener errores o imprecisiones. El documento original en su idioma nativo debe considerarse como la fuente autorizada. Para información crítica, se recomienda una traducción profesional realizada por humanos. No nos hacemos responsables de malentendidos o interpretaciones erróneas que puedan surgir del uso de esta traducción.