# Un Mundo Más Realista

En nuestra situación, Peter podía moverse casi sin cansarse ni tener hambre. En un mundo más realista, tiene que sentarse y descansar de vez en cuando, y también alimentarse. Hagamos nuestro mundo más realista implementando las siguientes reglas:

1. Al moverse de un lugar a otro, Peter pierde **energía** y gana algo de **fatiga**.
2. Peter puede ganar más energía comiendo manzanas.
3. Peter puede deshacerse de la fatiga descansando bajo el árbol o en el césped (es decir, caminando hacia una ubicación en el tablero con un árbol o césped - campo verde).
4. Peter necesita encontrar y matar al lobo.
5. Para matar al lobo, Peter necesita tener ciertos niveles de energía y fatiga, de lo contrario pierde la batalla.

## Instrucciones

Usa el cuaderno original [notebook.ipynb](../../../../8-Reinforcement/1-QLearning/notebook.ipynb) como punto de partida para tu solución.

Modifica la función de recompensa anterior de acuerdo con las reglas del juego, ejecuta el algoritmo de aprendizaje por refuerzo para aprender la mejor estrategia para ganar el juego y compara los resultados del paseo aleatorio con tu algoritmo en términos de número de juegos ganados y perdidos.

> **Note**: En tu nuevo mundo, el estado es más complejo y, además de la posición del humano, también incluye los niveles de fatiga y energía. Puedes elegir representar el estado como una tupla (Board, energy, fatigue), o definir una clase para el estado (también puedes derivarla de `Board`), o incluso modificar la clase original `Board` dentro de [rlboard.py](../../../../8-Reinforcement/1-QLearning/rlboard.py).

En tu solución, por favor mantén el código responsable de la estrategia de paseo aleatorio y compara los resultados de tu algoritmo con el paseo aleatorio al final.

> **Note**: Es posible que necesites ajustar los hiperparámetros para que funcione, especialmente el número de épocas. Debido a que el éxito del juego (luchar contra el lobo) es un evento raro, puedes esperar un tiempo de entrenamiento mucho más largo.

## Rúbrica

| Criterios | Ejemplar                                                                                                                                                                                                 | Adecuado                                                                                                                                                                                | Necesita Mejorar                                                                                                                        |
| --------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
|           | Se presenta un cuaderno con la definición de las nuevas reglas del mundo, el algoritmo de Q-Learning y algunas explicaciones textuales. Q-Learning es capaz de mejorar significativamente los resultados en comparación con el paseo aleatorio. | Se presenta un cuaderno, se implementa Q-Learning y mejora los resultados en comparación con el paseo aleatorio, pero no significativamente; o el cuaderno está mal documentado y el código no está bien estructurado. | Se hace algún intento de redefinir las reglas del mundo, pero el algoritmo de Q-Learning no funciona, o la función de recompensa no está completamente definida. |

**Descargo de responsabilidad**:
Este documento ha sido traducido utilizando servicios de traducción automática por IA. Si bien nos esforzamos por lograr precisión, tenga en cuenta que las traducciones automáticas pueden contener errores o inexactitudes. El documento original en su idioma nativo debe considerarse la fuente autorizada. Para información crítica, se recomienda la traducción humana profesional. No somos responsables de ningún malentendido o interpretación errónea que surja del uso de esta traducción.