# Un mundo más realista

En nuestro caso, Pedro (Peter) fue capaz de moverse y casi sin cansarse o estar hambriento. En un mundo más realista, él tiene que sentarse y descansar de vez en cuando, así como también alimentarse. Hagamos nuestro mundo más realista, al implementar las siguientes reglas:

1. Al moverse de un lugar a otro, Pedro pierde **energía** y obtiene algo de **fatiga**.
2. Pedro puede adquirir más energía al comer manzanas.
3. Pedro puede deshacerse de la fatiga al descansar bajo un árbol o en el pasto (ej. caminar en una ubicación del tablero con un árbol o pasto - campo verde)
4. Pedro necesita encontrar y matar al lobo
5. Con el fin de matar al lobo, Pedro necesita tener ciertos niveles de energía y fatiga, de lo contrario pierde la batalla.

## Instrucciones

Usa el notebook original [notebook.ipynb](../notebook.ipynb) como punto de partida para tu solución.

Modifica la función reward de arriba de acuerdo a las reglas del juego, ejecuta el algoritmo de aprendizaje reforzado para aprender la mejor estrategia para ganar el juego, y compara los resultados de caminata aleatoria con tu algoritmo en términos de el número de juegos ganados y perdidos.

> **Nota**: En tu nuevo mundo, el estado es más complejo, y además a la posición humana también incluye la fatiga y los niveles de energía. Puedes optar por representar el estado como una tupla (tablero, energía, fatiga), o definir una clase para el estado (también puedes querer derivarlo de `Board`), o incluso modifica la clase original `Board` dentro de [rlboard.py](../rlboard.py).

En tu solución, mantén el código responsable de la estrategia de caminata aleatoria, y compara los resultados de tu algoritmo con la caminata aleatoria al final.

> **Nota**: Puedes necesitar ajustar los hiperparámetros para hacerlo funcionar, especialmente el número de épocas. Porque el éxito del juego (pelear contra el lobo) es un evento raro, puedes esperar un tiempo de entrenamiento mayor.

## Rúbrica

| Criterio | Ejemplar                                                                                                                                                                                             | Adecuado                                                                                                                                                                                | Necesita mejorar                                                                                                                          |
| -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
|          | Se presentó un notebook con la definición de las nuevas reglas del mundo. El algoritmo Q-Learning y algunas explicaciones textuales. Q-Learning es capaz de mejorar significativamente los resultados comparado con la caminata aleatoria. | Se presentó un notebook, Q-Learning se implementó y mejoró los resultados comparado con la caminata aleatoria, pero no de forma significativa; o el notebook está pobremente documentado y el código no está bien estructurado | Se hicieron algunos intentos para redefinir las reglas del mundo, pero el algoritmo de Q-Learning no funciona, o la función reward no está totalmente definida |
