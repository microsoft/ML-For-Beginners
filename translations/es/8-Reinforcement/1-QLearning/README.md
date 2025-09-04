<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "0ffe994d1cc881bdeb49226a064116e5",
  "translation_date": "2025-09-04T00:19:42+00:00",
  "source_file": "8-Reinforcement/1-QLearning/README.md",
  "language_code": "es"
}
-->
# Introducción al Aprendizaje por Refuerzo y Q-Learning

![Resumen del refuerzo en el aprendizaje automático en un sketchnote](../../../../translated_images/ml-reinforcement.94024374d63348dbb3571c343ca7ddabef72adac0b8086d47164b769ba3a8a1d.es.png)
> Sketchnote por [Tomomi Imura](https://www.twitter.com/girlie_mac)

El aprendizaje por refuerzo implica tres conceptos importantes: el agente, algunos estados y un conjunto de acciones por estado. Al ejecutar una acción en un estado específico, el agente recibe una recompensa. Imagina nuevamente el videojuego Super Mario. Tú eres Mario, estás en un nivel del juego, parado junto al borde de un acantilado. Sobre ti hay una moneda. Tú, siendo Mario, en un nivel del juego, en una posición específica... ese es tu estado. Moverte un paso hacia la derecha (una acción) te llevará al borde del acantilado, y eso te daría una puntuación numérica baja. Sin embargo, presionar el botón de salto te permitiría obtener un punto y seguir vivo. Ese es un resultado positivo y debería otorgarte una puntuación numérica positiva.

Usando aprendizaje por refuerzo y un simulador (el juego), puedes aprender a jugar para maximizar la recompensa, que es mantenerse vivo y obtener la mayor cantidad de puntos posible.

[![Introducción al Aprendizaje por Refuerzo](https://img.youtube.com/vi/lDq_en8RNOo/0.jpg)](https://www.youtube.com/watch?v=lDq_en8RNOo)

> 🎥 Haz clic en la imagen de arriba para escuchar a Dmitry hablar sobre el Aprendizaje por Refuerzo

## [Cuestionario previo a la lección](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/45/)

## Prerrequisitos y Configuración

En esta lección, experimentaremos con algo de código en Python. Deberías poder ejecutar el código del Jupyter Notebook de esta lección, ya sea en tu computadora o en la nube.

Puedes abrir [el notebook de la lección](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/notebook.ipynb) y seguir esta lección para construir.

> **Nota:** Si estás abriendo este código desde la nube, también necesitas obtener el archivo [`rlboard.py`](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/rlboard.py), que se utiliza en el código del notebook. Agrégalo al mismo directorio que el notebook.

## Introducción

En esta lección, exploraremos el mundo de **[Pedro y el Lobo](https://es.wikipedia.org/wiki/Pedro_y_el_lobo)**, inspirado en un cuento musical de un compositor ruso, [Sergei Prokofiev](https://es.wikipedia.org/wiki/Sergu%C3%A9i_Prok%C3%B3fiev). Usaremos **Aprendizaje por Refuerzo** para permitir que Pedro explore su entorno, recoja manzanas deliciosas y evite encontrarse con el lobo.

El **Aprendizaje por Refuerzo** (RL) es una técnica de aprendizaje que nos permite aprender un comportamiento óptimo de un **agente** en algún **entorno** mediante la ejecución de muchos experimentos. Un agente en este entorno debe tener algún **objetivo**, definido por una **función de recompensa**.

## El entorno

Para simplificar, consideremos el mundo de Pedro como un tablero cuadrado de tamaño `ancho` x `alto`, como este:

![Entorno de Pedro](../../../../translated_images/environment.40ba3cb66256c93fa7e92f6f7214e1d1f588aafa97d266c11d108c5c5d101b6c.es.png)

Cada celda en este tablero puede ser:

* **suelo**, sobre el cual Pedro y otras criaturas pueden caminar.
* **agua**, sobre la cual obviamente no puedes caminar.
* un **árbol** o **hierba**, un lugar donde puedes descansar.
* una **manzana**, que representa algo que Pedro estaría encantado de encontrar para alimentarse.
* un **lobo**, que es peligroso y debe evitarse.

Hay un módulo de Python separado, [`rlboard.py`](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/rlboard.py), que contiene el código para trabajar con este entorno. Dado que este código no es importante para entender nuestros conceptos, importaremos el módulo y lo usaremos para crear el tablero de muestra (bloque de código 1):

```python
from rlboard import *

width, height = 8,8
m = Board(width,height)
m.randomize(seed=13)
m.plot()
```

Este código debería imprimir una imagen del entorno similar a la anterior.

## Acciones y política

En nuestro ejemplo, el objetivo de Pedro sería encontrar una manzana, mientras evita al lobo y otros obstáculos. Para ello, esencialmente puede caminar por el tablero hasta encontrar una manzana.

Por lo tanto, en cualquier posición, puede elegir entre una de las siguientes acciones: arriba, abajo, izquierda y derecha.

Definiremos esas acciones como un diccionario y las mapearemos a pares de cambios de coordenadas correspondientes. Por ejemplo, moverse a la derecha (`R`) correspondería a un par `(1,0)`. (bloque de código 2):

```python
actions = { "U" : (0,-1), "D" : (0,1), "L" : (-1,0), "R" : (1,0) }
action_idx = { a : i for i,a in enumerate(actions.keys()) }
```

En resumen, la estrategia y el objetivo de este escenario son los siguientes:

- **La estrategia** de nuestro agente (Pedro) está definida por una llamada **política**. Una política es una función que devuelve la acción en cualquier estado dado. En nuestro caso, el estado del problema está representado por el tablero, incluida la posición actual del jugador.

- **El objetivo** del aprendizaje por refuerzo es eventualmente aprender una buena política que nos permita resolver el problema de manera eficiente. Sin embargo, como línea base, consideremos la política más simple llamada **camino aleatorio**.

## Camino aleatorio

Primero resolvamos nuestro problema implementando una estrategia de camino aleatorio. Con el camino aleatorio, elegiremos aleatoriamente la siguiente acción de las acciones permitidas, hasta que lleguemos a la manzana (bloque de código 3).

1. Implementa el camino aleatorio con el siguiente código:

    ```python
    def random_policy(m):
        return random.choice(list(actions))
    
    def walk(m,policy,start_position=None):
        n = 0 # number of steps
        # set initial position
        if start_position:
            m.human = start_position 
        else:
            m.random_start()
        while True:
            if m.at() == Board.Cell.apple:
                return n # success!
            if m.at() in [Board.Cell.wolf, Board.Cell.water]:
                return -1 # eaten by wolf or drowned
            while True:
                a = actions[policy(m)]
                new_pos = m.move_pos(m.human,a)
                if m.is_valid(new_pos) and m.at(new_pos)!=Board.Cell.water:
                    m.move(a) # do the actual move
                    break
            n+=1
    
    walk(m,random_policy)
    ```

    La llamada a `walk` debería devolver la longitud del camino correspondiente, que puede variar de una ejecución a otra.

1. Ejecuta el experimento de camino varias veces (digamos, 100), y muestra las estadísticas resultantes (bloque de código 4):

    ```python
    def print_statistics(policy):
        s,w,n = 0,0,0
        for _ in range(100):
            z = walk(m,policy)
            if z<0:
                w+=1
            else:
                s += z
                n += 1
        print(f"Average path length = {s/n}, eaten by wolf: {w} times")
    
    print_statistics(random_policy)
    ```

    Nota que la longitud promedio de un camino es de alrededor de 30-40 pasos, lo cual es bastante, dado que la distancia promedio a la manzana más cercana es de alrededor de 5-6 pasos.

    También puedes ver cómo se mueve Pedro durante el camino aleatorio:

    ![Camino aleatorio de Pedro](../../../../8-Reinforcement/1-QLearning/images/random_walk.gif)

## Función de recompensa

Para hacer nuestra política más inteligente, necesitamos entender qué movimientos son "mejores" que otros. Para ello, necesitamos definir nuestro objetivo.

El objetivo puede definirse en términos de una **función de recompensa**, que devolverá algún valor de puntuación para cada estado. Cuanto mayor sea el número, mejor será la función de recompensa. (bloque de código 5)

```python
move_reward = -0.1
goal_reward = 10
end_reward = -10

def reward(m,pos=None):
    pos = pos or m.human
    if not m.is_valid(pos):
        return end_reward
    x = m.at(pos)
    if x==Board.Cell.water or x == Board.Cell.wolf:
        return end_reward
    if x==Board.Cell.apple:
        return goal_reward
    return move_reward
```

Una cosa interesante sobre las funciones de recompensa es que en la mayoría de los casos, *solo se nos da una recompensa sustancial al final del juego*. Esto significa que nuestro algoritmo debería recordar los pasos "buenos" que conducen a una recompensa positiva al final y aumentar su importancia. De manera similar, todos los movimientos que conducen a malos resultados deberían desalentarse.

## Q-Learning

El algoritmo que discutiremos aquí se llama **Q-Learning**. En este algoritmo, la política está definida por una función (o una estructura de datos) llamada **Q-Table**. Registra la "bondad" de cada una de las acciones en un estado dado.

Se llama Q-Table porque a menudo es conveniente representarla como una tabla o un arreglo multidimensional. Dado que nuestro tablero tiene dimensiones `ancho` x `alto`, podemos representar la Q-Table usando un arreglo numpy con forma `ancho` x `alto` x `len(actions)`: (bloque de código 6)

```python
Q = np.ones((width,height,len(actions)),dtype=np.float)*1.0/len(actions)
```

Nota que inicializamos todos los valores de la Q-Table con un valor igual, en nuestro caso - 0.25. Esto corresponde a la política de "camino aleatorio", porque todos los movimientos en cada estado son igualmente buenos. Podemos pasar la Q-Table a la función `plot` para visualizar la tabla en el tablero: `m.plot(Q)`.

![Entorno de Pedro](../../../../translated_images/env_init.04e8f26d2d60089e128f21d22e5fef57d580e559f0d5937b06c689e5e7cdd438.es.png)

En el centro de cada celda hay una "flecha" que indica la dirección preferida de movimiento. Dado que todas las direcciones son iguales, se muestra un punto.

Ahora necesitamos ejecutar la simulación, explorar nuestro entorno y aprender una mejor distribución de valores de la Q-Table, lo que nos permitirá encontrar el camino hacia la manzana mucho más rápido.

## Esencia del Q-Learning: Ecuación de Bellman

Una vez que comenzamos a movernos, cada acción tendrá una recompensa correspondiente, es decir, teóricamente podemos seleccionar la siguiente acción basada en la recompensa inmediata más alta. Sin embargo, en la mayoría de los estados, el movimiento no logrará nuestro objetivo de alcanzar la manzana, y por lo tanto no podemos decidir inmediatamente qué dirección es mejor.

> Recuerda que no importa el resultado inmediato, sino el resultado final, que obtendremos al final de la simulación.

Para tener en cuenta esta recompensa diferida, necesitamos usar los principios de **[programación dinámica](https://es.wikipedia.org/wiki/Programaci%C3%B3n_din%C3%A1mica)**, que nos permiten pensar en nuestro problema de manera recursiva.

Supongamos que ahora estamos en el estado *s*, y queremos movernos al siguiente estado *s'*. Al hacerlo, recibiremos la recompensa inmediata *r(s,a)*, definida por la función de recompensa, más alguna recompensa futura. Si suponemos que nuestra Q-Table refleja correctamente la "atractividad" de cada acción, entonces en el estado *s'* elegiremos una acción *a* que corresponda al valor máximo de *Q(s',a')*. Así, la mejor recompensa futura posible que podríamos obtener en el estado *s* se definirá como `max`

## Verificando la política

Dado que la Q-Table enumera la "atractividad" de cada acción en cada estado, es bastante sencillo usarla para definir la navegación eficiente en nuestro mundo. En el caso más simple, podemos seleccionar la acción correspondiente al valor más alto de la Q-Table: (bloque de código 9)

```python
def qpolicy_strict(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = list(actions)[np.argmax(v)]
        return a

walk(m,qpolicy_strict)
```

> Si pruebas el código anterior varias veces, puede que notes que a veces se "queda colgado" y necesitas presionar el botón STOP en el notebook para interrumpirlo. Esto ocurre porque podría haber situaciones en las que dos estados "apuntan" el uno al otro en términos de valor óptimo de Q, en cuyo caso el agente termina moviéndose entre esos estados indefinidamente.

## 🚀Desafío

> **Tarea 1:** Modifica la función `walk` para limitar la longitud máxima del camino a un cierto número de pasos (por ejemplo, 100), y observa cómo el código anterior devuelve este valor de vez en cuando.

> **Tarea 2:** Modifica la función `walk` para que no regrese a los lugares donde ya ha estado previamente. Esto evitará que `walk` entre en bucles, sin embargo, el agente aún puede terminar "atrapado" en una ubicación de la que no pueda escapar.

## Navegación

Una política de navegación mejor sería la que usamos durante el entrenamiento, que combina explotación y exploración. En esta política, seleccionaremos cada acción con cierta probabilidad, proporcional a los valores en la Q-Table. Esta estrategia aún puede resultar en que el agente regrese a una posición que ya ha explorado, pero, como puedes ver en el código a continuación, resulta en un camino promedio muy corto hacia la ubicación deseada (recuerda que `print_statistics` ejecuta la simulación 100 veces): (bloque de código 10)

```python
def qpolicy(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = random.choices(list(actions),weights=v)[0]
        return a

print_statistics(qpolicy)
```

Después de ejecutar este código, deberías obtener una longitud promedio de camino mucho más pequeña que antes, en el rango de 3-6.

## Investigando el proceso de aprendizaje

Como hemos mencionado, el proceso de aprendizaje es un equilibrio entre la exploración y la explotación del conocimiento adquirido sobre la estructura del espacio del problema. Hemos visto que los resultados del aprendizaje (la capacidad de ayudar a un agente a encontrar un camino corto hacia el objetivo) han mejorado, pero también es interesante observar cómo se comporta la longitud promedio del camino durante el proceso de aprendizaje:

## Los aprendizajes pueden resumirse como:

- **La longitud promedio del camino aumenta**. Lo que vemos aquí es que al principio, la longitud promedio del camino aumenta. Esto probablemente se debe al hecho de que cuando no sabemos nada sobre el entorno, es más probable que quedemos atrapados en estados desfavorables, como agua o lobos. A medida que aprendemos más y comenzamos a usar este conocimiento, podemos explorar el entorno por más tiempo, pero aún no sabemos muy bien dónde están las manzanas.

- **La longitud del camino disminuye a medida que aprendemos más**. Una vez que aprendemos lo suficiente, se vuelve más fácil para el agente alcanzar el objetivo, y la longitud del camino comienza a disminuir. Sin embargo, aún estamos abiertos a la exploración, por lo que a menudo nos desviamos del mejor camino y exploramos nuevas opciones, haciendo que el camino sea más largo de lo óptimo.

- **La longitud aumenta abruptamente**. Lo que también observamos en este gráfico es que en algún momento, la longitud aumentó abruptamente. Esto indica la naturaleza estocástica del proceso, y que en algún momento podemos "estropear" los coeficientes de la Q-Table sobrescribiéndolos con nuevos valores. Esto idealmente debería minimizarse disminuyendo la tasa de aprendizaje (por ejemplo, hacia el final del entrenamiento, solo ajustamos los valores de la Q-Table por un pequeño valor).

En general, es importante recordar que el éxito y la calidad del proceso de aprendizaje dependen significativamente de parámetros como la tasa de aprendizaje, la disminución de la tasa de aprendizaje y el factor de descuento. A menudo se les llama **hiperparámetros**, para distinguirlos de los **parámetros**, que optimizamos durante el entrenamiento (por ejemplo, los coeficientes de la Q-Table). El proceso de encontrar los mejores valores de hiperparámetros se llama **optimización de hiperparámetros**, y merece un tema aparte.

## [Cuestionario post-lectura](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/46/)

## Asignación 
[Un Mundo Más Realista](assignment.md)

---

**Descargo de responsabilidad**:  
Este documento ha sido traducido utilizando el servicio de traducción automática [Co-op Translator](https://github.com/Azure/co-op-translator). Si bien nos esforzamos por garantizar la precisión, tenga en cuenta que las traducciones automatizadas pueden contener errores o imprecisiones. El documento original en su idioma nativo debe considerarse la fuente autorizada. Para información crítica, se recomienda una traducción profesional realizada por humanos. No nos hacemos responsables de malentendidos o interpretaciones erróneas que puedan surgir del uso de esta traducción.