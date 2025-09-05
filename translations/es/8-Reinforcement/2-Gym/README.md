<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "107d5bb29da8a562e7ae72262d251a75",
  "translation_date": "2025-09-04T22:26:14+00:00",
  "source_file": "8-Reinforcement/2-Gym/README.md",
  "language_code": "es"
}
-->
# CartPole Patinaje

El problema que resolvimos en la lección anterior puede parecer un problema de juguete, sin mucha aplicación en escenarios de la vida real. Sin embargo, este no es el caso, ya que muchos problemas del mundo real comparten características similares, como jugar al ajedrez o al Go. Son similares porque también tenemos un tablero con reglas definidas y un **estado discreto**.

## [Cuestionario previo a la lección](https://ff-quizzes.netlify.app/en/ml/)

## Introducción

En esta lección aplicaremos los mismos principios de Q-Learning a un problema con un **estado continuo**, es decir, un estado definido por uno o más números reales. Abordaremos el siguiente problema:

> **Problema**: Si Pedro quiere escapar del lobo, necesita aprender a moverse más rápido. Veremos cómo Pedro puede aprender a patinar, en particular, a mantener el equilibrio, utilizando Q-Learning.

![¡La gran escapada!](../../../../8-Reinforcement/2-Gym/images/escape.png)

> ¡Pedro y sus amigos se ponen creativos para escapar del lobo! Imagen de [Jen Looper](https://twitter.com/jenlooper)

Usaremos una versión simplificada del equilibrio conocida como el problema del **CartPole**. En el mundo del CartPole, tenemos un deslizador horizontal que puede moverse a la izquierda o a la derecha, y el objetivo es equilibrar un palo vertical sobre el deslizador.

## Prerrequisitos

En esta lección, utilizaremos una biblioteca llamada **OpenAI Gym** para simular diferentes **entornos**. Puedes ejecutar el código de esta lección localmente (por ejemplo, desde Visual Studio Code), en cuyo caso la simulación se abrirá en una nueva ventana. Si ejecutas el código en línea, es posible que necesites hacer algunos ajustes, como se describe [aquí](https://towardsdatascience.com/rendering-openai-gym-envs-on-binder-and-google-colab-536f99391cc7).

## OpenAI Gym

En la lección anterior, las reglas del juego y el estado estaban definidos por la clase `Board` que creamos nosotros mismos. Aquí utilizaremos un **entorno de simulación** especial, que simulará la física detrás del palo en equilibrio. Uno de los entornos de simulación más populares para entrenar algoritmos de aprendizaje por refuerzo se llama [Gym](https://gym.openai.com/), mantenido por [OpenAI](https://openai.com/). Usando este Gym, podemos crear diferentes **entornos**, desde una simulación de CartPole hasta juegos de Atari.

> **Nota**: Puedes ver otros entornos disponibles en OpenAI Gym [aquí](https://gym.openai.com/envs/#classic_control).

Primero, instalemos Gym e importemos las bibliotecas necesarias (bloque de código 1):

```python
import sys
!{sys.executable} -m pip install gym 

import gym
import matplotlib.pyplot as plt
import numpy as np
import random
```

## Ejercicio - inicializar un entorno de CartPole

Para trabajar con el problema de equilibrio de CartPole, necesitamos inicializar el entorno correspondiente. Cada entorno está asociado con:

- Un **espacio de observación** que define la estructura de la información que recibimos del entorno. Para el problema de CartPole, recibimos la posición del palo, la velocidad y otros valores.

- Un **espacio de acción** que define las acciones posibles. En nuestro caso, el espacio de acción es discreto y consta de dos acciones: **izquierda** y **derecha**. (bloque de código 2)

1. Para inicializar, escribe el siguiente código:

    ```python
    env = gym.make("CartPole-v1")
    print(env.action_space)
    print(env.observation_space)
    print(env.action_space.sample())
    ```

Para ver cómo funciona el entorno, ejecutemos una simulación corta de 100 pasos. En cada paso, proporcionamos una de las acciones a realizar; en esta simulación simplemente seleccionamos una acción aleatoriamente del `action_space`.

1. Ejecuta el siguiente código y observa el resultado.

    ✅ Recuerda que es preferible ejecutar este código en una instalación local de Python. (bloque de código 3)

    ```python
    env.reset()
    
    for i in range(100):
       env.render()
       env.step(env.action_space.sample())
    env.close()
    ```

    Deberías ver algo similar a esta imagen:

    ![CartPole sin equilibrio](../../../../8-Reinforcement/2-Gym/images/cartpole-nobalance.gif)

1. Durante la simulación, necesitamos obtener observaciones para decidir cómo actuar. De hecho, la función `step` devuelve las observaciones actuales, una función de recompensa y una bandera `done` que indica si tiene sentido continuar la simulación o no: (bloque de código 4)

    ```python
    env.reset()
    
    done = False
    while not done:
       env.render()
       obs, rew, done, info = env.step(env.action_space.sample())
       print(f"{obs} -> {rew}")
    env.close()
    ```

    Verás algo como esto en la salida del notebook:

    ```text
    [ 0.03403272 -0.24301182  0.02669811  0.2895829 ] -> 1.0
    [ 0.02917248 -0.04828055  0.03248977  0.00543839] -> 1.0
    [ 0.02820687  0.14636075  0.03259854 -0.27681916] -> 1.0
    [ 0.03113408  0.34100283  0.02706215 -0.55904489] -> 1.0
    [ 0.03795414  0.53573468  0.01588125 -0.84308041] -> 1.0
    ...
    [ 0.17299878  0.15868546 -0.20754175 -0.55975453] -> 1.0
    [ 0.17617249  0.35602306 -0.21873684 -0.90998894] -> 1.0
    ```

    El vector de observación que se devuelve en cada paso de la simulación contiene los siguientes valores:
    - Posición del carrito
    - Velocidad del carrito
    - Ángulo del palo
    - Tasa de rotación del palo

1. Obtén el valor mínimo y máximo de esos números: (bloque de código 5)

    ```python
    print(env.observation_space.low)
    print(env.observation_space.high)
    ```

    También notarás que el valor de recompensa en cada paso de la simulación siempre es 1. Esto se debe a que nuestro objetivo es sobrevivir el mayor tiempo posible, es decir, mantener el palo en una posición razonablemente vertical durante el mayor tiempo posible.

    ✅ De hecho, la simulación de CartPole se considera resuelta si logramos obtener una recompensa promedio de 195 en 100 ensayos consecutivos.

## Discretización del estado

En Q-Learning, necesitamos construir una Q-Table que defina qué hacer en cada estado. Para poder hacer esto, el estado debe ser **discreto**, más precisamente, debe contener un número finito de valores discretos. Por lo tanto, necesitamos de alguna manera **discretizar** nuestras observaciones, mapeándolas a un conjunto finito de estados.

Hay algunas formas de hacer esto:

- **Dividir en intervalos**. Si conocemos el intervalo de un valor determinado, podemos dividir este intervalo en un número de **intervalos**, y luego reemplazar el valor por el número del intervalo al que pertenece. Esto se puede hacer utilizando el método [`digitize`](https://numpy.org/doc/stable/reference/generated/numpy.digitize.html) de numpy. En este caso, conoceremos con precisión el tamaño del estado, ya que dependerá del número de intervalos que seleccionemos para la digitalización.

✅ Podemos usar interpolación lineal para llevar los valores a un intervalo finito (por ejemplo, de -20 a 20), y luego convertir los números a enteros redondeándolos. Esto nos da un poco menos de control sobre el tamaño del estado, especialmente si no conocemos los rangos exactos de los valores de entrada. Por ejemplo, en nuestro caso, 2 de los 4 valores no tienen límites superiores/inferiores, lo que puede resultar en un número infinito de estados.

En nuestro ejemplo, utilizaremos el segundo enfoque. Como notarás más adelante, a pesar de los límites superiores/inferiores indefinidos, esos valores rara vez toman valores fuera de ciertos intervalos finitos, por lo que esos estados con valores extremos serán muy raros.

1. Aquí está la función que tomará la observación de nuestro modelo y producirá una tupla de 4 valores enteros: (bloque de código 6)

    ```python
    def discretize(x):
        return tuple((x/np.array([0.25, 0.25, 0.01, 0.1])).astype(np.int))
    ```

1. Exploremos también otro método de discretización utilizando intervalos: (bloque de código 7)

    ```python
    def create_bins(i,num):
        return np.arange(num+1)*(i[1]-i[0])/num+i[0]
    
    print("Sample bins for interval (-5,5) with 10 bins\n",create_bins((-5,5),10))
    
    ints = [(-5,5),(-2,2),(-0.5,0.5),(-2,2)] # intervals of values for each parameter
    nbins = [20,20,10,10] # number of bins for each parameter
    bins = [create_bins(ints[i],nbins[i]) for i in range(4)]
    
    def discretize_bins(x):
        return tuple(np.digitize(x[i],bins[i]) for i in range(4))
    ```

1. Ahora ejecutemos una simulación corta y observemos esos valores discretos del entorno. Siéntete libre de probar tanto `discretize` como `discretize_bins` y observa si hay alguna diferencia.

    ✅ `discretize_bins` devuelve el número del intervalo, que comienza en 0. Por lo tanto, para valores de la variable de entrada cercanos a 0, devuelve el número del medio del intervalo (10). En `discretize`, no nos preocupamos por el rango de los valores de salida, permitiendo que sean negativos, por lo que los valores del estado no están desplazados, y 0 corresponde a 0. (bloque de código 8)

    ```python
    env.reset()
    
    done = False
    while not done:
       #env.render()
       obs, rew, done, info = env.step(env.action_space.sample())
       #print(discretize_bins(obs))
       print(discretize(obs))
    env.close()
    ```

    ✅ Descomenta la línea que comienza con `env.render` si deseas ver cómo se ejecuta el entorno. De lo contrario, puedes ejecutarlo en segundo plano, lo cual es más rápido. Usaremos esta ejecución "invisible" durante nuestro proceso de Q-Learning.

## La estructura de la Q-Table

En nuestra lección anterior, el estado era un simple par de números del 0 al 8, por lo que era conveniente representar la Q-Table con un tensor de numpy con una forma de 8x8x2. Si usamos la discretización por intervalos, el tamaño de nuestro vector de estado también es conocido, por lo que podemos usar el mismo enfoque y representar el estado con un array de forma 20x20x10x10x2 (aquí 2 es la dimensión del espacio de acción, y las primeras dimensiones corresponden al número de intervalos que seleccionamos para cada uno de los parámetros en el espacio de observación).

Sin embargo, a veces las dimensiones precisas del espacio de observación no son conocidas. En el caso de la función `discretize`, nunca podemos estar seguros de que nuestro estado se mantenga dentro de ciertos límites, ya que algunos de los valores originales no están acotados. Por lo tanto, utilizaremos un enfoque ligeramente diferente y representaremos la Q-Table con un diccionario.

1. Usa el par *(estado, acción)* como clave del diccionario, y el valor corresponderá al valor de la entrada en la Q-Table. (bloque de código 9)

    ```python
    Q = {}
    actions = (0,1)
    
    def qvalues(state):
        return [Q.get((state,a),0) for a in actions]
    ```

    Aquí también definimos una función `qvalues()`, que devuelve una lista de valores de la Q-Table para un estado dado que corresponde a todas las acciones posibles. Si la entrada no está presente en la Q-Table, devolveremos 0 como valor predeterminado.

## ¡Comencemos con Q-Learning!

Ahora estamos listos para enseñar a Pedro a mantener el equilibrio.

1. Primero, definamos algunos hiperparámetros: (bloque de código 10)

    ```python
    # hyperparameters
    alpha = 0.3
    gamma = 0.9
    epsilon = 0.90
    ```

    Aquí, `alpha` es la **tasa de aprendizaje** que define en qué medida debemos ajustar los valores actuales de la Q-Table en cada paso. En la lección anterior comenzamos con 1 y luego disminuimos `alpha` a valores más bajos durante el entrenamiento. En este ejemplo lo mantendremos constante por simplicidad, pero puedes experimentar ajustando los valores de `alpha` más adelante.

    `gamma` es el **factor de descuento** que muestra en qué medida debemos priorizar la recompensa futura sobre la recompensa actual.

    `epsilon` es el **factor de exploración/explotación** que determina si debemos preferir la exploración o la explotación. En nuestro algoritmo, en un porcentaje `epsilon` de los casos seleccionaremos la siguiente acción según los valores de la Q-Table, y en el porcentaje restante ejecutaremos una acción aleatoria. Esto nos permitirá explorar áreas del espacio de búsqueda que nunca hemos visto antes.

    ✅ En términos de equilibrio, elegir una acción aleatoria (exploración) actuaría como un empujón aleatorio en la dirección equivocada, y el palo tendría que aprender a recuperar el equilibrio de esos "errores".

### Mejorar el algoritmo

Podemos hacer dos mejoras a nuestro algoritmo de la lección anterior:

- **Calcular la recompensa acumulativa promedio**, durante un número de simulaciones. Imprimiremos el progreso cada 5000 iteraciones, y promediaremos nuestra recompensa acumulativa durante ese período de tiempo. Esto significa que si obtenemos más de 195 puntos, podemos considerar el problema resuelto, con una calidad incluso mayor a la requerida.

- **Calcular el resultado acumulativo promedio máximo**, `Qmax`, y almacenaremos la Q-Table correspondiente a ese resultado. Cuando ejecutes el entrenamiento, notarás que a veces el resultado acumulativo promedio comienza a disminuir, y queremos conservar los valores de la Q-Table que corresponden al mejor modelo observado durante el entrenamiento.

1. Recopila todas las recompensas acumulativas en cada simulación en el vector `rewards` para su posterior representación gráfica. (bloque de código 11)

    ```python
    def probs(v,eps=1e-4):
        v = v-v.min()+eps
        v = v/v.sum()
        return v
    
    Qmax = 0
    cum_rewards = []
    rewards = []
    for epoch in range(100000):
        obs = env.reset()
        done = False
        cum_reward=0
        # == do the simulation ==
        while not done:
            s = discretize(obs)
            if random.random()<epsilon:
                # exploitation - chose the action according to Q-Table probabilities
                v = probs(np.array(qvalues(s)))
                a = random.choices(actions,weights=v)[0]
            else:
                # exploration - randomly chose the action
                a = np.random.randint(env.action_space.n)
    
            obs, rew, done, info = env.step(a)
            cum_reward+=rew
            ns = discretize(obs)
            Q[(s,a)] = (1 - alpha) * Q.get((s,a),0) + alpha * (rew + gamma * max(qvalues(ns)))
        cum_rewards.append(cum_reward)
        rewards.append(cum_reward)
        # == Periodically print results and calculate average reward ==
        if epoch%5000==0:
            print(f"{epoch}: {np.average(cum_rewards)}, alpha={alpha}, epsilon={epsilon}")
            if np.average(cum_rewards) > Qmax:
                Qmax = np.average(cum_rewards)
                Qbest = Q
            cum_rewards=[]
    ```

Lo que puedes notar de estos resultados:

- **Cerca de nuestro objetivo**. Estamos muy cerca de alcanzar el objetivo de obtener 195 recompensas acumulativas en más de 100 ejecuciones consecutivas de la simulación, ¡o incluso podemos haberlo logrado! Incluso si obtenemos números más bajos, no lo sabremos con certeza, ya que promediamos sobre 5000 ejecuciones, y solo se requieren 100 ejecuciones según el criterio formal.

- **La recompensa comienza a disminuir**. A veces la recompensa comienza a disminuir, lo que significa que podemos "destruir" valores ya aprendidos en la Q-Table con otros que empeoran la situación.

Esta observación es más clara si graficamos el progreso del entrenamiento.

## Graficar el progreso del entrenamiento

Durante el entrenamiento, hemos recopilado el valor de la recompensa acumulativa en cada una de las iteraciones en el vector `rewards`. Así es como se ve cuando lo graficamos contra el número de iteración:

```python
plt.plot(rewards)
```

![progreso sin procesar](../../../../8-Reinforcement/2-Gym/images/train_progress_raw.png)

En este gráfico no es posible sacar conclusiones, ya que debido a la naturaleza del proceso de entrenamiento estocástico, la duración de las sesiones de entrenamiento varía mucho. Para darle más sentido a este gráfico, podemos calcular el **promedio móvil** sobre una serie de experimentos, digamos 100. Esto se puede hacer convenientemente usando `np.convolve`: (bloque de código 12)

```python
def running_average(x,window):
    return np.convolve(x,np.ones(window)/window,mode='valid')

plt.plot(running_average(rewards,100))
```

![progreso del entrenamiento](../../../../8-Reinforcement/2-Gym/images/train_progress_runav.png)

## Variar los hiperparámetros

Para hacer el aprendizaje más estable, tiene sentido ajustar algunos de nuestros hiperparámetros durante el entrenamiento. En particular:

- **Para la tasa de aprendizaje**, `alpha`, podemos comenzar con valores cercanos a 1 y luego ir disminuyendo el parámetro. Con el tiempo, obtendremos buenos valores de probabilidad en la Q-Table, y por lo tanto deberíamos ajustarlos ligeramente, y no sobrescribirlos completamente con nuevos valores.

- **Aumentar epsilon**. Podríamos querer aumentar lentamente el valor de `epsilon`, para explorar menos y explotar más. Probablemente tenga sentido comenzar con un valor bajo de `epsilon` y aumentarlo hasta casi 1.
> **Tarea 1**: Prueba con diferentes valores de hiperparámetros y observa si puedes lograr una recompensa acumulativa más alta. ¿Estás obteniendo más de 195?
> **Tarea 2**: Para resolver formalmente el problema, necesitas alcanzar un promedio de recompensa de 195 a lo largo de 100 ejecuciones consecutivas. Mide eso durante el entrenamiento y asegúrate de que has resuelto el problema formalmente.

## Ver el resultado en acción

Sería interesante ver cómo se comporta el modelo entrenado. Vamos a ejecutar la simulación y seguir la misma estrategia de selección de acciones que durante el entrenamiento, muestreando según la distribución de probabilidad en la Q-Table: (bloque de código 13)

```python
obs = env.reset()
done = False
while not done:
   s = discretize(obs)
   env.render()
   v = probs(np.array(qvalues(s)))
   a = random.choices(actions,weights=v)[0]
   obs,_,done,_ = env.step(a)
env.close()
```

Deberías ver algo como esto:

![un carrito equilibrando](../../../../8-Reinforcement/2-Gym/images/cartpole-balance.gif)

---

## 🚀Desafío

> **Tarea 3**: Aquí estuvimos utilizando la copia final de la Q-Table, que puede no ser la mejor. Recuerda que hemos almacenado la Q-Table con mejor rendimiento en la variable `Qbest`. ¡Prueba el mismo ejemplo con la Q-Table de mejor rendimiento copiando `Qbest` sobre `Q` y observa si notas alguna diferencia!

> **Tarea 4**: Aquí no estábamos seleccionando la mejor acción en cada paso, sino muestreando con la distribución de probabilidad correspondiente. ¿Tendría más sentido seleccionar siempre la mejor acción, con el valor más alto en la Q-Table? Esto se puede hacer utilizando la función `np.argmax` para encontrar el número de acción correspondiente al valor más alto en la Q-Table. Implementa esta estrategia y observa si mejora el equilibrio.

## [Cuestionario post-clase](https://ff-quizzes.netlify.app/en/ml/)

## Asignación
[Entrena un Mountain Car](assignment.md)

## Conclusión

Ahora hemos aprendido cómo entrenar agentes para lograr buenos resultados simplemente proporcionando una función de recompensa que define el estado deseado del juego y dándoles la oportunidad de explorar inteligentemente el espacio de búsqueda. Hemos aplicado con éxito el algoritmo de Q-Learning en casos de entornos discretos y continuos, pero con acciones discretas.

Es importante también estudiar situaciones donde el estado de las acciones sea continuo y cuando el espacio de observación sea mucho más complejo, como la imagen de la pantalla de un juego de Atari. En esos problemas, a menudo necesitamos usar técnicas de aprendizaje automático más poderosas, como redes neuronales, para lograr buenos resultados. Estos temas más avanzados serán el enfoque de nuestro próximo curso avanzado de IA.

---

**Descargo de responsabilidad**:  
Este documento ha sido traducido utilizando el servicio de traducción automática [Co-op Translator](https://github.com/Azure/co-op-translator). Aunque nos esforzamos por garantizar la precisión, tenga en cuenta que las traducciones automatizadas pueden contener errores o imprecisiones. El documento original en su idioma nativo debe considerarse como la fuente autorizada. Para información crítica, se recomienda una traducción profesional realizada por humanos. No nos hacemos responsables de malentendidos o interpretaciones erróneas que puedan surgir del uso de esta traducción.