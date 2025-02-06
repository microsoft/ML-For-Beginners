## CartPole Patinaje

El problema que hemos estado resolviendo en la lección anterior podría parecer un problema de juguete, no realmente aplicable a escenarios de la vida real. Este no es el caso, porque muchos problemas del mundo real también comparten este escenario, incluyendo jugar al Ajedrez o al Go. Son similares, porque también tenemos un tablero con reglas dadas y un **estado discreto**.

## [Cuestionario previo a la lección](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/47/)

## Introducción

En esta lección aplicaremos los mismos principios de Q-Learning a un problema con **estado continuo**, es decir, un estado que se da por uno o más números reales. Abordaremos el siguiente problema:

> **Problema**: Si Peter quiere escapar del lobo, necesita poder moverse más rápido. Veremos cómo Peter puede aprender a patinar, en particular, a mantener el equilibrio, utilizando Q-Learning.

![¡La gran escapada!](../../../../translated_images/escape.18862db9930337e3fce23a9b6a76a06445f229dadea2268e12a6f0a1fde12115.es.png)

> ¡Peter y sus amigos se ponen creativos para escapar del lobo! Imagen de [Jen Looper](https://twitter.com/jenlooper)

Usaremos una versión simplificada del equilibrio conocida como el problema **CartPole**. En el mundo de CartPole, tenemos un deslizador horizontal que puede moverse hacia la izquierda o hacia la derecha, y el objetivo es equilibrar un poste vertical sobre el deslizador.

## Requisitos previos

En esta lección, utilizaremos una biblioteca llamada **OpenAI Gym** para simular diferentes **entornos**. Puedes ejecutar el código de esta lección localmente (por ejemplo, desde Visual Studio Code), en cuyo caso la simulación se abrirá en una nueva ventana. Al ejecutar el código en línea, es posible que necesites hacer algunos ajustes en el código, como se describe [aquí](https://towardsdatascience.com/rendering-openai-gym-envs-on-binder-and-google-colab-536f99391cc7).

## OpenAI Gym

En la lección anterior, las reglas del juego y el estado fueron dados por la clase `Board` que definimos nosotros mismos. Aquí utilizaremos un **entorno de simulación** especial, que simulará la física detrás del poste en equilibrio. Uno de los entornos de simulación más populares para entrenar algoritmos de aprendizaje por refuerzo se llama [Gym](https://gym.openai.com/), que es mantenido por [OpenAI](https://openai.com/). Usando este gym, podemos crear diferentes **entornos** desde una simulación de CartPole hasta juegos de Atari.

> **Nota**: Puedes ver otros entornos disponibles en OpenAI Gym [aquí](https://gym.openai.com/envs/#classic_control).

Primero, instalemos el gym e importemos las bibliotecas requeridas (bloque de código 1):

```python
import sys
!{sys.executable} -m pip install gym 

import gym
import matplotlib.pyplot as plt
import numpy as np
import random
```

## Ejercicio - inicializar un entorno de CartPole

Para trabajar con un problema de equilibrio de CartPole, necesitamos inicializar el entorno correspondiente. Cada entorno está asociado con:

- **Espacio de observación** que define la estructura de la información que recibimos del entorno. Para el problema de CartPole, recibimos la posición del poste, la velocidad y otros valores.

- **Espacio de acción** que define las posibles acciones. En nuestro caso, el espacio de acción es discreto y consta de dos acciones: **izquierda** y **derecha**. (bloque de código 2)

1. Para inicializar, escribe el siguiente código:

    ```python
    env = gym.make("CartPole-v1")
    print(env.action_space)
    print(env.observation_space)
    print(env.action_space.sample())
    ```

Para ver cómo funciona el entorno, ejecutemos una simulación corta de 100 pasos. En cada paso, proporcionamos una de las acciones a realizar; en esta simulación simplemente seleccionamos aleatoriamente una acción de `action_space`.

1. Ejecuta el código a continuación y observa qué sucede.

    ✅ Recuerda que es preferible ejecutar este código en una instalación local de Python. (bloque de código 3)

    ```python
    env.reset()
    
    for i in range(100):
       env.render()
       env.step(env.action_space.sample())
    env.close()
    ```

    Deberías estar viendo algo similar a esta imagen:

    ![CartPole sin equilibrio](../../../../8-Reinforcement/2-Gym/images/cartpole-nobalance.gif)

1. Durante la simulación, necesitamos obtener observaciones para decidir cómo actuar. De hecho, la función step devuelve las observaciones actuales, una función de recompensa y una bandera de finalización que indica si tiene sentido continuar la simulación o no: (bloque de código 4)

    ```python
    env.reset()
    
    done = False
    while not done:
       env.render()
       obs, rew, done, info = env.step(env.action_space.sample())
       print(f"{obs} -> {rew}")
    env.close()
    ```

    Terminarás viendo algo como esto en la salida del notebook:

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
    - Ángulo del poste
    - Tasa de rotación del poste

1. Obtén el valor mínimo y máximo de esos números: (bloque de código 5)

    ```python
    print(env.observation_space.low)
    print(env.observation_space.high)
    ```

    También puedes notar que el valor de recompensa en cada paso de la simulación es siempre 1. Esto se debe a que nuestro objetivo es sobrevivir el mayor tiempo posible, es decir, mantener el poste en una posición razonablemente vertical durante el mayor tiempo posible.

    ✅ De hecho, la simulación de CartPole se considera resuelta si logramos obtener una recompensa promedio de 195 durante 100 pruebas consecutivas.

## Discretización del estado

En Q-Learning, necesitamos construir una Q-Table que defina qué hacer en cada estado. Para poder hacer esto, necesitamos que el estado sea **discreto**, más precisamente, debe contener un número finito de valores discretos. Por lo tanto, necesitamos de alguna manera **discretizar** nuestras observaciones, mapeándolas a un conjunto finito de estados.

Hay algunas formas de hacer esto:

- **Dividir en contenedores**. Si conocemos el intervalo de un cierto valor, podemos dividir este intervalo en un número de **contenedores**, y luego reemplazar el valor por el número del contenedor al que pertenece. Esto se puede hacer utilizando el método [`digitize`](https://numpy.org/doc/stable/reference/generated/numpy.digitize.html) de numpy. En este caso, conoceremos precisamente el tamaño del estado, porque dependerá del número de contenedores que seleccionemos para la digitalización.

✅ Podemos usar la interpolación lineal para llevar los valores a un intervalo finito (digamos, de -20 a 20), y luego convertir los números en enteros redondeándolos. Esto nos da un poco menos de control sobre el tamaño del estado, especialmente si no conocemos los rangos exactos de los valores de entrada. Por ejemplo, en nuestro caso, 2 de los 4 valores no tienen límites superiores/inferiores en sus valores, lo que puede resultar en un número infinito de estados.

En nuestro ejemplo, utilizaremos el segundo enfoque. Como puedes notar más adelante, a pesar de los límites superiores/inferiores indefinidos, esos valores rara vez toman valores fuera de ciertos intervalos finitos, por lo que esos estados con valores extremos serán muy raros.

1. Aquí está la función que tomará la observación de nuestro modelo y producirá una tupla de 4 valores enteros: (bloque de código 6)

    ```python
    def discretize(x):
        return tuple((x/np.array([0.25, 0.25, 0.01, 0.1])).astype(np.int))
    ```

1. Exploremos también otro método de discretización utilizando contenedores: (bloque de código 7)

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

1. Ahora ejecutemos una simulación corta y observemos esos valores discretos del entorno. Siéntete libre de probar ambos `discretize` and `discretize_bins` y ver si hay alguna diferencia.

    ✅ discretize_bins devuelve el número del contenedor, que es basado en 0. Por lo tanto, para valores de la variable de entrada alrededor de 0, devuelve el número del medio del intervalo (10). En discretize, no nos preocupamos por el rango de valores de salida, permitiéndoles ser negativos, por lo que los valores del estado no están desplazados, y 0 corresponde a 0. (bloque de código 8)

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

    ✅ Descomenta la línea que comienza con env.render si deseas ver cómo se ejecuta el entorno. De lo contrario, puedes ejecutarlo en segundo plano, lo cual es más rápido. Usaremos esta ejecución "invisible" durante nuestro proceso de Q-Learning.

## La estructura de la Q-Table

En nuestra lección anterior, el estado era un simple par de números del 0 al 8, y por lo tanto era conveniente representar la Q-Table con un tensor numpy con una forma de 8x8x2. Si usamos la discretización de contenedores, el tamaño de nuestro vector de estado también es conocido, por lo que podemos usar el mismo enfoque y representar el estado con un array de forma 20x20x10x10x2 (aquí 2 es la dimensión del espacio de acción, y las primeras dimensiones corresponden al número de contenedores que hemos seleccionado para usar para cada uno de los parámetros en el espacio de observación).

Sin embargo, a veces las dimensiones precisas del espacio de observación no son conocidas. En el caso de la función `discretize`, nunca podemos estar seguros de que nuestro estado se mantenga dentro de ciertos límites, porque algunos de los valores originales no están limitados. Por lo tanto, usaremos un enfoque ligeramente diferente y representaremos la Q-Table con un diccionario.

1. Usa el par *(estado, acción)* como la clave del diccionario, y el valor correspondería al valor de entrada de la Q-Table. (bloque de código 9)

    ```python
    Q = {}
    actions = (0,1)
    
    def qvalues(state):
        return [Q.get((state,a),0) for a in actions]
    ```

    Aquí también definimos una función `qvalues()`, que devuelve una lista de valores de la Q-Table para un estado dado que corresponde a todas las posibles acciones. Si la entrada no está presente en la Q-Table, devolveremos 0 como valor predeterminado.

## Comencemos con el Q-Learning

¡Ahora estamos listos para enseñar a Peter a mantener el equilibrio!

1. Primero, establezcamos algunos hiperparámetros: (bloque de código 10)

    ```python
    # hyperparameters
    alpha = 0.3
    gamma = 0.9
    epsilon = 0.90
    ```

    Aquí, `alpha` is the **learning rate** that defines to which extent we should adjust the current values of Q-Table at each step. In the previous lesson we started with 1, and then decreased `alpha` to lower values during training. In this example we will keep it constant just for simplicity, and you can experiment with adjusting `alpha` values later.

    `gamma` is the **discount factor** that shows to which extent we should prioritize future reward over current reward.

    `epsilon` is the **exploration/exploitation factor** that determines whether we should prefer exploration to exploitation or vice versa. In our algorithm, we will in `epsilon` percent of the cases select the next action according to Q-Table values, and in the remaining number of cases we will execute a random action. This will allow us to explore areas of the search space that we have never seen before. 

    ✅ In terms of balancing - choosing random action (exploration) would act as a random punch in the wrong direction, and the pole would have to learn how to recover the balance from those "mistakes"

### Improve the algorithm

We can also make two improvements to our algorithm from the previous lesson:

- **Calculate average cumulative reward**, over a number of simulations. We will print the progress each 5000 iterations, and we will average out our cumulative reward over that period of time. It means that if we get more than 195 point - we can consider the problem solved, with even higher quality than required.
  
- **Calculate maximum average cumulative result**, `Qmax`, and we will store the Q-Table corresponding to that result. When you run the training you will notice that sometimes the average cumulative result starts to drop, and we want to keep the values of Q-Table that correspond to the best model observed during training.

1. Collect all cumulative rewards at each simulation at `rewards` vector para su posterior representación gráfica. (bloque de código  11)

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

Lo que puedes notar de esos resultados:

- **Cerca de nuestro objetivo**. Estamos muy cerca de alcanzar el objetivo de obtener 195 recompensas acumuladas en más de 100 ejecuciones consecutivas de la simulación, ¡o podríamos haberlo logrado! Incluso si obtenemos números más pequeños, aún no lo sabemos, porque promediamos sobre 5000 ejecuciones, y solo se requieren 100 ejecuciones en el criterio formal.

- **La recompensa comienza a disminuir**. A veces la recompensa comienza a disminuir, lo que significa que podemos "destruir" valores ya aprendidos en la Q-Table con los que empeoran la situación.

Esta observación es más claramente visible si graficamos el progreso del entrenamiento.

## Graficando el progreso del entrenamiento

Durante el entrenamiento, hemos recopilado el valor de recompensa acumulada en cada una de las iteraciones en el vector `rewards`. Así es como se ve cuando lo graficamos contra el número de iteración:

```python
plt.plot(rewards)
```

![progreso bruto](../../../../translated_images/train_progress_raw.2adfdf2daea09c596fc786fa347a23e9aceffe1b463e2257d20a9505794823ec.es.png)

De este gráfico, no es posible decir nada, porque debido a la naturaleza del proceso de entrenamiento estocástico, la duración de las sesiones de entrenamiento varía mucho. Para darle más sentido a este gráfico, podemos calcular el **promedio móvil** sobre una serie de experimentos, digamos 100. Esto se puede hacer convenientemente usando `np.convolve`: (bloque de código 12)

```python
def running_average(x,window):
    return np.convolve(x,np.ones(window)/window,mode='valid')

plt.plot(running_average(rewards,100))
```

![progreso del entrenamiento](../../../../translated_images/train_progress_runav.c71694a8fa9ab35935aff6f109e5ecdfdbdf1b0ae265da49479a81b5fae8f0aa.es.png)

## Variando los hiperparámetros

Para hacer el aprendizaje más estable, tiene sentido ajustar algunos de nuestros hiperparámetros durante el entrenamiento. En particular:

- **Para la tasa de aprendizaje**, `alpha`, we may start with values close to 1, and then keep decreasing the parameter. With time, we will be getting good probability values in the Q-Table, and thus we should be adjusting them slightly, and not overwriting completely with new values.

- **Increase epsilon**. We may want to increase the `epsilon` slowly, in order to explore less and exploit more. It probably makes sense to start with lower value of `epsilon`, y subir hasta casi 1.

> **Tarea 1**: Juega con los valores de los hiperparámetros y ve si puedes lograr una mayor recompensa acumulada. ¿Estás obteniendo más de 195?

> **Tarea 2**: Para resolver formalmente el problema, necesitas obtener una recompensa promedio de 195 en 100 ejecuciones consecutivas. Mide eso durante el entrenamiento y asegúrate de haber resuelto formalmente el problema.

## Viendo el resultado en acción

Sería interesante ver cómo se comporta el modelo entrenado. Ejecutemos la simulación y sigamos la misma estrategia de selección de acciones que durante el entrenamiento, muestreando según la distribución de probabilidad en la Q-Table: (bloque de código 13)

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

![un CartPole equilibrado](../../../../8-Reinforcement/2-Gym/images/cartpole-balance.gif)

---

## 🚀Desafío

> **Tarea 3**: Aquí, estábamos usando la copia final de la Q-Table, que puede no ser la mejor. Recuerda que hemos almacenado la Q-Table con mejor rendimiento en `Qbest` variable! Try the same example with the best-performing Q-Table by copying `Qbest` over to `Q` and see if you notice the difference.

> **Task 4**: Here we were not selecting the best action on each step, but rather sampling with corresponding probability distribution. Would it make more sense to always select the best action, with the highest Q-Table value? This can be done by using `np.argmax` para encontrar el número de acción correspondiente al valor más alto de la Q-Table. Implementa esta estrategia y ve si mejora el equilibrio.

## [Cuestionario posterior a la lección](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/48/)

## Tarea
[Entrena un Mountain Car](assignment.md)

## Conclusión

Ahora hemos aprendido cómo entrenar agentes para lograr buenos resultados simplemente proporcionándoles una función de recompensa que define el estado deseado del juego, y dándoles la oportunidad de explorar inteligentemente el espacio de búsqueda. Hemos aplicado con éxito el algoritmo de Q-Learning en los casos de entornos discretos y continuos, pero con acciones discretas.

Es importante también estudiar situaciones donde el estado de la acción también es continuo, y cuando el espacio de observación es mucho más complejo, como la imagen de la pantalla del juego de Atari. En esos problemas a menudo necesitamos usar técnicas de aprendizaje automático más poderosas, como redes neuronales, para lograr buenos resultados. Esos temas más avanzados son el tema de nuestro próximo curso más avanzado de IA.

**Descargo de responsabilidad**:
Este documento ha sido traducido utilizando servicios de traducción automática basados en inteligencia artificial. Si bien nos esforzamos por lograr precisión, tenga en cuenta que las traducciones automáticas pueden contener errores o imprecisiones. El documento original en su idioma nativo debe considerarse la fuente autorizada. Para información crítica, se recomienda la traducción humana profesional. No somos responsables de ningún malentendido o interpretación errónea que surja del uso de esta traducción.