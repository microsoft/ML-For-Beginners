# Introducción al aprendizaje por refuerzo

El aprendizaje por refuerzo, RL, es visto como uno de los paradigmas básicos del aprendizaje automático, junto con el aprendizaje supervisado y el aprendizaje no supervisado. RL trata sobre decisiones: tomar las decisiones correctas o al menos aprender de ellas.

Imagina que tienes un entorno simulado como el mercado de valores. ¿Qué pasa si impones una regulación determinada? ¿Tiene un efecto positivo o negativo? Si ocurre algo negativo, necesitas tomar este _refuerzo negativo_, aprender de él y cambiar de rumbo. Si es un resultado positivo, necesitas construir sobre ese _refuerzo positivo_.

![Pedro y el lobo](../../../translated_images/peter.779730f9ba3a8a8d9290600dcf55f2e491c0640c785af7ac0d64f583c49b8864.es.png)

> ¡Pedro y sus amigos necesitan escapar del lobo hambriento! Imagen por [Jen Looper](https://twitter.com/jenlooper)

## Tema regional: Pedro y el lobo (Rusia)

[Pedro y el lobo](https://es.wikipedia.org/wiki/Pedro_y_el_lobo) es un cuento musical escrito por el compositor ruso [Sergei Prokofiev](https://es.wikipedia.org/wiki/Serguéi_Prokófiev). Es una historia sobre el joven pionero Pedro, que valientemente sale de su casa hacia el claro del bosque para perseguir al lobo. En esta sección, entrenaremos algoritmos de aprendizaje automático que ayudarán a Pedro:

- **Explorar** el área circundante y construir un mapa de navegación óptimo.
- **Aprender** a usar una patineta y equilibrarse en ella, para moverse más rápido.

[![Pedro y el lobo](https://img.youtube.com/vi/Fmi5zHg4QSM/0.jpg)](https://www.youtube.com/watch?v=Fmi5zHg4QSM)

> 🎥 Haz clic en la imagen de arriba para escuchar Pedro y el lobo por Prokofiev

## Aprendizaje por refuerzo

En secciones anteriores, has visto dos ejemplos de problemas de aprendizaje automático:

- **Supervisado**, donde tenemos conjuntos de datos que sugieren soluciones de muestra al problema que queremos resolver. [Clasificación](../4-Classification/README.md) y [regresión](../2-Regression/README.md) son tareas de aprendizaje supervisado.
- **No supervisado**, en el cual no tenemos datos de entrenamiento etiquetados. El principal ejemplo de aprendizaje no supervisado es [Agrupamiento](../5-Clustering/README.md).

En esta sección, te presentaremos un nuevo tipo de problema de aprendizaje que no requiere datos de entrenamiento etiquetados. Hay varios tipos de estos problemas:

- **[Aprendizaje semi-supervisado](https://es.wikipedia.org/wiki/Aprendizaje_semi-supervisado)**, donde tenemos muchos datos no etiquetados que pueden usarse para pre-entrenar el modelo.
- **[Aprendizaje por refuerzo](https://es.wikipedia.org/wiki/Aprendizaje_por_refuerzo)**, en el cual un agente aprende cómo comportarse realizando experimentos en algún entorno simulado.

### Ejemplo - juego de computadora

Supongamos que quieres enseñar a una computadora a jugar un juego, como el ajedrez, o [Super Mario](https://es.wikipedia.org/wiki/Super_Mario). Para que la computadora juegue un juego, necesitamos que prediga qué movimiento hacer en cada uno de los estados del juego. Aunque esto pueda parecer un problema de clasificación, no lo es, porque no tenemos un conjunto de datos con estados y acciones correspondientes. Aunque podamos tener algunos datos como partidas de ajedrez existentes o grabaciones de jugadores jugando Super Mario, es probable que esos datos no cubran suficientemente un número grande de estados posibles.

En lugar de buscar datos de juego existentes, el **Aprendizaje por Refuerzo** (RL) se basa en la idea de *hacer que la computadora juegue* muchas veces y observar el resultado. Así, para aplicar el Aprendizaje por Refuerzo, necesitamos dos cosas:

- **Un entorno** y **un simulador** que nos permitan jugar un juego muchas veces. Este simulador definiría todas las reglas del juego, así como los posibles estados y acciones.

- **Una función de recompensa**, que nos diría qué tan bien lo hicimos durante cada movimiento o juego.

La principal diferencia entre otros tipos de aprendizaje automático y RL es que en RL típicamente no sabemos si ganamos o perdemos hasta que terminamos el juego. Por lo tanto, no podemos decir si un cierto movimiento solo es bueno o no - solo recibimos una recompensa al final del juego. Y nuestro objetivo es diseñar algoritmos que nos permitan entrenar un modelo bajo condiciones inciertas. Aprenderemos sobre un algoritmo de RL llamado **Q-learning**.

## Lecciones

1. [Introducción al aprendizaje por refuerzo y Q-Learning](1-QLearning/README.md)
2. [Uso de un entorno de simulación de gimnasio](2-Gym/README.md)

## Créditos

"La Introducción al Aprendizaje por Refuerzo" fue escrita con ♥️ por [Dmitry Soshnikov](http://soshnikov.com)

**Descargo de responsabilidad**:
Este documento ha sido traducido utilizando servicios de traducción automática basados en inteligencia artificial. Aunque nos esforzamos por lograr precisión, tenga en cuenta que las traducciones automatizadas pueden contener errores o inexactitudes. El documento original en su idioma nativo debe considerarse la fuente autorizada. Para información crítica, se recomienda una traducción profesional realizada por humanos. No nos hacemos responsables de ningún malentendido o interpretación errónea que surja del uso de esta traducción.