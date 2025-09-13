<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "20ca019012b1725de956681d036d8b18",
  "translation_date": "2025-09-04T00:14:18+00:00",
  "source_file": "8-Reinforcement/README.md",
  "language_code": "es"
}
-->
# Introducción al aprendizaje por refuerzo

El aprendizaje por refuerzo, RL, se considera uno de los paradigmas básicos del aprendizaje automático, junto con el aprendizaje supervisado y el aprendizaje no supervisado. RL trata sobre decisiones: tomar las decisiones correctas o, al menos, aprender de ellas.

Imagina que tienes un entorno simulado como el mercado de valores. ¿Qué sucede si impones una regulación específica? ¿Tiene un efecto positivo o negativo? Si ocurre algo negativo, necesitas tomar este _refuerzo negativo_, aprender de ello y cambiar de rumbo. Si el resultado es positivo, necesitas construir sobre ese _refuerzo positivo_.

![Pedro y el lobo](../../../translated_images/peter.779730f9ba3a8a8d9290600dcf55f2e491c0640c785af7ac0d64f583c49b8864.es.png)

> ¡Pedro y sus amigos necesitan escapar del lobo hambriento! Imagen por [Jen Looper](https://twitter.com/jenlooper)

## Tema regional: Pedro y el Lobo (Rusia)

[Pedro y el Lobo](https://es.wikipedia.org/wiki/Pedro_y_el_lobo) es un cuento musical escrito por el compositor ruso [Sergei Prokofiev](https://es.wikipedia.org/wiki/Sergu%C3%A9i_Prok%C3%B3fiev). Es una historia sobre el joven pionero Pedro, quien valientemente sale de su casa hacia el claro del bosque para perseguir al lobo. En esta sección, entrenaremos algoritmos de aprendizaje automático que ayudarán a Pedro:

- **Explorar** el área circundante y construir un mapa de navegación óptimo.
- **Aprender** a usar un monopatín y mantener el equilibrio en él, para moverse más rápido.

[![Pedro y el Lobo](https://img.youtube.com/vi/Fmi5zHg4QSM/0.jpg)](https://www.youtube.com/watch?v=Fmi5zHg4QSM)

> 🎥 Haz clic en la imagen de arriba para escuchar Pedro y el Lobo de Prokofiev

## Aprendizaje por refuerzo

En secciones anteriores, has visto dos ejemplos de problemas de aprendizaje automático:

- **Supervisado**, donde tenemos conjuntos de datos que sugieren soluciones de muestra para el problema que queremos resolver. [Clasificación](../4-Classification/README.md) y [regresión](../2-Regression/README.md) son tareas de aprendizaje supervisado.
- **No supervisado**, en el que no tenemos datos de entrenamiento etiquetados. El principal ejemplo de aprendizaje no supervisado es [Agrupamiento](../5-Clustering/README.md).

En esta sección, te presentaremos un nuevo tipo de problema de aprendizaje que no requiere datos de entrenamiento etiquetados. Hay varios tipos de problemas de este tipo:

- **[Aprendizaje semisupervisado](https://es.wikipedia.org/wiki/Aprendizaje_semisupervisado)**, donde tenemos una gran cantidad de datos no etiquetados que pueden usarse para preentrenar el modelo.
- **[Aprendizaje por refuerzo](https://es.wikipedia.org/wiki/Aprendizaje_por_refuerzo)**, en el que un agente aprende cómo comportarse realizando experimentos en algún entorno simulado.

### Ejemplo - videojuego

Supongamos que quieres enseñar a una computadora a jugar un juego, como ajedrez o [Super Mario](https://es.wikipedia.org/wiki/Super_Mario). Para que la computadora juegue, necesitamos que prediga qué movimiento realizar en cada estado del juego. Aunque esto pueda parecer un problema de clasificación, no lo es, porque no tenemos un conjunto de datos con estados y acciones correspondientes. Aunque podríamos tener algunos datos como partidas de ajedrez existentes o grabaciones de jugadores jugando Super Mario, es probable que esos datos no cubran suficientemente una gran cantidad de estados posibles.

En lugar de buscar datos existentes del juego, el **Aprendizaje por Refuerzo** (RL) se basa en la idea de *hacer que la computadora juegue* muchas veces y observar el resultado. Por lo tanto, para aplicar el Aprendizaje por Refuerzo, necesitamos dos cosas:

- **Un entorno** y **un simulador** que nos permitan jugar muchas veces. Este simulador definiría todas las reglas del juego, así como los posibles estados y acciones.

- **Una función de recompensa**, que nos indique qué tan bien lo hicimos durante cada movimiento o partida.

La principal diferencia entre otros tipos de aprendizaje automático y RL es que en RL típicamente no sabemos si ganamos o perdemos hasta que terminamos el juego. Por lo tanto, no podemos decir si un movimiento en particular es bueno o no: solo recibimos una recompensa al final del juego. Y nuestro objetivo es diseñar algoritmos que nos permitan entrenar un modelo bajo condiciones inciertas. Aprenderemos sobre un algoritmo de RL llamado **Q-learning**.

## Lecciones

1. [Introducción al aprendizaje por refuerzo y Q-Learning](1-QLearning/README.md)
2. [Uso de un entorno de simulación gym](2-Gym/README.md)

## Créditos

"Introducción al Aprendizaje por Refuerzo" fue escrito con ♥️ por [Dmitry Soshnikov](http://soshnikov.com)

---

**Descargo de responsabilidad**:  
Este documento ha sido traducido utilizando el servicio de traducción automática [Co-op Translator](https://github.com/Azure/co-op-translator). Si bien nos esforzamos por lograr precisión, tenga en cuenta que las traducciones automáticas pueden contener errores o imprecisiones. El documento original en su idioma nativo debe considerarse como la fuente autorizada. Para información crítica, se recomienda una traducción profesional realizada por humanos. No nos hacemos responsables de malentendidos o interpretaciones erróneas que puedan surgir del uso de esta traducción.