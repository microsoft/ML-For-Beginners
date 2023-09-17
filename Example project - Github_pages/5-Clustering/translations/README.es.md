# Modelos de agrupamiento en el aprendizaje automático

El agrupamiento es una tarea del aprendizaje automático en donde se buscan objetos que se parezcan a otros y estos se agrupan en elementos llamados clusters. Lo que hace diferente al clustering de otros enfoques en el aprendizaje automático, es que las cosas suceden de forma automática, de hecho, es justo decir que es lo opuesto al aprendizaje supervisado.

## Tema regional: modelos de clustering para los gustos musicales de la audiencia Nigeriana 🎧

La audiencia Nigeriana tiene diversos gustos musicales. Usando datos extraídos de Spotify (inspirados por [este artículo](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421), daremos un vistazo a alguna música popular en Nigeria. Este conjunto de datos incluye datos acerca de varias canciones de acuerdo a la puntuación por 'bailables', 'acústica', 'volumen', 'habla', popularidad y energía. ¡Será interesante descubrir patrones en estos datos!

![Un tornamesa](../images/turntable.jpg)

> Foto por <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Marcela Laskoski</a> en <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
En esta serie de lecciones, descubrirás nuevas formas de analizar datos usando técnicas de clustering. El clustering es particularmente útil cuando tu conjunto de datos carece de etiquetas. Si este sí tiene etiquetas, entonces las técnicas de clasificación como las que has aprendido en lecciones previas son más útiles. Pero en casos donde pretendes agrupar datos sin etiquetas, el clustering es una gran forma de descubrir patrones.


> Existen herramientas low-code útiles que te pueden ayudar a trabajar con modelos de clustering. Prueba [Azure ML para esta tarea](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

## Lecciones

1. [Introducción al clustering](1-Visualize/README.md)
2. [K-Means clustering](2-K-Means/README.md)

## Créditos

Éstas lecciones fueron escritas con 🎶 por [Jen Looper](https://www.twitter.com/jenlooper) con la útil ayuda de [Rishit Dagli](https://rishit_dagli) y [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan).

El conjunto de datos de las [Canciones Nigerianas](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) fue obtenido de Kaggle como raspado de datos de Spotify.

Algunos ejemplos útiles de K-means que ayudaron en la creación de esta lección son [exploración de iris](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering), este [notebook introductorio](https://www.kaggle.com/prashant111/k-means-clustering-with-python), y este [ejemplo hipotético de ONG](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering).
