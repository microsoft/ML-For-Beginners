<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "b28a3a4911584062772c537b653ebbc7",
  "translation_date": "2025-09-03T22:56:08+00:00",
  "source_file": "5-Clustering/README.md",
  "language_code": "es"
}
-->
# Modelos de agrupamiento para aprendizaje automático

El agrupamiento es una tarea de aprendizaje automático que busca encontrar objetos que se asemejen entre sí y agruparlos en grupos llamados clústeres. Lo que diferencia el agrupamiento de otros enfoques en el aprendizaje automático es que todo sucede automáticamente; de hecho, es justo decir que es lo opuesto al aprendizaje supervisado.

## Tema regional: modelos de agrupamiento para los gustos musicales de una audiencia nigeriana 🎧

La diversa audiencia de Nigeria tiene gustos musicales variados. Usando datos extraídos de Spotify (inspirados en [este artículo](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421)), analicemos algo de la música popular en Nigeria. Este conjunto de datos incluye información sobre el puntaje de 'bailabilidad', 'acústica', volumen, 'hablabilidad', popularidad y energía de varias canciones. ¡Será interesante descubrir patrones en estos datos!

![Un tocadiscos](../../../translated_images/turntable.f2b86b13c53302dc106aa741de9dc96ac372864cf458dd6f879119857aab01da.es.jpg)

> Foto de <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Marcela Laskoski</a> en <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
En esta serie de lecciones, descubrirás nuevas formas de analizar datos utilizando técnicas de agrupamiento. El agrupamiento es particularmente útil cuando tu conjunto de datos carece de etiquetas. Si tiene etiquetas, entonces las técnicas de clasificación como las que aprendiste en lecciones anteriores podrían ser más útiles. Pero en casos donde buscas agrupar datos sin etiquetar, el agrupamiento es una excelente manera de descubrir patrones.

> Hay herramientas útiles de bajo código que pueden ayudarte a aprender a trabajar con modelos de agrupamiento. Prueba [Azure ML para esta tarea](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

## Lecciones

1. [Introducción al agrupamiento](1-Visualize/README.md)
2. [Agrupamiento K-Means](2-K-Means/README.md)

## Créditos

Estas lecciones fueron escritas con 🎶 por [Jen Looper](https://www.twitter.com/jenlooper) con revisiones útiles de [Rishit Dagli](https://rishit_dagli) y [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan).

El conjunto de datos [Nigerian Songs](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) fue obtenido de Kaggle como datos extraídos de Spotify.

Ejemplos útiles de K-Means que ayudaron a crear esta lección incluyen esta [exploración de iris](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering), este [notebook introductorio](https://www.kaggle.com/prashant111/k-means-clustering-with-python), y este [ejemplo hipotético de una ONG](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering).

---

**Descargo de responsabilidad**:  
Este documento ha sido traducido utilizando el servicio de traducción automática [Co-op Translator](https://github.com/Azure/co-op-translator). Si bien nos esforzamos por garantizar la precisión, tenga en cuenta que las traducciones automáticas pueden contener errores o imprecisiones. El documento original en su idioma nativo debe considerarse como la fuente autorizada. Para información crítica, se recomienda una traducción profesional realizada por humanos. No nos hacemos responsables de malentendidos o interpretaciones erróneas que puedan surgir del uso de esta traducción.