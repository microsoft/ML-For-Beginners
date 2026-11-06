# Modèles de clustering pour l'apprentissage automatique

Le clustering est une tâche d'apprentissage automatique où l'on cherche à trouver des objets qui se ressemblent et à les regrouper en groupes appelés clusters. Ce qui différencie le clustering des autres approches en apprentissage automatique, c'est que tout se fait automatiquement, en fait, on peut dire que c'est l'opposé de l'apprentissage supervisé.

## Sujet régional : modèles de clustering pour les goûts musicaux du public nigérian 🎧

Le public diversifié du Nigeria a des goûts musicaux variés. En utilisant des données extraites de Spotify (inspirées par [cet article](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421), examinons quelques musiques populaires au Nigeria. Ce jeu de données inclut des données sur le score de 'danceability', 'acousticness', la loudness, la 'speechiness', la popularité et l'énergie de différentes chansons. Il sera intéressant de découvrir des motifs dans ces données !

![Une platine](../../../translated_images/fr/turntable.f2b86b13c53302dc.webp)

> Photo par <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Marcela Laskoski</a> sur <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
Dans cette série de leçons, vous découvrirez de nouvelles façons d'analyser les données en utilisant des techniques de clustering. Le clustering est particulièrement utile lorsque votre jeu de données ne comporte pas d'étiquettes. S'il en comporte, alors les techniques de classification que vous avez apprises dans les leçons précédentes pourraient être plus utiles. Mais dans les cas où vous cherchez à regrouper des données non étiquetées, le clustering est un excellent moyen de découvrir des motifs.

> Il existe des outils low-code utiles qui peuvent vous aider à apprendre à travailler avec des modèles de clustering. Essayez [Azure ML pour cette tâche](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

## Leçons

1. [Introduction au clustering](1-Visualize/README.md)
2. [Clustering K-Means](2-K-Means/README.md)

## Crédits

Ces leçons ont été écrites avec 🎶 par [Jen Looper](https://www.twitter.com/jenlooper) avec des relectures utiles de [Rishit Dagli](https://rishit_dagli/) et [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan).

Le jeu de données [Nigerian Songs](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) provient de Kaggle, extrait de Spotify.

Des exemples utiles de K-Means ayant aidé à la création de cette leçon incluent cette [exploration des iris](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering), ce [carnet d’introduction](https://www.kaggle.com/prashant111/k-means-clustering-with-python), et cet [exemple hypothétique d’ONG](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering).

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Avertissement** :
Ce document a été traduit à l'aide du service de traduction automatique [Co-op Translator](https://github.com/Azure/co-op-translator). Bien que nous nous efforçions d'assurer l'exactitude, veuillez noter que les traductions automatisées peuvent contenir des erreurs ou des inexactitudes. Le document original dans sa langue native doit être considéré comme la source faisant autorité. Pour les informations critiques, il est recommandé de recourir à une traduction professionnelle réalisée par un humain. Nous ne saurions être tenus responsables des malentendus ou erreurs d'interprétation découlant de l'utilisation de cette traduction.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->