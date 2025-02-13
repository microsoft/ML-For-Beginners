# Modèles de clustering pour l'apprentissage automatique

Le clustering est une tâche d'apprentissage automatique qui vise à trouver des objets semblables et à les regrouper en ensembles appelés clusters. Ce qui distingue le clustering des autres approches en apprentissage automatique, c'est que les choses se passent automatiquement ; en fait, on peut dire que c'est l'opposé de l'apprentissage supervisé.

## Sujet régional : modèles de clustering pour les goûts musicaux d'un public nigérian 🎧

Le public diversifié du Nigéria a des goûts musicaux variés. En utilisant des données extraites de Spotify (inspirées par [cet article](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421)), examinons quelques musiques populaires au Nigéria. Cet ensemble de données inclut des informations sur le score de 'dansabilité' de diverses chansons, l' 'acoustique', le volume, la 'parole', la popularité et l'énergie. Il sera intéressant de découvrir des motifs dans ces données !

![Un tourne-disque](../../../translated_images/turntable.f2b86b13c53302dc106aa741de9dc96ac372864cf458dd6f879119857aab01da.fr.jpg)

> Photo par <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Marcela Laskoski</a> sur <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>

Dans cette série de leçons, vous découvrirez de nouvelles façons d'analyser des données en utilisant des techniques de clustering. Le clustering est particulièrement utile lorsque votre ensemble de données manque d'étiquettes. S'il a des étiquettes, alors des techniques de classification, comme celles que vous avez apprises dans les leçons précédentes, pourraient être plus utiles. Mais dans les cas où vous cherchez à regrouper des données non étiquetées, le clustering est un excellent moyen de découvrir des motifs.

> Il existe des outils low-code utiles qui peuvent vous aider à apprendre à travailler avec des modèles de clustering. Essayez [Azure ML pour cette tâche](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

## Leçons

1. [Introduction au clustering](1-Visualize/README.md)
2. [Clustering K-Means](2-K-Means/README.md)

## Crédits

Ces leçons ont été écrites avec 🎶 par [Jen Looper](https://www.twitter.com/jenlooper) avec des revues utiles de [Rishit Dagli](https://rishit_dagli) et [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan).

L'ensemble de données [Nigerian Songs](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) a été obtenu sur Kaggle en étant extrait de Spotify.

Des exemples utiles de K-Means qui ont aidé à créer cette leçon incluent cette [exploration de l'iris](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering), ce [carnet d'introduction](https://www.kaggle.com/prashant111/k-means-clustering-with-python), et cet [exemple d'ONG hypothétique](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering).

**Avertissement** :  
Ce document a été traduit à l'aide de services de traduction automatisée basés sur l'IA. Bien que nous nous efforçons d'assurer l'exactitude, veuillez noter que les traductions automatisées peuvent contenir des erreurs ou des inexactitudes. Le document original dans sa langue native doit être considéré comme la source autoritaire. Pour des informations critiques, une traduction humaine professionnelle est recommandée. Nous ne sommes pas responsables des malentendus ou des interprétations erronées résultant de l'utilisation de cette traduction.