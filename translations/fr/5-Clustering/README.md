<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "b28a3a4911584062772c537b653ebbc7",
  "translation_date": "2025-09-03T22:55:31+00:00",
  "source_file": "5-Clustering/README.md",
  "language_code": "fr"
}
-->
# Modèles de clustering pour l'apprentissage automatique

Le clustering est une tâche d'apprentissage automatique qui cherche à identifier des objets similaires et à les regrouper dans des groupes appelés clusters. Ce qui distingue le clustering des autres approches en apprentissage automatique, c'est que tout se fait automatiquement. En fait, on peut dire que c'est l'opposé de l'apprentissage supervisé.

## Sujet régional : modèles de clustering pour les goûts musicaux d'un public nigérian 🎧

Le public diversifié du Nigeria a des goûts musicaux variés. En utilisant des données extraites de Spotify (inspiré par [cet article](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421)), examinons certaines musiques populaires au Nigeria. Ce jeu de données inclut des informations sur le score de 'danseabilité', l'acoustique, le volume sonore, le caractère 'parlé', la popularité et l'énergie de diverses chansons. Il sera intéressant de découvrir des motifs dans ces données !

![Une platine vinyle](../../../translated_images/turntable.f2b86b13c53302dc106aa741de9dc96ac372864cf458dd6f879119857aab01da.fr.jpg)

> Photo par <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Marcela Laskoski</a> sur <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
Dans cette série de leçons, vous découvrirez de nouvelles façons d'analyser des données en utilisant des techniques de clustering. Le clustering est particulièrement utile lorsque votre jeu de données ne contient pas de labels. S'il contient des labels, alors des techniques de classification comme celles que vous avez apprises dans les leçons précédentes pourraient être plus adaptées. Mais dans les cas où vous cherchez à regrouper des données non étiquetées, le clustering est un excellent moyen de découvrir des motifs.

> Il existe des outils low-code utiles qui peuvent vous aider à travailler avec des modèles de clustering. Essayez [Azure ML pour cette tâche](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

## Leçons

1. [Introduction au clustering](1-Visualize/README.md)
2. [Clustering avec K-Means](2-K-Means/README.md)

## Crédits

Ces leçons ont été écrites avec 🎶 par [Jen Looper](https://www.twitter.com/jenlooper) avec des avis utiles de [Rishit Dagli](https://rishit_dagli) et [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan).

Le jeu de données [Nigerian Songs](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) a été obtenu sur Kaggle à partir de données extraites de Spotify.

Des exemples utiles de K-Means qui ont aidé à créer cette leçon incluent cette [exploration des iris](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering), ce [notebook introductif](https://www.kaggle.com/prashant111/k-means-clustering-with-python), et cet [exemple hypothétique d'ONG](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering).

---

**Avertissement** :  
Ce document a été traduit à l'aide du service de traduction automatique [Co-op Translator](https://github.com/Azure/co-op-translator). Bien que nous nous efforcions d'assurer l'exactitude, veuillez noter que les traductions automatisées peuvent contenir des erreurs ou des inexactitudes. Le document original dans sa langue d'origine doit être considéré comme la source faisant autorité. Pour des informations critiques, il est recommandé de faire appel à une traduction humaine professionnelle. Nous déclinons toute responsabilité en cas de malentendus ou d'interprétations erronées résultant de l'utilisation de cette traduction.