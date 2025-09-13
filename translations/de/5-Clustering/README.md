<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "b28a3a4911584062772c537b653ebbc7",
  "translation_date": "2025-09-03T21:45:03+00:00",
  "source_file": "5-Clustering/README.md",
  "language_code": "de"
}
-->
# Clustering-Modelle für maschinelles Lernen

Clustering ist eine Aufgabe des maschinellen Lernens, bei der versucht wird, Objekte zu finden, die einander ähneln, und diese in Gruppen, sogenannte Cluster, zu unterteilen. Was Clustering von anderen Ansätzen im maschinellen Lernen unterscheidet, ist, dass alles automatisch geschieht. Tatsächlich kann man sagen, dass es das Gegenteil von überwachten Lernmethoden ist.

## Regionales Thema: Clustering-Modelle für den Musikgeschmack eines nigerianischen Publikums 🎧

Das vielfältige Publikum in Nigeria hat ebenso vielfältige musikalische Vorlieben. Mithilfe von Daten, die von Spotify gesammelt wurden (inspiriert von [diesem Artikel](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421)), werfen wir einen Blick auf einige beliebte Musikstücke in Nigeria. Dieses Datenset enthält Informationen über verschiedene Songs, wie deren 'Danceability'-Score, 'Acousticness', Lautstärke, 'Speechiness', Popularität und Energie. Es wird spannend sein, Muster in diesen Daten zu entdecken!

![Ein Plattenspieler](../../../translated_images/turntable.f2b86b13c53302dc106aa741de9dc96ac372864cf458dd6f879119857aab01da.de.jpg)

> Foto von <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Marcela Laskoski</a> auf <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
In dieser Reihe von Lektionen wirst du neue Wege entdecken, Daten mithilfe von Clustering-Techniken zu analysieren. Clustering ist besonders nützlich, wenn dein Datensatz keine Labels enthält. Falls Labels vorhanden sind, könnten Klassifikationstechniken, wie die, die du in früheren Lektionen gelernt hast, hilfreicher sein. Aber in Fällen, in denen du unbeschriftete Daten gruppieren möchtest, ist Clustering eine großartige Methode, um Muster zu erkennen.

> Es gibt nützliche Low-Code-Tools, die dir helfen können, mit Clustering-Modellen zu arbeiten. Probiere [Azure ML für diese Aufgabe](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott) aus.

## Lektionen

1. [Einführung in Clustering](1-Visualize/README.md)
2. [K-Means Clustering](2-K-Means/README.md)

## Credits

Diese Lektionen wurden mit 🎶 von [Jen Looper](https://www.twitter.com/jenlooper) geschrieben, mit hilfreichen Reviews von [Rishit Dagli](https://rishit_dagli) und [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan).

Das [Nigerian Songs](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify)-Datenset wurde von Kaggle bezogen und von Spotify gesammelt.

Nützliche K-Means-Beispiele, die bei der Erstellung dieser Lektion geholfen haben, umfassen diese [Iris-Analyse](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering), dieses [einführende Notebook](https://www.kaggle.com/prashant111/k-means-clustering-with-python) und dieses [hypothetische NGO-Beispiel](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering).

---

**Haftungsausschluss**:  
Dieses Dokument wurde mit dem KI-Übersetzungsdienst [Co-op Translator](https://github.com/Azure/co-op-translator) übersetzt. Obwohl wir uns um Genauigkeit bemühen, beachten Sie bitte, dass automatisierte Übersetzungen Fehler oder Ungenauigkeiten enthalten können. Das Originaldokument in seiner ursprünglichen Sprache sollte als maßgebliche Quelle betrachtet werden. Für kritische Informationen wird eine professionelle menschliche Übersetzung empfohlen. Wir übernehmen keine Haftung für Missverständnisse oder Fehlinterpretationen, die sich aus der Nutzung dieser Übersetzung ergeben.