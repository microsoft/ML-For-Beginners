<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "b28a3a4911584062772c537b653ebbc7",
  "translation_date": "2025-09-05T15:39:21+00:00",
  "source_file": "5-Clustering/README.md",
  "language_code": "hu"
}
-->
# Gépi tanulási klaszterezési modellek

A klaszterezés egy gépi tanulási feladat, amelynek célja, hogy megtalálja az egymáshoz hasonló objektumokat, és ezeket csoportokba, úgynevezett klaszterekbe rendezze. Ami megkülönbözteti a klaszterezést a gépi tanulás más megközelítéseitől, az az, hogy a folyamat automatikusan történik; valójában mondhatjuk, hogy ez az ellenkezője a felügyelt tanulásnak.

## Regionális téma: klaszterezési modellek a nigériai közönség zenei ízléséhez 🎧

Nigéria sokszínű közönsége sokféle zenei ízléssel rendelkezik. A Spotify-ról gyűjtött adatok felhasználásával (az [ebben a cikkben](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421) inspirálódva) nézzük meg néhány Nigériában népszerű zenét. Ez az adatállomány tartalmaz információkat különböző dalok "táncolhatósági" pontszámáról, "akusztikusságáról", hangosságáról, "beszédességéről", népszerűségéről és energiájáról. Érdekes lesz mintázatokat felfedezni ezekben az adatokban!

![Egy lemezjátszó](../../../5-Clustering/images/turntable.jpg)

> Fotó: <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Marcela Laskoski</a> az <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a> oldalán
  
Ebben a leckesorozatban új módszereket fedezhetsz fel az adatok elemzésére klaszterezési technikák segítségével. A klaszterezés különösen hasznos, ha az adatállományod nem tartalmaz címkéket. Ha vannak címkék, akkor az előző leckékben tanult osztályozási technikák hasznosabbak lehetnek. De ha címkézetlen adatokat szeretnél csoportosítani, a klaszterezés nagyszerű módja a mintázatok felfedezésének.

> Hasznos alacsony kódú eszközök állnak rendelkezésre, amelyek segítenek a klaszterezési modellekkel való munkában. Próbáld ki az [Azure ML-t erre a feladatra](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

## Leckék

1. [Bevezetés a klaszterezésbe](1-Visualize/README.md)
2. [K-Means klaszterezés](2-K-Means/README.md)

## Köszönetnyilvánítás

Ezeket a leckéket 🎶-vel írta [Jen Looper](https://www.twitter.com/jenlooper), hasznos véleményekkel [Rishit Dagli](https://rishit_dagli) és [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan) részéről.

A [Nigériai dalok](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) adatállományt a Kaggle-ről származtatták, a Spotify-ról gyűjtve.

Hasznos K-Means példák, amelyek segítettek a lecke elkészítésében: ez az [iris elemzés](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering), ez a [bevezető notebook](https://www.kaggle.com/prashant111/k-means-clustering-with-python), és ez a [hipotetikus NGO példa](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering).

---

**Felelősség kizárása**:  
Ez a dokumentum az AI fordítási szolgáltatás, a [Co-op Translator](https://github.com/Azure/co-op-translator) segítségével lett lefordítva. Bár törekszünk a pontosságra, kérjük, vegye figyelembe, hogy az automatikus fordítások hibákat vagy pontatlanságokat tartalmazhatnak. Az eredeti dokumentum az eredeti nyelvén tekintendő hiteles forrásnak. Kritikus információk esetén javasolt professzionális emberi fordítást igénybe venni. Nem vállalunk felelősséget semmilyen félreértésért vagy téves értelmezésért, amely a fordítás használatából eredhet.