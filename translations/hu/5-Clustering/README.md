# Klasszterező modellek gépi tanuláshoz

A klaszterezés egy gépi tanulási feladat, ahol olyan objektumokat keresünk, amelyek hasonlítanak egymásra, és ezeket csoportokba, úgynevezett klaszterekbe rendezzük. Ami megkülönbözteti a klaszterezést a gépi tanulás más megközelítéseitől, hogy a folyamat automatikusan zajlik, valójában azt mondhatjuk, hogy ez a felügyelt tanulás ellentéte.

## Regionális téma: klaszterező modellek a nigériai közönség zenei ízléséhez 🎧

Nigéria sokszínű közönsége sokszínű zenei ízléssel rendelkezik. A Spotify-ról összegyűjtött adatok (inspirációként [ez a cikk](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421)) alapján nézzünk meg néhány, Nigériában népszerű zenét. Ez az adathalmaz tartalmaz adatokat különböző dalok 'táncolhatóság' pontszámáról, 'akusztikusságáról', hangerőről, 'beszédességéről', népszerűségéről és energiájáról. Érdekes lesz felfedezni mintákat ezekben az adatokban!

![Lemezjátszó](../../../translated_images/hu/turntable.f2b86b13c53302dc.webp)

> Fotó: <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Marcela Laskoski</a> az <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>-ról
  
Ebben a leckesorozatban új módszereket fedezhetsz fel az adatelemzésre klaszterezési technikák segítségével. A klaszterezés különösen hasznos akkor, ha az adathalmazban nincsenek címkék. Ha vannak címkék, akkor a korábban tanult osztályozási technikák lehetnek hasznosabbak. De ha címkézetlen adatok csoportosítására törekszel, a klaszterezés nagyszerű mód a mintázatok felfedezésére.

> Hasznos alacsony kódú eszközök is vannak, amelyek segíthetnek megismerni a klaszterező modellek használatát. Próbáld ki [az Azure ML-t erre a feladatra](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

## Leckék

1. [Bevezetés a klaszterezésbe](1-Visualize/README.md)
2. [K-Means klaszterezés](2-K-Means/README.md)

## Köszönetnyilvánítás

Ezeket a leckéket 🎶 írta [Jen Looper](https://www.twitter.com/jenlooper), hasznos véleményezéssel [Rishit Dagli](https://rishit_dagli/) és [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan) közreműködésével.

A [Nigerian Songs](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) adathalmaz a Kaggle-ról származik, Spotify-ról összegyűjtve.

Hasznos K-Means példák, amelyek segítettek ennek a leckének a létrehozásában: ez az [iriszek feltérképezése](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering), ez a [bevezető jegyzetfüzet](https://www.kaggle.com/prashant111/k-means-clustering-with-python), és ez a [hipotetikus NGO példa](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering).

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Jogi nyilatkozat**:
Ez a dokumentum az AI fordítási szolgáltatás, a [Co-op Translator](https://github.com/Azure/co-op-translator) segítségével készült. Bár az pontosságra törekszünk, kérjük, vegye figyelembe, hogy az automatikus fordítások hibákat vagy pontatlanságokat tartalmazhatnak. Az eredeti dokumentum az anyanyelvén tekintendő hiteles forrásnak. Fontos információk esetén professzionális emberi fordítást javasolunk. Nem vállalunk felelősséget semmilyen félreértésért vagy téves értelmezésért, amely ebből a fordításból ered.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->