<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "730225ea274c9174fe688b21d421539d",
  "translation_date": "2025-09-05T15:41:05+00:00",
  "source_file": "5-Clustering/1-Visualize/README.md",
  "language_code": "hu"
}
-->
# Bevezetés a klaszterezéshez

A klaszterezés a [felügyelet nélküli tanulás](https://wikipedia.org/wiki/Unsupervised_learning) egyik típusa, amely feltételezi, hogy az adathalmaz címkézetlen, vagy hogy a bemenetek nincsenek előre meghatározott kimenetekhez társítva. Különböző algoritmusokat használ a címkézetlen adatok rendezésére, és csoportosításokat hoz létre az adatokban észlelt minták alapján.

[![No One Like You by PSquare](https://img.youtube.com/vi/ty2advRiWJM/0.jpg)](https://youtu.be/ty2advRiWJM "No One Like You by PSquare")

> 🎥 Kattints a fenti képre egy videóért. Miközben a klaszterezéssel kapcsolatos gépi tanulást tanulmányozod, élvezd néhány nigériai Dance Hall számot - ez egy nagyon népszerű dal 2014-ből a PSquare-től.

## [Előadás előtti kvíz](https://ff-quizzes.netlify.app/en/ml/)

### Bevezetés

A [klaszterezés](https://link.springer.com/referenceworkentry/10.1007%2F978-0-387-30164-8_124) nagyon hasznos az adatok feltárásában. Nézzük meg, hogy segíthet-e trendek és minták felfedezésében a nigériai közönség zenehallgatási szokásai kapcsán.

✅ Gondolkodj el egy percig a klaszterezés felhasználási lehetőségein. A való életben klaszterezés történik, amikor van egy halom mosnivaló, és szét kell válogatnod a családtagok ruháit 🧦👕👖🩲. Az adatkutatásban klaszterezés történik, amikor megpróbáljuk elemezni a felhasználó preferenciáit, vagy meghatározni egy címkézetlen adathalmaz jellemzőit. A klaszterezés bizonyos értelemben segít rendet teremteni a káoszban, mint például egy zoknis fiókban.

[![Introduction to ML](https://img.youtube.com/vi/esmzYhuFnds/0.jpg)](https://youtu.be/esmzYhuFnds "Introduction to Clustering")

> 🎥 Kattints a fenti képre egy videóért: MIT John Guttag bemutatja a klaszterezést

Egy szakmai környezetben a klaszterezést például piaci szegmentáció meghatározására lehet használni, például annak megállapítására, hogy mely korcsoportok vásárolnak milyen termékeket. Egy másik felhasználási terület lehet az anomáliák észlelése, például csalások felderítése egy hitelkártya-tranzakciókat tartalmazó adathalmazból. Vagy használhatod a klaszterezést daganatok azonosítására egy orvosi szkenekből álló adathalmazban.

✅ Gondolkodj el egy percig azon, hogy találkoztál-e már klaszterezéssel a való életben, például banki, e-kereskedelmi vagy üzleti környezetben.

> 🎓 Érdekes módon a klaszterelemzés az antropológia és pszichológia területén kezdődött az 1930-as években. El tudod képzelni, hogyan használhatták akkoriban?

Alternatívaként használhatod keresési eredmények csoportosítására is - például vásárlási linkek, képek vagy vélemények alapján. A klaszterezés hasznos, ha van egy nagy adathalmaz, amelyet csökkenteni szeretnél, és amelyen részletesebb elemzést szeretnél végezni, így a technika segíthet az adatok megértésében, mielőtt más modelleket építenél.

✅ Miután az adataid klaszterekbe szerveződtek, hozzárendelhetsz egy klaszterazonosítót, és ez a technika hasznos lehet az adathalmaz adatvédelmének megőrzésében; az adatpontokra a klaszterazonosítóval hivatkozhatsz, ahelyett, hogy azonosítható adatokat használnál. Tudsz más okokat is mondani, hogy miért hivatkoznál egy klaszterazonosítóra a klaszter más elemei helyett?

Mélyítsd el a klaszterezési technikák megértését ebben a [Learn modulban](https://docs.microsoft.com/learn/modules/train-evaluate-cluster-models?WT.mc_id=academic-77952-leestott)

## Klaszterezés kezdőknek

[A Scikit-learn számos módszert kínál](https://scikit-learn.org/stable/modules/clustering.html) a klaszterezés elvégzésére. Az, hogy melyiket választod, az esettől függ. A dokumentáció szerint minden módszernek megvannak a maga előnyei. Íme egy egyszerűsített táblázat a Scikit-learn által támogatott módszerekről és azok megfelelő felhasználási eseteiről:

| Módszer neve                 | Felhasználási eset                                                     |
| :--------------------------- | :--------------------------------------------------------------------- |
| K-Means                      | általános célú, induktív                                               |
| Affinity propagation         | sok, egyenetlen klaszterek, induktív                                   |
| Mean-shift                   | sok, egyenetlen klaszterek, induktív                                   |
| Spectral clustering          | kevés, egyenletes klaszterek, transzduktív                            |
| Ward hierarchical clustering | sok, korlátozott klaszterek, transzduktív                              |
| Agglomerative clustering     | sok, korlátozott, nem euklideszi távolságok, transzduktív              |
| DBSCAN                       | nem sík geometria, egyenetlen klaszterek, transzduktív                 |
| OPTICS                       | nem sík geometria, egyenetlen klaszterek változó sűrűséggel, transzduktív |
| Gaussian mixtures            | sík geometria, induktív                                                |
| BIRCH                        | nagy adathalmaz kiugró értékekkel, induktív                            |

> 🎓 Az, hogy hogyan hozunk létre klasztereket, nagyban függ attól, hogyan gyűjtjük össze az adatpontokat csoportokba. Nézzük meg néhány szakkifejezést:
>
> 🎓 ['Transzduktív' vs. 'induktív'](https://wikipedia.org/wiki/Transduction_(machine_learning))
> 
> A transzduktív következtetés megfigyelt tanulási esetekből származik, amelyek konkrét tesztesetekhez kapcsolódnak. Az induktív következtetés tanulási esetekből származik, amelyek általános szabályokat alkotnak, amelyeket csak ezután alkalmaznak a tesztesetekre. 
> 
> Példa: Képzeld el, hogy van egy adathalmazod, amely csak részben van címkézve. Néhány elem 'lemezek', néhány 'cd-k', és néhány üres. A feladatod az üres elemek címkézése. Ha induktív megközelítést választasz, egy modellt tanítasz 'lemezek' és 'cd-k' keresésére, és ezeket a címkéket alkalmazod a címkézetlen adatokra. Ez a megközelítés nehézségekbe ütközhet olyan dolgok osztályozásában, amelyek valójában 'kazetták'. A transzduktív megközelítés viszont hatékonyabban kezeli ezt az ismeretlen adatot, mivel hasonló elemeket csoportosít, majd címkét alkalmaz egy csoportra. Ebben az esetben a klaszterek lehetnek 'kerek zenei dolgok' és 'szögletes zenei dolgok'.
> 
> 🎓 ['Nem sík' vs. 'sík' geometria](https://datascience.stackexchange.com/questions/52260/terminology-flat-geometry-in-the-context-of-clustering)
> 
> Matematikai terminológiából származik, a nem sík vs. sík geometria az adatpontok közötti távolságok mérésére utal, akár 'sík' ([euklideszi](https://wikipedia.org/wiki/Euclidean_geometry)), akár 'nem sík' (nem euklideszi) geometriai módszerekkel. 
>
>'Sík' ebben az összefüggésben az euklideszi geometriára utal (amelynek részeit 'síkmértan' néven tanítják), míg a nem sík a nem euklideszi geometriára utal. Mi köze van a geometriának a gépi tanuláshoz? Nos, mivel mindkét terület matematikai alapokon nyugszik, szükség van egy közös módszerre az adatpontok közötti távolságok mérésére a klaszterekben, és ezt 'sík' vagy 'nem sík' módon lehet megtenni, az adatok természetétől függően. Az [euklideszi távolságokat](https://wikipedia.org/wiki/Euclidean_distance) két pont közötti vonalszakasz hosszával mérik. A [nem euklideszi távolságokat](https://wikipedia.org/wiki/Non-Euclidean_geometry) görbe mentén mérik. Ha az adataid, vizualizálva, nem síkban léteznek, akkor speciális algoritmusra lehet szükséged a kezelésükhöz.
>
![Sík vs Nem sík geometria Infografika](../../../../5-Clustering/1-Visualize/images/flat-nonflat.png)
> Infografika: [Dasani Madipalli](https://twitter.com/dasani_decoded)
> 
> 🎓 ['Távolságok'](https://web.stanford.edu/class/cs345a/slides/12-clustering.pdf)
> 
> A klasztereket a távolságmátrixuk határozza meg, például az adatpontok közötti távolságok. Ez a távolság többféleképpen mérhető. Az euklideszi klasztereket az adatpontok értékeinek átlaga határozza meg, és tartalmaznak egy 'centroidot' vagy középpontot. A távolságokat így a centroidtól való távolság alapján mérik. A nem euklideszi távolságok 'clustroidok'-ra utalnak, az adatpontra, amely a legközelebb van más pontokhoz. A clustroidokat különböző módon lehet meghatározni.
> 
> 🎓 ['Korlátozott'](https://wikipedia.org/wiki/Constrained_clustering)
> 
> A [korlátozott klaszterezés](https://web.cs.ucdavis.edu/~davidson/Publications/ICDMTutorial.pdf) 'félig felügyelt' tanulást vezet be ebbe a felügyelet nélküli módszerbe. Az adatpontok közötti kapcsolatokat 'nem kapcsolható' vagy 'kapcsolható' címkékkel jelölik, így bizonyos szabályokat kényszerítenek az adathalmazra.
>
>Példa: Ha egy algoritmus szabadon működik egy címkézetlen vagy félig címkézett adathalmazon, az általa létrehozott klaszterek gyenge minőségűek lehetnek. A fenti példában a klaszterek lehetnek 'kerek zenei dolgok', 'szögletes zenei dolgok', 'háromszög alakú dolgok' és 'sütik'. Ha néhány korlátozást vagy szabályt adunk meg ("az elemnek műanyagból kell készülnie", "az elemnek zenét kell tudnia produkálni"), ez segíthet az algoritmusnak jobb döntéseket hozni.
> 
> 🎓 'Sűrűség'
> 
> Az 'zajos' adatokat 'sűrűnek' tekintik. Az egyes klaszterekben lévő pontok közötti távolságok vizsgálatakor kiderülhet, hogy ezek a távolságok többé-kevésbé sűrűek, vagy 'zsúfoltak', és így az ilyen adatokat megfelelő klaszterezési módszerrel kell elemezni. [Ez a cikk](https://www.kdnuggets.com/2020/02/understanding-density-based-clustering.html) bemutatja a különbséget a K-Means klaszterezés és a HDBSCAN algoritmusok használata között egy zajos adathalmaz egyenetlen klasztersűrűségének feltárására.

## Klaszterezési algoritmusok

Több mint 100 klaszterezési algoritmus létezik, és használatuk az adott adatok természetétől függ. Nézzük meg néhány főbb típust:

- **Hierarchikus klaszterezés**. Ha egy objektumot a közeli objektumhoz való közelsége alapján osztályoznak, nem pedig egy távolabbihoz, akkor a klaszterek az objektumok egymáshoz való távolsága alapján alakulnak ki. A Scikit-learn agglomeratív klaszterezése hierarchikus.

   ![Hierarchikus klaszterezés Infografika](../../../../5-Clustering/1-Visualize/images/hierarchical.png)
   > Infografika: [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Centroid klaszterezés**. Ez a népszerű algoritmus megköveteli a 'k' választását, vagyis a létrehozandó klaszterek számát, majd az algoritmus meghatározza a klaszter középpontját, és az adatokat e pont köré gyűjti. A [K-means klaszterezés](https://wikipedia.org/wiki/K-means_clustering) a centroid klaszterezés népszerű változata. A középpontot a legközelebbi átlag határozza meg, innen ered a neve. A klasztertől való négyzetes távolság minimalizálva van.

   ![Centroid klaszterezés Infografika](../../../../5-Clustering/1-Visualize/images/centroid.png)
   > Infografika: [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Eloszlás-alapú klaszterezés**. Statisztikai modellezésen alapul, az eloszlás-alapú klaszterezés középpontjában annak valószínűsége áll, hogy egy adatpont egy klaszterhez tartozik, és ennek megfelelően osztja be. A Gauss-keverék módszerek ehhez a típushoz tartoznak.

- **Sűrűség-alapú klaszterezés**. Az adatpontokat klaszterekhez rendelik azok sűrűsége, vagy egymás körüli csoportosulásuk alapján. Az adatpontokat, amelyek távol vannak a csoporttól, kiugró értékeknek vagy zajnak tekintik. A DBSCAN, Mean-shift és OPTICS ehhez a típushoz tartoznak.

- **Rács-alapú klaszterezés**. Többdimenziós adathalmazok esetén egy rácsot hoznak létre, és az adatokat a rács cellái között osztják el, így klasztereket hozva létre.

## Gyakorlat - klaszterezd az adataidat

A klaszterezés mint technika nagyban segíti a megfelelő vizualizáció, ezért kezdjük azzal, hogy vizualizáljuk a zenei adatainkat. Ez a gyakorlat segít eldönteni, hogy a klaszterezési módszerek közül melyiket használjuk a legjobban az adatok természetéhez.

1. Nyisd meg a [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/1-Visualize/notebook.ipynb) fájlt ebben a mappában.

1. Importáld a `Seaborn` csomagot a jó adatvizualizáció érdekében.

    ```python
    !pip install seaborn
    ```

1. Töltsd be a daladatokat a [_nigerian-songs.csv_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/data/nigerian-songs.csv) fájlból. Töltsd be egy adatkeretbe néhány adatot a dalokról. Készülj fel az adatok feltárására a könyvtárak importálásával és az adatok kiírásával:

    ```python
    import matplotlib.pyplot as plt
    import pandas as pd
    
    df = pd.read_csv("../data/nigerian-songs.csv")
    df.head()
    ```

    Ellenőrizd az adatok első néhány sorát:

    |     | név                     | album                        | előadó              | előadó_top_műfaj | megjelenési_dátum | hossz | népszerűség | táncolhatóság | akusztikusság | energia | hangszeresség | élénkség | hangosság | beszédesség | tempó   | idő_aláírás |
    | --- | ------------------------ | ---------------------------- | ------------------- | ---------------- | ----------------
| 2   | LITT!                    | LITT!                        | AYLØ                | indie r&b        | 2018         | 207758 | 40         | 0.836        | 0.272        | 0.564  | 0.000537         | 0.11     | -7.127   | 0.0424      | 130.005 | 4              |
| 3   | Confident / Feeling Cool | Enjoy Your Life              | Lady Donli          | nigerian pop     | 2019         | 175135 | 14         | 0.894        | 0.798        | 0.611  | 0.000187         | 0.0964   | -4.961   | 0.113       | 111.087 | 4              |
| 4   | wanted you               | rare.                        | Odunsi (The Engine) | afropop          | 2018         | 152049 | 25         | 0.702        | 0.116        | 0.833  | 0.91             | 0.348    | -6.044   | 0.0447      | 105.115 | 4              |

1. Szerezzünk némi információt az adatkeretről az `info()` hívásával:

    ```python
    df.info()
    ```

   Az eredmény így néz ki:

    ```output
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 530 entries, 0 to 529
    Data columns (total 16 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   name              530 non-null    object 
     1   album             530 non-null    object 
     2   artist            530 non-null    object 
     3   artist_top_genre  530 non-null    object 
     4   release_date      530 non-null    int64  
     5   length            530 non-null    int64  
     6   popularity        530 non-null    int64  
     7   danceability      530 non-null    float64
     8   acousticness      530 non-null    float64
     9   energy            530 non-null    float64
     10  instrumentalness  530 non-null    float64
     11  liveness          530 non-null    float64
     12  loudness          530 non-null    float64
     13  speechiness       530 non-null    float64
     14  tempo             530 non-null    float64
     15  time_signature    530 non-null    int64  
    dtypes: float64(8), int64(4), object(4)
    memory usage: 66.4+ KB
    ```

1. Ellenőrizzük a null értékeket az `isnull()` hívásával, és győződjünk meg róla, hogy az összeg 0:

    ```python
    df.isnull().sum()
    ```

    Minden rendben:

    ```output
    name                0
    album               0
    artist              0
    artist_top_genre    0
    release_date        0
    length              0
    popularity          0
    danceability        0
    acousticness        0
    energy              0
    instrumentalness    0
    liveness            0
    loudness            0
    speechiness         0
    tempo               0
    time_signature      0
    dtype: int64
    ```

1. Írjuk le az adatokat:

    ```python
    df.describe()
    ```

    |       | release_date | length      | popularity | danceability | acousticness | energy   | instrumentalness | liveness | loudness  | speechiness | tempo      | time_signature |
    | ----- | ------------ | ----------- | ---------- | ------------ | ------------ | -------- | ---------------- | -------- | --------- | ----------- | ---------- | -------------- |
    | count | 530          | 530         | 530        | 530          | 530          | 530      | 530              | 530      | 530       | 530         | 530        | 530            |
    | mean  | 2015.390566  | 222298.1698 | 17.507547  | 0.741619     | 0.265412     | 0.760623 | 0.016305         | 0.147308 | -4.953011 | 0.130748    | 116.487864 | 3.986792       |
    | std   | 3.131688     | 39696.82226 | 18.992212  | 0.117522     | 0.208342     | 0.148533 | 0.090321         | 0.123588 | 2.464186  | 0.092939    | 23.518601  | 0.333701       |
    | min   | 1998         | 89488       | 0          | 0.255        | 0.000665     | 0.111    | 0                | 0.0283   | -19.362   | 0.0278      | 61.695     | 3              |
    | 25%   | 2014         | 199305      | 0          | 0.681        | 0.089525     | 0.669    | 0                | 0.07565  | -6.29875  | 0.0591      | 102.96125  | 4              |
    | 50%   | 2016         | 218509      | 13         | 0.761        | 0.2205       | 0.7845   | 0.000004         | 0.1035   | -4.5585   | 0.09795     | 112.7145   | 4              |
    | 75%   | 2017         | 242098.5    | 31         | 0.8295       | 0.403        | 0.87575  | 0.000234         | 0.164    | -3.331    | 0.177       | 125.03925  | 4              |
    | max   | 2020         | 511738      | 73         | 0.966        | 0.954        | 0.995    | 0.91             | 0.811    | 0.582     | 0.514       | 206.007    | 5              |

> 🤔 Ha klaszterezéssel dolgozunk, ami egy felügyelet nélküli módszer, amely nem igényel címkézett adatokat, miért mutatjuk ezeket az adatokat címkékkel? Az adatfeltárási fázisban hasznosak lehetnek, de a klaszterezési algoritmusok működéséhez nem szükségesek. Akár el is távolíthatnánk az oszlopfejléceket, és az adatokra oszlopszám alapján hivatkozhatnánk.

Nézzük meg az adatok általános értékeit. Vegyük észre, hogy a népszerűség lehet '0', ami azt mutatja, hogy a daloknak nincs rangsorolása. Távolítsuk el ezeket hamarosan.

1. Használjunk oszlopdiagramot a legnépszerűbb műfajok megállapításához:

    ```python
    import seaborn as sns
    
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top[:5].index,y=top[:5].values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    ![legnépszerűbb](../../../../5-Clustering/1-Visualize/images/popular.png)

✅ Ha szeretnél több legjobb értéket látni, változtasd meg a top `[:5]` értékét nagyobbra, vagy távolítsd el, hogy mindet lásd.

Figyelj, ha a legnépszerűbb műfaj 'Missing'-ként van leírva, az azt jelenti, hogy a Spotify nem osztályozta, ezért távolítsuk el.

1. Távolítsuk el a hiányzó adatokat szűréssel:

    ```python
    df = df[df['artist_top_genre'] != 'Missing']
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    Most ellenőrizzük újra a műfajokat:

    ![legnépszerűbb](../../../../5-Clustering/1-Visualize/images/all-genres.png)

1. Messze a három legnépszerűbb műfaj uralja ezt az adatállományt. Koncentráljunk az `afro dancehall`, `afropop` és `nigerian pop` műfajokra, és szűrjük az adatállományt, hogy eltávolítsuk azokat, amelyek népszerűségi értéke 0 (ami azt jelenti, hogy nem osztályozták népszerűséggel az adatállományban, és zajnak tekinthetők a céljaink szempontjából):

    ```python
    df = df[(df['artist_top_genre'] == 'afro dancehall') | (df['artist_top_genre'] == 'afropop') | (df['artist_top_genre'] == 'nigerian pop')]
    df = df[(df['popularity'] > 0)]
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

1. Végezzünk egy gyors tesztet, hogy lássuk, van-e az adatok között különösen erős korreláció:

    ```python
    corrmat = df.corr(numeric_only=True)
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True)
    ```

    ![korrelációk](../../../../5-Clustering/1-Visualize/images/correlation.png)

    Az egyetlen erős korreláció az `energy` és a `loudness` között van, ami nem túl meglepő, mivel a hangos zene általában elég energikus. Egyébként a korrelációk viszonylag gyengék. Érdekes lesz látni, hogy mit tud kezdeni egy klaszterezési algoritmus ezekkel az adatokkal.

    > 🎓 Ne feledd, hogy a korreláció nem jelent ok-okozati összefüggést! Van bizonyítékunk a korrelációra, de nincs bizonyítékunk az ok-okozati összefüggésre. Egy [szórakoztató weboldal](https://tylervigen.com/spurious-correlations) vizuális példákat mutat be, amelyek hangsúlyozzák ezt a pontot.

Van-e bármilyen konvergencia ebben az adatállományban a dalok érzékelt népszerűsége és táncolhatósága körül? Egy FacetGrid megmutatja, hogy koncentrikus körök alakulnak ki, műfajtól függetlenül. Lehet, hogy a nigériai ízlés egy bizonyos táncolhatósági szinten konvergál ezeknél a műfajoknál?

✅ Próbálj ki különböző adatpontokat (energy, loudness, speechiness) és több vagy más zenei műfajokat. Mit fedezhetsz fel? Nézd meg a `df.describe()` táblázatot, hogy lásd az adatpontok általános eloszlását.

### Gyakorlat - adateloszlás

Jelentősen különbözik-e ez a három műfaj a táncolhatóság érzékelésében a népszerűségük alapján?

1. Vizsgáljuk meg a három legnépszerűbb műfaj adateloszlását a népszerűség és táncolhatóság mentén egy adott x és y tengelyen.

    ```python
    sns.set_theme(style="ticks")
    
    g = sns.jointplot(
        data=df,
        x="popularity", y="danceability", hue="artist_top_genre",
        kind="kde",
    )
    ```

    Felfedezhetsz koncentrikus köröket egy általános konvergenciapont körül, amelyek az eloszlási pontokat mutatják.

    > 🎓 Ne feledd, hogy ez a példa egy KDE (Kernel Density Estimate) grafikont használ, amely az adatokat egy folyamatos valószínűségi sűrűség görbével ábrázolja. Ez lehetővé teszi az adatok értelmezését több eloszlás esetén.

    Általánosságban elmondható, hogy a három műfaj lazán igazodik a népszerűségük és táncolhatóságuk tekintetében. Klaszterek meghatározása ebben a lazán igazodó adatokban kihívást jelent:

    ![eloszlás](../../../../5-Clustering/1-Visualize/images/distribution.png)

1. Készítsünk egy szórásdiagramot:

    ```python
    sns.FacetGrid(df, hue="artist_top_genre", height=5) \
       .map(plt.scatter, "popularity", "danceability") \
       .add_legend()
    ```

    Ugyanazon tengelyek szórásdiagramja hasonló konvergenciamintát mutat

    ![Facetgrid](../../../../5-Clustering/1-Visualize/images/facetgrid.png)

Általánosságban elmondható, hogy a klaszterezéshez használhatsz szórásdiagramokat az adatok klasztereinek megjelenítésére, így ennek a vizualizációs típusnak a elsajátítása nagyon hasznos. A következő leckében ezt a szűrt adatot fogjuk használni, és k-means klaszterezéssel fedezünk fel csoportokat az adatokban, amelyek érdekes módon átfedhetnek.

---

## 🚀Kihívás

A következő lecke előkészítéseként készíts egy diagramot a különböző klaszterezési algoritmusokról, amelyeket felfedezhetsz és használhatsz egy termelési környezetben. Milyen problémákat próbál megoldani a klaszterezés?

## [Utó-lecke kvíz](https://ff-quizzes.netlify.app/en/ml/)

## Áttekintés és önálló tanulás

Mielőtt klaszterezési algoritmusokat alkalmaznál, ahogy megtanultuk, jó ötlet megérteni az adatállomány természetét. Olvass többet erről a témáról [itt](https://www.kdnuggets.com/2019/10/right-clustering-algorithm.html)

[Ez a hasznos cikk](https://www.freecodecamp.org/news/8-clustering-algorithms-in-machine-learning-that-all-data-scientists-should-know/) bemutatja, hogyan viselkednek különböző klaszterezési algoritmusok különböző adatformák esetén.

## Feladat

[Kutatás más vizualizációkról a klaszterezéshez](assignment.md)

---

**Felelősség kizárása**:  
Ez a dokumentum az AI fordítási szolgáltatás, a [Co-op Translator](https://github.com/Azure/co-op-translator) segítségével lett lefordítva. Bár törekszünk a pontosságra, kérjük, vegye figyelembe, hogy az automatikus fordítások hibákat vagy pontatlanságokat tartalmazhatnak. Az eredeti dokumentum az eredeti nyelvén tekintendő hiteles forrásnak. Kritikus információk esetén javasolt professzionális emberi fordítást igénybe venni. Nem vállalunk felelősséget semmilyen félreértésért vagy téves értelmezésért, amely a fordítás használatából eredhet.