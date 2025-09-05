<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "730225ea274c9174fe688b21d421539d",
  "translation_date": "2025-09-05T18:14:20+00:00",
  "source_file": "5-Clustering/1-Visualize/README.md",
  "language_code": "tl"
}
-->
# Panimula sa clustering

Ang clustering ay isang uri ng [Unsupervised Learning](https://wikipedia.org/wiki/Unsupervised_learning) na ipinapalagay na ang dataset ay walang label o ang mga input nito ay hindi tumutugma sa mga pre-defined na output. Gumagamit ito ng iba't ibang algorithm upang suriin ang unlabeled na data at magbigay ng mga pangkat batay sa mga pattern na natutuklasan nito sa data.

[![No One Like You by PSquare](https://img.youtube.com/vi/ty2advRiWJM/0.jpg)](https://youtu.be/ty2advRiWJM "No One Like You by PSquare")

> 🎥 I-click ang imahe sa itaas para sa isang video. Habang nag-aaral ka ng machine learning gamit ang clustering, mag-enjoy sa ilang Nigerian Dance Hall tracks - ito ay isang highly rated na kanta mula 2014 ng PSquare.

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

### Panimula

Ang [Clustering](https://link.springer.com/referenceworkentry/10.1007%2F978-0-387-30164-8_124) ay napaka-kapaki-pakinabang para sa paggalugad ng data. Tingnan natin kung makakatulong ito sa pagtuklas ng mga trend at pattern sa paraan ng pagkonsumo ng musika ng mga Nigerian audience.

✅ Maglaan ng isang minuto upang pag-isipan ang mga gamit ng clustering. Sa totoong buhay, nangyayari ang clustering tuwing mayroon kang tambak ng labahan at kailangang ayusin ang mga damit ng iyong pamilya 🧦👕👖🩲. Sa data science, nangyayari ang clustering kapag sinusubukang suriin ang mga kagustuhan ng isang user, o tukuyin ang mga katangian ng anumang unlabeled dataset. Sa isang paraan, ang clustering ay tumutulong upang maunawaan ang kaguluhan, tulad ng drawer ng medyas.

[![Introduction to ML](https://img.youtube.com/vi/esmzYhuFnds/0.jpg)](https://youtu.be/esmzYhuFnds "Introduction to Clustering")

> 🎥 I-click ang imahe sa itaas para sa isang video: Ipinakikilala ni John Guttag ng MIT ang clustering

Sa isang propesyonal na setting, maaaring gamitin ang clustering upang matukoy ang segmentation ng merkado, tulad ng pagtukoy kung anong mga age group ang bumibili ng mga partikular na produkto. Isa pang gamit ay ang anomaly detection, halimbawa upang matukoy ang pandaraya mula sa dataset ng mga transaksyon sa credit card. Maaari mo ring gamitin ang clustering upang matukoy ang mga tumor sa batch ng mga medical scan.

✅ Maglaan ng isang minuto upang pag-isipan kung paano mo maaaring naranasan ang clustering 'sa totoong buhay', sa isang banking, e-commerce, o business setting.

> 🎓 Nakakatuwa, ang cluster analysis ay nagmula sa mga larangan ng Anthropology at Psychology noong 1930s. Maiisip mo ba kung paano ito ginamit noon?

Bukod dito, maaari mo itong gamitin para sa pag-grupo ng mga resulta ng paghahanap - tulad ng mga shopping link, imahe, o review, halimbawa. Kapaki-pakinabang ang clustering kapag mayroon kang malaking dataset na nais mong bawasan at kung saan nais mong magsagawa ng mas detalyadong pagsusuri, kaya ang teknik na ito ay maaaring gamitin upang matuto tungkol sa data bago bumuo ng iba pang mga modelo.

✅ Kapag ang iyong data ay nakaayos na sa mga cluster, maaari kang magtalaga ng cluster Id dito, at ang teknik na ito ay maaaring maging kapaki-pakinabang sa pagpapanatili ng privacy ng dataset; maaari mong tukuyin ang isang data point sa pamamagitan ng cluster id nito, sa halip na sa pamamagitan ng mas nakakapagpakilalang data. Maiisip mo ba ang iba pang dahilan kung bakit mas gugustuhin mong tukuyin ang isang cluster Id kaysa sa iba pang elemento ng cluster upang kilalanin ito?

Palalimin ang iyong pag-unawa sa mga clustering techniques sa [Learn module](https://docs.microsoft.com/learn/modules/train-evaluate-cluster-models?WT.mc_id=academic-77952-leestott)

## Pagsisimula sa clustering

[Ang Scikit-learn ay nag-aalok ng malawak na hanay](https://scikit-learn.org/stable/modules/clustering.html) ng mga pamamaraan upang magsagawa ng clustering. Ang uri na pipiliin mo ay depende sa iyong use case. Ayon sa dokumentasyon, bawat pamamaraan ay may iba't ibang benepisyo. Narito ang isang pinasimpleng talahanayan ng mga pamamaraan na sinusuportahan ng Scikit-learn at ang kanilang naaangkop na mga use case:

| Pangalan ng Pamamaraan        | Use case                                                               |
| :--------------------------- | :--------------------------------------------------------------------- |
| K-Means                      | pangkalahatang layunin, inductive                                      |
| Affinity propagation         | marami, hindi pantay na mga cluster, inductive                        |
| Mean-shift                   | marami, hindi pantay na mga cluster, inductive                        |
| Spectral clustering          | kaunti, pantay na mga cluster, transductive                           |
| Ward hierarchical clustering | marami, constrained na mga cluster, transductive                      |
| Agglomerative clustering     | marami, constrained, non Euclidean distances, transductive            |
| DBSCAN                       | non-flat geometry, hindi pantay na mga cluster, transductive          |
| OPTICS                       | non-flat geometry, hindi pantay na mga cluster na may variable density, transductive |
| Gaussian mixtures            | flat geometry, inductive                                              |
| BIRCH                        | malaking dataset na may outliers, inductive                           |

> 🎓 Ang paraan ng paglikha natin ng mga cluster ay may kinalaman sa kung paano natin pinagsasama-sama ang mga data point sa mga grupo. Tuklasin natin ang ilang bokabularyo:
>
> 🎓 ['Transductive' vs. 'inductive'](https://wikipedia.org/wiki/Transduction_(machine_learning))
> 
> Ang transductive inference ay nagmumula sa mga naobserbahang training cases na tumutugma sa mga partikular na test cases. Ang inductive inference ay nagmumula sa mga training cases na tumutugma sa mga pangkalahatang panuntunan na pagkatapos ay inilalapat sa mga test cases.
> 
> Halimbawa: Isipin na mayroon kang dataset na bahagyang may label. Ang ilan ay 'records', ang ilan ay 'cds', at ang ilan ay blangko. Ang iyong trabaho ay magbigay ng label para sa mga blangko. Kung pipili ka ng inductive approach, magte-train ka ng model na naghahanap ng 'records' at 'cds', at ilalapat ang mga label na iyon sa iyong unlabeled na data. Ang approach na ito ay mahihirapan sa pag-classify ng mga bagay na aktwal na 'cassettes'. Ang transductive approach, sa kabilang banda, ay mas epektibong humahawak sa hindi kilalang data dahil gumagana ito upang mag-grupo ng mga magkatulad na item at pagkatapos ay maglalapat ng label sa isang grupo. Sa kasong ito, maaaring magpakita ang mga cluster ng 'bilog na musical things' at 'parisukat na musical things'.
> 
> 🎓 ['Non-flat' vs. 'flat' geometry](https://datascience.stackexchange.com/questions/52260/terminology-flat-geometry-in-the-context-of-clustering)
> 
> Nagmula sa terminolohiyang matematika, ang non-flat vs. flat geometry ay tumutukoy sa pagsukat ng distansya sa pagitan ng mga punto sa pamamagitan ng 'flat' ([Euclidean](https://wikipedia.org/wiki/Euclidean_geometry)) o 'non-flat' (non-Euclidean) na mga geometrical na pamamaraan.
>
>'Flat' sa kontekstong ito ay tumutukoy sa Euclidean geometry (ang ilang bahagi nito ay itinuturo bilang 'plane' geometry), at ang non-flat ay tumutukoy sa non-Euclidean geometry. Ano ang kinalaman ng geometry sa machine learning? Bilang dalawang larangan na nakaugat sa matematika, dapat mayroong karaniwang paraan upang sukatin ang distansya sa pagitan ng mga punto sa mga cluster, at maaaring gawin ito sa 'flat' o 'non-flat' na paraan, depende sa kalikasan ng data. [Euclidean distances](https://wikipedia.org/wiki/Euclidean_distance) ay sinusukat bilang haba ng segment ng linya sa pagitan ng dalawang punto. [Non-Euclidean distances](https://wikipedia.org/wiki/Non-Euclidean_geometry) ay sinusukat sa kahabaan ng kurba. Kung ang iyong data, kapag na-visualize, ay tila hindi umiiral sa isang plane, maaaring kailanganin mong gumamit ng specialized algorithm upang hawakan ito.
>
![Flat vs Nonflat Geometry Infographic](../../../../5-Clustering/1-Visualize/images/flat-nonflat.png)
> Infographic ni [Dasani Madipalli](https://twitter.com/dasani_decoded)
> 
> 🎓 ['Distances'](https://web.stanford.edu/class/cs345a/slides/12-clustering.pdf)
> 
> Ang mga cluster ay tinutukoy ng kanilang distance matrix, halimbawa ang mga distansya sa pagitan ng mga punto. Ang distansya ay maaaring sukatin sa ilang paraan. Ang mga Euclidean cluster ay tinutukoy ng average ng mga halaga ng punto, at naglalaman ng 'centroid' o gitnang punto. Ang mga distansya ay sinusukat sa pamamagitan ng distansya sa centroid na iyon. Ang mga non-Euclidean distances ay tumutukoy sa 'clustroids', ang punto na pinakamalapit sa iba pang mga punto. Ang mga clustroids naman ay maaaring tukuyin sa iba't ibang paraan.
> 
> 🎓 ['Constrained'](https://wikipedia.org/wiki/Constrained_clustering)
> 
> [Constrained Clustering](https://web.cs.ucdavis.edu/~davidson/Publications/ICDMTutorial.pdf) ay nagpapakilala ng 'semi-supervised' learning sa unsupervised na pamamaraang ito. Ang mga relasyon sa pagitan ng mga punto ay minarkahan bilang 'cannot link' o 'must-link' kaya ang ilang mga panuntunan ay ipinapataw sa dataset.
>
>Halimbawa: Kung ang isang algorithm ay pinakawalan sa batch ng unlabeled o semi-labelled na data, ang mga cluster na nabuo nito ay maaaring hindi maganda ang kalidad. Sa halimbawa sa itaas, maaaring mag-grupo ang mga cluster ng 'bilog na musical things' at 'parisukat na musical things' at 'triangular things' at 'cookies'. Kung bibigyan ng ilang constraints, o mga panuntunan na susundin ("ang item ay dapat gawa sa plastic", "ang item ay kailangang makagawa ng musika") makakatulong ito upang 'pigilan' ang algorithm na gumawa ng mas mahusay na mga pagpipilian.
> 
> 🎓 'Density'
> 
> Ang data na 'noisy' ay itinuturing na 'dense'. Ang mga distansya sa pagitan ng mga punto sa bawat isa sa mga cluster nito ay maaaring magpakita, sa pagsusuri, na mas siksik o 'crowded' kaya ang data na ito ay kailangang suriin gamit ang naaangkop na clustering method. [Ang artikulong ito](https://www.kdnuggets.com/2020/02/understanding-density-based-clustering.html) ay nagpapakita ng pagkakaiba sa pagitan ng paggamit ng K-Means clustering vs. HDBSCAN algorithms upang galugarin ang isang noisy dataset na may hindi pantay na cluster density.

## Mga clustering algorithm

Mayroong higit sa 100 clustering algorithms, at ang kanilang paggamit ay nakadepende sa kalikasan ng data na hawak. Talakayin natin ang ilan sa mga pangunahing uri:

- **Hierarchical clustering**. Kung ang isang object ay na-classify batay sa kalapitan nito sa isang kalapit na object, sa halip na sa mas malayong object, ang mga cluster ay nabubuo batay sa distansya ng mga miyembro nito sa iba pang mga object. Ang agglomerative clustering ng Scikit-learn ay hierarchical.

   ![Hierarchical clustering Infographic](../../../../5-Clustering/1-Visualize/images/hierarchical.png)
   > Infographic ni [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Centroid clustering**. Ang sikat na algorithm na ito ay nangangailangan ng pagpili ng 'k', o ang bilang ng mga cluster na bubuuin, pagkatapos nito ay tinutukoy ng algorithm ang gitnang punto ng isang cluster at kinukuha ang data sa paligid ng puntong iyon. Ang [K-means clustering](https://wikipedia.org/wiki/K-means_clustering) ay isang sikat na bersyon ng centroid clustering. Ang gitna ay tinutukoy ng pinakamalapit na mean, kaya ang pangalan. Ang squared distance mula sa cluster ay pinapaliit.

   ![Centroid clustering Infographic](../../../../5-Clustering/1-Visualize/images/centroid.png)
   > Infographic ni [Dasani Madipalli](https://twitter.com/dasani_decoded)

- **Distribution-based clustering**. Batay sa statistical modeling, ang distribution-based clustering ay nakatuon sa pagtukoy ng posibilidad na ang isang data point ay kabilang sa isang cluster, at itinalaga ito nang naaayon. Ang Gaussian mixture methods ay kabilang sa ganitong uri.

- **Density-based clustering**. Ang mga data point ay itinalaga sa mga cluster batay sa kanilang density, o ang kanilang pag-grupo sa paligid ng isa't isa. Ang mga data point na malayo sa grupo ay itinuturing na outliers o noise. Ang DBSCAN, Mean-shift, at OPTICS ay kabilang sa ganitong uri ng clustering.

- **Grid-based clustering**. Para sa multi-dimensional datasets, isang grid ang nilikha at ang data ay hinahati sa mga cell ng grid, kaya't nabubuo ang mga cluster.

## Ehersisyo - i-cluster ang iyong data

Ang clustering bilang isang teknik ay lubos na natutulungan ng tamang visualization, kaya't magsimula tayo sa pag-visualize ng ating music data. Ang ehersisyong ito ay makakatulong sa atin na magpasya kung alin sa mga pamamaraan ng clustering ang pinaka-epektibong gamitin para sa kalikasan ng data na ito.

1. Buksan ang [_notebook.ipynb_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/1-Visualize/notebook.ipynb) file sa folder na ito.

1. I-import ang `Seaborn` package para sa mahusay na data visualization.

    ```python
    !pip install seaborn
    ```

1. I-append ang song data mula sa [_nigerian-songs.csv_](https://github.com/microsoft/ML-For-Beginners/blob/main/5-Clustering/data/nigerian-songs.csv). Mag-load ng dataframe na may ilang data tungkol sa mga kanta. Maghanda upang galugarin ang data na ito sa pamamagitan ng pag-import ng mga library at pag-dump ng data:

    ```python
    import matplotlib.pyplot as plt
    import pandas as pd
    
    df = pd.read_csv("../data/nigerian-songs.csv")
    df.head()
    ```

    Tingnan ang unang ilang linya ng data:

    |     | name                     | album                        | artist              | artist_top_genre | release_date | length | popularity | danceability | acousticness | energy | instrumentalness | liveness | loudness | speechiness | tempo   | time_signature |
    | --- | ------------------------ | ---------------------------- | ------------------- | ---------------- | ------------ | ------ | ---------- | ------------ | ------------ | ------ | ---------------- | -------- | -------- | ----------- | ------- | -------------- |
    | 0   | Sparky                   | Mandy & The Jungle           | Cruel Santino       | alternative r&b  | 2019         | 144000 | 48         | 0.666        | 0.851        | 0.42   | 0.534            | 0.11     | -6.699   | 0.0829      | 133.015 | 5              |
    | 1   | shuga rush               | EVERYTHING YOU HEARD IS TRUE | Odunsi (The Engine) | afropop          | 2020         | 89488  | 30         | 0.71         | 0.0822       | 0.683  | 0.000169         | 0.101    | -5.64    | 0.36        | 129.993 | 3              |
| 2   | LITT!                    | LITT!                        | AYLØ                | indie r&b        | 2018         | 207758 | 40         | 0.836        | 0.272        | 0.564  | 0.000537         | 0.11     | -7.127   | 0.0424      | 130.005 | 4              |
| 3   | Confident / Feeling Cool | Enjoy Your Life              | Lady Donli          | nigerian pop     | 2019         | 175135 | 14         | 0.894        | 0.798        | 0.611  | 0.000187         | 0.0964   | -4.961   | 0.113       | 111.087 | 4              |
| 4   | wanted you               | rare.                        | Odunsi (The Engine) | afropop          | 2018         | 152049 | 25         | 0.702        | 0.116        | 0.833  | 0.91             | 0.348    | -6.044   | 0.0447      | 105.115 | 4              |

1. Kunin ang ilang impormasyon tungkol sa dataframe, gamit ang `info()`:

    ```python
    df.info()
    ```

   Ang output ay ganito ang hitsura:

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

1. Siguraduhing walang null values, sa pamamagitan ng pagtawag sa `isnull()` at pag-verify na ang kabuuan ay 0:

    ```python
    df.isnull().sum()
    ```

    Mukhang maayos:

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

1. Ilarawan ang data:

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

> 🤔 Kung ang clustering ay isang unsupervised method na hindi nangangailangan ng labeled data, bakit natin ipinapakita ang data na may labels? Sa yugto ng pagsusuri ng data, ito ay kapaki-pakinabang, ngunit hindi ito kinakailangan para gumana ang clustering algorithms. Maaari mong alisin ang mga column headers at tukuyin ang data sa pamamagitan ng column number.

Tingnan ang pangkalahatang halaga ng data. Tandaan na ang popularity ay maaaring '0', na nagpapakita ng mga kanta na walang ranking. Alisin natin ang mga ito sa lalong madaling panahon.

1. Gumamit ng barplot upang malaman ang pinakasikat na genres:

    ```python
    import seaborn as sns
    
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top[:5].index,y=top[:5].values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    ![most popular](../../../../5-Clustering/1-Visualize/images/popular.png)

✅ Kung nais mong makita ang mas maraming top values, palitan ang top `[:5]` sa mas malaking halaga, o alisin ito upang makita ang lahat.

Tandaan, kapag ang top genre ay inilarawan bilang 'Missing', nangangahulugan ito na hindi ito na-classify ng Spotify, kaya alisin natin ito.

1. Alisin ang nawawalang data sa pamamagitan ng pag-filter nito

    ```python
    df = df[df['artist_top_genre'] != 'Missing']
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

    Ngayon muling suriin ang mga genres:

    ![most popular](../../../../5-Clustering/1-Visualize/images/all-genres.png)

1. Sa ngayon, ang tatlong nangungunang genres ang nangingibabaw sa dataset na ito. Mag-focus tayo sa `afro dancehall`, `afropop`, at `nigerian pop`, at karagdagang i-filter ang dataset upang alisin ang anumang may 0 popularity value (na nangangahulugang hindi ito na-classify na may popularity sa dataset at maaaring ituring na noise para sa ating layunin):

    ```python
    df = df[(df['artist_top_genre'] == 'afro dancehall') | (df['artist_top_genre'] == 'afropop') | (df['artist_top_genre'] == 'nigerian pop')]
    df = df[(df['popularity'] > 0)]
    top = df['artist_top_genre'].value_counts()
    plt.figure(figsize=(10,7))
    sns.barplot(x=top.index,y=top.values)
    plt.xticks(rotation=45)
    plt.title('Top genres',color = 'blue')
    ```

1. Gumawa ng mabilis na pagsusuri upang makita kung ang data ay may malakas na correlation:

    ```python
    corrmat = df.corr(numeric_only=True)
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True)
    ```

    ![correlations](../../../../5-Clustering/1-Visualize/images/correlation.png)

    Ang tanging malakas na correlation ay sa pagitan ng `energy` at `loudness`, na hindi masyadong nakakagulat, dahil ang malakas na musika ay karaniwang mas energetic. Bukod dito, ang correlations ay medyo mahina. Magiging interesante ang makita kung ano ang magagawa ng clustering algorithm sa data na ito.

    > 🎓 Tandaan na ang correlation ay hindi nangangahulugan ng causation! Mayroon tayong patunay ng correlation ngunit walang patunay ng causation. Ang [nakakatawang web site](https://tylervigen.com/spurious-correlations) ay may mga visual na nag-eemphasize sa puntong ito.

Mayroon bang convergence sa dataset na ito sa paligid ng perceived popularity at danceability ng isang kanta? Ang isang FacetGrid ay nagpapakita na may mga concentric circles na nagkakatugma, anuman ang genre. Posible kaya na ang mga Nigerian tastes ay nagkakatugma sa isang tiyak na antas ng danceability para sa genre na ito?

✅ Subukan ang iba't ibang datapoints (energy, loudness, speechiness) at mas marami o iba't ibang musical genres. Ano ang maaari mong matuklasan? Tingnan ang `df.describe()` table upang makita ang pangkalahatang spread ng data points.

### Ehersisyo - distribusyon ng data

Ang tatlong genres ba na ito ay makabuluhang naiiba sa perception ng kanilang danceability, base sa kanilang popularity?

1. Suriin ang distribusyon ng data ng ating top three genres para sa popularity at danceability sa isang ibinigay na x at y axis.

    ```python
    sns.set_theme(style="ticks")
    
    g = sns.jointplot(
        data=df,
        x="popularity", y="danceability", hue="artist_top_genre",
        kind="kde",
    )
    ```

    Maaari kang makakita ng concentric circles sa paligid ng isang pangkalahatang punto ng convergence, na nagpapakita ng distribusyon ng mga puntos.

    > 🎓 Tandaan na ang halimbawang ito ay gumagamit ng isang KDE (Kernel Density Estimate) graph na kumakatawan sa data gamit ang isang tuloy-tuloy na probability density curve. Pinapayagan nito tayong ma-interpret ang data kapag nagtatrabaho sa maraming distribusyon.

    Sa pangkalahatan, ang tatlong genres ay maluwag na nagkakatugma sa kanilang popularity at danceability. Ang pagtukoy ng clusters sa maluwag na data na ito ay magiging hamon:

    ![distribution](../../../../5-Clustering/1-Visualize/images/distribution.png)

1. Gumawa ng scatter plot:

    ```python
    sns.FacetGrid(df, hue="artist_top_genre", height=5) \
       .map(plt.scatter, "popularity", "danceability") \
       .add_legend()
    ```

    Ang scatterplot ng parehong axes ay nagpapakita ng katulad na pattern ng convergence

    ![Facetgrid](../../../../5-Clustering/1-Visualize/images/facetgrid.png)

Sa pangkalahatan, para sa clustering, maaari mong gamitin ang scatterplots upang ipakita ang clusters ng data, kaya ang pag-master ng ganitong uri ng visualization ay napaka-kapaki-pakinabang. Sa susunod na aralin, gagamitin natin ang filtered data na ito at gagamit ng k-means clustering upang matuklasan ang mga grupo sa data na ito na tila nag-o-overlap sa mga interesanteng paraan.

---

## 🚀Hamunin

Bilang paghahanda para sa susunod na aralin, gumawa ng chart tungkol sa iba't ibang clustering algorithms na maaari mong matuklasan at gamitin sa isang production environment. Anong mga uri ng problema ang sinusubukang tugunan ng clustering?

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Review at Pag-aaral ng Sarili

Bago ka mag-apply ng clustering algorithms, tulad ng natutunan natin, magandang ideya na maunawaan ang likas na katangian ng iyong dataset. Magbasa pa tungkol sa paksang ito [dito](https://www.kdnuggets.com/2019/10/right-clustering-algorithm.html)

[Ang kapaki-pakinabang na artikulong ito](https://www.freecodecamp.org/news/8-clustering-algorithms-in-machine-learning-that-all-data-scientists-should-know/) ay naglalakad sa iyo sa iba't ibang paraan kung paano kumikilos ang iba't ibang clustering algorithms, base sa iba't ibang hugis ng data.

## Takdang-Aralin

[Mag-research ng iba pang visualizations para sa clustering](assignment.md)

---

**Paunawa**:  
Ang dokumentong ito ay isinalin gamit ang AI translation service na [Co-op Translator](https://github.com/Azure/co-op-translator). Bagama't sinisikap naming maging tumpak, pakitandaan na ang mga awtomatikong pagsasalin ay maaaring maglaman ng mga pagkakamali o hindi pagkakatugma. Ang orihinal na dokumento sa kanyang katutubong wika ang dapat ituring na opisyal na sanggunian. Para sa mahalagang impormasyon, inirerekomenda ang propesyonal na pagsasalin ng tao. Hindi kami mananagot sa anumang hindi pagkakaunawaan o maling interpretasyon na maaaring magmula sa paggamit ng pagsasaling ito.