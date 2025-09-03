<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "b28a3a4911584062772c537b653ebbc7",
  "translation_date": "2025-09-03T17:01:31+00:00",
  "source_file": "5-Clustering/README.md",
  "language_code": "lt"
}
-->
# Klasterizavimo modeliai mašininio mokymosi srityje

Klasterizavimas yra mašininio mokymosi užduotis, kurios tikslas – surasti objektus, panašius vienas į kitą, ir sugrupuoti juos į grupes, vadinamas klasteriais. Kas skiria klasterizavimą nuo kitų mašininio mokymosi metodų, yra tai, kad procesas vyksta automatiškai. Iš tiesų, galima sakyti, kad tai yra priešingybė prižiūrimam mokymuisi.

## Regioninė tema: klasterizavimo modeliai Nigerijos auditorijos muzikiniam skoniui 🎧

Nigerijos įvairialypė auditorija turi skirtingus muzikinius skonius. Naudojant duomenis, surinktus iš Spotify (įkvėpta [šio straipsnio](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421)), pažvelkime į kai kurias Nigerijoje populiarias dainas. Šis duomenų rinkinys apima informaciją apie įvairių dainų „šokamumo“ balą, „akustiškumą“, garsumą, „kalbamumą“, populiarumą ir energiją. Bus įdomu atrasti šių duomenų dėsningumus!

![Patefono nuotrauka](../../../translated_images/turntable.f2b86b13c53302dc106aa741de9dc96ac372864cf458dd6f879119857aab01da.lt.jpg)

> Nuotrauka <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Marcela Laskoski</a> iš <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
Šioje pamokų serijoje jūs atrasite naujus būdus analizuoti duomenis naudojant klasterizavimo technikas. Klasterizavimas yra ypač naudingas, kai jūsų duomenų rinkinyje nėra etikečių. Jei etiketės yra, tada klasifikavimo technikos, kurias išmokote ankstesnėse pamokose, gali būti naudingesnės. Tačiau tais atvejais, kai norite grupuoti nepažymėtus duomenis, klasterizavimas yra puikus būdas atrasti dėsningumus.

> Yra naudingų mažo kodo įrankių, kurie gali padėti jums išmokti dirbti su klasterizavimo modeliais. Išbandykite [Azure ML šiai užduočiai](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

## Pamokos

1. [Įvadas į klasterizavimą](1-Visualize/README.md)
2. [K-Means klasterizavimas](2-K-Means/README.md)

## Kreditas

Šios pamokos buvo parašytos su 🎶 [Jen Looper](https://www.twitter.com/jenlooper), su naudingomis apžvalgomis iš [Rishit Dagli](https://rishit_dagli) ir [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan).

[Nigerijos dainų](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) duomenų rinkinys buvo gautas iš Kaggle, surinktas iš Spotify.

Naudingi K-Means pavyzdžiai, kurie padėjo sukurti šią pamoką, apima šį [irisų tyrimą](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering), šį [įvadinį užrašų knygelės pavyzdį](https://www.kaggle.com/prashant111/k-means-clustering-with-python) ir šį [hipotetinį NVO pavyzdį](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering).

---

**Atsakomybės apribojimas**:  
Šis dokumentas buvo išverstas naudojant AI vertimo paslaugą [Co-op Translator](https://github.com/Azure/co-op-translator). Nors siekiame tikslumo, prašome atkreipti dėmesį, kad automatiniai vertimai gali turėti klaidų ar netikslumų. Originalus dokumentas jo gimtąja kalba turėtų būti laikomas autoritetingu šaltiniu. Kritinei informacijai rekomenduojama naudoti profesionalų žmogaus vertimą. Mes neprisiimame atsakomybės už nesusipratimus ar klaidingus interpretavimus, atsiradusius dėl šio vertimo naudojimo.