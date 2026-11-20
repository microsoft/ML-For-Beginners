# Klasterizavimo modeliai mašininiam mokymuisi

Klasterizavimas yra mašininio mokymosi užduotis, kurioje siekiama rasti tarpusavyje panašius objektus ir sugrupuoti juos į grupes, vadinamas klasteriais. Kas skiria klasterizavimą nuo kitų mašininio mokymosi metodų, tai kad procesas vyksta automatiškai, iš tikrųjų, galima sakyti, kad tai priešingybė priežiūriniam mokymuisi.

## Regioninė tema: klasterizavimo modeliai Nigerijos auditorijos muzikos skonio analizėje 🎧

Nigerijos įvairialypė auditorija turi įvairius muzikos skonius. Naudodami duomenis, surinktus iš Spotify (įkvėpti [šio straipsnio](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421)), pažvelkime į kai kurias Nigerijoje populiarias dainas. Ši duomenų bazė apima duomenis apie įvairių dainų „šokiamumą“, „akustinį lygį“, garsumą, „kalbėjimą“, populiarumą ir energiją. Bus įdomu atrasti šių duomenų modelius!

![Grotuvas](../../../translated_images/lt/turntable.f2b86b13c53302dc.webp)

> Nuotrauka <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Marcelos Laskoski</a> iš <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
Šių pamokų serijoje atraskite naujus duomenų analizės būdus, naudodami klasterizavimo technikas. Klasterizavimas ypač naudingas, kai jūsų duomenų rinkinyje trūksta žymių. Jei žymės yra, tada galbūt naudingesnės gali būti klasifikavimo technikos, kurias mokėtės ankstesnėse pamokose. Bet jei norite sugrupuoti nepažymėtus duomenis, klasterizavimas yra puikus būdas atrasti modelius.

> Yra naudingi žemo kodo įrankiai, kurie gali padėti išmokti dirbti su klasterizavimo modeliais. Išbandykite [Azure ML šiai užduočiai](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

## Pamokos

1. [Įvadas į klasterizavimą](1-Visualize/README.md)
2. [K-Means klasterizavimas](2-K-Means/README.md)

## Autoriai

Šios pamokos sukurtos su 🎶 pagal [Jen Looper](https://www.twitter.com/jenlooper) už pagalbą dėkingi [Rishit Dagli](https://rishit_dagli/) ir [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan).

[Nigerijos dainų](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) duomenų rinkinys gautas iš Kaggle, surinktas iš Spotify.

Naudingi K-Means pavyzdžiai, kurie padėjo sudaryti šią pamoką, yra šis [gėlės iris tyrimas](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering), šis [įvadinis užrašų knygelės pavyzdys](https://www.kaggle.com/prashant111/k-means-clustering-with-python) ir šis [hipotetinis NVO pavyzdys](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering).

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Atsakomybės apribojimas**:
Šis dokumentas buvo išverstas naudojant dirbtinio intelekto vertimo paslaugą [Co-op Translator](https://github.com/Azure/co-op-translator). Nors siekiame tikslumo, prašome atkreipti dėmesį, kad automatiniai vertimai gali turėti klaidų ar netikslumų. Originalus dokumentas jo gimtąja kalba laikomas autoritetingu šaltiniu. Svarbiai informacijai rekomenduojama naudoti profesionalų žmogiškąjį vertimą. Mes neatsakome už jokius nesusipratimus ar neteisingą interpretaciją, kilusią naudojantis šiuo vertimu.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->