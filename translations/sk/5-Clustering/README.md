# Zhlukovacie modely pre strojové učenie

Zhlukovanie je úloha strojového učenia, kde sa snaží nájsť objekty, ktoré si navzájom pripomínajú, a tieto zoskupiť do skupín nazývaných zhluky. Čo odlišuje zhlukovanie od iných prístupov v strojovom učení, je to, že sa deje automaticky, v skutočnosti je spravodlivé povedať, že je to opak riadeného učenia.

## Regionálna téma: zhlukovacie modely pre hudobný vkus nigerijského publika 🎧

Nigerijské rôznorodé publikum má rôznorodé hudobné chute. Použitím údajov stiahnutých zo Spotify (inšpirované [týmto článkom](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421)), sa pozrime na niektorú hudbu populárnu v Nigérii. Táto dátová sada obsahuje údaje o rôznych skladbách ako skóre „danceability“ (tančiteľnosť), „acousticness“ (akustickosť), hlasitosť, „speechiness“ (rečovitosť), popularita a energia. Bude zaujímavé objaviť vzory v týchto údajoch!

![Gramofón](../../../translated_images/sk/turntable.f2b86b13c53302dc.webp)

> Foto od <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Marcela Laskoski</a> na <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
V tejto sérii lekcií objavíte nové spôsoby analýzy údajov pomocou zhlukovacích techník. Zhlukovanie je obzvlášť užitočné, keď vaša dátová sada nemá štítky. Ak však štítky má, potom by mohli byť užitočnejšie klasifikačné techniky, ako ste sa naučili v predchádzajúcich lekciách. Ale v prípadoch, keď chcete zoskupiť neoznačené dáta, zhlukovanie je skvelý spôsob, ako objaviť vzory.

> Existujú užitočné nástroje s nízkym kódom, ktoré vám môžu pomôcť naučiť sa pracovať so zhlukovacími modelmi. Skúste [Azure ML pre túto úlohu](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

## Lekcie

1. [Úvod do zhlukovania](1-Visualize/README.md)
2. [Zhlukovanie K-means](2-K-Means/README.md)

## Kredity

Tieto lekcie boli napísané s 🎶 od [Jen Looper](https://www.twitter.com/jenlooper) s užitočnými recenziami od [Rishit Dagli](https://rishit_dagli/) a [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan).

Dátová sada [Nigerijské piesne](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) bola získaná z Kaggle zo Spotify.

Užitočné príklady K-Means, ktoré pomohli pri tvorbe tejto lekcie, zahŕňajú túto [prieskum irisov](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering), tento [úvodný notebook](https://www.kaggle.com/prashant111/k-means-clustering-with-python) a tento [hypotetický príklad mimovládnej organizácie](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering).

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Vyhlásenie o zodpovednosti**:
Tento dokument bol preložený pomocou AI prekladateľskej služby [Co-op Translator](https://github.com/Azure/co-op-translator). Hoci sa snažíme o presnosť, vezmite prosím na vedomie, že automatické preklady môžu obsahovať chyby alebo nepresnosti. Pôvodný dokument v jeho natívnom jazyku by mal byť považovaný za autoritatívny zdroj. Pre kritické informácie sa odporúča profesionálny ľudský preklad. Nie sme zodpovední za žiadne nedorozumenia alebo nesprávne interpretácie vyplývajúce z použitia tohto prekladu.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->