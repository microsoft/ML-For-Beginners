<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "b28a3a4911584062772c537b653ebbc7",
  "translation_date": "2025-09-05T15:39:34+00:00",
  "source_file": "5-Clustering/README.md",
  "language_code": "sk"
}
-->
# Modely zhlukovania pre strojové učenie

Zhlukovanie je úloha strojového učenia, ktorá sa snaží nájsť objekty, ktoré sa navzájom podobajú, a zoskupiť ich do skupín nazývaných zhluky. Čo odlišuje zhlukovanie od iných prístupov v strojovom učení, je to, že veci sa dejú automaticky. V skutočnosti je spravodlivé povedať, že je to opak učenia s učiteľom.

## Regionálna téma: modely zhlukovania pre hudobný vkus nigérijského publika 🎧

Rozmanité publikum v Nigérii má rozmanitý hudobný vkus. Pomocou údajov získaných zo Spotify (inšpirované [týmto článkom](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421)), sa pozrime na niektoré populárne skladby v Nigérii. Tento dataset obsahuje údaje o rôznych skladbách, ako napríklad skóre 'tanečnosti', 'akustickosti', hlasitosti, 'rečovosti', popularite a energii. Bude zaujímavé objaviť vzory v týchto údajoch!

![Gramofón](../../../5-Clustering/images/turntable.jpg)

> Foto od <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Marcela Laskoski</a> na <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
V tejto sérii lekcií objavíte nové spôsoby analýzy údajov pomocou techník zhlukovania. Zhlukovanie je obzvlášť užitočné, keď váš dataset nemá štítky. Ak však štítky má, potom môžu byť klasifikačné techniky, ktoré ste sa naučili v predchádzajúcich lekciách, užitočnejšie. Ale v prípadoch, keď chcete zoskupiť neoznačené údaje, zhlukovanie je skvelý spôsob, ako objaviť vzory.

> Existujú užitočné nástroje s nízkym kódom, ktoré vám môžu pomôcť naučiť sa pracovať s modelmi zhlukovania. Vyskúšajte [Azure ML pre túto úlohu](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

## Lekcie

1. [Úvod do zhlukovania](1-Visualize/README.md)
2. [Zhlukovanie pomocou K-Means](2-K-Means/README.md)

## Kredity

Tieto lekcie boli napísané s 🎶 od [Jen Looper](https://www.twitter.com/jenlooper) s užitočnými recenziami od [Rishit Dagli](https://rishit_dagli) a [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan).

Dataset [Nigérijské skladby](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) bol získaný z Kaggle ako údaje zozbierané zo Spotify.

Užitočné príklady K-Means, ktoré pomohli pri tvorbe tejto lekcie, zahŕňajú túto [analýzu kosatcov](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering), tento [úvodný notebook](https://www.kaggle.com/prashant111/k-means-clustering-with-python) a tento [hypotetický príklad NGO](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering).

---

**Upozornenie**:  
Tento dokument bol preložený pomocou služby AI prekladu [Co-op Translator](https://github.com/Azure/co-op-translator). Hoci sa snažíme o presnosť, prosím, berte na vedomie, že automatizované preklady môžu obsahovať chyby alebo nepresnosti. Pôvodný dokument v jeho rodnom jazyku by mal byť považovaný za autoritatívny zdroj. Pre kritické informácie sa odporúča profesionálny ľudský preklad. Nie sme zodpovední za akékoľvek nedorozumenia alebo nesprávne interpretácie vyplývajúce z použitia tohto prekladu.