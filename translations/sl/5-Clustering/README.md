<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "b28a3a4911584062772c537b653ebbc7",
  "translation_date": "2025-09-05T12:08:46+00:00",
  "source_file": "5-Clustering/README.md",
  "language_code": "sl"
}
-->
# Modeli za gručenje v strojnem učenju

Gručenje je naloga strojnega učenja, pri kateri iščemo predmete, ki so si med seboj podobni, in jih združujemo v skupine, imenovane gruče. Kar razlikuje gručenje od drugih pristopov v strojnem učenju, je dejstvo, da se stvari dogajajo samodejno; pravzaprav lahko rečemo, da je to nasprotje nadzorovanega učenja.

## Regionalna tema: modeli za gručenje glasbenih okusov nigerijskega občinstva 🎧

Raznoliko nigerijsko občinstvo ima raznolike glasbene okuse. Z uporabo podatkov, pridobljenih s Spotifyja (navdihnjeno s [tem člankom](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421)), si poglejmo nekaj glasbe, ki je priljubljena v Nigeriji. Ta podatkovni niz vključuje podatke o različnih pesmih, kot so ocena 'plesnosti', 'akustičnosti', glasnosti, 'govorljivosti', priljubljenosti in energije. Zanimivo bo odkriti vzorce v teh podatkih!

![Gramofon](../../../5-Clustering/images/turntable.jpg)

> Fotografija <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Marcele Laskoski</a> na <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
V tej seriji lekcij boste odkrili nove načine za analizo podatkov z uporabo tehnik gručenja. Gručenje je še posebej uporabno, kadar vaš podatkovni niz nima oznak. Če pa oznake obstajajo, so tehnike klasifikacije, kot ste jih spoznali v prejšnjih lekcijah, morda bolj uporabne. V primerih, ko želite združiti neoznačene podatke, je gručenje odličen način za odkrivanje vzorcev.

> Obstajajo uporabna orodja z malo kode, ki vam lahko pomagajo pri delu z modeli za gručenje. Poskusite [Azure ML za to nalogo](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

## Lekcije

1. [Uvod v gručenje](1-Visualize/README.md)
2. [Gručenje s K-Means](2-K-Means/README.md)

## Zasluge

Te lekcije so bile napisane z 🎶 s strani [Jen Looper](https://www.twitter.com/jenlooper) z uporabnimi pregledi s strani [Rishit Dagli](https://rishit_dagli) in [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan).

Podatkovni niz [Nigerijske pesmi](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) je bil pridobljen s Kaggleja, kot je bil pridobljen s Spotifyja.

Uporabni primeri K-Means, ki so pomagali pri ustvarjanju te lekcije, vključujejo to [raziskovanje irisa](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering), ta [uvodni zvezek](https://www.kaggle.com/prashant111/k-means-clustering-with-python) in ta [hipotetični primer nevladne organizacije](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering).

---

**Omejitev odgovornosti**:  
Ta dokument je bil preveden z uporabo storitve za strojno prevajanje [Co-op Translator](https://github.com/Azure/co-op-translator). Čeprav si prizadevamo za natančnost, vas prosimo, da upoštevate, da lahko avtomatizirani prevodi vsebujejo napake ali netočnosti. Izvirni dokument v njegovem izvirnem jeziku je treba obravnavati kot avtoritativni vir. Za ključne informacije priporočamo strokovno človeško prevajanje. Ne prevzemamo odgovornosti za morebitna nesporazumevanja ali napačne razlage, ki izhajajo iz uporabe tega prevoda.