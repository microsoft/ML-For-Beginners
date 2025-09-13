<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "b28a3a4911584062772c537b653ebbc7",
  "translation_date": "2025-09-05T12:08:35+00:00",
  "source_file": "5-Clustering/README.md",
  "language_code": "hr"
}
-->
# Modeli grupiranja za strojno učenje

Grupiranje je zadatak strojnog učenja koji nastoji pronaći objekte koji su slični jedni drugima i grupirati ih u skupine koje nazivamo klasterima. Ono što razlikuje grupiranje od drugih pristupa u strojnome učenju jest činjenica da se proces odvija automatski; zapravo, može se reći da je to suprotnost nadziranom učenju.

## Regionalna tema: modeli grupiranja za glazbeni ukus nigerijske publike 🎧

Raznolika publika u Nigeriji ima raznolike glazbene ukuse. Koristeći podatke prikupljene sa Spotifyja (inspirirano [ovim člankom](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421)), pogledajmo neke od popularnih pjesama u Nigeriji. Ovaj skup podataka uključuje informacije o raznim pjesmama, poput ocjene 'plesnosti', 'akustičnosti', glasnoće, 'govorljivosti', popularnosti i energije. Bit će zanimljivo otkriti obrasce u ovim podacima!

![Gramofon](../../../5-Clustering/images/turntable.jpg)

> Fotografija autora <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Marcela Laskoski</a> na <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
U ovom nizu lekcija otkrit ćete nove načine analize podataka koristeći tehnike grupiranja. Grupiranje je posebno korisno kada vaš skup podataka nema oznake. Ako ima oznake, tada bi tehnike klasifikacije, poput onih koje ste naučili u prethodnim lekcijama, mogle biti korisnije. No, u slučajevima kada želite grupirati nepovezane podatke, grupiranje je odličan način za otkrivanje obrazaca.

> Postoje korisni alati s malo koda koji vam mogu pomoći u radu s modelima grupiranja. Isprobajte [Azure ML za ovaj zadatak](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

## Lekcije

1. [Uvod u grupiranje](1-Visualize/README.md)
2. [K-Means grupiranje](2-K-Means/README.md)

## Zasluge

Ove lekcije napisane su uz 🎶 od strane [Jen Looper](https://www.twitter.com/jenlooper) uz korisne recenzije od [Rishit Dagli](https://rishit_dagli) i [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan).

Skup podataka [Nigerijske pjesme](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) preuzet je s Kagglea, prikupljen sa Spotifyja.

Korisni primjeri K-Means grupiranja koji su pomogli u stvaranju ove lekcije uključuju ovu [analizu irisa](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering), ovaj [uvodni notebook](https://www.kaggle.com/prashant111/k-means-clustering-with-python) i ovaj [hipotetski primjer za nevladinu organizaciju](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering).

---

**Odricanje od odgovornosti**:  
Ovaj dokument je preveden korištenjem AI usluge za prevođenje [Co-op Translator](https://github.com/Azure/co-op-translator). Iako nastojimo osigurati točnost, imajte na umu da automatski prijevodi mogu sadržavati pogreške ili netočnosti. Izvorni dokument na izvornom jeziku treba smatrati mjerodavnim izvorom. Za ključne informacije preporučuje se profesionalni prijevod od strane ljudskog prevoditelja. Ne preuzimamo odgovornost za bilo kakva nesporazuma ili pogrešna tumačenja koja mogu proizaći iz korištenja ovog prijevoda.