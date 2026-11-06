# Modeli za grupiranje u strojnome učenju

Grupiranje je zadatak strojnog učenja gdje se traže objekti koji su jedan drugome slični i grupiraju u skupine zvane klasteri. Ono po čemu se grupiranje razlikuje od drugih pristupa u strojnome učenju jest to što se sve odvija automatski, zapravo, može se reći da je to suprotno od nadziranog učenja.

## Regijski predmet: modeli grupiranja za glazbeni ukus nigerijske publike 🎧

Nigerijska raznolika publika ima raznolike glazbene ukuse. Koristeći podatke prikupljene s Spotifya (inspirirano [ovim člankom](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421)), pogledajmo nešto glazbe popularne u Nigeriji. Ova skupina podataka uključuje podatke o 'plesnosti', 'akustičnosti', glasnoći, 'govornosti', popularnosti i energiji različitih pjesama. Bit će zanimljivo otkriti obrasce u ovim podacima!

![Gramofon](../../../translated_images/hr/turntable.f2b86b13c53302dc.webp)

> Fotografija <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Marcele Laskoski</a> na <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
U ovoj seriji lekcija otkrit ćete nove načine analize podataka koristeći tehnike grupiranja. Grupiranje je posebno korisno kada vaša skupina podataka nema oznake. Ako oznake postoje, tada su tehnike klasifikacije, poput onih koje ste naučili u prethodnim lekcijama, možda korisnije. Ali u slučajevima kada želite grupirati podatke bez oznaka, grupiranje je odličan način za otkrivanje obrazaca.

> Postoje korisni alati s malo koda koji vam mogu pomoći da naučite raditi s modelima grupiranja. Isprobajte [Azure ML za ovaj zadatak](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

## Lekcije

1. [Uvod u grupiranje](1-Visualize/README.md)
2. [K-means grupiranje](2-K-Means/README.md)

## Zasluge

Ove lekcije napisala je 🎶 [Jen Looper](https://www.twitter.com/jenlooper) uz korisne recenzije [Rishita Daglija](https://rishit_dagli/) i [Muhammada Sakiba Khana Inana](https://twitter.com/Sakibinan).

Skup podataka [Nigerijske pjesme](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) preuzet je s Kagglea, prikupljen sa Spotifya.

Korisni primjeri K-means algoritma koji su pomogli u stvaranju ove lekcije uključuju ovu [analizu irisa](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering), ovaj [uvodni notebook](https://www.kaggle.com/prashant111/k-means-clustering-with-python) i ovaj [hipotetski primjer nevladine organizacije](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering).

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Napomena**:
Ovaj dokument je preveden korištenjem AI prevoditeljskog servisa [Co-op Translator](https://github.com/Azure/co-op-translator). Iako težimo točnosti, imajte na umu da automatski prijevodi mogu sadržavati greške ili netočnosti. Izvorni dokument na izvornom jeziku treba smatrati autoritativnim izvorom. Za važne informacije preporuča se profesionalni ljudski prijevod. Nismo odgovorni za bilo kakva nesporazumevanja ili pogrešne interpretacije koje proizlaze iz korištenja ovog prijevoda.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->