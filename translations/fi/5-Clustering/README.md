# Klusterointimallit koneoppimiseen

Klusterointi on koneoppimisen tehtävä, jossa pyritään löytämään toisiaan muistuttavia objekteja ja ryhmittelemään ne klustereiksi kutsuttuihin ryhmiin. Se, mikä erottaa klusteroinnin muista koneoppimisen lähestymistavoista, on se, että kaikki tapahtuu automaattisesti, itse asiassa on oikeutettua sanoa, että se on päinvastaista valvotulle oppimiselle.

## Alueellinen aihe: klusterointimallit Nigerian yleisön musiikkimakua varten 🎧

Nigerian monimuotoisella yleisöllä on monimuotoiset musiikkimieltymykset. Käyttäen Spotifylta poimittua dataa (innoittajana [tämä artikkeli](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421)), katsotaan hetki Nigeriassa suosittua musiikkia. Tämä tietojoukko sisältää tietoa eri kappaleiden ’tanssittavuus’-pisteestä, 'akustisuudesta', äänenvoimakkuudesta, ’puhekielisyydestä’, suosiosta ja energiasta. On mielenkiintoista tutkia, millaisia kuvioita tästä datasta löytyy!

![Vinyylisoitin](../../../translated_images/fi/turntable.f2b86b13c53302dc.webp)

> Kuva: <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Marcela Laskoski</a> sivustolla <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
Tässä oppitesarjassa opit uusia tapoja analysoida dataa klusterointimenetelmillä. Klusterointi on erityisen hyödyllistä silloin, kun tietojoukossasi ei ole tunnisteita. Jos tunnisteita kuitenkin on, silloin luokittelumenetelmät, kuten aiemmin oppimasi, voivat olla hyödyllisempiä. Mutta silloin, kun haluat ryhmitellä tunnistamattomia tietoja, klusterointi on erinomainen tapa löytää malleja.

> On olemassa hyödyllisiä vähän koodia vaativia työkaluja, jotka auttavat sinua oppimaan klusterointimallien kanssa työskentelyä. Kokeile [Azure ML -ratkaisua tähän tehtävään](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

## Oppitunnit

1. [Johdatus klusterointiin](1-Visualize/README.md)
2. [K-Means-klusterointi](2-K-Means/README.md)

## Tekijät

Nämä oppitunnit kirjoitti ja sävelsi 🎶 [Jen Looper](https://www.twitter.com/jenlooper) avustavien arviointien kera [Rishit Dagli](https://rishit_dagli/) ja [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan).

[Nigerian Songs](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) -tietojoukko on peräisin Kagglesta, ja se on poimittu Spotifylta.

Hyödyllisiä K-Means-esimerkkejä, jotka auttoivat tämän oppitunnin luomisessa, ovat muun muassa tämä [iris-kartoitus](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering), tämä [aloitusmuistio](https://www.kaggle.com/prashant111/k-means-clustering-with-python) sekä tämä [hypoteettinen NGO-esimerkki](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering).

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Vastuuvapauslauseke**:
Tämä asiakirja on käännetty käyttämällä tekoälypohjaista käännöspalvelua [Co-op Translator](https://github.com/Azure/co-op-translator). Vaikka pyrimme tarkkuuteen, otathan huomioon, että automaattiset käännökset saattavat sisältää virheitä tai epätarkkuuksia. Alkuperäinen asiakirja sen alkuperäiskielellä on virallinen lähde. Tärkeissä asioissa suositellaan ammattimaista ihmiskäännöstä. Emme ole vastuussa tämän käännöksen käytöstä aiheutuvista väärinymmärryksistä tai tulkinnoista.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->