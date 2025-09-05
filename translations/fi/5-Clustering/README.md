<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "b28a3a4911584062772c537b653ebbc7",
  "translation_date": "2025-09-04T23:58:19+00:00",
  "source_file": "5-Clustering/README.md",
  "language_code": "fi"
}
-->
# Klusterointimallit koneoppimiseen

Klusterointi on koneoppimisen tehtävä, jossa pyritään löytämään toisiaan muistuttavia objekteja ja ryhmittelemään ne ryhmiin, joita kutsutaan klustereiksi. Se, mikä erottaa klusteroinnin muista koneoppimisen lähestymistavoista, on se, että prosessi tapahtuu automaattisesti. Itse asiassa voidaan sanoa, että se on päinvastainen valvotulle oppimiselle.

## Alueellinen aihe: klusterointimallit Nigerian yleisön musiikkimakua varten 🎧

Nigerian monimuotoinen yleisö nauttii monenlaisesta musiikista. Käyttämällä Spotifysta kerättyä dataa (inspiroituna [tästä artikkelista](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421)), tarkastellaan joitakin Nigeriassa suosittuja kappaleita. Tämä datasetti sisältää tietoa kappaleiden "tanssittavuudesta", "akustisuudesta", äänenvoimakkuudesta, "puheisuudesta", suosiosta ja energiasta. On mielenkiintoista löytää kuvioita tästä datasta!

![Levysoitin](../../../5-Clustering/images/turntable.jpg)

> Kuva: <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Marcela Laskoski</a> palvelussa <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
Tässä oppituntisarjassa opit uusia tapoja analysoida dataa klusterointitekniikoiden avulla. Klusterointi on erityisen hyödyllistä silloin, kun datasetistä puuttuvat etiketit. Jos datasetissä on etiketit, luokittelutekniikat, kuten aiemmissa oppitunneissa opitut, voivat olla hyödyllisempiä. Mutta tilanteissa, joissa haluat ryhmitellä etiketöimätöntä dataa, klusterointi on erinomainen tapa löytää kuvioita.

> On olemassa hyödyllisiä vähäkoodisia työkaluja, jotka voivat auttaa sinua oppimaan klusterointimallien kanssa työskentelyä. Kokeile [Azure ML:ää tähän tehtävään](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

## Oppitunnit

1. [Johdatus klusterointiin](1-Visualize/README.md)
2. [K-Means-klusterointi](2-K-Means/README.md)

## Tekijät

Nämä oppitunnit kirjoitettiin 🎶 [Jen Looperin](https://www.twitter.com/jenlooper) toimesta, ja niitä tarkistivat hyödyllisesti [Rishit Dagli](https://rishit_dagli) ja [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan).

[Nigerian Songs](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) datasetti on peräisin Kagglesta ja kerätty Spotifysta.

Hyödyllisiä K-Means-esimerkkejä, jotka auttoivat tämän oppitunnin luomisessa, ovat tämä [iris-tutkimus](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering), tämä [aloitusnotebook](https://www.kaggle.com/prashant111/k-means-clustering-with-python) ja tämä [hypoteettinen NGO-esimerkki](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering).

---

**Vastuuvapauslauseke**:  
Tämä asiakirja on käännetty käyttämällä tekoälypohjaista käännöspalvelua [Co-op Translator](https://github.com/Azure/co-op-translator). Vaikka pyrimme tarkkuuteen, huomioithan, että automaattiset käännökset voivat sisältää virheitä tai epätarkkuuksia. Alkuperäistä asiakirjaa sen alkuperäisellä kielellä tulisi pitää ensisijaisena lähteenä. Kriittisen tiedon osalta suositellaan ammattimaista ihmiskäännöstä. Emme ole vastuussa väärinkäsityksistä tai virhetulkinnoista, jotka johtuvat tämän käännöksen käytöstä.