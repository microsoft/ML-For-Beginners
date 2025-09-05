<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9836ff53cfef716ddfd70e06c5f43436",
  "translation_date": "2025-09-05T00:36:44+00:00",
  "source_file": "3-Web-App/README.md",
  "language_code": "fi"
}
-->
# Rakenna verkkosovellus ML-mallisi käyttöön

Tässä osiossa tutustut soveltavaan koneoppimisen aiheeseen: kuinka tallentaa Scikit-learn-malli tiedostoksi, jota voidaan käyttää ennusteiden tekemiseen verkkosovelluksessa. Kun malli on tallennettu, opit käyttämään sitä Flaskilla rakennetussa verkkosovelluksessa. Ensin luot mallin käyttäen dataa, joka liittyy UFO-havaintoihin! Sen jälkeen rakennat verkkosovelluksen, jonka avulla voit syöttää sekuntimäärän sekä leveys- ja pituusasteen arvot ennustaaksesi, mikä maa raportoi UFO-havainnon.

![UFO-pysäköinti](../../../3-Web-App/images/ufo.jpg)

Kuva: <a href="https://unsplash.com/@mdherren?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Michael Herren</a> palvelussa <a href="https://unsplash.com/s/photos/ufo?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>

## Oppitunnit

1. [Rakenna verkkosovellus](1-Web-App/README.md)

## Tekijät

"Rakenna verkkosovellus" on kirjoitettu ♥️:lla [Jen Looper](https://twitter.com/jenlooper).

♥️ Visailut on kirjoittanut Rohan Raj.

Datasetti on peräisin [Kagglesta](https://www.kaggle.com/NUFORC/ufo-sightings).

Verkkosovelluksen arkkitehtuuria ehdotettiin osittain [tässä artikkelissa](https://towardsdatascience.com/how-to-easily-deploy-machine-learning-models-using-flask-b95af8fe34d4) ja [tässä repossa](https://github.com/abhinavsagar/machine-learning-deployment) Abhinav Sagarin toimesta.

---

**Vastuuvapauslauseke**:  
Tämä asiakirja on käännetty käyttämällä tekoälypohjaista käännöspalvelua [Co-op Translator](https://github.com/Azure/co-op-translator). Vaikka pyrimme tarkkuuteen, huomioithan, että automaattiset käännökset voivat sisältää virheitä tai epätarkkuuksia. Alkuperäistä asiakirjaa sen alkuperäisellä kielellä tulisi pitää ensisijaisena lähteenä. Kriittisen tiedon osalta suositellaan ammattimaista ihmiskäännöstä. Emme ole vastuussa väärinkäsityksistä tai virhetulkinnoista, jotka johtuvat tämän käännöksen käytöstä.