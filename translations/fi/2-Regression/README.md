<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "508582278dbb8edd2a8a80ac96ef416c",
  "translation_date": "2025-09-04T23:19:10+00:00",
  "source_file": "2-Regression/README.md",
  "language_code": "fi"
}
-->
# Regressiomallit koneoppimisessa
## Alueellinen aihe: Regressiomallit kurpitsan hinnoille Pohjois-Amerikassa 🎃

Pohjois-Amerikassa kurpitsat kaiverretaan usein pelottaviksi kasvoiksi Halloweenia varten. Tutustutaanpa tarkemmin näihin kiehtoviin vihanneksiin!

![jack-o-lanterns](../../../2-Regression/images/jack-o-lanterns.jpg)
> Kuva: <a href="https://unsplash.com/@teutschmann?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Beth Teutschmann</a> osoitteessa <a href="https://unsplash.com/s/photos/jack-o-lanterns?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
## Mitä opit

[![Johdanto regressioon](https://img.youtube.com/vi/5QnJtDad4iQ/0.jpg)](https://youtu.be/5QnJtDad4iQ "Johdantovideo regressioon - Klikkaa katsoaksesi!")
> 🎥 Klikkaa yllä olevaa kuvaa nähdäksesi lyhyen johdantovideon tähän oppituntiin

Tämän osion oppitunnit käsittelevät regressiotyyppejä koneoppimisen kontekstissa. Regressiomallit voivat auttaa määrittämään _suhteen_ muuttujien välillä. Tämän tyyppinen malli voi ennustaa arvoja, kuten pituutta, lämpötilaa tai ikää, ja paljastaa muuttujien välisiä yhteyksiä analysoidessaan datapisteitä.

Näissä oppitunneissa opit lineaarisen ja logistisen regression erot sekä sen, milloin kumpaakin kannattaa käyttää.

[![ML aloittelijoille - Johdanto regressiomalleihin koneoppimisessa](https://img.youtube.com/vi/XA3OaoW86R8/0.jpg)](https://youtu.be/XA3OaoW86R8 "ML aloittelijoille - Johdanto regressiomalleihin koneoppimisessa")

> 🎥 Klikkaa yllä olevaa kuvaa nähdäksesi lyhyen videon regressiomalleista.

Tässä oppituntisarjassa opit aloittamaan koneoppimistehtävät, mukaan lukien Visual Studio Coden konfiguroinnin muistikirjojen hallintaan, joka on yleinen ympäristö datatieteilijöille. Tutustut Scikit-learniin, koneoppimiskirjastoon, ja rakennat ensimmäiset mallisi, keskittyen tässä luvussa regressiomalleihin.

> On olemassa hyödyllisiä vähäkoodisia työkaluja, jotka voivat auttaa sinua oppimaan regressiomallien kanssa työskentelyä. Kokeile [Azure ML:ää tähän tehtävään](https://docs.microsoft.com/learn/modules/create-regression-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

### Oppitunnit

1. [Työkalut käyttöön](1-Tools/README.md)
2. [Datan hallinta](2-Data/README.md)
3. [Lineaarinen ja polynominen regressio](3-Linear/README.md)
4. [Logistinen regressio](4-Logistic/README.md)

---
### Tekijät

"ML regressiolla" on kirjoitettu ♥️:lla [Jen Looperin](https://twitter.com/jenlooper) toimesta.

♥️ Kysymysten laatijoihin kuuluvat: [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan) ja [Ornella Altunyan](https://twitter.com/ornelladotcom)

Kurpitsadatan on ehdottanut [tämä Kaggle-projekti](https://www.kaggle.com/usda/a-year-of-pumpkin-prices), ja sen tiedot ovat peräisin [Specialty Crops Terminal Markets Standard Reports](https://www.marketnews.usda.gov/mnp/fv-report-config-step1?type=termPrice) -raporteista, joita jakaa Yhdysvaltain maatalousministeriö. Olemme lisänneet joitakin pisteitä värin mukaan lajikkeen perusteella normalisoidaksemme jakaumaa. Tämä data on julkista tietoa.

---

**Vastuuvapauslauseke**:  
Tämä asiakirja on käännetty käyttämällä tekoälypohjaista käännöspalvelua [Co-op Translator](https://github.com/Azure/co-op-translator). Vaikka pyrimme tarkkuuteen, huomioithan, että automaattiset käännökset voivat sisältää virheitä tai epätarkkuuksia. Alkuperäinen asiakirja sen alkuperäisellä kielellä tulisi pitää ensisijaisena lähteenä. Kriittisen tiedon osalta suositellaan ammattimaista ihmiskäännöstä. Emme ole vastuussa väärinkäsityksistä tai virhetulkinnoista, jotka johtuvat tämän käännöksen käytöstä.