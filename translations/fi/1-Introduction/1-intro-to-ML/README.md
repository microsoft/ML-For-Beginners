<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "69389392fa6346e0dfa30f664b7b6fec",
  "translation_date": "2025-09-05T00:31:44+00:00",
  "source_file": "1-Introduction/1-intro-to-ML/README.md",
  "language_code": "fi"
}
-->
# Johdatus koneoppimiseen

## [Ennakkokysely](https://ff-quizzes.netlify.app/en/ml/)

---

[![ML aloittelijoille - Johdatus koneoppimiseen aloittelijoille](https://img.youtube.com/vi/6mSx_KJxcHI/0.jpg)](https://youtu.be/6mSx_KJxcHI "ML aloittelijoille - Johdatus koneoppimiseen aloittelijoille")

> 🎥 Klikkaa yllä olevaa kuvaa lyhyen videon katsomiseksi, jossa käydään läpi tämän oppitunnin sisältöä.

Tervetuloa tähän klassisen koneoppimisen kurssiin aloittelijoille! Olitpa täysin uusi tämän aiheen parissa tai kokenut koneoppimisen ammattilainen, joka haluaa kerrata tiettyjä osa-alueita, olemme iloisia, että liityit mukaan! Haluamme luoda ystävällisen lähtökohdan koneoppimisen opiskelullesi ja otamme mielellämme vastaan [palautettasi](https://github.com/microsoft/ML-For-Beginners/discussions).

[![Johdatus koneoppimiseen](https://img.youtube.com/vi/h0e2HAPTGF4/0.jpg)](https://youtu.be/h0e2HAPTGF4 "Johdatus koneoppimiseen")

> 🎥 Klikkaa yllä olevaa kuvaa videon katsomiseksi: MIT:n John Guttag esittelee koneoppimista

---
## Koneoppimisen aloittaminen

Ennen kuin aloitat tämän kurssin, sinun tulee varmistaa, että tietokoneesi on valmis ajamaan muistikirjoja paikallisesti.

- **Konfiguroi koneesi näiden videoiden avulla**. Käytä seuraavia linkkejä oppiaksesi [Pythonin asentamisen](https://youtu.be/CXZYvNRIAKM) järjestelmääsi ja [tekstieditorin asettamisen](https://youtu.be/EU8eayHWoZg) kehitystä varten.
- **Opiskele Pythonia**. On myös suositeltavaa, että sinulla on perustiedot [Pythonista](https://docs.microsoft.com/learn/paths/python-language/?WT.mc_id=academic-77952-leestott), ohjelmointikielestä, joka on hyödyllinen datatieteilijöille ja jota käytämme tässä kurssissa.
- **Opiskele Node.js ja JavaScript**. Käytämme JavaScriptiä muutaman kerran tässä kurssissa web-sovellusten rakentamiseen, joten sinun tulee asentaa [node](https://nodejs.org) ja [npm](https://www.npmjs.com/), sekä [Visual Studio Code](https://code.visualstudio.com/) Python- ja JavaScript-kehitystä varten.
- **Luo GitHub-tili**. Koska löysit meidät [GitHubista](https://github.com), sinulla saattaa jo olla tili, mutta jos ei, luo sellainen ja haarauta tämä kurssi omaan käyttöösi. (Voit myös antaa meille tähden 😊)
- **Tutustu Scikit-learniin**. Perehdy [Scikit-learniin](https://scikit-learn.org/stable/user_guide.html), koneoppimiskirjastoon, jota käytämme näissä oppitunneissa.

---
## Mitä koneoppiminen on?

Termi 'koneoppiminen' on yksi nykyajan suosituimmista ja eniten käytetyistä termeistä. On melko todennäköistä, että olet kuullut tämän termin ainakin kerran, jos sinulla on jonkinlaista teknologiaan liittyvää taustaa, riippumatta siitä, millä alalla työskentelet. Koneoppimisen mekanismit ovat kuitenkin mysteeri monille. Koneoppimisen aloittelijalle aihe voi joskus tuntua ylivoimaiselta. Siksi on tärkeää ymmärtää, mitä koneoppiminen oikeasti on, ja oppia siitä askel kerrallaan käytännön esimerkkien avulla.

---
## Hype-käyrä

![ml hype curve](../../../../1-Introduction/1-intro-to-ML/images/hype.png)

> Google Trends näyttää termin 'koneoppiminen' viimeaikaisen hype-käyrän

---
## Mysteerien täyttämä universumi

Elämme universumissa, joka on täynnä kiehtovia mysteerejä. Suuret tiedemiehet, kuten Stephen Hawking, Albert Einstein ja monet muut, ovat omistaneet elämänsä merkityksellisen tiedon etsimiseen, joka paljastaa ympäröivän maailman mysteerejä. Tämä on ihmisen oppimisen perusta: ihmislapsi oppii uusia asioita ja paljastaa maailmansa rakenteen vuosi vuodelta kasvaessaan aikuiseksi.

---
## Lapsen aivot

Lapsen aivot ja aistit havaitsevat ympäristönsä tosiasiat ja oppivat vähitellen elämän piilotettuja kaavoja, jotka auttavat lasta luomaan loogisia sääntöjä tunnistamaan opittuja kaavoja. Ihmisaivojen oppimisprosessi tekee ihmisistä tämän maailman kehittyneimmän elävän olennon. Jatkuva oppiminen piilotettujen kaavojen löytämisen kautta ja niiden innovointi mahdollistaa meille itsensä kehittämisen läpi elämän. Tämä oppimiskyky ja kehittymiskyky liittyvät käsitteeseen nimeltä [aivojen plastisuus](https://www.simplypsychology.org/brain-plasticity.html). Pintapuolisesti voimme nähdä joitakin motivoivia yhtäläisyyksiä ihmisaivojen oppimisprosessin ja koneoppimisen käsitteiden välillä.

---
## Ihmisaivot

[Ihmisaivot](https://www.livescience.com/29365-human-brain.html) havaitsevat asioita todellisesta maailmasta, käsittelevät havaittua tietoa, tekevät järkeviä päätöksiä ja suorittavat tiettyjä toimia olosuhteiden perusteella. Tätä kutsutaan älykkääksi käyttäytymiseksi. Kun ohjelmoimme älykkään käyttäytymisprosessin jäljitelmän koneelle, sitä kutsutaan tekoälyksi (AI).

---
## Termistöä

Vaikka termit voivat olla hämmentäviä, koneoppiminen (ML) on tärkeä tekoälyn osa-alue. **ML keskittyy erikoistuneiden algoritmien käyttöön merkityksellisen tiedon löytämiseksi ja piilotettujen kaavojen tunnistamiseksi havaituista tiedoista järkevän päätöksenteon tukemiseksi**.

---
## AI, ML, syväoppiminen

![AI, ML, deep learning, data science](../../../../1-Introduction/1-intro-to-ML/images/ai-ml-ds.png)

> Kaavio, joka näyttää tekoälyn, koneoppimisen, syväoppimisen ja datatieteen väliset suhteet. Infografiikka: [Jen Looper](https://twitter.com/jenlooper), inspiroitunut [tästä grafiikasta](https://softwareengineering.stackexchange.com/questions/366996/distinction-between-ai-ml-neural-networks-deep-learning-and-data-mining)

---
## Käsitteet, joita käsitellään

Tässä kurssissa käsittelemme vain koneoppimisen ydinkäsitteitä, jotka aloittelijan tulee tietää. Keskitymme siihen, mitä kutsumme 'klassiseksi koneoppimiseksi', pääasiassa käyttäen Scikit-learnia, erinomaista kirjastoa, jota monet opiskelijat käyttävät perusasioiden oppimiseen. Laajempien tekoälyn tai syväoppimisen käsitteiden ymmärtämiseksi vahva koneoppimisen perustietämys on välttämätön, ja haluamme tarjota sen tässä.

---
## Tässä kurssissa opit:

- koneoppimisen ydinkäsitteet
- koneoppimisen historiaa
- koneoppiminen ja oikeudenmukaisuus
- regressio-koneoppimistekniikat
- luokittelu-koneoppimistekniikat
- klusterointi-koneoppimistekniikat
- luonnollisen kielen käsittelyyn liittyvät koneoppimistekniikat
- aikasarjojen ennustaminen koneoppimistekniikoilla
- vahvistusoppiminen
- koneoppimisen käytännön sovellukset

---
## Mitä emme käsittele

- syväoppiminen
- neuroverkot
- tekoäly

Oppimiskokemuksen parantamiseksi vältämme neuroverkkojen monimutkaisuutta, 'syväoppimista' - monikerroksista mallinrakennusta neuroverkkojen avulla - ja tekoälyä, joita käsittelemme eri kurssilla. Tarjoamme myös tulevan datatieteen kurssin, joka keskittyy tämän laajemman alan siihen osa-alueeseen.

---
## Miksi opiskella koneoppimista?

Koneoppiminen järjestelmän näkökulmasta määritellään automatisoitujen järjestelmien luomiseksi, jotka voivat oppia piilotettuja kaavoja datasta älykkään päätöksenteon tukemiseksi.

Tämä motivaatio on löyhästi inspiroitunut siitä, miten ihmisaivot oppivat tiettyjä asioita ulkomaailmasta havaitsemansa datan perusteella.

✅ Mieti hetki, miksi yritys haluaisi käyttää koneoppimisstrategioita sen sijaan, että loisi kovakoodatun sääntöpohjaisen moottorin.

---
## Koneoppimisen sovellukset

Koneoppimisen sovellukset ovat nykyään lähes kaikkialla ja yhtä yleisiä kuin data, joka virtaa yhteiskunnissamme, älypuhelimiemme, yhdistettyjen laitteiden ja muiden järjestelmien tuottamana. Ottaen huomioon huippuluokan koneoppimisalgoritmien valtavan potentiaalin, tutkijat ovat tutkineet niiden kykyä ratkaista monimutkaisia ja monialaisia tosielämän ongelmia erinomaisin tuloksin.

---
## Esimerkkejä sovelletusta koneoppimisesta

**Koneoppimista voi käyttää monin tavoin**:

- Ennustamaan sairauden todennäköisyyttä potilaan sairaushistorian tai raporttien perusteella.
- Hyödyntämään säätietoja sääilmiöiden ennustamiseen.
- Ymmärtämään tekstin sentimenttiä.
- Tunnistamaan valeuutisia propagandan leviämisen estämiseksi.

Rahoitus, taloustiede, maantiede, avaruustutkimus, biolääketieteen tekniikka, kognitiotiede ja jopa humanistiset alat ovat ottaneet koneoppimisen käyttöön ratkaistakseen alansa vaativia, datankäsittelyyn liittyviä ongelmia.

---
## Yhteenveto

Koneoppiminen automatisoi kaavojen löytämisen prosessin löytämällä merkityksellisiä oivalluksia todellisesta tai tuotetusta datasta. Se on osoittautunut erittäin arvokkaaksi liiketoiminnassa, terveydenhuollossa ja rahoituksessa, muiden alojen ohella.

Lähitulevaisuudessa koneoppimisen perusteiden ymmärtäminen tulee olemaan välttämätöntä kaikille aloille sen laajan käyttöönoton vuoksi.

---
# 🚀 Haaste

Piirrä paperille tai käytä online-sovellusta, kuten [Excalidraw](https://excalidraw.com/), hahmottaaksesi tekoälyn, koneoppimisen, syväoppimisen ja datatieteen väliset erot. Lisää ideoita ongelmista, joita kukin näistä tekniikoista on hyvä ratkaisemaan.

# [Jälkioppituntikysely](https://ff-quizzes.netlify.app/en/ml/)

---
# Kertaus ja itseopiskelu

Lisätietoja siitä, miten voit työskennellä koneoppimisalgoritmien kanssa pilvessä, seuraa tätä [oppimispolkua](https://docs.microsoft.com/learn/paths/create-no-code-predictive-models-azure-machine-learning/?WT.mc_id=academic-77952-leestott).

Osallistu [oppimispolkuun](https://docs.microsoft.com/learn/modules/introduction-to-machine-learning/?WT.mc_id=academic-77952-leestott) koneoppimisen perusteista.

---
# Tehtävä

[Ota käyttöön](assignment.md)

---

**Vastuuvapauslauseke**:  
Tämä asiakirja on käännetty käyttämällä tekoälypohjaista käännöspalvelua [Co-op Translator](https://github.com/Azure/co-op-translator). Vaikka pyrimme tarkkuuteen, huomioithan, että automaattiset käännökset voivat sisältää virheitä tai epätarkkuuksia. Alkuperäinen asiakirja sen alkuperäisellä kielellä tulisi pitää ensisijaisena lähteenä. Kriittisen tiedon osalta suositellaan ammattimaista ihmiskäännöstä. Emme ole vastuussa väärinkäsityksistä tai virhetulkinnoista, jotka johtuvat tämän käännöksen käytöstä.