<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "6a05fec147e734c3e6bfa54505648e2b",
  "translation_date": "2025-09-05T00:35:08+00:00",
  "source_file": "1-Introduction/2-history-of-ML/README.md",
  "language_code": "fi"
}
-->
# Koneoppimisen historia

![Yhteenveto koneoppimisen historiasta sketchnotena](../../../../sketchnotes/ml-history.png)
> Sketchnote: [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Esiluennon kysely](https://ff-quizzes.netlify.app/en/ml/)

---

[![ML for beginners - Koneoppimisen historia](https://img.youtube.com/vi/N6wxM4wZ7V0/0.jpg)](https://youtu.be/N6wxM4wZ7V0 "ML for beginners - Koneoppimisen historia")

> 🎥 Klikkaa yllä olevaa kuvaa katsellaksesi lyhyen videon tästä oppitunnista.

Tässä oppitunnissa käymme läpi koneoppimisen ja tekoälyn historian merkittäviä virstanpylväitä.

Tekoälyn (AI) historia tieteenalana on kietoutunut koneoppimisen historiaan, sillä koneoppimisen algoritmit ja laskennalliset edistysaskeleet ovat olleet keskeisiä tekoälyn kehityksessä. On hyvä muistaa, että vaikka nämä alat alkoivat muotoutua erillisiksi tutkimusalueiksi 1950-luvulla, tärkeät [algoritmiset, tilastolliset, matemaattiset, laskennalliset ja tekniset löydöt](https://wikipedia.org/wiki/Timeline_of_machine_learning) edelsivät ja limittyivät tähän aikakauteen. Ihmiset ovat itse asiassa pohtineet näitä kysymyksiä jo [satojen vuosien ajan](https://wikipedia.org/wiki/History_of_artificial_intelligence): tämä artikkeli käsittelee ajatusta "ajattelevasta koneesta" ja sen historiallisia älyllisiä perusteita.

---
## Merkittäviä löytöjä

- 1763, 1812 [Bayesin kaava](https://wikipedia.org/wiki/Bayes%27_theorem) ja sen edeltäjät. Tämä kaava ja sen sovellukset ovat keskeisiä päättelyssä, sillä ne kuvaavat tapahtuman todennäköisyyttä aiemman tiedon perusteella.
- 1805 [Pienimmän neliösumman menetelmä](https://wikipedia.org/wiki/Least_squares) ranskalaisen matemaatikon Adrien-Marie Legendren kehittämänä. Tämä teoria, josta opit lisää regressio-osiossa, auttaa datan sovittamisessa.
- 1913 [Markovin ketjut](https://wikipedia.org/wiki/Markov_chain), venäläisen matemaatikon Andrey Markovin mukaan nimettynä, kuvaavat tapahtumien sarjaa aiemman tilan perusteella.
- 1957 [Perceptron](https://wikipedia.org/wiki/Perceptron) on amerikkalaisen psykologin Frank Rosenblattin kehittämä lineaarinen luokittelija, joka on ollut keskeinen syväoppimisen edistysaskelissa.

---

- 1967 [Lähimmän naapurin algoritmi](https://wikipedia.org/wiki/Nearest_neighbor) suunniteltiin alun perin reittien kartoittamiseen. Koneoppimisen kontekstissa sitä käytetään kuvioiden tunnistamiseen.
- 1970 [Takaisinkytkentä](https://wikipedia.org/wiki/Backpropagation) käytetään [syöttöverkkojen](https://wikipedia.org/wiki/Feedforward_neural_network) kouluttamiseen.
- 1982 [Toistuvat neuroverkot](https://wikipedia.org/wiki/Recurrent_neural_network) ovat syöttöverkkojen johdannaisia, jotka luovat ajallisia graafeja.

✅ Tee hieman tutkimusta. Mitkä muut päivämäärät ovat merkittäviä koneoppimisen ja tekoälyn historiassa?

---
## 1950: Ajattelevat koneet

Alan Turing, poikkeuksellinen henkilö, joka valittiin [yleisön toimesta vuonna 2019](https://wikipedia.org/wiki/Icons:_The_Greatest_Person_of_the_20th_Century) 1900-luvun suurimmaksi tiedemieheksi, on tunnettu siitä, että hän auttoi luomaan perustan ajatukselle "ajattelevasta koneesta". Hän kohtasi skeptikkoja ja pyrki todistamaan konseptin empiirisesti luomalla [Turingin testin](https://www.bbc.com/news/technology-18475646), jota käsittelet NLP-osiossa.

---
## 1956: Dartmouthin kesätutkimusprojekti

"Dartmouthin kesätutkimusprojekti tekoälystä oli merkittävä tapahtuma tekoälyn tieteenalalle," ja siellä keksittiin termi 'tekoäly' ([lähde](https://250.dartmouth.edu/highlights/artificial-intelligence-ai-coined-dartmouth)).

> Jokainen oppimisen tai minkä tahansa älykkyyden piirre voidaan periaatteessa kuvata niin tarkasti, että kone voidaan ohjelmoida simuloimaan sitä.

---

Johtava tutkija, matematiikan professori John McCarthy, toivoi "voivansa edetä hypoteesin pohjalta, että jokainen oppimisen tai minkä tahansa älykkyyden piirre voidaan periaatteessa kuvata niin tarkasti, että kone voidaan ohjelmoida simuloimaan sitä." Osallistujina oli myös toinen alan merkittävä hahmo, Marvin Minsky.

Työpajaa pidetään keskustelujen käynnistäjänä ja edistäjänä, mukaan lukien "symbolisten menetelmien nousu, järjestelmät, jotka keskittyivät rajattuihin alueisiin (varhaiset asiantuntijajärjestelmät), ja deduktiiviset järjestelmät vastaan induktiiviset järjestelmät." ([lähde](https://wikipedia.org/wiki/Dartmouth_workshop)).

---
## 1956 - 1974: "Kultaiset vuodet"

1950-luvulta 1970-luvun puoliväliin vallitsi suuri optimismi tekoälyn kyvystä ratkaista monia ongelmia. Vuonna 1967 Marvin Minsky totesi luottavaisesti, että "Sukupolven kuluessa ... tekoälyn luomisen ongelma tulee olemaan olennaisesti ratkaistu." (Minsky, Marvin (1967), Computation: Finite and Infinite Machines, Englewood Cliffs, N.J.: Prentice-Hall)

Luonnollisen kielen käsittelyn tutkimus kukoisti, hakua kehitettiin ja tehtiin tehokkaammaksi, ja luotiin "mikromaailmojen" käsite, jossa yksinkertaisia tehtäviä suoritettiin tavallisen kielen ohjeilla.

---

Tutkimusta rahoitettiin hyvin valtion virastojen toimesta, laskentateho ja algoritmit kehittyivät, ja älykkäiden koneiden prototyyppejä rakennettiin. Joitakin näistä koneista ovat:

* [Shakey-robotti](https://wikipedia.org/wiki/Shakey_the_robot), joka pystyi liikkumaan ja päättämään, miten suorittaa tehtäviä "älykkäästi".

    ![Shakey, älykäs robotti](../../../../1-Introduction/2-history-of-ML/images/shakey.jpg)
    > Shakey vuonna 1972

---

* Eliza, varhainen "chatterbot", pystyi keskustelemaan ihmisten kanssa ja toimimaan alkeellisena "terapeuttina". Opit lisää Elizasta NLP-osiossa.

    ![Eliza, botti](../../../../1-Introduction/2-history-of-ML/images/eliza.png)
    > Versio Elizasta, chatbotista

---

* "Blocks world" oli esimerkki mikromaailmasta, jossa palikoita voitiin pinota ja lajitella, ja koneiden päätöksentekokokeita voitiin testata. Kirjastojen, kuten [SHRDLU](https://wikipedia.org/wiki/SHRDLU), avulla tehdyt edistysaskeleet auttoivat kielen käsittelyä eteenpäin.

    [![blocks world SHRDLU:n kanssa](https://img.youtube.com/vi/QAJz4YKUwqw/0.jpg)](https://www.youtube.com/watch?v=QAJz4YKUwqw "blocks world SHRDLU:n kanssa")

    > 🎥 Klikkaa yllä olevaa kuvaa katsellaksesi videon: Blocks world SHRDLU:n kanssa

---
## 1974 - 1980: "AI-talvi"

1970-luvun puoliväliin mennessä kävi ilmi, että "älykkäiden koneiden" luomisen monimutkaisuutta oli aliarvioitu ja sen lupauksia, ottaen huomioon käytettävissä oleva laskentateho, oli liioiteltu. Rahoitus kuivui ja luottamus alaan hiipui. Joitakin tekijöitä, jotka vaikuttivat luottamuksen laskuun, olivat:
---
- **Rajoitukset**. Laskentateho oli liian rajallinen.
- **Kombinatorinen räjähdys**. Parametrien määrä, joita piti kouluttaa, kasvoi eksponentiaalisesti, kun tietokoneilta vaadittiin enemmän, ilman laskentatehon ja kyvykkyyden rinnakkaista kehitystä.
- **Datan puute**. Datan puute haittasi algoritmien testaamista, kehittämistä ja hienosäätöä.
- **Kysymmekö oikeita kysymyksiä?**. Itse kysymykset, joita esitettiin, alkoivat herättää kysymyksiä. Tutkijat kohtasivat kritiikkiä lähestymistavoistaan:
  - Turingin testit kyseenalaistettiin muun muassa "kiinalaisen huoneen teorian" kautta, joka esitti, että "digitaalisen tietokoneen ohjelmointi voi saada sen näyttämään ymmärtävän kieltä, mutta ei voi tuottaa todellista ymmärrystä." ([lähde](https://plato.stanford.edu/entries/chinese-room/))
  - Tekoälyn, kuten "terapeutti" ELIZAn, eettisyys yhteiskunnassa herätti huolta.

---

Samalla tekoälyn eri koulukunnat alkoivat muodostua. Syntyi kahtiajako ["scruffy" vs. "neat AI"](https://wikipedia.org/wiki/Neats_and_scruffies) -käytäntöjen välillä. _Scruffy_-laboratoriot muokkasivat ohjelmia tuntikausia haluttujen tulosten saavuttamiseksi. _Neat_-laboratoriot "keskittyivät logiikkaan ja muodolliseen ongelmanratkaisuun". ELIZA ja SHRDLU olivat tunnettuja _scruffy_-järjestelmiä. 1980-luvulla, kun syntyi tarve tehdä koneoppimisjärjestelmistä toistettavia, _neat_-lähestymistapa nousi vähitellen etualalle, koska sen tulokset ovat selitettävämpiä.

---
## 1980-luvun asiantuntijajärjestelmät

Kun ala kasvoi, sen hyöty liiketoiminnalle tuli selkeämmäksi, ja 1980-luvulla asiantuntijajärjestelmät yleistyivät. "Asiantuntijajärjestelmät olivat ensimmäisiä todella menestyksekkäitä tekoälyn (AI) ohjelmistomuotoja." ([lähde](https://wikipedia.org/wiki/Expert_system)).

Tämäntyyppinen järjestelmä on itse asiassa _hybridi_, joka koostuu osittain sääntömoottorista, joka määrittelee liiketoimintavaatimukset, ja päättelymoottorista, joka hyödyntää sääntöjärjestelmää uusien faktojen päättelemiseksi.

Tämä aikakausi toi myös lisää huomiota neuroverkoille.

---
## 1987 - 1993: AI:n "jäähtyminen"

Erikoistuneiden asiantuntijajärjestelmien laitteistojen yleistyminen johti valitettavasti niiden liialliseen erikoistumiseen. Henkilökohtaisten tietokoneiden nousu kilpaili näiden suurten, erikoistuneiden, keskitettyjen järjestelmien kanssa. Laskennan demokratisointi oli alkanut, ja se lopulta tasoitti tietä modernille suurten datamäärien räjähdykselle.

---
## 1993 - 2011

Tämä aikakausi toi uuden vaiheen koneoppimiselle ja tekoälylle, jotka pystyivät ratkaisemaan aiemmin datan ja laskentatehon puutteesta johtuneita ongelmia. Datan määrä alkoi kasvaa nopeasti ja tulla laajemmin saataville, hyvässä ja pahassa, erityisesti älypuhelimen tulon myötä vuonna 2007. Laskentateho kasvoi eksponentiaalisesti, ja algoritmit kehittyivät rinnalla. Ala alkoi saavuttaa kypsyyttä, kun aiempien vuosien vapaamuotoisuus alkoi kiteytyä todelliseksi tieteenalaksi.

---
## Nykyhetki

Nykyään koneoppiminen ja tekoäly koskettavat lähes jokaista elämämme osa-aluetta. Tämä aikakausi vaatii huolellista ymmärrystä näiden algoritmien riskeistä ja mahdollisista vaikutuksista ihmisten elämään. Kuten Microsoftin Brad Smith on todennut: "Tietotekniikka nostaa esiin kysymyksiä, jotka liittyvät keskeisiin ihmisoikeuksien suojeluihin, kuten yksityisyyteen ja ilmaisunvapauteen. Nämä kysymykset lisäävät vastuuta teknologiayrityksille, jotka luovat näitä tuotteita. Meidän näkemyksemme mukaan ne myös vaativat harkittua hallituksen sääntelyä ja normien kehittämistä hyväksyttävistä käyttötavoista" ([lähde](https://www.technologyreview.com/2019/12/18/102365/the-future-of-ais-impact-on-society/)).

---

On vielä nähtävissä, mitä tulevaisuus tuo tullessaan, mutta on tärkeää ymmärtää nämä tietokonejärjestelmät sekä ohjelmistot ja algoritmit, joita ne käyttävät. Toivomme, että tämä opetusohjelma auttaa sinua saamaan paremman ymmärryksen, jotta voit tehdä omat johtopäätöksesi.

[![Syväoppimisen historia](https://img.youtube.com/vi/mTtDfKgLm54/0.jpg)](https://www.youtube.com/watch?v=mTtDfKgLm54 "Syväoppimisen historia")
> 🎥 Klikkaa yllä olevaa kuvaa katsellaksesi videon: Yann LeCun käsittelee syväoppimisen historiaa tässä luennossa

---
## 🚀Haaste

Tutki yhtä näistä historiallisista hetkistä ja opi lisää niiden takana olevista ihmisistä. Tiedemaailma on täynnä kiehtovia hahmoja, eikä mikään tieteellinen löytö ole syntynyt kulttuurisessa tyhjiössä. Mitä löydät?

## [Jälkiluennon kysely](https://ff-quizzes.netlify.app/en/ml/)

---
## Kertaus ja itseopiskelu

Tässä katsottavaa ja kuunneltavaa:

[Podcast, jossa Amy Boyd keskustelee tekoälyn kehityksestä](http://runasradio.com/Shows/Show/739)

[![Amy Boyd: Tekoälyn historia](https://img.youtube.com/vi/EJt3_bFYKss/0.jpg)](https://www.youtube.com/watch?v=EJt3_bFYKss "Amy Boyd: Tekoälyn historia")

---

## Tehtävä

[Luo aikajana](assignment.md)

---

**Vastuuvapauslauseke**:  
Tämä asiakirja on käännetty käyttämällä tekoälypohjaista käännöspalvelua [Co-op Translator](https://github.com/Azure/co-op-translator). Vaikka pyrimme tarkkuuteen, huomioithan, että automaattiset käännökset voivat sisältää virheitä tai epätarkkuuksia. Alkuperäistä asiakirjaa sen alkuperäisellä kielellä tulisi pitää ensisijaisena lähteenä. Kriittisen tiedon osalta suositellaan ammattimaista ihmiskäännöstä. Emme ole vastuussa väärinkäsityksistä tai virhetulkinnoista, jotka johtuvat tämän käännöksen käytöstä.