<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1c2ec40cf55c98a028a359c27ef7e45a",
  "translation_date": "2025-09-05T01:35:59+00:00",
  "source_file": "6-NLP/1-Introduction-to-NLP/README.md",
  "language_code": "fi"
}
-->
# Johdatus luonnollisen kielen käsittelyyn

Tässä oppitunnissa käsitellään lyhyesti *luonnollisen kielen käsittelyn* historiaa ja keskeisiä käsitteitä, joka on osa-alue *laskennallisesta kielitieteestä*.

## [Ennakkokysely](https://ff-quizzes.netlify.app/en/ml/)

## Johdanto

NLP, kuten sitä yleisesti kutsutaan, on yksi tunnetuimmista alueista, joilla koneoppimista on sovellettu ja käytetty tuotanto-ohjelmistoissa.

✅ Voitko keksiä ohjelmistoja, joita käytät päivittäin ja joissa todennäköisesti on mukana NLP:tä? Entä tekstinkäsittelyohjelmat tai mobiilisovellukset, joita käytät säännöllisesti?

Opit seuraavista aiheista:

- **Kielten idea**. Miten kielet kehittyivät ja mitkä ovat olleet keskeiset tutkimusalueet.
- **Määritelmät ja käsitteet**. Opit myös määritelmiä ja käsitteitä siitä, miten tietokoneet käsittelevät tekstiä, mukaan lukien jäsennys, kielioppi sekä substantiivien ja verbien tunnistaminen. Oppitunnilla on joitakin koodausharjoituksia, ja useita tärkeitä käsitteitä esitellään, joita opit koodaamaan myöhemmin seuraavissa oppitunneissa.

## Laskennallinen kielitiede

Laskennallinen kielitiede on tutkimus- ja kehitysalue, joka on ollut olemassa vuosikymmeniä ja tutkii, miten tietokoneet voivat työskennellä kielten kanssa, jopa ymmärtää, kääntää ja kommunikoida niiden avulla. Luonnollisen kielen käsittely (NLP) on siihen liittyvä ala, joka keskittyy siihen, miten tietokoneet voivat käsitellä "luonnollisia" eli ihmisten käyttämiä kieliä.

### Esimerkki - puheentunnistus puhelimessa

Jos olet koskaan sanellut puhelimellesi sen sijaan, että kirjoittaisit, tai kysynyt virtuaaliassistentilta kysymyksen, puheesi on muutettu tekstimuotoon ja sitten käsitelty tai *jäsennetty* puhumallasi kielellä. Tunnistetut avainsanat on sitten käsitelty muotoon, jonka puhelin tai assistentti voi ymmärtää ja käyttää.

![ymmärrys](../../../../6-NLP/1-Introduction-to-NLP/images/comprehension.png)
> Todellinen kielellinen ymmärrys on vaikeaa! Kuva: [Jen Looper](https://twitter.com/jenlooper)

### Miten tämä teknologia on mahdollista?

Tämä on mahdollista, koska joku kirjoitti tietokoneohjelman tekemään sen. Muutama vuosikymmen sitten jotkut tieteiskirjailijat ennustivat, että ihmiset puhuisivat enimmäkseen tietokoneilleen ja tietokoneet ymmärtäisivät aina tarkalleen, mitä he tarkoittavat. Valitettavasti ongelma osoittautui vaikeammaksi kuin monet kuvittelivat, ja vaikka se on nykyään paljon paremmin ymmärretty ongelma, täydellisen luonnollisen kielen käsittelyn saavuttamisessa on merkittäviä haasteita, erityisesti lauseen merkityksen ymmärtämisessä. Tämä on erityisen vaikea ongelma, kun kyseessä on huumorin ymmärtäminen tai tunteiden, kuten sarkasmin, havaitseminen lauseessa.

Tässä vaiheessa saatat muistaa koulutunnit, joissa opettaja käsitteli lauseen kieliopillisia osia. Joissakin maissa oppilaille opetetaan kielioppia ja kielitiedettä omana oppiaineenaan, mutta monissa maissa nämä aiheet sisältyvät kielen oppimiseen: joko ensimmäisen kielen oppimiseen alakoulussa (lukemisen ja kirjoittamisen oppiminen) ja mahdollisesti toisen kielen oppimiseen yläkoulussa tai lukiossa. Älä huoli, jos et ole asiantuntija erottamaan substantiiveja verbeistä tai adverbejä adjektiiveista!

Jos sinulla on vaikeuksia erottaa *yksinkertainen preesens* ja *preesensin kestomuoto*, et ole yksin. Tämä on haastavaa monille ihmisille, jopa kielen äidinkielisille puhujille. Hyvä uutinen on, että tietokoneet ovat todella hyviä soveltamaan muodollisia sääntöjä, ja opit kirjoittamaan koodia, joka voi *jäsentää* lauseen yhtä hyvin kuin ihminen. Suurempi haaste, jota tarkastelet myöhemmin, on lauseen *merkityksen* ja *tunteen* ymmärtäminen.

## Esitiedot

Tämän oppitunnin pääasiallinen esitieto on kyky lukea ja ymmärtää oppitunnin kieltä. Oppitunnilla ei ole matemaattisia ongelmia tai yhtälöitä ratkaistavaksi. Vaikka alkuperäinen kirjoittaja kirjoitti tämän oppitunnin englanniksi, se on myös käännetty muille kielille, joten saatat lukea käännöstä. Esimerkeissä käytetään useita eri kieliä (vertaillaan eri kielten kielioppisääntöjä). Näitä ei *käännetä*, mutta selittävä teksti on, joten merkitys pitäisi olla selvä.

Koodausharjoituksissa käytät Pythonia, ja esimerkit käyttävät Python 3.8:aa.

Tässä osiossa tarvitset ja käytät:

- **Python 3 -osaaminen**. Python 3 -ohjelmointikielen ymmärtäminen, tämä oppitunti käyttää syötettä, silmukoita, tiedostojen lukemista ja taulukoita.
- **Visual Studio Code + laajennus**. Käytämme Visual Studio Codea ja sen Python-laajennusta. Voit myös käyttää haluamaasi Python-IDE:tä.
- **TextBlob**. [TextBlob](https://github.com/sloria/TextBlob) on yksinkertaistettu tekstinkäsittelykirjasto Pythonille. Seuraa TextBlob-sivuston ohjeita asentaaksesi sen järjestelmääsi (asenna myös korpukset, kuten alla näytetään):

   ```bash
   pip install -U textblob
   python -m textblob.download_corpora
   ```

> 💡 Vinkki: Voit ajaa Pythonia suoraan VS Code -ympäristöissä. Katso [dokumentaatio](https://code.visualstudio.com/docs/languages/python?WT.mc_id=academic-77952-leestott) saadaksesi lisätietoja.

## Keskustelu koneiden kanssa

Historia siitä, miten tietokoneet saadaan ymmärtämään ihmisten kieltä, ulottuu vuosikymmenten taakse, ja yksi varhaisimmista tutkijoista, joka pohti luonnollisen kielen käsittelyä, oli *Alan Turing*.

### 'Turingin testi'

Kun Turing tutki *tekoälyä* 1950-luvulla, hän pohti, voisiko ihmisen ja tietokoneen välille (kirjoitetun viestinnän kautta) järjestää keskustelutestin, jossa ihminen ei olisi varma, keskusteleeko hän toisen ihmisen vai tietokoneen kanssa.

Jos tietyn keskustelun jälkeen ihminen ei pystyisi määrittämään, olivatko vastaukset peräisin tietokoneelta vai ei, voitaisiinko tietokoneen sanoa *ajattelevan*?

### Inspiraatio - 'imitaatiopeli'

Idea tähän tuli juhlapelistä nimeltä *Imitaatiopeli*, jossa kuulustelija on yksin huoneessa ja hänen tehtävänään on määrittää, kumpi kahdesta henkilöstä (toisessa huoneessa) on mies ja kumpi nainen. Kuulustelija voi lähettää muistiinpanoja ja hänen täytyy yrittää keksiä kysymyksiä, joissa kirjalliset vastaukset paljastavat mysteerihenkilön sukupuolen. Tietenkin toisen huoneen pelaajat yrittävät hämätä kuulustelijaa vastaamalla kysymyksiin tavalla, joka johtaa harhaan tai hämmentää kuulustelijaa, samalla kun he antavat vaikutelman rehellisestä vastauksesta.

### Elizan kehittäminen

1960-luvulla MIT:n tutkija *Joseph Weizenbaum* kehitti [*Elizan*](https://wikipedia.org/wiki/ELIZA), tietokoneen "terapeutin", joka kysyi ihmiseltä kysymyksiä ja antoi vaikutelman ymmärtävänsä heidän vastauksensa. Kuitenkin, vaikka Eliza pystyi jäsentämään lauseen ja tunnistamaan tiettyjä kieliopillisia rakenteita ja avainsanoja antaakseen järkevän vastauksen, sitä ei voitu sanoa *ymmärtävän* lausetta. Jos Elizalle esitettiin lause, joka noudatti muotoa "**Olen** <u>surullinen</u>", se saattoi järjestää ja korvata sanoja lauseessa muodostaakseen vastauksen "Kuinka kauan olet ollut <u>surullinen</u>". 

Tämä antoi vaikutelman, että Eliza ymmärsi väitteen ja esitti jatkokysymyksen, kun todellisuudessa se vain muutti aikamuotoa ja lisäsi joitakin sanoja. Jos Eliza ei pystynyt tunnistamaan avainsanaa, johon sillä oli vastaus, se antoi sen sijaan satunnaisen vastauksen, joka saattoi sopia moniin eri väitteisiin. Elizaa oli helppo huijata, esimerkiksi jos käyttäjä kirjoitti "**Sinä olet** <u>polkupyörä</u>", se saattoi vastata "Kuinka kauan olen ollut <u>polkupyörä</u>?", sen sijaan että antaisi järkevämmän vastauksen.

[![Keskustelu Elizan kanssa](https://img.youtube.com/vi/RMK9AphfLco/0.jpg)](https://youtu.be/RMK9AphfLco "Keskustelu Elizan kanssa")

> 🎥 Klikkaa yllä olevaa kuvaa nähdäksesi videon alkuperäisestä ELIZA-ohjelmasta

> Huom: Voit lukea alkuperäisen kuvauksen [Elizasta](https://cacm.acm.org/magazines/1966/1/13317-elizaa-computer-program-for-the-study-of-natural-language-communication-between-man-and-machine/abstract), joka julkaistiin vuonna 1966, jos sinulla on ACM-tili. Vaihtoehtoisesti voit lukea Elizasta [wikipediassa](https://wikipedia.org/wiki/ELIZA).

## Harjoitus - yksinkertaisen keskustelubotin koodaaminen

Keskustelubotti, kuten Eliza, on ohjelma, joka pyytää käyttäjän syötettä ja vaikuttaa ymmärtävän ja vastaavan älykkäästi. Toisin kuin Eliza, meidän bottimme ei sisällä useita sääntöjä, jotka antavat sille vaikutelman älykkäästä keskustelusta. Sen sijaan botillamme on vain yksi kyky: pitää keskustelu käynnissä satunnaisilla vastauksilla, jotka saattavat toimia melkein missä tahansa triviaalissa keskustelussa.

### Suunnitelma

Vaiheet keskustelubotin rakentamisessa:

1. Tulosta ohjeet, joissa neuvotaan käyttäjää, miten olla vuorovaikutuksessa botin kanssa
2. Käynnistä silmukka
   1. Hyväksy käyttäjän syöte
   2. Jos käyttäjä pyytää lopettamaan, lopeta
   3. Käsittele käyttäjän syöte ja määritä vastaus (tässä tapauksessa vastaus on satunnainen valinta mahdollisten yleisten vastausten listasta)
   4. Tulosta vastaus
3. Palaa kohtaan 2

### Botin rakentaminen

Luodaan botti seuraavaksi. Aloitetaan määrittelemällä joitakin lauseita.

1. Luo tämä botti itse Pythonilla seuraavilla satunnaisilla vastauksilla:

    ```python
    random_responses = ["That is quite interesting, please tell me more.",
                        "I see. Do go on.",
                        "Why do you say that?",
                        "Funny weather we've been having, isn't it?",
                        "Let's change the subject.",
                        "Did you catch the game last night?"]
    ```

    Tässä on esimerkkituloste ohjeeksi (käyttäjän syöte alkaa `>`-merkillä):

    ```output
    Hello, I am Marvin, the simple robot.
    You can end this conversation at any time by typing 'bye'
    After typing each answer, press 'enter'
    How are you today?
    > I am good thanks
    That is quite interesting, please tell me more.
    > today I went for a walk     
    Did you catch the game last night?
    > I did, but my team lost
    Funny weather we've been having, isn't it?
    > yes but I hope next week is better
    Let's change the subject.
    > ok, lets talk about music
    Why do you say that?
    > because I like music!
    Why do you say that?
    > bye
    It was nice talking to you, goodbye!
    ```

    Yksi mahdollinen ratkaisu tehtävään löytyy [täältä](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/1-Introduction-to-NLP/solution/bot.py)

    ✅ Pysähdy ja pohdi

    1. Uskotko, että satunnaiset vastaukset voisivat "huijata" jonkun ajattelemaan, että botti todella ymmärtää häntä?
    2. Mitä ominaisuuksia botilla pitäisi olla, jotta se olisi tehokkaampi?
    3. Jos botti todella ymmärtäisi lauseen merkityksen, pitäisikö sen myös "muistaa" aiempien lauseiden merkitys keskustelussa?

---

## 🚀Haaste

Valitse yksi yllä olevista "pysähdy ja pohdi" -elementeistä ja yritä joko toteuttaa se koodissa tai kirjoittaa ratkaisu paperille pseudokoodina.

Seuraavassa oppitunnissa opit useista muista lähestymistavoista luonnollisen kielen jäsentämiseen ja koneoppimiseen.

## [Jälkikysely](https://ff-quizzes.netlify.app/en/ml/)

## Kertaus ja itseopiskelu

Tutustu alla oleviin viitteisiin lisälukumahdollisuuksina.

### Viitteet

1. Schubert, Lenhart, "Computational Linguistics", *The Stanford Encyclopedia of Philosophy* (Spring 2020 Edition), Edward N. Zalta (ed.), URL = <https://plato.stanford.edu/archives/spr2020/entries/computational-linguistics/>.
2. Princeton University "About WordNet." [WordNet](https://wordnet.princeton.edu/). Princeton University. 2010. 

## Tehtävä 

[Etsi botti](assignment.md)

---

**Vastuuvapauslauseke**:  
Tämä asiakirja on käännetty käyttämällä tekoälypohjaista käännöspalvelua [Co-op Translator](https://github.com/Azure/co-op-translator). Vaikka pyrimme tarkkuuteen, huomioithan, että automaattiset käännökset voivat sisältää virheitä tai epätarkkuuksia. Alkuperäistä asiakirjaa sen alkuperäisellä kielellä tulisi pitää ensisijaisena lähteenä. Kriittisen tiedon osalta suositellaan ammattimaista ihmiskäännöstä. Emme ole vastuussa väärinkäsityksistä tai virhetulkinnoista, jotka johtuvat tämän käännöksen käytöstä.