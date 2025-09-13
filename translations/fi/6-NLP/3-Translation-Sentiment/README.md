<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "be03c8182982b87ced155e4e9d1438e8",
  "translation_date": "2025-09-05T01:40:08+00:00",
  "source_file": "6-NLP/3-Translation-Sentiment/README.md",
  "language_code": "fi"
}
-->
# Käännös ja sentimenttianalyysi koneoppimisen avulla

Aiemmissa oppitunneissa opit rakentamaan yksinkertaisen botin käyttämällä `TextBlob`-kirjastoa, joka hyödyntää koneoppimista taustalla suorittaakseen perusluonteisia luonnollisen kielen käsittelytehtäviä, kuten substantiivilauseiden tunnistamista. Toinen tärkeä haaste laskennallisessa kielitieteessä on lauseen tarkka _kääntäminen_ yhdestä puhutuista tai kirjoitetuista kielistä toiseen.

## [Ennakkokysely](https://ff-quizzes.netlify.app/en/ml/)

Kääntäminen on erittäin vaikea ongelma, jota monimutkaistaa tuhansien kielten olemassaolo ja niiden hyvin erilaiset kielioppisäännöt. Yksi lähestymistapa on muuntaa yhden kielen, kuten englannin, muodolliset kielioppisäännöt kieleen riippumattomaan rakenteeseen ja sitten kääntää se muuntamalla takaisin toiseen kieleen. Tämä lähestymistapa tarkoittaa seuraavia vaiheita:

1. **Tunnistaminen**. Tunnista tai merkitse syöttökielen sanat substantiiveiksi, verbeiksi jne.
2. **Käännöksen luominen**. Tuota suora käännös jokaisesta sanasta kohdekielen muodossa.

### Esimerkkilause, englannista iiriin

Englanniksi lause _I feel happy_ koostuu kolmesta sanasta seuraavassa järjestyksessä:

- **subjekti** (I)
- **verbi** (feel)
- **adjektiivi** (happy)

Kuitenkin iirin kielessä sama lause noudattaa hyvin erilaista kielioppirakennetta – tunteet kuten "*happy*" tai "*sad*" ilmaistaan olevan *jonkun päällä*.

Englanninkielinen lause `I feel happy` olisi iiriksi `Tá athas orm`. *Kirjaimellinen* käännös olisi `Happy is upon me`.

Iiriä puhuva henkilö, joka kääntää englanniksi, sanoisi `I feel happy`, ei `Happy is upon me`, koska hän ymmärtää lauseen merkityksen, vaikka sanat ja lauserakenne olisivat erilaisia.

Iirin kielen muodollinen järjestys lauseelle on:

- **verbi** (Tá eli is)
- **adjektiivi** (athas eli happy)
- **subjekti** (orm eli upon me)

## Kääntäminen

Naivi käännösohjelma saattaisi kääntää vain sanat, jättäen huomiotta lauserakenteen.

✅ Jos olet oppinut toisen (tai kolmannen tai useamman) kielen aikuisena, olet saattanut aloittaa ajattelemalla omalla äidinkielelläsi, kääntämällä käsitteen sana sanalta päässäsi toiselle kielelle ja sitten puhumalla käännöksesi. Tämä on samanlaista kuin mitä naivit käännösohjelmat tekevät. On tärkeää päästä tämän vaiheen yli, jotta saavutetaan sujuvuus!

Naivi kääntäminen johtaa huonoihin (ja joskus huvittaviin) virhekäännöksiin: `I feel happy` kääntyy kirjaimellisesti `Mise bhraitheann athas` iiriksi. Tämä tarkoittaa (kirjaimellisesti) `me feel happy` eikä ole kelvollinen iirinkielinen lause. Vaikka englanti ja iiri ovat kieliä, joita puhutaan kahdella lähekkäin sijaitsevalla saarella, ne ovat hyvin erilaisia kieliä, joilla on erilaiset kielioppirakenteet.

> Voit katsoa joitakin videoita iirin kielellisistä perinteistä, kuten [tämän](https://www.youtube.com/watch?v=mRIaLSdRMMs)

### Koneoppimisen lähestymistavat

Tähän mennessä olet oppinut muodollisten sääntöjen lähestymistavasta luonnollisen kielen käsittelyyn. Toinen lähestymistapa on jättää sanojen merkitys huomiotta ja _sen sijaan käyttää koneoppimista havaitsemaan kaavoja_. Tämä voi toimia kääntämisessä, jos sinulla on paljon tekstiä (*korpus*) tai tekstejä (*korpukset*) sekä alkuperäisellä että kohdekielellä.

Esimerkiksi, ajatellaan *Ylpeys ja ennakkoluulo* -teosta, tunnettua englantilaista romaania, jonka Jane Austen kirjoitti vuonna 1813. Jos tarkastelet kirjaa englanniksi ja sen ihmisen tekemää käännöstä *ranskaksi*, voisit havaita lauseita, jotka ovat _idiomaattisesti_ käännettyjä toiselle kielelle. Teet tämän kohta.

Esimerkiksi, kun englanninkielinen lause `I have no money` käännetään kirjaimellisesti ranskaksi, siitä saattaa tulla `Je n'ai pas de monnaie`. "Monnaie" on hankala ranskalainen 'väärä ystävä', sillä 'money' ja 'monnaie' eivät ole synonyymejä. Parempi käännös, jonka ihminen voisi tehdä, olisi `Je n'ai pas d'argent`, koska se välittää paremmin merkityksen, että sinulla ei ole rahaa (eikä 'pikkurahaa', joka on 'monnaie'-sanan merkitys).

![monnaie](../../../../6-NLP/3-Translation-Sentiment/images/monnaie.png)

> Kuva: [Jen Looper](https://twitter.com/jenlooper)

Jos koneoppimismalli saa tarpeeksi ihmisen tekemiä käännöksiä, joiden perusteella rakentaa mallin, se voi parantaa käännösten tarkkuutta tunnistamalla yleisiä kaavoja teksteissä, jotka asiantuntijakielten puhujat ovat aiemmin kääntäneet.

### Harjoitus - kääntäminen

Voit käyttää `TextBlob`-kirjastoa lauseiden kääntämiseen. Kokeile kuuluisan ensimmäisen lauseen kääntämistä **Ylpeys ja ennakkoluulo** -teoksesta:

```python
from textblob import TextBlob

blob = TextBlob(
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife!"
)
print(blob.translate(to="fr"))

```

`TextBlob` tekee melko hyvän käännöksen: "C'est une vérité universellement reconnue, qu'un homme célibataire en possession d'une bonne fortune doit avoir besoin d'une femme!". 

Voidaan väittää, että TextBlobin käännös on itse asiassa paljon tarkempi kuin kirjan vuoden 1932 ranskankielinen käännös, jonka tekivät V. Leconte ja Ch. Pressoir:

"C'est une vérité universelle qu'un célibataire pourvu d'une belle fortune doit avoir envie de se marier, et, si peu que l'on sache de son sentiment à cet egard, lorsqu'il arrive dans une nouvelle résidence, cette idée est si bien fixée dans l'esprit de ses voisins qu'ils le considèrent sur-le-champ comme la propriété légitime de l'une ou l'autre de leurs filles."

Tässä tapauksessa koneoppimisen avulla tehty käännös tekee parempaa työtä kuin ihmiskääntäjä, joka tarpeettomasti lisää sanoja alkuperäisen kirjoittajan suuhun selkeyden vuoksi.

> Mikä tässä tapahtuu? Ja miksi TextBlob on niin hyvä kääntämisessä? No, taustalla se käyttää Google Translatea, kehittynyttä tekoälyä, joka pystyy käsittelemään miljoonia lauseita ennustaakseen parhaita merkkijonoja tehtävään. Tässä ei tapahdu mitään manuaalista, ja tarvitset internetyhteyden käyttääksesi `blob.translate`.

✅ Kokeile lisää lauseita. Kumpi on parempi, koneoppiminen vai ihmiskäännös? Missä tapauksissa?

## Sentimenttianalyysi

Toinen alue, jossa koneoppiminen voi toimia erittäin hyvin, on sentimenttianalyysi. Ei-koneoppimiseen perustuva lähestymistapa sentimenttiin on tunnistaa sanat ja lauseet, jotka ovat 'positiivisia' ja 'negatiivisia'. Sitten, kun annetaan uusi tekstikappale, lasketaan positiivisten, negatiivisten ja neutraalien sanojen kokonaisarvo sentimentin määrittämiseksi.

Tämä lähestymistapa on helposti huijattavissa, kuten olet saattanut huomata Marvin-tehtävässä – lause `Great, that was a wonderful waste of time, I'm glad we are lost on this dark road` on sarkastinen, negatiivinen sentimenttilause, mutta yksinkertainen algoritmi tunnistaa 'great', 'wonderful', 'glad' positiivisiksi ja 'waste', 'lost' ja 'dark' negatiivisiksi. Kokonaisvaltainen sentimentti kallistuu näiden ristiriitaisten sanojen vuoksi.

✅ Pysähdy hetkeksi ja mieti, miten me ihmiset välitämme sarkasmia puhuessamme. Äänenpainolla on suuri merkitys. Kokeile sanoa lause "Well, that film was awesome" eri tavoilla ja huomaa, miten äänesi välittää merkityksen.

### Koneoppimisen lähestymistavat

Koneoppimisen lähestymistapa olisi kerätä manuaalisesti negatiivisia ja positiivisia tekstikappaleita – twiittejä, elokuva-arvosteluja tai mitä tahansa, missä ihminen on antanut pisteet *ja* kirjallisen mielipiteen. Sitten NLP-tekniikoita voidaan soveltaa mielipiteisiin ja pisteisiin, jotta kaavat tulevat esiin (esim. positiivisissa elokuva-arvosteluissa esiintyy usein lause 'Oscar worthy' enemmän kuin negatiivisissa arvosteluissa, tai positiivisissa ravintola-arvosteluissa sanotaan 'gourmet' paljon useammin kuin 'disgusting').

> ⚖️ **Esimerkki**: Jos työskentelisit poliitikon toimistossa ja jokin uusi laki olisi keskustelun alla, äänestäjät saattaisivat kirjoittaa toimistoon sähköposteja, jotka tukevat tai vastustavat kyseistä uutta lakia. Oletetaan, että sinut on määrätty lukemaan sähköpostit ja lajittelemaan ne kahteen pinoon, *puolesta* ja *vastaan*. Jos sähköposteja olisi paljon, voisit tuntea olosi ylivoimaiseksi yrittäessäsi lukea ne kaikki. Eikö olisi mukavaa, jos botti voisi lukea ne kaikki puolestasi, ymmärtää ne ja kertoa, mihin pinoon kukin sähköposti kuuluu? 
> 
> Yksi tapa saavuttaa tämä on käyttää koneoppimista. Voisit kouluttaa mallin osalla *vastaan* olevista sähköposteista ja osalla *puolesta* olevista sähköposteista. Malli yhdistäisi tietyt lauseet ja sanat todennäköisemmin vastaan- tai puolesta-sähköposteihin, *mutta se ei ymmärtäisi mitään sisällöstä*, vain että tietyt sanat ja kaavat esiintyvät todennäköisemmin vastaan- tai puolesta-sähköposteissa. Voisit testata sitä joillakin sähköposteilla, joita et ollut käyttänyt mallin kouluttamiseen, ja nähdä, päätyisikö se samaan johtopäätökseen kuin sinä. Kun olisit tyytyväinen mallin tarkkuuteen, voisit käsitellä tulevia sähköposteja lukematta jokaista erikseen.

✅ Kuulostaako tämä prosessi samalta kuin prosessit, joita olet käyttänyt aiemmissa oppitunneissa?

## Harjoitus - sentimenttiset lauseet

Sentimentti mitataan *polariteetilla* välillä -1–1, mikä tarkoittaa, että -1 on kaikkein negatiivisin sentimentti ja 1 kaikkein positiivisin. Sentimentti mitataan myös 0–1 asteikolla objektiivisuudelle (0) ja subjektiivisuudelle (1).

Tarkastellaan uudelleen Jane Austenin *Ylpeys ja ennakkoluulo* -teosta. Teksti on saatavilla täällä: [Project Gutenberg](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm). Alla oleva esimerkki näyttää lyhyen ohjelman, joka analysoi kirjan ensimmäisen ja viimeisen lauseen sentimentin ja näyttää sen polariteetin sekä subjektiivisuus/objektiivisuus-pisteet.

Sinun tulisi käyttää `TextBlob`-kirjastoa (kuvattu yllä) sentimentin määrittämiseen (sinun ei tarvitse kirjoittaa omaa sentimenttilaskuria) seuraavassa tehtävässä.

```python
from textblob import TextBlob

quote1 = """It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife."""

quote2 = """Darcy, as well as Elizabeth, really loved them; and they were both ever sensible of the warmest gratitude towards the persons who, by bringing her into Derbyshire, had been the means of uniting them."""

sentiment1 = TextBlob(quote1).sentiment
sentiment2 = TextBlob(quote2).sentiment

print(quote1 + " has a sentiment of " + str(sentiment1))
print(quote2 + " has a sentiment of " + str(sentiment2))
```

Näet seuraavan tulosteen:

```output
It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want # of a wife. has a sentiment of Sentiment(polarity=0.20952380952380953, subjectivity=0.27142857142857146)

Darcy, as well as Elizabeth, really loved them; and they were
     both ever sensible of the warmest gratitude towards the persons
      who, by bringing her into Derbyshire, had been the means of
      uniting them. has a sentiment of Sentiment(polarity=0.7, subjectivity=0.8)
```

## Haaste - tarkista sentimentin polariteetti

Tehtäväsi on selvittää sentimentin polariteetin avulla, onko *Ylpeys ja ennakkoluulo* -teoksessa enemmän ehdottoman positiivisia lauseita kuin ehdottoman negatiivisia. Tässä tehtävässä voit olettaa, että polariteettipisteet 1 tai -1 ovat ehdottoman positiivisia tai negatiivisia.

**Vaiheet:**

1. Lataa [kopio Ylpeys ja ennakkoluulo -teoksesta](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm) Project Gutenbergista .txt-tiedostona. Poista metatiedot tiedoston alusta ja lopusta, jättäen vain alkuperäisen tekstin
2. Avaa tiedosto Pythonissa ja pura sisältö merkkijonona
3. Luo TextBlob kirjan merkkijonosta
4. Analysoi kirjan jokainen lause silmukassa
   1. Jos polariteetti on 1 tai -1, tallenna lause positiivisten tai negatiivisten viestien taulukkoon tai listaan
5. Lopuksi tulosta kaikki positiiviset ja negatiiviset lauseet (erikseen) ja niiden lukumäärä.

Tässä on esimerkkiratkaisu: [ratkaisu](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/3-Translation-Sentiment/solution/notebook.ipynb).

✅ Tietotarkistus

1. Sentimentti perustuu lauseessa käytettyihin sanoihin, mutta ymmärtääkö koodi *sanat*?
2. Onko mielestäsi sentimentin polariteetti tarkka, eli toisin sanoen, oletko *samaa mieltä* pisteiden kanssa?
   1. Erityisesti, oletko samaa mieltä tai eri mieltä seuraavien lauseiden ehdottoman **positiivisesta** polariteetista?
      * “What an excellent father you have, girls!” said she, when the door was shut.
      * “Your examination of Mr. Darcy is over, I presume,” said Miss Bingley; “and pray what is the result?” “I am perfectly convinced by it that Mr. Darcy has no defect.
      * How wonderfully these sort of things occur!
      * I have the greatest dislike in the world to that sort of thing.
      * Charlotte is an excellent manager, I dare say.
      * “This is delightful indeed!
      * I am so happy!
      * Your idea of the ponies is delightful.
   2. Seuraavat 3 lausetta saivat ehdottoman positiivisen sentimentin, mutta tarkemmin luettuna ne eivät ole positiivisia lauseita. Miksi sentimenttianalyysi ajatteli niiden olevan positiivisia lauseita?
      * Happy shall I be, when his stay at Netherfield is over!” “I wish I could say anything to comfort you,” replied Elizabeth; “but it is wholly out of my power.
      * If I could but see you as happy!
      * Our distress, my dear Lizzy, is very great.
   3. Oletko samaa mieltä tai eri mieltä seuraavien lauseiden ehdottoman **negatiivisesta** polariteetista?
      - Everybody is disgusted with his pride.
      - “I should like to know how he behaves among strangers.” “You shall hear then—but prepare yourself for something very dreadful.
      - The pause was to Elizabeth’s feelings dreadful.
      - It would be dreadful!

✅ Jane Austenin teosten tuntija ymmärtää, että hän käyttää usein kirjojaan kritisoidakseen Englannin regency-kauden yhteiskunnan naurettavimpia puolia. Elizabeth Bennett, *Ylpeys ja ennakkoluulo* -teoksen päähenkilö, on tarkka sosiaalinen havainnoija (kuten kirjailija itse), ja hänen kielensä on usein voimakkaasti vivahteikasta. Jopa Mr. Darcy (tarinan rakkauden kohde) huomauttaa Elizabethin leikkisästä ja kiusoittelevasta kielenkäytöstä: "I have had the pleasure of your acquaintance long enough to know that you find great enjoyment in occasionally professing opinions which in fact are not your own."

---

## 🚀Haaste

Voitko tehdä Marvinista vielä paremman ottamalla käyttäjän syötteestä muita ominaisuuksia?

## [Jälkikysely](https://ff-quizzes.netlify.app/en/ml/)

## Kertaus ja itseopiskelu
Tekstin sentimentin analysointiin on monia tapoja. Mieti liiketoimintasovelluksia, jotka voisivat hyödyntää tätä tekniikkaa. Pohdi myös, miten se voi mennä pieleen. Lue lisää kehittyneistä yrityskäyttöön tarkoitetuista järjestelmistä, jotka analysoivat sentimenttiä, kuten [Azure Text Analysis](https://docs.microsoft.com/azure/cognitive-services/Text-Analytics/how-tos/text-analytics-how-to-sentiment-analysis?tabs=version-3-1?WT.mc_id=academic-77952-leestott). Testaa joitakin yllä olevia Ylpeys ja ennakkoluulo -lauseita ja katso, pystyykö järjestelmä havaitsemaan vivahteita.

## Tehtävä

[Runoilijan vapaus](assignment.md)

---

**Vastuuvapauslauseke**:  
Tämä asiakirja on käännetty käyttämällä tekoälypohjaista käännöspalvelua [Co-op Translator](https://github.com/Azure/co-op-translator). Vaikka pyrimme tarkkuuteen, huomioithan, että automaattiset käännökset voivat sisältää virheitä tai epätarkkuuksia. Alkuperäinen asiakirja sen alkuperäisellä kielellä tulisi pitää ensisijaisena lähteenä. Kriittisen tiedon osalta suositellaan ammattimaista ihmiskäännöstä. Emme ole vastuussa väärinkäsityksistä tai virhetulkinnoista, jotka johtuvat tämän käännöksen käytöstä.