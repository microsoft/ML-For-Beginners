<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "5f3cb462e3122e1afe7ab0050ccf2bd3",
  "translation_date": "2025-09-05T01:23:13+00:00",
  "source_file": "6-NLP/2-Tasks/README.md",
  "language_code": "fi"
}
-->
# Yleisiä luonnollisen kielen käsittelyn tehtäviä ja tekniikoita

Useimmissa *luonnollisen kielen käsittelyn* tehtävissä käsiteltävä teksti täytyy pilkkoa, tutkia ja tallentaa tulokset tai verrata niitä sääntöihin ja tietokantoihin. Näiden tehtävien avulla ohjelmoija voi selvittää tekstin _merkityksen_, _tarkoituksen_ tai pelkästään _sanojen ja termien esiintymistiheyden_.

## [Esiluennon kysely](https://ff-quizzes.netlify.app/en/ml/)

Tutustutaan yleisiin tekniikoihin, joita käytetään tekstin käsittelyssä. Yhdistettynä koneoppimiseen nämä tekniikat auttavat analysoimaan suuria tekstimääriä tehokkaasti. Ennen kuin sovellamme koneoppimista näihin tehtäviin, on tärkeää ymmärtää ongelmat, joita NLP-asiantuntija kohtaa.

## NLP:n yleiset tehtävät

Tekstin analysointiin on monia eri tapoja. On olemassa tehtäviä, joita voit suorittaa, ja näiden tehtävien avulla voit ymmärtää tekstiä ja tehdä johtopäätöksiä. Yleensä nämä tehtävät suoritetaan tietyssä järjestyksessä.

### Tokenisointi

Todennäköisesti ensimmäinen asia, jonka useimmat NLP-algoritmit tekevät, on tekstin jakaminen tokeneiksi eli sanoiksi. Vaikka tämä kuulostaa yksinkertaiselta, välimerkkien ja eri kielten sanan- ja lauseenrajoittimien huomioiminen voi tehdä siitä haastavaa. Saatat joutua käyttämään erilaisia menetelmiä rajojen määrittämiseksi.

![tokenisointi](../../../../6-NLP/2-Tasks/images/tokenization.png)
> Tokenisointi lauseesta **Ylpeys ja ennakkoluulo**. Infografiikka: [Jen Looper](https://twitter.com/jenlooper)

### Upotukset

[Sana-upotukset](https://wikipedia.org/wiki/Word_embedding) ovat tapa muuttaa tekstidata numeeriseen muotoon. Upotukset tehdään siten, että samankaltaista merkitystä omaavat tai yhdessä käytetyt sanat ryhmittyvät yhteen.

![sana-upotukset](../../../../6-NLP/2-Tasks/images/embedding.png)
> "Kunnioitan suuresti hermojasi, ne ovat vanhoja ystäviäni." - Sana-upotukset lauseesta **Ylpeys ja ennakkoluulo**. Infografiikka: [Jen Looper](https://twitter.com/jenlooper)

✅ Kokeile [tätä mielenkiintoista työkalua](https://projector.tensorflow.org/) tutkiaksesi sana-upotuksia. Klikkaamalla yhtä sanaa näet samankaltaisten sanojen ryhmiä: 'toy' ryhmittyy sanojen 'disney', 'lego', 'playstation' ja 'console' kanssa.

### Jäsennys ja sanaluokkien tunnistus

Jokainen tokenisoitu sana voidaan merkitä sanaluokaksi, kuten substantiiviksi, verbiksi tai adjektiiviksi. Lause `the quick red fox jumped over the lazy brown dog` voidaan merkitä sanaluokittain esimerkiksi fox = substantiivi, jumped = verbi.

![jäsennys](../../../../6-NLP/2-Tasks/images/parse.png)

> Lauseen jäsennys **Ylpeys ja ennakkoluulo**. Infografiikka: [Jen Looper](https://twitter.com/jenlooper)

Jäsennys tarkoittaa sanojen välisten suhteiden tunnistamista lauseessa – esimerkiksi `the quick red fox jumped` on adjektiivi-substantiivi-verbi-sekvenssi, joka on erillinen `lazy brown dog` -sekvenssistä.

### Sanojen ja fraasien esiintymistiheys

Kun analysoidaan suurta tekstimassaa, on hyödyllistä rakentaa sanakirja, joka sisältää kaikki kiinnostavat sanat tai fraasit ja niiden esiintymistiheyden. Lauseessa `the quick red fox jumped over the lazy brown dog` sanan the esiintymistiheys on 2.

Tarkastellaan esimerkkitekstiä, jossa lasketaan sanojen esiintymistiheys. Rudyard Kiplingin runo The Winners sisältää seuraavan säkeen:

```output
What the moral? Who rides may read.
When the night is thick and the tracks are blind
A friend at a pinch is a friend, indeed,
But a fool to wait for the laggard behind.
Down to Gehenna or up to the Throne,
He travels the fastest who travels alone.
```

Koska fraasien esiintymistiheys voi olla kirjainkoolle herkkä tai herkkä, fraasin `a friend` esiintymistiheys on 2, `the` esiintymistiheys on 6 ja `travels` esiintymistiheys on 2.

### N-grammit

Teksti voidaan jakaa tietyn pituisiksi sanasekvensseiksi: yksi sana (unigrammi), kaksi sanaa (bigrammi), kolme sanaa (trigrammi) tai mikä tahansa määrä sanoja (n-grammi).

Esimerkiksi lause `the quick red fox jumped over the lazy brown dog` n-grammiarvolla 2 tuottaa seuraavat n-grammit:

1. the quick 
2. quick red 
3. red fox
4. fox jumped 
5. jumped over 
6. over the 
7. the lazy 
8. lazy brown 
9. brown dog

Tätä voi olla helpompi visualisoida liukuvana laatikkona lauseen päällä. Tässä esimerkki 3 sanan n-grammeista, n-grammi on lihavoitu jokaisessa lauseessa:

1.   <u>**the quick red**</u> fox jumped over the lazy brown dog
2.   the **<u>quick red fox</u>** jumped over the lazy brown dog
3.   the quick **<u>red fox jumped</u>** over the lazy brown dog
4.   the quick red **<u>fox jumped over</u>** the lazy brown dog
5.   the quick red fox **<u>jumped over the</u>** lazy brown dog
6.   the quick red fox jumped **<u>over the lazy</u>** brown dog
7.   the quick red fox jumped over <u>**the lazy brown**</u> dog
8.   the quick red fox jumped over the **<u>lazy brown dog</u>**

![n-grammit liukuva ikkuna](../../../../6-NLP/2-Tasks/images/n-grams.gif)

> N-grammiarvo 3: Infografiikka: [Jen Looper](https://twitter.com/jenlooper)

### Substantiivifraasien tunnistus

Useimmissa lauseissa on substantiivi, joka toimii subjektina tai objektina. Englannissa sen voi usein tunnistaa sanoista 'a', 'an' tai 'the', jotka edeltävät sitä. Subjektin tai objektin tunnistaminen lauseesta 'substantiivifraasin tunnistamisella' on yleinen tehtävä NLP:ssä, kun pyritään ymmärtämään lauseen merkitystä.

✅ Lauseessa "I cannot fix on the hour, or the spot, or the look or the words, which laid the foundation. It is too long ago. I was in the middle before I knew that I had begun." voitko tunnistaa substantiivifraasit?

Lauseessa `the quick red fox jumped over the lazy brown dog` on 2 substantiivifraasia: **quick red fox** ja **lazy brown dog**.

### Tunteiden analysointi

Lause tai teksti voidaan analysoida sen tunnesävyn perusteella, eli kuinka *positiivinen* tai *negatiivinen* se on. Tunnesävyä mitataan *polariteetilla* ja *objektiivisuudella/subjektiivisuudella*. Polariteetti mitataan välillä -1.0–1.0 (negatiivisesta positiiviseen) ja 0.0–1.0 (objektiivisimmasta subjektiivisimpaan).

✅ Myöhemmin opit, että tunnesävyn määrittämiseen on erilaisia tapoja koneoppimisen avulla, mutta yksi tapa on käyttää listaa sanoista ja fraaseista, jotka ihmisen asiantuntija on luokitellut positiivisiksi tai negatiivisiksi, ja soveltaa tätä mallia tekstiin polariteettipisteen laskemiseksi. Näetkö, miten tämä toimisi joissakin tilanteissa ja vähemmän hyvin toisissa?

### Taivutus

Taivutus mahdollistaa sanan muuttamisen yksikkö- tai monikkomuotoon.

### Lemmatisaatio

*Lemma* on sanan perusmuoto tai kantasana, esimerkiksi *flew*, *flies*, *flying* ovat verbin *fly* lemma.

NLP-tutkijalle on myös saatavilla hyödyllisiä tietokantoja, kuten:

### WordNet

[WordNet](https://wordnet.princeton.edu/) on tietokanta, joka sisältää sanoja, synonyymejä, antonyymejä ja monia muita yksityiskohtia eri kielten sanoista. Se on erittäin hyödyllinen, kun pyritään rakentamaan käännöksiä, oikeinkirjoituksen tarkistimia tai minkä tahansa tyyppisiä kielityökaluja.

## NLP-kirjastot

Onneksi sinun ei tarvitse rakentaa kaikkia näitä tekniikoita itse, sillä saatavilla on erinomaisia Python-kirjastoja, jotka tekevät NLP:stä paljon helpommin lähestyttävää kehittäjille, jotka eivät ole erikoistuneet luonnollisen kielen käsittelyyn tai koneoppimiseen. Seuraavissa oppitunneissa on lisää esimerkkejä näistä, mutta tässä opit joitakin hyödyllisiä esimerkkejä seuraavaa tehtävää varten.

### Harjoitus - `TextBlob`-kirjaston käyttö

Käytetään kirjastoa nimeltä TextBlob, sillä se sisältää hyödyllisiä API-rajapintoja näiden tehtävien käsittelyyn. TextBlob "perustuu [NLTK](https://nltk.org)- ja [pattern](https://github.com/clips/pattern)-kirjastojen vahvuuksiin ja toimii hyvin molempien kanssa." Sen API sisältää huomattavan määrän koneoppimista.

> Huom: Hyödyllinen [Quick Start](https://textblob.readthedocs.io/en/dev/quickstart.html#quickstart) -opas on saatavilla TextBlobille, ja sitä suositellaan kokeneille Python-kehittäjille.

Kun yritetään tunnistaa *substantiivifraaseja*, TextBlob tarjoaa useita vaihtoehtoja fraasien tunnistamiseen.

1. Tutustu `ConllExtractor`-luokkaan.

    ```python
    from textblob import TextBlob
    from textblob.np_extractors import ConllExtractor
    # import and create a Conll extractor to use later 
    extractor = ConllExtractor()
    
    # later when you need a noun phrase extractor:
    user_input = input("> ")
    user_input_blob = TextBlob(user_input, np_extractor=extractor)  # note non-default extractor specified
    np = user_input_blob.noun_phrases                                    
    ```

    > Mitä tässä tapahtuu? [ConllExtractor](https://textblob.readthedocs.io/en/dev/api_reference.html?highlight=Conll#textblob.en.np_extractors.ConllExtractor) on "substantiivifraasien tunnistaja, joka käyttää chunk-jäsennystä, joka on koulutettu ConLL-2000-koulutusdatalla." ConLL-2000 viittaa vuoden 2000 Computational Natural Language Learning -konferenssiin. Jokaisena vuonna konferenssi järjesti työpajan, jossa käsiteltiin haastavaa NLP-ongelmaa, ja vuonna 2000 se oli substantiivifraasien tunnistus. Malli koulutettiin Wall Street Journalin datalla, jossa "osat 15-18 toimivat koulutusdatana (211727 tokenia) ja osa 20 testidatana (47377 tokenia)". Voit tutustua käytettyihin menetelmiin [täällä](https://www.clips.uantwerpen.be/conll2000/chunking/) ja [tuloksiin](https://ifarm.nl/erikt/research/np-chunking.html).

### Haaste - paranna bottiasi NLP:n avulla

Edellisessä oppitunnissa rakensit hyvin yksinkertaisen kysymys-vastausbotin. Nyt teet Marvinista hieman empaattisemman analysoimalla syötteesi tunnesävyn ja tulostamalla vastauksen, joka vastaa tunnesävyä. Sinun täytyy myös tunnistaa `substantiivifraasi` ja kysyä siitä lisää.

Askeleet paremman keskustelubotin rakentamiseksi:

1. Tulosta ohjeet, jotka neuvovat käyttäjää, miten botin kanssa voi keskustella
2. Aloita silmukka 
   1. Hyväksy käyttäjän syöte
   2. Jos käyttäjä haluaa lopettaa, lopeta
   3. Käsittele käyttäjän syöte ja määritä sopiva tunnesävyn vastaus
   4. Jos tunnesävyssä havaitaan substantiivifraasi, monikkomuotoile se ja kysy lisää aiheesta
   5. Tulosta vastaus
3. Palaa kohtaan 2

Tässä on koodinpätkä tunnesävyn määrittämiseksi TextBlobin avulla. Huomaa, että tunnesävyn vastauksia on vain neljä *sävyä* (voit lisätä enemmän, jos haluat):

```python
if user_input_blob.polarity <= -0.5:
  response = "Oh dear, that sounds bad. "
elif user_input_blob.polarity <= 0:
  response = "Hmm, that's not great. "
elif user_input_blob.polarity <= 0.5:
  response = "Well, that sounds positive. "
elif user_input_blob.polarity <= 1:
  response = "Wow, that sounds great. "
```

Tässä on esimerkkitulostus ohjeeksi (käyttäjän syöte alkaa rivillä, jossa on >):

```output
Hello, I am Marvin, the friendly robot.
You can end this conversation at any time by typing 'bye'
After typing each answer, press 'enter'
How are you today?
> I am ok
Well, that sounds positive. Can you tell me more?
> I went for a walk and saw a lovely cat
Well, that sounds positive. Can you tell me more about lovely cats?
> cats are the best. But I also have a cool dog
Wow, that sounds great. Can you tell me more about cool dogs?
> I have an old hounddog but he is sick
Hmm, that's not great. Can you tell me more about old hounddogs?
> bye
It was nice talking to you, goodbye!
```

Yksi mahdollinen ratkaisu tehtävään löytyy [täältä](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/2-Tasks/solution/bot.py)

✅ Tietotesti

1. Uskotko, että empaattiset vastaukset voisivat 'huijata' jonkun ajattelemaan, että botti todella ymmärtää heitä?
2. Tekevätkö substantiivifraasin tunnistaminen botista 'uskottavamman'?
3. Miksi substantiivifraasin tunnistaminen lauseesta olisi hyödyllistä?

---

Toteuta botti edellisessä tietotestissä ja testaa sitä ystävälläsi. Voiko se huijata heitä? Voitko tehdä botista 'uskottavamman'?

## 🚀Haaste

Valitse tehtävä edellisestä tietotestistä ja yritä toteuttaa se. Testaa bottia ystävälläsi. Voiko se huijata heitä? Voitko tehdä botista 'uskottavamman'?

## [Jälkiluennon kysely](https://ff-quizzes.netlify.app/en/ml/)

## Kertaus ja itseopiskelu

Seuraavissa oppitunneissa opit lisää tunnesävyn analysoinnista. Tutki tätä mielenkiintoista tekniikkaa esimerkiksi [KDNuggets](https://www.kdnuggets.com/tag/nlp)-artikkeleista.

## Tehtävä 

[Saata botti keskustelemaan](assignment.md)

---

**Vastuuvapauslauseke**:  
Tämä asiakirja on käännetty käyttämällä tekoälypohjaista käännöspalvelua [Co-op Translator](https://github.com/Azure/co-op-translator). Vaikka pyrimme tarkkuuteen, huomioithan, että automaattiset käännökset voivat sisältää virheitä tai epätarkkuuksia. Alkuperäistä asiakirjaa sen alkuperäisellä kielellä tulisi pitää ensisijaisena lähteenä. Kriittisen tiedon osalta suositellaan ammattimaista ihmiskäännöstä. Emme ole vastuussa väärinkäsityksistä tai virhetulkinnoista, jotka johtuvat tämän käännöksen käytöstä.