<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9d91f3af3758fdd4569fb410575995ef",
  "translation_date": "2025-09-05T00:28:39+00:00",
  "source_file": "1-Introduction/4-techniques-of-ML/README.md",
  "language_code": "fi"
}
-->
# Koneoppimisen tekniikat

Koneoppimismallien ja niiden käyttämän datan rakentaminen, käyttäminen ja ylläpito eroaa merkittävästi monista muista kehitysprosesseista. Tässä oppitunnissa selvitämme prosessin ja hahmotamme tärkeimmät tekniikat, jotka sinun tulee hallita. Opit:

- Ymmärtämään koneoppimisen taustalla olevat prosessit yleisellä tasolla.
- Tutustumaan peruskäsitteisiin, kuten "mallit", "ennusteet" ja "opetusdata".

## [Esiluennon kysely](https://ff-quizzes.netlify.app/en/ml/)

[![ML for beginners - Techniques of Machine Learning](https://img.youtube.com/vi/4NGM0U2ZSHU/0.jpg)](https://youtu.be/4NGM0U2ZSHU "ML for beginners - Techniques of Machine Learning")

> 🎥 Klikkaa yllä olevaa kuvaa lyhyen videon katsomiseksi, jossa käydään läpi tämän oppitunnin sisältöä.

## Johdanto

Koneoppimisen (ML) prosessien luominen koostuu useista vaiheista:

1. **Määrittele kysymys**. Useimmat ML-prosessit alkavat kysymyksestä, johon ei voida vastata yksinkertaisella ehdollisella ohjelmalla tai sääntöpohjaisella moottorilla. Nämä kysymykset liittyvät usein ennusteisiin, jotka perustuvat datakokoelmaan.
2. **Kerää ja valmistele data**. Kysymykseen vastaaminen vaatii dataa. Datan laatu ja joskus myös määrä määrittävät, kuinka hyvin voit vastata alkuperäiseen kysymykseesi. Datan visualisointi on tärkeä osa tätä vaihetta. Tämä vaihe sisältää myös datan jakamisen opetus- ja testiryhmiin mallin rakentamista varten.
3. **Valitse opetusmenetelmä**. Kysymyksesi ja datasi luonteen perusteella sinun tulee valita, miten haluat opettaa mallia, jotta se parhaiten heijastaisi dataasi ja tekisi tarkkoja ennusteita. Tämä ML-prosessin osa vaatii erityistä asiantuntemusta ja usein huomattavan määrän kokeilua.
4. **Opeta malli**. Käyttämällä opetusdataa käytät erilaisia algoritmeja opettaaksesi mallin tunnistamaan datan kuvioita. Malli voi hyödyntää sisäisiä painotuksia, joita voidaan säätää korostamaan tiettyjä datan osia paremman mallin rakentamiseksi.
5. **Arvioi malli**. Käytä aiemmin näkemätöntä dataa (testidatasi) arvioidaksesi, kuinka hyvin malli toimii.
6. **Parametrien säätö**. Mallin suorituskyvyn perusteella voit tehdä prosessin uudelleen käyttämällä erilaisia parametreja tai muuttujia, jotka ohjaavat mallin opetusalgoritmien toimintaa.
7. **Ennusta**. Käytä uusia syötteitä testataksesi mallisi tarkkuutta.

## Mitä kysymystä kysyä

Tietokoneet ovat erityisen taitavia löytämään piilotettuja kuvioita datasta. Tämä ominaisuus on erittäin hyödyllinen tutkijoille, joilla on kysymyksiä tietystä aihealueesta, joihin ei voida helposti vastata luomalla ehdollisuuspohjainen sääntömoottori. Esimerkiksi vakuutusmatemaattisessa tehtävässä data-analyytikko voisi rakentaa käsintehtyjä sääntöjä tupakoitsijoiden ja ei-tupakoitsijoiden kuolleisuudesta.

Kun mukaan tuodaan monia muita muuttujia, ML-malli voi kuitenkin osoittautua tehokkaammaksi ennustamaan tulevia kuolleisuuslukuja aiemman terveystiedon perusteella. Iloisempi esimerkki voisi olla sääennusteiden tekeminen huhtikuulle tietyssä paikassa datan perusteella, joka sisältää leveys- ja pituusasteet, ilmastonmuutoksen, etäisyyden merestä, suihkuvirtauksen kuviot ja paljon muuta.

✅ Tämä [esitysmateriaali](https://www2.cisl.ucar.edu/sites/default/files/2021-10/0900%20June%2024%20Haupt_0.pdf) säämalleista tarjoaa historiallisen näkökulman ML:n käytöstä sääanalyysissä.  

## Ennen mallin rakentamista

Ennen kuin aloitat mallin rakentamisen, sinun tulee suorittaa useita tehtäviä. Testataksesi kysymystäsi ja muodostaaksesi hypoteesin mallin ennusteiden perusteella, sinun tulee tunnistaa ja määrittää useita elementtejä.

### Data

Jotta voit vastata kysymykseesi minkäänlaisella varmuudella, tarvitset riittävän määrän oikeanlaista dataa. Tässä vaiheessa sinun tulee tehdä kaksi asiaa:

- **Kerää data**. Muista aiemman oppitunnin oikeudenmukaisuudesta data-analyysissä, kerää datasi huolellisesti. Ole tietoinen datan lähteistä, mahdollisista sisäisistä ennakkoluuloista ja dokumentoi sen alkuperä.
- **Valmistele data**. Datavalmisteluprosessissa on useita vaiheita. Saatat joutua yhdistämään dataa ja normalisoimaan sen, jos se tulee eri lähteistä. Voit parantaa datan laatua ja määrää eri menetelmillä, kuten muuntamalla merkkijonoja numeroiksi (kuten teemme [Klusteroinnissa](../../5-Clustering/1-Visualize/README.md)). Voit myös luoda uutta dataa alkuperäisen datan perusteella (kuten teemme [Luokittelussa](../../4-Classification/1-Introduction/README.md)). Voit puhdistaa ja muokata dataa (kuten teemme ennen [Web-sovellus](../../3-Web-App/README.md) -oppituntia). Lopuksi saatat joutua myös satunnaistamaan ja sekoittamaan dataa riippuen opetusmenetelmistäsi.

✅ Kun olet kerännyt ja käsitellyt datasi, ota hetki aikaa tarkistaaksesi, voiko sen muoto auttaa sinua vastaamaan aiottuun kysymykseesi. Saattaa olla, että data ei suoriudu hyvin annetussa tehtävässä, kuten huomaamme [Klusterointi](../../5-Clustering/1-Visualize/README.md) -oppitunneilla!

### Ominaisuudet ja kohde

[Ominaisuus](https://www.datasciencecentral.com/profiles/blogs/an-introduction-to-variable-and-feature-selection) on mitattavissa oleva datan ominaisuus. Monissa datasetissä se ilmaistaan sarakeotsikkona, kuten "päivämäärä", "koko" tai "väri". Ominaisuusmuuttuja, jota yleensä edustaa `X` koodissa, edustaa syötemuuttujaa, jota käytetään mallin opettamiseen.

Kohde on asia, jota yrität ennustaa. Kohde, jota yleensä edustaa `y` koodissa, edustaa vastausta kysymykseen, jonka yrität esittää datallesi: joulukuussa, minkä **väriset** kurpitsat ovat halvimpia? San Franciscossa, mitkä naapurustot tarjoavat parhaat kiinteistöjen **hinnat**? Joskus kohdetta kutsutaan myös nimikeattribuutiksi.

### Ominaisuusmuuttujan valinta

🎓 **Ominaisuusvalinta ja ominaisuuksien uuttaminen** Miten tiedät, minkä muuttujan valitset mallia rakentaessasi? Käyt läpi todennäköisesti prosessin, jossa valitset oikeat muuttujat parhaiten toimivaan malliin joko ominaisuusvalinnan tai ominaisuuksien uuttamisen avulla. Ne eivät kuitenkaan ole sama asia: "Ominaisuuksien uuttaminen luo uusia ominaisuuksia alkuperäisten ominaisuuksien funktioista, kun taas ominaisuusvalinta palauttaa alkuperäisten ominaisuuksien alijoukon." ([lähde](https://wikipedia.org/wiki/Feature_selection))

### Visualisoi datasi

Data-analyytikon työkalupakin tärkeä osa on kyky visualisoida dataa useiden erinomaisen kirjastojen, kuten Seabornin tai MatPlotLibin, avulla. Datan visuaalinen esittäminen voi auttaa sinua paljastamaan piilotettuja korrelaatioita, joita voit hyödyntää. Visualisoinnit voivat myös auttaa sinua havaitsemaan ennakkoluuloja tai epätasapainoista dataa (kuten huomaamme [Luokittelu](../../4-Classification/2-Classifiers-1/README.md) -oppitunneilla).

### Jaa datasetti

Ennen opettamista sinun tulee jakaa datasetti kahteen tai useampaan erikokoiseen osaan, jotka edustavat dataa hyvin.

- **Opetus**. Tämä osa datasetistä sovitetaan malliin sen opettamiseksi. Tämä osuus muodostaa suurimman osan alkuperäisestä datasetistä.
- **Testaus**. Testidatasetti on itsenäinen dataryhmä, joka usein kerätään alkuperäisestä datasta ja jota käytetään rakennetun mallin suorituskyvyn vahvistamiseen.
- **Validointi**. Validointijoukko on pienempi itsenäinen esimerkkiryhmä, jota käytetään mallin hyperparametrien tai rakenteen säätämiseen mallin parantamiseksi. Datasi koosta ja kysymyksestäsi riippuen et välttämättä tarvitse tätä kolmatta joukkoa (kuten huomaamme [Aikasarjojen ennustaminen](../../7-TimeSeries/1-Introduction/README.md) -oppitunneilla).

## Mallin rakentaminen

Käyttämällä opetusdataasi tavoitteesi on rakentaa malli, eli tilastollinen esitys datastasi, käyttämällä erilaisia algoritmeja sen **opettamiseksi**. Mallin opettaminen altistaa sen datalle ja antaa sen tehdä oletuksia havaitsemistaan kuvioista, validoida ja hyväksyä tai hylätä ne.

### Valitse opetusmenetelmä

Kysymyksesi ja datasi luonteen perusteella valitset menetelmän sen opettamiseksi. Käymällä läpi [Scikit-learnin dokumentaatiota](https://scikit-learn.org/stable/user_guide.html) - jota käytämme tässä kurssissa - voit tutkia monia tapoja opettaa mallia. Kokemuksesi perusteella saatat joutua kokeilemaan useita eri menetelmiä parhaan mallin rakentamiseksi. Todennäköisesti käyt läpi prosessin, jossa data-analyytikot arvioivat mallin suorituskykyä syöttämällä sille aiemmin näkemätöntä dataa, tarkistamalla tarkkuutta, ennakkoluuloja ja muita laatua heikentäviä ongelmia ja valitsemalla tehtävään sopivimman opetusmenetelmän.

### Opeta malli

Kun sinulla on opetusdata, olet valmis "sovittamaan" sen mallin luomiseksi. Huomaat, että monissa ML-kirjastoissa löytyy koodi "model.fit" - tässä vaiheessa syötät ominaisuusmuuttujasi taulukkomuodossa (yleensä "X") ja kohdemuuttujasi (yleensä "y").

### Arvioi malli

Kun opetusprosessi on valmis (suuren mallin opettaminen voi vaatia useita iteraatioita tai "epookkeja"), voit arvioida mallin laatua käyttämällä testidataa sen suorituskyvyn mittaamiseen. Tämä data on osa alkuperäisestä datasta, jota malli ei ole aiemmin analysoinut. Voit tulostaa taulukon mallin laadun mittareista.

🎓 **Mallin sovittaminen**

Koneoppimisen kontekstissa mallin sovittaminen viittaa mallin taustalla olevan funktion tarkkuuteen, kun se yrittää analysoida dataa, jota se ei tunne.

🎓 **Alioppiminen** ja **ylioppiminen** ovat yleisiä ongelmia, jotka heikentävät mallin laatua, kun malli sovittuu joko liian huonosti tai liian hyvin. Tämä aiheuttaa sen, että malli tekee ennusteita joko liian tiukasti tai liian löyhästi suhteessa opetusdataansa. Ylioppinut malli ennustaa opetusdataa liian hyvin, koska se on oppinut datan yksityiskohdat ja kohinan liian hyvin. Alioppinut malli ei ole tarkka, koska se ei pysty analysoimaan tarkasti opetusdataansa eikä dataa, jota se ei ole vielä "nähnyt".

![ylioppiminen malli](../../../../1-Introduction/4-techniques-of-ML/images/overfitting.png)
> Infografiikka: [Jen Looper](https://twitter.com/jenlooper)

## Parametrien säätö

Kun alkuperäinen opetus on valmis, tarkkaile mallin laatua ja harkitse sen parantamista säätämällä sen "hyperparametreja". Lue lisää prosessista [dokumentaatiosta](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters?WT.mc_id=academic-77952-leestott).

## Ennustaminen

Tämä on hetki, jolloin voit käyttää täysin uutta dataa testataksesi mallisi tarkkuutta. Sovelletussa ML-ympäristössä, jossa rakennat verkkosovelluksia mallin käyttöön tuotannossa, tämä prosessi voi sisältää käyttäjän syötteen keräämisen (esimerkiksi painikkeen painallus) muuttujan asettamiseksi ja sen lähettämiseksi mallille inferenssiä tai arviointia varten.

Näissä oppitunneissa opit käyttämään näitä vaiheita datan valmisteluun, mallin rakentamiseen, testaamiseen, arviointiin ja ennustamiseen - kaikki data-analyytikon eleet ja enemmän, kun etenet matkallasi kohti "full stack" ML-insinööriksi.

---

## 🚀Haaste

Piirrä vuokaavio, joka kuvaa ML-asiantuntijan työvaiheet. Missä näet itsesi tällä hetkellä prosessissa? Missä ennustat kohtaavasi vaikeuksia? Mikä vaikuttaa sinulle helpolta?

## [Jälkiluennon kysely](https://ff-quizzes.netlify.app/en/ml/)

## Kertaus ja itseopiskelu

Etsi verkosta haastatteluja data-analyytikoista, jotka keskustelevat päivittäisestä työstään. Tässä on [yksi](https://www.youtube.com/watch?v=Z3IjgbbCEfs).

## Tehtävä

[Haastattele data-analyytikkoa](assignment.md)

---

**Vastuuvapauslauseke**:  
Tämä asiakirja on käännetty käyttämällä tekoälypohjaista käännöspalvelua [Co-op Translator](https://github.com/Azure/co-op-translator). Vaikka pyrimme tarkkuuteen, huomioithan, että automaattiset käännökset voivat sisältää virheitä tai epätarkkuuksia. Alkuperäinen asiakirja sen alkuperäisellä kielellä tulisi pitää ensisijaisena lähteenä. Kriittisen tiedon osalta suositellaan ammattimaista ihmiskäännöstä. Emme ole vastuussa väärinkäsityksistä tai virhetulkinnoista, jotka johtuvat tämän käännöksen käytöstä.