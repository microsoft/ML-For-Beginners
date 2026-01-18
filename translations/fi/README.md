<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "0a6f4476a4f3934a4aa47c1bf47158bc",
  "translation_date": "2026-01-16T13:01:36+00:00",
  "source_file": "README.md",
  "language_code": "fi"
}
-->
[![GitHub-lisenssi](https://img.shields.io/github/license/microsoft/ML-For-Beginners.svg)](https://github.com/microsoft/ML-For-Beginners/blob/master/LICENSE)
[![GitHub-avustajat](https://img.shields.io/github/contributors/microsoft/ML-For-Beginners.svg)](https://GitHub.com/microsoft/ML-For-Beginners/graphs/contributors/)
[![GitHub-ongelmat](https://img.shields.io/github/issues/microsoft/ML-For-Beginners.svg)](https://GitHub.com/microsoft/ML-For-Beginners/issues/)
[![GitHub pull-pyynnöt](https://img.shields.io/github/issues-pr/microsoft/ML-For-Beginners.svg)](https://GitHub.com/microsoft/ML-For-Beginners/pulls/)
[![PR:t tervetulleita](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

[![GitHub-seuraajat](https://img.shields.io/github/watchers/microsoft/ML-For-Beginners.svg?style=social&label=Watch)](https://GitHub.com/microsoft/ML-For-Beginners/watchers/)
[![GitHub haarukat](https://img.shields.io/github/forks/microsoft/ML-For-Beginners.svg?style=social&label=Fork)](https://GitHub.com/microsoft/ML-For-Beginners/network/)
[![GitHub tähdet](https://img.shields.io/github/stars/microsoft/ML-For-Beginners.svg?style=social&label=Star)](https://GitHub.com/microsoft/ML-For-Beginners/stargazers/)

### 🌐 Monikielinen tuki

#### Tuettu GitHub Actionin avulla (automaattinen & aina ajan tasalla)

<!-- CO-OP TRANSLATOR LANGUAGES TABLE START -->
[Arabia](../ar/README.md) | [Bengali](../bn/README.md) | [Bulgaria](../bg/README.md) | [Burma (Myanmar)](../my/README.md) | [Kiina (yksinkertaistettu)](../zh/README.md) | [Kiina (perinteinen, Hong Kong)](../hk/README.md) | [Kiina (perinteinen, Macau)](../mo/README.md) | [Kiina (perinteinen, Taiwan)](../tw/README.md) | [Kroatia](../hr/README.md) | [Tšekki](../cs/README.md) | [Tanska](../da/README.md) | [Hollanti](../nl/README.md) | [Viro](../et/README.md) | [Suomi](./README.md) | [Ranska](../fr/README.md) | [Saksa](../de/README.md) | [Kreikka](../el/README.md) | [Heprea](../he/README.md) | [Hindi](../hi/README.md) | [Unkari](../hu/README.md) | [Indonesia](../id/README.md) | [Italia](../it/README.md) | [Japani](../ja/README.md) | [Kannada](../kn/README.md) | [Korea](../ko/README.md) | [Liettua](../lt/README.md) | [Malaiji](../ms/README.md) | [Malajalam](../ml/README.md) | [Marathi](../mr/README.md) | [Nepali](../ne/README.md) | [Nigerian Pidgin](../pcm/README.md) | [Norja](../no/README.md) | [Persia (Farsi)](../fa/README.md) | [Puola](../pl/README.md) | [Portugali (Brasilia)](../br/README.md) | [Portugali (Portugali)](../pt/README.md) | [Punjabi (Gurmukhi)](../pa/README.md) | [Romania](../ro/README.md) | [Venäjä](../ru/README.md) | [Serbia (kyrillinen)](../sr/README.md) | [Slovakki](../sk/README.md) | [Sloveeni](../sl/README.md) | [Espanja](../es/README.md) | [Swahili](../sw/README.md) | [Ruotsi](../sv/README.md) | [Tagalog (Filipino)](../tl/README.md) | [Tamil](../ta/README.md) | [Telugu](../te/README.md) | [Thai](../th/README.md) | [Turkki](../tr/README.md) | [Ukraina](../uk/README.md) | [Urdu](../ur/README.md) | [Vietnam](../vi/README.md)

> **Haluatko mieluummin kloonata paikallisesti?**

> Tämä repositorio sisältää yli 50 kielen käännökset, mikä kasvattaa latauskokoa merkittävästi. Jos haluat kloonata ilman käännöksiä, käytä harvaa checkoutia:
> ```bash
> git clone --filter=blob:none --sparse https://github.com/microsoft/ML-For-Beginners.git
> cd ML-For-Beginners
> git sparse-checkout set --no-cone '/*' '!translations' '!translated_images'
> ```
> Tämä tarjoaa kaiken tarvittavan kurssin suorittamiseen huomattavasti nopeammalla latauksella.
<!-- CO-OP TRANSLATOR LANGUAGES TABLE END -->

#### Liity yhteisöömme

[![Microsoft Foundry Discord](https://dcbadge.limes.pink/api/server/nTYy5BXMWG)](https://discord.gg/nTYy5BXMWG)

Meillä on käynnissä Discordin Learn with AI -sarja, opi lisää ja liity mukaan osoitteessa [Learn with AI Series](https://aka.ms/learnwithai/discord) 18. – 30. syyskuuta 2025. Saat vinkkejä ja temppuja GitHub Copilotin käyttämiseen Data Scienticessä.

![Learn with AI series](../../../../translated_images/fi/3.9b58fd8d6c373c20.webp)

# Koneoppiminen aloittelijoille – opetussuunnitelma

> 🌍 Matkusta ympäri maailmaa tutkiskellessamme koneoppimista maailman kulttuurien kautta 🌍

Microsoftin Cloud Advocates tarjoaa 12 viikon ja 26 oppitunnin opetussuunnitelman, joka käsittelee **koneoppimista**. Tässä opetussuunnitelmassa opit siitä, mitä joskus kutsutaan **klassiseksi koneoppimiseksi**, pääasiassa Scikit-learn-kirjastoa käyttäen ja välttäen syväoppimista, joka käsitellään [AI for Beginners -opetussuunnitelmassamme](https://aka.ms/ai4beginners). Yhdistä nämä oppitunnit myös ['Data Science for Beginners' -opetussuunnitelmaamme](https://aka.ms/ds4beginners)!

Matkusta kanssamme ympäri maailmaa soveltaen näitä klassisia tekniikoita eri alueiden dataan. Jokainen oppitunti sisältää ennakko- ja jälkikäteen suoritettavat testit, kirjalliset ohjeet oppitunnin suorittamiseen, ratkaisun, tehtävän ja muuta. Projektipohjainen pedagogiikkamme mahdollistaa oppimisen rakentamisen kautta, mikä on todistettu tapa omaksua uusia taitoja.

**✍️ Suuret kiitokset kirjoittajillemme** Jen Looper, Stephen Howell, Francesca Lazzeri, Tomomi Imura, Cassie Breviu, Dmitry Soshnikov, Chris Noring, Anirban Mukherjee, Ornella Altunyan, Ruth Yakubu ja Amy Boyd

**🎨 Kiitos myös kuvittajillemme** Tomomi Imura, Dasani Madipalli ja Jen Looper

**🙏 Erityiskiitos 🙏 Microsoft Student Ambassador -kirjoittajille, arvioijille ja sisällöntuottajillemme**, erityisesti Rishit Dagli, Muhammad Sakib Khan Inan, Rohan Raj, Alexandru Petrescu, Abhishek Jaiswal, Nawrin Tabassum, Ioan Samuila ja Snigdha Agarwal

**🤩 Lisäkiitos Microsoft Student Ambassadors Eric Wanjau, Jasleen Sondhi ja Vidushi Gupta R-oppitunneistamme!**

# Aloittaminen

Seuraa näitä ohjeita:
1. **Forkkaa repositorio**: Klikkaa oikeassa yläkulmassa olevaa "Fork"-painiketta.
2. **Kloonaa repositorio**: `git clone https://github.com/microsoft/ML-For-Beginners.git`

> [löydät kaikki kurssin lisäresurssit Microsoft Learn -kokoelmastamme](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)

> 🔧 **Tarvitsetko apua?** Tarkista [Vianmääritysohjeistuksemme](TROUBLESHOOTING.md) yleisiin asennus-, käyttöönotto- ja oppituntien suorittamisen ongelmiin.


**[Opiskelijat](https://aka.ms/student-page)**, käyttääksesi tätä opetussuunnitelmaa, kloonaa koko repositorio omaan GitHub-tiliisi ja suorita tehtävät itseksesi tai ryhmässä:

- Aloita ennakkotentin tekemisellä.
- Lue luento ja tee aktiviteetit, pysähdy ja pohdi jokaisen tietokyselyn kohdalla.
- Yritä luoda projektit ymmärtämällä oppitunnit sen sijaan, että suoritat ratkaisukoodin; koodi on kuitenkin saatavilla kunkin projektipainotteisen oppitunnin `/solution`-kansiossa.
- Tee jälkitentti.
- Suorita haaste.
- Tee tehtävä.
- Oppituntikokonaisuuden suorittamisen jälkeen käy [Keskustelupalstalla](https://github.com/microsoft/ML-For-Beginners/discussions) ja "opiskele ääneen" täyttämällä asianmukainen PAT-arviointi. PAT on edistymisarviointityökalu, jonka avulla voit edistää oppimistasi. Voit myös reagoida muiden PAT-arviointeihin, jotta voimme oppia yhdessä.

> Jatko-opiskeluun suosittelemme seuraamaan näitä [Microsoft Learn](https://docs.microsoft.com/en-us/users/jenlooper-2911/collections/k7o7tg1gp306q4?WT.mc_id=academic-77952-leestott) moduuleja ja oppimispolkuja.

**Opettajat**, olemme liittäneet [joitakin ehdotuksia](for-teachers.md) tämän opetussuunnitelman käyttämiseen.

---

## Video-kävelyoppaat

Jotkin oppitunnit ovat saatavilla lyhyinä videoina. Löydät ne kaikki oppitunneista tai Microsoft Developerin YouTube-kanavan [ML for Beginners -soittolistalta](https://aka.ms/ml-beginners-videos) klikkaamalla alla olevaa kuvaa.

[![ML for beginners banner](../../../../translated_images/fi/ml-for-beginners-video-banner.63f694a100034bc6.webp)](https://aka.ms/ml-beginners-videos)

---

## Tapaa tiimi

[![Promo video](../../images/ml.gif)](https://youtu.be/Tj1XWrDSYJU)

**Gif:** [Mohit Jaisal](https://linkedin.com/in/mohitjaisal)

> 🎥 Klikkaa yllä olevaa kuvaa katsoaksesi videon projektista ja sen tekijöistä!

---

## Pedagogiikka

Olemme valinneet tämän opetussuunnitelman rakentamiseen kaksi pedagogista periaatetta: varmistaa, että se on käytännönläheinen **projektipohjainen** ja sisältää **tiheät tentit**. Lisäksi opetussuunnitelmalla on yhtenäinen **teema**, joka antaa sille johdonmukaisuuden.

Sisällön sovittaminen projekteihin tekee prosessista opiskelijoille mielekkäämmän ja käsitteiden omaksuminen paranee. Lisäksi matalan panoksen tentti ennen luentoa asettaa opiskelijan oppimistavoitteen aiheelle, ja jälkitentti varmistaa käsitteiden paremman omaksumisen. Tämä opetussuunnitelma on suunniteltu joustavaksi ja hauskaksi, ja se voidaan suorittaa kokonaan tai osittain. Projektit alkavat pieninä ja kasvavat monimutkaisemmiksi 12 viikon prosessin loppua kohti. Tämä opetussuunnitelma sisältää myös jälkikirjoituksen ML:n todellisista sovelluksista, jota voi käyttää lisäpisteenä tai keskustelun pohjana.

> Löydät [käyttäytymissäännöstömme](CODE_OF_CONDUCT.md), [Osallistumisohjeet](CONTRIBUTING.md), [Käännösohjeet](TRANSLATIONS.md) ja [Vianmääritysohjeet](TROUBLESHOOTING.md). Otamme mielellämme vastaan rakentavaa palautettasi!

## Jokainen oppitunti sisältää

- valinnaisen luonnoksen
- valinnaisen lisävideon
- video-kävelyoppaan (vain osassa oppitunteja)
- [ennakko-oppitentin](https://ff-quizzes.netlify.app/en/ml/)
- kirjallisen oppitunnin
- projektipohjaisissa oppitunneissa askel askeleelta ohjeet projektin rakentamiseen
- tietokyselyitä
- haasteen
- lisälukemista
- tehtävän
- [jälkikäteen tehtävän tentin](https://ff-quizzes.netlify.app/en/ml/)

> **Huomio kielistä**: Nämä oppitunnit on kirjoitettu pääasiassa Pythonilla, mutta monia on myös saatavilla R-kielellä. R-oppitunnin suorittamiseksi siirry `/solution`-kansioon ja etsi R-oppitunteja. Niissä on .rmd-pääte, joka tarkoittaa **R Markdown** -tiedostoa, joka on yksinkertaisesti määritelty `koodilohkojen` (R:n tai muiden kielien) ja `YAML-otsikon` (joka ohjaa, miten muotoilla tulokset kuten PDF) yhdistämisenä `Markdown-dokumenttiin`. Näin se toimii esimerkillisenä kirjoituskehyksenä data-analytiikassa, koska sen avulla voit yhdistää koodisi, sen tuottaman tuloksen ja ajatuksesi kirjoittamalla ne Markdowniin. Lisäksi R Markdown -tiedostot voidaan renderöidä erilaisiin tulostusmuotoihin, kuten PDF, HTML tai Word.
> **Huomautus visailuista**: Kaikki visailut löytyvät kansiosta [Quiz App folder](../../quiz-app), yhteensä 52 visailua, joissa kussakin on kolme kysymystä. Ne on linkitetty oppitunteihin, mutta visailusovellusta voi ajaa paikallisesti; noudata `quiz-app` -kansion ohjeita sovelluksen paikalliseen ajamiseen tai Azureen siirtämiseen.

| Oppitunnin numero |                             Aihe                              |                   Oppituntiryhmä                   | Oppimistavoitteet                                                                                                               |                                                              Linkitetty oppitunti                                                               |                        Tekijä                        |
| :---------------: | :----------------------------------------------------------: | :------------------------------------------------: | ------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------: |
|        01         |                Johdatus koneoppimiseen                      |      [Introduction](1-Introduction/README.md)     | Opi koneoppimisen peruskäsitteet                                                                                                |                                            [Oppitunti](1-Introduction/1-intro-to-ML/README.md)                                             |                       Muhammad                      |
|        02         |               Koneoppimisen historia                        |      [Introduction](1-Introduction/README.md)     | Opi tämän alan historia                                                                                                        |                                            [Oppitunti](1-Introduction/2-history-of-ML/README.md)                                            |                     Jen ja Amy                      |
|        03         |                Oikeudenmukaisuus ja koneoppiminen           |      [Introduction](1-Introduction/README.md)     | Mitkä ovat tärkeät filosofiset kysymykset oikeudenmukaisuudesta, jotka opiskelijoiden tulisi huomioida rakentaessaan ja käyttäessään ML-malleja? |                                              [Oppitunti](1-Introduction/3-fairness/README.md)                                               |                        Tomomi                       |
|        04         |               Koneoppimisen menetelmiä                      |      [Introduction](1-Introduction/README.md)     | Mitä menetelmiä ML-tutkijat käyttävät mallien rakentamiseen?                                                                   |                                          [Oppitunti](1-Introduction/4-techniques-of-ML/README.md)                                           |                    Chris ja Jen                     |
|        05         |                      Johdatus regressioon                   |        [Regression](2-Regression/README.md)       | Aloita Pythonilla ja Scikit-learnillä regressiomalleihin                                                                     |         [Python](2-Regression/1-Tools/README.md) • [R](../../2-Regression/1-Tools/solution/R/lesson_1.html)         |      Jen • Eric Wanjau       |
|        06         |                 Pohjoisamerikkalaiset kurpitsahinnat 🎃      |        [Regression](2-Regression/README.md)       | Visualisoi ja puhdista dataa koneoppimista varten                                                                               |          [Python](2-Regression/2-Data/README.md) • [R](../../2-Regression/2-Data/solution/R/lesson_2.html)          |      Jen • Eric Wanjau       |
|        07         |                 Pohjoisamerikkalaiset kurpitsahinnat 🎃      |        [Regression](2-Regression/README.md)       | Rakenna lineaarinen ja polynominen regressiomalli                                                                              |        [Python](2-Regression/3-Linear/README.md) • [R](../../2-Regression/3-Linear/solution/R/lesson_3.html)        |      Jen ja Dmitry • Eric Wanjau       |
|        08         |                 Pohjoisamerikkalaiset kurpitsahinnat 🎃      |        [Regression](2-Regression/README.md)       | Rakenna logistinen regressiomalli                                                                                              |     [Python](2-Regression/4-Logistic/README.md) • [R](../../2-Regression/4-Logistic/solution/R/lesson_4.html)      |      Jen • Eric Wanjau       |
|        09         |                           Web-sovellus 🔌                    |           [Web App](3-Web-App/README.md)           | Rakenna web-sovellus koulutetun mallisi käyttöön                                                                               |                                                 [Python](3-Web-App/1-Web-App/README.md)                                                  |                         Jen                          |
|        10         |                    Johdatus luokitteluun                     |    [Classification](4-Classification/README.md)   | Puhdista, valmistele ja visualisoi data; johdatus luokitteluun                                                                 | [Python](4-Classification/1-Introduction/README.md) • [R](../../4-Classification/1-Introduction/solution/R/lesson_10.html)  | Jen ja Cassie • Eric Wanjau |
|        11         |              Herkulliset aasialaiset ja intialaiset ruoat 🍜  |    [Classification](4-Classification/README.md)   | Johdatus luokittelijoihin                                                                                                      | [Python](4-Classification/2-Classifiers-1/README.md) • [R](../../4-Classification/2-Classifiers-1/solution/R/lesson_11.html) | Jen ja Cassie • Eric Wanjau |
|        12         |              Herkulliset aasialaiset ja intialaiset ruoat 🍜  |    [Classification](4-Classification/README.md)   | Lisää luokittelijoita                                                                                                          | [Python](4-Classification/3-Classifiers-2/README.md) • [R](../../4-Classification/3-Classifiers-2/solution/R/lesson_12.html) | Jen ja Cassie • Eric Wanjau |
|        13         |              Herkulliset aasialaiset ja intialaiset ruoat 🍜  |    [Classification](4-Classification/README.md)   | Rakenna suosittelujärjestelmän web-sovellus mallisi avulla                                                                     |                                              [Python](4-Classification/4-Applied/README.md)                                              |                         Jen                          |
|        14         |                      Johdatus klusterointiin                 |        [Clustering](5-Clustering/README.md)        | Puhdista, valmistele ja visualisoi datasi; Johdatus klusterointiin                                                              |         [Python](5-Clustering/1-Visualize/README.md) • [R](../../5-Clustering/1-Visualize/solution/R/lesson_14.html)         |      Jen • Eric Wanjau       |
|        15         |             Nigerialaisen musiikkimaun tutkimista 🎧        |        [Clustering](5-Clustering/README.md)        | Tutustu K-Means-klusterointimenetelmään                                                                                         |           [Python](5-Clustering/2-K-Means/README.md) • [R](../../5-Clustering/2-K-Means/solution/R/lesson_15.html)           |      Jen • Eric Wanjau       |
|        16         |        Johdatus luonnollisen kielen käsittelyyn ☕️            |   [Natural language processing](6-NLP/README.md)  | Opi NLP:n perusteet rakentamalla yksinkertainen botti                                                                           |                                             [Python](6-NLP/1-Introduction-to-NLP/README.md)                                              |                       Stephen                        |
|        17         |                         Yleisiä NLP-tehtäviä ☕️               |   [Natural language processing](6-NLP/README.md)  | Syvennä NLP-tietoasi ymmärtämällä yleisiä kielirakenteissa tarvittavia tehtäviä                                                |                                                    [Python](6-NLP/2-Tasks/README.md)                                                     |                       Stephen                        |
|        18         |             Käännös ja tunteiden analysointi ♥️              |   [Natural language processing](6-NLP/README.md)  | Käännös ja tunteiden analysointi Jane Austenin kanssa                                                                           |                                            [Python](6-NLP/3-Translation-Sentiment/README.md)                                             |                       Stephen                        |
|        19         |                  Euroopan romanttiset hotellit ♥️            |   [Natural language processing](6-NLP/README.md)  | Tunteiden analysointi hotelliarvioiden 1 avulla                                                                                 |                                               [Python](6-NLP/4-Hotel-Reviews-1/README.md)                                                |                       Stephen                        |
|        20         |                  Euroopan romanttiset hotellit ♥️            |   [Natural language processing](6-NLP/README.md)  | Tunteiden analysointi hotelliarvioiden 2 avulla                                                                                 |                                               [Python](6-NLP/5-Hotel-Reviews-2/README.md)                                                |                       Stephen                        |
|        21         |            Johdatus aikasarjaennusteisiin                    |        [Time series](7-TimeSeries/README.md)       | Johdatus aikasarjaennusteisiin                                                                                                 |                                             [Python](7-TimeSeries/1-Introduction/README.md)                                              |                      Francesca                       |
|        22         | ⚡️ Maailman sähkönkulutus ⚡️ - aikasarjaennuste ARIMA:lla   |        [Time series](7-TimeSeries/README.md)       | Aikasarjaennuste ARIMA-mallilla                                                                                                |                                                 [Python](7-TimeSeries/2-ARIMA/README.md)                                                 |                      Francesca                       |
|        23         |  ⚡️ Maailman sähkönkulutus ⚡️ - aikasarjaennuste SVR:llä    |        [Time series](7-TimeSeries/README.md)       | Aikasarjaennuste tukivektoriregressiolla                                                                                       |                                                  [Python](7-TimeSeries/3-SVR/README.md)                                                  |                       Anirban                        |
|        24         |             Johdatus vahvistusoppimiseen                     | [Reinforcement learning](8-Reinforcement/README.md) | Johdatus vahvistusoppimiseen Q-Learningin avulla                                                                               |                                             [Python](8-Reinforcement/1-QLearning/README.md)                                              |                        Dmitry                        |
|        25         |                Auta Peteriä välttelemään sutta! 🐺           | [Reinforcement learning](8-Reinforcement/README.md) | Vahvistusoppiminen Gymissä                                                                                                     |                                                [Python](8-Reinforcement/2-Gym/README.md)                                                 |                        Dmitry                        |
|  Jälkikirjoitus  |            Käytännön ML-tilanteita ja sovelluksia            |      [ML in the Wild](9-Real-World/README.md)      | Mielenkiintoisia ja paljastavia käytännön esimerkkejä perinteisestä ML:stä                                                      |                                             [Oppitunti](9-Real-World/1-Applications/README.md)                                              |                         Tiimi                         |
|  Jälkikirjoitus  |            Mallin virheenkorjaus ML:ssä RAI-kojelautaa käyttäen |      [ML in the Wild](9-Real-World/README.md)      | Mallin virheenkorjaus koneoppimisessa Responsible AI -kojelauta-komponenttien avulla                                            |                                             [Oppitunti](9-Real-World/2-Debugging-ML-Models/README.md)                                              |                         Ruth Yakubu                       |

> [löydä kaikki tämän kurssin lisäresurssit Microsoft Learn -kokoelmastamme](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)

## Offline-käyttö

Voit käyttää tätä dokumentaatiota offline-tilassa käyttämällä [Docsify](https://docsify.js.org/#/). Tee fork tähän repositorioon, [asenna Docsify](https://docsify.js.org/#/quickstart) paikalliselle koneellesi, ja tämän repositorion juuressa kirjoita `docsify serve`. Verkkosivusto palvelee portissa 3000 paikallisella koneellasi: `localhost:3000`.

## PDF:t

Löydät opetussuunnitelman pdf-version linkkeineen [täältä](https://microsoft.github.io/ML-For-Beginners/pdf/readme.pdf).


## 🎒 Muut kurssit 

Tiimimme tuottaa myös muita kursseja! Tutustu:

<!-- CO-OP TRANSLATOR OTHER COURSES START -->
### LangChain
[![LangChain4j for Beginners](https://img.shields.io/badge/LangChain4j%20for%20Beginners-22C55E?style=for-the-badge&&labelColor=E5E7EB&color=0553D6)](https://aka.ms/langchain4j-for-beginners)
[![LangChain.js for Beginners](https://img.shields.io/badge/LangChain.js%20for%20Beginners-22C55E?style=for-the-badge&labelColor=E5E7EB&color=0553D6)](https://aka.ms/langchainjs-for-beginners?WT.mc_id=m365-94501-dwahlin)

---

### Azure / Edge / MCP / Agents
[![AZD for Beginners](https://img.shields.io/badge/AZD%20for%20Beginners-0078D4?style=for-the-badge&labelColor=E5E7EB&color=0078D4)](https://github.com/microsoft/AZD-for-beginners?WT.mc_id=academic-105485-koreyst)
[![Edge AI for Beginners](https://img.shields.io/badge/Edge%20AI%20for%20Beginners-00B8E4?style=for-the-badge&labelColor=E5E7EB&color=00B8E4)](https://github.com/microsoft/edgeai-for-beginners?WT.mc_id=academic-105485-koreyst)
[![MCP for Beginners](https://img.shields.io/badge/MCP%20for%20Beginners-009688?style=for-the-badge&labelColor=E5E7EB&color=009688)](https://github.com/microsoft/mcp-for-beginners?WT.mc_id=academic-105485-koreyst)
[![AI Agents for Beginners](https://img.shields.io/badge/AI%20Agents%20for%20Beginners-00C49A?style=for-the-badge&labelColor=E5E7EB&color=00C49A)](https://github.com/microsoft/ai-agents-for-beginners?WT.mc_id=academic-105485-koreyst)

---
 
### Generative AI Series
[![Generatiivinen tekoäly aloittelijoille](https://img.shields.io/badge/Generative%20AI%20for%20Beginners-8B5CF6?style=for-the-badge&labelColor=E5E7EB&color=8B5CF6)](https://github.com/microsoft/generative-ai-for-beginners?WT.mc_id=academic-105485-koreyst)
[![Generatiivinen tekoäly (.NET)](https://img.shields.io/badge/Generative%20AI%20(.NET)-9333EA?style=for-the-badge&labelColor=E5E7EB&color=9333EA)](https://github.com/microsoft/Generative-AI-for-beginners-dotnet?WT.mc_id=academic-105485-koreyst)
[![Generatiivinen tekoäly (Java)](https://img.shields.io/badge/Generative%20AI%20(Java)-C084FC?style=for-the-badge&labelColor=E5E7EB&color=C084FC)](https://github.com/microsoft/generative-ai-for-beginners-java?WT.mc_id=academic-105485-koreyst)
[![Generatiivinen tekoäly (JavaScript)](https://img.shields.io/badge/Generative%20AI%20(JavaScript)-E879F9?style=for-the-badge&labelColor=E5E7EB&color=E879F9)](https://github.com/microsoft/generative-ai-with-javascript?WT.mc_id=academic-105485-koreyst)

---

### Perusopetus
[![ML aloittelijoille](https://img.shields.io/badge/ML%20for%20Beginners-22C55E?style=for-the-badge&labelColor=E5E7EB&color=22C55E)](https://aka.ms/ml-beginners?WT.mc_id=academic-105485-koreyst)
[![Datatiede aloittelijoille](https://img.shields.io/badge/Data%20Science%20for%20Beginners-84CC16?style=for-the-badge&labelColor=E5E7EB&color=84CC16)](https://aka.ms/datascience-beginners?WT.mc_id=academic-105485-koreyst)
[![Tekoäly aloittelijoille](https://img.shields.io/badge/AI%20for%20Beginners-A3E635?style=for-the-badge&labelColor=E5E7EB&color=A3E635)](https://aka.ms/ai-beginners?WT.mc_id=academic-105485-koreyst)
[![Kyberturvallisuus aloittelijoille](https://img.shields.io/badge/Cybersecurity%20for%20Beginners-F97316?style=for-the-badge&labelColor=E5E7EB&color=F97316)](https://github.com/microsoft/Security-101?WT.mc_id=academic-96948-sayoung)
[![Web-kehitys aloittelijoille](https://img.shields.io/badge/Web%20Dev%20for%20Beginners-EC4899?style=for-the-badge&labelColor=E5E7EB&color=EC4899)](https://aka.ms/webdev-beginners?WT.mc_id=academic-105485-koreyst)
[![IoT aloittelijoille](https://img.shields.io/badge/IoT%20for%20Beginners-14B8A6?style=for-the-badge&labelColor=E5E7EB&color=14B8A6)](https://aka.ms/iot-beginners?WT.mc_id=academic-105485-koreyst)
[![XR-kehitys aloittelijoille](https://img.shields.io/badge/XR%20Development%20for%20Beginners-38BDF8?style=for-the-badge&labelColor=E5E7EB&color=38BDF8)](https://github.com/microsoft/xr-development-for-beginners?WT.mc_id=academic-105485-koreyst)

---

### Copilot-sarja
[![Copilot tekoälyn pariohjelmointiin](https://img.shields.io/badge/Copilot%20for%20AI%20Paired%20Programming-FACC15?style=for-the-badge&labelColor=E5E7EB&color=FACC15)](https://aka.ms/GitHubCopilotAI?WT.mc_id=academic-105485-koreyst)
[![Copilot C#/.NET:lle](https://img.shields.io/badge/Copilot%20for%20C%23/.NET-FBBF24?style=for-the-badge&labelColor=E5E7EB&color=FBBF24)](https://github.com/microsoft/mastering-github-copilot-for-dotnet-csharp-developers?WT.mc_id=academic-105485-koreyst)
[![Copilot-seikkailu](https://img.shields.io/badge/Copilot%20Adventure-FDE68A?style=for-the-badge&labelColor=E5E7EB&color=FDE68A)](https://github.com/microsoft/CopilotAdventures?WT.mc_id=academic-105485-koreyst)
<!-- CO-OP TRANSLATOR OTHER COURSES END -->

## Apua saamassa

Jos jäät jumiin tai sinulla on kysyttävää tekoälysovellusten rakentamisesta, liity muiden oppijoiden ja kokeneiden kehittäjien keskusteluihin MCP:stä. Se on kannustava yhteisö, jossa kysymykset ovat tervetulleita ja tieto jaetaan vapaaehtoisesti.

[![Microsoft Foundry Discord](https://dcbadge.limes.pink/api/server/nTYy5BXMWG)](https://discord.gg/nTYy5BXMWG)

Jos sinulla on tuotepalaute tai kohtaat virheitä rakennusvaiheessa, vieraile:

[![Microsoft Foundry Developer Forum](https://img.shields.io/badge/GitHub-Microsoft_Foundry_Developer_Forum-blue?style=for-the-badge&logo=github&color=000000&logoColor=fff)](https://aka.ms/foundry/forum)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Vastuuvapauslauseke**:
Tämä asiakirja on käännetty käyttäen tekoälypohjaista käännöspalvelua [Co-op Translator](https://github.com/Azure/co-op-translator). Vaikka pyrimme tarkkuuteen, otathan huomioon, että automaattikäännöksissä saattaa esiintyä virheitä tai epätarkkuuksia. Alkuperäinen asiakirja omalla kielellään on virallinen ja auktoriteettinen lähde. Kriittisissä tiedoissa suosittelemme ammattimaisen ihmiskääntäjän käyttöä. Emme ole vastuussa mahdollisista väärinymmärryksistä tai tulkinnoista, jotka johtuvat tämän käännöksen käytöstä.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->