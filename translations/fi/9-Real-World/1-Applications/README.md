<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "83320d6b6994909e35d830cebf214039",
  "translation_date": "2025-09-05T00:13:02+00:00",
  "source_file": "9-Real-World/1-Applications/README.md",
  "language_code": "fi"
}
-->
# Jälkikirjoitus: Koneoppiminen tosielämässä

![Yhteenveto koneoppimisesta tosielämässä sketchnotessa](../../../../sketchnotes/ml-realworld.png)
> Sketchnote: [Tomomi Imura](https://www.twitter.com/girlie_mac)

Tässä oppimateriaalissa olet oppinut monia tapoja valmistella dataa koulutusta varten ja luoda koneoppimismalleja. Olet rakentanut sarjan klassisia regressio-, klusterointi-, luokittelu-, luonnollisen kielen käsittely- ja aikasarjamalleja. Onnittelut! Nyt saatat miettiä, mihin tämä kaikki johtaa... mitkä ovat näiden mallien tosielämän sovellukset?

Vaikka teollisuudessa on paljon kiinnostusta tekoälyyn, joka yleensä hyödyntää syväoppimista, klassisilla koneoppimismalleilla on edelleen arvokkaita sovelluksia. Saatat jopa käyttää joitakin näistä sovelluksista tänään! Tässä oppitunnissa tutustut siihen, miten kahdeksan eri toimialaa ja asiantuntija-alueet käyttävät näitä malleja tehdäkseen sovelluksistaan suorituskykyisempiä, luotettavampia, älykkäämpiä ja käyttäjille arvokkaampia.

## [Ennakkokysely](https://ff-quizzes.netlify.app/en/ml/)

## 💰 Rahoitus

Rahoitussektori tarjoaa monia mahdollisuuksia koneoppimiselle. Monet tämän alueen ongelmat soveltuvat mallinnettaviksi ja ratkaistaviksi koneoppimisen avulla.

### Luottokorttipetosten tunnistaminen

Olemme oppineet [k-means-klusteroinnista](../../5-Clustering/2-K-Means/README.md) aiemmin kurssilla, mutta miten sitä voidaan käyttää luottokorttipetoksiin liittyvien ongelmien ratkaisemiseen?

K-means-klusterointi on hyödyllinen luottokorttipetosten tunnistustekniikassa, jota kutsutaan **poikkeavuuksien tunnistamiseksi**. Poikkeavuudet tai havainnot, jotka poikkeavat datan joukosta, voivat kertoa meille, käytetäänkö luottokorttia normaalisti vai tapahtuuko jotain epätavallista. Alla olevassa artikkelissa kuvataan, kuinka luottokorttidata voidaan järjestää k-means-klusterointialgoritmin avulla ja jokainen tapahtuma voidaan liittää klusteriin sen perusteella, kuinka poikkeava se vaikuttaa olevan. Tämän jälkeen voidaan arvioida riskialttiimmat klusterit petollisten ja laillisten tapahtumien osalta.
[Viite](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.680.1195&rep=rep1&type=pdf)

### Varainhoito

Varainhoidossa yksilö tai yritys hallinnoi sijoituksia asiakkaidensa puolesta. Heidän tehtävänsä on ylläpitää ja kasvattaa varallisuutta pitkällä aikavälillä, joten on tärkeää valita sijoituksia, jotka tuottavat hyvin.

Yksi tapa arvioida, miten tietty sijoitus tuottaa, on tilastollinen regressio. [Lineaarinen regressio](../../2-Regression/1-Tools/README.md) on arvokas työkalu rahaston suorituskyvyn ymmärtämiseen suhteessa vertailuarvoon. Voimme myös päätellä, ovatko regressiotulokset tilastollisesti merkittäviä eli kuinka paljon ne vaikuttaisivat asiakkaan sijoituksiin. Analyysiä voidaan laajentaa käyttämällä monimuuttujaregressiota, jossa otetaan huomioon lisäriskitekijöitä. Esimerkki siitä, miten tämä toimisi tietyn rahaston kohdalla, löytyy alla olevasta artikkelista.
[Viite](http://www.brightwoodventures.com/evaluating-fund-performance-using-regression/)

## 🎓 Koulutus

Koulutussektori on myös erittäin mielenkiintoinen alue, jossa koneoppimista voidaan soveltaa. Täällä voidaan käsitellä kiinnostavia ongelmia, kuten huijaamisen tunnistamista kokeissa tai esseissä tai puolueellisuuden hallintaa, tahallista tai tahatonta, korjausprosessissa.

### Opiskelijoiden käyttäytymisen ennustaminen

[Coursera](https://coursera.com), verkossa toimiva avoin kurssitarjoaja, ylläpitää erinomaista teknistä blogia, jossa he keskustelevat monista teknisistä päätöksistä. Tässä tapaustutkimuksessa he piirsivät regressioviivan yrittääkseen tutkia korrelaatiota matalan NPS-arvosanan (Net Promoter Score) ja kurssin säilyttämisen tai keskeyttämisen välillä.
[Viite](https://medium.com/coursera-engineering/controlled-regression-quantifying-the-impact-of-course-quality-on-learner-retention-31f956bd592a)

### Puolueellisuuden vähentäminen

[Grammarly](https://grammarly.com), kirjoitusavustaja, joka tarkistaa oikeinkirjoitus- ja kielioppivirheet, käyttää kehittyneitä [luonnollisen kielen käsittelyjärjestelmiä](../../6-NLP/README.md) tuotteissaan. He julkaisivat mielenkiintoisen tapaustutkimuksen teknisessä blogissaan siitä, miten he käsittelivät sukupuolten välistä puolueellisuutta koneoppimisessa, mistä opit [reiluuden alkeisoppitunnilla](../../1-Introduction/3-fairness/README.md).
[Viite](https://www.grammarly.com/blog/engineering/mitigating-gender-bias-in-autocorrect/)

## 👜 Vähittäiskauppa

Vähittäiskauppasektori voi ehdottomasti hyötyä koneoppimisen käytöstä, aina paremman asiakaspolun luomisesta varaston optimoituun hallintaan.

### Asiakaspolun personointi

Wayfairilla, yrityksellä, joka myy kodin tavaroita kuten huonekaluja, asiakkaiden auttaminen löytämään oikeat tuotteet heidän makuunsa ja tarpeisiinsa on ensisijaisen tärkeää. Tässä artikkelissa yrityksen insinöörit kuvaavat, kuinka he käyttävät koneoppimista ja NLP:tä "tuodakseen esiin oikeat tulokset asiakkaille". Erityisesti heidän Query Intent Engine -järjestelmänsä on rakennettu käyttämään entiteettien tunnistusta, luokittelijan koulutusta, omaisuuden ja mielipiteiden tunnistusta sekä sentimenttien merkitsemistä asiakasarvosteluissa. Tämä on klassinen esimerkki siitä, miten NLP toimii verkkokaupassa.
[Viite](https://www.aboutwayfair.com/tech-innovation/how-we-use-machine-learning-and-natural-language-processing-to-empower-search)

### Varastonhallinta

Innovatiiviset, ketterät yritykset kuten [StitchFix](https://stitchfix.com), laatikkopalvelu, joka lähettää vaatteita kuluttajille, luottavat vahvasti koneoppimiseen suositusten ja varastonhallinnan osalta. Heidän stailausryhmänsä työskentelevät yhdessä heidän kaupallisten tiimiensä kanssa: "yksi datatieteilijämme kokeili geneettistä algoritmia ja sovelsi sitä vaatteisiin ennustaakseen, mikä olisi menestyksekäs vaatekappale, joka ei vielä ole olemassa. Esittelimme sen kaupalliselle tiimille, ja nyt he voivat käyttää sitä työkaluna."
[Viite](https://www.zdnet.com/article/how-stitch-fix-uses-machine-learning-to-master-the-science-of-styling/)

## 🏥 Terveysala

Terveysala voi hyödyntää koneoppimista tutkimustehtävien optimointiin sekä logistisiin ongelmiin, kuten potilaiden uudelleenottamiseen tai tautien leviämisen estämiseen.

### Kliinisten tutkimusten hallinta

Kliinisten tutkimusten toksisuus on suuri huolenaihe lääkkeiden valmistajille. Kuinka paljon toksisuutta on siedettävää? Tässä tutkimuksessa eri kliinisten tutkimusmenetelmien analysointi johti uuden lähestymistavan kehittämiseen kliinisten tutkimusten tulosten ennustamiseksi. Erityisesti he pystyivät käyttämään random forest -menetelmää tuottaakseen [luokittelijan](../../4-Classification/README.md), joka pystyy erottamaan lääkeryhmät toisistaan.
[Viite](https://www.sciencedirect.com/science/article/pii/S2451945616302914)

### Sairaalan uudelleenottamisen hallinta

Sairaalahoito on kallista, erityisesti silloin, kun potilaat täytyy ottaa uudelleen hoitoon. Tässä artikkelissa käsitellään yritystä, joka käyttää koneoppimista ennustamaan uudelleenottamisen todennäköisyyttä [klusterointialgoritmien](../../5-Clustering/README.md) avulla. Nämä klusterit auttavat analyytikkoja "löytämään ryhmiä uudelleenottamisista, joilla saattaa olla yhteinen syy".
[Viite](https://healthmanagement.org/c/healthmanagement/issuearticle/hospital-readmissions-and-machine-learning)

### Tautien hallinta

Viimeisin pandemia on tuonut esiin tapoja, joilla koneoppiminen voi auttaa tautien leviämisen estämisessä. Tässä artikkelissa tunnistat ARIMA:n, logistiset käyrät, lineaarisen regression ja SARIMA:n käytön. "Tämä työ pyrkii laskemaan viruksen leviämisnopeuden ja ennustamaan kuolemat, parantumiset ja vahvistetut tapaukset, jotta voimme valmistautua paremmin ja selviytyä."
[Viite](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7979218/)

## 🌲 Ekologia ja vihreä teknologia

Luonto ja ekologia koostuvat monista herkistä järjestelmistä, joissa eläinten ja luonnon vuorovaikutus on keskiössä. On tärkeää pystyä mittaamaan näitä järjestelmiä tarkasti ja toimimaan asianmukaisesti, jos jotain tapahtuu, kuten metsäpalo tai eläinkannan väheneminen.

### Metsänhoito

Olet oppinut [vahvistusoppimisesta](../../8-Reinforcement/README.md) aiemmissa oppitunneissa. Se voi olla erittäin hyödyllinen, kun yritetään ennustaa luonnon ilmiöitä. Erityisesti sitä voidaan käyttää ekologisten ongelmien, kuten metsäpalojen ja vieraslajien leviämisen, seuraamiseen. Kanadassa ryhmä tutkijoita käytti vahvistusoppimista rakentaakseen metsäpalojen dynamiikkamalleja satelliittikuvista. Käyttämällä innovatiivista "spatially spreading process (SSP)" -menetelmää he kuvittelivat metsäpalon "agenttina missä tahansa maiseman solussa." "Toimintojen joukko, joita palo voi tehdä sijainnista missä tahansa ajanhetkessä, sisältää leviämisen pohjoiseen, etelään, itään tai länteen tai ei leviämistä."

Tämä lähestymistapa kääntää tavallisen RL-asetelman, koska vastaavan Markov Decision Process (MDP) -prosessin dynamiikka on tunnettu funktio välittömälle palon leviämiselle." Lue lisää tämän ryhmän käyttämistä klassisista algoritmeista alla olevasta linkistä.
[Viite](https://www.frontiersin.org/articles/10.3389/fict.2018.00006/full)

### Eläinten liikkeiden tunnistaminen

Vaikka syväoppiminen on luonut vallankumouksen eläinten liikkeiden visuaalisessa seurannassa (voit rakentaa oman [jääkarhuseurannan](https://docs.microsoft.com/learn/modules/build-ml-model-with-azure-stream-analytics/?WT.mc_id=academic-77952-leestott) täällä), klassisella koneoppimisella on edelleen paikkansa tässä tehtävässä.

Anturit maatilan eläinten liikkeiden seuraamiseen ja IoT hyödyntävät tämän tyyppistä visuaalista käsittelyä, mutta yksinkertaisemmat koneoppimistekniikat ovat hyödyllisiä datan esikäsittelyssä. Esimerkiksi tässä artikkelissa lampaiden asentoja seurattiin ja analysoitiin eri luokittelija-algoritmien avulla. Saatat tunnistaa ROC-käyrän sivulla 335.
[Viite](https://druckhaus-hofmann.de/gallery/31-wj-feb-2020.pdf)

### ⚡️ Energianhallinta

Aikasarjojen ennustamista käsittelevissä oppitunneissamme (../../7-TimeSeries/README.md) otimme esille älykkäiden pysäköintimittareiden käsitteen, joiden avulla kaupunki voi tuottaa tuloja ymmärtämällä kysyntää ja tarjontaa. Tämä artikkeli käsittelee yksityiskohtaisesti, kuinka klusterointi, regressio ja aikasarjojen ennustaminen yhdistettiin auttamaan tulevan energiankäytön ennustamisessa Irlannissa älykkäiden mittareiden avulla.
[Viite](https://www-cdn.knime.com/sites/default/files/inline-images/knime_bigdata_energy_timeseries_whitepaper.pdf)

## 💼 Vakuutus

Vakuutussektori on toinen sektori, joka käyttää koneoppimista elinkelpoisten taloudellisten ja aktuaaristen mallien rakentamiseen ja optimointiin.

### Volatiliteetin hallinta

MetLife, henkivakuutuspalveluntarjoaja, on avoin tavoistaan analysoida ja vähentää volatiliteettia taloudellisissa malleissaan. Tässä artikkelissa huomaat binääriset ja järjestysluokittelun visualisoinnit. Löydät myös ennustamisen visualisointeja.
[Viite](https://investments.metlife.com/content/dam/metlifecom/us/investments/insights/research-topics/macro-strategy/pdf/MetLifeInvestmentManagement_MachineLearnedRanking_070920.pdf)

## 🎨 Taide, kulttuuri ja kirjallisuus

Taiteessa, esimerkiksi journalismissa, on monia kiinnostavia ongelmia. Valeuutisten tunnistaminen on suuri ongelma, sillä on todistettu, että ne voivat vaikuttaa ihmisten mielipiteisiin ja jopa horjuttaa demokratioita. Museot voivat myös hyötyä koneoppimisen käytöstä, esimerkiksi artefaktien välisten yhteyksien löytämisessä tai resurssien suunnittelussa.

### Valeuutisten tunnistaminen

Valeuutisten tunnistaminen on nykyajan median kissa-hiiri-leikkiä. Tässä artikkelissa tutkijat ehdottavat järjestelmää, joka yhdistää useita koneoppimistekniikoita, joita olemme opiskelleet, ja testaa parhaan mallin käyttöönottoa: "Tämä järjestelmä perustuu luonnollisen kielen käsittelyyn datasta ominaisuuksien eristämiseksi, ja sitten näitä ominaisuuksia käytetään koneoppimisen luokittelijoiden, kuten Naive Bayes, Support Vector Machine (SVM), Random Forest (RF), Stochastic Gradient Descent (SGD) ja Logistic Regression (LR), kouluttamiseen."
[Viite](https://www.irjet.net/archives/V7/i6/IRJET-V7I6688.pdf)

Tämä artikkeli osoittaa, kuinka eri koneoppimisalueiden yhdistäminen voi tuottaa mielenkiintoisia tuloksia, jotka voivat auttaa estämään valeuutisten leviämistä ja aiheuttamasta todellista vahinkoa; tässä tapauksessa syynä oli COVID-hoitoja koskevien huhujen leviäminen, jotka aiheuttivat väkivaltaisia mellakoita.

### Museon koneoppiminen

Museot ovat tekoälyvallankumouksen kynnyksellä, jossa kokoelmien luettelointi ja digitalisointi sekä artefaktien välisten yhteyksien löytäminen helpottuvat teknologian edistyessä. Projektit kuten [In Codice Ratio](https://www.sciencedirect.com/science/article/abs/pii/S0306457321001035#:~:text=1.,studies%20over%20large%20historical%20sources.) auttavat avaamaan pääsyä vaikeasti saavutettaviin kokoelmiin, kuten Vatikaanin arkistoihin. Mutta museojen liiketoiminta hyötyy myös koneoppimismalleista.

Esimerkiksi Chicagon taideinstituutti rakensi malleja ennustamaan, mistä yleisöt ovat kiinnostuneita ja milloin he osallistuvat näyttelyihin. Tavoitteena on luoda yksilöllisiä ja optimoituja vierailijakokemuksia joka kerta, kun käyttäjä vierailee museossa. "Vuonna 2017 malli ennusti osallistumisen ja pääsymaksut yhden prosentin tarkkuudella, kertoo Andrew Simnick, Chicagon taideinstituutin vanhempi varatoimitusjohtaja."
[Viite](https://www.chicagobusiness.com/article/20180518/ISSUE01/180519840/art-institute-of-chicago-uses-data-to-make-exhibit-choices)

## 🏷 Markkinointi

### Asiakassegmentointi

Tehokkaimmat markkinointistrategiat kohdistavat asiakkaat eri tavoin erilaisten ryhmittelyjen perusteella. Tässä artikkelissa käsitellään klusterointialgoritmien käyttöä tukemaan eriytettyä markkinointia. Eriytetty markkinointi auttaa yrityksiä parantamaan brändin tunnettuutta, tavoittamaan enemmän asiakkaita ja ansaitsemaan enemmän rahaa.
[Viite](https://ai.inqline.com/machine-learning-for-marketing-customer-segmentation/)

## 🚀 Haaste

Tunnista toinen sektori, joka hyötyy joistakin tämän oppimateriaalin tekniikoista, ja
## [Luennon jälkeinen kysely](https://ff-quizzes.netlify.app/en/ml/)

## Kertaus ja itseopiskelu

Wayfairin data science -tiimillä on useita mielenkiintoisia videoita siitä, miten he käyttävät koneoppimista yrityksessään. Kannattaa [tutustua](https://www.youtube.com/channel/UCe2PjkQXqOuwkW1gw6Ameuw/videos)!

## Tehtävä

[ML-aarteenetsintä](assignment.md)

---

**Vastuuvapauslauseke**:  
Tämä asiakirja on käännetty käyttämällä tekoälypohjaista käännöspalvelua [Co-op Translator](https://github.com/Azure/co-op-translator). Vaikka pyrimme tarkkuuteen, huomioithan, että automaattiset käännökset voivat sisältää virheitä tai epätarkkuuksia. Alkuperäinen asiakirja sen alkuperäisellä kielellä tulisi pitää ensisijaisena lähteenä. Kriittisen tiedon osalta suositellaan ammattimaista ihmiskäännöstä. Emme ole vastuussa väärinkäsityksistä tai virhetulkinnoista, jotka johtuvat tämän käännöksen käytöstä.