<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "83320d6b6994909e35d830cebf214039",
  "translation_date": "2025-09-05T19:21:14+00:00",
  "source_file": "9-Real-World/1-Applications/README.md",
  "language_code": "nl"
}
-->
# Postscript: Machine learning in de echte wereld

![Samenvatting van machine learning in de echte wereld in een sketchnote](../../../../sketchnotes/ml-realworld.png)
> Sketchnote door [Tomomi Imura](https://www.twitter.com/girlie_mac)

In dit curriculum heb je veel manieren geleerd om data voor te bereiden voor training en machine learning-modellen te maken. Je hebt een reeks klassieke regressie-, clustering-, classificatie-, natuurlijke taalverwerking- en tijdreeksmodellen gebouwd. Gefeliciteerd! Nu vraag je je misschien af waar dit allemaal voor dient... wat zijn de toepassingen van deze modellen in de echte wereld?

Hoewel er in de industrie veel interesse is in AI, die meestal gebruik maakt van deep learning, zijn er nog steeds waardevolle toepassingen voor klassieke machine learning-modellen. Je gebruikt misschien zelfs vandaag al enkele van deze toepassingen! In deze les verken je hoe acht verschillende industrieën en vakgebieden deze soorten modellen gebruiken om hun toepassingen performanter, betrouwbaarder, intelligenter en waardevoller te maken voor gebruikers.

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## 💰 Financiën

De financiële sector biedt veel mogelijkheden voor machine learning. Veel problemen in deze sector lenen zich goed om gemodelleerd en opgelost te worden met ML.

### Detectie van creditcardfraude

We hebben eerder in de cursus geleerd over [k-means clustering](../../5-Clustering/2-K-Means/README.md), maar hoe kan dit worden gebruikt om problemen met betrekking tot creditcardfraude op te lossen?

K-means clustering is handig bij een techniek voor het detecteren van creditcardfraude genaamd **outlier detection**. Outliers, of afwijkingen in waarnemingen binnen een dataset, kunnen ons vertellen of een creditcard op normale wijze wordt gebruikt of dat er iets ongewoons gebeurt. Zoals beschreven in het onderstaande artikel, kun je creditcarddata sorteren met een k-means clustering-algoritme en elke transactie toewijzen aan een cluster op basis van hoe afwijkend deze lijkt te zijn. Vervolgens kun je de meest risicovolle clusters evalueren op frauduleuze versus legitieme transacties.
[Referentie](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.680.1195&rep=rep1&type=pdf)

### Vermogensbeheer

Bij vermogensbeheer beheert een individu of bedrijf investeringen namens hun klanten. Hun taak is om op lange termijn vermogen te behouden en te laten groeien, dus het is essentieel om investeringen te kiezen die goed presteren.

Een manier om te evalueren hoe een bepaalde investering presteert, is door middel van statistische regressie. [Lineaire regressie](../../2-Regression/1-Tools/README.md) is een waardevol hulpmiddel om te begrijpen hoe een fonds presteert ten opzichte van een benchmark. We kunnen ook afleiden of de resultaten van de regressie statistisch significant zijn, of hoeveel ze de investeringen van een klant zouden beïnvloeden. Je kunt je analyse zelfs verder uitbreiden met meervoudige regressie, waarbij aanvullende risicofactoren worden meegenomen. Voor een voorbeeld van hoe dit zou werken voor een specifiek fonds, bekijk het onderstaande artikel over het evalueren van fondsprestaties met regressie.
[Referentie](http://www.brightwoodventures.com/evaluating-fund-performance-using-regression/)

## 🎓 Onderwijs

De onderwijssector is ook een zeer interessant gebied waar ML kan worden toegepast. Er zijn interessante problemen aan te pakken, zoals het detecteren van fraude bij toetsen of essays, of het beheren van (onbedoelde) vooringenomenheid in het correctieproces.

### Voorspellen van studentengedrag

[Coursera](https://coursera.com), een online aanbieder van open cursussen, heeft een geweldige techblog waar ze veel technische beslissingen bespreken. In deze casestudy hebben ze een regressielijn uitgezet om te proberen een correlatie te vinden tussen een lage NPS (Net Promoter Score) en het behoud of uitval van cursisten.
[Referentie](https://medium.com/coursera-engineering/controlled-regression-quantifying-the-impact-of-course-quality-on-learner-retention-31f956bd592a)

### Vooringenomenheid verminderen

[Grammarly](https://grammarly.com), een schrijfassistent die controleert op spelling- en grammaticafouten, gebruikt geavanceerde [natuurlijke taalverwerkingssystemen](../../6-NLP/README.md) in al zijn producten. Ze publiceerden een interessante casestudy in hun techblog over hoe ze omgingen met genderbias in machine learning, wat je hebt geleerd in onze [introductieles over eerlijkheid](../../1-Introduction/3-fairness/README.md).
[Referentie](https://www.grammarly.com/blog/engineering/mitigating-gender-bias-in-autocorrect/)

## 👜 Retail

De retailsector kan zeker profiteren van het gebruik van ML, van het creëren van een betere klantreis tot het optimaal beheren van voorraad.

### Personaliseren van de klantreis

Bij Wayfair, een bedrijf dat huishoudelijke artikelen zoals meubels verkoopt, is het helpen van klanten om de juiste producten te vinden voor hun smaak en behoeften van groot belang. In dit artikel beschrijven ingenieurs van het bedrijf hoe ze ML en NLP gebruiken om "de juiste resultaten voor klanten naar voren te brengen". Hun Query Intent Engine is gebouwd om entiteiten te extraheren, classifiers te trainen, assets en meningen te extraheren, en sentimenten te taggen in klantbeoordelingen. Dit is een klassiek voorbeeld van hoe NLP werkt in online retail.
[Referentie](https://www.aboutwayfair.com/tech-innovation/how-we-use-machine-learning-and-natural-language-processing-to-empower-search)

### Voorraadbeheer

Innovatieve, flexibele bedrijven zoals [StitchFix](https://stitchfix.com), een abonnementsservice die kleding naar consumenten verzendt, vertrouwen sterk op ML voor aanbevelingen en voorraadbeheer. Hun stylingteams werken zelfs samen met hun merchandisingteams: "een van onze datawetenschappers experimenteerde met een genetisch algoritme en paste het toe op kleding om te voorspellen wat een succesvol kledingstuk zou zijn dat vandaag nog niet bestaat. We brachten dat naar het merchandisingteam en nu kunnen ze dat als hulpmiddel gebruiken."
[Referentie](https://www.zdnet.com/article/how-stitch-fix-uses-machine-learning-to-master-the-science-of-styling/)

## 🏥 Gezondheidszorg

De gezondheidszorgsector kan ML gebruiken om onderzoekstaken te optimaliseren en logistieke problemen zoals heropnames van patiënten of het stoppen van de verspreiding van ziekten aan te pakken.

### Beheer van klinische proeven

Toxiciteit in klinische proeven is een groot probleem voor medicijnfabrikanten. Hoeveel toxiciteit is acceptabel? In deze studie leidde het analyseren van verschillende methoden voor klinische proeven tot de ontwikkeling van een nieuwe aanpak om de kans op uitkomsten van klinische proeven te voorspellen. Ze konden specifiek random forest gebruiken om een [classifier](../../4-Classification/README.md) te produceren die groepen medicijnen onderscheidt.
[Referentie](https://www.sciencedirect.com/science/article/pii/S2451945616302914)

### Beheer van ziekenhuisheropnames

Ziekenhuiszorg is duur, vooral wanneer patiënten opnieuw moeten worden opgenomen. Dit artikel bespreekt een bedrijf dat ML gebruikt om het potentieel voor heropname te voorspellen met behulp van [clustering](../../5-Clustering/README.md) algoritmen. Deze clusters helpen analisten om "groepen heropnames te ontdekken die mogelijk een gemeenschappelijke oorzaak delen".
[Referentie](https://healthmanagement.org/c/healthmanagement/issuearticle/hospital-readmissions-and-machine-learning)

### Ziektebeheer

De recente pandemie heeft duidelijk gemaakt hoe machine learning kan helpen bij het stoppen van de verspreiding van ziekten. In dit artikel herken je het gebruik van ARIMA, logistische curves, lineaire regressie en SARIMA. "Dit werk is een poging om de verspreidingssnelheid van dit virus te berekenen en zo de sterfgevallen, herstelgevallen en bevestigde gevallen te voorspellen, zodat we ons beter kunnen voorbereiden en overleven."
[Referentie](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7979218/)

## 🌲 Ecologie en Groene Technologie

De natuur en ecologie bestaan uit veel gevoelige systemen waarin de interactie tussen dieren en de natuur centraal staat. Het is belangrijk om deze systemen nauwkeurig te kunnen meten en gepast te handelen als er iets gebeurt, zoals een bosbrand of een daling in de dierenpopulatie.

### Bosbeheer

Je hebt geleerd over [Reinforcement Learning](../../8-Reinforcement/README.md) in eerdere lessen. Het kan zeer nuttig zijn bij het voorspellen van patronen in de natuur. In het bijzonder kan het worden gebruikt om ecologische problemen zoals bosbranden en de verspreiding van invasieve soorten te volgen. In Canada gebruikte een groep onderzoekers Reinforcement Learning om modellen voor bosbranddynamiek te bouwen op basis van satellietbeelden. Met behulp van een innovatieve "spatially spreading process (SSP)" zagen ze een bosbrand als "de agent op elke cel in het landschap." "De set acties die de brand kan ondernemen vanaf een locatie op elk moment omvat verspreiding naar het noorden, zuiden, oosten of westen, of niet verspreiden."

Deze aanpak keert de gebruikelijke RL-opzet om, aangezien de dynamiek van het bijbehorende Markov Decision Process (MDP) een bekende functie is voor onmiddellijke bosbrandverspreiding. Lees meer over de klassieke algoritmen die door deze groep zijn gebruikt via de onderstaande link.
[Referentie](https://www.frontiersin.org/articles/10.3389/fict.2018.00006/full)

### Bewegingsdetectie van dieren

Hoewel deep learning een revolutie heeft veroorzaakt in het visueel volgen van dierbewegingen (je kunt je eigen [ijsbeertracker](https://docs.microsoft.com/learn/modules/build-ml-model-with-azure-stream-analytics/?WT.mc_id=academic-77952-leestott) hier bouwen), heeft klassieke ML nog steeds een plaats in deze taak.

Sensoren om bewegingen van boerderijdieren te volgen en IoT maken gebruik van dit soort visuele verwerking, maar meer basale ML-technieken zijn nuttig om data voor te verwerken. Bijvoorbeeld, in dit artikel werden houdingen van schapen gemonitord en geanalyseerd met verschillende classifier-algoritmen. Je herkent misschien de ROC-curve op pagina 335.
[Referentie](https://druckhaus-hofmann.de/gallery/31-wj-feb-2020.pdf)

### ⚡️ Energiebeheer

In onze lessen over [tijdreeksvoorspelling](../../7-TimeSeries/README.md) hebben we het concept van slimme parkeermeters gebruikt om inkomsten te genereren voor een stad op basis van het begrijpen van vraag en aanbod. Dit artikel bespreekt in detail hoe clustering, regressie en tijdreeksvoorspelling gecombineerd werden om het toekomstige energiegebruik in Ierland te voorspellen, gebaseerd op slimme meters.
[Referentie](https://www-cdn.knime.com/sites/default/files/inline-images/knime_bigdata_energy_timeseries_whitepaper.pdf)

## 💼 Verzekeringen

De verzekeringssector is een andere sector die ML gebruikt om levensvatbare financiële en actuariële modellen te bouwen en te optimaliseren.

### Volatiliteitsbeheer

MetLife, een aanbieder van levensverzekeringen, is open over hoe ze volatiliteit in hun financiële modellen analyseren en verminderen. In dit artikel zie je visualisaties van binaire en ordinale classificatie. Je zult ook visualisaties van voorspellingen ontdekken.
[Referentie](https://investments.metlife.com/content/dam/metlifecom/us/investments/insights/research-topics/macro-strategy/pdf/MetLifeInvestmentManagement_MachineLearnedRanking_070920.pdf)

## 🎨 Kunst, Cultuur en Literatuur

In de kunst, bijvoorbeeld in de journalistiek, zijn er veel interessante problemen. Het detecteren van nepnieuws is een groot probleem, omdat bewezen is dat het de mening van mensen kan beïnvloeden en zelfs democratieën kan ondermijnen. Musea kunnen ook profiteren van het gebruik van ML, van het vinden van verbanden tussen artefacten tot resourceplanning.

### Detectie van nepnieuws

Het detecteren van nepnieuws is tegenwoordig een kat-en-muisspel in de media. In dit artikel stellen onderzoekers voor dat een systeem dat verschillende van de ML-technieken die we hebben bestudeerd combineert, kan worden getest en het beste model kan worden ingezet: "Dit systeem is gebaseerd op natuurlijke taalverwerking om kenmerken uit de data te extraheren en vervolgens worden deze kenmerken gebruikt voor het trainen van machine learning-classifiers zoals Naive Bayes, Support Vector Machine (SVM), Random Forest (RF), Stochastic Gradient Descent (SGD) en Logistic Regression (LR)."
[Referentie](https://www.irjet.net/archives/V7/i6/IRJET-V7I6688.pdf)

Dit artikel laat zien hoe het combineren van verschillende ML-domeinen interessante resultaten kan opleveren die kunnen helpen om nepnieuws te stoppen en echte schade te voorkomen; in dit geval was de aanleiding de verspreiding van geruchten over COVID-behandelingen die tot geweld door menigten leidden.

### Museum ML

Musea staan aan de vooravond van een AI-revolutie waarin het catalogiseren en digitaliseren van collecties en het vinden van verbanden tussen artefacten gemakkelijker wordt naarmate technologie vordert. Projecten zoals [In Codice Ratio](https://www.sciencedirect.com/science/article/abs/pii/S0306457321001035#:~:text=1.,studies%20over%20large%20historical%20sources.) helpen de mysteries van ontoegankelijke collecties zoals de Vaticaanse archieven te ontsluiten. Maar ook het zakelijke aspect van musea profiteert van ML-modellen.

Bijvoorbeeld, het Art Institute of Chicago bouwde modellen om te voorspellen waar bezoekers geïnteresseerd in zijn en wanneer ze tentoonstellingen zullen bezoeken. Het doel is om elke keer dat de gebruiker het museum bezoekt een geïndividualiseerde en geoptimaliseerde bezoekerservaring te creëren. "Tijdens het fiscale jaar 2017 voorspelde het model de bezoekersaantallen en toegangsinkomsten met een nauwkeurigheid van 1 procent, zegt Andrew Simnick, senior vice president bij het Art Institute."
[Referentie](https://www.chicagobusiness.com/article/20180518/ISSUE01/180519840/art-institute-of-chicago-uses-data-to-make-exhibit-choices)

## 🏷 Marketing

### Klantsegmentatie

De meest effectieve marketingstrategieën richten zich op klanten op verschillende manieren op basis van diverse groeperingen. In dit artikel worden de toepassingen van clustering-algoritmen besproken om gedifferentieerde marketing te ondersteunen. Gedifferentieerde marketing helpt bedrijven hun merkbekendheid te verbeteren, meer klanten te bereiken en meer geld te verdienen.
[Referentie](https://ai.inqline.com/machine-learning-for-marketing-customer-segmentation/)

## 🚀 Uitdaging

Identificeer een andere sector die profiteert van enkele van de technieken die je in dit curriculum hebt geleerd, en ontdek hoe deze sector ML gebruikt.
## [Quiz na de les](https://ff-quizzes.netlify.app/en/ml/)

## Herziening & Zelfstudie

Het data science-team van Wayfair heeft verschillende interessante video's over hoe zij ML gebruiken binnen hun bedrijf. Het is de moeite waard om [een kijkje te nemen](https://www.youtube.com/channel/UCe2PjkQXqOuwkW1gw6Ameuw/videos)!

## Opdracht

[Een ML-speurtocht](assignment.md)

---

**Disclaimer**:  
Dit document is vertaald met behulp van de AI-vertalingsservice [Co-op Translator](https://github.com/Azure/co-op-translator). Hoewel we streven naar nauwkeurigheid, dient u zich ervan bewust te zijn dat geautomatiseerde vertalingen fouten of onnauwkeurigheden kunnen bevatten. Het originele document in zijn oorspronkelijke taal moet worden beschouwd als de gezaghebbende bron. Voor cruciale informatie wordt professionele menselijke vertaling aanbevolen. Wij zijn niet aansprakelijk voor eventuele misverstanden of verkeerde interpretaties die voortvloeien uit het gebruik van deze vertaling.