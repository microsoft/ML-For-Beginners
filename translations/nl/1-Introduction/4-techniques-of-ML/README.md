<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9d91f3af3758fdd4569fb410575995ef",
  "translation_date": "2025-09-05T19:34:36+00:00",
  "source_file": "1-Introduction/4-techniques-of-ML/README.md",
  "language_code": "nl"
}
-->
# Technieken van Machine Learning

Het proces van het bouwen, gebruiken en onderhouden van machine learning-modellen en de gegevens die ze gebruiken, verschilt sterk van veel andere ontwikkelworkflows. In deze les zullen we het proces verduidelijken en de belangrijkste technieken bespreken die je moet kennen. Je zult:

- Begrijpen welke processen ten grondslag liggen aan machine learning op een hoog niveau.
- Basisconcepten verkennen zoals 'modellen', 'voorspellingen' en 'trainingsdata'.

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

[![ML voor beginners - Technieken van Machine Learning](https://img.youtube.com/vi/4NGM0U2ZSHU/0.jpg)](https://youtu.be/4NGM0U2ZSHU "ML voor beginners - Technieken van Machine Learning")

> 🎥 Klik op de afbeelding hierboven voor een korte video over deze les.

## Introductie

Op een hoog niveau bestaat het vak van het creëren van machine learning (ML)-processen uit een aantal stappen:

1. **Bepaal de vraag**. De meeste ML-processen beginnen met het stellen van een vraag die niet kan worden beantwoord met een eenvoudig conditioneel programma of een op regels gebaseerde engine. Deze vragen draaien vaak om voorspellingen op basis van een verzameling gegevens.
2. **Verzamel en bereid gegevens voor**. Om je vraag te kunnen beantwoorden, heb je gegevens nodig. De kwaliteit en soms de hoeveelheid van je gegevens bepalen hoe goed je je oorspronkelijke vraag kunt beantwoorden. Het visualiseren van gegevens is een belangrijk aspect van deze fase. Deze fase omvat ook het splitsen van de gegevens in een trainings- en testgroep om een model te bouwen.
3. **Kies een trainingsmethode**. Afhankelijk van je vraag en de aard van je gegevens moet je kiezen hoe je een model wilt trainen om je gegevens het beste te weerspiegelen en nauwkeurige voorspellingen te doen. Dit is het deel van je ML-proces dat specifieke expertise vereist en vaak een aanzienlijke hoeveelheid experimenten.
4. **Train het model**. Met behulp van je trainingsgegevens gebruik je verschillende algoritmen om een model te trainen dat patronen in de gegevens herkent. Het model kan interne gewichten gebruiken die kunnen worden aangepast om bepaalde delen van de gegevens te bevoordelen boven andere om een beter model te bouwen.
5. **Evalueer het model**. Je gebruikt gegevens die het model nog niet eerder heeft gezien (je testgegevens) uit je verzamelde set om te zien hoe het model presteert.
6. **Parameterafstemming**. Op basis van de prestaties van je model kun je het proces opnieuw uitvoeren met verschillende parameters of variabelen die het gedrag van de algoritmen bepalen die worden gebruikt om het model te trainen.
7. **Voorspel**. Gebruik nieuwe invoer om de nauwkeurigheid van je model te testen.

## Welke vraag stel je?

Computers zijn bijzonder goed in het ontdekken van verborgen patronen in gegevens. Deze eigenschap is erg nuttig voor onderzoekers die vragen hebben over een bepaald domein die niet gemakkelijk kunnen worden beantwoord door een op regels gebaseerde engine te maken. Bij een actuarieel vraagstuk kan een datawetenschapper bijvoorbeeld handgemaakte regels opstellen over de sterfte van rokers versus niet-rokers.

Wanneer veel andere variabelen in de vergelijking worden gebracht, kan een ML-model echter efficiënter zijn om toekomstige sterftecijfers te voorspellen op basis van eerdere gezondheidsgegevens. Een vrolijker voorbeeld zou het maken van weersvoorspellingen voor de maand april op een bepaalde locatie kunnen zijn, gebaseerd op gegevens zoals breedtegraad, lengtegraad, klimaatverandering, nabijheid van de oceaan, patronen van de straalstroom en meer.

✅ Deze [presentatie](https://www2.cisl.ucar.edu/sites/default/files/2021-10/0900%20June%2024%20Haupt_0.pdf) over weermodellen biedt een historisch perspectief op het gebruik van ML in weersanalyse.  

## Taken vóór het bouwen

Voordat je begint met het bouwen van je model, zijn er verschillende taken die je moet voltooien. Om je vraag te testen en een hypothese te vormen op basis van de voorspellingen van een model, moet je verschillende elementen identificeren en configureren.

### Gegevens

Om je vraag met enige zekerheid te kunnen beantwoorden, heb je een goede hoeveelheid gegevens van het juiste type nodig. Er zijn twee dingen die je op dit punt moet doen:

- **Gegevens verzamelen**. Houd rekening met de vorige les over eerlijkheid in data-analyse en verzamel je gegevens zorgvuldig. Wees je bewust van de bronnen van deze gegevens, eventuele inherente vooroordelen en documenteer de oorsprong ervan.
- **Gegevens voorbereiden**. Er zijn verschillende stappen in het proces van gegevensvoorbereiding. Je moet mogelijk gegevens samenvoegen en normaliseren als ze uit diverse bronnen komen. Je kunt de kwaliteit en kwantiteit van de gegevens verbeteren via verschillende methoden, zoals het converteren van strings naar getallen (zoals we doen in [Clustering](../../5-Clustering/1-Visualize/README.md)). Je kunt ook nieuwe gegevens genereren op basis van de oorspronkelijke gegevens (zoals we doen in [Classificatie](../../4-Classification/1-Introduction/README.md)). Je kunt de gegevens opschonen en bewerken (zoals we doen voorafgaand aan de [Web App](../../3-Web-App/README.md) les). Tot slot kun je de gegevens willekeurig maken en schudden, afhankelijk van je trainingstechnieken.

✅ Nadat je je gegevens hebt verzameld en verwerkt, neem een moment om te zien of de vorm ervan je in staat stelt je beoogde vraag te beantwoorden. Het kan zijn dat de gegevens niet goed presteren in je gegeven taak, zoals we ontdekken in onze [Clustering](../../5-Clustering/1-Visualize/README.md) lessen!

### Kenmerken en Doel

Een [kenmerk](https://www.datasciencecentral.com/profiles/blogs/an-introduction-to-variable-and-feature-selection) is een meetbare eigenschap van je gegevens. In veel datasets wordt dit uitgedrukt als een kolomkop zoals 'datum', 'grootte' of 'kleur'. Je kenmerkvariabelen, meestal weergegeven als `X` in code, vertegenwoordigen de invoervariabelen die worden gebruikt om het model te trainen.

Een doel is datgene wat je probeert te voorspellen. Het doel, meestal weergegeven als `y` in code, vertegenwoordigt het antwoord op de vraag die je probeert te stellen over je gegevens: in december, welke **kleur** pompoenen zullen het goedkoopst zijn? In San Francisco, welke buurten zullen de beste vastgoed**prijzen** hebben? Soms wordt het doel ook wel labelattribuut genoemd.

### Selecteer je kenmerkvariabelen

🎓 **Kenmerkselectie en Kenmerkextractie** Hoe weet je welke variabelen je moet kiezen bij het bouwen van een model? Je zult waarschijnlijk een proces van kenmerkselectie of kenmerkextractie doorlopen om de juiste variabelen te kiezen voor het meest presterende model. Ze zijn echter niet hetzelfde: "Kenmerkextractie creëert nieuwe kenmerken uit functies van de oorspronkelijke kenmerken, terwijl kenmerkselectie een subset van de kenmerken retourneert." ([bron](https://wikipedia.org/wiki/Feature_selection))

### Visualiseer je gegevens

Een belangrijk aspect van de toolkit van een datawetenschapper is de kracht om gegevens te visualiseren met behulp van verschillende uitstekende bibliotheken zoals Seaborn of MatPlotLib. Het visueel weergeven van je gegevens kan je in staat stellen verborgen correlaties te ontdekken die je kunt benutten. Je visualisaties kunnen je ook helpen om vooroordelen of onevenwichtige gegevens te ontdekken (zoals we ontdekken in [Classificatie](../../4-Classification/2-Classifiers-1/README.md)).

### Splits je dataset

Voordat je gaat trainen, moet je je dataset splitsen in twee of meer delen van ongelijke grootte die de gegevens nog steeds goed vertegenwoordigen.

- **Training**. Dit deel van de dataset wordt gebruikt om je model te trainen. Dit deel vormt het grootste deel van de oorspronkelijke dataset.
- **Testen**. Een testdataset is een onafhankelijke groep gegevens, vaak verzameld uit de oorspronkelijke gegevens, die je gebruikt om de prestaties van het gebouwde model te bevestigen.
- **Valideren**. Een validatieset is een kleinere onafhankelijke groep voorbeelden die je gebruikt om de hyperparameters of architectuur van het model af te stemmen om het model te verbeteren. Afhankelijk van de grootte van je gegevens en de vraag die je stelt, hoef je deze derde set mogelijk niet te bouwen (zoals we opmerken in [Tijdreeksvoorspelling](../../7-TimeSeries/1-Introduction/README.md)).

## Een model bouwen

Met behulp van je trainingsgegevens is je doel om een model te bouwen, of een statistische representatie van je gegevens, met behulp van verschillende algoritmen om het te **trainen**. Het trainen van een model stelt het bloot aan gegevens en stelt het in staat aannames te doen over waargenomen patronen die het ontdekt, valideert en accepteert of afwijst.

### Kies een trainingsmethode

Afhankelijk van je vraag en de aard van je gegevens kies je een methode om het te trainen. Door [Scikit-learn's documentatie](https://scikit-learn.org/stable/user_guide.html) door te nemen - die we in deze cursus gebruiken - kun je veel manieren verkennen om een model te trainen. Afhankelijk van je ervaring moet je mogelijk verschillende methoden proberen om het beste model te bouwen. Je zult waarschijnlijk een proces doorlopen waarbij datawetenschappers de prestaties van een model evalueren door het ongeziene gegevens te voeren, te controleren op nauwkeurigheid, vooroordelen en andere kwaliteitsverminderende problemen, en de meest geschikte trainingsmethode voor de taak te selecteren.

### Train een model

Met je trainingsgegevens ben je klaar om ze te 'fitten' om een model te creëren. Je zult merken dat in veel ML-bibliotheken de code 'model.fit' voorkomt - op dit moment stuur je je kenmerkvariabelen als een array van waarden (meestal 'X') en een doelvariabele (meestal 'y').

### Evalueer het model

Zodra het trainingsproces is voltooid (het kan veel iteraties, of 'epochs', duren om een groot model te trainen), kun je de kwaliteit van het model evalueren door testgegevens te gebruiken om de prestaties te meten. Deze gegevens zijn een subset van de oorspronkelijke gegevens die het model nog niet eerder heeft geanalyseerd. Je kunt een tabel met statistieken over de kwaliteit van je model afdrukken.

🎓 **Model fitten**

In de context van machine learning verwijst model fitten naar de nauwkeurigheid van de onderliggende functie van het model terwijl het probeert gegevens te analyseren waarmee het niet bekend is.

🎓 **Underfitting** en **overfitting** zijn veelvoorkomende problemen die de kwaliteit van het model verminderen, omdat het model ofwel niet goed genoeg past of te goed past. Dit zorgt ervoor dat het model voorspellingen doet die ofwel te nauw aansluiten of te los aansluiten bij de trainingsgegevens. Een overfit model voorspelt trainingsgegevens te goed omdat het de details en ruis van de gegevens te goed heeft geleerd. Een underfit model is niet nauwkeurig omdat het noch zijn trainingsgegevens noch gegevens die het nog niet heeft 'gezien' nauwkeurig kan analyseren.

![overfitting model](../../../../1-Introduction/4-techniques-of-ML/images/overfitting.png)
> Infographic door [Jen Looper](https://twitter.com/jenlooper)

## Parameterafstemming

Zodra je eerste training is voltooid, observeer de kwaliteit van het model en overweeg het te verbeteren door de 'hyperparameters' aan te passen. Lees meer over het proces [in de documentatie](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters?WT.mc_id=academic-77952-leestott).

## Voorspelling

Dit is het moment waarop je volledig nieuwe gegevens kunt gebruiken om de nauwkeurigheid van je model te testen. In een 'toegepaste' ML-instelling, waar je webassets bouwt om het model in productie te gebruiken, kan dit proces het verzamelen van gebruikersinvoer omvatten (bijvoorbeeld een druk op een knop) om een variabele in te stellen en deze naar het model te sturen voor inferentie of evaluatie.

In deze lessen ontdek je hoe je deze stappen kunt gebruiken om voor te bereiden, te bouwen, te testen, te evalueren en te voorspellen - alle handelingen van een datawetenschapper en meer, terwijl je vordert in je reis om een 'full stack' ML-engineer te worden.

---

## 🚀Uitdaging

Teken een stroomdiagram dat de stappen van een ML-practitioner weergeeft. Waar zie je jezelf op dit moment in het proces? Waar denk je dat je moeilijkheden zult ondervinden? Wat lijkt je gemakkelijk?

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Review & Zelfstudie

Zoek online naar interviews met datawetenschappers die hun dagelijkse werk bespreken. Hier is [een](https://www.youtube.com/watch?v=Z3IjgbbCEfs).

## Opdracht

[Interview een datawetenschapper](assignment.md)

---

**Disclaimer**:  
Dit document is vertaald met behulp van de AI-vertalingsservice [Co-op Translator](https://github.com/Azure/co-op-translator). Hoewel we streven naar nauwkeurigheid, dient u zich ervan bewust te zijn dat geautomatiseerde vertalingen fouten of onnauwkeurigheden kunnen bevatten. Het originele document in de oorspronkelijke taal moet worden beschouwd als de gezaghebbende bron. Voor cruciale informatie wordt professionele menselijke vertaling aanbevolen. Wij zijn niet aansprakelijk voor misverstanden of verkeerde interpretaties die voortvloeien uit het gebruik van deze vertaling.