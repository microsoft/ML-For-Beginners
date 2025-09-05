<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9a6b702d1437c0467e3c5c28d763dac2",
  "translation_date": "2025-09-05T19:30:29+00:00",
  "source_file": "1-Introduction/3-fairness/README.md",
  "language_code": "nl"
}
-->
# Machine Learning-oplossingen bouwen met verantwoorde AI

![Samenvatting van verantwoorde AI in Machine Learning in een sketchnote](../../../../sketchnotes/ml-fairness.png)
> Sketchnote door [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Introductie

In dit curriculum ontdek je hoe machine learning ons dagelijks leven beïnvloedt. Systemen en modellen spelen nu al een rol in dagelijkse besluitvormingsprocessen, zoals medische diagnoses, leninggoedkeuringen of het opsporen van fraude. Het is daarom belangrijk dat deze modellen goed functioneren en betrouwbare resultaten leveren. Net zoals bij elke softwaretoepassing kunnen AI-systemen niet aan verwachtingen voldoen of ongewenste uitkomsten hebben. Daarom is het essentieel om het gedrag van een AI-model te begrijpen en uit te leggen.

Stel je voor wat er kan gebeuren als de gegevens die je gebruikt om deze modellen te bouwen bepaalde demografische gegevens missen, zoals ras, geslacht, politieke overtuiging, religie, of als ze deze disproportioneel vertegenwoordigen. Wat gebeurt er als de output van het model wordt geïnterpreteerd om een bepaalde demografische groep te bevoordelen? Wat zijn de gevolgen voor de toepassing? En wat gebeurt er als het model een nadelige uitkomst heeft en schadelijk is voor mensen? Wie is verantwoordelijk voor het gedrag van AI-systemen? Dit zijn enkele vragen die we in dit curriculum zullen verkennen.

In deze les leer je:

- Het belang van eerlijkheid in machine learning en de schade die oneerlijkheid kan veroorzaken.
- Het verkennen van uitschieters en ongebruikelijke scenario's om betrouwbaarheid en veiligheid te waarborgen.
- Het belang van inclusieve systemen ontwerpen om iedereen te empoweren.
- Hoe cruciaal het is om de privacy en veiligheid van gegevens en mensen te beschermen.
- Het belang van een transparante aanpak om het gedrag van AI-modellen uit te leggen.
- Hoe verantwoordelijkheid essentieel is om vertrouwen in AI-systemen op te bouwen.

## Vereisten

Als vereiste, volg het "Responsible AI Principles" leerpad en bekijk de onderstaande video over dit onderwerp:

Leer meer over verantwoorde AI via dit [leerpad](https://docs.microsoft.com/learn/modules/responsible-ai-principles/?WT.mc_id=academic-77952-leestott)

[![Microsoft's Approach to Responsible AI](https://img.youtube.com/vi/dnC8-uUZXSc/0.jpg)](https://youtu.be/dnC8-uUZXSc "Microsoft's Approach to Responsible AI")

> 🎥 Klik op de afbeelding hierboven voor een video: Microsoft's Approach to Responsible AI

## Eerlijkheid

AI-systemen moeten iedereen eerlijk behandelen en vermijden dat vergelijkbare groepen mensen verschillend worden beïnvloed. Bijvoorbeeld, wanneer AI-systemen advies geven over medische behandelingen, leningaanvragen of werkgelegenheid, moeten ze dezelfde aanbevelingen doen aan iedereen met vergelijkbare symptomen, financiële omstandigheden of professionele kwalificaties. Elk van ons draagt inherente vooroordelen met zich mee die onze beslissingen en acties beïnvloeden. Deze vooroordelen kunnen zichtbaar zijn in de gegevens die we gebruiken om AI-systemen te trainen. Dergelijke manipulatie kan soms onbedoeld gebeuren. Het is vaak moeilijk om bewust te weten wanneer je vooroordelen in gegevens introduceert.

**"Oneerlijkheid"** omvat negatieve gevolgen, of "schade", voor een groep mensen, zoals die gedefinieerd in termen van ras, geslacht, leeftijd of handicapstatus. De belangrijkste vormen van schade gerelateerd aan eerlijkheid kunnen worden geclassificeerd als:

- **Toewijzing**, bijvoorbeeld als een geslacht of etniciteit wordt bevoordeeld boven een andere.
- **Kwaliteit van dienstverlening**. Als je gegevens traint voor één specifiek scenario, maar de werkelijkheid veel complexer is, leidt dit tot een slecht presterende dienst. Bijvoorbeeld een handzeepdispenser die mensen met een donkere huid niet lijkt te kunnen detecteren. [Referentie](https://gizmodo.com/why-cant-this-soap-dispenser-identify-dark-skin-1797931773)
- **Denigratie**. Het oneerlijk bekritiseren en labelen van iets of iemand. Bijvoorbeeld, een beeldlabeltechnologie die berucht is om het verkeerd labelen van afbeeldingen van mensen met een donkere huid als gorilla's.
- **Over- of ondervertegenwoordiging**. Het idee dat een bepaalde groep niet wordt gezien in een bepaald beroep, en elke dienst of functie die dat blijft promoten, draagt bij aan schade.
- **Stereotypering**. Het associëren van een bepaalde groep met vooraf toegewezen eigenschappen. Bijvoorbeeld, een taalvertalingssysteem tussen Engels en Turks kan onnauwkeurigheden hebben door woorden met stereotypische associaties met geslacht.

![vertaling naar Turks](../../../../1-Introduction/3-fairness/images/gender-bias-translate-en-tr.png)
> vertaling naar Turks

![vertaling terug naar Engels](../../../../1-Introduction/3-fairness/images/gender-bias-translate-tr-en.png)
> vertaling terug naar Engels

Bij het ontwerpen en testen van AI-systemen moeten we ervoor zorgen dat AI eerlijk is en niet geprogrammeerd is om bevooroordeelde of discriminerende beslissingen te nemen, die ook verboden zijn voor mensen. Eerlijkheid garanderen in AI en machine learning blijft een complexe sociaal-technische uitdaging.

### Betrouwbaarheid en veiligheid

Om vertrouwen op te bouwen, moeten AI-systemen betrouwbaar, veilig en consistent zijn onder normale en onverwachte omstandigheden. Het is belangrijk om te weten hoe AI-systemen zich gedragen in verschillende situaties, vooral wanneer ze uitschieters zijn. Bij het bouwen van AI-oplossingen moet er veel aandacht worden besteed aan hoe om te gaan met een breed scala aan omstandigheden waarmee de AI-oplossingen te maken kunnen krijgen. Bijvoorbeeld, een zelfrijdende auto moet de veiligheid van mensen als topprioriteit stellen. Als gevolg hiervan moet de AI die de auto aandrijft rekening houden met alle mogelijke scenario's waarmee de auto te maken kan krijgen, zoals nacht, onweer of sneeuwstormen, kinderen die de straat oversteken, huisdieren, wegwerkzaamheden, enzovoort. Hoe goed een AI-systeem een breed scala aan omstandigheden betrouwbaar en veilig kan verwerken, weerspiegelt het niveau van anticipatie dat de datawetenschapper of AI-ontwikkelaar heeft overwogen tijdens het ontwerp of de test van het systeem.

> [🎥 Klik hier voor een video: ](https://www.microsoft.com/videoplayer/embed/RE4vvIl)

### Inclusiviteit

AI-systemen moeten worden ontworpen om iedereen te betrekken en te empoweren. Bij het ontwerpen en implementeren van AI-systemen identificeren en aanpakken datawetenschappers en AI-ontwikkelaars potentiële barrières in het systeem die mensen onbedoeld kunnen uitsluiten. Bijvoorbeeld, er zijn 1 miljard mensen met een handicap wereldwijd. Met de vooruitgang van AI kunnen zij gemakkelijker toegang krijgen tot een breed scala aan informatie en kansen in hun dagelijks leven. Door de barrières aan te pakken, ontstaan er mogelijkheden om te innoveren en AI-producten te ontwikkelen met betere ervaringen die iedereen ten goede komen.

> [🎥 Klik hier voor een video: inclusiviteit in AI](https://www.microsoft.com/videoplayer/embed/RE4vl9v)

### Veiligheid en privacy

AI-systemen moeten veilig zijn en de privacy van mensen respecteren. Mensen hebben minder vertrouwen in systemen die hun privacy, informatie of leven in gevaar brengen. Bij het trainen van machine learning-modellen vertrouwen we op gegevens om de beste resultaten te produceren. Daarbij moet rekening worden gehouden met de herkomst en integriteit van de gegevens. Bijvoorbeeld, zijn de gegevens door gebruikers ingediend of openbaar beschikbaar? Vervolgens, bij het werken met de gegevens, is het cruciaal om AI-systemen te ontwikkelen die vertrouwelijke informatie kunnen beschermen en aanvallen kunnen weerstaan. Naarmate AI steeds vaker voorkomt, wordt het beschermen van privacy en het beveiligen van belangrijke persoonlijke en zakelijke informatie steeds belangrijker en complexer. Privacy- en gegevensbeveiligingskwesties vereisen speciale aandacht voor AI, omdat toegang tot gegevens essentieel is voor AI-systemen om nauwkeurige en geïnformeerde voorspellingen en beslissingen over mensen te maken.

> [🎥 Klik hier voor een video: veiligheid in AI](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Als industrie hebben we aanzienlijke vooruitgang geboekt op het gebied van privacy en veiligheid, grotendeels gestimuleerd door regelgeving zoals de GDPR (General Data Protection Regulation).
- Toch moeten we bij AI-systemen de spanning erkennen tussen de behoefte aan meer persoonlijke gegevens om systemen persoonlijker en effectiever te maken – en privacy.
- Net zoals bij de geboorte van verbonden computers met het internet, zien we ook een enorme toename van het aantal beveiligingsproblemen met betrekking tot AI.
- Tegelijkertijd hebben we gezien dat AI wordt gebruikt om de beveiliging te verbeteren. Bijvoorbeeld, de meeste moderne antivirusprogramma's worden tegenwoordig aangedreven door AI-heuristieken.
- We moeten ervoor zorgen dat onze data science-processen harmonieus samengaan met de nieuwste privacy- en beveiligingspraktijken.

### Transparantie

AI-systemen moeten begrijpelijk zijn. Een cruciaal onderdeel van transparantie is het uitleggen van het gedrag van AI-systemen en hun componenten. Het verbeteren van het begrip van AI-systemen vereist dat belanghebbenden begrijpen hoe en waarom ze functioneren, zodat ze potentiële prestatieproblemen, veiligheids- en privacykwesties, vooroordelen, uitsluitingspraktijken of onbedoelde uitkomsten kunnen identificeren. We geloven ook dat degenen die AI-systemen gebruiken eerlijk en open moeten zijn over wanneer, waarom en hoe ze ervoor kiezen om ze in te zetten. Evenals de beperkingen van de systemen die ze gebruiken. Bijvoorbeeld, als een bank een AI-systeem gebruikt om zijn beslissingen over consumentenkredieten te ondersteunen, is het belangrijk om de uitkomsten te onderzoeken en te begrijpen welke gegevens de aanbevelingen van het systeem beïnvloeden. Overheden beginnen AI in verschillende sectoren te reguleren, dus datawetenschappers en organisaties moeten uitleggen of een AI-systeem voldoet aan de regelgeving, vooral wanneer er een ongewenste uitkomst is.

> [🎥 Klik hier voor een video: transparantie in AI](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Omdat AI-systemen zo complex zijn, is het moeilijk te begrijpen hoe ze werken en de resultaten te interpreteren.
- Dit gebrek aan begrip beïnvloedt de manier waarop deze systemen worden beheerd, operationeel worden gemaakt en gedocumenteerd.
- Dit gebrek aan begrip beïnvloedt nog belangrijker de beslissingen die worden genomen op basis van de resultaten die deze systemen produceren.

### Verantwoordelijkheid

De mensen die AI-systemen ontwerpen en implementeren moeten verantwoordelijk zijn voor hoe hun systemen functioneren. De behoefte aan verantwoordelijkheid is vooral cruciaal bij gevoelige technologieën zoals gezichtsherkenning. Recentelijk is er een groeiende vraag naar gezichtsherkenningstechnologie, vooral van wetshandhavingsorganisaties die het potentieel van de technologie zien in toepassingen zoals het vinden van vermiste kinderen. Deze technologieën kunnen echter mogelijk door een overheid worden gebruikt om de fundamentele vrijheden van haar burgers in gevaar te brengen, bijvoorbeeld door voortdurende surveillance van specifieke individuen mogelijk te maken. Daarom moeten datawetenschappers en organisaties verantwoordelijk zijn voor hoe hun AI-systeem individuen of de samenleving beïnvloedt.

[![Leading AI Researcher Warns of Mass Surveillance Through Facial Recognition](../../../../1-Introduction/3-fairness/images/accountability.png)](https://www.youtube.com/watch?v=Wldt8P5V6D0 "Microsoft's Approach to Responsible AI")

> 🎥 Klik op de afbeelding hierboven voor een video: Waarschuwingen over massale surveillance door gezichtsherkenning

Uiteindelijk is een van de grootste vragen voor onze generatie, als de eerste generatie die AI naar de samenleving brengt, hoe we ervoor kunnen zorgen dat computers verantwoordelijk blijven tegenover mensen en hoe we ervoor kunnen zorgen dat de mensen die computers ontwerpen verantwoordelijk blijven tegenover iedereen.

## Impactanalyse

Voordat je een machine learning-model traint, is het belangrijk om een impactanalyse uit te voeren om het doel van het AI-systeem te begrijpen; wat het beoogde gebruik is; waar het zal worden ingezet; en wie met het systeem zal omgaan. Dit is nuttig voor beoordelaars of testers die het systeem evalueren om te weten welke factoren ze in overweging moeten nemen bij het identificeren van potentiële risico's en verwachte gevolgen.

De volgende gebieden zijn van belang bij het uitvoeren van een impactanalyse:

* **Nadelige impact op individuen**. Bewust zijn van eventuele beperkingen of vereisten, niet-ondersteund gebruik of bekende beperkingen die de prestaties van het systeem belemmeren, is essentieel om ervoor te zorgen dat het systeem niet op een manier wordt gebruikt die individuen kan schaden.
* **Gegevensvereisten**. Begrijpen hoe en waar het systeem gegevens zal gebruiken stelt beoordelaars in staat om eventuele gegevensvereisten te onderzoeken waar je rekening mee moet houden (bijv. GDPR- of HIPPA-gegevensreguleringen). Daarnaast moet worden onderzocht of de bron of hoeveelheid gegevens voldoende is voor training.
* **Samenvatting van impact**. Verzamel een lijst van potentiële schade die kan ontstaan door het gebruik van het systeem. Tijdens de ML-levenscyclus moet worden beoordeeld of de geïdentificeerde problemen worden aangepakt of opgelost.
* **Toepasbare doelen** voor elk van de zes kernprincipes. Beoordeel of de doelen van elk van de principes worden gehaald en of er hiaten zijn.

## Debuggen met verantwoorde AI

Net zoals bij het debuggen van een softwaretoepassing, is het debuggen van een AI-systeem een noodzakelijk proces om problemen in het systeem te identificeren en op te lossen. Er zijn veel factoren die ervoor kunnen zorgen dat een model niet presteert zoals verwacht of niet verantwoordelijk is. De meeste traditionele modelprestatiemetrics zijn kwantitatieve aggregaten van de prestaties van een model, die niet voldoende zijn om te analyseren hoe een model de principes van verantwoorde AI schendt. Bovendien is een machine learning-model een black box, wat het moeilijk maakt om te begrijpen wat de uitkomst beïnvloedt of uitleg te geven wanneer het een fout maakt. Later in deze cursus leren we hoe we het Responsible AI-dashboard kunnen gebruiken om AI-systemen te debuggen. Het dashboard biedt een holistisch hulpmiddel voor datawetenschappers en AI-ontwikkelaars om:

* **Foutanalyse**. Om de foutverdeling van het model te identificeren die de eerlijkheid of betrouwbaarheid van het systeem kan beïnvloeden.
* **Modeloverzicht**. Om te ontdekken waar er verschillen zijn in de prestaties van het model over verschillende datacohorten.
* **Gegevensanalyse**. Om de gegevensverdeling te begrijpen en eventuele potentiële vooroordelen in de gegevens te identificeren die kunnen leiden tot problemen met eerlijkheid, inclusiviteit en betrouwbaarheid.
* **Modelinterpretatie**. Om te begrijpen wat de voorspellingen van het model beïnvloedt. Dit helpt bij het uitleggen van het gedrag van het model, wat belangrijk is voor transparantie en verantwoordelijkheid.

## 🚀 Uitdaging

Om schade te voorkomen voordat deze wordt geïntroduceerd, moeten we:

- een diversiteit aan achtergronden en perspectieven hebben onder de mensen die aan systemen werken
- investeren in datasets die de diversiteit van onze samenleving weerspiegelen
- betere methoden ontwikkelen gedurende de hele machine learning-levenscyclus om verantwoorde AI te detecteren en te corrigeren wanneer deze optreedt

Denk na over real-life scenario's waarin de onbetrouwbaarheid van een model duidelijk is in modelbouw en gebruik. Wat moeten we nog meer overwegen?

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Review & Zelfstudie

In deze les heb je enkele basisconcepten geleerd over eerlijkheid en oneerlijkheid in machine learning.
Bekijk deze workshop om dieper in te gaan op de onderwerpen:

- Op zoek naar verantwoorde AI: Principes in de praktijk brengen door Besmira Nushi, Mehrnoosh Sameki en Amit Sharma

[![Responsible AI Toolbox: Een open-source framework voor het bouwen van verantwoorde AI](https://img.youtube.com/vi/tGgJCrA-MZU/0.jpg)](https://www.youtube.com/watch?v=tGgJCrA-MZU "RAI Toolbox: Een open-source framework voor het bouwen van verantwoorde AI")

> 🎥 Klik op de afbeelding hierboven voor een video: RAI Toolbox: Een open-source framework voor het bouwen van verantwoorde AI door Besmira Nushi, Mehrnoosh Sameki en Amit Sharma

Lees ook:

- Microsoft’s RAI resource center: [Responsible AI Resources – Microsoft AI](https://www.microsoft.com/ai/responsible-ai-resources?activetab=pivot1%3aprimaryr4)

- Microsoft’s FATE onderzoeksgroep: [FATE: Fairness, Accountability, Transparency, and Ethics in AI - Microsoft Research](https://www.microsoft.com/research/theme/fate/)

RAI Toolbox:

- [Responsible AI Toolbox GitHub repository](https://github.com/microsoft/responsible-ai-toolbox)

Lees meer over de tools van Azure Machine Learning om eerlijkheid te waarborgen:

- [Azure Machine Learning](https://docs.microsoft.com/azure/machine-learning/concept-fairness-ml?WT.mc_id=academic-77952-leestott)

## Opdracht

[Verken RAI Toolbox](assignment.md)

---

**Disclaimer**:  
Dit document is vertaald met behulp van de AI-vertalingsservice [Co-op Translator](https://github.com/Azure/co-op-translator). Hoewel we streven naar nauwkeurigheid, dient u zich ervan bewust te zijn dat geautomatiseerde vertalingen fouten of onnauwkeurigheden kunnen bevatten. Het originele document in zijn oorspronkelijke taal moet worden beschouwd als de gezaghebbende bron. Voor cruciale informatie wordt professionele menselijke vertaling aanbevolen. Wij zijn niet aansprakelijk voor eventuele misverstanden of verkeerde interpretaties die voortvloeien uit het gebruik van deze vertaling.