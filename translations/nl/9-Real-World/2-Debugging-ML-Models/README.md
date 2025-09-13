<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "df2b538e8fbb3e91cf0419ae2f858675",
  "translation_date": "2025-09-05T19:25:07+00:00",
  "source_file": "9-Real-World/2-Debugging-ML-Models/README.md",
  "language_code": "nl"
}
-->
# Postscript: Model Debugging in Machine Learning met Responsible AI-dashboardcomponenten

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Introductie

Machine learning beïnvloedt ons dagelijks leven. AI vindt zijn weg naar enkele van de belangrijkste systemen die ons als individu en onze samenleving raken, zoals gezondheidszorg, financiën, onderwijs en werkgelegenheid. Bijvoorbeeld, systemen en modellen worden gebruikt bij dagelijkse besluitvormingsprocessen, zoals medische diagnoses of het opsporen van fraude. Hierdoor worden de vooruitgangen in AI, samen met de versnelde adoptie, geconfronteerd met veranderende maatschappelijke verwachtingen en groeiende regelgeving. We zien voortdurend gebieden waar AI-systemen niet aan de verwachtingen voldoen; ze brengen nieuwe uitdagingen aan het licht; en overheden beginnen AI-oplossingen te reguleren. Het is daarom belangrijk dat deze modellen worden geanalyseerd om eerlijke, betrouwbare, inclusieve, transparante en verantwoorde resultaten voor iedereen te bieden.

In dit curriculum bekijken we praktische tools die kunnen worden gebruikt om te beoordelen of een model problemen heeft met Responsible AI. Traditionele debuggingtechnieken voor machine learning zijn vaak gebaseerd op kwantitatieve berekeningen zoals geaggregeerde nauwkeurigheid of gemiddelde foutverlies. Stel je voor wat er kan gebeuren als de gegevens die je gebruikt om deze modellen te bouwen bepaalde demografische gegevens missen, zoals ras, geslacht, politieke opvattingen, religie, of als deze demografische gegevens onevenredig worden vertegenwoordigd. Wat als de output van het model wordt geïnterpreteerd om een bepaalde demografische groep te bevoordelen? Dit kan leiden tot een over- of ondervertegenwoordiging van deze gevoelige kenmerken, wat resulteert in problemen met eerlijkheid, inclusiviteit of betrouwbaarheid van het model. Een ander probleem is dat machine learning-modellen vaak worden beschouwd als "black boxes", wat het moeilijk maakt om te begrijpen en uit te leggen wat de voorspellingen van een model beïnvloedt. Dit zijn allemaal uitdagingen waarmee datawetenschappers en AI-ontwikkelaars worden geconfronteerd wanneer ze niet over voldoende tools beschikken om de eerlijkheid of betrouwbaarheid van een model te debuggen en te beoordelen.

In deze les leer je hoe je je modellen kunt debuggen met behulp van:

- **Foutanalyse**: Identificeer waar in je dataverdeling het model hoge foutpercentages heeft.
- **Modeloverzicht**: Voer vergelijkende analyses uit over verschillende datacohorten om verschillen in de prestatiemetrics van je model te ontdekken.
- **Data-analyse**: Onderzoek waar er mogelijk een over- of ondervertegenwoordiging van je gegevens is die je model kan beïnvloeden om een bepaalde demografische groep te bevoordelen boven een andere.
- **Feature-importance**: Begrijp welke kenmerken de voorspellingen van je model beïnvloeden op een globaal of lokaal niveau.

## Vereisten

Als vereiste, neem de review [Responsible AI tools voor ontwikkelaars](https://www.microsoft.com/ai/ai-lab-responsible-ai-dashboard)

> ![Gif over Responsible AI Tools](../../../../9-Real-World/2-Debugging-ML-Models/images/rai-overview.gif)

## Foutanalyse

Traditionele prestatiemetrics voor modellen die worden gebruikt om nauwkeurigheid te meten, zijn meestal berekeningen gebaseerd op correcte versus incorrecte voorspellingen. Bijvoorbeeld, het bepalen dat een model 89% van de tijd nauwkeurig is met een foutverlies van 0,001 kan worden beschouwd als een goede prestatie. Fouten zijn echter vaak niet gelijkmatig verdeeld in je onderliggende dataset. Je kunt een nauwkeurigheidsscore van 89% krijgen, maar ontdekken dat er verschillende gebieden in je gegevens zijn waar het model 42% van de tijd faalt. De gevolgen van deze foutpatronen bij bepaalde datagroepen kunnen leiden tot problemen met eerlijkheid of betrouwbaarheid. Het is essentieel om te begrijpen in welke gebieden het model goed of slecht presteert. De datagebieden waar er een hoog aantal onnauwkeurigheden in je model zijn, kunnen belangrijke demografische gegevens blijken te zijn.

![Analyseer en debug model fouten](../../../../9-Real-World/2-Debugging-ML-Models/images/ea-error-distribution.png)

De Foutanalysecomponent op het RAI-dashboard illustreert hoe modelfalen is verdeeld over verschillende cohorten met een boomvisualisatie. Dit is nuttig om kenmerken of gebieden te identificeren waar er een hoog foutpercentage is in je dataset. Door te zien waar de meeste onnauwkeurigheden van het model vandaan komen, kun je beginnen met het onderzoeken van de oorzaak. Je kunt ook cohorten van gegevens maken om analyses uit te voeren. Deze datacohorten helpen bij het debuggen om te bepalen waarom de modelprestaties goed zijn in het ene cohort, maar foutief in een ander.

![Foutanalyse](../../../../9-Real-World/2-Debugging-ML-Models/images/ea-error-cohort.png)

De visuele indicatoren op de boomkaart helpen om probleemgebieden sneller te lokaliseren. Bijvoorbeeld, hoe donkerder de rode kleur van een boomknooppunt, hoe hoger het foutpercentage.

Een heatmap is een andere visualisatiefunctie die gebruikers kunnen gebruiken om het foutpercentage te onderzoeken met behulp van één of twee kenmerken om een bijdrage aan de model fouten te vinden over een hele dataset of cohorten.

![Foutanalyse Heatmap](../../../../9-Real-World/2-Debugging-ML-Models/images/ea-heatmap.png)

Gebruik foutanalyse wanneer je:

* Een diepgaand begrip wilt krijgen van hoe modelfalen is verdeeld over een dataset en over verschillende invoer- en kenmerkdimensies.
* De geaggregeerde prestatiemetrics wilt opsplitsen om automatisch foutieve cohorten te ontdekken en gerichte stappen voor mitigatie te informeren.

## Modeloverzicht

Het evalueren van de prestaties van een machine learning-model vereist een holistisch begrip van zijn gedrag. Dit kan worden bereikt door meer dan één metric te bekijken, zoals foutpercentage, nauwkeurigheid, recall, precisie of MAE (Mean Absolute Error), om verschillen tussen prestatiemetrics te vinden. Eén prestatiemetric kan er goed uitzien, maar onnauwkeurigheden kunnen worden blootgelegd in een andere metric. Bovendien helpt het vergelijken van de metrics voor verschillen over de hele dataset of cohorten om licht te werpen op waar het model goed of slecht presteert. Dit is vooral belangrijk om de prestaties van het model te zien tussen gevoelige versus ongevoelige kenmerken (bijv. ras, geslacht of leeftijd van patiënten) om mogelijke oneerlijkheid van het model te onthullen. Bijvoorbeeld, het ontdekken dat het model meer fouten maakt in een cohort met gevoelige kenmerken kan mogelijke oneerlijkheid van het model onthullen.

De Modeloverzichtcomponent van het RAI-dashboard helpt niet alleen bij het analyseren van de prestatiemetrics van de gegevensrepresentatie in een cohort, maar geeft gebruikers ook de mogelijkheid om het gedrag van het model te vergelijken over verschillende cohorten.

![Dataset cohorten - modeloverzicht in RAI-dashboard](../../../../9-Real-World/2-Debugging-ML-Models/images/model-overview-dataset-cohorts.png)

De functie-gebaseerde analysefunctionaliteit van de component stelt gebruikers in staat om gegevenssubgroepen binnen een specifiek kenmerk te verfijnen om anomalieën op een gedetailleerd niveau te identificeren. Bijvoorbeeld, het dashboard heeft ingebouwde intelligentie om automatisch cohorten te genereren voor een door de gebruiker geselecteerd kenmerk (bijv. *"time_in_hospital < 3"* of *"time_in_hospital >= 7"*). Dit stelt een gebruiker in staat om een specifiek kenmerk te isoleren van een grotere gegevensgroep om te zien of het een belangrijke beïnvloeder is van de foutieve uitkomsten van het model.

![Feature cohorten - modeloverzicht in RAI-dashboard](../../../../9-Real-World/2-Debugging-ML-Models/images/model-overview-feature-cohorts.png)

De Modeloverzichtcomponent ondersteunt twee klassen van verschilmetrics:

**Verschil in modelprestaties**: Deze sets van metrics berekenen het verschil in de waarden van de geselecteerde prestatiemetric over subgroepen van gegevens. Hier zijn enkele voorbeelden:

* Verschil in nauwkeurigheidspercentage
* Verschil in foutpercentage
* Verschil in precisie
* Verschil in recall
* Verschil in gemiddelde absolute fout (MAE)

**Verschil in selectieratio**: Deze metric bevat het verschil in selectieratio (gunstige voorspelling) tussen subgroepen. Een voorbeeld hiervan is het verschil in goedkeuringspercentages voor leningen. Selectieratio betekent het aandeel gegevenspunten in elke klasse die als 1 worden geclassificeerd (bij binaire classificatie) of de verdeling van voorspelde waarden (bij regressie).

## Data-analyse

> "Als je de gegevens lang genoeg martelt, zullen ze alles bekennen" - Ronald Coase

Deze uitspraak klinkt extreem, maar het is waar dat gegevens kunnen worden gemanipuleerd om elke conclusie te ondersteunen. Dergelijke manipulatie kan soms onbedoeld gebeuren. Als mensen hebben we allemaal vooroordelen, en het is vaak moeilijk om bewust te weten wanneer je vooroordelen in gegevens introduceert. Eerlijkheid garanderen in AI en machine learning blijft een complexe uitdaging.

Gegevens zijn een grote blinde vlek voor traditionele prestatiemetrics van modellen. Je kunt hoge nauwkeurigheidsscores hebben, maar dit weerspiegelt niet altijd de onderliggende gegevensbias die in je dataset kan zitten. Bijvoorbeeld, als een dataset van werknemers 27% vrouwen in leidinggevende posities in een bedrijf heeft en 73% mannen op hetzelfde niveau, kan een AI-model voor vacatureadvertenties dat op deze gegevens is getraind, voornamelijk een mannelijk publiek targeten voor senior functies. Deze onbalans in gegevens heeft de voorspelling van het model beïnvloed om één geslacht te bevoordelen. Dit onthult een eerlijkheidsprobleem waarbij er een genderbias in het AI-model is.

De Data-analysecomponent op het RAI-dashboard helpt om gebieden te identificeren waar er een over- en ondervertegenwoordiging in de dataset is. Het helpt gebruikers om de oorzaak van fouten en eerlijkheidsproblemen te diagnosticeren die worden geïntroduceerd door gegevensonevenwichtigheden of een gebrek aan vertegenwoordiging van een bepaalde gegevensgroep. Dit geeft gebruikers de mogelijkheid om datasets te visualiseren op basis van voorspelde en werkelijke uitkomsten, foutgroepen en specifieke kenmerken. Soms kan het ontdekken van een ondervertegenwoordigde gegevensgroep ook onthullen dat het model niet goed leert, vandaar de hoge onnauwkeurigheden. Een model met gegevensbias is niet alleen een eerlijkheidsprobleem, maar toont ook aan dat het model niet inclusief of betrouwbaar is.

![Data-analysecomponent op RAI-dashboard](../../../../9-Real-World/2-Debugging-ML-Models/images/dataanalysis-cover.png)

Gebruik data-analyse wanneer je:

* De statistieken van je dataset wilt verkennen door verschillende filters te selecteren om je gegevens in verschillende dimensies (ook wel cohorten genoemd) op te splitsen.
* De verdeling van je dataset over verschillende cohorten en kenmerkengroepen wilt begrijpen.
* Wilt bepalen of je bevindingen met betrekking tot eerlijkheid, foutanalyse en causaliteit (afgeleid van andere dashboardcomponenten) het resultaat zijn van de verdeling van je dataset.
* Wilt beslissen in welke gebieden je meer gegevens moet verzamelen om fouten te verminderen die voortkomen uit representatieproblemen, labelruis, kenmerkruis, labelbias en vergelijkbare factoren.

## Modelinterpretatie

Machine learning-modellen worden vaak beschouwd als "black boxes". Begrijpen welke belangrijke gegevenskenmerken de voorspelling van een model beïnvloeden kan een uitdaging zijn. Het is belangrijk om transparantie te bieden over waarom een model een bepaalde voorspelling doet. Bijvoorbeeld, als een AI-systeem voorspelt dat een diabetespatiënt het risico loopt om binnen 30 dagen opnieuw in een ziekenhuis te worden opgenomen, moet het ondersteunende gegevens kunnen bieden die tot zijn voorspelling hebben geleid. Het hebben van ondersteunende gegevensindicatoren brengt transparantie, zodat clinici of ziekenhuizen goed geïnformeerde beslissingen kunnen nemen. Bovendien stelt het kunnen uitleggen waarom een model een voorspelling deed voor een individuele patiënt verantwoording mogelijk met gezondheidsregelgeving. Wanneer je machine learning-modellen gebruikt op manieren die het leven van mensen beïnvloeden, is het cruciaal om te begrijpen en uit te leggen wat het gedrag van een model beïnvloedt. Modeluitlegbaarheid en interpretatie helpen vragen te beantwoorden in scenario's zoals:

* Modeldebugging: Waarom maakte mijn model deze fout? Hoe kan ik mijn model verbeteren?
* Samenwerking tussen mens en AI: Hoe kan ik de beslissingen van het model begrijpen en vertrouwen?
* Regelgevingsnaleving: Voldoet mijn model aan wettelijke vereisten?

De Feature-importancecomponent van het RAI-dashboard helpt je om te debuggen en een uitgebreid begrip te krijgen van hoe een model voorspellingen maakt. Het is ook een nuttige tool voor machine learning-professionals en besluitvormers om uit te leggen en bewijs te tonen van kenmerken die het gedrag van een model beïnvloeden voor regelgevingsnaleving. Gebruikers kunnen vervolgens zowel globale als lokale verklaringen verkennen om te valideren welke kenmerken de voorspelling van een model beïnvloeden. Globale verklaringen tonen de belangrijkste kenmerken die de algehele voorspelling van een model beïnvloeden. Lokale verklaringen tonen welke kenmerken hebben geleid tot de voorspelling van een model voor een individueel geval. Het vermogen om lokale verklaringen te evalueren is ook nuttig bij het debuggen of auditen van een specifiek geval om beter te begrijpen en te interpreteren waarom een model een nauwkeurige of onnauwkeurige voorspelling deed.

![Feature-importancecomponent van het RAI-dashboard](../../../../9-Real-World/2-Debugging-ML-Models/images/9-feature-importance.png)

* Globale verklaringen: Bijvoorbeeld, welke kenmerken beïnvloeden het algehele gedrag van een diabetes ziekenhuisopname-model?
* Lokale verklaringen: Bijvoorbeeld, waarom werd een diabetespatiënt ouder dan 60 jaar met eerdere ziekenhuisopnames voorspeld om binnen 30 dagen opnieuw te worden opgenomen of niet opgenomen in een ziekenhuis?

In het debugproces van het onderzoeken van de prestaties van een model over verschillende cohorten, toont Feature-importance welk niveau van impact een kenmerk heeft over de cohorten. Het helpt om anomalieën te onthullen bij het vergelijken van het niveau van invloed dat het kenmerk heeft bij het sturen van foutieve voorspellingen van een model. De Feature-importancecomponent kan tonen welke waarden in een kenmerk positief of negatief de uitkomst van het model beïnvloeden. Bijvoorbeeld, als een model een onnauwkeurige voorspelling deed, geeft de component je de mogelijkheid om in te zoomen en te bepalen welke kenmerken of kenmerkwaarden de voorspelling hebben beïnvloed. Dit detailniveau helpt niet alleen bij het debuggen, maar biedt ook transparantie en verantwoording in auditsituaties. Ten slotte kan de component helpen om eerlijkheidsproblemen te identificeren. Bijvoorbeeld, als een gevoelig kenmerk zoals etniciteit of geslacht zeer invloedrijk is bij het sturen van de voorspelling van een model, kan dit een teken zijn van ras- of genderbias in het model.

![Feature-importance](../../../../9-Real-World/2-Debugging-ML-Models/images/9-features-influence.png)

Gebruik interpretatie wanneer je:

* Wilt bepalen hoe betrouwbaar de voorspellingen van je AI-systeem zijn door te begrijpen welke kenmerken het belangrijkst zijn voor de voorspellingen.
* Het debuggen van je model wilt benaderen door het eerst te begrijpen en te identificeren of het model gezonde kenmerken gebruikt of slechts valse correlaties.
* Potentiële bronnen van oneerlijkheid wilt onthullen door te begrijpen of het model voorspellingen baseert op gevoelige kenmerken of op kenmerken die sterk gecorreleerd zijn met hen.
* Gebruikersvertrouwen wilt opbouwen in de beslissingen van je model door lokale verklaringen te genereren om hun uitkomsten te illustreren.
* Een regelgevingsaudit van een AI-systeem wilt voltooien om modellen te valideren en de impact van modelbeslissingen op mensen te monitoren.

## Conclusie

Alle componenten van het RAI-dashboard zijn praktische tools om machine learning-modellen te bouwen die minder schadelijk en meer betrouwbaar zijn voor de samenleving. Ze verbeteren de preventie van bedreigingen voor mensenrechten; het discrimineren of uitsluiten van bepaalde groepen van levensmogelijkheden; en het risico op fysieke of psychologische schade. Ze helpen ook om vertrouwen op te bouwen in de beslissingen van je model door lokale verklaringen te genereren om hun uitkomsten te illustreren. Sommige van de potentiële schade kan worden geclassificeerd als:

- **Toewijzing**, als bijvoorbeeld een geslacht of etniciteit wordt bevoordeeld boven een ander.
- **Kwaliteit van dienstverlening**. Als je de gegevens traint voor één specifiek scenario, maar de realiteit veel complexer is, leidt dit tot een slecht presterende dienst.
- **Stereotypering**. Het associëren van een bepaalde groep met vooraf toegewezen attributen.
- **Denigratie**. Het oneerlijk bekritiseren en labelen van iets of iemand.
- **Over- of ondervertegenwoordiging**. Het idee is dat een bepaalde groep niet zichtbaar is in een bepaald beroep, en elke dienst of functie die dat blijft promoten, draagt bij aan schade.

### Azure RAI-dashboard

[Azure RAI-dashboard](https://learn.microsoft.com/en-us/azure/machine-learning/concept-responsible-ai-dashboard?WT.mc_id=aiml-90525-ruyakubu) is gebaseerd op open-source tools ontwikkeld door toonaangevende academische instellingen en organisaties, waaronder Microsoft. Deze tools zijn essentieel voor datawetenschappers en AI-ontwikkelaars om het gedrag van modellen beter te begrijpen, ongewenste problemen in AI-modellen te ontdekken en te mitigeren.

- Leer hoe je de verschillende componenten kunt gebruiken door de RAI-dashboard [documentatie](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-responsible-ai-dashboard?WT.mc_id=aiml-90525-ruyakubu) te bekijken.

- Bekijk enkele RAI-dashboard [voorbeeldnotebooks](https://github.com/Azure/RAI-vNext-Preview/tree/main/examples/notebooks) om meer verantwoorde AI-scenario's in Azure Machine Learning te debuggen.

---
## 🚀 Uitdaging

Om te voorkomen dat statistische of databiases überhaupt worden geïntroduceerd, zouden we:

- een diversiteit aan achtergronden en perspectieven moeten hebben onder de mensen die aan systemen werken
- investeren in datasets die de diversiteit van onze samenleving weerspiegelen
- betere methoden ontwikkelen om bias te detecteren en te corrigeren wanneer het optreedt

Denk na over real-life scenario's waarin oneerlijkheid duidelijk is bij het bouwen en gebruiken van modellen. Wat moeten we nog meer overwegen?

## [Quiz na de les](https://ff-quizzes.netlify.app/en/ml/)
## Review & Zelfstudie

In deze les heb je enkele praktische tools geleerd om verantwoorde AI te integreren in machine learning.

Bekijk deze workshop om dieper in te gaan op de onderwerpen:

- Responsible AI Dashboard: One-stop shop voor het operationaliseren van RAI in de praktijk door Besmira Nushi en Mehrnoosh Sameki

[![Responsible AI Dashboard: One-stop shop voor het operationaliseren van RAI in de praktijk](https://img.youtube.com/vi/f1oaDNl3djg/0.jpg)](https://www.youtube.com/watch?v=f1oaDNl3djg "Responsible AI Dashboard: One-stop shop voor het operationaliseren van RAI in de praktijk")


> 🎥 Klik op de afbeelding hierboven voor een video: Responsible AI Dashboard: One-stop shop voor het operationaliseren van RAI in de praktijk door Besmira Nushi en Mehrnoosh Sameki

Raadpleeg de volgende materialen om meer te leren over verantwoorde AI en hoe je meer betrouwbare modellen kunt bouwen:

- Microsoft’s RAI-dashboardtools voor het debuggen van ML-modellen: [Responsible AI tools resources](https://aka.ms/rai-dashboard)

- Verken de Responsible AI-toolkit: [Github](https://github.com/microsoft/responsible-ai-toolbox)

- Microsoft’s RAI-resourcecentrum: [Responsible AI Resources – Microsoft AI](https://www.microsoft.com/ai/responsible-ai-resources?activetab=pivot1%3aprimaryr4)

- Microsoft’s FATE-onderzoeksgroep: [FATE: Fairness, Accountability, Transparency, and Ethics in AI - Microsoft Research](https://www.microsoft.com/research/theme/fate/)

## Opdracht

[Verken RAI-dashboard](assignment.md)

---

**Disclaimer**:  
Dit document is vertaald met behulp van de AI-vertalingsservice [Co-op Translator](https://github.com/Azure/co-op-translator). Hoewel we streven naar nauwkeurigheid, dient u zich ervan bewust te zijn dat geautomatiseerde vertalingen fouten of onnauwkeurigheden kunnen bevatten. Het originele document in zijn oorspronkelijke taal moet worden beschouwd als de gezaghebbende bron. Voor cruciale informatie wordt professionele menselijke vertaling aanbevolen. Wij zijn niet aansprakelijk voor eventuele misverstanden of verkeerde interpretaties die voortvloeien uit het gebruik van deze vertaling.