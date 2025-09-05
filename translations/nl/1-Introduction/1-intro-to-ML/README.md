<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "69389392fa6346e0dfa30f664b7b6fec",
  "translation_date": "2025-09-05T19:37:48+00:00",
  "source_file": "1-Introduction/1-intro-to-ML/README.md",
  "language_code": "nl"
}
-->
# Introductie tot machine learning

## [Quiz voorafgaand aan de les](https://ff-quizzes.netlify.app/en/ml/)

---

[![ML voor beginners - Introductie tot Machine Learning voor Beginners](https://img.youtube.com/vi/6mSx_KJxcHI/0.jpg)](https://youtu.be/6mSx_KJxcHI "ML voor beginners - Introductie tot Machine Learning voor Beginners")

> 🎥 Klik op de afbeelding hierboven voor een korte video over deze les.

Welkom bij deze cursus over klassieke machine learning voor beginners! Of je nu helemaal nieuw bent op dit gebied, of een ervaren ML-practitioner die zijn kennis wil opfrissen, we zijn blij dat je meedoet! We willen een vriendelijke startplek creëren voor je ML-studie en staan open voor jouw [feedback](https://github.com/microsoft/ML-For-Beginners/discussions).

[![Introductie tot ML](https://img.youtube.com/vi/h0e2HAPTGF4/0.jpg)](https://youtu.be/h0e2HAPTGF4 "Introductie tot ML")

> 🎥 Klik op de afbeelding hierboven voor een video: MIT's John Guttag introduceert machine learning

---
## Aan de slag met machine learning

Voordat je begint met deze cursus, moet je je computer instellen om lokaal notebooks te kunnen draaien.

- **Configureer je computer met deze video's**. Gebruik de volgende links om te leren [hoe je Python installeert](https://youtu.be/CXZYvNRIAKM) op je systeem en [hoe je een teksteditor instelt](https://youtu.be/EU8eayHWoZg) voor ontwikkeling.
- **Leer Python**. Het wordt ook aanbevolen om een basisbegrip te hebben van [Python](https://docs.microsoft.com/learn/paths/python-language/?WT.mc_id=academic-77952-leestott), een programmeertaal die nuttig is voor datawetenschappers en die we in deze cursus gebruiken.
- **Leer Node.js en JavaScript**. We gebruiken ook een paar keer JavaScript in deze cursus bij het bouwen van webapps, dus je moet [node](https://nodejs.org) en [npm](https://www.npmjs.com/) installeren, evenals [Visual Studio Code](https://code.visualstudio.com/) beschikbaar hebben voor zowel Python- als JavaScript-ontwikkeling.
- **Maak een GitHub-account aan**. Aangezien je ons hier op [GitHub](https://github.com) hebt gevonden, heb je misschien al een account, maar zo niet, maak er dan een aan en fork deze cursus om zelf te gebruiken. (Geef ons gerust een ster, ook 😊)
- **Verken Scikit-learn**. Maak jezelf vertrouwd met [Scikit-learn](https://scikit-learn.org/stable/user_guide.html), een set ML-bibliotheken die we in deze lessen gebruiken.

---
## Wat is machine learning?

De term 'machine learning' is een van de meest populaire en vaak gebruikte termen van vandaag. Er is een aanzienlijke kans dat je deze term minstens één keer hebt gehoord als je enige bekendheid hebt met technologie, ongeacht in welk domein je werkt. De werking van machine learning is echter voor de meeste mensen een mysterie. Voor een beginner in machine learning kan het onderwerp soms overweldigend aanvoelen. Daarom is het belangrijk om te begrijpen wat machine learning eigenlijk is en er stap voor stap over te leren, aan de hand van praktische voorbeelden.

---
## De hypecurve

![ml hype curve](../../../../1-Introduction/1-intro-to-ML/images/hype.png)

> Google Trends toont de recente 'hypecurve' van de term 'machine learning'

---
## Een mysterieus universum

We leven in een universum vol fascinerende mysteries. Grote wetenschappers zoals Stephen Hawking, Albert Einstein en vele anderen hebben hun leven gewijd aan het zoeken naar betekenisvolle informatie die de mysteries van de wereld om ons heen onthult. Dit is de menselijke conditie van leren: een menselijk kind leert nieuwe dingen en ontdekt de structuur van zijn wereld jaar na jaar terwijl het opgroeit tot volwassenheid.

---
## Het brein van een kind

Het brein en de zintuigen van een kind nemen de feiten van hun omgeving waar en leren geleidelijk de verborgen patronen van het leven, die het kind helpen logische regels te ontwikkelen om de geleerde patronen te identificeren. Het leerproces van het menselijk brein maakt mensen tot de meest geavanceerde levende wezens van deze wereld. Continu leren door verborgen patronen te ontdekken en vervolgens innoveren op die patronen stelt ons in staat om onszelf steeds beter te maken gedurende ons hele leven. Dit leervermogen en evolutiecapaciteit is gerelateerd aan een concept genaamd [hersenenplasticiteit](https://www.simplypsychology.org/brain-plasticity.html). Op een oppervlakkige manier kunnen we enkele motiverende overeenkomsten trekken tussen het leerproces van het menselijk brein en de concepten van machine learning.

---
## Het menselijk brein

Het [menselijk brein](https://www.livescience.com/29365-human-brain.html) neemt dingen waar uit de echte wereld, verwerkt de waargenomen informatie, neemt rationele beslissingen en voert bepaalde acties uit op basis van de omstandigheden. Dit noemen we intelligent gedrag. Wanneer we een facsimile van het intelligente gedragsproces programmeren in een machine, noemen we dat kunstmatige intelligentie (AI).

---
## Enkele terminologie

Hoewel de termen soms door elkaar worden gehaald, is machine learning (ML) een belangrijk onderdeel van kunstmatige intelligentie. **ML houdt zich bezig met het gebruik van gespecialiseerde algoritmen om betekenisvolle informatie te ontdekken en verborgen patronen te vinden in waargenomen data om het rationele besluitvormingsproces te ondersteunen**.

---
## AI, ML, Deep Learning

![AI, ML, deep learning, data science](../../../../1-Introduction/1-intro-to-ML/images/ai-ml-ds.png)

> Een diagram dat de relaties toont tussen AI, ML, deep learning en data science. Infographic door [Jen Looper](https://twitter.com/jenlooper) geïnspireerd door [deze grafiek](https://softwareengineering.stackexchange.com/questions/366996/distinction-between-ai-ml-neural-networks-deep-learning-and-data-mining)

---
## Concepten die we behandelen

In deze cursus behandelen we alleen de kernconcepten van machine learning die een beginner moet kennen. We behandelen wat we 'klassieke machine learning' noemen, voornamelijk met behulp van Scikit-learn, een uitstekende bibliotheek die veel studenten gebruiken om de basis te leren. Om bredere concepten van kunstmatige intelligentie of deep learning te begrijpen, is een sterke fundamentele kennis van machine learning onmisbaar, en die willen we hier aanbieden.

---
## In deze cursus leer je:

- kernconcepten van machine learning
- de geschiedenis van ML
- ML en eerlijkheid
- regressie ML-technieken
- classificatie ML-technieken
- clustering ML-technieken
- natuurlijke taalverwerking ML-technieken
- tijdreeksvoorspelling ML-technieken
- reinforcement learning
- toepassingen van ML in de echte wereld

---
## Wat we niet behandelen

- deep learning
- neurale netwerken
- AI

Om de leerervaring te verbeteren, vermijden we de complexiteit van neurale netwerken, 'deep learning' - het bouwen van modellen met meerdere lagen met behulp van neurale netwerken - en AI, die we in een andere cursus zullen bespreken. We bieden ook een toekomstige data science-cursus aan om ons te richten op dat aspect van dit grotere veld.

---
## Waarom machine learning studeren?

Machine learning wordt vanuit een systeemperspectief gedefinieerd als het creëren van geautomatiseerde systemen die verborgen patronen uit data kunnen leren om te helpen bij het nemen van intelligente beslissingen.

Deze motivatie is losjes geïnspireerd door hoe het menselijk brein bepaalde dingen leert op basis van de data die het waarneemt uit de buitenwereld.

✅ Denk eens na waarom een bedrijf zou willen proberen machine learning-strategieën te gebruiken in plaats van een hard-coded regelsysteem te maken.

---
## Toepassingen van machine learning

Toepassingen van machine learning zijn tegenwoordig bijna overal en zijn net zo alomtegenwoordig als de data die door onze samenlevingen stroomt, gegenereerd door onze smartphones, verbonden apparaten en andere systemen. Gezien het immense potentieel van geavanceerde machine learning-algoritmen, hebben onderzoekers hun mogelijkheden verkend om multidimensionale en multidisciplinaire problemen uit het echte leven op te lossen met geweldige positieve resultaten.

---
## Voorbeelden van toegepaste ML

**Je kunt machine learning op veel manieren gebruiken**:

- Om de kans op ziekte te voorspellen op basis van de medische geschiedenis of rapporten van een patiënt.
- Om weersgegevens te gebruiken om weersomstandigheden te voorspellen.
- Om de sentimenten van een tekst te begrijpen.
- Om nepnieuws te detecteren en de verspreiding van propaganda te stoppen.

Financiën, economie, aardwetenschappen, ruimteonderzoek, biomedische techniek, cognitieve wetenschap en zelfs gebieden binnen de geesteswetenschappen hebben machine learning aangepast om de zware, data-intensieve problemen van hun domein op te lossen.

---
## Conclusie

Machine learning automatiseert het proces van patroonontdekking door betekenisvolle inzichten te vinden uit echte of gegenereerde data. Het heeft zichzelf bewezen als zeer waardevol in zakelijke, gezondheids- en financiële toepassingen, onder andere.

In de nabije toekomst zal het begrijpen van de basisprincipes van machine learning een must worden voor mensen uit elk domein vanwege de brede adoptie ervan.

---
# 🚀 Uitdaging

Schets, op papier of met een online app zoals [Excalidraw](https://excalidraw.com/), jouw begrip van de verschillen tussen AI, ML, deep learning en data science. Voeg enkele ideeën toe over problemen die elk van deze technieken goed kunnen oplossen.

# [Quiz na de les](https://ff-quizzes.netlify.app/en/ml/)

---
# Review & Zelfstudie

Om meer te leren over hoe je met ML-algoritmen in de cloud kunt werken, volg dit [Leerpad](https://docs.microsoft.com/learn/paths/create-no-code-predictive-models-azure-machine-learning/?WT.mc_id=academic-77952-leestott).

Volg een [Leerpad](https://docs.microsoft.com/learn/modules/introduction-to-machine-learning/?WT.mc_id=academic-77952-leestott) over de basisprincipes van ML.

---
# Opdracht

[Begin met werken](assignment.md)

---

**Disclaimer**:  
Dit document is vertaald met behulp van de AI-vertalingsservice [Co-op Translator](https://github.com/Azure/co-op-translator). Hoewel we streven naar nauwkeurigheid, dient u zich ervan bewust te zijn dat geautomatiseerde vertalingen fouten of onnauwkeurigheden kunnen bevatten. Het originele document in zijn oorspronkelijke taal moet worden beschouwd als de gezaghebbende bron. Voor kritieke informatie wordt professionele menselijke vertaling aanbevolen. Wij zijn niet aansprakelijk voor eventuele misverstanden of verkeerde interpretaties die voortvloeien uit het gebruik van deze vertaling.