<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "69389392fa6346e0dfa30f664b7b6fec",
  "translation_date": "2025-09-05T00:30:43+00:00",
  "source_file": "1-Introduction/1-intro-to-ML/README.md",
  "language_code": "da"
}
-->
# Introduktion til maskinlæring

## [Quiz før lektionen](https://ff-quizzes.netlify.app/en/ml/)

---

[![ML for begyndere - Introduktion til maskinlæring for begyndere](https://img.youtube.com/vi/6mSx_KJxcHI/0.jpg)](https://youtu.be/6mSx_KJxcHI "ML for begyndere - Introduktion til maskinlæring for begyndere")

> 🎥 Klik på billedet ovenfor for en kort video, der gennemgår denne lektion.

Velkommen til dette kursus om klassisk maskinlæring for begyndere! Uanset om du er helt ny inden for emnet eller en erfaren ML-praktiker, der ønsker at genopfriske et område, er vi glade for at have dig med! Vi ønsker at skabe et venligt startpunkt for din ML-studie og vil gerne evaluere, reagere på og integrere din [feedback](https://github.com/microsoft/ML-For-Beginners/discussions).

[![Introduktion til ML](https://img.youtube.com/vi/h0e2HAPTGF4/0.jpg)](https://youtu.be/h0e2HAPTGF4 "Introduktion til ML")

> 🎥 Klik på billedet ovenfor for en video: MIT's John Guttag introducerer maskinlæring

---
## Kom godt i gang med maskinlæring

Før du starter med dette pensum, skal du have din computer klar til at køre notebooks lokalt.

- **Konfigurer din computer med disse videoer**. Brug følgende links til at lære [hvordan man installerer Python](https://youtu.be/CXZYvNRIAKM) på dit system og [opsætter en teksteditor](https://youtu.be/EU8eayHWoZg) til udvikling.
- **Lær Python**. Det anbefales også at have en grundlæggende forståelse af [Python](https://docs.microsoft.com/learn/paths/python-language/?WT.mc_id=academic-77952-leestott), et programmeringssprog, der er nyttigt for dataforskere, og som vi bruger i dette kursus.
- **Lær Node.js og JavaScript**. Vi bruger også JavaScript et par gange i dette kursus, når vi bygger webapplikationer, så du skal have [node](https://nodejs.org) og [npm](https://www.npmjs.com/) installeret samt [Visual Studio Code](https://code.visualstudio.com/) til både Python- og JavaScript-udvikling.
- **Opret en GitHub-konto**. Da du fandt os her på [GitHub](https://github.com), har du måske allerede en konto, men hvis ikke, så opret en og fork derefter dette pensum til brug for dig selv. (Du er også velkommen til at give os en stjerne 😊)
- **Udforsk Scikit-learn**. Bliv bekendt med [Scikit-learn](https://scikit-learn.org/stable/user_guide.html), et sæt ML-biblioteker, som vi refererer til i disse lektioner.

---
## Hvad er maskinlæring?

Begrebet 'maskinlæring' er et af de mest populære og ofte anvendte begreber i dag. Der er en ikke ubetydelig sandsynlighed for, at du har hørt dette begreb mindst én gang, hvis du har en vis form for kendskab til teknologi, uanset hvilket område du arbejder inden for. Mekanikken bag maskinlæring er dog en gåde for de fleste mennesker. For en nybegynder inden for maskinlæring kan emnet nogle gange føles overvældende. Derfor er det vigtigt at forstå, hvad maskinlæring egentlig er, og at lære om det trin for trin gennem praktiske eksempler.

---
## Hypekurven

![ml hype curve](../../../../1-Introduction/1-intro-to-ML/images/hype.png)

> Google Trends viser den seneste 'hypekurve' for begrebet 'maskinlæring'

---
## Et mystisk univers

Vi lever i et univers fyldt med fascinerende mysterier. Store videnskabsfolk som Stephen Hawking, Albert Einstein og mange flere har viet deres liv til at søge meningsfuld information, der afslører mysterierne i verden omkring os. Dette er den menneskelige tilstand af læring: Et menneskebarn lærer nye ting og opdager strukturen i deres verden år for år, mens de vokser op.

---
## Barnets hjerne

Et barns hjerne og sanser opfatter fakta om deres omgivelser og lærer gradvist de skjulte mønstre i livet, som hjælper barnet med at skabe logiske regler for at identificere lærte mønstre. Den menneskelige hjernes læringsproces gør mennesker til de mest sofistikerede levende væsener i denne verden. At lære kontinuerligt ved at opdage skjulte mønstre og derefter innovere på disse mønstre gør os i stand til at forbedre os selv gennem hele livet. Denne læringskapacitet og evne til at udvikle sig er relateret til et koncept kaldet [hjernens plasticitet](https://www.simplypsychology.org/brain-plasticity.html). Overfladisk kan vi drage nogle motiverende ligheder mellem den menneskelige hjernes læringsproces og begreberne inden for maskinlæring.

---
## Den menneskelige hjerne

Den [menneskelige hjerne](https://www.livescience.com/29365-human-brain.html) opfatter ting fra den virkelige verden, behandler den opfattede information, træffer rationelle beslutninger og udfører visse handlinger baseret på omstændighederne. Dette kalder vi at opføre sig intelligent. Når vi programmerer en efterligning af den intelligente adfærdsproces til en maskine, kaldes det kunstig intelligens (AI).

---
## Nogle terminologier

Selvom begreberne kan forveksles, er maskinlæring (ML) en vigtig underkategori af kunstig intelligens. **ML handler om at bruge specialiserede algoritmer til at finde meningsfuld information og opdage skjulte mønstre fra opfattet data for at understøtte den rationelle beslutningsproces**.

---
## AI, ML, Deep Learning

![AI, ML, deep learning, data science](../../../../1-Introduction/1-intro-to-ML/images/ai-ml-ds.png)

> Et diagram, der viser forholdet mellem AI, ML, deep learning og data science. Infografik af [Jen Looper](https://twitter.com/jenlooper) inspireret af [denne grafik](https://softwareengineering.stackexchange.com/questions/366996/distinction-between-ai-ml-neural-networks-deep-learning-and-data-mining)

---
## Begreber, vi vil dække

I dette pensum vil vi kun dække de grundlæggende begreber inden for maskinlæring, som en nybegynder skal kende. Vi dækker det, vi kalder 'klassisk maskinlæring', primært ved hjælp af Scikit-learn, et fremragende bibliotek, som mange studerende bruger til at lære det grundlæggende. For at forstå bredere begreber inden for kunstig intelligens eller deep learning er en stærk grundlæggende viden om maskinlæring uundværlig, og det vil vi gerne tilbyde her.

---
## I dette kursus vil du lære:

- grundlæggende begreber inden for maskinlæring
- historien om ML
- ML og retfærdighed
- regressionsteknikker inden for ML
- klassifikationsteknikker inden for ML
- clusteringteknikker inden for ML
- naturlig sprogbehandlingsteknikker inden for ML
- tidsserieprognoseteknikker inden for ML
- forstærkningslæring
- virkelige anvendelser af ML

---
## Hvad vi ikke vil dække

- deep learning
- neurale netværk
- AI

For at skabe en bedre læringsoplevelse vil vi undgå kompleksiteten af neurale netværk, 'deep learning' - opbygning af modeller med mange lag ved hjælp af neurale netværk - og AI, som vi vil diskutere i et andet pensum. Vi vil også tilbyde et kommende pensum om data science for at fokusere på den del af dette større felt.

---
## Hvorfor studere maskinlæring?

Maskinlæring, fra et systemperspektiv, defineres som skabelsen af automatiserede systemer, der kan lære skjulte mønstre fra data for at hjælpe med at træffe intelligente beslutninger.

Denne motivation er løst inspireret af, hvordan den menneskelige hjerne lærer visse ting baseret på de data, den opfatter fra omverdenen.

✅ Tænk et øjeblik over, hvorfor en virksomhed ville ønske at bruge maskinlæringsstrategier i stedet for at skabe en hardkodet regelbaseret motor.

---
## Anvendelser af maskinlæring

Anvendelser af maskinlæring er nu næsten overalt og er lige så udbredte som de data, der flyder rundt i vores samfund, genereret af vores smartphones, tilsluttede enheder og andre systemer. I betragtning af det enorme potentiale i avancerede maskinlæringsalgoritmer har forskere udforsket deres evne til at løse multidimensionale og tværfaglige virkelige problemer med store positive resultater.

---
## Eksempler på anvendt ML

**Du kan bruge maskinlæring på mange måder**:

- Til at forudsige sandsynligheden for sygdom ud fra en patients medicinske historie eller rapporter.
- Til at udnytte vejrdata til at forudsige vejrbegivenheder.
- Til at forstå sentimentet i en tekst.
- Til at opdage falske nyheder for at stoppe spredningen af propaganda.

Finans, økonomi, jordvidenskab, rumforskning, biomedicinsk ingeniørkunst, kognitiv videnskab og endda områder inden for humaniora har tilpasset maskinlæring til at løse de tunge databehandlingsproblemer i deres domæne.

---
## Konklusion

Maskinlæring automatiserer processen med mønsteropdagelse ved at finde meningsfulde indsigter fra virkelige eller genererede data. Det har vist sig at være yderst værdifuldt inden for forretning, sundhed og finansielle anvendelser, blandt andre.

I den nærmeste fremtid vil det at forstå det grundlæggende i maskinlæring blive et must for folk fra alle områder på grund af dets udbredte anvendelse.

---
# 🚀 Udfordring

Lav en skitse, enten på papir eller ved hjælp af en online app som [Excalidraw](https://excalidraw.com/), af din forståelse af forskellene mellem AI, ML, deep learning og data science. Tilføj nogle idéer om problemer, som hver af disse teknikker er gode til at løse.

# [Quiz efter lektionen](https://ff-quizzes.netlify.app/en/ml/)

---
# Gennemgang & Selvstudie

For at lære mere om, hvordan du kan arbejde med ML-algoritmer i skyen, følg denne [Learning Path](https://docs.microsoft.com/learn/paths/create-no-code-predictive-models-azure-machine-learning/?WT.mc_id=academic-77952-leestott).

Tag en [Learning Path](https://docs.microsoft.com/learn/modules/introduction-to-machine-learning/?WT.mc_id=academic-77952-leestott) om det grundlæggende i ML.

---
# Opgave

[Kom i gang](assignment.md)

---

**Ansvarsfraskrivelse**:  
Dette dokument er blevet oversat ved hjælp af AI-oversættelsestjenesten [Co-op Translator](https://github.com/Azure/co-op-translator). Selvom vi bestræber os på at opnå nøjagtighed, skal det bemærkes, at automatiserede oversættelser kan indeholde fejl eller unøjagtigheder. Det originale dokument på dets oprindelige sprog bør betragtes som den autoritative kilde. For kritisk information anbefales professionel menneskelig oversættelse. Vi påtager os ikke ansvar for eventuelle misforståelser eller fejltolkninger, der måtte opstå som følge af brugen af denne oversættelse.