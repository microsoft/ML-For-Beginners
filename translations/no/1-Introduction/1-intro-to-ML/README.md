<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "69389392fa6346e0dfa30f664b7b6fec",
  "translation_date": "2025-09-05T21:43:11+00:00",
  "source_file": "1-Introduction/1-intro-to-ML/README.md",
  "language_code": "no"
}
-->
# Introduksjon til maskinlæring

## [Quiz før forelesning](https://ff-quizzes.netlify.app/en/ml/)

---

[![ML for nybegynnere - Introduksjon til maskinlæring for nybegynnere](https://img.youtube.com/vi/6mSx_KJxcHI/0.jpg)](https://youtu.be/6mSx_KJxcHI "ML for nybegynnere - Introduksjon til maskinlæring for nybegynnere")

> 🎥 Klikk på bildet over for en kort video som går gjennom denne leksjonen.

Velkommen til dette kurset om klassisk maskinlæring for nybegynnere! Enten du er helt ny på dette temaet, eller en erfaren ML-praktiker som ønsker å friske opp kunnskapen, er vi glade for å ha deg med! Vi ønsker å skape et vennlig startpunkt for din ML-studie og vil gjerne evaluere, svare på og inkludere din [tilbakemelding](https://github.com/microsoft/ML-For-Beginners/discussions).

[![Introduksjon til ML](https://img.youtube.com/vi/h0e2HAPTGF4/0.jpg)](https://youtu.be/h0e2HAPTGF4 "Introduksjon til ML")

> 🎥 Klikk på bildet over for en video: MITs John Guttag introduserer maskinlæring

---
## Komme i gang med maskinlæring

Før du starter med dette pensumet, må du ha datamaskinen din satt opp og klar til å kjøre notatbøker lokalt.

- **Konfigurer maskinen din med disse videoene**. Bruk følgende lenker for å lære [hvordan du installerer Python](https://youtu.be/CXZYvNRIAKM) på systemet ditt og [setter opp en teksteditor](https://youtu.be/EU8eayHWoZg) for utvikling.
- **Lær Python**. Det anbefales også å ha en grunnleggende forståelse av [Python](https://docs.microsoft.com/learn/paths/python-language/?WT.mc_id=academic-77952-leestott), et programmeringsspråk som er nyttig for dataforskere og som vi bruker i dette kurset.
- **Lær Node.js og JavaScript**. Vi bruker også JavaScript noen ganger i dette kurset når vi bygger webapplikasjoner, så du må ha [node](https://nodejs.org) og [npm](https://www.npmjs.com/) installert, samt [Visual Studio Code](https://code.visualstudio.com/) tilgjengelig for både Python- og JavaScript-utvikling.
- **Opprett en GitHub-konto**. Siden du fant oss her på [GitHub](https://github.com), har du kanskje allerede en konto, men hvis ikke, opprett en og deretter fork dette pensumet for å bruke det selv. (Gi oss gjerne en stjerne også 😊)
- **Utforsk Scikit-learn**. Gjør deg kjent med [Scikit-learn](https://scikit-learn.org/stable/user_guide.html), et sett med ML-biblioteker som vi refererer til i disse leksjonene.

---
## Hva er maskinlæring?

Begrepet 'maskinlæring' er et av de mest populære og ofte brukte begrepene i dag. Det er en betydelig sannsynlighet for at du har hørt dette begrepet minst én gang hvis du har en viss kjennskap til teknologi, uansett hvilket felt du jobber i. Mekanismene bak maskinlæring er imidlertid en gåte for de fleste. For en nybegynner innen maskinlæring kan emnet noen ganger føles overveldende. Derfor er det viktig å forstå hva maskinlæring faktisk er, og lære om det steg for steg, gjennom praktiske eksempler.

---
## Hypekurven

![ml hype curve](../../../../1-Introduction/1-intro-to-ML/images/hype.png)

> Google Trends viser den nylige 'hypekurven' for begrepet 'maskinlæring'

---
## Et mystisk univers

Vi lever i et univers fullt av fascinerende mysterier. Store vitenskapsmenn som Stephen Hawking, Albert Einstein og mange flere har viet sine liv til å søke etter meningsfull informasjon som avdekker mysteriene i verden rundt oss. Dette er den menneskelige tilstanden for læring: et menneskebarn lærer nye ting og avdekker strukturen i sin verden år for år mens de vokser opp.

---
## Barnets hjerne

Et barns hjerne og sanser oppfatter fakta fra omgivelsene og lærer gradvis de skjulte mønstrene i livet som hjelper barnet med å lage logiske regler for å identifisere lærte mønstre. Læringsprosessen til den menneskelige hjernen gjør mennesker til de mest sofistikerte levende skapningene i denne verden. Å lære kontinuerlig ved å oppdage skjulte mønstre og deretter innovere på disse mønstrene gjør oss i stand til å forbedre oss selv gjennom hele livet. Denne læringsevnen og utviklingskapasiteten er knyttet til et konsept kalt [hjernens plastisitet](https://www.simplypsychology.org/brain-plasticity.html). Overfladisk kan vi trekke noen motiverende likheter mellom læringsprosessen til den menneskelige hjernen og konseptene for maskinlæring.

---
## Den menneskelige hjernen

Den [menneskelige hjernen](https://www.livescience.com/29365-human-brain.html) oppfatter ting fra den virkelige verden, behandler den oppfattede informasjonen, tar rasjonelle beslutninger og utfører visse handlinger basert på omstendigheter. Dette kaller vi å oppføre seg intelligent. Når vi programmerer en etterligning av den intelligente atferdsprosessen til en maskin, kalles det kunstig intelligens (AI).

---
## Noen begreper

Selv om begrepene kan forveksles, er maskinlæring (ML) en viktig underkategori av kunstig intelligens. **ML handler om å bruke spesialiserte algoritmer for å avdekke meningsfull informasjon og finne skjulte mønstre fra oppfattet data for å støtte den rasjonelle beslutningsprosessen**.

---
## AI, ML, Deep Learning

![AI, ML, deep learning, data science](../../../../1-Introduction/1-intro-to-ML/images/ai-ml-ds.png)

> Et diagram som viser forholdet mellom AI, ML, deep learning og data science. Infografikk av [Jen Looper](https://twitter.com/jenlooper) inspirert av [denne grafikken](https://softwareengineering.stackexchange.com/questions/366996/distinction-between-ai-ml-neural-networks-deep-learning-and-data-mining)

---
## Konsepter vi skal dekke

I dette pensumet skal vi dekke kun kjernebegrepene innen maskinlæring som en nybegynner må kjenne til. Vi dekker det vi kaller 'klassisk maskinlæring' hovedsakelig ved bruk av Scikit-learn, et utmerket bibliotek mange studenter bruker for å lære det grunnleggende. For å forstå bredere konsepter innen kunstig intelligens eller deep learning, er en sterk grunnleggende kunnskap om maskinlæring uunnværlig, og det ønsker vi å tilby her.

---
## I dette kurset vil du lære:

- kjernebegreper innen maskinlæring
- historien til ML
- ML og rettferdighet
- regresjonsteknikker innen ML
- klassifiseringsteknikker innen ML
- klyngingsteknikker innen ML
- naturlig språkbehandlingsteknikker innen ML
- tidsserieprognoseteknikker innen ML
- forsterkende læring
- virkelige applikasjoner for ML

---
## Hva vi ikke vil dekke

- deep learning
- nevrale nettverk
- AI

For å gi en bedre læringsopplevelse, vil vi unngå kompleksiteten til nevrale nettverk, 'deep learning' - modellbygging med mange lag ved bruk av nevrale nettverk - og AI, som vi vil diskutere i et annet pensum. Vi vil også tilby et kommende pensum om data science for å fokusere på den delen av dette større feltet.

---
## Hvorfor studere maskinlæring?

Maskinlæring, fra et systemperspektiv, defineres som opprettelsen av automatiserte systemer som kan lære skjulte mønstre fra data for å hjelpe til med å ta intelligente beslutninger.

Denne motivasjonen er løst inspirert av hvordan den menneskelige hjernen lærer visse ting basert på data den oppfatter fra omverdenen.

✅ Tenk et øyeblikk på hvorfor en bedrift ville ønske å bruke maskinlæringsstrategier i stedet for å lage en hardkodet regelbasert motor.

---
## Applikasjoner av maskinlæring

Applikasjoner av maskinlæring er nå nesten overalt, og er like utbredt som dataene som flyter rundt i våre samfunn, generert av våre smarttelefoner, tilkoblede enheter og andre systemer. Med tanke på det enorme potensialet til moderne maskinlæringsalgoritmer, har forskere utforsket deres evne til å løse multidimensjonale og tverrfaglige virkelige problemer med svært positive resultater.

---
## Eksempler på anvendt ML

**Du kan bruke maskinlæring på mange måter**:

- For å forutsi sannsynligheten for sykdom basert på en pasients medisinske historie eller rapporter.
- For å utnytte værdata til å forutsi værhendelser.
- For å forstå sentimentet i en tekst.
- For å oppdage falske nyheter for å stoppe spredningen av propaganda.

Finans, økonomi, geovitenskap, romutforskning, biomedisinsk ingeniørkunst, kognitiv vitenskap og til og med humanistiske fag har tilpasset maskinlæring for å løse de krevende, databehandlingsintensive problemene i deres felt.

---
## Konklusjon

Maskinlæring automatiserer prosessen med mønsteroppdagelse ved å finne meningsfulle innsikter fra virkelige eller genererte data. Det har vist seg å være svært verdifullt i forretnings-, helse- og finansapplikasjoner, blant andre.

I nær fremtid vil det å forstå det grunnleggende om maskinlæring bli et must for folk fra alle felt på grunn av dens utbredte adopsjon.

---
# 🚀 Utfordring

Skisser, på papir eller ved bruk av en online app som [Excalidraw](https://excalidraw.com/), din forståelse av forskjellene mellom AI, ML, deep learning og data science. Legg til noen ideer om problemer som hver av disse teknikkene er gode til å løse.

# [Quiz etter forelesning](https://ff-quizzes.netlify.app/en/ml/)

---
# Gjennomgang & Selvstudie

For å lære mer om hvordan du kan jobbe med ML-algoritmer i skyen, følg denne [Læringsstien](https://docs.microsoft.com/learn/paths/create-no-code-predictive-models-azure-machine-learning/?WT.mc_id=academic-77952-leestott).

Ta en [Læringssti](https://docs.microsoft.com/learn/modules/introduction-to-machine-learning/?WT.mc_id=academic-77952-leestott) om det grunnleggende innen ML.

---
# Oppgave

[Kom i gang](assignment.md)

---

**Ansvarsfraskrivelse**:  
Dette dokumentet er oversatt ved hjelp av AI-oversettelsestjenesten [Co-op Translator](https://github.com/Azure/co-op-translator). Selv om vi tilstreber nøyaktighet, vennligst vær oppmerksom på at automatiske oversettelser kan inneholde feil eller unøyaktigheter. Det originale dokumentet på sitt opprinnelige språk bør anses som den autoritative kilden. For kritisk informasjon anbefales profesjonell menneskelig oversettelse. Vi er ikke ansvarlige for eventuelle misforståelser eller feiltolkninger som oppstår ved bruk av denne oversettelsen.