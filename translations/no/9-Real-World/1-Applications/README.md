<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "83320d6b6994909e35d830cebf214039",
  "translation_date": "2025-09-05T21:33:30+00:00",
  "source_file": "9-Real-World/1-Applications/README.md",
  "language_code": "no"
}
-->
# Postscript: Maskinlæring i den virkelige verden

![Oppsummering av maskinlæring i den virkelige verden i en sketchnote](../../../../sketchnotes/ml-realworld.png)
> Sketchnote av [Tomomi Imura](https://www.twitter.com/girlie_mac)

I dette kurset har du lært mange måter å forberede data for trening og lage maskinlæringsmodeller. Du har bygget en rekke klassiske regresjons-, klyngings-, klassifiserings-, naturlig språkbehandlings- og tidsseriemodeller. Gratulerer! Nå lurer du kanskje på hva alt dette skal brukes til... hva er de virkelige anvendelsene for disse modellene?

Selv om AI, som ofte bruker dyp læring, har fått mye oppmerksomhet i industrien, finnes det fortsatt verdifulle bruksområder for klassiske maskinlæringsmodeller. Du bruker kanskje noen av disse anvendelsene allerede i dag! I denne leksjonen skal du utforske hvordan åtte ulike industrier og fagområder bruker disse typene modeller for å gjøre sine applikasjoner mer effektive, pålitelige, intelligente og verdifulle for brukerne.

## [Quiz før leksjonen](https://ff-quizzes.netlify.app/en/ml/)

## 💰 Finans

Finanssektoren tilbyr mange muligheter for maskinlæring. Mange problemer i dette området egner seg godt til å bli modellert og løst ved hjelp av ML.

### Kredittkortsvindel

Vi lærte om [k-means klynging](../../5-Clustering/2-K-Means/README.md) tidligere i kurset, men hvordan kan det brukes til å løse problemer relatert til kredittkortsvindel?

K-means klynging er nyttig i en teknikk for kredittkortsvindel kalt **uteliggermåling**. Uteliggere, eller avvik i observasjoner om et datasett, kan fortelle oss om et kredittkort brukes på en normal måte eller om noe uvanlig skjer. Som vist i artikkelen nedenfor, kan du sortere kredittkortdata ved hjelp av en k-means klyngingsalgoritme og tilordne hver transaksjon til en klynge basert på hvor mye den skiller seg ut. Deretter kan du evaluere de mest risikable klyngene for å avgjøre om transaksjonene er svindel eller legitime.
[Referanse](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.680.1195&rep=rep1&type=pdf)

### Formuesforvaltning

I formuesforvaltning håndterer en person eller firma investeringer på vegne av sine klienter. Jobben deres er å opprettholde og øke formuen på lang sikt, så det er viktig å velge investeringer som gir god avkastning.

En måte å evaluere hvordan en investering presterer på er gjennom statistisk regresjon. [Lineær regresjon](../../2-Regression/1-Tools/README.md) er et verdifullt verktøy for å forstå hvordan et fond presterer i forhold til en referanseindeks. Vi kan også finne ut om resultatene av regresjonen er statistisk signifikante, eller hvor mye de vil påvirke en klients investeringer. Du kan til og med utvide analysen ved hjelp av multippel regresjon, hvor flere risikofaktorer tas med i betraktning. For et eksempel på hvordan dette fungerer for et spesifikt fond, se artikkelen nedenfor om evaluering av fondsprestasjoner ved hjelp av regresjon.
[Referanse](http://www.brightwoodventures.com/evaluating-fund-performance-using-regression/)

## 🎓 Utdanning

Utdanningssektoren er også et veldig interessant område hvor ML kan brukes. Det finnes spennende problemer å løse, som å oppdage juks på prøver eller essays, eller håndtere skjevheter, enten de er utilsiktede eller ikke, i vurderingsprosessen.

### Forutsi studentatferd

[Coursera](https://coursera.com), en leverandør av åpne nettkurs, har en flott teknologiblogg hvor de diskuterer mange ingeniørbeslutninger. I denne casestudien plottet de en regresjonslinje for å utforske en mulig korrelasjon mellom en lav NPS (Net Promoter Score)-vurdering og kursretensjon eller frafall.
[Referanse](https://medium.com/coursera-engineering/controlled-regression-quantifying-the-impact-of-course-quality-on-learner-retention-31f956bd592a)

### Redusere skjevheter

[Grammarly](https://grammarly.com), en skriveassistent som sjekker for stave- og grammatikkfeil, bruker sofistikerte [naturlig språkbehandlingssystemer](../../6-NLP/README.md) i sine produkter. De publiserte en interessant casestudie i sin teknologiblogg om hvordan de håndterte kjønnsbias i maskinlæring, som du lærte om i vår [introduksjonsleksjon om rettferdighet](../../1-Introduction/3-fairness/README.md).
[Referanse](https://www.grammarly.com/blog/engineering/mitigating-gender-bias-in-autocorrect/)

## 👜 Detaljhandel

Detaljhandelssektoren kan definitivt dra nytte av bruk av ML, med alt fra å skape en bedre kundereise til å optimalisere lagerbeholdning.

### Personalisere kundereisen

Hos Wayfair, et selskap som selger hjemmevarer som møbler, er det avgjørende å hjelpe kundene med å finne de riktige produktene for deres smak og behov. I denne artikkelen beskriver ingeniører fra selskapet hvordan de bruker ML og NLP for å "vise de riktige resultatene for kundene". Spesielt har deres Query Intent Engine blitt bygget for å bruke enhetsuttrekking, klassifiseringstrening, uttrekking av eiendeler og meninger, og sentimentmerking på kundeanmeldelser. Dette er et klassisk eksempel på hvordan NLP fungerer i netthandel.
[Referanse](https://www.aboutwayfair.com/tech-innovation/how-we-use-machine-learning-and-natural-language-processing-to-empower-search)

### Lagerstyring

Innovative, smidige selskaper som [StitchFix](https://stitchfix.com), en abonnementstjeneste som sender klær til forbrukere, er sterkt avhengige av ML for anbefalinger og lagerstyring. Deres stylingteam samarbeider med deres innkjøpsteam, faktisk: "en av våre dataforskere eksperimenterte med en genetisk algoritme og brukte den på klær for å forutsi hva som ville være et vellykket klesplagg som ikke eksisterer i dag. Vi presenterte dette for innkjøpsteamet, og nå kan de bruke det som et verktøy."
[Referanse](https://www.zdnet.com/article/how-stitch-fix-uses-machine-learning-to-master-the-science-of-styling/)

## 🏥 Helsevesen

Helsevesenet kan bruke ML til å optimalisere forskningsoppgaver og logistiske problemer som å forhindre gjeninnleggelser eller stoppe sykdomsspredning.

### Administrere kliniske studier

Toksisitet i kliniske studier er en stor bekymring for legemiddelprodusenter. Hvor mye toksisitet er akseptabelt? I denne studien førte analyser av ulike metoder for kliniske studier til utviklingen av en ny tilnærming for å forutsi sannsynligheten for utfall i kliniske studier. Spesielt var de i stand til å bruke random forest for å lage en [klassifiserer](../../4-Classification/README.md) som kan skille mellom grupper av legemidler.
[Referanse](https://www.sciencedirect.com/science/article/pii/S2451945616302914)

### Administrere sykehusgjeninnleggelser

Sykehusbehandling er kostbart, spesielt når pasienter må gjeninnlegges. Denne artikkelen diskuterer et selskap som bruker ML for å forutsi potensialet for gjeninnleggelser ved hjelp av [klynging](../../5-Clustering/README.md)-algoritmer. Disse klyngene hjelper analytikere med å "oppdage grupper av gjeninnleggelser som kan ha en felles årsak".
[Referanse](https://healthmanagement.org/c/healthmanagement/issuearticle/hospital-readmissions-and-machine-learning)

### Sykdomshåndtering

Den nylige pandemien har kastet lys over hvordan maskinlæring kan bidra til å stoppe spredningen av sykdommer. I denne artikkelen vil du kjenne igjen bruken av ARIMA, logistiske kurver, lineær regresjon og SARIMA. "Dette arbeidet er et forsøk på å beregne spredningshastigheten for dette viruset og dermed forutsi dødsfall, tilfriskninger og bekreftede tilfeller, slik at det kan hjelpe oss med å forberede oss bedre og overleve."
[Referanse](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7979218/)

## 🌲 Økologi og grønn teknologi

Natur og økologi består av mange sensitive systemer hvor samspillet mellom dyr og natur kommer i fokus. Det er viktig å kunne måle disse systemene nøyaktig og handle riktig hvis noe skjer, som en skogbrann eller en nedgang i dyrepopulasjonen.

### Skogforvaltning

Du lærte om [forsterkende læring](../../8-Reinforcement/README.md) i tidligere leksjoner. Det kan være veldig nyttig når man prøver å forutsi mønstre i naturen. Spesielt kan det brukes til å spore økologiske problemer som skogbranner og spredning av invasive arter. I Canada brukte en gruppe forskere forsterkende læring for å bygge modeller for skogbrannens dynamikk basert på satellittbilder. Ved å bruke en innovativ "spatially spreading process (SSP)" forestilte de seg en skogbrann som "agenten ved enhver celle i landskapet." "Sett med handlinger brannen kan ta fra et sted på et gitt tidspunkt inkluderer å spre seg nord, sør, øst eller vest, eller ikke spre seg."

Denne tilnærmingen snur den vanlige RL-oppsettet siden dynamikken i den tilsvarende Markov Decision Process (MDP) er en kjent funksjon for umiddelbar spredning av skogbrann." Les mer om de klassiske algoritmene brukt av denne gruppen på lenken nedenfor.
[Referanse](https://www.frontiersin.org/articles/10.3389/fict.2018.00006/full)

### Bevegelsessporing av dyr

Mens dyp læring har skapt en revolusjon i visuell sporing av dyrebevegelser (du kan bygge din egen [isbjørnsporer](https://docs.microsoft.com/learn/modules/build-ml-model-with-azure-stream-analytics/?WT.mc_id=academic-77952-leestott) her), har klassisk ML fortsatt en plass i denne oppgaven.

Sensorer for å spore bevegelser hos husdyr og IoT bruker denne typen visuell prosessering, men mer grunnleggende ML-teknikker er nyttige for å forhåndsbehandle data. For eksempel, i denne artikkelen ble sauers kroppsholdninger overvåket og analysert ved hjelp av ulike klassifiseringsalgoritmer. Du vil kanskje kjenne igjen ROC-kurven på side 335.
[Referanse](https://druckhaus-hofmann.de/gallery/31-wj-feb-2020.pdf)

### ⚡️ Energistyring

I våre leksjoner om [tidsserieprognoser](../../7-TimeSeries/README.md) introduserte vi konseptet med smarte parkeringsmålere for å generere inntekter for en by basert på forståelse av tilbud og etterspørsel. Denne artikkelen diskuterer i detalj hvordan klynging, regresjon og tidsserieprognoser kombineres for å hjelpe med å forutsi fremtidig energibruk i Irland, basert på smarte målere.
[Referanse](https://www-cdn.knime.com/sites/default/files/inline-images/knime_bigdata_energy_timeseries_whitepaper.pdf)

## 💼 Forsikring

Forsikringssektoren er en annen sektor som bruker ML for å konstruere og optimalisere levedyktige finansielle og aktuarielle modeller.

### Volatilitetsstyring

MetLife, en leverandør av livsforsikring, er åpen om hvordan de analyserer og reduserer volatilitet i sine finansielle modeller. I denne artikkelen vil du legge merke til visualiseringer av binær og ordinal klassifisering. Du vil også oppdage visualiseringer av prognoser.
[Referanse](https://investments.metlife.com/content/dam/metlifecom/us/investments/insights/research-topics/macro-strategy/pdf/MetLifeInvestmentManagement_MachineLearnedRanking_070920.pdf)

## 🎨 Kunst, kultur og litteratur

Innen kunst, for eksempel journalistikk, finnes det mange interessante problemer. Å oppdage falske nyheter er et stort problem, da det har vist seg å påvirke folks meninger og til og med undergrave demokratier. Museer kan også dra nytte av bruk av ML i alt fra å finne koblinger mellom artefakter til ressursplanlegging.

### Oppdage falske nyheter

Å oppdage falske nyheter har blitt et katt-og-mus-spill i dagens medier. I denne artikkelen foreslår forskere at et system som kombinerer flere av ML-teknikkene vi har studert kan testes, og den beste modellen kan implementeres: "Dette systemet er basert på naturlig språkbehandling for å trekke ut funksjoner fra dataene, og deretter brukes disse funksjonene til opplæring av maskinlæringsklassifiserere som Naive Bayes, Support Vector Machine (SVM), Random Forest (RF), Stochastic Gradient Descent (SGD) og Logistic Regression (LR)."
[Referanse](https://www.irjet.net/archives/V7/i6/IRJET-V7I6688.pdf)

Denne artikkelen viser hvordan kombinasjonen av ulike ML-domener kan gi interessante resultater som kan bidra til å stoppe spredningen av falske nyheter og forhindre reell skade; i dette tilfellet var motivasjonen spredningen av rykter om COVID-behandlinger som utløste voldelige opptøyer.

### Museum ML

Museer er på terskelen til en AI-revolusjon hvor katalogisering og digitalisering av samlinger og finne koblinger mellom artefakter blir enklere etter hvert som teknologien utvikler seg. Prosjekter som [In Codice Ratio](https://www.sciencedirect.com/science/article/abs/pii/S0306457321001035#:~:text=1.,studies%20over%20large%20historical%20sources.) hjelper med å låse opp mysteriene i utilgjengelige samlinger som Vatikanets arkiver. Men, den forretningsmessige siden av museer drar også nytte av ML-modeller.

For eksempel bygde Art Institute of Chicago modeller for å forutsi hva publikum er interessert i og når de vil besøke utstillinger. Målet er å skape individualiserte og optimaliserte besøksopplevelser hver gang brukeren besøker museet. "I regnskapsåret 2017 forutså modellen besøkstall og inntekter med en nøyaktighet på 1 prosent, sier Andrew Simnick, senior visepresident ved Art Institute."
[Referanse](https://www.chicagobusiness.com/article/20180518/ISSUE01/180519840/art-institute-of-chicago-uses-data-to-make-exhibit-choices)

## 🏷 Markedsføring

### Kundesegmentering

De mest effektive markedsføringsstrategiene retter seg mot kunder på ulike måter basert på forskjellige grupperinger. I denne artikkelen diskuteres bruken av klyngingsalgoritmer for å støtte differensiert markedsføring. Differensiert markedsføring hjelper selskaper med å forbedre merkevaregjenkjenning, nå flere kunder og tjene mer penger.
[Referanse](https://ai.inqline.com/machine-learning-for-marketing-customer-segmentation/)

## 🚀 Utfordring

Identifiser en annen sektor som drar nytte av noen av teknikkene du har lært i dette kurset, og oppdag hvordan den bruker ML.
## [Quiz etter forelesning](https://ff-quizzes.netlify.app/en/ml/)

## Gjennomgang og selvstudium

Wayfair sitt data science-team har flere interessante videoer om hvordan de bruker ML i selskapet sitt. Det er verdt [å ta en titt](https://www.youtube.com/channel/UCe2PjkQXqOuwkW1gw6Ameuw/videos)!

## Oppgave

[En ML skattejakt](assignment.md)

---

**Ansvarsfraskrivelse**:  
Dette dokumentet er oversatt ved hjelp av AI-oversettelsestjenesten [Co-op Translator](https://github.com/Azure/co-op-translator). Selv om vi tilstreber nøyaktighet, vennligst vær oppmerksom på at automatiske oversettelser kan inneholde feil eller unøyaktigheter. Det originale dokumentet på sitt opprinnelige språk bør anses som den autoritative kilden. For kritisk informasjon anbefales profesjonell menneskelig oversettelse. Vi er ikke ansvarlige for eventuelle misforståelser eller feiltolkninger som oppstår ved bruk av denne oversettelsen.