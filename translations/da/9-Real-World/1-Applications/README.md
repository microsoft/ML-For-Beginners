<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "83320d6b6994909e35d830cebf214039",
  "translation_date": "2025-09-05T00:10:53+00:00",
  "source_file": "9-Real-World/1-Applications/README.md",
  "language_code": "da"
}
-->
# Postscript: Maskinlæring i den virkelige verden

![Oversigt over maskinlæring i den virkelige verden i en sketchnote](../../../../sketchnotes/ml-realworld.png)
> Sketchnote af [Tomomi Imura](https://www.twitter.com/girlie_mac)

I dette pensum har du lært mange måder at forberede data til træning og skabe maskinlæringsmodeller. Du har bygget en række klassiske modeller inden for regression, klyngedannelse, klassifikation, naturlig sprogbehandling og tidsserier. Tillykke! Nu undrer du dig måske over, hvad det hele skal bruges til... hvad er de virkelige anvendelser af disse modeller?

Selvom AI, som ofte benytter dyb læring, har vakt stor interesse i industrien, er der stadig værdifulde anvendelser for klassiske maskinlæringsmodeller. Du bruger måske allerede nogle af disse anvendelser i dag! I denne lektion vil du udforske, hvordan otte forskellige industrier og fagområder bruger disse typer modeller til at gøre deres applikationer mere effektive, pålidelige, intelligente og værdifulde for brugerne.

## [Quiz før lektionen](https://ff-quizzes.netlify.app/en/ml/)

## 💰 Finans

Finanssektoren tilbyder mange muligheder for maskinlæring. Mange problemer inden for dette område egner sig til at blive modelleret og løst ved hjælp af ML.

### Kreditkortsvindel

Vi lærte om [k-means clustering](../../5-Clustering/2-K-Means/README.md) tidligere i kurset, men hvordan kan det bruges til at løse problemer relateret til kreditkortsvindel?

K-means clustering er nyttigt i en teknik til kreditkortsvindel kaldet **outlier detection**. Outliers, eller afvigelser i observationer om et datasæt, kan fortælle os, om et kreditkort bruges normalt eller om noget usædvanligt foregår. Som vist i det linkede papir kan du sortere kreditkortdata ved hjælp af en k-means clustering-algoritme og tildele hver transaktion til en klynge baseret på, hvor meget den afviger. Derefter kan du evaluere de mest risikable klynger for at afgøre, om transaktionerne er svigagtige eller legitime.
[Reference](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.680.1195&rep=rep1&type=pdf)

### Formueforvaltning

I formueforvaltning håndterer en person eller virksomhed investeringer på vegne af deres klienter. Deres opgave er at opretholde og øge formuen på lang sigt, så det er afgørende at vælge investeringer, der klarer sig godt.

En måde at evaluere, hvordan en bestemt investering klarer sig, er gennem statistisk regression. [Lineær regression](../../2-Regression/1-Tools/README.md) er et værdifuldt værktøj til at forstå, hvordan en fond klarer sig i forhold til en benchmark. Vi kan også vurdere, om resultaterne af regressionen er statistisk signifikante, eller hvor meget de vil påvirke en klients investeringer. Du kan endda udvide din analyse med multipel regression, hvor yderligere risikofaktorer kan tages i betragtning. For et eksempel på, hvordan dette ville fungere for en specifik fond, kan du se papiret nedenfor om evaluering af fondsperformance ved hjælp af regression.
[Reference](http://www.brightwoodventures.com/evaluating-fund-performance-using-regression/)

## 🎓 Uddannelse

Uddannelsessektoren er også et meget interessant område, hvor ML kan anvendes. Der er spændende problemer at tackle, såsom at opdage snyd i tests eller essays eller håndtere bias, bevidst eller ubevidst, i bedømmelsesprocessen.

### Forudsigelse af studerendes adfærd

[Coursera](https://coursera.com), en online udbyder af åbne kurser, har en fantastisk teknologiblog, hvor de diskuterer mange ingeniørbeslutninger. I denne case study plotter de en regressionslinje for at undersøge en mulig korrelation mellem en lav NPS (Net Promoter Score) og fastholdelse eller frafald fra kurser.
[Reference](https://medium.com/coursera-engineering/controlled-regression-quantifying-the-impact-of-course-quality-on-learner-retention-31f956bd592a)

### Reducering af bias

[Grammarly](https://grammarly.com), en skriveassistent, der tjekker for stave- og grammatikfejl, bruger sofistikerede [naturlige sprogbehandlingssystemer](../../6-NLP/README.md) i sine produkter. De har offentliggjort en interessant case study i deres teknologiblog om, hvordan de håndterede kønsbias i maskinlæring, som du lærte om i vores [introduktionslektion om fairness](../../1-Introduction/3-fairness/README.md).
[Reference](https://www.grammarly.com/blog/engineering/mitigating-gender-bias-in-autocorrect/)

## 👜 Detailhandel

Detailsektoren kan bestemt drage fordel af brugen af ML, lige fra at skabe en bedre kunderejse til at optimere lagerstyring.

### Personalisering af kunderejsen

Hos Wayfair, en virksomhed der sælger boligartikler som møbler, er det afgørende at hjælpe kunderne med at finde de rigtige produkter til deres smag og behov. I denne artikel beskriver ingeniører fra virksomheden, hvordan de bruger ML og NLP til at "vise de rigtige resultater for kunderne". Deres Query Intent Engine er bygget til at bruge enhedsekstraktion, klassifikatortræning, udtrækning af aktiver og meninger samt sentiment-tagging på kundeanmeldelser. Dette er et klassisk eksempel på, hvordan NLP fungerer i online detailhandel.
[Reference](https://www.aboutwayfair.com/tech-innovation/how-we-use-machine-learning-and-natural-language-processing-to-empower-search)

### Lagerstyring

Innovative, agile virksomheder som [StitchFix](https://stitchfix.com), en abonnementsservice der sender tøj til forbrugere, er stærkt afhængige af ML til anbefalinger og lagerstyring. Deres stylingteams arbejder sammen med deres merchandisingteams: "En af vores dataforskere eksperimenterede med en genetisk algoritme og anvendte den på beklædning for at forudsige, hvad der ville være et succesfuldt stykke tøj, der ikke eksisterer i dag. Vi præsenterede det for merchandise-teamet, og nu kan de bruge det som et værktøj."
[Reference](https://www.zdnet.com/article/how-stitch-fix-uses-machine-learning-to-master-the-science-of-styling/)

## 🏥 Sundhedssektoren

Sundhedssektoren kan bruge ML til at optimere forskningsopgaver og logistiske problemer som genindlæggelse af patienter eller at stoppe sygdomme fra at sprede sig.

### Håndtering af kliniske forsøg

Toksicitet i kliniske forsøg er en stor bekymring for medicinalvirksomheder. Hvor meget toksicitet er acceptabelt? I denne undersøgelse førte analyser af forskellige kliniske forsøgsmetoder til udviklingen af en ny tilgang til at forudsige sandsynligheden for kliniske forsøgsresultater. Specifikt var de i stand til at bruge random forest til at producere en [klassifikator](../../4-Classification/README.md), der kan skelne mellem grupper af lægemidler.
[Reference](https://www.sciencedirect.com/science/article/pii/S2451945616302914)

### Håndtering af genindlæggelser på hospitaler

Hospitalpleje er dyrt, især når patienter skal genindlægges. Denne artikel diskuterer en virksomhed, der bruger ML til at forudsige potentialet for genindlæggelse ved hjælp af [klyngedannelse](../../5-Clustering/README.md) algoritmer. Disse klynger hjælper analytikere med at "opdage grupper af genindlæggelser, der kan have en fælles årsag".
[Reference](https://healthmanagement.org/c/healthmanagement/issuearticle/hospital-readmissions-and-machine-learning)

### Sygdomshåndtering

Den nylige pandemi har kastet lys over, hvordan maskinlæring kan hjælpe med at stoppe spredningen af sygdomme. I denne artikel vil du genkende brugen af ARIMA, logistiske kurver, lineær regression og SARIMA. "Dette arbejde er et forsøg på at beregne spredningshastigheden for denne virus og dermed forudsige dødsfald, helbredelser og bekræftede tilfælde, så det kan hjælpe os med at forberede os bedre og overleve."
[Reference](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7979218/)

## 🌲 Økologi og grøn teknologi

Natur og økologi består af mange følsomme systemer, hvor samspillet mellem dyr og natur kommer i fokus. Det er vigtigt at kunne måle disse systemer nøjagtigt og handle passende, hvis noget sker, som en skovbrand eller et fald i dyrepopulationen.

### Skovforvaltning

Du lærte om [Reinforcement Learning](../../8-Reinforcement/README.md) i tidligere lektioner. Det kan være meget nyttigt, når man forsøger at forudsige mønstre i naturen. Især kan det bruges til at spore økologiske problemer som skovbrande og spredning af invasive arter. I Canada brugte en gruppe forskere Reinforcement Learning til at bygge modeller for skovbranddynamik baseret på satellitbilleder. Ved hjælp af en innovativ "spatially spreading process (SSP)" forestillede de sig en skovbrand som "agenten ved enhver celle i landskabet." "Sættet af handlinger, som branden kan tage fra en placering på ethvert tidspunkt, inkluderer spredning nord, syd, øst eller vest eller ikke at sprede sig."

Denne tilgang vender den sædvanlige RL-opsætning på hovedet, da dynamikken i den tilsvarende Markov Decision Process (MDP) er en kendt funktion for øjeblikkelig skovbrandspredning. Læs mere om de klassiske algoritmer, som denne gruppe brugte, på linket nedenfor.
[Reference](https://www.frontiersin.org/articles/10.3389/fict.2018.00006/full)

### Bevægelsessporing af dyr

Mens dyb læring har skabt en revolution i visuel sporing af dyrebevægelser (du kan bygge din egen [isbjørnesporer](https://docs.microsoft.com/learn/modules/build-ml-model-with-azure-stream-analytics/?WT.mc_id=academic-77952-leestott) her), har klassisk ML stadig en plads i denne opgave.

Sensorer til at spore bevægelser af husdyr og IoT gør brug af denne type visuel behandling, men mere grundlæggende ML-teknikker er nyttige til at forbehandle data. For eksempel blev fåres holdninger overvåget og analyseret ved hjælp af forskellige klassifikatoralgoritmer. Du vil måske genkende ROC-kurven på side 335.
[Reference](https://druckhaus-hofmann.de/gallery/31-wj-feb-2020.pdf)

### ⚡️ Energistyring

I vores lektioner om [tidsserieprognoser](../../7-TimeSeries/README.md) introducerede vi konceptet med smarte parkeringsmålere til at generere indtægter for en by baseret på forståelse af udbud og efterspørgsel. Denne artikel diskuterer detaljeret, hvordan klyngedannelse, regression og tidsserieprognoser kombineres for at hjælpe med at forudsige fremtidigt energiforbrug i Irland baseret på smarte målere.
[Reference](https://www-cdn.knime.com/sites/default/files/inline-images/knime_bigdata_energy_timeseries_whitepaper.pdf)

## 💼 Forsikring

Forsikringssektoren er en anden sektor, der bruger ML til at konstruere og optimere levedygtige finansielle og aktuarielle modeller.

### Volatilitetsstyring

MetLife, en livsforsikringsudbyder, er åben omkring, hvordan de analyserer og reducerer volatilitet i deres finansielle modeller. I denne artikel vil du bemærke visualiseringer af binær og ordinal klassifikation. Du vil også opdage visualiseringer af prognoser.
[Reference](https://investments.metlife.com/content/dam/metlifecom/us/investments/insights/research-topics/macro-strategy/pdf/MetLifeInvestmentManagement_MachineLearnedRanking_070920.pdf)

## 🎨 Kunst, kultur og litteratur

Inden for kunst, for eksempel journalistik, er der mange interessante problemer. At opdage falske nyheder er et stort problem, da det har vist sig at påvirke folks meninger og endda vælte demokratier. Museer kan også drage fordel af at bruge ML til alt fra at finde forbindelser mellem artefakter til ressourceplanlægning.

### Falske nyheder

At opdage falske nyheder er blevet en kamp mellem kat og mus i dagens medier. I denne artikel foreslår forskere, at et system, der kombinerer flere af de ML-teknikker, vi har studeret, kan testes, og den bedste model kan implementeres: "Dette system er baseret på naturlig sprogbehandling til at udtrække funktioner fra dataene, og derefter bruges disse funktioner til træning af maskinlæringsklassifikatorer såsom Naive Bayes, Support Vector Machine (SVM), Random Forest (RF), Stochastic Gradient Descent (SGD) og Logistic Regression (LR)."
[Reference](https://www.irjet.net/archives/V7/i6/IRJET-V7I6688.pdf)

Denne artikel viser, hvordan kombinationen af forskellige ML-domæner kan producere interessante resultater, der kan hjælpe med at stoppe falske nyheder fra at sprede sig og skabe reel skade; i dette tilfælde var drivkraften spredningen af rygter om COVID-behandlinger, der inciterede voldelige optøjer.

### Museum ML

Museer står på tærsklen til en AI-revolution, hvor katalogisering og digitalisering af samlinger samt at finde forbindelser mellem artefakter bliver lettere, efterhånden som teknologien udvikler sig. Projekter som [In Codice Ratio](https://www.sciencedirect.com/science/article/abs/pii/S0306457321001035#:~:text=1.,studies%20over%20large%20historical%20sources.) hjælper med at låse op for mysterierne i utilgængelige samlinger som Vatikanets arkiver. Men den forretningsmæssige side af museer drager også fordel af ML-modeller.

For eksempel byggede Art Institute of Chicago modeller til at forudsige, hvad publikum er interesseret i, og hvornår de vil besøge udstillinger. Målet er at skabe individualiserede og optimerede besøgsoplevelser hver gang brugeren besøger museet. "I regnskabsåret 2017 forudsagde modellen besøgstal og indtægter med en nøjagtighed på 1 procent, siger Andrew Simnick, senior vice president ved Art Institute."
[Reference](https://www.chicagobusiness.com/article/20180518/ISSUE01/180519840/art-institute-of-chicago-uses-data-to-make-exhibit-choices)

## 🏷 Marketing

### Kundesegmentering

De mest effektive marketingstrategier målretter kunder på forskellige måder baseret på forskellige grupperinger. I denne artikel diskuteres brugen af klyngedannelsesalgoritmer til at understøtte differentieret marketing. Differentieret marketing hjælper virksomheder med at forbedre brandgenkendelse, nå flere kunder og tjene flere penge.
[Reference](https://ai.inqline.com/machine-learning-for-marketing-customer-segmentation/)

## 🚀 Udfordring

Identificer en anden sektor, der drager fordel af nogle af de teknikker, du har lært i dette pensum, og undersøg, hvordan den bruger ML.
## [Quiz efter forelæsning](https://ff-quizzes.netlify.app/en/ml/)

## Gennemgang & Selvstudie

Wayfair's data science-team har flere interessante videoer om, hvordan de bruger ML i deres virksomhed. Det er værd at [tage et kig](https://www.youtube.com/channel/UCe2PjkQXqOuwkW1gw6Ameuw/videos)!

## Opgave

[En ML skattejagt](assignment.md)

---

**Ansvarsfraskrivelse**:  
Dette dokument er blevet oversat ved hjælp af AI-oversættelsestjenesten [Co-op Translator](https://github.com/Azure/co-op-translator). Selvom vi bestræber os på nøjagtighed, skal du være opmærksom på, at automatiserede oversættelser kan indeholde fejl eller unøjagtigheder. Det originale dokument på dets oprindelige sprog bør betragtes som den autoritative kilde. For kritisk information anbefales professionel menneskelig oversættelse. Vi påtager os ikke ansvar for eventuelle misforståelser eller fejltolkninger, der opstår som følge af brugen af denne oversættelse.