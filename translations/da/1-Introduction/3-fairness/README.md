<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9a6b702d1437c0467e3c5c28d763dac2",
  "translation_date": "2025-09-05T00:22:19+00:00",
  "source_file": "1-Introduction/3-fairness/README.md",
  "language_code": "da"
}
-->
# Bygge maskinlæringsløsninger med ansvarlig AI

![Oversigt over ansvarlig AI i maskinlæring i en sketchnote](../../../../sketchnotes/ml-fairness.png)
> Sketchnote af [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Quiz før lektionen](https://ff-quizzes.netlify.app/en/ml/)

## Introduktion

I dette pensum vil du begynde at opdage, hvordan maskinlæring kan og allerede påvirker vores daglige liv. Selv nu er systemer og modeller involveret i daglige beslutningsopgaver, såsom sundhedsdiagnoser, låneansøgninger eller afsløring af svindel. Derfor er det vigtigt, at disse modeller fungerer godt for at levere resultater, der er troværdige. Ligesom enhver softwareapplikation vil AI-systemer kunne fejle eller have uønskede resultater. Derfor er det afgørende at kunne forstå og forklare adfærden af en AI-model.

Forestil dig, hvad der kan ske, når de data, du bruger til at bygge disse modeller, mangler visse demografiske grupper, såsom race, køn, politisk holdning, religion, eller uforholdsmæssigt repræsenterer sådanne grupper. Hvad med når modellens output tolkes til at favorisere en bestemt demografisk gruppe? Hvad er konsekvensen for applikationen? Derudover, hvad sker der, når modellen har et negativt resultat og skader mennesker? Hvem er ansvarlig for AI-systemets adfærd? Dette er nogle af de spørgsmål, vi vil udforske i dette pensum.

I denne lektion vil du:

- Øge din bevidsthed om vigtigheden af retfærdighed i maskinlæring og skader relateret til retfærdighed.
- Blive bekendt med praksis for at udforske outliers og usædvanlige scenarier for at sikre pålidelighed og sikkerhed.
- Få forståelse for behovet for at styrke alle ved at designe inkluderende systemer.
- Udforske, hvor vigtigt det er at beskytte privatliv og sikkerhed for data og mennesker.
- Se vigtigheden af en "glasboks"-tilgang til at forklare adfærden af AI-modeller.
- Være opmærksom på, hvordan ansvarlighed er afgørende for at opbygge tillid til AI-systemer.

## Forudsætning

Som forudsætning bedes du tage "Principper for ansvarlig AI"-læringsstien og se videoen nedenfor om emnet:

Lær mere om ansvarlig AI ved at følge denne [læringssti](https://docs.microsoft.com/learn/modules/responsible-ai-principles/?WT.mc_id=academic-77952-leestott)

[![Microsofts tilgang til ansvarlig AI](https://img.youtube.com/vi/dnC8-uUZXSc/0.jpg)](https://youtu.be/dnC8-uUZXSc "Microsofts tilgang til ansvarlig AI")

> 🎥 Klik på billedet ovenfor for en video: Microsofts tilgang til ansvarlig AI

## Retfærdighed

AI-systemer bør behandle alle retfærdigt og undgå at påvirke lignende grupper af mennesker på forskellige måder. For eksempel, når AI-systemer giver vejledning om medicinsk behandling, låneansøgninger eller ansættelse, bør de give de samme anbefalinger til alle med lignende symptomer, økonomiske forhold eller faglige kvalifikationer. Hver af os som mennesker bærer med os arvede fordomme, der påvirker vores beslutninger og handlinger. Disse fordomme kan være tydelige i de data, vi bruger til at træne AI-systemer. Sådan manipulation kan nogle gange ske utilsigtet. Det er ofte svært bevidst at vide, hvornår man introducerer bias i data.

**"Uretfærdighed"** omfatter negative konsekvenser eller "skader" for en gruppe mennesker, såsom dem defineret ud fra race, køn, alder eller handicapstatus. De vigtigste skader relateret til retfærdighed kan klassificeres som:

- **Allokering**, hvis et køn eller en etnicitet for eksempel favoriseres frem for en anden.
- **Kvalitet af service**. Hvis du træner data til et specifikt scenarie, men virkeligheden er meget mere kompleks, fører det til en dårligt fungerende service. For eksempel en håndsæbedispenser, der ikke kunne registrere personer med mørk hud. [Reference](https://gizmodo.com/why-cant-this-soap-dispenser-identify-dark-skin-1797931773)
- **Nedgørelse**. At kritisere og mærke noget eller nogen uretfærdigt. For eksempel mislabeled en billedmærkningsteknologi berømt billeder af mørkhudede mennesker som gorillaer.
- **Over- eller underrepræsentation**. Ideen er, at en bestemt gruppe ikke ses i en bestemt profession, og enhver service eller funktion, der fortsat fremmer dette, bidrager til skade.
- **Stereotyper**. At associere en given gruppe med forudbestemte attributter. For eksempel kan et sprogoversættelsessystem mellem engelsk og tyrkisk have unøjagtigheder på grund af ord med stereotype associationer til køn.

![oversættelse til tyrkisk](../../../../1-Introduction/3-fairness/images/gender-bias-translate-en-tr.png)
> oversættelse til tyrkisk

![oversættelse tilbage til engelsk](../../../../1-Introduction/3-fairness/images/gender-bias-translate-tr-en.png)
> oversættelse tilbage til engelsk

Når vi designer og tester AI-systemer, skal vi sikre, at AI er retfærdig og ikke programmeret til at træffe biased eller diskriminerende beslutninger, som mennesker også er forbudt mod at træffe. At garantere retfærdighed i AI og maskinlæring forbliver en kompleks socioteknisk udfordring.

### Pålidelighed og sikkerhed

For at opbygge tillid skal AI-systemer være pålidelige, sikre og konsistente under normale og uventede forhold. Det er vigtigt at vide, hvordan AI-systemer vil opføre sig i en række forskellige situationer, især når de er outliers. Når man bygger AI-løsninger, skal der være betydelig fokus på, hvordan man håndterer en bred vifte af omstændigheder, som AI-løsningerne ville støde på. For eksempel skal en selvkørende bil prioritere menneskers sikkerhed højt. Som et resultat skal AI, der driver bilen, tage højde for alle de mulige scenarier, bilen kunne støde på, såsom nat, tordenvejr eller snestorme, børn der løber over gaden, kæledyr, vejarbejder osv. Hvor godt et AI-system kan håndtere en bred vifte af forhold pålideligt og sikkert afspejler niveauet af forudseenhed, som dataforskeren eller AI-udvikleren har overvejet under design eller test af systemet.

> [🎥 Klik her for en video: ](https://www.microsoft.com/videoplayer/embed/RE4vvIl)

### Inklusion

AI-systemer bør designes til at engagere og styrke alle. Når dataforskere og AI-udviklere designer og implementerer AI-systemer, identificerer og adresserer de potentielle barrierer i systemet, der utilsigtet kunne ekskludere mennesker. For eksempel er der 1 milliard mennesker med handicap verden over. Med fremskridt inden for AI kan de få adgang til en bred vifte af information og muligheder lettere i deres daglige liv. Ved at adressere barriererne skabes der muligheder for at innovere og udvikle AI-produkter med bedre oplevelser, der gavner alle.

> [🎥 Klik her for en video: inklusion i AI](https://www.microsoft.com/videoplayer/embed/RE4vl9v)

### Sikkerhed og privatliv

AI-systemer bør være sikre og respektere folks privatliv. Folk har mindre tillid til systemer, der sætter deres privatliv, information eller liv i fare. Når vi træner maskinlæringsmodeller, stoler vi på data for at producere de bedste resultater. I den proces skal dataenes oprindelse og integritet overvejes. For eksempel, blev dataene indsendt af brugere eller offentligt tilgængelige? Dernæst, mens vi arbejder med dataene, er det afgørende at udvikle AI-systemer, der kan beskytte fortrolige oplysninger og modstå angreb. Efterhånden som AI bliver mere udbredt, bliver beskyttelse af privatliv og sikring af vigtige personlige og forretningsmæssige oplysninger mere kritisk og komplekst. Privatlivs- og datasikkerhedsproblemer kræver særlig opmærksomhed for AI, fordi adgang til data er afgørende for, at AI-systemer kan lave præcise og informerede forudsigelser og beslutninger om mennesker.

> [🎥 Klik her for en video: sikkerhed i AI](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Som industri har vi gjort betydelige fremskridt inden for privatliv og sikkerhed, drevet væsentligt af reguleringer som GDPR (General Data Protection Regulation).
- Alligevel må vi med AI-systemer erkende spændingen mellem behovet for mere personlige data for at gøre systemer mere personlige og effektive – og privatliv.
- Ligesom med internettets fødsel ser vi også en stor stigning i antallet af sikkerhedsproblemer relateret til AI.
- Samtidig har vi set AI blive brugt til at forbedre sikkerheden. For eksempel drives de fleste moderne antivirus-scannere i dag af AI-heuristikker.
- Vi skal sikre, at vores dataforskningsprocesser harmonisk integreres med de nyeste privatlivs- og sikkerhedspraksisser.

### Transparens

AI-systemer bør være forståelige. En afgørende del af transparens er at forklare adfærden af AI-systemer og deres komponenter. At forbedre forståelsen af AI-systemer kræver, at interessenter forstår, hvordan og hvorfor de fungerer, så de kan identificere potentielle præstationsproblemer, sikkerheds- og privatlivsbekymringer, bias, ekskluderende praksis eller utilsigtede resultater. Vi mener også, at de, der bruger AI-systemer, bør være ærlige og åbne om, hvornår, hvorfor og hvordan de vælger at implementere dem. Samt begrænsningerne af de systemer, de bruger. For eksempel, hvis en bank bruger et AI-system til at understøtte sine forbrugerlånsbeslutninger, er det vigtigt at undersøge resultaterne og forstå, hvilke data der påvirker systemets anbefalinger. Regeringer begynder at regulere AI på tværs af industrier, så dataforskere og organisationer skal forklare, om et AI-system opfylder lovgivningsmæssige krav, især når der er et uønsket resultat.

> [🎥 Klik her for en video: transparens i AI](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Fordi AI-systemer er så komplekse, er det svært at forstå, hvordan de fungerer og tolke resultaterne.
- Denne manglende forståelse påvirker, hvordan disse systemer administreres, operationaliseres og dokumenteres.
- Denne manglende forståelse påvirker endnu vigtigere de beslutninger, der træffes ved hjælp af de resultater, disse systemer producerer.

### Ansvarlighed

De mennesker, der designer og implementerer AI-systemer, skal være ansvarlige for, hvordan deres systemer fungerer. Behovet for ansvarlighed er særligt vigtigt med følsomme teknologier som ansigtsgenkendelse. For nylig har der været en stigende efterspørgsel efter ansigtsgenkendelsesteknologi, især fra retshåndhævende organisationer, der ser potentialet i teknologien til anvendelser som at finde forsvundne børn. Men disse teknologier kunne potentielt bruges af en regering til at sætte borgernes grundlæggende friheder i fare ved for eksempel at muliggøre kontinuerlig overvågning af specifikke individer. Derfor skal dataforskere og organisationer være ansvarlige for, hvordan deres AI-system påvirker individer eller samfundet.

[![Ledende AI-forsker advarer om masseovervågning gennem ansigtsgenkendelse](../../../../1-Introduction/3-fairness/images/accountability.png)](https://www.youtube.com/watch?v=Wldt8P5V6D0 "Microsofts tilgang til ansvarlig AI")

> 🎥 Klik på billedet ovenfor for en video: Advarsler om masseovervågning gennem ansigtsgenkendelse

I sidste ende er et af de største spørgsmål for vores generation, som den første generation, der bringer AI til samfundet, hvordan vi sikrer, at computere forbliver ansvarlige over for mennesker, og hvordan vi sikrer, at de mennesker, der designer computere, forbliver ansvarlige over for alle andre.

## Vurdering af påvirkning

Før du træner en maskinlæringsmodel, er det vigtigt at gennemføre en vurdering af påvirkning for at forstå formålet med AI-systemet; hvad den tilsigtede anvendelse er; hvor det vil blive implementeret; og hvem der vil interagere med systemet. Disse er nyttige for anmelder(e) eller testere, der evaluerer systemet, for at vide, hvilke faktorer der skal tages i betragtning, når man identificerer potentielle risici og forventede konsekvenser.

Følgende er fokusområder, når man gennemfører en vurdering af påvirkning:

* **Negative konsekvenser for individer**. At være opmærksom på eventuelle begrænsninger eller krav, ikke-understøttet brug eller kendte begrænsninger, der hindrer systemets ydeevne, er afgørende for at sikre, at systemet ikke bruges på en måde, der kan skade individer.
* **Data krav**. At få en forståelse af, hvordan og hvor systemet vil bruge data, gør det muligt for anmeldere at udforske eventuelle datakrav, du skal være opmærksom på (f.eks. GDPR eller HIPPA dataregler). Derudover skal du undersøge, om kilden eller mængden af data er tilstrækkelig til træning.
* **Opsummering af påvirkning**. Saml en liste over potentielle skader, der kunne opstå ved brug af systemet. Gennem hele ML-livscyklussen skal du gennemgå, om de identificerede problemer er afhjulpet eller adresseret.
* **Anvendelige mål** for hver af de seks kerneprincipper. Vurder, om målene fra hvert af principperne er opfyldt, og om der er nogen mangler.

## Fejlfinding med ansvarlig AI

Ligesom fejlfinding af en softwareapplikation er fejlfinding af et AI-system en nødvendig proces for at identificere og løse problemer i systemet. Der er mange faktorer, der kan påvirke en models ydeevne eller ansvarlighed. De fleste traditionelle modelpræstationsmålinger er kvantitative aggregater af en models ydeevne, hvilket ikke er tilstrækkeligt til at analysere, hvordan en model overtræder principperne for ansvarlig AI. Desuden er en maskinlæringsmodel en "black box", der gør det svært at forstå, hvad der driver dens resultater eller give forklaringer, når den begår fejl. Senere i dette kursus vil vi lære, hvordan man bruger Responsible AI-dashboardet til at hjælpe med fejlfinding af AI-systemer. Dashboardet giver et holistisk værktøj til dataforskere og AI-udviklere til at udføre:

* **Fejlanalyse**. For at identificere fejlfordelingen i modellen, der kan påvirke systemets retfærdighed eller pålidelighed.
* **Modeloversigt**. For at opdage, hvor der er forskelle i modellens ydeevne på tværs af datakohorter.
* **Dataanalyse**. For at forstå datadistributionen og identificere eventuel bias i dataene, der kunne føre til problemer med retfærdighed, inklusion og pålidelighed.
* **Modelfortolkning**. For at forstå, hvad der påvirker eller influerer modellens forudsigelser. Dette hjælper med at forklare modellens adfærd, hvilket er vigtigt for transparens og ansvarlighed.

## 🚀 Udfordring

For at forhindre skader i at blive introduceret i første omgang bør vi:

- have en mangfoldighed af baggrunde og perspektiver blandt de mennesker, der arbejder på systemer
- investere i datasæt, der afspejler mangfoldigheden i vores samfund
- udvikle bedre metoder gennem hele maskinlæringslivscyklussen til at opdage og rette ansvarlig AI, når det opstår

Tænk på virkelige scenarier, hvor en models utroværdighed er tydelig i modelbygning og brug. Hvad bør vi ellers overveje?

## [Quiz efter lektionen](https://ff-quizzes.netlify.app/en/ml/)

## Gennemgang & Selvstudie

I denne lektion har du lært nogle grundlæggende begreber om retfærdighed og uretfærdighed i maskinlæring.
Se denne workshop for at dykke dybere ned i emnerne:

- På jagt efter ansvarlig AI: Fra principper til praksis af Besmira Nushi, Mehrnoosh Sameki og Amit Sharma

[![Responsible AI Toolbox: En open-source ramme for at bygge ansvarlig AI](https://img.youtube.com/vi/tGgJCrA-MZU/0.jpg)](https://www.youtube.com/watch?v=tGgJCrA-MZU "RAI Toolbox: En open-source ramme for at bygge ansvarlig AI")

> 🎥 Klik på billedet ovenfor for en video: RAI Toolbox: En open-source ramme for at bygge ansvarlig AI af Besmira Nushi, Mehrnoosh Sameki og Amit Sharma

Læs også:

- Microsofts RAI ressourcecenter: [Responsible AI Resources – Microsoft AI](https://www.microsoft.com/ai/responsible-ai-resources?activetab=pivot1%3aprimaryr4)

- Microsofts FATE forskningsgruppe: [FATE: Fairness, Accountability, Transparency, and Ethics in AI - Microsoft Research](https://www.microsoft.com/research/theme/fate/)

RAI Toolbox:

- [Responsible AI Toolbox GitHub repository](https://github.com/microsoft/responsible-ai-toolbox)

Læs om Azure Machine Learnings værktøjer til at sikre retfærdighed:

- [Azure Machine Learning](https://docs.microsoft.com/azure/machine-learning/concept-fairness-ml?WT.mc_id=academic-77952-leestott)

## Opgave

[Udforsk RAI Toolbox](assignment.md)

---

**Ansvarsfraskrivelse**:  
Dette dokument er blevet oversat ved hjælp af AI-oversættelsestjenesten [Co-op Translator](https://github.com/Azure/co-op-translator). Selvom vi bestræber os på nøjagtighed, skal du være opmærksom på, at automatiserede oversættelser kan indeholde fejl eller unøjagtigheder. Det originale dokument på dets oprindelige sprog bør betragtes som den autoritative kilde. For kritisk information anbefales professionel menneskelig oversættelse. Vi er ikke ansvarlige for eventuelle misforståelser eller fejltolkninger, der opstår som følge af brugen af denne oversættelse.