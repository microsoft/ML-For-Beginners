# Postscript: Modellfelsökning i maskininlärning med komponenter från Responsible AI-dashboarden

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Introduktion

Maskininlärning påverkar våra dagliga liv. AI hittar sin väg in i några av de mest betydelsefulla systemen som påverkar oss som individer och vårt samhälle, från sjukvård, finans, utbildning och anställning. Till exempel används system och modeller i dagliga beslutsprocesser, såsom diagnoser inom sjukvården eller att upptäcka bedrägerier. Följaktligen möts framstegen inom AI och dess snabba adoption av förändrade samhällsförväntningar och ökande regleringar. Vi ser ständigt områden där AI-system inte lever upp till förväntningarna; de avslöjar nya utmaningar, och regeringar börjar reglera AI-lösningar. Därför är det viktigt att dessa modeller analyseras för att säkerställa rättvisa, pålitliga, inkluderande, transparenta och ansvarsfulla resultat för alla.

I denna kurs kommer vi att titta på praktiska verktyg som kan användas för att bedöma om en modell har problem med ansvarsfull AI. Traditionella felsökningstekniker inom maskininlärning tenderar att baseras på kvantitativa beräkningar, såsom aggregerad noggrannhet eller genomsnittlig felkostnad. Föreställ dig vad som kan hända när datan du använder för att bygga dessa modeller saknar vissa demografiska grupper, såsom ras, kön, politiska åsikter, religion, eller oproportionerligt representerar sådana grupper. Vad händer när modellens resultat tolkas som att gynna vissa demografiska grupper? Detta kan leda till över- eller underrepresentation av dessa känsliga egenskapsgrupper, vilket resulterar i rättvise-, inkluderings- eller tillförlitlighetsproblem från modellen. En annan faktor är att maskininlärningsmodeller ofta betraktas som "svarta lådor", vilket gör det svårt att förstå och förklara vad som driver modellens förutsägelser. Alla dessa är utmaningar som dataforskare och AI-utvecklare står inför när de saknar tillräckliga verktyg för att felsöka och bedöma rättvisan eller tillförlitligheten hos en modell.

I denna lektion kommer du att lära dig att felsöka dina modeller med hjälp av:

- **Felsanalys**: identifiera var i din datadistribution modellen har höga felfrekvenser.
- **Modellöversikt**: utför jämförande analyser mellan olika datakohorter för att upptäcka skillnader i modellens prestationsmått.
- **Dataanalys**: undersök var det kan finnas över- eller underrepresentation i din data som kan snedvrida modellen till att gynna en demografisk grupp framför en annan.
- **Egenskapsbetydelse**: förstå vilka egenskaper som driver modellens förutsägelser på en global eller lokal nivå.

## Förkunskaper

Som förkunskap, vänligen granska [Responsible AI tools for developers](https://www.microsoft.com/ai/ai-lab-responsible-ai-dashboard)

> ![Gif om Responsible AI Tools](../../../../9-Real-World/2-Debugging-ML-Models/images/rai-overview.gif)

## Felsanalys

Traditionella prestationsmått för modeller som används för att mäta noggrannhet är oftast beräkningar baserade på korrekta kontra felaktiga förutsägelser. Till exempel kan det anses vara en bra prestation att fastställa att en modell är korrekt 89 % av tiden med en felkostnad på 0,001. Fel är dock ofta inte jämnt fördelade i din underliggande dataset. Du kan få en modellnoggrannhet på 89 %, men upptäcka att det finns olika områden i din data där modellen misslyckas 42 % av tiden. Konsekvensen av dessa felmönster med vissa datagrupper kan leda till rättvise- eller tillförlitlighetsproblem. Det är avgörande att förstå områden där modellen presterar bra eller inte. De dataområden där det finns ett högt antal felaktigheter i din modell kan visa sig vara en viktig demografisk grupp.

![Analysera och felsök modellfel](../../../../9-Real-World/2-Debugging-ML-Models/images/ea-error-distribution.png)

Felsanalyskomponenten på RAI-dashboarden illustrerar hur modellfel är fördelade över olika kohorter med en trädvisualisering. Detta är användbart för att identifiera egenskaper eller områden där det finns en hög felfrekvens i din dataset. Genom att se var de flesta av modellens felaktigheter kommer ifrån kan du börja undersöka grundorsaken. Du kan också skapa datakohorter för att utföra analyser. Dessa datakohorter hjälper i felsökningsprocessen för att avgöra varför modellens prestation är bra i en kohort men felaktig i en annan.

![Felsanalys](../../../../9-Real-World/2-Debugging-ML-Models/images/ea-error-cohort.png)

De visuella indikatorerna på trädkartan hjälper till att snabbare lokalisera problemområden. Till exempel, ju mörkare röd färg en trädnod har, desto högre är felfrekvensen.

Värmekarta är en annan visualiseringsfunktion som användare kan använda för att undersöka felfrekvensen med hjälp av en eller två egenskaper för att hitta bidragande faktorer till modellfel över hela datasetet eller kohorter.

![Felsanalys Värmekarta](../../../../9-Real-World/2-Debugging-ML-Models/images/ea-heatmap.png)

Använd felsanalys när du behöver:

* Få en djup förståelse för hur modellfel är fördelade över en dataset och över flera indata- och egenskapsdimensioner.
* Bryta ner de aggregerade prestationsmåtten för att automatiskt upptäcka felaktiga kohorter och informera om dina riktade åtgärdssteg.

## Modellöversikt

Att utvärdera en maskininlärningsmodells prestation kräver en holistisk förståelse av dess beteende. Detta kan uppnås genom att granska mer än ett mått, såsom felfrekvens, noggrannhet, återkallelse, precision eller MAE (Mean Absolute Error) för att hitta skillnader bland prestationsmått. Ett prestationsmått kan se bra ut, men felaktigheter kan avslöjas i ett annat mått. Dessutom hjälper jämförelse av måtten för skillnader över hela datasetet eller kohorter till att belysa var modellen presterar bra eller inte. Detta är särskilt viktigt för att se modellens prestation bland känsliga kontra okänsliga egenskaper (t.ex. patientens ras, kön eller ålder) för att upptäcka potentiell orättvisa modellen kan ha. Till exempel kan upptäckten att modellen är mer felaktig i en kohort med känsliga egenskaper avslöja potentiell orättvisa modellen kan ha.

Modellöversiktskomponenten på RAI-dashboarden hjälper inte bara till att analysera prestationsmåtten för datarepresentationen i en kohort, utan ger användare möjlighet att jämföra modellens beteende över olika kohorter.

![Datasetkohorter - modellöversikt i RAI-dashboarden](../../../../9-Real-World/2-Debugging-ML-Models/images/model-overview-dataset-cohorts.png)

Komponentens egenskapsbaserade analysfunktion gör det möjligt för användare att begränsa datasubgrupper inom en viss egenskap för att identifiera avvikelser på en detaljerad nivå. Till exempel har dashboarden inbyggd intelligens för att automatiskt generera kohorter för en användarvald egenskap (t.ex. *"time_in_hospital < 3"* eller *"time_in_hospital >= 7"*). Detta gör det möjligt för en användare att isolera en viss egenskap från en större datagrupp för att se om den är en nyckelpåverkare av modellens felaktiga resultat.

![Egenskapskohorter - modellöversikt i RAI-dashboarden](../../../../9-Real-World/2-Debugging-ML-Models/images/model-overview-feature-cohorts.png)

Modellöversiktskomponenten stödjer två klasser av skillnadsmått:

**Skillnad i modellprestation**: Dessa mått beräknar skillnaden i värdena för det valda prestationsmåttet över datagrupper. Här är några exempel:

* Skillnad i noggrannhetsgrad
* Skillnad i felfrekvens
* Skillnad i precision
* Skillnad i återkallelse
* Skillnad i medelfel (MAE)

**Skillnad i urvalsfrekvens**: Detta mått innehåller skillnaden i urvalsfrekvens (gynnsam förutsägelse) bland datagrupper. Ett exempel på detta är skillnaden i lånegodkännandefrekvens. Urvalsfrekvens betyder andelen datapunkter i varje klass som klassificeras som 1 (i binär klassificering) eller fördelningen av förutsägelsevärden (i regression).

## Dataanalys

> "Om du torterar datan tillräckligt länge kommer den att erkänna vad som helst" - Ronald Coase

Detta uttalande låter extremt, men det är sant att data kan manipuleras för att stödja vilken slutsats som helst. Sådan manipulation kan ibland ske oavsiktligt. Som människor har vi alla fördomar, och det är ofta svårt att medvetet veta när man introducerar fördomar i data. Att garantera rättvisa inom AI och maskininlärning förblir en komplex utmaning.

Data är en stor blind fläck för traditionella prestationsmått för modeller. Du kan ha höga noggrannhetspoäng, men detta återspeglar inte alltid den underliggande databias som kan finnas i din dataset. Till exempel, om en dataset med anställda har 27 % kvinnor i ledande positioner i ett företag och 73 % män på samma nivå, kan en AI-modell för jobbannonsering som tränats på denna data främst rikta sig mot en manlig publik för seniora jobbtjänster. Denna obalans i data snedvrider modellens förutsägelse till att gynna ett kön. Detta avslöjar ett rättviseproblem där det finns en könsbias i AI-modellen.

Dataanalyskomponenten på RAI-dashboarden hjälper till att identifiera områden där det finns över- och underrepresentation i datasetet. Den hjälper användare att diagnostisera grundorsaken till fel och rättviseproblem som introducerats från dataobalanser eller brist på representation av en viss datagrupp. Detta ger användare möjlighet att visualisera dataset baserat på förutsagda och faktiska resultat, felgrupper och specifika egenskaper. Ibland kan upptäckten av en underrepresenterad datagrupp också avslöja att modellen inte lär sig väl, vilket leder till höga felaktigheter. Att ha en modell med databias är inte bara ett rättviseproblem utan visar att modellen inte är inkluderande eller pålitlig.

![Dataanalyskomponent på RAI-dashboarden](../../../../9-Real-World/2-Debugging-ML-Models/images/dataanalysis-cover.png)

Använd dataanalys när du behöver:

* Utforska din datasets statistik genom att välja olika filter för att dela upp din data i olika dimensioner (även kallade kohorter).
* Förstå fördelningen av din dataset över olika kohorter och egenskapsgrupper.
* Avgöra om dina upptäckter relaterade till rättvisa, felsanalys och kausalitet (härledda från andra dashboardkomponenter) är ett resultat av din datasets fördelning.
* Besluta i vilka områden du ska samla in mer data för att minska fel som kommer från representationsproblem, etikettbrus, egenskapsbrus, etikettbias och liknande faktorer.

## Modelltolkning

Maskininlärningsmodeller tenderar att vara "svarta lådor". Att förstå vilka nyckeldataegenskaper som driver en modells förutsägelse kan vara utmanande. Det är viktigt att ge transparens kring varför en modell gör en viss förutsägelse. Till exempel, om ett AI-system förutspår att en diabetiker riskerar att bli återinlagd på sjukhus inom mindre än 30 dagar, bör det kunna ge stödjande data som ledde till dess förutsägelse. Att ha stödjande dataindikatorer ger transparens för att hjälpa kliniker eller sjukhus att kunna fatta välgrundade beslut. Dessutom gör det möjligt att förklara varför en modell gjorde en förutsägelse för en enskild patient att ansvarstagande med hälsoregleringar. När du använder maskininlärningsmodeller på sätt som påverkar människors liv är det avgörande att förstå och förklara vad som påverkar en modells beteende. Modellförklarbarhet och tolkning hjälper till att besvara frågor i scenarier såsom:

* Modellfelsökning: Varför gjorde min modell detta misstag? Hur kan jag förbättra min modell?
* Människa-AI-samarbete: Hur kan jag förstå och lita på modellens beslut?
* Regelöverensstämmelse: Uppfyller min modell juridiska krav?

Egenskapsbetydelsekomponenten på RAI-dashboarden hjälper dig att felsöka och få en omfattande förståelse för hur en modell gör förutsägelser. Det är också ett användbart verktyg för maskininlärningsproffs och beslutsfattare att förklara och visa bevis på egenskaper som påverkar modellens beteende för regelöverensstämmelse. Nästa steg är att användare kan utforska både globala och lokala förklaringar för att validera vilka egenskaper som driver en modells förutsägelse. Globala förklaringar listar de viktigaste egenskaperna som påverkat modellens övergripande förutsägelse. Lokala förklaringar visar vilka egenskaper som ledde till en modells förutsägelse för ett enskilt fall. Möjligheten att utvärdera lokala förklaringar är också användbar vid felsökning eller granskning av ett specifikt fall för att bättre förstå och tolka varför en modell gjorde en korrekt eller felaktig förutsägelse.

![Egenskapsbetydelsekomponent på RAI-dashboarden](../../../../9-Real-World/2-Debugging-ML-Models/images/9-feature-importance.png)

* Globala förklaringar: Till exempel, vilka egenskaper påverkar det övergripande beteendet hos en diabetesmodell för sjukhusåterinläggning?
* Lokala förklaringar: Till exempel, varför förutspåddes en diabetiker över 60 år med tidigare sjukhusvistelser att bli återinlagd eller inte återinlagd inom 30 dagar på ett sjukhus?

I felsökningsprocessen för att undersöka en modells prestation över olika kohorter visar Egenskapsbetydelse vilken nivå av påverkan en egenskap har över kohorterna. Den hjälper till att avslöja avvikelser när man jämför nivån av påverkan egenskapen har på att driva modellens felaktiga förutsägelser. Egenskapsbetydelsekomponenten kan visa vilka värden i en egenskap som positivt eller negativt påverkat modellens resultat. Till exempel, om en modell gjorde en felaktig förutsägelse, ger komponenten dig möjlighet att borra ner och identifiera vilka egenskaper eller egenskapsvärden som drev förutsägelsen. Denna detaljnivå hjälper inte bara vid felsökning utan ger transparens och ansvarstagande vid granskningssituationer. Slutligen kan komponenten hjälpa dig att identifiera rättviseproblem. För att illustrera, om en känslig egenskap såsom etnicitet eller kön är mycket inflytelserik i att driva en modells förutsägelse, kan detta vara ett tecken på ras- eller könsbias i modellen.

![Egenskapsbetydelse](../../../../9-Real-World/2-Debugging-ML-Models/images/9-features-influence.png)

Använd tolkning när du behöver:

* Avgöra hur pålitliga din AI-systems förutsägelser är genom att förstå vilka egenskaper som är viktigast för förutsägelserna.
* Närma dig felsökningen av din modell genom att först förstå den och identifiera om modellen använder sunda egenskaper eller bara falska korrelationer.
* Avslöja potentiella källor till orättvisa genom att förstå om modellen baserar förutsägelser på känsliga egenskaper eller på egenskaper som är starkt korrelerade med dem.
* Bygga användarförtroende för modellens beslut genom att generera lokala förklaringar för att illustrera deras resultat.
* Slutföra en regelgranskning av ett AI-system för att validera modeller och övervaka effekten av modellens beslut på människor.

## Slutsats

Alla komponenter i RAI-dashboarden är praktiska verktyg för att hjälpa dig bygga maskininlärningsmodeller som är mindre skadliga och mer pålitliga för samhället. De förbättrar förebyggandet av hot mot mänskliga rättigheter; diskriminering eller uteslutning av vissa grupper från livsmöjligheter; och risken för fysisk eller psykologisk skada. De hjälper också till att bygga förtroende för modellens beslut genom att generera lokala förklaringar för att illustrera deras resultat. Några av de potentiella skadorna kan klassificeras som:

- **Tilldelning**, om ett kön eller en etnicitet till exempel gynnas framför en annan.
- **Kvalitet på tjänsten**. Om du tränar data för ett specifikt scenario men verkligheten är mycket mer komplex, leder det till en dåligt presterande tjänst.
- **Stereotypisering**. Att associera en viss grupp med förutbestämda attribut.
- **Nedvärdering**. Att orättvist kritisera och ge negativa etiketter till något eller någon.
- **Över- eller underrepresentation**. Tanken är att en viss grupp inte syns inom ett visst yrke, och att varje tjänst eller funktion som fortsätter att främja detta bidrar till skada.

### Azure RAI-dashboard

[Azure RAI-dashboard](https://learn.microsoft.com/en-us/azure/machine-learning/concept-responsible-ai-dashboard?WT.mc_id=aiml-90525-ruyakubu) är byggd på öppen källkod utvecklad av ledande akademiska institutioner och organisationer, inklusive Microsoft, och är avgörande för dataforskare och AI-utvecklare för att bättre förstå modellbeteende, upptäcka och åtgärda oönskade problem från AI-modeller.

- Lär dig hur du använder de olika komponenterna genom att kolla in RAI-dashboardens [dokumentation.](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-responsible-ai-dashboard?WT.mc_id=aiml-90525-ruyakubu)

- Kolla in några exempel på RAI-dashboardens [notebooks](https://github.com/Azure/RAI-vNext-Preview/tree/main/examples/notebooks) för att felsöka mer ansvarsfulla AI-scenarier i Azure Machine Learning.

---
## 🚀 Utmaning

För att förhindra att statistiska eller datamässiga fördomar introduceras från början bör vi:

- ha en mångfald av bakgrunder och perspektiv bland de personer som arbetar med systemen
- investera i dataset som speglar mångfalden i vårt samhälle
- utveckla bättre metoder för att upptäcka och korrigera fördomar när de uppstår

Fundera på verkliga scenarier där orättvisa är tydlig i modellbyggande och användning. Vad mer bör vi ta hänsyn till?

## [Quiz efter föreläsningen](https://ff-quizzes.netlify.app/en/ml/)
## Granskning & Självstudier

I denna lektion har du lärt dig några praktiska verktyg för att integrera ansvarsfull AI i maskininlärning.

Titta på denna workshop för att fördjupa dig i ämnena:

- Responsible AI Dashboard: En helhetslösning för att operationalisera RAI i praktiken av Besmira Nushi och Mehrnoosh Sameki

[![Responsible AI Dashboard: En helhetslösning för att operationalisera RAI i praktiken](https://img.youtube.com/vi/f1oaDNl3djg/0.jpg)](https://www.youtube.com/watch?v=f1oaDNl3djg "Responsible AI Dashboard: En helhetslösning för att operationalisera RAI i praktiken")


> 🎥 Klicka på bilden ovan för en video: Responsible AI Dashboard: En helhetslösning för att operationalisera RAI i praktiken av Besmira Nushi och Mehrnoosh Sameki

Referera till följande material för att lära dig mer om ansvarsfull AI och hur man bygger mer pålitliga modeller:

- Microsofts RAI-dashboardverktyg för att felsöka ML-modeller: [Resurser för Responsible AI-verktyg](https://aka.ms/rai-dashboard)

- Utforska Responsible AI-verktygslådan: [Github](https://github.com/microsoft/responsible-ai-toolbox)

- Microsofts RAI-resurscenter: [Resurser för Responsible AI – Microsoft AI](https://www.microsoft.com/en-us/ai/tools-practices)

- Microsofts FATE-forskningsgrupp: [FATE: Rättvisa, Ansvar, Transparens och Etik inom AI - Microsoft Research](https://www.microsoft.com/research/theme/fate/)

## Uppgift

[Utforska RAI-dashboarden](assignment.md)

---

**Ansvarsfriskrivning**:  
Detta dokument har översatts med hjälp av AI-översättningstjänsten [Co-op Translator](https://github.com/Azure/co-op-translator). Även om vi strävar efter noggrannhet, vänligen notera att automatiska översättningar kan innehålla fel eller felaktigheter. Det ursprungliga dokumentet på sitt originalspråk bör betraktas som den auktoritativa källan. För kritisk information rekommenderas professionell mänsklig översättning. Vi ansvarar inte för eventuella missförstånd eller feltolkningar som uppstår vid användning av denna översättning.