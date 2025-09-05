<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "83320d6b6994909e35d830cebf214039",
  "translation_date": "2025-09-05T21:32:51+00:00",
  "source_file": "9-Real-World/1-Applications/README.md",
  "language_code": "sv"
}
-->
# Postscript: Maskininlärning i verkligheten

![Sammanfattning av maskininlärning i verkligheten i en sketchnote](../../../../sketchnotes/ml-realworld.png)
> Sketchnote av [Tomomi Imura](https://www.twitter.com/girlie_mac)

I den här kursen har du lärt dig många sätt att förbereda data för träning och skapa maskininlärningsmodeller. Du har byggt en serie klassiska modeller för regression, klustring, klassificering, naturlig språkbehandling och tidsserier. Grattis! Nu kanske du undrar vad allt detta ska användas till... vilka är de verkliga tillämpningarna för dessa modeller?

Även om AI, som ofta använder sig av djupinlärning, har fått mycket uppmärksamhet inom industrin, finns det fortfarande värdefulla tillämpningar för klassiska maskininlärningsmodeller. Du kanske till och med använder några av dessa tillämpningar idag! I den här lektionen kommer du att utforska hur åtta olika industrier och ämnesområden använder dessa typer av modeller för att göra sina applikationer mer effektiva, pålitliga, intelligenta och värdefulla för användarna.

## [Quiz före föreläsningen](https://ff-quizzes.netlify.app/en/ml/)

## 💰 Finans

Finanssektorn erbjuder många möjligheter för maskininlärning. Många problem inom detta område lämpar sig väl för att modelleras och lösas med hjälp av ML.

### Upptäckt av kreditkortsbedrägerier

Vi lärde oss om [k-means klustring](../../5-Clustering/2-K-Means/README.md) tidigare i kursen, men hur kan det användas för att lösa problem relaterade till kreditkortsbedrägerier?

K-means klustring är användbart vid en teknik för att upptäcka kreditkortsbedrägerier som kallas **outlier detection**. Avvikelser, eller avvikande observationer i en datamängd, kan indikera om ett kreditkort används normalt eller om något ovanligt pågår. Som visas i den länkade artikeln nedan kan du sortera kreditkortsdata med en k-means klustringsalgoritm och tilldela varje transaktion till en kluster baserat på hur mycket avvikande den verkar vara. Därefter kan du utvärdera de mest riskfyllda klustren för att avgöra om transaktionerna är bedrägliga eller legitima.
[Referens](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.680.1195&rep=rep1&type=pdf)

### Förmögenhetsförvaltning

Inom förmögenhetsförvaltning hanterar en individ eller firma investeringar för sina klienter. Deras uppgift är att långsiktigt bevara och öka förmögenheten, vilket gör det avgörande att välja investeringar som presterar väl.

Ett sätt att utvärdera hur en specifik investering presterar är genom statistisk regression. [Linjär regression](../../2-Regression/1-Tools/README.md) är ett värdefullt verktyg för att förstå hur en fond presterar i förhållande till en benchmark. Vi kan också avgöra om resultaten av regressionen är statistiskt signifikanta, eller hur mycket de skulle påverka en klients investeringar. Du kan till och med utöka din analys med multipel regression, där ytterligare riskfaktorer tas med i beräkningen. För ett exempel på hur detta skulle fungera för en specifik fond, kolla in artikeln nedan om att utvärdera fondprestanda med regression.
[Referens](http://www.brightwoodventures.com/evaluating-fund-performance-using-regression/)

## 🎓 Utbildning

Utbildningssektorn är också ett mycket intressant område där ML kan tillämpas. Det finns spännande problem att lösa, såsom att upptäcka fusk på prov eller uppsatser, eller hantera bias, medveten eller omedveten, i rättningsprocessen.

### Förutsäga studentbeteende

[Coursera](https://coursera.com), en leverantör av öppna onlinekurser, har en fantastisk teknisk blogg där de diskuterar många ingenjörsbeslut. I denna fallstudie plottade de en regressionslinje för att undersöka om det finns någon korrelation mellan ett lågt NPS (Net Promoter Score) och kursretention eller avhopp.
[Referens](https://medium.com/coursera-engineering/controlled-regression-quantifying-the-impact-of-course-quality-on-learner-retention-31f956bd592a)

### Minska bias

[Grammarly](https://grammarly.com), en skrivassistent som kontrollerar stavning och grammatikfel, använder sofistikerade [system för naturlig språkbehandling](../../6-NLP/README.md) i sina produkter. De publicerade en intressant fallstudie i sin tekniska blogg om hur de hanterade könsbias i maskininlärning, vilket du lärde dig om i vår [introduktionslektion om rättvisa](../../1-Introduction/3-fairness/README.md).
[Referens](https://www.grammarly.com/blog/engineering/mitigating-gender-bias-in-autocorrect/)

## 👜 Detaljhandel

Detaljhandelssektorn kan definitivt dra nytta av användningen av ML, med allt från att skapa en bättre kundresa till att optimera lagerhantering.

### Personalisera kundresan

På Wayfair, ett företag som säljer heminredning som möbler, är det avgörande att hjälpa kunder att hitta rätt produkter för deras smak och behov. I denna artikel beskriver ingenjörer från företaget hur de använder ML och NLP för att "visa rätt resultat för kunder". Deras Query Intent Engine har byggts för att använda entitetsutvinning, klassificeringsträning, utvinning av tillgångar och åsikter, samt sentimenttaggning på kundrecensioner. Detta är ett klassiskt användningsfall för hur NLP fungerar inom online-detaljhandel.
[Referens](https://www.aboutwayfair.com/tech-innovation/how-we-use-machine-learning-and-natural-language-processing-to-empower-search)

### Lagerhantering

Innovativa, flexibla företag som [StitchFix](https://stitchfix.com), en boxservice som skickar kläder till konsumenter, förlitar sig starkt på ML för rekommendationer och lagerhantering. Deras stylingteam samarbetar med deras merchandisingteam: "en av våra dataforskare experimenterade med en genetisk algoritm och tillämpade den på kläder för att förutsäga vad som skulle vara en framgångsrik klädesplagg som inte existerar idag. Vi presenterade detta för merchandisingteamet, och nu kan de använda det som ett verktyg."
[Referens](https://www.zdnet.com/article/how-stitch-fix-uses-machine-learning-to-master-the-science-of-styling/)

## 🏥 Hälso- och sjukvård

Hälso- och sjukvårdssektorn kan använda ML för att optimera forskningsuppgifter och logistiska problem som att återinlägga patienter eller stoppa spridningen av sjukdomar.

### Hantering av kliniska prövningar

Toxicitet i kliniska prövningar är en stor oro för läkemedelsföretag. Hur mycket toxicitet är tolerabelt? I denna studie ledde analysen av olika metoder för kliniska prövningar till utvecklingen av en ny metod för att förutsäga oddsen för kliniska prövningsresultat. Specifikt kunde de använda random forest för att skapa en [klassificerare](../../4-Classification/README.md) som kan skilja mellan grupper av läkemedel.
[Referens](https://www.sciencedirect.com/science/article/pii/S2451945616302914)

### Hantering av återinläggningar på sjukhus

Sjukhusvård är kostsam, särskilt när patienter måste återinläggas. Denna artikel diskuterar ett företag som använder ML för att förutsäga potentialen för återinläggning med hjälp av [klustringsalgoritmer](../../5-Clustering/README.md). Dessa kluster hjälper analytiker att "upptäcka grupper av återinläggningar som kan ha en gemensam orsak".
[Referens](https://healthmanagement.org/c/healthmanagement/issuearticle/hospital-readmissions-and-machine-learning)

### Hantering av sjukdomar

Den senaste pandemin har belyst hur maskininlärning kan hjälpa till att stoppa spridningen av sjukdomar. I denna artikel kommer du att känna igen användningen av ARIMA, logistiska kurvor, linjär regression och SARIMA. "Detta arbete är ett försök att beräkna spridningshastigheten för detta virus och därmed förutsäga dödsfall, återhämtningar och bekräftade fall, så att vi kan förbereda oss bättre och överleva."
[Referens](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7979218/)

## 🌲 Ekologi och grön teknik

Natur och ekologi består av många känsliga system där samspelet mellan djur och natur står i fokus. Det är viktigt att kunna mäta dessa system noggrant och agera lämpligt om något händer, som en skogsbrand eller en minskning av djurpopulationen.

### Skogsförvaltning

Du lärde dig om [Reinforcement Learning](../../8-Reinforcement/README.md) i tidigare lektioner. Det kan vara mycket användbart när man försöker förutsäga mönster i naturen. Särskilt kan det användas för att spåra ekologiska problem som skogsbränder och spridningen av invasiva arter. I Kanada använde en grupp forskare Reinforcement Learning för att bygga modeller för skogsbrandsdynamik från satellitbilder. Med hjälp av en innovativ "spatially spreading process (SSP)" föreställde de sig en skogsbrand som "agenten vid vilken cell som helst i landskapet." "Uppsättningen av åtgärder som branden kan vidta från en plats vid vilken tidpunkt som helst inkluderar att sprida sig norrut, söderut, österut eller västerut eller att inte sprida sig."

Denna metod inverterar den vanliga RL-uppsättningen eftersom dynamiken i den motsvarande Markov Decision Process (MDP) är en känd funktion för omedelbar spridning av skogsbranden." Läs mer om de klassiska algoritmer som användes av denna grupp på länken nedan.
[Referens](https://www.frontiersin.org/articles/10.3389/fict.2018.00006/full)

### Rörelsesensorer för djur

Även om djupinlärning har skapat en revolution i att visuellt spåra djurrörelser (du kan bygga din egen [isbjörnspårare](https://docs.microsoft.com/learn/modules/build-ml-model-with-azure-stream-analytics/?WT.mc_id=academic-77952-leestott) här), har klassisk ML fortfarande en plats i denna uppgift.

Sensorer för att spåra rörelser hos gårdsdjur och IoT använder denna typ av visuell bearbetning, men mer grundläggande ML-tekniker är användbara för att förbehandla data. Till exempel övervakades och analyserades fårs kroppshållningar med olika klassificeringsalgoritmer i denna artikel. Du kanske känner igen ROC-kurvan på sida 335.
[Referens](https://druckhaus-hofmann.de/gallery/31-wj-feb-2020.pdf)

### ⚡️ Energihantering

I våra lektioner om [tidsserieprognoser](../../7-TimeSeries/README.md) tog vi upp konceptet med smarta parkeringsmätare för att generera intäkter för en stad baserat på att förstå utbud och efterfrågan. Denna artikel diskuterar i detalj hur klustring, regression och tidsserieprognoser kombinerades för att hjälpa till att förutsäga framtida energianvändning i Irland, baserat på smarta mätare.
[Referens](https://www-cdn.knime.com/sites/default/files/inline-images/knime_bigdata_energy_timeseries_whitepaper.pdf)

## 💼 Försäkring

Försäkringssektorn är ytterligare en sektor som använder ML för att konstruera och optimera hållbara finansiella och aktuariska modeller.

### Hantering av volatilitet

MetLife, en leverantör av livförsäkringar, är öppen med hur de analyserar och minskar volatilitet i sina finansiella modeller. I denna artikel kommer du att märka visualiseringar av binär och ordinal klassificering. Du kommer också att upptäcka visualiseringar för prognoser.
[Referens](https://investments.metlife.com/content/dam/metlifecom/us/investments/insights/research-topics/macro-strategy/pdf/MetLifeInvestmentManagement_MachineLearnedRanking_070920.pdf)

## 🎨 Konst, kultur och litteratur

Inom konsten, till exempel journalistik, finns det många intressanta problem. Att upptäcka falska nyheter är ett stort problem eftersom det har visat sig påverka människors åsikter och till och med kunna störta demokratier. Museer kan också dra nytta av att använda ML i allt från att hitta länkar mellan artefakter till resursplanering.

### Upptäckt av falska nyheter

Att upptäcka falska nyheter har blivit en katt-och-råtta-lek i dagens media. I denna artikel föreslår forskare att ett system som kombinerar flera av de ML-tekniker vi har studerat kan testas och den bästa modellen implementeras: "Detta system är baserat på naturlig språkbehandling för att extrahera funktioner från data och sedan används dessa funktioner för att träna maskininlärningsklassificerare såsom Naive Bayes, Support Vector Machine (SVM), Random Forest (RF), Stochastic Gradient Descent (SGD) och Logistic Regression (LR)."
[Referens](https://www.irjet.net/archives/V7/i6/IRJET-V7I6688.pdf)

Denna artikel visar hur kombinationen av olika ML-områden kan producera intressanta resultat som kan hjälpa till att stoppa spridningen av falska nyheter och skapa verklig skada; i detta fall var drivkraften spridningen av rykten om COVID-behandlingar som ledde till våldsamma upplopp.

### Museum ML

Museer står på tröskeln till en AI-revolution där katalogisering och digitalisering av samlingar samt att hitta länkar mellan artefakter blir enklare i takt med att tekniken utvecklas. Projekt som [In Codice Ratio](https://www.sciencedirect.com/science/article/abs/pii/S0306457321001035#:~:text=1.,studies%20over%20large%20historical%20sources.) hjälper till att låsa upp mysterierna i otillgängliga samlingar som Vatikanens arkiv. Men den affärsmässiga aspekten av museer drar också nytta av ML-modeller.

Till exempel byggde Art Institute of Chicago modeller för att förutsäga vad publiken är intresserad av och när de kommer att besöka utställningar. Målet är att skapa individuella och optimerade besöksupplevelser varje gång användaren besöker museet. "Under räkenskapsåret 2017 förutsade modellen besöksantal och intäkter med en noggrannhet på 1 procent, säger Andrew Simnick, senior vice president på Art Institute."
[Referens](https://www.chicagobusiness.com/article/20180518/ISSUE01/180519840/art-institute-of-chicago-uses-data-to-make-exhibit-choices)

## 🏷 Marknadsföring

### Kundsegmentering

De mest effektiva marknadsföringsstrategierna riktar sig till kunder på olika sätt baserat på olika grupperingar. I denna artikel diskuteras användningen av klustringsalgoritmer för att stödja differentierad marknadsföring. Differentierad marknadsföring hjälper företag att förbättra varumärkesigenkänning, nå fler kunder och tjäna mer pengar.
[Referens](https://ai.inqline.com/machine-learning-for-marketing-customer-segmentation/)

## 🚀 Utmaning

Identifiera en annan sektor som drar nytta av några av de tekniker du lärt dig i denna kurs och upptäck hur den använder ML.
## [Quiz efter föreläsningen](https://ff-quizzes.netlify.app/en/ml/)

## Granskning & Självstudier

Wayfairs data science-team har flera intressanta videor om hur de använder ML på sitt företag. Det är värt att [kolla in dem](https://www.youtube.com/channel/UCe2PjkQXqOuwkW1gw6Ameuw/videos)!

## Uppgift

[En ML-skattjakt](assignment.md)

---

**Ansvarsfriskrivning**:  
Detta dokument har översatts med hjälp av AI-översättningstjänsten [Co-op Translator](https://github.com/Azure/co-op-translator). Även om vi strävar efter noggrannhet, vänligen notera att automatiska översättningar kan innehålla fel eller felaktigheter. Det ursprungliga dokumentet på dess originalspråk bör betraktas som den auktoritativa källan. För kritisk information rekommenderas professionell mänsklig översättning. Vi ansvarar inte för eventuella missförstånd eller feltolkningar som uppstår vid användning av denna översättning.