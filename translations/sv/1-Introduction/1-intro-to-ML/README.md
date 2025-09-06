<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "69389392fa6346e0dfa30f664b7b6fec",
  "translation_date": "2025-09-05T21:42:42+00:00",
  "source_file": "1-Introduction/1-intro-to-ML/README.md",
  "language_code": "sv"
}
-->
# Introduktion till maskininlärning

## [Quiz före föreläsning](https://ff-quizzes.netlify.app/en/ml/)

---

[![ML för nybörjare - Introduktion till maskininlärning för nybörjare](https://img.youtube.com/vi/6mSx_KJxcHI/0.jpg)](https://youtu.be/6mSx_KJxcHI "ML för nybörjare - Introduktion till maskininlärning för nybörjare")

> 🎥 Klicka på bilden ovan för en kort video som går igenom denna lektion.

Välkommen till denna kurs om klassisk maskininlärning för nybörjare! Oavsett om du är helt ny inom detta ämne eller en erfaren ML-praktiker som vill fräscha upp dina kunskaper, är vi glada att ha dig med! Vi vill skapa en vänlig startpunkt för dina studier i maskininlärning och välkomnar gärna din [feedback](https://github.com/microsoft/ML-For-Beginners/discussions).

[![Introduktion till ML](https://img.youtube.com/vi/h0e2HAPTGF4/0.jpg)](https://youtu.be/h0e2HAPTGF4 "Introduktion till ML")

> 🎥 Klicka på bilden ovan för en video: MIT:s John Guttag introducerar maskininlärning

---
## Komma igång med maskininlärning

Innan du börjar med detta kursmaterial behöver du ha din dator konfigurerad och redo att köra notebooks lokalt.

- **Konfigurera din dator med dessa videor**. Använd följande länkar för att lära dig [hur du installerar Python](https://youtu.be/CXZYvNRIAKM) på ditt system och [ställer in en textredigerare](https://youtu.be/EU8eayHWoZg) för utveckling.
- **Lär dig Python**. Det rekommenderas också att ha en grundläggande förståelse för [Python](https://docs.microsoft.com/learn/paths/python-language/?WT.mc_id=academic-77952-leestott), ett programmeringsspråk som är användbart för dataforskare och som vi använder i denna kurs.
- **Lär dig Node.js och JavaScript**. Vi använder också JavaScript några gånger i denna kurs när vi bygger webbappar, så du behöver ha [node](https://nodejs.org) och [npm](https://www.npmjs.com/) installerade, samt [Visual Studio Code](https://code.visualstudio.com/) tillgängligt för både Python- och JavaScript-utveckling.
- **Skapa ett GitHub-konto**. Eftersom du hittade oss här på [GitHub](https://github.com), har du kanske redan ett konto, men om inte, skapa ett och fork:a sedan detta kursmaterial för att använda det själv. (Ge oss gärna en stjärna också 😊)
- **Utforska Scikit-learn**. Bekanta dig med [Scikit-learn](https://scikit-learn.org/stable/user_guide.html), en uppsättning ML-bibliotek som vi refererar till i dessa lektioner.

---
## Vad är maskininlärning?

Begreppet 'maskininlärning' är ett av de mest populära och frekvent använda termerna idag. Det är inte osannolikt att du har hört detta begrepp åtminstone en gång om du har någon form av bekantskap med teknik, oavsett vilket område du arbetar inom. Mekaniken bakom maskininlärning är dock ett mysterium för de flesta. För en nybörjare inom maskininlärning kan ämnet ibland kännas överväldigande. Därför är det viktigt att förstå vad maskininlärning faktiskt är och att lära sig om det steg för steg, genom praktiska exempel.

---
## Hypekurvan

![ml hype curve](../../../../1-Introduction/1-intro-to-ML/images/hype.png)

> Google Trends visar den senaste 'hypekurvan' för termen 'maskininlärning'

---
## Ett mystiskt universum

Vi lever i ett universum fullt av fascinerande mysterier. Stora vetenskapsmän som Stephen Hawking, Albert Einstein och många fler har ägnat sina liv åt att söka meningsfull information som avslöjar mysterierna i världen omkring oss. Detta är människans lärandevillkor: ett barn lär sig nya saker och upptäcker strukturen i sin värld år för år när det växer upp.

---
## Barnets hjärna

Ett barns hjärna och sinnen uppfattar fakta från sin omgivning och lär sig gradvis de dolda mönstren i livet, vilket hjälper barnet att skapa logiska regler för att identifiera inlärda mönster. Den mänskliga hjärnans inlärningsprocess gör människor till världens mest sofistikerade levande varelse. Att kontinuerligt lära sig genom att upptäcka dolda mönster och sedan innovera på dessa mönster gör att vi kan bli bättre och bättre under hela vår livstid. Denna inlärningsförmåga och utvecklingskapacitet är relaterad till ett koncept som kallas [hjärnplasticitet](https://www.simplypsychology.org/brain-plasticity.html). Ytligt sett kan vi dra vissa motiverande likheter mellan den mänskliga hjärnans inlärningsprocess och koncepten inom maskininlärning.

---
## Den mänskliga hjärnan

Den [mänskliga hjärnan](https://www.livescience.com/29365-human-brain.html) uppfattar saker från den verkliga världen, bearbetar den uppfattade informationen, fattar rationella beslut och utför vissa handlingar baserat på omständigheterna. Detta är vad vi kallar att bete sig intelligent. När vi programmerar en kopia av den intelligenta beteendeprocessen till en maskin kallas det artificiell intelligens (AI).

---
## Några termer

Även om termerna kan förväxlas är maskininlärning (ML) en viktig delmängd av artificiell intelligens. **ML handlar om att använda specialiserade algoritmer för att upptäcka meningsfull information och hitta dolda mönster från uppfattad data för att stödja den rationella beslutsprocessen**.

---
## AI, ML, djupinlärning

![AI, ML, deep learning, data science](../../../../1-Introduction/1-intro-to-ML/images/ai-ml-ds.png)

> En diagram som visar relationerna mellan AI, ML, djupinlärning och data science. Infografik av [Jen Looper](https://twitter.com/jenlooper) inspirerad av [denna grafik](https://softwareengineering.stackexchange.com/questions/366996/distinction-between-ai-ml-neural-networks-deep-learning-and-data-mining)

---
## Koncept att täcka

I detta kursmaterial kommer vi att täcka endast kärnkoncepten inom maskininlärning som en nybörjare måste känna till. Vi täcker det vi kallar 'klassisk maskininlärning' främst med hjälp av Scikit-learn, ett utmärkt bibliotek som många studenter använder för att lära sig grunderna. För att förstå bredare koncept inom artificiell intelligens eller djupinlärning är en stark grundläggande kunskap om maskininlärning oumbärlig, och därför vill vi erbjuda det här.

---
## I denna kurs kommer du att lära dig:

- kärnkoncept inom maskininlärning
- maskininlärningens historia
- ML och rättvisa
- regressionstekniker inom ML
- klassificeringstekniker inom ML
- klustringstekniker inom ML
- tekniker för naturlig språkbehandling inom ML
- tekniker för tidsserieprognoser inom ML
- förstärkningsinlärning
- verkliga tillämpningar av ML

---
## Vad vi inte kommer att täcka

- djupinlärning
- neurala nätverk
- AI

För att skapa en bättre inlärningsupplevelse kommer vi att undvika komplexiteten i neurala nätverk, 'djupinlärning' - flerskiktad modellbyggnad med neurala nätverk - och AI, som vi kommer att diskutera i ett annat kursmaterial. Vi kommer också att erbjuda ett kommande kursmaterial om data science för att fokusera på den aspekten av detta större område.

---
## Varför studera maskininlärning?

Maskininlärning, ur ett systemperspektiv, definieras som skapandet av automatiserade system som kan lära sig dolda mönster från data för att hjälpa till att fatta intelligenta beslut.

Denna motivation är löst inspirerad av hur den mänskliga hjärnan lär sig vissa saker baserat på data den uppfattar från omvärlden.

✅ Fundera en stund på varför ett företag skulle vilja använda strategier för maskininlärning istället för att skapa en hårdkodad regelbaserad motor.

---
## Tillämpningar av maskininlärning

Tillämpningar av maskininlärning finns nu nästan överallt och är lika allestädes närvarande som den data som flödar runt i våra samhällen, genererad av våra smartphones, uppkopplade enheter och andra system. Med tanke på den enorma potentialen hos moderna maskininlärningsalgoritmer har forskare utforskat deras förmåga att lösa multidimensionella och tvärvetenskapliga verkliga problem med stora positiva resultat.

---
## Exempel på tillämpad ML

**Du kan använda maskininlärning på många sätt**:

- För att förutsäga sannolikheten för sjukdom utifrån en patients medicinska historia eller rapporter.
- För att använda väderdata för att förutsäga väderhändelser.
- För att förstå känslan i en text.
- För att upptäcka falska nyheter och stoppa spridningen av propaganda.

Finans, ekonomi, geovetenskap, rymdforskning, biomedicinsk teknik, kognitiv vetenskap och till och med humaniora har anpassat maskininlärning för att lösa de arbetskrävande, databehandlingsintensiva problemen inom sina områden.

---
## Slutsats

Maskininlärning automatiserar processen att upptäcka mönster genom att hitta meningsfulla insikter från verklig eller genererad data. Det har visat sig vara mycket värdefullt inom affärs-, hälso- och finansiella tillämpningar, bland andra.

I en nära framtid kommer förståelsen av grunderna i maskininlärning att bli ett måste för människor inom alla områden på grund av dess utbredda användning.

---
# 🚀 Utmaning

Skissa, på papper eller med hjälp av en onlineapp som [Excalidraw](https://excalidraw.com/), din förståelse av skillnaderna mellan AI, ML, djupinlärning och data science. Lägg till några idéer om problem som var och en av dessa tekniker är bra på att lösa.

# [Quiz efter föreläsning](https://ff-quizzes.netlify.app/en/ml/)

---
# Granskning & Självstudier

För att lära dig mer om hur du kan arbeta med ML-algoritmer i molnet, följ denna [Lärväg](https://docs.microsoft.com/learn/paths/create-no-code-predictive-models-azure-machine-learning/?WT.mc_id=academic-77952-leestott).

Ta en [Lärväg](https://docs.microsoft.com/learn/modules/introduction-to-machine-learning/?WT.mc_id=academic-77952-leestott) om grunderna i ML.

---
# Uppgift

[Kom igång](assignment.md)

---

**Ansvarsfriskrivning**:  
Detta dokument har översatts med hjälp av AI-översättningstjänsten [Co-op Translator](https://github.com/Azure/co-op-translator). Även om vi strävar efter noggrannhet, vänligen notera att automatiska översättningar kan innehålla fel eller felaktigheter. Det ursprungliga dokumentet på sitt ursprungliga språk bör betraktas som den auktoritativa källan. För kritisk information rekommenderas professionell mänsklig översättning. Vi ansvarar inte för eventuella missförstånd eller feltolkningar som uppstår vid användning av denna översättning.