<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "6a05fec147e734c3e6bfa54505648e2b",
  "translation_date": "2025-09-05T21:44:44+00:00",
  "source_file": "1-Introduction/2-history-of-ML/README.md",
  "language_code": "sv"
}
-->
# Historien om maskininlärning

![Sammanfattning av historien om maskininlärning i en sketchnote](../../../../sketchnotes/ml-history.png)
> Sketchnote av [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Quiz före föreläsningen](https://ff-quizzes.netlify.app/en/ml/)

---

[![ML för nybörjare - Historien om maskininlärning](https://img.youtube.com/vi/N6wxM4wZ7V0/0.jpg)](https://youtu.be/N6wxM4wZ7V0 "ML för nybörjare - Historien om maskininlärning")

> 🎥 Klicka på bilden ovan för en kort video som går igenom denna lektion.

I denna lektion kommer vi att gå igenom de viktigaste milstolparna i historien om maskininlärning och artificiell intelligens.

Historien om artificiell intelligens (AI) som forskningsområde är nära sammanflätad med historien om maskininlärning, eftersom de algoritmer och tekniska framsteg som ligger till grund för ML bidrog till utvecklingen av AI. Det är bra att komma ihåg att även om dessa områden som separata forskningsfält började ta form på 1950-talet, föregicks och överlappades denna era av viktiga [algoritmiska, statistiska, matematiska, beräkningsmässiga och tekniska upptäckter](https://wikipedia.org/wiki/Timeline_of_machine_learning). Faktum är att människor har funderat på dessa frågor i [hundratals år](https://wikipedia.org/wiki/History_of_artificial_intelligence): denna artikel diskuterar de historiska intellektuella grunderna för idén om en "tänkande maskin."

---
## Viktiga upptäckter

- 1763, 1812 [Bayes sats](https://wikipedia.org/wiki/Bayes%27_theorem) och dess föregångare. Denna sats och dess tillämpningar ligger till grund för inferens och beskriver sannolikheten för att en händelse inträffar baserat på tidigare kunskap.
- 1805 [Minsta kvadratmetoden](https://wikipedia.org/wiki/Least_squares) av den franske matematikern Adrien-Marie Legendre. Denna teori, som du kommer att lära dig om i vår Regression-enhet, hjälper till med dataanpassning.
- 1913 [Markovkedjor](https://wikipedia.org/wiki/Markov_chain), uppkallad efter den ryske matematikern Andrey Markov, används för att beskriva en sekvens av möjliga händelser baserat på ett tidigare tillstånd.
- 1957 [Perceptron](https://wikipedia.org/wiki/Perceptron) är en typ av linjär klassificerare uppfunnen av den amerikanske psykologen Frank Rosenblatt som ligger till grund för framsteg inom djupinlärning.

---

- 1967 [Närmaste granne](https://wikipedia.org/wiki/Nearest_neighbor) är en algoritm som ursprungligen designades för att kartlägga rutter. Inom ML används den för att upptäcka mönster.
- 1970 [Backpropagation](https://wikipedia.org/wiki/Backpropagation) används för att träna [feedforward-nätverk](https://wikipedia.org/wiki/Feedforward_neural_network).
- 1982 [Recurrent Neural Networks](https://wikipedia.org/wiki/Recurrent_neural_network) är artificiella neurala nätverk som härstammar från feedforward-nätverk och skapar temporala grafer.

✅ Gör lite research. Vilka andra datum sticker ut som avgörande i historien om ML och AI?

---
## 1950: Maskiner som tänker

Alan Turing, en verkligen enastående person som [av allmänheten 2019](https://wikipedia.org/wiki/Icons:_The_Greatest_Person_of_the_20th_Century) röstades fram som 1900-talets största vetenskapsman, anses ha hjälpt till att lägga grunden för konceptet "en maskin som kan tänka." Han brottades med skeptiker och sitt eget behov av empiriska bevis för detta koncept, bland annat genom att skapa [Turingtestet](https://www.bbc.com/news/technology-18475646), som du kommer att utforska i våra NLP-lektioner.

---
## 1956: Dartmouth Summer Research Project

"Dartmouth Summer Research Project on artificial intelligence var en avgörande händelse för artificiell intelligens som forskningsområde," och det var här termen "artificiell intelligens" myntades ([källa](https://250.dartmouth.edu/highlights/artificial-intelligence-ai-coined-dartmouth)).

> Varje aspekt av lärande eller någon annan egenskap hos intelligens kan i princip beskrivas så exakt att en maskin kan göras för att simulera den.

---

Huvudforskaren, matematikprofessorn John McCarthy, hoppades "att gå vidare på grundval av hypotesen att varje aspekt av lärande eller någon annan egenskap hos intelligens i princip kan beskrivas så exakt att en maskin kan göras för att simulera den." Deltagarna inkluderade en annan framstående person inom området, Marvin Minsky.

Workshoppen anses ha initierat och uppmuntrat flera diskussioner, inklusive "framväxten av symboliska metoder, system fokuserade på begränsade domäner (tidiga expertsystem) och deduktiva system kontra induktiva system." ([källa](https://wikipedia.org/wiki/Dartmouth_workshop)).

---
## 1956 - 1974: "De gyllene åren"

Från 1950-talet till mitten av 70-talet var optimismen hög kring hoppet att AI kunde lösa många problem. 1967 uttalade Marvin Minsky självsäkert att "Inom en generation ... kommer problemet med att skapa 'artificiell intelligens' i stort sett att vara löst." (Minsky, Marvin (1967), Computation: Finite and Infinite Machines, Englewood Cliffs, N.J.: Prentice-Hall)

Forskning inom naturlig språkbehandling blomstrade, sökning förfinades och blev mer kraftfull, och konceptet "mikrovärldar" skapades, där enkla uppgifter utfördes med hjälp av instruktioner på vanligt språk.

---

Forskningen finansierades väl av statliga organ, framsteg gjordes inom beräkning och algoritmer, och prototyper av intelligenta maskiner byggdes. Några av dessa maskiner inkluderar:

* [Shakey the robot](https://wikipedia.org/wiki/Shakey_the_robot), som kunde manövrera och besluta hur uppgifter skulle utföras "intelligent".

    ![Shakey, en intelligent robot](../../../../1-Introduction/2-history-of-ML/images/shakey.jpg)
    > Shakey år 1972

---

* Eliza, en tidig "chatterbot", kunde samtala med människor och fungera som en primitiv "terapeut". Du kommer att lära dig mer om Eliza i NLP-lektionerna.

    ![Eliza, en bot](../../../../1-Introduction/2-history-of-ML/images/eliza.png)
    > En version av Eliza, en chatbot

---

* "Blocks world" var ett exempel på en mikrovärld där block kunde staplas och sorteras, och experiment i att lära maskiner att fatta beslut kunde testas. Framsteg byggda med bibliotek som [SHRDLU](https://wikipedia.org/wiki/SHRDLU) hjälpte till att driva språkbehandling framåt.

    [![blocks world med SHRDLU](https://img.youtube.com/vi/QAJz4YKUwqw/0.jpg)](https://www.youtube.com/watch?v=QAJz4YKUwqw "blocks world med SHRDLU")

    > 🎥 Klicka på bilden ovan för en video: Blocks world med SHRDLU

---
## 1974 - 1980: "AI-vintern"

I mitten av 1970-talet blev det uppenbart att komplexiteten i att skapa "intelligenta maskiner" hade underskattats och att dess löften, givet den tillgängliga beräkningskraften, hade överdrivits. Finansieringen torkade upp och förtroendet för området minskade. Några problem som påverkade förtroendet inkluderade:
---
- **Begränsningar**. Beräkningskraften var för begränsad.
- **Kombinatorisk explosion**. Antalet parametrar som behövde tränas växte exponentiellt när mer krävdes av datorer, utan en parallell utveckling av beräkningskraft och kapacitet.
- **Brist på data**. Det fanns en brist på data som hindrade processen att testa, utveckla och förfina algoritmer.
- **Ställer vi rätt frågor?**. Själva frågorna som ställdes började ifrågasättas. Forskare började möta kritik kring sina tillvägagångssätt:
  - Turingtestet ifrågasattes bland annat genom teorin om "den kinesiska rummet", som hävdade att "programmering av en digital dator kan få den att verka förstå språk men kan inte producera verklig förståelse." ([källa](https://plato.stanford.edu/entries/chinese-room/))
  - Etiken kring att introducera artificiella intelligenser som "terapeuten" ELIZA i samhället utmanades.

---

Samtidigt började olika AI-skolor bildas. En dikotomi etablerades mellan ["scruffy" vs. "neat AI"](https://wikipedia.org/wiki/Neats_and_scruffies) metoder. _Scruffy_-labb justerade program i timmar tills de fick önskade resultat. _Neat_-labb "fokuserade på logik och formell problemlösning". ELIZA och SHRDLU var välkända _scruffy_-system. På 1980-talet, när efterfrågan på att göra ML-system reproducerbara ökade, tog _neat_-metoden gradvis ledningen eftersom dess resultat är mer förklarbara.

---
## 1980-talets expertsystem

När området växte blev dess nytta för företag tydligare, och på 1980-talet ökade också spridningen av "expertsystem". "Expertsystem var bland de första verkligt framgångsrika formerna av artificiell intelligens (AI)-programvara." ([källa](https://wikipedia.org/wiki/Expert_system)).

Denna typ av system är faktiskt _hybrid_, bestående delvis av en regelmotor som definierar affärskrav och en inferensmotor som utnyttjar regelsystemet för att härleda nya fakta.

Denna era såg också ökat fokus på neurala nätverk.

---
## 1987 - 1993: AI "kyla"

Spridningen av specialiserad hårdvara för expertsystem hade den olyckliga effekten att bli för specialiserad. Framväxten av persondatorer konkurrerade också med dessa stora, specialiserade, centraliserade system. Demokratiseringen av databehandling hade börjat, och den banade så småningom väg för den moderna explosionen av big data.

---
## 1993 - 2011

Denna epok såg en ny era för ML och AI att kunna lösa några av de problem som tidigare orsakats av bristen på data och beräkningskraft. Mängden data började snabbt öka och bli mer tillgänglig, på gott och ont, särskilt med framväxten av smarttelefonen runt 2007. Beräkningskraften expanderade exponentiellt, och algoritmer utvecklades parallellt. Fältet började mogna när de fria dagarna från det förflutna började kristalliseras till en verklig disciplin.

---
## Nu

Idag berör maskininlärning och AI nästan alla delar av våra liv. Denna era kräver noggrann förståelse för riskerna och de potentiella effekterna av dessa algoritmer på människors liv. Som Microsofts Brad Smith har sagt, "Informationsteknologi väcker frågor som går till kärnan av grundläggande mänskliga rättigheter som integritet och yttrandefrihet. Dessa frågor ökar ansvaret för teknikföretag som skapar dessa produkter. Enligt vår uppfattning kräver de också genomtänkt statlig reglering och utveckling av normer kring acceptabla användningar" ([källa](https://www.technologyreview.com/2019/12/18/102365/the-future-of-ais-impact-on-society/)).

---

Det återstår att se vad framtiden har att erbjuda, men det är viktigt att förstå dessa datorsystem och den programvara och de algoritmer de kör. Vi hoppas att denna kursplan hjälper dig att få en bättre förståelse så att du kan bestämma själv.

[![Historien om djupinlärning](https://img.youtube.com/vi/mTtDfKgLm54/0.jpg)](https://www.youtube.com/watch?v=mTtDfKgLm54 "Historien om djupinlärning")
> 🎥 Klicka på bilden ovan för en video: Yann LeCun diskuterar historien om djupinlärning i denna föreläsning

---
## 🚀Utmaning

Gräv djupare i ett av dessa historiska ögonblick och lär dig mer om personerna bakom dem. Det finns fascinerande karaktärer, och ingen vetenskaplig upptäckt har någonsin skapats i ett kulturellt vakuum. Vad upptäcker du?

## [Quiz efter föreläsningen](https://ff-quizzes.netlify.app/en/ml/)

---
## Granskning & Självstudier

Här är saker att titta på och lyssna på:

[Denna podcast där Amy Boyd diskuterar AI:s utveckling](http://runasradio.com/Shows/Show/739)

[![Historien om AI av Amy Boyd](https://img.youtube.com/vi/EJt3_bFYKss/0.jpg)](https://www.youtube.com/watch?v=EJt3_bFYKss "Historien om AI av Amy Boyd")

---

## Uppgift

[Skapa en tidslinje](assignment.md)

---

**Ansvarsfriskrivning**:  
Detta dokument har översatts med hjälp av AI-översättningstjänsten [Co-op Translator](https://github.com/Azure/co-op-translator). Även om vi strävar efter noggrannhet, bör det noteras att automatiska översättningar kan innehålla fel eller felaktigheter. Det ursprungliga dokumentet på dess originalspråk bör betraktas som den auktoritativa källan. För kritisk information rekommenderas professionell mänsklig översättning. Vi ansvarar inte för eventuella missförstånd eller feltolkningar som uppstår vid användning av denna översättning.