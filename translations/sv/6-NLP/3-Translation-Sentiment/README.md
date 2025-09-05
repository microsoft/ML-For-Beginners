<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "be03c8182982b87ced155e4e9d1438e8",
  "translation_date": "2025-09-05T22:24:00+00:00",
  "source_file": "6-NLP/3-Translation-Sentiment/README.md",
  "language_code": "sv"
}
-->
# Översättning och sentimentanalys med ML

I de tidigare lektionerna lärde du dig att bygga en grundläggande bot med hjälp av `TextBlob`, ett bibliotek som använder maskininlärning bakom kulisserna för att utföra grundläggande NLP-uppgifter som att extrahera substantivfraser. En annan viktig utmaning inom datalingvistik är att korrekt _översätta_ en mening från ett talat eller skrivet språk till ett annat.

## [Förtest](https://ff-quizzes.netlify.app/en/ml/)

Översättning är ett mycket svårt problem, förvärrat av det faktum att det finns tusentals språk, alla med mycket olika grammatiska regler. En metod är att omvandla de formella grammatiska reglerna för ett språk, som engelska, till en språkoberoende struktur och sedan översätta tillbaka till ett annat språk. Denna metod innebär att du skulle ta följande steg:

1. **Identifiering**. Identifiera eller tagga orden i ingångsspråket som substantiv, verb etc.
2. **Skapa översättning**. Producera en direkt översättning av varje ord i målformatet för det andra språket.

### Exempelfras, engelska till iriska

På 'engelska' är meningen _I feel happy_ tre ord i ordningen:

- **subjekt** (I)
- **verb** (feel)
- **adjektiv** (happy)

Men på 'iriska' har samma mening en mycket annorlunda grammatisk struktur – känslor som "*happy*" eller "*sad*" uttrycks som att de är *på* dig.

Den engelska frasen `I feel happy` skulle på iriska vara `Tá athas orm`. En *bokstavlig* översättning skulle vara `Happy is upon me`.

En irisktalande som översätter till engelska skulle säga `I feel happy`, inte `Happy is upon me`, eftersom de förstår meningen med meningen, även om orden och meningsstrukturen är olika.

Den formella ordningen för meningen på iriska är:

- **verb** (Tá eller is)
- **adjektiv** (athas, eller happy)
- **subjekt** (orm, eller upon me)

## Översättning

Ett naivt översättningsprogram kanske bara översätter ord och ignorerar meningsstrukturen.

✅ Om du har lärt dig ett andra (eller tredje eller fler) språk som vuxen, kanske du började med att tänka på ditt modersmål, översätta ett koncept ord för ord i huvudet till det andra språket och sedan säga din översättning högt. Detta liknar vad naiva översättningsprogram för datorer gör. Det är viktigt att komma förbi denna fas för att uppnå flyt!

Naiv översättning leder till dåliga (och ibland roliga) felöversättningar: `I feel happy` översätts bokstavligen till `Mise bhraitheann athas` på iriska. Det betyder (bokstavligen) `me feel happy` och är inte en giltig irisk mening. Även om engelska och iriska är språk som talas på två närliggande öar, är de mycket olika språk med olika grammatiska strukturer.

> Du kan titta på några videor om iriska språktraditioner, som [den här](https://www.youtube.com/watch?v=mRIaLSdRMMs)

### Maskininlärningsmetoder

Hittills har du lärt dig om den formella regeltillvägagångssättet för naturlig språkbehandling. Ett annat tillvägagångssätt är att ignorera ordens betydelse och _istället använda maskininlärning för att upptäcka mönster_. Detta kan fungera vid översättning om du har mycket text (en *korpus*) eller texter (*korpora*) på både ursprungs- och målspråket.

Till exempel, tänk på fallet med *Pride and Prejudice*, en välkänd engelsk roman skriven av Jane Austen 1813. Om du konsulterar boken på engelska och en mänsklig översättning av boken till *franska*, kan du upptäcka fraser i den ena som är _idiomatiskt_ översatta till den andra. Det ska du göra om en stund.

Till exempel, när en engelsk fras som `I have no money` översätts bokstavligen till franska, kan det bli `Je n'ai pas de monnaie`. "Monnaie" är en knepig fransk 'falsk vän', eftersom 'money' och 'monnaie' inte är synonymer. En bättre översättning som en människa skulle kunna göra är `Je n'ai pas d'argent`, eftersom det bättre förmedlar betydelsen att du inte har några pengar (snarare än 'växel' som är betydelsen av 'monnaie').

![monnaie](../../../../6-NLP/3-Translation-Sentiment/images/monnaie.png)

> Bild av [Jen Looper](https://twitter.com/jenlooper)

Om en ML-modell har tillräckligt många mänskliga översättningar att bygga en modell på, kan den förbättra översättningens noggrannhet genom att identifiera vanliga mönster i texter som tidigare har översatts av experttalare av båda språken.

### Övning - översättning

Du kan använda `TextBlob` för att översätta meningar. Prova den berömda första raden i **Pride and Prejudice**:

```python
from textblob import TextBlob

blob = TextBlob(
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife!"
)
print(blob.translate(to="fr"))

```

`TextBlob` gör ett ganska bra jobb med översättningen: "C'est une vérité universellement reconnue, qu'un homme célibataire en possession d'une bonne fortune doit avoir besoin d'une femme!". 

Det kan hävdas att TextBlobs översättning är mycket mer exakt än den franska översättningen från 1932 av V. Leconte och Ch. Pressoir:

"C'est une vérité universelle qu'un célibataire pourvu d'une belle fortune doit avoir envie de se marier, et, si peu que l'on sache de son sentiment à cet egard, lorsqu'il arrive dans une nouvelle résidence, cette idée est si bien fixée dans l'esprit de ses voisins qu'ils le considèrent sur-le-champ comme la propriété légitime de l'une ou l'autre de leurs filles."

I detta fall gör den ML-baserade översättningen ett bättre jobb än den mänskliga översättaren, som onödigt lägger ord i den ursprungliga författarens mun för 'tydlighet'.

> Vad händer här? Och varför är TextBlob så bra på översättning? Jo, bakom kulisserna använder den Google Translate, en sofistikerad AI som kan analysera miljontals fraser för att förutsäga de bästa strängarna för uppgiften. Det finns inget manuellt arbete här, och du behöver en internetanslutning för att använda `blob.translate`.

✅ Prova några fler meningar. Vilken är bättre, ML eller mänsklig översättning? I vilka fall?

## Sentimentanalys

Ett annat område där maskininlärning kan fungera mycket bra är sentimentanalys. Ett icke-ML-tillvägagångssätt för sentiment är att identifiera ord och fraser som är 'positiva' och 'negativa'. Sedan, givet en ny text, beräkna det totala värdet av de positiva, negativa och neutrala orden för att identifiera det övergripande sentimentet. 

Detta tillvägagångssätt är lätt att lura, som du kanske har sett i Marvin-uppgiften – meningen `Great, that was a wonderful waste of time, I'm glad we are lost on this dark road` är en sarkastisk, negativ mening, men den enkla algoritmen upptäcker 'great', 'wonderful', 'glad' som positiva och 'waste', 'lost' och 'dark' som negativa. Det övergripande sentimentet påverkas av dessa motstridiga ord.

✅ Stanna upp en stund och fundera på hur vi som människor förmedlar sarkasm. Tonfallet spelar en stor roll. Försök att säga frasen "Well, that film was awesome" på olika sätt för att upptäcka hur din röst förmedlar betydelse.

### ML-metoder

ML-tillvägagångssättet skulle vara att manuellt samla negativa och positiva textkroppar – tweets, filmrecensioner eller något annat där en människa har gett ett betyg *och* en skriftlig åsikt. Sedan kan NLP-tekniker tillämpas på åsikter och betyg, så att mönster framträder (t.ex. positiva filmrecensioner tenderar att innehålla frasen 'Oscar worthy' oftare än negativa filmrecensioner, eller positiva restaurangrecensioner säger 'gourmet' mycket oftare än 'disgusting').

> ⚖️ **Exempel**: Om du arbetade på en politikers kontor och det fanns en ny lag som diskuterades, kanske väljare skrev till kontoret med e-post som stödjer eller motsätter sig den nya lagen. Låt oss säga att du fick i uppdrag att läsa e-posten och sortera dem i två högar, *för* och *emot*. Om det fanns många e-postmeddelanden kanske du skulle känna dig överväldigad av att försöka läsa alla. Skulle det inte vara trevligt om en bot kunde läsa dem åt dig, förstå dem och berätta i vilken hög varje e-postmeddelande hörde hemma? 
> 
> Ett sätt att uppnå detta är att använda maskininlärning. Du skulle träna modellen med en del av *emot*-e-posten och en del av *för*-e-posten. Modellen skulle tendera att associera fraser och ord med emot-sidan och för-sidan, *men den skulle inte förstå något av innehållet*, bara att vissa ord och mönster var mer sannolika att dyka upp i en *emot*- eller en *för*-e-post. Du skulle kunna testa den med några e-postmeddelanden som du inte hade använt för att träna modellen och se om den kom till samma slutsats som du gjorde. Sedan, när du var nöjd med modellens noggrannhet, skulle du kunna bearbeta framtida e-postmeddelanden utan att behöva läsa varje enskilt.

✅ Låter denna process som processer du har använt i tidigare lektioner?

## Övning - sentimentala meningar

Sentiment mäts med en *polaritet* från -1 till 1, där -1 är det mest negativa sentimentet och 1 är det mest positiva. Sentiment mäts också med en 0 - 1-skala för objektivitet (0) och subjektivitet (1).

Titta igen på Jane Austens *Pride and Prejudice*. Texten finns här på [Project Gutenberg](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm). Exemplet nedan visar ett kort program som analyserar sentimentet i första och sista meningarna från boken och visar dess sentimentpolaritet och subjektivitets-/objektivitetsvärde.

Du bör använda `TextBlob`-biblioteket (beskrivet ovan) för att bestämma `sentiment` (du behöver inte skriva din egen sentimentkalkylator) i följande uppgift.

```python
from textblob import TextBlob

quote1 = """It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife."""

quote2 = """Darcy, as well as Elizabeth, really loved them; and they were both ever sensible of the warmest gratitude towards the persons who, by bringing her into Derbyshire, had been the means of uniting them."""

sentiment1 = TextBlob(quote1).sentiment
sentiment2 = TextBlob(quote2).sentiment

print(quote1 + " has a sentiment of " + str(sentiment1))
print(quote2 + " has a sentiment of " + str(sentiment2))
```

Du ser följande utdata:

```output
It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want # of a wife. has a sentiment of Sentiment(polarity=0.20952380952380953, subjectivity=0.27142857142857146)

Darcy, as well as Elizabeth, really loved them; and they were
     both ever sensible of the warmest gratitude towards the persons
      who, by bringing her into Derbyshire, had been the means of
      uniting them. has a sentiment of Sentiment(polarity=0.7, subjectivity=0.8)
```

## Utmaning - kontrollera sentimentpolaritet

Din uppgift är att avgöra, med hjälp av sentimentpolaritet, om *Pride and Prejudice* har fler absolut positiva meningar än absolut negativa. För denna uppgift kan du anta att ett polaritetsvärde på 1 eller -1 är absolut positivt eller negativt.

**Steg:**

1. Ladda ner en [kopia av Pride and Prejudice](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm) från Project Gutenberg som en .txt-fil. Ta bort metadata i början och slutet av filen, så att endast originaltexten återstår.
2. Öppna filen i Python och extrahera innehållet som en sträng.
3. Skapa en TextBlob med boksträngen.
4. Analysera varje mening i boken i en loop.
   1. Om polariteten är 1 eller -1, lagra meningen i en array eller lista med positiva eller negativa meddelanden.
5. I slutet, skriv ut alla positiva meningar och negativa meningar (separat) och antalet av varje.

Här är en [lösning](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/3-Translation-Sentiment/solution/notebook.ipynb).

✅ Kunskapskontroll

1. Sentimentet baseras på ord som används i meningen, men *förstår* koden orden?
2. Tycker du att sentimentpolariteten är korrekt, eller med andra ord, håller du *med* om poängen?
   1. Håller du i synnerhet med eller inte med den absoluta **positiva** polariteten i följande meningar?
      * “What an excellent father you have, girls!” said she, when the door was shut.
      * “Your examination of Mr. Darcy is over, I presume,” said Miss Bingley; “and pray what is the result?” “I am perfectly convinced by it that Mr. Darcy has no defect.
      * How wonderfully these sort of things occur!
      * I have the greatest dislike in the world to that sort of thing.
      * Charlotte is an excellent manager, I dare say.
      * “This is delightful indeed!
      * I am so happy!
      * Your idea of the ponies is delightful.
   2. De nästa tre meningarna fick en absolut positiv sentiment, men vid närmare granskning är de inte positiva meningar. Varför trodde sentimentanalysen att de var positiva meningar?
      * Happy shall I be, when his stay at Netherfield is over!” “I wish I could say anything to comfort you,” replied Elizabeth; “but it is wholly out of my power.
      * If I could but see you as happy!
      * Our distress, my dear Lizzy, is very great.
   3. Håller du med eller inte med den absoluta **negativa** polariteten i följande meningar?
      - Everybody is disgusted with his pride.
      - “I should like to know how he behaves among strangers.” “You shall hear then—but prepare yourself for something very dreadful.
      - The pause was to Elizabeth’s feelings dreadful.
      - It would be dreadful!

✅ Alla som är förtjusta i Jane Austen förstår att hon ofta använder sina böcker för att kritisera de mer löjliga aspekterna av det engelska regentsamhället. Elizabeth Bennett, huvudpersonen i *Pride and Prejudice*, är en skarp social observatör (som författaren) och hennes språk är ofta starkt nyanserat. Till och med Mr. Darcy (kärleksintresset i berättelsen) noterar Elizabeths lekfulla och retfulla användning av språket: "I have had the pleasure of your acquaintance long enough to know that you find great enjoyment in occasionally professing opinions which in fact are not your own."

---

## 🚀Utmaning

Kan du göra Marvin ännu bättre genom att extrahera andra funktioner från användarinmatningen?

## [Eftertest](https://ff-quizzes.netlify.app/en/ml/)

## Recension & Självstudier
Det finns många sätt att extrahera sentiment från text. Tänk på de affärsapplikationer som kan dra nytta av denna teknik. Fundera på hur det kan gå fel. Läs mer om sofistikerade företagsklara system som analyserar sentiment, såsom [Azure Text Analysis](https://docs.microsoft.com/azure/cognitive-services/Text-Analytics/how-tos/text-analytics-how-to-sentiment-analysis?tabs=version-3-1?WT.mc_id=academic-77952-leestott). Testa några av meningarna från Stolthet och fördom ovan och se om det kan upptäcka nyanser.

## Uppgift

[Poetisk frihet](assignment.md)

---

**Ansvarsfriskrivning**:  
Detta dokument har översatts med hjälp av AI-översättningstjänsten [Co-op Translator](https://github.com/Azure/co-op-translator). Även om vi strävar efter noggrannhet, vänligen notera att automatiska översättningar kan innehålla fel eller felaktigheter. Det ursprungliga dokumentet på sitt originalspråk bör betraktas som den auktoritativa källan. För kritisk information rekommenderas professionell mänsklig översättning. Vi ansvarar inte för eventuella missförstånd eller feltolkningar som uppstår vid användning av denna översättning.