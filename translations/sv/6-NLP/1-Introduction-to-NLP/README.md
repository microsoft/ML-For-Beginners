<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1c2ec40cf55c98a028a359c27ef7e45a",
  "translation_date": "2025-09-05T22:21:40+00:00",
  "source_file": "6-NLP/1-Introduction-to-NLP/README.md",
  "language_code": "sv"
}
-->
# Introduktion till naturlig språkbehandling

Den här lektionen täcker en kort historik och viktiga koncept inom *naturlig språkbehandling*, ett delområde inom *datorlingvistik*.

## [Quiz före föreläsningen](https://ff-quizzes.netlify.app/en/ml/)

## Introduktion

NLP, som det ofta kallas, är ett av de mest kända områdena där maskininlärning har tillämpats och används i produktionsprogramvara.

✅ Kan du komma på programvara som du använder dagligen som troligen har någon NLP inbyggd? Vad sägs om dina ordbehandlingsprogram eller mobilappar som du använder regelbundet?

Du kommer att lära dig om:

- **Idén om språk**. Hur språk utvecklades och vilka de stora studieområdena har varit.
- **Definition och koncept**. Du kommer också att lära dig definitioner och koncept om hur datorer bearbetar text, inklusive parsning, grammatik och identifiering av substantiv och verb. Det finns några kodningsuppgifter i den här lektionen, och flera viktiga koncept introduceras som du kommer att lära dig att koda senare i de kommande lektionerna.

## Datorlingvistik

Datorlingvistik är ett forsknings- och utvecklingsområde som har pågått i många decennier och studerar hur datorer kan arbeta med, och till och med förstå, översätta och kommunicera med språk. Naturlig språkbehandling (NLP) är ett relaterat område som fokuserar på hur datorer kan bearbeta "naturliga", eller mänskliga, språk.

### Exempel - telefonens diktering

Om du någonsin har dikterat till din telefon istället för att skriva eller ställt en fråga till en virtuell assistent, har ditt tal konverterats till textform och sedan bearbetats eller *parsats* från det språk du talade. De upptäckta nyckelorden bearbetades sedan till ett format som telefonen eller assistenten kunde förstå och agera på.

![förståelse](../../../../6-NLP/1-Introduction-to-NLP/images/comprehension.png)
> Riktig språklig förståelse är svårt! Bild av [Jen Looper](https://twitter.com/jenlooper)

### Hur är denna teknik möjlig?

Detta är möjligt eftersom någon har skrivit ett datorprogram för att göra detta. För några decennier sedan förutspådde vissa science fiction-författare att människor mestadels skulle prata med sina datorer och att datorerna alltid skulle förstå exakt vad de menade. Tyvärr visade det sig vara ett svårare problem än många föreställde sig, och även om det är ett mycket bättre förstått problem idag, finns det betydande utmaningar med att uppnå "perfekt" naturlig språkbehandling när det gäller att förstå innebörden av en mening. Detta är särskilt svårt när det gäller att förstå humor eller att upptäcka känslor som sarkasm i en mening.

Vid det här laget kanske du minns skollektioner där läraren gick igenom grammatikens delar i en mening. I vissa länder lärs grammatik och lingvistik ut som ett dedikerat ämne, men i många ingår dessa ämnen som en del av att lära sig ett språk: antingen ditt första språk i grundskolan (att lära sig läsa och skriva) och kanske ett andra språk i högstadiet eller gymnasiet. Oroa dig inte om du inte är expert på att skilja substantiv från verb eller adverb från adjektiv!

Om du har svårt med skillnaden mellan *simple present* och *present progressive*, är du inte ensam. Detta är en utmaning för många människor, även modersmålstalare av ett språk. Den goda nyheten är att datorer är riktigt bra på att tillämpa formella regler, och du kommer att lära dig att skriva kod som kan *parsa* en mening lika bra som en människa. Den större utmaningen som du kommer att undersöka senare är att förstå *innebörden* och *känslan* av en mening.

## Förkunskaper

För den här lektionen är den huvudsakliga förkunskapen att kunna läsa och förstå språket i den här lektionen. Det finns inga matematiska problem eller ekvationer att lösa. Även om den ursprungliga författaren skrev den här lektionen på engelska, är den också översatt till andra språk, så du kan läsa en översättning. Det finns exempel där ett antal olika språk används (för att jämföra de olika grammatiska reglerna för olika språk). Dessa är *inte* översatta, men den förklarande texten är det, så innebörden bör vara tydlig.

För kodningsuppgifterna kommer du att använda Python och exemplen använder Python 3.8.

I det här avsnittet kommer du att behöva och använda:

- **Python 3 förståelse**. Förståelse för programmeringsspråket Python 3, den här lektionen använder input, loopar, filinläsning, arrayer.
- **Visual Studio Code + tillägg**. Vi kommer att använda Visual Studio Code och dess Python-tillägg. Du kan också använda en Python IDE som du föredrar.
- **TextBlob**. [TextBlob](https://github.com/sloria/TextBlob) är ett förenklat textbearbetningsbibliotek för Python. Följ instruktionerna på TextBlob-webbplatsen för att installera det på ditt system (installera även corpora, som visas nedan):

   ```bash
   pip install -U textblob
   python -m textblob.download_corpora
   ```

> 💡 Tips: Du kan köra Python direkt i VS Code-miljöer. Kolla [dokumentationen](https://code.visualstudio.com/docs/languages/python?WT.mc_id=academic-77952-leestott) för mer information.

## Att prata med maskiner

Historien om att försöka få datorer att förstå mänskligt språk går tillbaka flera decennier, och en av de tidigaste forskarna som funderade på naturlig språkbehandling var *Alan Turing*.

### 'Turingtestet'

När Turing forskade om *artificiell intelligens* på 1950-talet funderade han på om ett konversationstest kunde ges till en människa och en dator (via skriftlig korrespondens) där människan i konversationen inte var säker på om de samtalade med en annan människa eller en dator.

Om människan efter en viss längd av konversation inte kunde avgöra om svaren kom från en dator eller inte, kunde datorn då sägas *tänka*?

### Inspirationen - 'imitationsspelet'

Idén till detta kom från ett sällskapsspel som kallades *Imitationsspelet* där en förhörsledare är ensam i ett rum och har i uppgift att avgöra vilka av två personer (i ett annat rum) som är man respektive kvinna. Förhörsledaren kan skicka lappar och måste försöka komma på frågor där de skriftliga svaren avslöjar könet på den mystiska personen. Naturligtvis försöker spelarna i det andra rummet lura förhörsledaren genom att svara på frågor på ett sätt som vilseleder eller förvirrar förhörsledaren, samtidigt som de ger intryck av att svara ärligt.

### Utvecklingen av Eliza

På 1960-talet utvecklade en MIT-forskare vid namn *Joseph Weizenbaum* [*Eliza*](https://wikipedia.org/wiki/ELIZA), en dator "terapeut" som skulle ställa frågor till människan och ge intryck av att förstå deras svar. Men även om Eliza kunde parsa en mening och identifiera vissa grammatiska konstruktioner och nyckelord för att ge ett rimligt svar, kunde den inte sägas *förstå* meningen. Om Eliza presenterades med en mening som följde formatet "**Jag är** <u>ledsen</u>" kunde den omarrangera och ersätta ord i meningen för att bilda svaret "Hur länge har **du varit** <u>ledsen</u>". 

Detta gav intrycket att Eliza förstod uttalandet och ställde en följdfråga, medan den i verkligheten ändrade tempus och lade till några ord. Om Eliza inte kunde identifiera ett nyckelord som den hade ett svar för, skulle den istället ge ett slumpmässigt svar som kunde vara tillämpligt på många olika uttalanden. Eliza kunde lätt luras, till exempel om en användare skrev "**Du är** en <u>cykel</u>" kunde den svara med "Hur länge har **jag varit** en <u>cykel</u>?", istället för ett mer genomtänkt svar.

[![Prata med Eliza](https://img.youtube.com/vi/RMK9AphfLco/0.jpg)](https://youtu.be/RMK9AphfLco "Prata med Eliza")

> 🎥 Klicka på bilden ovan för en video om det ursprungliga ELIZA-programmet

> Obs: Du kan läsa den ursprungliga beskrivningen av [Eliza](https://cacm.acm.org/magazines/1966/1/13317-elizaa-computer-program-for-the-study-of-natural-language-communication-between-man-and-machine/abstract) publicerad 1966 om du har ett ACM-konto. Alternativt kan du läsa om Eliza på [wikipedia](https://wikipedia.org/wiki/ELIZA)

## Övning - koda en enkel konversationsbot

En konversationsbot, som Eliza, är ett program som tar emot användarinmatning och verkar förstå och svara intelligent. Till skillnad från Eliza kommer vår bot inte att ha flera regler som ger den intrycket av att ha en intelligent konversation. Istället kommer vår bot att ha en enda förmåga, att hålla konversationen igång med slumpmässiga svar som kan fungera i nästan vilken trivial konversation som helst.

### Planen

Dina steg när du bygger en konversationsbot:

1. Skriv ut instruktioner som informerar användaren om hur man interagerar med boten
2. Starta en loop
   1. Ta emot användarinmatning
   2. Om användaren har bett om att avsluta, avsluta
   3. Bearbeta användarinmatning och bestäm svar (i detta fall är svaret ett slumpmässigt val från en lista med möjliga generiska svar)
   4. Skriv ut svar
3. Gå tillbaka till steg 2

### Bygga boten

Låt oss skapa boten nu. Vi börjar med att definiera några fraser.

1. Skapa den här boten själv i Python med följande slumpmässiga svar:

    ```python
    random_responses = ["That is quite interesting, please tell me more.",
                        "I see. Do go on.",
                        "Why do you say that?",
                        "Funny weather we've been having, isn't it?",
                        "Let's change the subject.",
                        "Did you catch the game last night?"]
    ```

    Här är ett exempel på utdata för att vägleda dig (användarinmatning är på rader som börjar med `>`):

    ```output
    Hello, I am Marvin, the simple robot.
    You can end this conversation at any time by typing 'bye'
    After typing each answer, press 'enter'
    How are you today?
    > I am good thanks
    That is quite interesting, please tell me more.
    > today I went for a walk     
    Did you catch the game last night?
    > I did, but my team lost
    Funny weather we've been having, isn't it?
    > yes but I hope next week is better
    Let's change the subject.
    > ok, lets talk about music
    Why do you say that?
    > because I like music!
    Why do you say that?
    > bye
    It was nice talking to you, goodbye!
    ```

    En möjlig lösning på uppgiften finns [här](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/1-Introduction-to-NLP/solution/bot.py)

    ✅ Stanna upp och fundera

    1. Tror du att de slumpmässiga svaren skulle "lura" någon att tro att boten faktiskt förstod dem?
    2. Vilka funktioner skulle boten behöva för att vara mer effektiv?
    3. Om en bot verkligen kunde "förstå" innebörden av en mening, skulle den behöva "komma ihåg" innebörden av tidigare meningar i en konversation också?

---

## 🚀Utmaning

Välj ett av elementen ovan under "stanna upp och fundera" och försök antingen implementera det i kod eller skriv en lösning på papper med pseudokod.

I nästa lektion kommer du att lära dig om ett antal andra metoder för att parsa naturligt språk och maskininlärning.

## [Quiz efter föreläsningen](https://ff-quizzes.netlify.app/en/ml/)

## Granskning & Självstudier

Ta en titt på referenserna nedan som ytterligare läsmöjligheter.

### Referenser

1. Schubert, Lenhart, "Computational Linguistics", *The Stanford Encyclopedia of Philosophy* (Spring 2020 Edition), Edward N. Zalta (ed.), URL = <https://plato.stanford.edu/archives/spr2020/entries/computational-linguistics/>.
2. Princeton University "About WordNet." [WordNet](https://wordnet.princeton.edu/). Princeton University. 2010. 

## Uppgift 

[Sök efter en bot](assignment.md)

---

**Ansvarsfriskrivning**:  
Detta dokument har översatts med hjälp av AI-översättningstjänsten [Co-op Translator](https://github.com/Azure/co-op-translator). Även om vi strävar efter noggrannhet, vänligen notera att automatiska översättningar kan innehålla fel eller felaktigheter. Det ursprungliga dokumentet på sitt originalspråk bör betraktas som den auktoritativa källan. För kritisk information rekommenderas professionell mänsklig översättning. Vi ansvarar inte för eventuella missförstånd eller feltolkningar som uppstår vid användning av denna översättning.