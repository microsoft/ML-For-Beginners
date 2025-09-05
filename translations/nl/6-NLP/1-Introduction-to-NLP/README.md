<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1c2ec40cf55c98a028a359c27ef7e45a",
  "translation_date": "2025-09-05T20:35:21+00:00",
  "source_file": "6-NLP/1-Introduction-to-NLP/README.md",
  "language_code": "nl"
}
-->
# Introductie tot natuurlijke taalverwerking

Deze les behandelt een korte geschiedenis en belangrijke concepten van *natuurlijke taalverwerking*, een subveld van *computational linguistics*.

## [Quiz voorafgaand aan de les](https://ff-quizzes.netlify.app/en/ml/)

## Introductie

NLP, zoals het vaak wordt genoemd, is een van de bekendste gebieden waar machine learning is toegepast en gebruikt in productie-software.

✅ Kun je software bedenken die je dagelijks gebruikt en waarschijnlijk wat NLP ingebouwd heeft? Wat dacht je van je tekstverwerkingsprogramma's of mobiele apps die je regelmatig gebruikt?

Je leert over:

- **Het idee van talen**. Hoe talen zich hebben ontwikkeld en wat de belangrijkste studiegebieden zijn geweest.
- **Definities en concepten**. Je leert ook definities en concepten over hoe computers tekst verwerken, inclusief parsing, grammatica en het identificeren van zelfstandige naamwoorden en werkwoorden. Er zijn enkele programmeertaken in deze les, en er worden verschillende belangrijke concepten geïntroduceerd die je later in de volgende lessen zult leren coderen.

## Computational linguistics

Computational linguistics is een onderzoeks- en ontwikkelingsgebied dat al tientallen jaren bestudeert hoe computers kunnen werken met, en zelfs begrijpen, vertalen en communiceren in talen. Natuurlijke taalverwerking (NLP) is een gerelateerd veld dat zich richt op hoe computers 'natuurlijke', of menselijke, talen kunnen verwerken.

### Voorbeeld - telefoon dicteren

Als je ooit tegen je telefoon hebt gedicteerd in plaats van te typen of een virtuele assistent een vraag hebt gesteld, is je spraak omgezet in tekstvorm en vervolgens verwerkt of *geparsed* vanuit de taal die je sprak. De gedetecteerde trefwoorden werden vervolgens verwerkt in een formaat dat de telefoon of assistent kon begrijpen en waarop kon worden gehandeld.

![begrip](../../../../6-NLP/1-Introduction-to-NLP/images/comprehension.png)
> Echt taalkundig begrip is moeilijk! Afbeelding door [Jen Looper](https://twitter.com/jenlooper)

### Hoe is deze technologie mogelijk gemaakt?

Dit is mogelijk omdat iemand een computerprogramma heeft geschreven om dit te doen. Enkele decennia geleden voorspelden sommige sciencefiction-schrijvers dat mensen voornamelijk met hun computers zouden praten en dat de computers altijd precies zouden begrijpen wat ze bedoelden. Helaas bleek het een moeilijker probleem te zijn dan velen hadden gedacht, en hoewel het tegenwoordig een veel beter begrepen probleem is, zijn er aanzienlijke uitdagingen bij het bereiken van 'perfecte' natuurlijke taalverwerking als het gaat om het begrijpen van de betekenis van een zin. Dit is een bijzonder moeilijk probleem als het gaat om het begrijpen van humor of het detecteren van emoties zoals sarcasme in een zin.

Op dit moment herinner je je misschien schoollessen waarin de leraar de onderdelen van grammatica in een zin behandelde. In sommige landen krijgen studenten grammatica en taalkunde als een apart vak, maar in veel landen worden deze onderwerpen opgenomen als onderdeel van het leren van een taal: ofwel je eerste taal op de basisschool (leren lezen en schrijven) en misschien een tweede taal op de middelbare school. Maak je geen zorgen als je geen expert bent in het onderscheiden van zelfstandige naamwoorden van werkwoorden of bijwoorden van bijvoeglijke naamwoorden!

Als je moeite hebt met het verschil tussen de *eenvoudige tegenwoordige tijd* en de *tegenwoordige progressieve tijd*, ben je niet de enige. Dit is een uitdaging voor veel mensen, zelfs moedertaalsprekers van een taal. Het goede nieuws is dat computers heel goed zijn in het toepassen van formele regels, en je zult leren code schrijven die een zin kan *parsen* net zo goed als een mens. De grotere uitdaging die je later zult onderzoeken, is het begrijpen van de *betekenis* en *sentiment* van een zin.

## Vereisten

Voor deze les is de belangrijkste vereiste dat je de taal van deze les kunt lezen en begrijpen. Er zijn geen wiskundige problemen of vergelijkingen om op te lossen. Hoewel de oorspronkelijke auteur deze les in het Engels heeft geschreven, is deze ook vertaald in andere talen, dus je zou een vertaling kunnen lezen. Er zijn voorbeelden waarin een aantal verschillende talen worden gebruikt (om de verschillende grammaticaregels van verschillende talen te vergelijken). Deze worden *niet* vertaald, maar de verklarende tekst wel, zodat de betekenis duidelijk moet zijn.

Voor de programmeertaken gebruik je Python en de voorbeelden gebruiken Python 3.8.

In deze sectie heb je nodig, en gebruik je:

- **Python 3 begrip**. Begrip van de programmeertaal Python 3, deze les gebruikt invoer, loops, bestand lezen, arrays.
- **Visual Studio Code + extensie**. We gebruiken Visual Studio Code en de Python-extensie. Je kunt ook een Python IDE naar keuze gebruiken.
- **TextBlob**. [TextBlob](https://github.com/sloria/TextBlob) is een vereenvoudigde tekstverwerkingsbibliotheek voor Python. Volg de instructies op de TextBlob-site om het op je systeem te installeren (installeer ook de corpora, zoals hieronder weergegeven):

   ```bash
   pip install -U textblob
   python -m textblob.download_corpora
   ```

> 💡 Tip: Je kunt Python direct uitvoeren in VS Code-omgevingen. Bekijk de [documentatie](https://code.visualstudio.com/docs/languages/python?WT.mc_id=academic-77952-leestott) voor meer informatie.

## Praten met machines

De geschiedenis van het proberen computers menselijke taal te laten begrijpen gaat tientallen jaren terug, en een van de vroegste wetenschappers die natuurlijke taalverwerking overwoog was *Alan Turing*.

### De 'Turing-test'

Toen Turing in de jaren 50 onderzoek deed naar *kunstmatige intelligentie*, overwoog hij of een gesprekstest kon worden gegeven aan een mens en een computer (via getypte correspondentie) waarbij de mens in het gesprek niet zeker wist of ze met een andere mens of een computer aan het praten waren.

Als de mens na een bepaalde lengte van het gesprek niet kon bepalen of de antwoorden van een computer kwamen of niet, kon de computer dan worden gezegd te *denken*?

### De inspiratie - 'het imitatiespel'

Het idee hiervoor kwam van een gezelschapsspel genaamd *Het Imitatiespel* waarbij een ondervrager alleen in een kamer is en de taak heeft te bepalen welke van twee mensen (in een andere kamer) respectievelijk man en vrouw zijn. De ondervrager kan notities sturen en moet proberen vragen te bedenken waarbij de geschreven antwoorden het geslacht van de mysterieuze persoon onthullen. Natuurlijk proberen de spelers in de andere kamer de ondervrager te misleiden door vragen op een manier te beantwoorden die de ondervrager misleidt of verwart, terwijl ze ook de indruk wekken eerlijk te antwoorden.

### Het ontwikkelen van Eliza

In de jaren 60 ontwikkelde een MIT-wetenschapper genaamd *Joseph Weizenbaum* [*Eliza*](https://wikipedia.org/wiki/ELIZA), een computer-'therapeut' die de mens vragen zou stellen en de indruk zou wekken hun antwoorden te begrijpen. Echter, hoewel Eliza een zin kon parsen en bepaalde grammaticale constructies en trefwoorden kon identificeren om een redelijk antwoord te geven, kon niet worden gezegd dat het de zin *begrijpt*. Als Eliza een zin kreeg in het formaat "**Ik ben** <u>verdrietig</u>", zou het woorden in de zin kunnen herschikken en vervangen om de reactie "Hoe lang ben **jij** <u>verdrietig</u>" te vormen.

Dit gaf de indruk dat Eliza de uitspraak begreep en een vervolgvraag stelde, terwijl het in werkelijkheid de tijd veranderde en enkele woorden toevoegde. Als Eliza geen trefwoord kon identificeren waarvoor het een reactie had, zou het in plaats daarvan een willekeurige reactie geven die van toepassing zou kunnen zijn op veel verschillende uitspraken. Eliza kon gemakkelijk worden misleid, bijvoorbeeld als een gebruiker schreef "**Jij bent** een <u>fiets</u>", zou het kunnen reageren met "Hoe lang ben **ik** een <u>fiets</u>?", in plaats van een meer redelijke reactie.

[![Chatten met Eliza](https://img.youtube.com/vi/RMK9AphfLco/0.jpg)](https://youtu.be/RMK9AphfLco "Chatten met Eliza")

> 🎥 Klik op de afbeelding hierboven voor een video over het originele ELIZA-programma

> Opmerking: Je kunt de originele beschrijving van [Eliza](https://cacm.acm.org/magazines/1966/1/13317-elizaa-computer-program-for-the-study-of-natural-language-communication-between-man-and-machine/abstract) gepubliceerd in 1966 lezen als je een ACM-account hebt. Alternatief, lees over Eliza op [wikipedia](https://wikipedia.org/wiki/ELIZA)

## Oefening - een eenvoudige conversatiebot coderen

Een conversatiebot, zoals Eliza, is een programma dat gebruikersinput uitlokt en lijkt te begrijpen en intelligent te reageren. In tegenstelling tot Eliza zal onze bot geen verschillende regels hebben die de indruk wekken dat het een intelligent gesprek voert. In plaats daarvan zal onze bot slechts één vaardigheid hebben: het gesprek gaande houden met willekeurige reacties die in bijna elk triviaal gesprek zouden kunnen werken.

### Het plan

Je stappen bij het bouwen van een conversatiebot:

1. Print instructies waarin de gebruiker wordt geadviseerd hoe met de bot te communiceren
2. Start een loop
   1. Accepteer gebruikersinput
   2. Als de gebruiker heeft gevraagd om te stoppen, stop dan
   3. Verwerk gebruikersinput en bepaal de reactie (in dit geval is de reactie een willekeurige keuze uit een lijst met mogelijke generieke reacties)
   4. Print reactie
3. Ga terug naar stap 2

### De bot bouwen

Laten we de bot nu maken. We beginnen met het definiëren van enkele zinnen.

1. Maak deze bot zelf in Python met de volgende willekeurige reacties:

    ```python
    random_responses = ["That is quite interesting, please tell me more.",
                        "I see. Do go on.",
                        "Why do you say that?",
                        "Funny weather we've been having, isn't it?",
                        "Let's change the subject.",
                        "Did you catch the game last night?"]
    ```

    Hier is een voorbeeldoutput om je te begeleiden (gebruikersinput staat op de regels die beginnen met `>`):

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

    Een mogelijke oplossing voor de taak is [hier](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/1-Introduction-to-NLP/solution/bot.py)

    ✅ Stop en overweeg

    1. Denk je dat de willekeurige reacties iemand zouden 'misleiden' om te denken dat de bot hen daadwerkelijk begreep?
    2. Welke functies zou de bot nodig hebben om effectiever te zijn?
    3. Als een bot echt de betekenis van een zin zou kunnen 'begrijpen', zou het dan ook de betekenis van eerdere zinnen in een gesprek moeten 'onthouden'?

---

## 🚀Uitdaging

Kies een van de "stop en overweeg"-elementen hierboven en probeer deze te implementeren in code of schrijf een oplossing op papier met pseudocode.

In de volgende les leer je over een aantal andere benaderingen voor het parsen van natuurlijke taal en machine learning.

## [Quiz na de les](https://ff-quizzes.netlify.app/en/ml/)

## Review & Zelfstudie

Bekijk de onderstaande referenties als verdere leesmogelijkheden.

### Referenties

1. Schubert, Lenhart, "Computational Linguistics", *The Stanford Encyclopedia of Philosophy* (Spring 2020 Edition), Edward N. Zalta (ed.), URL = <https://plato.stanford.edu/archives/spr2020/entries/computational-linguistics/>.
2. Princeton University "About WordNet." [WordNet](https://wordnet.princeton.edu/). Princeton University. 2010. 

## Opdracht 

[Zoek een bot](assignment.md)

---

**Disclaimer**:  
Dit document is vertaald met behulp van de AI-vertalingsservice [Co-op Translator](https://github.com/Azure/co-op-translator). Hoewel we streven naar nauwkeurigheid, dient u zich ervan bewust te zijn dat geautomatiseerde vertalingen fouten of onnauwkeurigheden kunnen bevatten. Het originele document in zijn oorspronkelijke taal moet worden beschouwd als de gezaghebbende bron. Voor cruciale informatie wordt professionele menselijke vertaling aanbevolen. Wij zijn niet aansprakelijk voor misverstanden of verkeerde interpretaties die voortvloeien uit het gebruik van deze vertaling.