<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "911efd5e595089000cb3c16fce1beab8",
  "translation_date": "2025-09-05T22:04:16+00:00",
  "source_file": "8-Reinforcement/1-QLearning/README.md",
  "language_code": "sv"
}
-->
# Introduktion till förstärkningsinlärning och Q-Learning

![Sammanfattning av förstärkning inom maskininlärning i en sketchnote](../../../../sketchnotes/ml-reinforcement.png)
> Sketchnote av [Tomomi Imura](https://www.twitter.com/girlie_mac)

Förstärkningsinlärning involverar tre viktiga koncept: agenten, några tillstånd och en uppsättning av handlingar per tillstånd. Genom att utföra en handling i ett specifikt tillstånd får agenten en belöning. Tänk dig datorspelet Super Mario. Du är Mario, du befinner dig på en nivå i spelet, stående vid kanten av en klippa. Ovanför dig finns ett mynt. Du som Mario, på en nivå i spelet, på en specifik position ... det är ditt tillstånd. Om du tar ett steg åt höger (en handling) faller du över kanten, vilket ger dig en låg numerisk poäng. Men om du trycker på hoppknappen får du en poäng och förblir vid liv. Det är ett positivt resultat och bör ge dig en positiv numerisk poäng.

Genom att använda förstärkningsinlärning och en simulator (spelet) kan du lära dig att spela spelet för att maximera belöningen, vilket innebär att hålla dig vid liv och samla så många poäng som möjligt.

[![Introduktion till förstärkningsinlärning](https://img.youtube.com/vi/lDq_en8RNOo/0.jpg)](https://www.youtube.com/watch?v=lDq_en8RNOo)

> 🎥 Klicka på bilden ovan för att höra Dmitry diskutera förstärkningsinlärning

## [Quiz före föreläsningen](https://ff-quizzes.netlify.app/en/ml/)

## Förutsättningar och inställning

I denna lektion kommer vi att experimentera med kod i Python. Du bör kunna köra Jupyter Notebook-koden från denna lektion, antingen på din dator eller i molnet.

Du kan öppna [lektionens notebook](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/notebook.ipynb) och följa med i lektionen för att bygga.

> **Obs:** Om du öppnar denna kod från molnet behöver du också hämta filen [`rlboard.py`](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/rlboard.py), som används i notebook-koden. Lägg den i samma katalog som notebook-filen.

## Introduktion

I denna lektion kommer vi att utforska världen av **[Peter och vargen](https://en.wikipedia.org/wiki/Peter_and_the_Wolf)**, inspirerad av en musikalisk saga av den ryske kompositören [Sergei Prokofiev](https://en.wikipedia.org/wiki/Sergei_Prokofiev). Vi kommer att använda **förstärkningsinlärning** för att låta Peter utforska sin miljö, samla goda äpplen och undvika att möta vargen.

**Förstärkningsinlärning** (RL) är en inlärningsteknik som gör det möjligt för oss att lära oss ett optimalt beteende hos en **agent** i en viss **miljö** genom att köra många experiment. En agent i denna miljö bör ha ett **mål**, definierat av en **belöningsfunktion**.

## Miljön

För enkelhetens skull kan vi tänka oss Peters värld som en kvadratisk spelbräda med storleken `width` x `height`, som denna:

![Peters miljö](../../../../8-Reinforcement/1-QLearning/images/environment.png)

Varje cell på denna bräda kan vara:

* **mark**, där Peter och andra varelser kan gå.
* **vatten**, där man uppenbarligen inte kan gå.
* ett **träd** eller **gräs**, en plats där man kan vila.
* ett **äpple**, som representerar något Peter gärna vill hitta för att äta.
* en **varg**, som är farlig och bör undvikas.

Det finns en separat Python-modul, [`rlboard.py`](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/rlboard.py), som innehåller koden för att arbeta med denna miljö. Eftersom denna kod inte är viktig för att förstå våra koncept, kommer vi att importera modulen och använda den för att skapa spelbrädan (kodblock 1):

```python
from rlboard import *

width, height = 8,8
m = Board(width,height)
m.randomize(seed=13)
m.plot()
```

Denna kod bör skriva ut en bild av miljön som liknar den ovan.

## Handlingar och strategi

I vårt exempel skulle Peters mål vara att hitta ett äpple, samtidigt som han undviker vargen och andra hinder. För att göra detta kan han i princip gå runt tills han hittar ett äpple.

Därför kan han vid varje position välja mellan följande handlingar: upp, ner, vänster och höger.

Vi kommer att definiera dessa handlingar som en ordbok och koppla dem till par av motsvarande koordinatförändringar. Till exempel skulle rörelse åt höger (`R`) motsvara paret `(1,0)`. (kodblock 2):

```python
actions = { "U" : (0,-1), "D" : (0,1), "L" : (-1,0), "R" : (1,0) }
action_idx = { a : i for i,a in enumerate(actions.keys()) }
```

Sammanfattningsvis är strategin och målet för detta scenario följande:

- **Strategin**, för vår agent (Peter) definieras av en så kallad **policy**. En policy är en funktion som returnerar handlingen vid ett givet tillstånd. I vårt fall representeras problemet av spelbrädan, inklusive spelarens aktuella position.

- **Målet**, med förstärkningsinlärning är att så småningom lära sig en bra policy som gör det möjligt för oss att lösa problemet effektivt. Men som en grundläggande utgångspunkt kan vi överväga den enklaste policyn som kallas **slumpvandring**.

## Slumpvandring

Låt oss först lösa vårt problem genom att implementera en slumpvandring-strategi. Med slumpvandring kommer vi slumpmässigt att välja nästa handling från de tillåtna handlingarna tills vi når äpplet (kodblock 3).

1. Implementera slumpvandringen med koden nedan:

    ```python
    def random_policy(m):
        return random.choice(list(actions))
    
    def walk(m,policy,start_position=None):
        n = 0 # number of steps
        # set initial position
        if start_position:
            m.human = start_position 
        else:
            m.random_start()
        while True:
            if m.at() == Board.Cell.apple:
                return n # success!
            if m.at() in [Board.Cell.wolf, Board.Cell.water]:
                return -1 # eaten by wolf or drowned
            while True:
                a = actions[policy(m)]
                new_pos = m.move_pos(m.human,a)
                if m.is_valid(new_pos) and m.at(new_pos)!=Board.Cell.water:
                    m.move(a) # do the actual move
                    break
            n+=1
    
    walk(m,random_policy)
    ```

    Anropet till `walk` bör returnera längden på den motsvarande vägen, som kan variera från en körning till en annan.

1. Kör experimentet med slumpvandring ett antal gånger (säg 100) och skriv ut de resulterande statistiken (kodblock 4):

    ```python
    def print_statistics(policy):
        s,w,n = 0,0,0
        for _ in range(100):
            z = walk(m,policy)
            if z<0:
                w+=1
            else:
                s += z
                n += 1
        print(f"Average path length = {s/n}, eaten by wolf: {w} times")
    
    print_statistics(random_policy)
    ```

    Observera att den genomsnittliga längden på en väg är cirka 30-40 steg, vilket är ganska mycket, med tanke på att det genomsnittliga avståndet till närmaste äpple är cirka 5-6 steg.

    Du kan också se hur Peters rörelse ser ut under slumpvandringen:

    ![Peters slumpvandring](../../../../8-Reinforcement/1-QLearning/images/random_walk.gif)

## Belöningsfunktion

För att göra vår policy mer intelligent behöver vi förstå vilka rörelser som är "bättre" än andra. För att göra detta måste vi definiera vårt mål.

Målet kan definieras i termer av en **belöningsfunktion**, som returnerar ett poängvärde för varje tillstånd. Ju högre nummer, desto bättre belöningsfunktion. (kodblock 5)

```python
move_reward = -0.1
goal_reward = 10
end_reward = -10

def reward(m,pos=None):
    pos = pos or m.human
    if not m.is_valid(pos):
        return end_reward
    x = m.at(pos)
    if x==Board.Cell.water or x == Board.Cell.wolf:
        return end_reward
    if x==Board.Cell.apple:
        return goal_reward
    return move_reward
```

En intressant sak med belöningsfunktioner är att i de flesta fall *får vi bara en betydande belöning i slutet av spelet*. Detta innebär att vår algoritm på något sätt måste komma ihåg "bra" steg som leder till en positiv belöning i slutet och öka deras betydelse. På samma sätt bör alla rörelser som leder till dåliga resultat avskräckas.

## Q-Learning

En algoritm som vi kommer att diskutera här kallas **Q-Learning**. I denna algoritm definieras policyn av en funktion (eller en datastruktur) som kallas en **Q-Tabell**. Den registrerar "kvaliteten" på varje handling i ett givet tillstånd.

Den kallas en Q-Tabell eftersom det ofta är bekvämt att representera den som en tabell eller en multidimensionell array. Eftersom vår bräda har dimensionerna `width` x `height`, kan vi representera Q-Tabellen med en numpy-array med formen `width` x `height` x `len(actions)`: (kodblock 6)

```python
Q = np.ones((width,height,len(actions)),dtype=np.float)*1.0/len(actions)
```

Observera att vi initialiserar alla värden i Q-Tabellen med ett lika värde, i vårt fall - 0.25. Detta motsvarar policyn "slumpvandring", eftersom alla rörelser i varje tillstånd är lika bra. Vi kan skicka Q-Tabellen till funktionen `plot` för att visualisera tabellen på brädan: `m.plot(Q)`.

![Peters miljö](../../../../8-Reinforcement/1-QLearning/images/env_init.png)

I mitten av varje cell finns en "pil" som indikerar den föredragna rörelseriktningen. Eftersom alla riktningar är lika visas en punkt.

Nu behöver vi köra simuleringen, utforska vår miljö och lära oss en bättre fördelning av Q-Tabellvärden, vilket gör att vi kan hitta vägen till äpplet mycket snabbare.

## Kärnan i Q-Learning: Bellman-ekvationen

När vi börjar röra oss kommer varje handling att ha en motsvarande belöning, dvs. vi kan teoretiskt välja nästa handling baserat på den högsta omedelbara belöningen. Men i de flesta tillstånd kommer rörelsen inte att uppnå vårt mål att nå äpplet, och vi kan därför inte omedelbart avgöra vilken riktning som är bättre.

> Kom ihåg att det inte är det omedelbara resultatet som spelar roll, utan snarare det slutliga resultatet, som vi kommer att få i slutet av simuleringen.

För att ta hänsyn till denna fördröjda belöning måste vi använda principerna för **[dynamisk programmering](https://en.wikipedia.org/wiki/Dynamic_programming)**, som gör det möjligt för oss att tänka på vårt problem rekursivt.

Anta att vi nu är i tillståndet *s*, och vi vill gå vidare till nästa tillstånd *s'*. Genom att göra det kommer vi att få den omedelbara belöningen *r(s,a)*, definierad av belöningsfunktionen, plus någon framtida belöning. Om vi antar att vår Q-Tabell korrekt återspeglar "attraktiviteten" hos varje handling, kommer vi vid tillståndet *s'* att välja en handling *a* som motsvarar det maximala värdet av *Q(s',a')*. Således definieras den bästa möjliga framtida belöningen vi kan få vid tillståndet *s* som `max`

## Kontrollera policyn

Eftersom Q-Tabellen listar "attraktiviteten" för varje handling i varje tillstånd, är det ganska enkelt att använda den för att definiera effektiv navigering i vår värld. I det enklaste fallet kan vi välja den handling som motsvarar det högsta värdet i Q-Tabellen: (kodblock 9)

```python
def qpolicy_strict(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = list(actions)[np.argmax(v)]
        return a

walk(m,qpolicy_strict)
```

> Om du testar koden ovan flera gånger kan du märka att den ibland "fastnar", och du måste trycka på STOP-knappen i notebooken för att avbryta den. Detta händer eftersom det kan finnas situationer där två tillstånd "pekar" på varandra i termer av optimalt Q-värde, vilket gör att agenten hamnar i en oändlig rörelse mellan dessa tillstånd.

## 🚀Utmaning

> **Uppgift 1:** Modifiera funktionen `walk` för att begränsa den maximala längden på vägen till ett visst antal steg (t.ex. 100), och se hur koden ovan returnerar detta värde då och då.

> **Uppgift 2:** Modifiera funktionen `walk` så att den inte går tillbaka till platser där den redan har varit tidigare. Detta kommer att förhindra att `walk` hamnar i en loop, men agenten kan fortfarande bli "fast" på en plats som den inte kan ta sig ifrån.

## Navigering

En bättre navigeringspolicy skulle vara den vi använde under träningen, som kombinerar exploatering och utforskning. I denna policy kommer vi att välja varje handling med en viss sannolikhet, proportionell mot värdena i Q-Tabellen. Denna strategi kan fortfarande leda till att agenten återvänder till en position den redan har utforskat, men som du kan se från koden nedan resulterar den i en mycket kortare genomsnittlig väg till den önskade platsen (kom ihåg att `print_statistics` kör simuleringen 100 gånger): (kodblock 10)

```python
def qpolicy(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = random.choices(list(actions),weights=v)[0]
        return a

print_statistics(qpolicy)
```

Efter att ha kört denna kod bör du få en mycket mindre genomsnittlig väg än tidigare, i intervallet 3-6.

## Undersöka inlärningsprocessen

Som vi har nämnt är inlärningsprocessen en balans mellan utforskning och utnyttjande av den kunskap vi har fått om problemområdets struktur. Vi har sett att resultaten av inlärningen (förmågan att hjälpa en agent att hitta en kort väg till målet) har förbättrats, men det är också intressant att observera hur den genomsnittliga väglängden beter sig under inlärningsprocessen:

## Sammanfattning av lärdomarna:

- **Genomsnittlig väglängd ökar**. Det vi ser här är att i början ökar den genomsnittliga väglängden. Detta beror troligen på att när vi inte vet något om miljön är vi mer benägna att fastna i dåliga tillstånd, som vatten eller varg. När vi lär oss mer och börjar använda denna kunskap kan vi utforska miljön längre, men vi vet fortfarande inte var äpplena finns särskilt väl.

- **Väglängden minskar när vi lär oss mer**. När vi har lärt oss tillräckligt blir det lättare för agenten att nå målet, och väglängden börjar minska. Vi är dock fortfarande öppna för utforskning, så vi avviker ofta från den bästa vägen och utforskar nya alternativ, vilket gör vägen längre än optimal.

- **Längden ökar plötsligt**. Det vi också observerar på denna graf är att vid någon punkt ökade längden plötsligt. Detta indikerar den stokastiska naturen hos processen, och att vi vid något tillfälle kan "förstöra" Q-Tabellens koefficienter genom att skriva över dem med nya värden. Detta bör idealt minimeras genom att minska inlärningshastigheten (till exempel mot slutet av träningen justerar vi endast Q-Tabellens värden med ett litet värde).

Sammanfattningsvis är det viktigt att komma ihåg att framgången och kvaliteten på inlärningsprocessen beror avsevärt på parametrar som inlärningshastighet, inlärningshastighetsminskning och diskonteringsfaktor. Dessa kallas ofta **hyperparametrar**, för att skilja dem från **parametrar**, som vi optimerar under träningen (till exempel Q-Tabellens koefficienter). Processen att hitta de bästa värdena för hyperparametrar kallas **hyperparameteroptimering**, och det förtjänar ett eget ämne.

## [Quiz efter föreläsningen](https://ff-quizzes.netlify.app/en/ml/)

## Uppgift 
[En mer realistisk värld](assignment.md)

---

**Ansvarsfriskrivning**:  
Detta dokument har översatts med hjälp av AI-översättningstjänsten [Co-op Translator](https://github.com/Azure/co-op-translator). Även om vi strävar efter noggrannhet, vänligen notera att automatiska översättningar kan innehålla fel eller felaktigheter. Det ursprungliga dokumentet på dess originalspråk bör betraktas som den auktoritativa källan. För kritisk information rekommenderas professionell mänsklig översättning. Vi ansvarar inte för eventuella missförstånd eller feltolkningar som uppstår vid användning av denna översättning.