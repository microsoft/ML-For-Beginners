<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "911efd5e595089000cb3c16fce1beab8",
  "translation_date": "2025-09-05T01:08:08+00:00",
  "source_file": "8-Reinforcement/1-QLearning/README.md",
  "language_code": "da"
}
-->
# Introduktion til Forstærkningslæring og Q-Learning

![Oversigt over forstærkning i maskinlæring i en sketchnote](../../../../sketchnotes/ml-reinforcement.png)
> Sketchnote af [Tomomi Imura](https://www.twitter.com/girlie_mac)

Forstærkningslæring involverer tre vigtige begreber: agenten, nogle tilstande og et sæt handlinger pr. tilstand. Ved at udføre en handling i en specifik tilstand får agenten en belønning. Forestil dig igen computerspillet Super Mario. Du er Mario, du befinder dig i et spilniveau, stående ved kanten af en klippe. Over dig er der en mønt. Du som Mario, i et spilniveau, på en specifik position ... det er din tilstand. Hvis du tager et skridt til højre (en handling), falder du ud over kanten, hvilket giver dig en lav numerisk score. Men hvis du trykker på hop-knappen, scorer du et point og forbliver i live. Det er et positivt resultat, og det bør give dig en positiv numerisk score.

Ved at bruge forstærkningslæring og en simulator (spillet) kan du lære at spille spillet for at maksimere belønningen, som er at forblive i live og score så mange point som muligt.

[![Introduktion til Forstærkningslæring](https://img.youtube.com/vi/lDq_en8RNOo/0.jpg)](https://www.youtube.com/watch?v=lDq_en8RNOo)

> 🎥 Klik på billedet ovenfor for at høre Dmitry diskutere Forstærkningslæring

## [Quiz før lektionen](https://ff-quizzes.netlify.app/en/ml/)

## Forudsætninger og Opsætning

I denne lektion vil vi eksperimentere med noget kode i Python. Du skal kunne køre Jupyter Notebook-koden fra denne lektion, enten på din computer eller i skyen.

Du kan åbne [lektionens notebook](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/notebook.ipynb) og gennemgå denne lektion for at bygge.

> **Bemærk:** Hvis du åbner denne kode fra skyen, skal du også hente filen [`rlboard.py`](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/rlboard.py), som bruges i notebook-koden. Tilføj den til samme mappe som notebooken.

## Introduktion

I denne lektion vil vi udforske verdenen af **[Peter og Ulven](https://en.wikipedia.org/wiki/Peter_and_the_Wolf)**, inspireret af et musikalsk eventyr af den russiske komponist [Sergei Prokofiev](https://en.wikipedia.org/wiki/Sergei_Prokofiev). Vi vil bruge **Forstærkningslæring** til at lade Peter udforske sit miljø, samle lækre æbler og undgå at møde ulven.

**Forstærkningslæring** (RL) er en læringsteknik, der giver os mulighed for at lære en optimal adfærd for en **agent** i et **miljø** ved at udføre mange eksperimenter. En agent i dette miljø skal have et **mål**, defineret af en **belønningsfunktion**.

## Miljøet

For enkelhedens skyld lad os antage, at Peters verden er en kvadratisk spilleplade med størrelsen `bredde` x `højde`, som denne:

![Peters Miljø](../../../../8-Reinforcement/1-QLearning/images/environment.png)

Hver celle på denne spilleplade kan enten være:

* **jord**, som Peter og andre væsener kan gå på.
* **vand**, som man selvfølgelig ikke kan gå på.
* et **træ** eller **græs**, et sted hvor man kan hvile sig.
* et **æble**, som repræsenterer noget, Peter ville være glad for at finde for at spise.
* en **ulv**, som er farlig og bør undgås.

Der er et separat Python-modul, [`rlboard.py`](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/rlboard.py), som indeholder koden til at arbejde med dette miljø. Fordi denne kode ikke er vigtig for at forstå vores begreber, vil vi importere modulet og bruge det til at oprette spillepladen (kodeblok 1):

```python
from rlboard import *

width, height = 8,8
m = Board(width,height)
m.randomize(seed=13)
m.plot()
```

Denne kode bør udskrive et billede af miljøet, der ligner det ovenfor.

## Handlinger og strategi

I vores eksempel vil Peters mål være at finde et æble, mens han undgår ulven og andre forhindringer. For at gøre dette kan han i princippet gå rundt, indtil han finder et æble.

Derfor kan han på enhver position vælge mellem følgende handlinger: op, ned, venstre og højre.

Vi vil definere disse handlinger som en ordbog og kortlægge dem til par af tilsvarende koordinatændringer. For eksempel vil bevægelse til højre (`R`) svare til et par `(1,0)`. (kodeblok 2):

```python
actions = { "U" : (0,-1), "D" : (0,1), "L" : (-1,0), "R" : (1,0) }
action_idx = { a : i for i,a in enumerate(actions.keys()) }
```

For at opsummere er strategien og målet for dette scenarie som følger:

- **Strategien**, for vores agent (Peter) er defineret af en såkaldt **politik**. En politik er en funktion, der returnerer handlingen i en given tilstand. I vores tilfælde er problemet repræsenteret af spillepladen, inklusive spillerens aktuelle position.

- **Målet**, med forstærkningslæring er at lære en god politik, der gør det muligt for os at løse problemet effektivt. Som en baseline lad os overveje den enkleste politik kaldet **tilfældig gang**.

## Tilfældig gang

Lad os først løse vores problem ved at implementere en strategi med tilfældig gang. Med tilfældig gang vil vi tilfældigt vælge den næste handling fra de tilladte handlinger, indtil vi når æblet (kodeblok 3).

1. Implementer den tilfældige gang med nedenstående kode:

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

    Kaldet til `walk` bør returnere længden af den tilsvarende sti, som kan variere fra den ene kørsel til den anden. 

1. Kør gangeksperimentet et antal gange (f.eks. 100), og udskriv de resulterende statistikker (kodeblok 4):

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

    Bemærk, at den gennemsnitlige længde af en sti er omkring 30-40 trin, hvilket er ret meget, i betragtning af at den gennemsnitlige afstand til det nærmeste æble er omkring 5-6 trin.

    Du kan også se, hvordan Peters bevægelse ser ud under den tilfældige gang:

    ![Peters Tilfældige Gang](../../../../8-Reinforcement/1-QLearning/images/random_walk.gif)

## Belønningsfunktion

For at gøre vores politik mere intelligent skal vi forstå, hvilke bevægelser der er "bedre" end andre. For at gøre dette skal vi definere vores mål.

Målet kan defineres i form af en **belønningsfunktion**, som vil returnere en scoreværdi for hver tilstand. Jo højere tallet er, desto bedre er belønningsfunktionen. (kodeblok 5)

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

En interessant ting ved belønningsfunktioner er, at i de fleste tilfælde *får vi kun en væsentlig belønning ved slutningen af spillet*. Det betyder, at vores algoritme på en eller anden måde skal huske "gode" trin, der fører til en positiv belønning til sidst, og øge deres betydning. Tilsvarende bør alle bevægelser, der fører til dårlige resultater, frarådes.

## Q-Learning

En algoritme, som vi vil diskutere her, kaldes **Q-Learning**. I denne algoritme defineres politikken af en funktion (eller en datastruktur) kaldet en **Q-Table**. Den registrerer "godheden" af hver af handlingerne i en given tilstand.

Den kaldes en Q-Table, fordi det ofte er praktisk at repræsentere den som en tabel eller en multidimensionel array. Da vores spilleplade har dimensionerne `bredde` x `højde`, kan vi repræsentere Q-Table ved hjælp af en numpy-array med formen `bredde` x `højde` x `len(actions)`: (kodeblok 6)

```python
Q = np.ones((width,height,len(actions)),dtype=np.float)*1.0/len(actions)
```

Bemærk, at vi initialiserer alle værdierne i Q-Table med en ens værdi, i vores tilfælde - 0.25. Dette svarer til politikken "tilfældig gang", fordi alle bevægelser i hver tilstand er lige gode. Vi kan sende Q-Table til `plot`-funktionen for at visualisere tabellen på spillepladen: `m.plot(Q)`.

![Peters Miljø](../../../../8-Reinforcement/1-QLearning/images/env_init.png)

I midten af hver celle er der en "pil", der angiver den foretrukne bevægelsesretning. Da alle retninger er ens, vises en prik.

Nu skal vi køre simuleringen, udforske vores miljø og lære en bedre fordeling af Q-Table-værdier, som gør det muligt for os at finde vejen til æblet meget hurtigere.

## Essensen af Q-Learning: Bellman-ligningen

Når vi begynder at bevæge os, vil hver handling have en tilsvarende belønning, dvs. vi kan teoretisk vælge den næste handling baseret på den højeste umiddelbare belønning. Men i de fleste tilstande vil bevægelsen ikke opnå vores mål om at nå æblet, og vi kan derfor ikke straks beslutte, hvilken retning der er bedre.

> Husk, at det ikke er det umiddelbare resultat, der betyder noget, men snarere det endelige resultat, som vi vil opnå ved slutningen af simuleringen.

For at tage højde for denne forsinkede belønning skal vi bruge principperne for **[dynamisk programmering](https://en.wikipedia.org/wiki/Dynamic_programming)**, som giver os mulighed for at tænke på vores problem rekursivt.

Antag, at vi nu er i tilstanden *s*, og vi ønsker at bevæge os til den næste tilstand *s'*. Ved at gøre det vil vi modtage den umiddelbare belønning *r(s,a)*, defineret af belønningsfunktionen, plus en fremtidig belønning. Hvis vi antager, at vores Q-Table korrekt afspejler "attraktiviteten" af hver handling, vil vi i tilstanden *s'* vælge en handling *a*, der svarer til den maksimale værdi af *Q(s',a')*. Således vil den bedste mulige fremtidige belønning, vi kunne få i tilstanden *s*, være defineret som `max`

## Kontrol af politikken

Da Q-Tabellen viser "attraktiviteten" af hver handling i hver tilstand, er det ret nemt at bruge den til at definere effektiv navigation i vores verden. I det simpleste tilfælde kan vi vælge den handling, der svarer til den højeste Q-Table-værdi: (kodeblok 9)

```python
def qpolicy_strict(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = list(actions)[np.argmax(v)]
        return a

walk(m,qpolicy_strict)
```

> Hvis du prøver koden ovenfor flere gange, vil du måske bemærke, at den nogle gange "hænger", og du skal trykke på STOP-knappen i notebooken for at afbryde den. Dette sker, fordi der kan være situationer, hvor to tilstande "peger" på hinanden i forhold til optimal Q-værdi, hvilket får agenten til at bevæge sig mellem disse tilstande uendeligt.

## 🚀Udfordring

> **Opgave 1:** Modificer `walk`-funktionen til at begrænse den maksimale længde af stien til et bestemt antal trin (f.eks. 100), og se koden ovenfor returnere denne værdi fra tid til anden.

> **Opgave 2:** Modificer `walk`-funktionen, så den ikke vender tilbage til steder, hvor den allerede har været tidligere. Dette vil forhindre `walk` i at gentage sig selv, men agenten kan stadig ende med at være "fanget" på et sted, hvorfra den ikke kan undslippe.

## Navigation

En bedre navigationspolitik ville være den, vi brugte under træningen, som kombinerer udnyttelse og udforskning. I denne politik vil vi vælge hver handling med en vis sandsynlighed, proportional med værdierne i Q-Tabellen. Denne strategi kan stadig resultere i, at agenten vender tilbage til en position, den allerede har udforsket, men som du kan se fra koden nedenfor, resulterer den i en meget kort gennemsnitlig sti til den ønskede placering (husk, at `print_statistics` kører simuleringen 100 gange): (kodeblok 10)

```python
def qpolicy(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = random.choices(list(actions),weights=v)[0]
        return a

print_statistics(qpolicy)
```

Efter at have kørt denne kode, bør du få en meget mindre gennemsnitlig stiens længde end før, i området 3-6.

## Undersøgelse af læringsprocessen

Som vi har nævnt, er læringsprocessen en balance mellem udforskning og udnyttelse af den viden, der er opnået om problemrummets struktur. Vi har set, at resultaterne af læringen (evnen til at hjælpe en agent med at finde en kort sti til målet) er forbedret, men det er også interessant at observere, hvordan den gennemsnitlige stiens længde opfører sig under læringsprocessen:

Læringen kan opsummeres som:

- **Gennemsnitlig stiens længde stiger**. Det, vi ser her, er, at i starten stiger den gennemsnitlige stiens længde. Dette skyldes sandsynligvis, at når vi intet ved om miljøet, er vi tilbøjelige til at blive fanget i dårlige tilstande, vand eller ulv. Når vi lærer mere og begynder at bruge denne viden, kan vi udforske miljøet længere, men vi ved stadig ikke særlig godt, hvor æblerne er.

- **Stiens længde falder, efterhånden som vi lærer mere**. Når vi lærer nok, bliver det lettere for agenten at nå målet, og stiens længde begynder at falde. Vi er dog stadig åbne for udforskning, så vi afviger ofte fra den bedste sti og udforsker nye muligheder, hvilket gør stien længere end optimal.

- **Længden stiger pludseligt**. Det, vi også observerer på denne graf, er, at længden på et tidspunkt steg pludseligt. Dette indikerer den stokastiske natur af processen, og at vi på et tidspunkt kan "ødelægge" Q-Table-koefficienterne ved at overskrive dem med nye værdier. Dette bør ideelt set minimeres ved at reducere læringsraten (for eksempel mod slutningen af træningen justerer vi kun Q-Table-værdierne med en lille værdi).

Samlet set er det vigtigt at huske, at succes og kvaliteten af læringsprocessen afhænger betydeligt af parametre som læringsrate, læringsrate-nedgang og diskonteringsfaktor. Disse kaldes ofte **hyperparametre** for at skelne dem fra **parametre**, som vi optimerer under træningen (for eksempel Q-Table-koefficienter). Processen med at finde de bedste hyperparameterværdier kaldes **hyperparameteroptimering**, og det fortjener et separat emne.

## [Quiz efter forelæsning](https://ff-quizzes.netlify.app/en/ml/)

## Opgave 
[En Mere Realistisk Verden](assignment.md)

---

**Ansvarsfraskrivelse**:  
Dette dokument er blevet oversat ved hjælp af AI-oversættelsestjenesten [Co-op Translator](https://github.com/Azure/co-op-translator). Selvom vi bestræber os på nøjagtighed, skal du være opmærksom på, at automatiserede oversættelser kan indeholde fejl eller unøjagtigheder. Det originale dokument på dets oprindelige sprog bør betragtes som den autoritative kilde. For kritisk information anbefales professionel menneskelig oversættelse. Vi er ikke ansvarlige for eventuelle misforståelser eller fejltolkninger, der opstår som følge af brugen af denne oversættelse.