<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "911efd5e595089000cb3c16fce1beab8",
  "translation_date": "2025-09-05T22:05:43+00:00",
  "source_file": "8-Reinforcement/1-QLearning/README.md",
  "language_code": "no"
}
-->
# Introduksjon til forsterkende læring og Q-Læring

![Oppsummering av forsterkning i maskinlæring i en sketchnote](../../../../sketchnotes/ml-reinforcement.png)
> Sketchnote av [Tomomi Imura](https://www.twitter.com/girlie_mac)

Forsterkende læring involverer tre viktige konsepter: agenten, noen tilstander, og et sett med handlinger per tilstand. Ved å utføre en handling i en spesifisert tilstand, får agenten en belønning. Tenk igjen på dataspillet Super Mario. Du er Mario, du er i et spillnivå, stående ved kanten av en klippe. Over deg er det en mynt. Du som Mario, i et spillnivå, på en spesifikk posisjon ... det er din tilstand. Å ta et steg til høyre (en handling) vil føre deg over kanten, og det vil gi deg en lav numerisk score. Men å trykke på hopp-knappen vil gi deg et poeng og holde deg i live. Det er et positivt utfall og bør gi deg en positiv numerisk score.

Ved å bruke forsterkende læring og en simulator (spillet), kan du lære hvordan du spiller spillet for å maksimere belønningen, som er å holde seg i live og score så mange poeng som mulig.

[![Intro til forsterkende læring](https://img.youtube.com/vi/lDq_en8RNOo/0.jpg)](https://www.youtube.com/watch?v=lDq_en8RNOo)

> 🎥 Klikk på bildet over for å høre Dmitry diskutere forsterkende læring

## [Quiz før forelesning](https://ff-quizzes.netlify.app/en/ml/)

## Forutsetninger og oppsett

I denne leksjonen skal vi eksperimentere med litt kode i Python. Du bør kunne kjøre Jupyter Notebook-koden fra denne leksjonen, enten på din egen datamaskin eller i skyen.

Du kan åpne [notatboken for leksjonen](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/notebook.ipynb) og gå gjennom denne leksjonen for å bygge.

> **Merk:** Hvis du åpner denne koden fra skyen, må du også hente filen [`rlboard.py`](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/rlboard.py), som brukes i notatbok-koden. Legg den i samme katalog som notatboken.

## Introduksjon

I denne leksjonen skal vi utforske verdenen til **[Peter og ulven](https://en.wikipedia.org/wiki/Peter_and_the_Wolf)**, inspirert av et musikalsk eventyr av den russiske komponisten [Sergei Prokofiev](https://en.wikipedia.org/wiki/Sergei_Prokofiev). Vi skal bruke **forsterkende læring** for å la Peter utforske sitt miljø, samle smakfulle epler og unngå å møte ulven.

**Forsterkende læring** (RL) er en læringsteknikk som lar oss lære optimal oppførsel for en **agent** i et **miljø** ved å kjøre mange eksperimenter. En agent i dette miljøet bør ha et **mål**, definert av en **belønningsfunksjon**.

## Miljøet

For enkelhets skyld, la oss anta at Peters verden er et kvadratisk brett med størrelse `bredde` x `høyde`, som dette:

![Peters miljø](../../../../8-Reinforcement/1-QLearning/images/environment.png)

Hver celle i dette brettet kan enten være:

* **bakke**, som Peter og andre skapninger kan gå på.
* **vann**, som man åpenbart ikke kan gå på.
* et **tre** eller **gress**, et sted hvor man kan hvile.
* et **eple**, som representerer noe Peter vil være glad for å finne for å mate seg selv.
* en **ulv**, som er farlig og bør unngås.

Det finnes et eget Python-modul, [`rlboard.py`](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/rlboard.py), som inneholder koden for å arbeide med dette miljøet. Fordi denne koden ikke er viktig for å forstå konseptene våre, vil vi importere modulen og bruke den til å lage brettet (kodeblokk 1):

```python
from rlboard import *

width, height = 8,8
m = Board(width,height)
m.randomize(seed=13)
m.plot()
```

Denne koden bør skrive ut et bilde av miljøet som ligner på det ovenfor.

## Handlinger og strategi

I vårt eksempel vil Peters mål være å finne et eple, mens han unngår ulven og andre hindringer. For å gjøre dette kan han i hovedsak gå rundt til han finner et eple.

Derfor kan han på enhver posisjon velge mellom en av følgende handlinger: opp, ned, venstre og høyre.

Vi vil definere disse handlingene som et ordbok, og knytte dem til par av tilsvarende koordinatendringer. For eksempel vil det å bevege seg til høyre (`R`) tilsvare paret `(1,0)`. (kodeblokk 2):

```python
actions = { "U" : (0,-1), "D" : (0,1), "L" : (-1,0), "R" : (1,0) }
action_idx = { a : i for i,a in enumerate(actions.keys()) }
```

For å oppsummere, strategien og målet for dette scenariet er som følger:

- **Strategien**, for vår agent (Peter) er definert av en såkalt **policy**. En policy er en funksjon som returnerer handlingen i en gitt tilstand. I vårt tilfelle er tilstanden til problemet representert av brettet, inkludert spillerens nåværende posisjon.

- **Målet**, med forsterkende læring er å til slutt lære en god policy som lar oss løse problemet effektivt. Men som en baseline, la oss vurdere den enkleste policyen kalt **tilfeldig vandring**.

## Tilfeldig vandring

La oss først løse problemet vårt ved å implementere en strategi for tilfeldig vandring. Med tilfeldig vandring vil vi tilfeldig velge neste handling fra de tillatte handlingene, til vi når eplet (kodeblokk 3).

1. Implementer den tilfeldige vandringen med koden nedenfor:

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

    Kallet til `walk` bør returnere lengden på den tilsvarende stien, som kan variere fra en kjøring til en annen. 

1. Kjør vandringseksperimentet et antall ganger (si, 100), og skriv ut de resulterende statistikkene (kodeblokk 4):

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

    Merk at gjennomsnittlig lengde på en sti er rundt 30-40 steg, noe som er ganske mye, gitt at gjennomsnittlig avstand til nærmeste eple er rundt 5-6 steg.

    Du kan også se hvordan Peters bevegelser ser ut under den tilfeldige vandringen:

    ![Peters tilfeldige vandring](../../../../8-Reinforcement/1-QLearning/images/random_walk.gif)

## Belønningsfunksjon

For å gjøre vår policy mer intelligent, må vi forstå hvilke bevegelser som er "bedre" enn andre. For å gjøre dette må vi definere målet vårt.

Målet kan defineres i form av en **belønningsfunksjon**, som vil returnere en scoreverdi for hver tilstand. Jo høyere tallet er, jo bedre er belønningsfunksjonen. (kodeblokk 5)

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

En interessant ting med belønningsfunksjoner er at i de fleste tilfeller *får vi bare en betydelig belønning på slutten av spillet*. Dette betyr at algoritmen vår på en eller annen måte må huske "gode" steg som fører til en positiv belønning til slutt, og øke deres betydning. På samme måte bør alle bevegelser som fører til dårlige resultater avskrekkes.

## Q-Læring

En algoritme som vi skal diskutere her kalles **Q-Læring**. I denne algoritmen er policyen definert av en funksjon (eller en datastruktur) kalt en **Q-Tabell**. Den registrerer "godheten" til hver av handlingene i en gitt tilstand.

Den kalles en Q-Tabell fordi det ofte er praktisk å representere den som en tabell, eller en flerdimensjonal matrise. Siden brettet vårt har dimensjonene `bredde` x `høyde`, kan vi representere Q-Tabellen ved hjelp av en numpy-matrise med formen `bredde` x `høyde` x `len(actions)`: (kodeblokk 6)

```python
Q = np.ones((width,height,len(actions)),dtype=np.float)*1.0/len(actions)
```

Legg merke til at vi initialiserer alle verdiene i Q-Tabellen med en lik verdi, i vårt tilfelle - 0.25. Dette tilsvarer policyen "tilfeldig vandring", fordi alle bevegelser i hver tilstand er like gode. Vi kan sende Q-Tabellen til `plot`-funksjonen for å visualisere tabellen på brettet: `m.plot(Q)`.

![Peters miljø](../../../../8-Reinforcement/1-QLearning/images/env_init.png)

I midten av hver celle er det en "pil" som indikerer den foretrukne retningen for bevegelse. Siden alle retninger er like, vises en prikk.

Nå må vi kjøre simuleringen, utforske miljøet vårt, og lære en bedre fordeling av Q-Tabell-verdier, som vil tillate oss å finne veien til eplet mye raskere.

## Essensen av Q-Læring: Bellman-likningen

Når vi begynner å bevege oss, vil hver handling ha en tilsvarende belønning, dvs. vi kan teoretisk velge neste handling basert på den høyeste umiddelbare belønningen. Men i de fleste tilstander vil ikke bevegelsen oppnå målet vårt om å nå eplet, og dermed kan vi ikke umiddelbart avgjøre hvilken retning som er bedre.

> Husk at det ikke er det umiddelbare resultatet som betyr noe, men heller det endelige resultatet, som vi vil oppnå på slutten av simuleringen.

For å ta hensyn til denne forsinkede belønningen, må vi bruke prinsippene for **[dynamisk programmering](https://en.wikipedia.org/wiki/Dynamic_programming)**, som lar oss tenke på problemet vårt rekursivt.

Anta at vi nå er i tilstanden *s*, og vi ønsker å bevege oss til neste tilstand *s'*. Ved å gjøre det vil vi motta den umiddelbare belønningen *r(s,a)*, definert av belønningsfunksjonen, pluss en fremtidig belønning. Hvis vi antar at Q-Tabellen vår korrekt reflekterer "attraktiviteten" til hver handling, vil vi i tilstanden *s'* velge en handling *a* som tilsvarer maksimal verdi av *Q(s',a')*. Dermed vil den beste mulige fremtidige belønningen vi kunne få i tilstanden *s* bli definert som `max`

## Sjekke policyen

Siden Q-Tabellen viser "attraktiviteten" til hver handling i hver tilstand, er det ganske enkelt å bruke den til å definere effektiv navigering i vår verden. I det enkleste tilfellet kan vi velge handlingen som tilsvarer den høyeste verdien i Q-Tabellen: (kodeblokk 9)

```python
def qpolicy_strict(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = list(actions)[np.argmax(v)]
        return a

walk(m,qpolicy_strict)
```

> Hvis du prøver koden ovenfor flere ganger, kan du legge merke til at den noen ganger "henger", og du må trykke på STOP-knappen i notatboken for å avbryte den. Dette skjer fordi det kan oppstå situasjoner der to tilstander "peker" på hverandre når det gjelder optimal Q-verdi, i så fall ender agenten opp med å bevege seg mellom disse tilstandene uendelig.

## 🚀Utfordring

> **Oppgave 1:** Endre `walk`-funksjonen slik at den begrenser maksimal lengde på stien til et visst antall steg (for eksempel 100), og se koden ovenfor returnere denne verdien fra tid til annen.

> **Oppgave 2:** Endre `walk`-funksjonen slik at den ikke går tilbake til steder den allerede har vært tidligere. Dette vil forhindre at `walk` går i loop, men agenten kan fortsatt ende opp med å bli "fanget" på et sted den ikke kan unnslippe fra.

## Navigering

En bedre navigeringspolicy ville være den vi brukte under trening, som kombinerer utnyttelse og utforskning. I denne policyen vil vi velge hver handling med en viss sannsynlighet, proporsjonal med verdiene i Q-Tabellen. Denne strategien kan fortsatt føre til at agenten returnerer til en posisjon den allerede har utforsket, men som du kan se fra koden nedenfor, resulterer den i en veldig kort gjennomsnittlig sti til ønsket lokasjon (husk at `print_statistics` kjører simuleringen 100 ganger): (kodeblokk 10)

```python
def qpolicy(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = random.choices(list(actions),weights=v)[0]
        return a

print_statistics(qpolicy)
```

Etter å ha kjørt denne koden, bør du få en mye kortere gjennomsnittlig stilengde enn før, i området 3-6.

## Undersøke læringsprosessen

Som vi har nevnt, er læringsprosessen en balanse mellom utforskning og utnyttelse av opparbeidet kunnskap om problemrommets struktur. Vi har sett at resultatene av læringen (evnen til å hjelpe en agent med å finne en kort sti til målet) har forbedret seg, men det er også interessant å observere hvordan gjennomsnittlig stilengde oppfører seg under læringsprosessen:

## Oppsummering av læringen:

- **Gjennomsnittlig stilengde øker**. Det vi ser her er at i starten øker gjennomsnittlig stilengde. Dette skyldes sannsynligvis at når vi ikke vet noe om miljøet, er vi mer sannsynlig å bli fanget i dårlige tilstander, som vann eller ulv. Etter hvert som vi lærer mer og begynner å bruke denne kunnskapen, kan vi utforske miljøet lenger, men vi vet fortsatt ikke veldig godt hvor eplene er.

- **Stilengden avtar etter hvert som vi lærer mer**. Når vi har lært nok, blir det lettere for agenten å nå målet, og stilengden begynner å avta. Vi er imidlertid fortsatt åpne for utforskning, så vi avviker ofte fra den beste stien og utforsker nye alternativer, noe som gjør stien lengre enn optimal.

- **Lengden øker brått**. Det vi også observerer på denne grafen er at på et tidspunkt økte lengden brått. Dette indikerer den stokastiske naturen til prosessen, og at vi på et tidspunkt kan "ødelegge" Q-Tabell-koeffisientene ved å overskrive dem med nye verdier. Dette bør ideelt sett minimeres ved å redusere læringsraten (for eksempel mot slutten av treningen justerer vi Q-Tabell-verdiene med en liten verdi).

Alt i alt er det viktig å huske at suksessen og kvaliteten på læringsprosessen avhenger betydelig av parametere, som læringsrate, læringsrate-avtakelse og diskonteringsfaktor. Disse kalles ofte **hyperparametere**, for å skille dem fra **parametere**, som vi optimaliserer under trening (for eksempel Q-Tabell-koeffisienter). Prosessen med å finne de beste verdiene for hyperparametere kalles **hyperparameteroptimalisering**, og det fortjener et eget tema.

## [Quiz etter forelesning](https://ff-quizzes.netlify.app/en/ml/)

## Oppgave 
[En mer realistisk verden](assignment.md)

---

**Ansvarsfraskrivelse**:  
Dette dokumentet er oversatt ved hjelp av AI-oversettelsestjenesten [Co-op Translator](https://github.com/Azure/co-op-translator). Selv om vi streber etter nøyaktighet, vær oppmerksom på at automatiserte oversettelser kan inneholde feil eller unøyaktigheter. Det originale dokumentet på sitt opprinnelige språk bør anses som den autoritative kilden. For kritisk informasjon anbefales profesjonell menneskelig oversettelse. Vi er ikke ansvarlige for eventuelle misforståelser eller feiltolkninger som oppstår ved bruk av denne oversettelsen.