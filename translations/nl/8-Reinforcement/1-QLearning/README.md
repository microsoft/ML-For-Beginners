<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "911efd5e595089000cb3c16fce1beab8",
  "translation_date": "2025-09-05T20:11:11+00:00",
  "source_file": "8-Reinforcement/1-QLearning/README.md",
  "language_code": "nl"
}
-->
# Introductie tot Reinforcement Learning en Q-Learning

![Samenvatting van reinforcement in machine learning in een sketchnote](../../../../sketchnotes/ml-reinforcement.png)
> Sketchnote door [Tomomi Imura](https://www.twitter.com/girlie_mac)

Reinforcement learning draait om drie belangrijke concepten: de agent, enkele toestanden, en een set acties per toestand. Door een actie uit te voeren in een bepaalde toestand, krijgt de agent een beloning. Stel je opnieuw het computerspel Super Mario voor. Jij bent Mario, je bevindt je in een level van het spel, naast een klifrand. Boven je hangt een munt. Jij als Mario, in een level, op een specifieke positie ... dat is jouw toestand. Eén stap naar rechts zetten (een actie) brengt je over de rand, wat je een lage numerieke score oplevert. Maar als je op de springknop drukt, scoor je een punt en blijf je in leven. Dat is een positieve uitkomst en zou je een positieve numerieke score moeten opleveren.

Met behulp van reinforcement learning en een simulator (het spel) kun je leren hoe je het spel speelt om de beloning te maximaliseren, namelijk in leven blijven en zoveel mogelijk punten scoren.

[![Introductie tot Reinforcement Learning](https://img.youtube.com/vi/lDq_en8RNOo/0.jpg)](https://www.youtube.com/watch?v=lDq_en8RNOo)

> 🎥 Klik op de afbeelding hierboven om Dmitry te horen praten over Reinforcement Learning

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Vereisten en Setup

In deze les gaan we experimenteren met wat code in Python. Je moet in staat zijn om de Jupyter Notebook-code uit deze les uit te voeren, ofwel op je computer of ergens in de cloud.

Je kunt [het lesnotebook](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/notebook.ipynb) openen en door deze les lopen om het op te bouwen.

> **Opmerking:** Als je deze code vanuit de cloud opent, moet je ook het bestand [`rlboard.py`](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/rlboard.py) ophalen, dat wordt gebruikt in de notebook-code. Voeg het toe aan dezelfde map als het notebook.

## Introductie

In deze les verkennen we de wereld van **[Peter en de Wolf](https://en.wikipedia.org/wiki/Peter_and_the_Wolf)**, geïnspireerd door een muzikaal sprookje van de Russische componist [Sergei Prokofiev](https://en.wikipedia.org/wiki/Sergei_Prokofiev). We gebruiken **Reinforcement Learning** om Peter zijn omgeving te laten verkennen, smakelijke appels te verzamelen en de wolf te vermijden.

**Reinforcement Learning** (RL) is een leertechniek waarmee we het optimale gedrag van een **agent** in een bepaalde **omgeving** kunnen leren door veel experimenten uit te voeren. Een agent in deze omgeving moet een **doel** hebben, gedefinieerd door een **beloningsfunctie**.

## De omgeving

Voor de eenvoud beschouwen we Peters wereld als een vierkant bord van grootte `breedte` x `hoogte`, zoals dit:

![Peters Omgeving](../../../../8-Reinforcement/1-QLearning/images/environment.png)

Elke cel op dit bord kan zijn:

* **grond**, waarop Peter en andere wezens kunnen lopen.
* **water**, waarop je uiteraard niet kunt lopen.
* een **boom** of **gras**, een plek waar je kunt uitrusten.
* een **appel**, iets wat Peter graag zou vinden om zichzelf te voeden.
* een **wolf**, die gevaarlijk is en vermeden moet worden.

Er is een apart Python-module, [`rlboard.py`](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/rlboard.py), die de code bevat om met deze omgeving te werken. Omdat deze code niet belangrijk is voor het begrijpen van onze concepten, importeren we de module en gebruiken we deze om het voorbeeldbord te maken (codeblok 1):

```python
from rlboard import *

width, height = 8,8
m = Board(width,height)
m.randomize(seed=13)
m.plot()
```

Deze code zou een afbeelding van de omgeving moeten afdrukken die lijkt op de bovenstaande.

## Acties en beleid

In ons voorbeeld is Peters doel om een appel te vinden, terwijl hij de wolf en andere obstakels vermijdt. Om dit te doen, kan hij in principe rondlopen totdat hij een appel vindt.

Daarom kan hij op elke positie kiezen uit een van de volgende acties: omhoog, omlaag, links en rechts.

We definiëren die acties als een dictionary en koppelen ze aan paren van bijbehorende coördinatenwijzigingen. Bijvoorbeeld, naar rechts bewegen (`R`) zou overeenkomen met een paar `(1,0)`. (codeblok 2):

```python
actions = { "U" : (0,-1), "D" : (0,1), "L" : (-1,0), "R" : (1,0) }
action_idx = { a : i for i,a in enumerate(actions.keys()) }
```

Samenvattend zijn de strategie en het doel van dit scenario als volgt:

- **De strategie**, van onze agent (Peter) wordt gedefinieerd door een zogenaamde **policy**. Een policy is een functie die de actie retourneert in een bepaalde toestand. In ons geval wordt de toestand van het probleem weergegeven door het bord, inclusief de huidige positie van de speler.

- **Het doel**, van reinforcement learning is uiteindelijk een goed beleid te leren dat ons in staat stelt het probleem efficiënt op te lossen. Als basislijn beschouwen we echter het eenvoudigste beleid, genaamd **random walk**.

## Random walk

Laten we eerst ons probleem oplossen door een random walk-strategie te implementeren. Bij random walk kiezen we willekeurig de volgende actie uit de toegestane acties, totdat we de appel bereiken (codeblok 3).

1. Implementeer de random walk met de onderstaande code:

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

    De oproep naar `walk` zou de lengte van het bijbehorende pad moeten retourneren, wat kan variëren van de ene run tot de andere. 

1. Voer het walk-experiment een aantal keren uit (bijvoorbeeld 100) en druk de resulterende statistieken af (codeblok 4):

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

    Merk op dat de gemiddelde lengte van een pad ongeveer 30-40 stappen is, wat behoorlijk veel is, gezien het feit dat de gemiddelde afstand tot de dichtstbijzijnde appel ongeveer 5-6 stappen is.

    Je kunt ook zien hoe Peters beweging eruitziet tijdens de random walk:

    ![Peters Random Walk](../../../../8-Reinforcement/1-QLearning/images/random_walk.gif)

## Beloningsfunctie

Om ons beleid intelligenter te maken, moeten we begrijpen welke bewegingen "beter" zijn dan andere. Om dit te doen, moeten we ons doel definiëren.

Het doel kan worden gedefinieerd in termen van een **beloningsfunctie**, die een scorewaarde retourneert voor elke toestand. Hoe hoger het getal, hoe beter de beloningsfunctie. (codeblok 5)

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

Een interessant aspect van beloningsfuncties is dat in de meeste gevallen *we alleen een substantiële beloning krijgen aan het einde van het spel*. Dit betekent dat ons algoritme op de een of andere manier "goede" stappen moet onthouden die leiden tot een positieve beloning aan het einde, en hun belang moet vergroten. Evenzo moeten alle bewegingen die tot slechte resultaten leiden worden ontmoedigd.

## Q-Learning

Een algoritme dat we hier zullen bespreken, heet **Q-Learning**. In dit algoritme wordt het beleid gedefinieerd door een functie (of een datastructuur) genaamd een **Q-Table**. Het registreert de "kwaliteit" van elke actie in een bepaalde toestand.

Het wordt een Q-Table genoemd omdat het vaak handig is om het weer te geven als een tabel of een multidimensionale array. Omdat ons bord afmetingen heeft van `breedte` x `hoogte`, kunnen we de Q-Table weergeven met behulp van een numpy-array met vorm `breedte` x `hoogte` x `len(actions)`: (codeblok 6)

```python
Q = np.ones((width,height,len(actions)),dtype=np.float)*1.0/len(actions)
```

Merk op dat we alle waarden van de Q-Table initialiseren met een gelijke waarde, in ons geval - 0.25. Dit komt overeen met het "random walk"-beleid, omdat alle bewegingen in elke toestand even goed zijn. We kunnen de Q-Table doorgeven aan de `plot`-functie om de tabel op het bord te visualiseren: `m.plot(Q)`.

![Peters Omgeving](../../../../8-Reinforcement/1-QLearning/images/env_init.png)

In het midden van elke cel staat een "pijl" die de voorkeursrichting van beweging aangeeft. Omdat alle richtingen gelijk zijn, wordt een stip weergegeven.

Nu moeten we de simulatie uitvoeren, onze omgeving verkennen en een betere verdeling van Q-Table-waarden leren, waarmee we veel sneller het pad naar de appel kunnen vinden.

## Essentie van Q-Learning: Bellman-vergelijking

Zodra we beginnen te bewegen, heeft elke actie een bijbehorende beloning, d.w.z. we kunnen theoretisch de volgende actie selecteren op basis van de hoogste directe beloning. Echter, in de meeste toestanden zal de beweging ons doel om de appel te bereiken niet bereiken, en dus kunnen we niet onmiddellijk beslissen welke richting beter is.

> Onthoud dat het niet het directe resultaat is dat telt, maar eerder het uiteindelijke resultaat, dat we aan het einde van de simulatie zullen verkrijgen.

Om rekening te houden met deze vertraagde beloning, moeten we de principes van **[dynamisch programmeren](https://en.wikipedia.org/wiki/Dynamic_programming)** gebruiken, waarmee we ons probleem recursief kunnen benaderen.

Stel dat we nu in toestand *s* zijn en we willen naar de volgende toestand *s'* gaan. Door dit te doen, ontvangen we de directe beloning *r(s,a)*, gedefinieerd door de beloningsfunctie, plus een toekomstige beloning. Als we aannemen dat onze Q-Table correct de "aantrekkelijkheid" van elke actie weergeeft, dan zullen we in toestand *s'* een actie *a* kiezen die overeenkomt met de maximale waarde van *Q(s',a')*. Dus de best mogelijke toekomstige beloning die we in toestand *s* zouden kunnen krijgen, wordt gedefinieerd als `max`

## Controle van het beleid

Aangezien de Q-Table de "aantrekkelijkheid" van elke actie in elke staat weergeeft, is het vrij eenvoudig om deze te gebruiken om efficiënte navigatie in onze wereld te definiëren. In het eenvoudigste geval kunnen we de actie selecteren die overeenkomt met de hoogste waarde in de Q-Table: (code block 9)

```python
def qpolicy_strict(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = list(actions)[np.argmax(v)]
        return a

walk(m,qpolicy_strict)
```

> Als je de bovenstaande code meerdere keren probeert, merk je misschien dat deze soms "vastloopt" en dat je op de STOP-knop in het notebook moet drukken om het te onderbreken. Dit gebeurt omdat er situaties kunnen zijn waarin twee staten elkaar "aanwijzen" in termen van optimale Q-waarde, waardoor de agent eindeloos tussen die staten blijft bewegen.

## 🚀Uitdaging

> **Taak 1:** Pas de `walk`-functie aan om de maximale lengte van het pad te beperken tot een bepaald aantal stappen (bijvoorbeeld 100), en kijk hoe de bovenstaande code deze waarde van tijd tot tijd retourneert.

> **Taak 2:** Pas de `walk`-functie aan zodat deze niet terugkeert naar plaatsen waar hij eerder is geweest. Dit voorkomt dat `walk` in een lus terechtkomt, maar de agent kan nog steeds "vast" komen te zitten op een locatie waaruit hij niet kan ontsnappen.

## Navigatie

Een betere navigatiebeleid zou het beleid zijn dat we tijdens de training hebben gebruikt, dat exploitatie en exploratie combineert. In dit beleid selecteren we elke actie met een bepaalde waarschijnlijkheid, evenredig aan de waarden in de Q-Table. Deze strategie kan er nog steeds toe leiden dat de agent terugkeert naar een positie die hij al heeft verkend, maar zoals je kunt zien in de onderstaande code, resulteert dit in een zeer kort gemiddeld pad naar de gewenste locatie (onthoud dat `print_statistics` de simulatie 100 keer uitvoert): (code block 10)

```python
def qpolicy(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = random.choices(list(actions),weights=v)[0]
        return a

print_statistics(qpolicy)
```

Na het uitvoeren van deze code zou je een veel kleinere gemiddelde padlengte moeten krijgen dan voorheen, in de orde van 3-6.

## Onderzoek naar het leerproces

Zoals we hebben vermeld, is het leerproces een balans tussen exploratie en het benutten van opgedane kennis over de structuur van de probleemruimte. We hebben gezien dat de resultaten van het leren (het vermogen om een agent te helpen een kort pad naar het doel te vinden) zijn verbeterd, maar het is ook interessant om te observeren hoe de gemiddelde padlengte zich gedraagt tijdens het leerproces:

De leerresultaten kunnen als volgt worden samengevat:

- **Gemiddelde padlengte neemt toe**. Wat we hier zien, is dat in het begin de gemiddelde padlengte toeneemt. Dit komt waarschijnlijk doordat we, wanneer we niets weten over de omgeving, geneigd zijn vast te lopen in slechte staten, zoals water of een wolf. Naarmate we meer leren en deze kennis beginnen te gebruiken, kunnen we de omgeving langer verkennen, maar we weten nog steeds niet goed waar de appels zijn.

- **Padlengte neemt af naarmate we meer leren**. Zodra we genoeg leren, wordt het gemakkelijker voor de agent om het doel te bereiken, en de padlengte begint af te nemen. We staan echter nog steeds open voor exploratie, dus we wijken vaak af van het beste pad en verkennen nieuwe opties, waardoor het pad langer wordt dan optimaal.

- **Lengte neemt abrupt toe**. Wat we ook op deze grafiek zien, is dat op een bepaald moment de lengte abrupt toenam. Dit geeft de stochastische aard van het proces aan, en dat we op een bepaald moment de Q-Table-coëfficiënten kunnen "bederven" door ze te overschrijven met nieuwe waarden. Dit zou idealiter moeten worden geminimaliseerd door de leersnelheid te verlagen (bijvoorbeeld tegen het einde van de training passen we de Q-Table-waarden slechts met een kleine waarde aan).

Over het algemeen is het belangrijk om te onthouden dat het succes en de kwaliteit van het leerproces sterk afhankelijk zijn van parameters, zoals leersnelheid, afname van de leersnelheid en de discontofactor. Deze worden vaak **hyperparameters** genoemd, om ze te onderscheiden van **parameters**, die we optimaliseren tijdens de training (bijvoorbeeld Q-Table-coëfficiënten). Het proces van het vinden van de beste hyperparameterwaarden wordt **hyperparameteroptimalisatie** genoemd, en dit verdient een apart onderwerp.

## [Quiz na de les](https://ff-quizzes.netlify.app/en/ml/)

## Opdracht 
[Een Meer Realistische Wereld](assignment.md)

---

**Disclaimer**:  
Dit document is vertaald met behulp van de AI-vertalingsservice [Co-op Translator](https://github.com/Azure/co-op-translator). Hoewel we ons best doen om nauwkeurigheid te garanderen, dient u zich ervan bewust te zijn dat geautomatiseerde vertalingen fouten of onnauwkeurigheden kunnen bevatten. Het originele document in zijn oorspronkelijke taal moet worden beschouwd als de gezaghebbende bron. Voor kritieke informatie wordt professionele menselijke vertaling aanbevolen. Wij zijn niet aansprakelijk voor misverstanden of verkeerde interpretaties die voortvloeien uit het gebruik van deze vertaling.