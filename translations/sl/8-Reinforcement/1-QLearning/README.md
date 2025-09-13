<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "911efd5e595089000cb3c16fce1beab8",
  "translation_date": "2025-09-05T13:37:15+00:00",
  "source_file": "8-Reinforcement/1-QLearning/README.md",
  "language_code": "sl"
}
-->
# Uvod v učenje z okrepitvijo in Q-učenje

![Povzetek učenja z okrepitvijo v strojnem učenju v obliki sketchnote](../../../../sketchnotes/ml-reinforcement.png)
> Sketchnote avtorja [Tomomi Imura](https://www.twitter.com/girlie_mac)

Učenje z okrepitvijo vključuje tri pomembne koncepte: agenta, določena stanja in niz dejanj za vsako stanje. Z izvajanjem dejanja v določenem stanju agent prejme nagrado. Predstavljajte si računalniško igro Super Mario. Vi ste Mario, nahajate se na ravni igre, stojite ob robu prepada. Nad vami je kovanec. Vi kot Mario, na določeni ravni igre, na določenem položaju ... to je vaše stanje. Če se premaknete korak v desno (dejanje), boste padli čez rob in prejeli nizko številčno oceno. Če pa pritisnete gumb za skok, boste dosegli točko in ostali živi. To je pozitiven izid, ki bi vam moral prinesti pozitivno številčno oceno.

Z uporabo učenja z okrepitvijo in simulatorja (igre) se lahko naučite igrati igro tako, da maksimizirate nagrado, kar pomeni, da ostanete živi in dosežete čim več točk.

[![Uvod v učenje z okrepitvijo](https://img.youtube.com/vi/lDq_en8RNOo/0.jpg)](https://www.youtube.com/watch?v=lDq_en8RNOo)

> 🎥 Kliknite zgornjo sliko, da poslušate Dmitryja, kako razpravlja o učenju z okrepitvijo

## [Pred-predavanje kviz](https://ff-quizzes.netlify.app/en/ml/)

## Predpogoji in nastavitev

V tej lekciji bomo eksperimentirali s kodo v Pythonu. Kodo iz Jupyter Notebooka iz te lekcije bi morali biti sposobni zagnati na svojem računalniku ali v oblaku.

Odprite [notebook lekcije](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/notebook.ipynb) in sledite lekciji za gradnjo.

> **Opomba:** Če odpirate to kodo iz oblaka, morate pridobiti tudi datoteko [`rlboard.py`](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/rlboard.py), ki se uporablja v kodi notebooka. Dodajte jo v isto mapo kot notebook.

## Uvod

V tej lekciji bomo raziskali svet **[Peter in volk](https://en.wikipedia.org/wiki/Peter_and_the_Wolf)**, ki ga je navdihnila glasbena pravljica ruskega skladatelja [Sergeja Prokofjeva](https://en.wikipedia.org/wiki/Sergei_Prokofiev). Uporabili bomo **učenje z okrepitvijo**, da bomo Petru omogočili raziskovanje njegovega okolja, zbiranje okusnih jabolk in izogibanje volku.

**Učenje z okrepitvijo** (RL) je tehnika učenja, ki nam omogoča, da se naučimo optimalnega vedenja **agenta** v določenem **okolju** z izvajanjem številnih eksperimentov. Agent v tem okolju mora imeti določen **cilj**, ki ga določa **funkcija nagrade**.

## Okolje

Za enostavnost si predstavljajmo Petrov svet kot kvadratno ploščo velikosti `širina` x `višina`, kot je ta:

![Petrovo okolje](../../../../8-Reinforcement/1-QLearning/images/environment.png)

Vsaka celica na tej plošči je lahko:

* **zemlja**, po kateri lahko Peter in druga bitja hodijo.
* **voda**, po kateri očitno ne morete hoditi.
* **drevo** ali **trava**, kjer se lahko spočijete.
* **jabolko**, ki predstavlja nekaj, kar bi Peter z veseljem našel, da se nahrani.
* **volk**, ki je nevaren in se mu je treba izogniti.

Obstaja ločen Python modul, [`rlboard.py`](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/rlboard.py), ki vsebuje kodo za delo s tem okoljem. Ker ta koda ni pomembna za razumevanje naših konceptov, bomo modul uvozili in ga uporabili za ustvarjanje vzorčne plošče (koda blok 1):

```python
from rlboard import *

width, height = 8,8
m = Board(width,height)
m.randomize(seed=13)
m.plot()
```

Ta koda bi morala natisniti sliko okolja, podobno zgornji.

## Dejanja in politika

V našem primeru bi bil Petrov cilj najti jabolko, medtem ko se izogiba volku in drugim oviram. Da bi to dosegel, se lahko preprosto sprehaja, dokler ne najde jabolka.

Zato lahko na katerem koli položaju izbere eno od naslednjih dejanj: gor, dol, levo in desno.

Ta dejanja bomo definirali kot slovar in jih preslikali v pare ustreznih sprememb koordinat. Na primer, premik v desno (`R`) bi ustrezal paru `(1,0)`. (koda blok 2):

```python
actions = { "U" : (0,-1), "D" : (0,1), "L" : (-1,0), "R" : (1,0) }
action_idx = { a : i for i,a in enumerate(actions.keys()) }
```

Če povzamemo, strategija in cilj tega scenarija sta naslednja:

- **Strategija** našega agenta (Petra) je definirana z tako imenovano **politiko**. Politika je funkcija, ki vrne dejanje v katerem koli danem stanju. V našem primeru je stanje problema predstavljeno s ploščo, vključno s trenutnim položajem igralca.

- **Cilj** učenja z okrepitvijo je sčasoma naučiti dobro politiko, ki nam bo omogočila učinkovito reševanje problema. Vendar pa kot osnovo upoštevajmo najpreprostejšo politiko, imenovano **naključna hoja**.

## Naključna hoja

Najprej rešimo naš problem z implementacijo strategije naključne hoje. Pri naključni hoji bomo naključno izbrali naslednje dejanje iz dovoljenih dejanj, dokler ne dosežemo jabolka (koda blok 3).

1. Implementirajte naključno hojo s spodnjo kodo:

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

    Klic funkcije `walk` bi moral vrniti dolžino ustrezne poti, ki se lahko razlikuje od enega zagona do drugega.

1. Izvedite eksperiment hoje večkrat (recimo 100-krat) in natisnite rezultate statistike (koda blok 4):

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

    Opazite, da je povprečna dolžina poti okoli 30-40 korakov, kar je precej, glede na to, da je povprečna razdalja do najbližjega jabolka okoli 5-6 korakov.

    Prav tako lahko vidite, kako izgleda Petrov premik med naključno hojo:

    ![Petrova naključna hoja](../../../../8-Reinforcement/1-QLearning/images/random_walk.gif)

## Funkcija nagrade

Da bi bila naša politika bolj inteligentna, moramo razumeti, katera dejanja so "boljša" od drugih. Za to moramo definirati naš cilj.

Cilj lahko definiramo v smislu **funkcije nagrade**, ki bo vrnila neko vrednost ocene za vsako stanje. Višja kot je številka, boljša je nagrada. (koda blok 5)

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

Zanimivo pri funkcijah nagrade je, da v večini primerov *prejmemo bistveno nagrado šele na koncu igre*. To pomeni, da mora naš algoritem nekako zapomniti "dobre" korake, ki vodijo do pozitivne nagrade na koncu, in povečati njihov pomen. Podobno je treba odvračati vse poteze, ki vodijo do slabih rezultatov.

## Q-učenje

Algoritem, ki ga bomo tukaj obravnavali, se imenuje **Q-učenje**. V tem algoritmu je politika definirana s funkcijo (ali podatkovno strukturo), imenovano **Q-tabela**. Ta beleži "dobroto" vsakega dejanja v določenem stanju.

Imenuje se Q-tabela, ker jo je pogosto priročno predstavljati kot tabelo ali večdimenzionalno matriko. Ker ima naša plošča dimenzije `širina` x `višina`, lahko Q-tabelo predstavimo z numpy matriko oblike `širina` x `višina` x `len(actions)`: (koda blok 6)

```python
Q = np.ones((width,height,len(actions)),dtype=np.float)*1.0/len(actions)
```

Opazite, da vse vrednosti Q-tabele inicializiramo z enako vrednostjo, v našem primeru - 0,25. To ustreza politiki "naključne hoje", ker so vse poteze v vsakem stanju enako dobre. Q-tabelo lahko posredujemo funkciji `plot`, da vizualiziramo tabelo na plošči: `m.plot(Q)`.

![Petrovo okolje](../../../../8-Reinforcement/1-QLearning/images/env_init.png)

V središču vsake celice je "puščica", ki označuje prednostno smer gibanja. Ker so vse smeri enake, je prikazana pika.

Zdaj moramo zagnati simulacijo, raziskati naše okolje in se naučiti boljše porazdelitve vrednosti Q-tabele, kar nam bo omogočilo veliko hitrejše iskanje poti do jabolka.

## Bistvo Q-učenja: Bellmanova enačba

Ko se začnemo premikati, bo vsako dejanje imelo ustrezno nagrado, tj. teoretično lahko izberemo naslednje dejanje na podlagi najvišje takojšnje nagrade. Vendar pa v večini stanj poteza ne bo dosegla našega cilja doseči jabolko, zato ne moremo takoj odločiti, katera smer je boljša.

> Ne pozabite, da ni pomemben trenutni rezultat, temveč končni rezultat, ki ga bomo dosegli na koncu simulacije.

Da bi upoštevali to odloženo nagrado, moramo uporabiti načela **[dinamičnega programiranja](https://en.wikipedia.org/wiki/Dynamic_programming)**, ki nam omogočajo, da o problemu razmišljamo rekurzivno.

Recimo, da smo zdaj v stanju *s* in se želimo premakniti v naslednje stanje *s'*. S tem bomo prejeli takojšnjo nagrado *r(s,a)*, določeno s funkcijo nagrade, plus neko prihodnjo nagrado. Če predpostavimo, da naša Q-tabela pravilno odraža "privlačnost" vsakega dejanja, bomo v stanju *s'* izbrali dejanje *a*, ki ustreza največji vrednosti *Q(s',a')*. Tako bo najboljša možna prihodnja nagrada, ki jo lahko dobimo v stanju *s*, definirana kot `max`

## Preverjanje politike

Ker Q-tabela prikazuje "privlačnost" vsakega dejanja v vsakem stanju, jo je precej enostavno uporabiti za določitev učinkovite navigacije v našem svetu. V najpreprostejšem primeru lahko izberemo dejanje, ki ustreza najvišji vrednosti v Q-tabeli: (koda 9)

```python
def qpolicy_strict(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = list(actions)[np.argmax(v)]
        return a

walk(m,qpolicy_strict)
```

> Če zgornjo kodo poskusite večkrat, boste morda opazili, da se včasih "zatakne" in morate pritisniti gumb STOP v beležnici, da jo prekinete. To se zgodi, ker lahko pride do situacij, ko si dve stanji "kažeta" druga na drugo glede na optimalno Q-vrednost, zaradi česar agent neskončno prehaja med tema dvema stanjema.

## 🚀Izziv

> **Naloga 1:** Spremenite funkcijo `walk`, da omejite največjo dolžino poti na določeno število korakov (na primer 100) in opazujte, kako zgornja koda občasno vrne to vrednost.

> **Naloga 2:** Spremenite funkcijo `walk`, da se ne vrača na mesta, kjer je že bil. To bo preprečilo, da bi se `walk` zanko ponavljal, vendar se agent še vedno lahko "ujame" na lokaciji, iz katere ne more pobegniti.

## Navigacija

Boljša navigacijska politika bi bila tista, ki smo jo uporabili med učenjem, in ki združuje izkoriščanje in raziskovanje. Pri tej politiki bomo vsako dejanje izbrali z določeno verjetnostjo, sorazmerno z vrednostmi v Q-tabeli. Ta strategija lahko še vedno povzroči, da se agent vrne na že raziskano mesto, vendar, kot lahko vidite iz spodnje kode, vodi do zelo kratke povprečne poti do želene lokacije (ne pozabite, da `print_statistics` simulacijo zažene 100-krat): (koda 10)

```python
def qpolicy(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = random.choices(list(actions),weights=v)[0]
        return a

print_statistics(qpolicy)
```

Po zagonu te kode bi morali dobiti precej krajšo povprečno dolžino poti kot prej, v razponu od 3 do 6.

## Preučevanje učnega procesa

Kot smo omenili, je učni proces ravnovesje med raziskovanjem in izkoriščanjem pridobljenega znanja o strukturi problemskega prostora. Videli smo, da so se rezultati učenja (sposobnost pomagati agentu najti kratko pot do cilja) izboljšali, vendar je zanimivo tudi opazovati, kako se povprečna dolžina poti spreminja med učnim procesom:

Ugotovitve lahko povzamemo takole:

- **Povprečna dolžina poti se poveča.** Na začetku opazimo, da se povprečna dolžina poti povečuje. To je verjetno zato, ker ko ne vemo ničesar o okolju, se zlahka ujamemo v slaba stanja, kot so voda ali volk. Ko se naučimo več in začnemo uporabljati to znanje, lahko okolje raziskujemo dlje, vendar še vedno ne vemo dobro, kje so jabolka.

- **Dolžina poti se zmanjša, ko se naučimo več.** Ko se naučimo dovolj, postane agentu lažje doseči cilj, dolžina poti pa se začne zmanjševati. Vendar smo še vedno odprti za raziskovanje, zato pogosto odstopimo od najboljše poti in raziskujemo nove možnosti, kar podaljša pot.

- **Dolžina se nenadoma poveča.** Na grafu opazimo tudi, da se dolžina na neki točki nenadoma poveča. To kaže na stohastično naravo procesa in na to, da lahko na neki točki "pokvarimo" koeficiente v Q-tabeli, tako da jih prepišemo z novimi vrednostmi. To bi morali idealno zmanjšati z zniževanjem učne stopnje (na primer, proti koncu učenja Q-tabeli dodajamo le majhne vrednosti).

Na splošno je pomembno vedeti, da uspeh in kakovost učnega procesa močno odvisna od parametrov, kot so učna stopnja, zmanjševanje učne stopnje in faktor diskontiranja. Ti se pogosto imenujejo **hiperparametri**, da jih ločimo od **parametrov**, ki jih optimiziramo med učenjem (na primer koeficienti Q-tabele). Proces iskanja najboljših vrednosti hiperparametrov se imenuje **optimizacija hiperparametrov** in si zasluži ločeno obravnavo.

## [Kvizi po predavanju](https://ff-quizzes.netlify.app/en/ml/)

## Naloga 
[Bolj realističen svet](assignment.md)

---

**Omejitev odgovornosti**:  
Ta dokument je bil preveden z uporabo storitve za prevajanje z umetno inteligenco [Co-op Translator](https://github.com/Azure/co-op-translator). Čeprav si prizadevamo za natančnost, vas prosimo, da upoštevate, da lahko avtomatizirani prevodi vsebujejo napake ali netočnosti. Izvirni dokument v njegovem maternem jeziku je treba obravnavati kot avtoritativni vir. Za ključne informacije priporočamo profesionalni človeški prevod. Ne prevzemamo odgovornosti za morebitne nesporazume ali napačne razlage, ki bi nastale zaradi uporabe tega prevoda.