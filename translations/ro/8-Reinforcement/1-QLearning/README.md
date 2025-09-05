<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "911efd5e595089000cb3c16fce1beab8",
  "translation_date": "2025-09-05T16:41:50+00:00",
  "source_file": "8-Reinforcement/1-QLearning/README.md",
  "language_code": "ro"
}
-->
# Introducere în Învățarea prin Recompensă și Q-Learning

![Rezumat al învățării prin recompensă în machine learning într-un sketchnote](../../../../sketchnotes/ml-reinforcement.png)
> Sketchnote de [Tomomi Imura](https://www.twitter.com/girlie_mac)

Învățarea prin recompensă implică trei concepte importante: agentul, câteva stări și un set de acțiuni pentru fiecare stare. Prin executarea unei acțiuni într-o stare specificată, agentul primește o recompensă. Imaginează din nou jocul pe calculator Super Mario. Tu ești Mario, te afli într-un nivel de joc, stând lângă marginea unei prăpăstii. Deasupra ta este o monedă. Tu, fiind Mario, într-un nivel de joc, într-o poziție specifică... aceasta este starea ta. Dacă faci un pas spre dreapta (o acțiune), vei cădea în prăpastie, ceea ce îți va aduce un scor numeric scăzut. Totuși, dacă apeși butonul de săritură, vei obține un punct și vei rămâne în viață. Acesta este un rezultat pozitiv și ar trebui să îți aducă un scor numeric pozitiv.

Folosind învățarea prin recompensă și un simulator (jocul), poți învăța cum să joci pentru a maximiza recompensa, adică să rămâi în viață și să obții cât mai multe puncte.

[![Introducere în Învățarea prin Recompensă](https://img.youtube.com/vi/lDq_en8RNOo/0.jpg)](https://www.youtube.com/watch?v=lDq_en8RNOo)

> 🎥 Click pe imaginea de mai sus pentru a-l asculta pe Dmitry discutând despre Învățarea prin Recompensă

## [Chestionar înainte de lecție](https://ff-quizzes.netlify.app/en/ml/)

## Cerințe preliminare și configurare

În această lecție, vom experimenta cu ceva cod în Python. Ar trebui să poți rula codul din Jupyter Notebook din această lecție, fie pe computerul tău, fie undeva în cloud.

Poți deschide [notebook-ul lecției](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/notebook.ipynb) și să parcurgi lecția pentru a construi.

> **Notă:** Dacă deschizi acest cod din cloud, trebuie să obții și fișierul [`rlboard.py`](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/rlboard.py), care este utilizat în codul notebook-ului. Adaugă-l în același director ca notebook-ul.

## Introducere

În această lecție, vom explora lumea **[Petru și Lupul](https://en.wikipedia.org/wiki/Peter_and_the_Wolf)**, inspirată de un basm muzical al compozitorului rus [Sergei Prokofiev](https://en.wikipedia.org/wiki/Sergei_Prokofiev). Vom folosi **Învățarea prin Recompensă** pentru a-l lăsa pe Petru să-și exploreze mediul, să colecteze mere gustoase și să evite întâlnirea cu lupul.

**Învățarea prin Recompensă** (RL) este o tehnică de învățare care ne permite să învățăm un comportament optim al unui **agent** într-un **mediu** prin efectuarea multor experimente. Un agent în acest mediu ar trebui să aibă un **scop**, definit de o **funcție de recompensă**.

## Mediul

Pentru simplitate, să considerăm lumea lui Petru ca o tablă pătrată de dimensiune `width` x `height`, ca aceasta:

![Mediul lui Petru](../../../../8-Reinforcement/1-QLearning/images/environment.png)

Fiecare celulă din această tablă poate fi:

* **teren**, pe care Petru și alte creaturi pot merge.
* **apă**, pe care evident nu poți merge.
* un **copac** sau **iarbă**, un loc unde te poți odihni.
* un **măr**, care reprezintă ceva ce Petru ar fi bucuros să găsească pentru a se hrăni.
* un **lup**, care este periculos și ar trebui evitat.

Există un modul Python separat, [`rlboard.py`](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/rlboard.py), care conține codul pentru a lucra cu acest mediu. Deoarece acest cod nu este important pentru înțelegerea conceptelor noastre, vom importa modulul și îl vom folosi pentru a crea tabla exemplu (bloc de cod 1):

```python
from rlboard import *

width, height = 8,8
m = Board(width,height)
m.randomize(seed=13)
m.plot()
```

Acest cod ar trebui să imprime o imagine a mediului similară cu cea de mai sus.

## Acțiuni și politică

În exemplul nostru, scopul lui Petru ar fi să găsească un măr, evitând lupul și alte obstacole. Pentru a face acest lucru, el poate, în esență, să se plimbe până găsește un măr.

Astfel, în orice poziție, el poate alege între una dintre următoarele acțiuni: sus, jos, stânga și dreapta.

Vom defini aceste acțiuni ca un dicționar și le vom mapa la perechi de modificări corespunzătoare ale coordonatelor. De exemplu, deplasarea spre dreapta (`R`) ar corespunde perechii `(1,0)`. (bloc de cod 2):

```python
actions = { "U" : (0,-1), "D" : (0,1), "L" : (-1,0), "R" : (1,0) }
action_idx = { a : i for i,a in enumerate(actions.keys()) }
```

Pentru a rezuma, strategia și scopul acestui scenariu sunt următoarele:

- **Strategia** agentului nostru (Petru) este definită de o așa-numită **politică**. O politică este o funcție care returnează acțiunea într-o stare dată. În cazul nostru, starea problemei este reprezentată de tablă, inclusiv poziția curentă a jucătorului.

- **Scopul** învățării prin recompensă este să învățăm în cele din urmă o politică bună care ne va permite să rezolvăm problema eficient. Totuși, ca bază, să considerăm cea mai simplă politică numită **plimbare aleatorie**.

## Plimbare aleatorie

Să rezolvăm mai întâi problema noastră implementând o strategie de plimbare aleatorie. Cu plimbarea aleatorie, vom alege aleator următoarea acțiune din acțiunile permise, până ajungem la măr (bloc de cod 3).

1. Implementează plimbarea aleatorie cu codul de mai jos:

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

    Apelul la `walk` ar trebui să returneze lungimea traseului corespunzător, care poate varia de la o rulare la alta.

1. Rulează experimentul de plimbare de mai multe ori (să zicem, 100) și imprimă statisticile rezultate (bloc de cod 4):

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

    Observă că lungimea medie a unui traseu este în jur de 30-40 de pași, ceea ce este destul de mult, având în vedere că distanța medie până la cel mai apropiat măr este în jur de 5-6 pași.

    Poți vedea și cum arată mișcarea lui Petru în timpul plimbării aleatorii:

    ![Plimbarea aleatorie a lui Petru](../../../../8-Reinforcement/1-QLearning/images/random_walk.gif)

## Funcția de recompensă

Pentru a face politica noastră mai inteligentă, trebuie să înțelegem care mișcări sunt "mai bune" decât altele. Pentru a face acest lucru, trebuie să definim scopul nostru.

Scopul poate fi definit în termeni de o **funcție de recompensă**, care va returna o valoare de scor pentru fiecare stare. Cu cât numărul este mai mare, cu atât funcția de recompensă este mai bună. (bloc de cod 5)

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

Un lucru interesant despre funcțiile de recompensă este că, în cele mai multe cazuri, *primim o recompensă substanțială doar la sfârșitul jocului*. Aceasta înseamnă că algoritmul nostru ar trebui cumva să-și amintească "pașii buni" care duc la o recompensă pozitivă la final și să le crească importanța. În mod similar, toate mișcările care duc la rezultate proaste ar trebui descurajate.

## Q-Learning

Un algoritm pe care îl vom discuta aici se numește **Q-Learning**. În acest algoritm, politica este definită de o funcție (sau o structură de date) numită **Q-Table**. Aceasta înregistrează "calitatea" fiecărei acțiuni într-o stare dată.

Se numește Q-Table deoarece este adesea convenabil să o reprezentăm ca o tabelă sau un array multidimensional. Deoarece tabla noastră are dimensiuni `width` x `height`, putem reprezenta Q-Table folosind un array numpy cu forma `width` x `height` x `len(actions)`: (bloc de cod 6)

```python
Q = np.ones((width,height,len(actions)),dtype=np.float)*1.0/len(actions)
```

Observă că inițializăm toate valorile din Q-Table cu o valoare egală, în cazul nostru - 0.25. Aceasta corespunde politicii de "plimbare aleatorie", deoarece toate mișcările din fiecare stare sunt la fel de bune. Putem transmite Q-Table funcției `plot` pentru a vizualiza tabela pe tablă: `m.plot(Q)`.

![Mediul lui Petru](../../../../8-Reinforcement/1-QLearning/images/env_init.png)

În centrul fiecărei celule există o "săgeată" care indică direcția preferată de mișcare. Deoarece toate direcțiile sunt egale, se afișează un punct.

Acum trebuie să rulăm simularea, să explorăm mediul nostru și să învățăm o distribuție mai bună a valorilor din Q-Table, care ne va permite să găsim drumul către măr mult mai rapid.

## Esența Q-Learning: Ecuația Bellman

Odată ce începem să ne mișcăm, fiecare acțiune va avea o recompensă corespunzătoare, adică teoretic putem selecta următoarea acțiune pe baza celei mai mari recompense imediate. Totuși, în cele mai multe stări, mișcarea nu va atinge scopul nostru de a ajunge la măr și, astfel, nu putem decide imediat care direcție este mai bună.

> Amintește-ți că nu rezultatul imediat contează, ci mai degrabă rezultatul final, pe care îl vom obține la sfârșitul simulării.

Pentru a ține cont de această recompensă întârziată, trebuie să folosim principiile **[programării dinamice](https://en.wikipedia.org/wiki/Dynamic_programming)**, care ne permit să gândim problema noastră recursiv.

Să presupunem că acum ne aflăm în starea *s* și vrem să ne mutăm în următoarea stare *s'*. Prin aceasta, vom primi recompensa imediată *r(s,a)*, definită de funcția de recompensă, plus o recompensă viitoare. Dacă presupunem că Q-Table reflectă corect "atractivitatea" fiecărei acțiuni, atunci în starea *s'* vom alege o acțiune *a'* care corespunde valorii maxime a *Q(s',a')*. Astfel, cea mai bună recompensă viitoare posibilă pe care am putea să o obținem în starea *s* va fi definită ca `max`

## Verificarea politicii

Deoarece Q-Table listează "atractivitatea" fiecărei acțiuni în fiecare stare, este destul de simplu să o folosim pentru a defini navigarea eficientă în lumea noastră. În cel mai simplu caz, putem selecta acțiunea corespunzătoare valorii maxime din Q-Table: (bloc de cod 9)

```python
def qpolicy_strict(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = list(actions)[np.argmax(v)]
        return a

walk(m,qpolicy_strict)
```

> Dacă încercați codul de mai sus de mai multe ori, este posibil să observați că uneori "se blochează" și trebuie să apăsați butonul STOP din notebook pentru a-l întrerupe. Acest lucru se întâmplă deoarece pot exista situații în care două stări "indică" una către cealaltă în termeni de valoare optimă Q, caz în care agentul ajunge să se miște între aceste stări la nesfârșit.

## 🚀Provocare

> **Sarcina 1:** Modificați funcția `walk` pentru a limita lungimea maximă a traseului la un anumit număr de pași (de exemplu, 100) și observați cum codul de mai sus returnează această valoare din când în când.

> **Sarcina 2:** Modificați funcția `walk` astfel încât să nu se întoarcă în locurile în care a fost deja anterior. Acest lucru va preveni ca `walk` să intre într-un ciclu, însă agentul poate ajunge totuși să fie "blocat" într-o locație din care nu poate scăpa.

## Navigare

O politică de navigare mai bună ar fi cea pe care am folosit-o în timpul antrenamentului, care combină exploatarea și explorarea. În această politică, vom selecta fiecare acțiune cu o anumită probabilitate, proporțională cu valorile din Q-Table. Această strategie poate duce totuși la revenirea agentului într-o poziție pe care a explorat-o deja, dar, după cum puteți vedea din codul de mai jos, rezultă într-un traseu mediu foarte scurt către locația dorită (amintiți-vă că `print_statistics` rulează simularea de 100 de ori): (bloc de cod 10)

```python
def qpolicy(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = random.choices(list(actions),weights=v)[0]
        return a

print_statistics(qpolicy)
```

După rularea acestui cod, ar trebui să obțineți o lungime medie a traseului mult mai mică decât înainte, în intervalul 3-6.

## Investigarea procesului de învățare

Așa cum am menționat, procesul de învățare este un echilibru între explorarea și exploatarea cunoștințelor dobândite despre structura spațiului problemei. Am observat că rezultatele învățării (capacitatea de a ajuta un agent să găsească un traseu scurt către obiectiv) s-au îmbunătățit, dar este, de asemenea, interesant să observăm cum se comportă lungimea medie a traseului în timpul procesului de învățare:

## Rezumarea învățării

- **Lungimea medie a traseului crește**. Ce vedem aici este că, la început, lungimea medie a traseului crește. Acest lucru se datorează probabil faptului că, atunci când nu știm nimic despre mediu, avem tendința să fim prinși în stări nefavorabile, cum ar fi apă sau lup. Pe măsură ce învățăm mai multe și începem să folosim aceste cunoștințe, putem explora mediul mai mult, dar încă nu știm foarte bine unde sunt merele.

- **Lungimea traseului scade, pe măsură ce învățăm mai mult**. Odată ce învățăm suficient, devine mai ușor pentru agent să atingă obiectivul, iar lungimea traseului începe să scadă. Totuși, suntem încă deschiși la explorare, așa că deseori ne abatem de la cel mai bun traseu și explorăm opțiuni noi, ceea ce face traseul mai lung decât optimul.

- **Creștere bruscă a lungimii**. Ce mai observăm pe acest grafic este că, la un moment dat, lungimea a crescut brusc. Acest lucru indică natura stochastică a procesului și faptul că, la un moment dat, putem "strica" coeficienții din Q-Table prin suprascrierea lor cu valori noi. Acest lucru ar trebui minimizat ideal prin reducerea ratei de învățare (de exemplu, spre sfârșitul antrenamentului, ajustăm valorile din Q-Table doar cu o valoare mică).

În general, este important să ne amintim că succesul și calitatea procesului de învățare depind semnificativ de parametri, cum ar fi rata de învățare, reducerea ratei de învățare și factorul de discount. Aceștia sunt adesea numiți **hiperparametri**, pentru a-i distinge de **parametri**, pe care îi optimizăm în timpul antrenamentului (de exemplu, coeficienții din Q-Table). Procesul de găsire a celor mai bune valori pentru hiperparametri se numește **optimizarea hiperparametrilor** și merită o discuție separată.

## [Test de verificare post-lectură](https://ff-quizzes.netlify.app/en/ml/)

## Temă 
[O lume mai realistă](assignment.md)

---

**Declinare de responsabilitate**:  
Acest document a fost tradus folosind serviciul de traducere AI [Co-op Translator](https://github.com/Azure/co-op-translator). Deși ne străduim să asigurăm acuratețea, vă rugăm să fiți conștienți că traducerile automate pot conține erori sau inexactități. Documentul original în limba sa natală ar trebui considerat sursa autoritară. Pentru informații critice, se recomandă traducerea profesională realizată de un specialist uman. Nu ne asumăm responsabilitatea pentru eventualele neînțelegeri sau interpretări greșite care pot apărea din utilizarea acestei traduceri.