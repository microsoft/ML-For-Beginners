<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "107d5bb29da8a562e7ae72262d251a75",
  "translation_date": "2025-09-05T16:47:06+00:00",
  "source_file": "8-Reinforcement/2-Gym/README.md",
  "language_code": "ro"
}
-->
## Cerințe preliminare

În această lecție, vom folosi o bibliotecă numită **OpenAI Gym** pentru a simula diferite **medii**. Poți rula codul lecției local (de exemplu, din Visual Studio Code), caz în care simularea se va deschide într-o fereastră nouă. Dacă rulezi codul online, poate fi necesar să faci unele ajustări, așa cum este descris [aici](https://towardsdatascience.com/rendering-openai-gym-envs-on-binder-and-google-colab-536f99391cc7).

## OpenAI Gym

În lecția anterioară, regulile jocului și starea erau definite de clasa `Board`, pe care am creat-o noi. Aici vom folosi un **mediu de simulare** special, care va simula fizica din spatele balansării unui stâlp. Unul dintre cele mai populare medii de simulare pentru antrenarea algoritmilor de învățare prin întărire se numește [Gym](https://gym.openai.com/), care este întreținut de [OpenAI](https://openai.com/). Folosind acest Gym, putem crea diverse **medii**, de la simulări de tip cartpole până la jocuri Atari.

> **Notă**: Poți vedea alte medii disponibile în OpenAI Gym [aici](https://gym.openai.com/envs/#classic_control).

Mai întâi, să instalăm Gym și să importăm bibliotecile necesare (bloc de cod 1):

```python
import sys
!{sys.executable} -m pip install gym 

import gym
import matplotlib.pyplot as plt
import numpy as np
import random
```

## Exercițiu - inițializează un mediu cartpole

Pentru a lucra cu problema balansării cartpole, trebuie să inițializăm mediul corespunzător. Fiecare mediu este asociat cu:

- **Spațiul de observație**, care definește structura informațiilor pe care le primim de la mediu. Pentru problema cartpole, primim poziția stâlpului, viteza și alte valori.

- **Spațiul de acțiune**, care definește acțiunile posibile. În cazul nostru, spațiul de acțiune este discret și constă din două acțiuni - **stânga** și **dreapta**. (bloc de cod 2)

1. Pentru a inițializa, tastează următorul cod:

    ```python
    env = gym.make("CartPole-v1")
    print(env.action_space)
    print(env.observation_space)
    print(env.action_space.sample())
    ```

Pentru a vedea cum funcționează mediul, să rulăm o simulare scurtă de 100 de pași. La fiecare pas, oferim una dintre acțiunile care trebuie luate - în această simulare selectăm aleatoriu o acțiune din `action_space`.

1. Rulează codul de mai jos și vezi ce rezultă.

    ✅ Amintește-ți că este preferabil să rulezi acest cod pe o instalare locală de Python! (bloc de cod 3)

    ```python
    env.reset()
    
    for i in range(100):
       env.render()
       env.step(env.action_space.sample())
    env.close()
    ```

    Ar trebui să vezi ceva similar cu această imagine:

    ![cartpole fără balans](../../../../8-Reinforcement/2-Gym/images/cartpole-nobalance.gif)

1. În timpul simulării, trebuie să obținem observații pentru a decide cum să acționăm. De fapt, funcția step returnează observațiile curente, o funcție de recompensă și un indicator `done` care arată dacă are sens să continuăm simularea sau nu: (bloc de cod 4)

    ```python
    env.reset()
    
    done = False
    while not done:
       env.render()
       obs, rew, done, info = env.step(env.action_space.sample())
       print(f"{obs} -> {rew}")
    env.close()
    ```

    Vei vedea ceva asemănător în output-ul notebook-ului:

    ```text
    [ 0.03403272 -0.24301182  0.02669811  0.2895829 ] -> 1.0
    [ 0.02917248 -0.04828055  0.03248977  0.00543839] -> 1.0
    [ 0.02820687  0.14636075  0.03259854 -0.27681916] -> 1.0
    [ 0.03113408  0.34100283  0.02706215 -0.55904489] -> 1.0
    [ 0.03795414  0.53573468  0.01588125 -0.84308041] -> 1.0
    ...
    [ 0.17299878  0.15868546 -0.20754175 -0.55975453] -> 1.0
    [ 0.17617249  0.35602306 -0.21873684 -0.90998894] -> 1.0
    ```

    Vectorul de observație returnat la fiecare pas al simulării conține următoarele valori:
    - Poziția căruciorului
    - Viteza căruciorului
    - Unghiul stâlpului
    - Rata de rotație a stâlpului

1. Obține valoarea minimă și maximă a acestor numere: (bloc de cod 5)

    ```python
    print(env.observation_space.low)
    print(env.observation_space.high)
    ```

    De asemenea, vei observa că valoarea recompensei la fiecare pas al simulării este întotdeauna 1. Acest lucru se datorează faptului că scopul nostru este să supraviețuim cât mai mult timp posibil, adică să menținem stâlpul într-o poziție rezonabil verticală pentru cea mai lungă perioadă de timp.

    ✅ De fapt, simularea CartPole este considerată rezolvată dacă reușim să obținem o recompensă medie de 195 pe parcursul a 100 de încercări consecutive.

## Discretizarea stării

În Q-Learning, trebuie să construim un Q-Table care definește ce să facem în fiecare stare. Pentru a face acest lucru, starea trebuie să fie **discretă**, mai precis, trebuie să conțină un număr finit de valori discrete. Astfel, trebuie să **discretizăm** cumva observațiile, mapându-le la un set finit de stări.

Există câteva moduri în care putem face acest lucru:

- **Împărțirea în intervale**. Dacă știm intervalul unei anumite valori, putem împărți acest interval în mai multe **intervale** și apoi să înlocuim valoarea cu numărul intervalului căruia îi aparține. Acest lucru poate fi realizat folosind metoda [`digitize`](https://numpy.org/doc/stable/reference/generated/numpy.digitize.html) din numpy. În acest caz, vom ști exact dimensiunea stării, deoarece va depinde de numărul de intervale pe care le selectăm pentru digitalizare.

✅ Putem folosi interpolarea liniară pentru a aduce valorile la un interval finit (de exemplu, de la -20 la 20) și apoi să convertim numerele în întregi prin rotunjire. Acest lucru ne oferă un control mai redus asupra dimensiunii stării, mai ales dacă nu cunoaștem intervalele exacte ale valorilor de intrare. De exemplu, în cazul nostru, 2 din cele 4 valori nu au limite superioare/inferioare, ceea ce poate duce la un număr infinit de stări.

În exemplul nostru, vom folosi a doua abordare. După cum vei observa mai târziu, în ciuda lipsei limitelor superioare/inferioare, aceste valori rareori iau valori în afara unor intervale finite, astfel încât stările cu valori extreme vor fi foarte rare.

1. Iată funcția care va lua observația din modelul nostru și va produce un tuplu de 4 valori întregi: (bloc de cod 6)

    ```python
    def discretize(x):
        return tuple((x/np.array([0.25, 0.25, 0.01, 0.1])).astype(np.int))
    ```

1. Să explorăm și o altă metodă de discretizare folosind intervale: (bloc de cod 7)

    ```python
    def create_bins(i,num):
        return np.arange(num+1)*(i[1]-i[0])/num+i[0]
    
    print("Sample bins for interval (-5,5) with 10 bins\n",create_bins((-5,5),10))
    
    ints = [(-5,5),(-2,2),(-0.5,0.5),(-2,2)] # intervals of values for each parameter
    nbins = [20,20,10,10] # number of bins for each parameter
    bins = [create_bins(ints[i],nbins[i]) for i in range(4)]
    
    def discretize_bins(x):
        return tuple(np.digitize(x[i],bins[i]) for i in range(4))
    ```

1. Acum să rulăm o simulare scurtă și să observăm aceste valori discrete ale mediului. Simte-te liber să încerci atât `discretize`, cât și `discretize_bins` și să vezi dacă există vreo diferență.

    ✅ `discretize_bins` returnează numărul intervalului, care începe de la 0. Astfel, pentru valorile variabilei de intrare în jurul valorii 0, returnează numărul din mijlocul intervalului (10). În `discretize`, nu ne-am preocupat de intervalul valorilor de ieșire, permițându-le să fie negative, astfel încât valorile stării nu sunt deplasate, iar 0 corespunde lui 0. (bloc de cod 8)

    ```python
    env.reset()
    
    done = False
    while not done:
       #env.render()
       obs, rew, done, info = env.step(env.action_space.sample())
       #print(discretize_bins(obs))
       print(discretize(obs))
    env.close()
    ```

    ✅ Debifează linia care începe cu `env.render` dacă vrei să vezi cum se execută mediul. În caz contrar, poți să-l execuți în fundal, ceea ce este mai rapid. Vom folosi această execuție "invizibilă" în timpul procesului de Q-Learning.

## Structura Q-Table

În lecția anterioară, starea era un simplu pereche de numere de la 0 la 8, și astfel era convenabil să reprezentăm Q-Table printr-un tensor numpy cu o formă de 8x8x2. Dacă folosim discretizarea prin intervale, dimensiunea vectorului de stare este de asemenea cunoscută, astfel încât putem folosi aceeași abordare și să reprezentăm starea printr-un array cu forma 20x20x10x10x2 (aici 2 este dimensiunea spațiului de acțiune, iar primele dimensiuni corespund numărului de intervale pe care le-am selectat pentru fiecare dintre parametrii din spațiul de observație).

Totuși, uneori dimensiunile precise ale spațiului de observație nu sunt cunoscute. În cazul funcției `discretize`, nu putem fi niciodată siguri că starea rămâne în anumite limite, deoarece unele dintre valorile originale nu sunt limitate. Astfel, vom folosi o abordare ușor diferită și vom reprezenta Q-Table printr-un dicționar.

1. Folosește perechea *(state,action)* ca cheie a dicționarului, iar valoarea ar corespunde valorii din Q-Table. (bloc de cod 9)

    ```python
    Q = {}
    actions = (0,1)
    
    def qvalues(state):
        return [Q.get((state,a),0) for a in actions]
    ```

    Aici definim și o funcție `qvalues()`, care returnează o listă de valori din Q-Table pentru o stare dată, corespunzătoare tuturor acțiunilor posibile. Dacă intrarea nu este prezentă în Q-Table, vom returna 0 ca valoare implicită.

## Să începem Q-Learning

Acum suntem gata să-l învățăm pe Peter să-și mențină echilibrul!

1. Mai întâi, să setăm câțiva hiperparametri: (bloc de cod 10)

    ```python
    # hyperparameters
    alpha = 0.3
    gamma = 0.9
    epsilon = 0.90
    ```

    Aici, `alpha` este **rata de învățare**, care definește în ce măsură ar trebui să ajustăm valorile curente din Q-Table la fiecare pas. În lecția anterioară am început cu 1 și apoi am scăzut `alpha` la valori mai mici în timpul antrenamentului. În acest exemplu, îl vom menține constant doar pentru simplitate, iar tu poți experimenta ajustarea valorilor `alpha` mai târziu.

    `gamma` este **factorul de reducere**, care arată în ce măsură ar trebui să prioritizăm recompensa viitoare față de recompensa curentă.

    `epsilon` este **factorul de explorare/exploatare**, care determină dacă ar trebui să preferăm explorarea sau exploatarea. În algoritmul nostru, vom selecta acțiunea următoare conform valorilor din Q-Table în procentul `epsilon` din cazuri, iar în restul cazurilor vom executa o acțiune aleatorie. Acest lucru ne va permite să explorăm zone ale spațiului de căutare pe care nu le-am văzut niciodată.

    ✅ În termeni de echilibrare - alegerea unei acțiuni aleatorii (explorare) ar acționa ca o împingere aleatorie în direcția greșită, iar stâlpul ar trebui să învețe cum să-și recupereze echilibrul din aceste "greșeli".

### Îmbunătățirea algoritmului

Putem face două îmbunătățiri algoritmului nostru din lecția anterioară:

- **Calcularea recompensei cumulative medii**, pe parcursul unui număr de simulări. Vom afișa progresul la fiecare 5000 de iterații și vom calcula media recompensei cumulative pe acea perioadă de timp. Asta înseamnă că, dacă obținem mai mult de 195 de puncte, putem considera problema rezolvată, cu o calitate chiar mai mare decât cea cerută.

- **Calcularea rezultatului cumulativ mediu maxim**, `Qmax`, și vom stoca Q-Table corespunzător acelui rezultat. Când rulezi antrenamentul, vei observa că uneori rezultatul cumulativ mediu începe să scadă, și vrem să păstrăm valorile din Q-Table care corespund celui mai bun model observat în timpul antrenamentului.

1. Colectează toate recompensele cumulative la fiecare simulare în vectorul `rewards` pentru a le reprezenta grafic ulterior. (bloc de cod 11)

    ```python
    def probs(v,eps=1e-4):
        v = v-v.min()+eps
        v = v/v.sum()
        return v
    
    Qmax = 0
    cum_rewards = []
    rewards = []
    for epoch in range(100000):
        obs = env.reset()
        done = False
        cum_reward=0
        # == do the simulation ==
        while not done:
            s = discretize(obs)
            if random.random()<epsilon:
                # exploitation - chose the action according to Q-Table probabilities
                v = probs(np.array(qvalues(s)))
                a = random.choices(actions,weights=v)[0]
            else:
                # exploration - randomly chose the action
                a = np.random.randint(env.action_space.n)
    
            obs, rew, done, info = env.step(a)
            cum_reward+=rew
            ns = discretize(obs)
            Q[(s,a)] = (1 - alpha) * Q.get((s,a),0) + alpha * (rew + gamma * max(qvalues(ns)))
        cum_rewards.append(cum_reward)
        rewards.append(cum_reward)
        # == Periodically print results and calculate average reward ==
        if epoch%5000==0:
            print(f"{epoch}: {np.average(cum_rewards)}, alpha={alpha}, epsilon={epsilon}")
            if np.average(cum_rewards) > Qmax:
                Qmax = np.average(cum_rewards)
                Qbest = Q
            cum_rewards=[]
    ```

Ce poți observa din aceste rezultate:

- **Aproape de obiectivul nostru**. Suntem foarte aproape de a atinge obiectivul de a obține 195 de recompense cumulative pe parcursul a 100+ rulări consecutive ale simulării, sau poate chiar l-am atins! Chiar dacă obținem numere mai mici, încă nu știm, deoarece calculăm media pe 5000 de rulări, iar doar 100 de rulări sunt necesare conform criteriilor formale.

- **Recompensa începe să scadă**. Uneori recompensa începe să scadă, ceea ce înseamnă că putem "distruge" valorile deja învățate din Q-Table cu cele care fac situația mai rea.

Această observație este mai clar vizibilă dacă reprezentăm grafic progresul antrenamentului.

## Reprezentarea grafică a progresului antrenamentului

În timpul antrenamentului, am colectat valoarea recompensei cumulative la fiecare dintre iterații în vectorul `rewards`. Iată cum arată când o reprezentăm grafic în funcție de numărul de iterații:

```python
plt.plot(rewards)
```

![progres brut](../../../../8-Reinforcement/2-Gym/images/train_progress_raw.png)

Din acest grafic, nu putem spune nimic, deoarece, datorită naturii procesului de antrenament stochastic, durata sesiunilor de antrenament variază foarte mult. Pentru a înțelege mai bine acest grafic, putem calcula **media mobilă** pe o serie de experimente, să zicem 100. Acest lucru poate fi realizat convenabil folosind `np.convolve`: (bloc de cod 12)

```python
def running_average(x,window):
    return np.convolve(x,np.ones(window)/window,mode='valid')

plt.plot(running_average(rewards,100))
```

![progres antrenament](../../../../8-Reinforcement/2-Gym/images/train_progress_runav.png)

## Variarea hiperparametrilor

Pentru a face învățarea mai stabilă, are sens să ajustăm unii dintre hiperparametrii noștri în timpul antrenamentului. În special:

- **Pentru rata de învățare**, `alpha`, putem începe cu valori apropiate de 1 și apoi să continuăm să scădem parametrul. Cu timpul, vom obține valori bune de probabilitate în Q-Table și, astfel, ar trebui să le ajustăm ușor, fără să le suprascriem complet cu valori noi.

- **Creșterea lui epsilon**. Poate fi util să creștem `epsilon` treptat, pentru a explora mai puțin și a exploata mai mult. Probabil are sens să începem cu o valoare mai mică pentru `epsilon` și să o creștem până aproape de 1.
> **Sarcina 1**: Joacă-te cu valorile hiperparametrilor și vezi dacă poți obține o recompensă cumulativă mai mare. Ajungi peste 195?
> **Sarcina 2**: Pentru a rezolva formal problema, trebuie să obții o recompensă medie de 195 pe parcursul a 100 runde consecutive. Măsoară acest lucru în timpul antrenamentului și asigură-te că ai rezolvat problema în mod formal!

## Vizualizarea rezultatului în acțiune

Ar fi interesant să vedem cum se comportă modelul antrenat. Să rulăm simularea și să urmăm aceeași strategie de selecție a acțiunilor ca în timpul antrenamentului, eșantionând conform distribuției de probabilitate din Q-Table: (bloc de cod 13)

```python
obs = env.reset()
done = False
while not done:
   s = discretize(obs)
   env.render()
   v = probs(np.array(qvalues(s)))
   a = random.choices(actions,weights=v)[0]
   obs,_,done,_ = env.step(a)
env.close()
```

Ar trebui să vezi ceva de genul acesta:

![un cartpole echilibrat](../../../../8-Reinforcement/2-Gym/images/cartpole-balance.gif)

---

## 🚀Provocare

> **Sarcina 3**: Aici, am folosit copia finală a Q-Table, care poate să nu fie cea mai bună. Amintește-ți că am salvat cea mai performantă Q-Table în variabila `Qbest`! Încearcă același exemplu cu cea mai performantă Q-Table, copiază `Qbest` peste `Q` și vezi dacă observi vreo diferență.

> **Sarcina 4**: Aici nu am selectat cea mai bună acțiune la fiecare pas, ci am eșantionat conform distribuției de probabilitate corespunzătoare. Ar avea mai mult sens să selectăm mereu cea mai bună acțiune, cu cea mai mare valoare din Q-Table? Acest lucru poate fi realizat folosind funcția `np.argmax` pentru a afla numărul acțiunii corespunzătoare celei mai mari valori din Q-Table. Implementează această strategie și vezi dacă îmbunătățește echilibrarea.

## [Quiz post-lectură](https://ff-quizzes.netlify.app/en/ml/)

## Temă
[Antrenează un Mountain Car](assignment.md)

## Concluzie

Am învățat acum cum să antrenăm agenți pentru a obține rezultate bune doar oferindu-le o funcție de recompensă care definește starea dorită a jocului și oferindu-le oportunitatea de a explora inteligent spațiul de căutare. Am aplicat cu succes algoritmul Q-Learning în cazurile de medii discrete și continue, dar cu acțiuni discrete.

Este important să studiem și situațiile în care starea acțiunii este continuă și când spațiul de observație este mult mai complex, cum ar fi imaginea de pe ecranul unui joc Atari. În astfel de probleme, deseori trebuie să folosim tehnici de învățare automată mai puternice, cum ar fi rețelele neuronale, pentru a obține rezultate bune. Aceste subiecte mai avansate vor fi abordate în cursul nostru viitor de AI avansat.

---

**Declinare de responsabilitate**:  
Acest document a fost tradus folosind serviciul de traducere AI [Co-op Translator](https://github.com/Azure/co-op-translator). Deși ne străduim să asigurăm acuratețea, vă rugăm să fiți conștienți că traducerile automate pot conține erori sau inexactități. Documentul original în limba sa natală ar trebui considerat sursa autoritară. Pentru informații critice, se recomandă traducerea profesională realizată de un specialist uman. Nu ne asumăm responsabilitatea pentru eventualele neînțelegeri sau interpretări greșite care pot apărea din utilizarea acestei traduceri.