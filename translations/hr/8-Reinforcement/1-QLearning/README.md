<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "911efd5e595089000cb3c16fce1beab8",
  "translation_date": "2025-09-05T13:35:49+00:00",
  "source_file": "8-Reinforcement/1-QLearning/README.md",
  "language_code": "hr"
}
-->
# Uvod u učenje pojačanjem i Q-učenje

![Sažetak učenja pojačanjem u strojnom učenju u obliku sketchnotea](../../../../sketchnotes/ml-reinforcement.png)
> Sketchnote autorice [Tomomi Imura](https://www.twitter.com/girlie_mac)

Učenje pojačanjem uključuje tri važna koncepta: agenta, određena stanja i skup akcija za svako stanje. Izvršavanjem akcije u određenom stanju, agent dobiva nagradu. Zamislite računalnu igru Super Mario. Vi ste Mario, nalazite se na razini igre, stojite pored ruba litice. Iznad vas je novčić. Vi, kao Mario, na razini igre, na određenoj poziciji... to je vaše stanje. Pomicanje korak udesno (akcija) odvest će vas preko ruba, što bi vam donijelo nisku numeričku ocjenu. Međutim, pritiskom na gumb za skok osvojili biste bod i ostali živi. To je pozitivan ishod i trebao bi vam donijeti pozitivnu numeričku ocjenu.

Korištenjem učenja pojačanjem i simulatora (igre), možete naučiti kako igrati igru kako biste maksimizirali nagradu, što znači ostati živ i osvojiti što više bodova.

[![Uvod u učenje pojačanjem](https://img.youtube.com/vi/lDq_en8RNOo/0.jpg)](https://www.youtube.com/watch?v=lDq_en8RNOo)

> 🎥 Kliknite na sliku iznad kako biste čuli Dmitryja kako govori o učenju pojačanjem

## [Kviz prije predavanja](https://ff-quizzes.netlify.app/en/ml/)

## Preduvjeti i postavljanje

U ovoj lekciji eksperimentirat ćemo s nekim kodom u Pythonu. Trebali biste moći pokrenuti Jupyter Notebook kod iz ove lekcije, bilo na svom računalu ili negdje u oblaku.

Možete otvoriti [bilježnicu lekcije](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/notebook.ipynb) i proći kroz ovu lekciju kako biste je izgradili.

> **Napomena:** Ako otvarate ovaj kod iz oblaka, također trebate preuzeti datoteku [`rlboard.py`](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/rlboard.py), koja se koristi u kodu bilježnice. Dodajte je u isti direktorij kao i bilježnicu.

## Uvod

U ovoj lekciji istražit ćemo svijet **[Petra i vuka](https://en.wikipedia.org/wiki/Peter_and_the_Wolf)**, inspiriran glazbenom bajkom ruskog skladatelja [Sergeja Prokofjeva](https://en.wikipedia.org/wiki/Sergei_Prokofiev). Koristit ćemo **učenje pojačanjem** kako bismo omogućili Petru da istraži svoje okruženje, prikupi ukusne jabuke i izbjegne susret s vukom.

**Učenje pojačanjem** (RL) je tehnika učenja koja nam omogućuje da naučimo optimalno ponašanje **agenta** u nekom **okruženju** izvođenjem mnogih eksperimenata. Agent u ovom okruženju treba imati neki **cilj**, definiran pomoću **funkcije nagrade**.

## Okruženje

Radi jednostavnosti, zamislimo Petrov svijet kao kvadratnu ploču veličine `širina` x `visina`, ovako:

![Petrovo okruženje](../../../../8-Reinforcement/1-QLearning/images/environment.png)

Svaka ćelija na ovoj ploči može biti:

* **tlo**, po kojem Peter i druga bića mogu hodati.
* **voda**, po kojoj očito ne možete hodati.
* **stablo** ili **trava**, mjesto gdje se možete odmoriti.
* **jabuka**, što predstavlja nešto što bi Peter rado pronašao kako bi se nahranio.
* **vuk**, koji je opasan i treba ga izbjegavati.

Postoji zaseban Python modul, [`rlboard.py`](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/rlboard.py), koji sadrži kod za rad s ovim okruženjem. Budući da ovaj kod nije važan za razumijevanje naših koncepata, uvest ćemo modul i koristiti ga za stvaranje uzorka ploče (blok koda 1):

```python
from rlboard import *

width, height = 8,8
m = Board(width,height)
m.randomize(seed=13)
m.plot()
```

Ovaj kod bi trebao ispisati sliku okruženja sličnu onoj gore.

## Akcije i politika

U našem primjeru, Petrov cilj bio bi pronaći jabuku, dok izbjegava vuka i druge prepreke. Da bi to učinio, može se kretati dok ne pronađe jabuku.

Dakle, na bilo kojoj poziciji, može birati između sljedećih akcija: gore, dolje, lijevo i desno.

Te ćemo akcije definirati kao rječnik i mapirati ih na parove odgovarajućih promjena koordinata. Na primjer, pomicanje udesno (`R`) odgovaralo bi paru `(1,0)`. (blok koda 2):

```python
actions = { "U" : (0,-1), "D" : (0,1), "L" : (-1,0), "R" : (1,0) }
action_idx = { a : i for i,a in enumerate(actions.keys()) }
```

Da rezimiramo, strategija i cilj ovog scenarija su sljedeći:

- **Strategija** našeg agenta (Petra) definirana je tzv. **politikom**. Politika je funkcija koja vraća akciju u bilo kojem danom stanju. U našem slučaju, stanje problema predstavljeno je pločom, uključujući trenutnu poziciju igrača.

- **Cilj** učenja pojačanjem je na kraju naučiti dobru politiku koja će nam omogućiti učinkovito rješavanje problema. Međutim, kao osnovnu liniju, razmotrit ćemo najjednostavniju politiku zvanu **slučajna šetnja**.

## Slučajna šetnja

Najprije ćemo riješiti naš problem implementacijom strategije slučajne šetnje. Kod slučajne šetnje, nasumično ćemo birati sljedeću akciju iz dopuštenih akcija, sve dok ne dođemo do jabuke (blok koda 3).

1. Implementirajte slučajnu šetnju pomoću donjeg koda:

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

    Poziv funkcije `walk` trebao bi vratiti duljinu odgovarajuće staze, koja može varirati od jednog pokretanja do drugog.

1. Pokrenite eksperiment šetnje nekoliko puta (recimo, 100) i ispišite dobivene statistike (blok koda 4):

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

    Primijetite da je prosječna duljina staze oko 30-40 koraka, što je prilično puno, s obzirom na to da je prosječna udaljenost do najbliže jabuke oko 5-6 koraka.

    Također možete vidjeti kako izgleda Petrov pokret tijekom slučajne šetnje:

    ![Petrova slučajna šetnja](../../../../8-Reinforcement/1-QLearning/images/random_walk.gif)

## Funkcija nagrade

Kako bismo našu politiku učinili inteligentnijom, trebamo razumjeti koji su potezi "bolji" od drugih. Da bismo to učinili, trebamo definirati naš cilj.

Cilj se može definirati u smislu **funkcije nagrade**, koja će vraćati neku vrijednost ocjene za svako stanje. Što je broj veći, to je funkcija nagrade bolja. (blok koda 5)

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

Zanimljivo je da u većini slučajeva *značajnu nagradu dobivamo tek na kraju igre*. To znači da naš algoritam nekako treba zapamtiti "dobre" korake koji vode do pozitivne nagrade na kraju i povećati njihovu važnost. Slično tome, svi potezi koji vode do loših rezultata trebaju se obeshrabriti.

## Q-učenje

Algoritam koji ćemo ovdje raspraviti zove se **Q-učenje**. U ovom algoritmu, politika je definirana funkcijom (ili strukturom podataka) zvanom **Q-Tablica**. Ona bilježi "dobrotu" svake od akcija u danom stanju.

Zove se Q-Tablica jer je često zgodno predstavljati je kao tablicu ili višedimenzionalni niz. Budući da naša ploča ima dimenzije `širina` x `visina`, možemo predstaviti Q-Tablicu pomoću numpy niza s oblikom `širina` x `visina` x `len(actions)`: (blok koda 6)

```python
Q = np.ones((width,height,len(actions)),dtype=np.float)*1.0/len(actions)
```

Primijetite da inicijaliziramo sve vrijednosti Q-Tablice s jednakom vrijednošću, u našem slučaju - 0.25. Ovo odgovara politici "slučajne šetnje", jer su svi potezi u svakom stanju jednako dobri. Q-Tablicu možemo proslijediti funkciji `plot` kako bismo vizualizirali tablicu na ploči: `m.plot(Q)`.

![Petrovo okruženje](../../../../8-Reinforcement/1-QLearning/images/env_init.png)

U središtu svake ćelije nalazi se "strelica" koja označava preferirani smjer kretanja. Budući da su svi smjerovi jednaki, prikazuje se točka.

Sada trebamo pokrenuti simulaciju, istražiti naše okruženje i naučiti bolju raspodjelu vrijednosti Q-Tablice, što će nam omogućiti da mnogo brže pronađemo put do jabuke.

## Suština Q-učenja: Bellmanova jednadžba

Jednom kada se počnemo kretati, svaka akcija imat će odgovarajuću nagradu, tj. teoretski možemo odabrati sljedeću akciju na temelju najveće neposredne nagrade. Međutim, u većini stanja potez neće postići naš cilj dolaska do jabuke, pa stoga ne možemo odmah odlučiti koji je smjer bolji.

> Zapamtite da nije važan neposredni rezultat, već konačni rezultat, koji ćemo dobiti na kraju simulacije.

Kako bismo uzeli u obzir ovu odgođenu nagradu, trebamo koristiti principe **[dinamičkog programiranja](https://en.wikipedia.org/wiki/Dynamic_programming)**, koji nam omogućuju da o našem problemu razmišljamo rekurzivno.

Pretpostavimo da se sada nalazimo u stanju *s* i želimo se pomaknuti u sljedeće stanje *s'*. Time ćemo dobiti neposrednu nagradu *r(s,a)*, definiranu funkcijom nagrade, plus neku buduću nagradu. Ako pretpostavimo da naša Q-Tablica točno odražava "privlačnost" svake akcije, tada ćemo u stanju *s'* odabrati akciju *a* koja odgovara maksimalnoj vrijednosti *Q(s',a')*. Tako će najbolja moguća buduća nagrada koju bismo mogli dobiti u stanju *s* biti definirana kao `max`

## Provjera politike

Budući da Q-Tablica prikazuje "privlačnost" svake akcije u svakom stanju, vrlo je jednostavno koristiti je za definiranje učinkovitog kretanja u našem svijetu. U najjednostavnijem slučaju, možemo odabrati akciju koja odgovara najvećoj vrijednosti u Q-Tablici: (kodni blok 9)

```python
def qpolicy_strict(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = list(actions)[np.argmax(v)]
        return a

walk(m,qpolicy_strict)
```

> Ako nekoliko puta isprobate gornji kod, možda ćete primijetiti da se ponekad "zaglavi" i morate pritisnuti gumb STOP u bilježnici kako biste ga prekinuli. To se događa jer mogu postojati situacije u kojima dva stanja "pokazuju" jedno na drugo u smislu optimalne Q-Vrijednosti, u kojem slučaju agent završava krećući se između tih stanja beskonačno.

## 🚀Izazov

> **Zadatak 1:** Modificirajte funkciju `walk` kako biste ograničili maksimalnu duljinu puta na određeni broj koraka (recimo, 100) i promatrajte kako gornji kod povremeno vraća ovu vrijednost.

> **Zadatak 2:** Modificirajte funkciju `walk` tako da se ne vraća na mjesta na kojima je već bio. Ovo će spriječiti da se `walk` ponavlja, no agent i dalje može završiti "zarobljen" na lokaciji s koje ne može pobjeći.

## Navigacija

Bolja navigacijska politika bila bi ona koju smo koristili tijekom treninga, a koja kombinira eksploataciju i istraživanje. U ovoj politici odabiremo svaku akciju s određenom vjerojatnošću, proporcionalno vrijednostima u Q-Tablici. Ova strategija može i dalje rezultirati time da se agent vraća na poziciju koju je već istražio, ali, kao što možete vidjeti iz koda dolje, rezultira vrlo kratkim prosječnim putem do željene lokacije (zapamtite da `print_statistics` pokreće simulaciju 100 puta): (kodni blok 10)

```python
def qpolicy(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = random.choices(list(actions),weights=v)[0]
        return a

print_statistics(qpolicy)
```

Nakon pokretanja ovog koda, trebali biste dobiti znatno manju prosječnu duljinu puta nego prije, u rasponu od 3-6.

## Istraživanje procesa učenja

Kao što smo spomenuli, proces učenja je ravnoteža između istraživanja i korištenja stečenog znanja o strukturi prostora problema. Vidjeli smo da su rezultati učenja (sposobnost pomaganja agentu da pronađe kratak put do cilja) poboljšani, ali također je zanimljivo promatrati kako se prosječna duljina puta ponaša tijekom procesa učenja:

## Sažetak naučenog:

- **Prosječna duljina puta raste**. Ono što ovdje vidimo jest da na početku prosječna duljina puta raste. To je vjerojatno zbog činjenice da, kada ništa ne znamo o okolišu, vjerojatno ćemo se zaglaviti u lošim stanjima, poput vode ili vuka. Kako učimo više i počnemo koristiti to znanje, možemo dulje istraživati okoliš, ali još uvijek ne znamo dobro gdje se nalaze jabuke.

- **Duljina puta se smanjuje kako učimo više**. Kada dovoljno naučimo, agentu postaje lakše postići cilj, a duljina puta počinje se smanjivati. Međutim, još uvijek smo otvoreni za istraživanje, pa često skrećemo s najboljeg puta i istražujemo nove opcije, čime put postaje dulji od optimalnog.

- **Duljina naglo raste**. Ono što također primjećujemo na ovom grafu jest da u nekom trenutku duljina naglo raste. To ukazuje na stohastičku prirodu procesa i da u nekom trenutku možemo "pokvariti" koeficijente u Q-Tablici prepisivanjem novih vrijednosti. Ovo bi idealno trebalo minimizirati smanjenjem stope učenja (na primjer, prema kraju treninga, prilagođavamo vrijednosti u Q-Tablici samo malim iznosom).

Sveukupno, važno je zapamtiti da uspjeh i kvaliteta procesa učenja značajno ovise o parametrima, poput stope učenja, smanjenja stope učenja i faktora diskonta. Ti se parametri često nazivaju **hiperparametri**, kako bi se razlikovali od **parametara**, koje optimiziramo tijekom treninga (na primjer, koeficijenti u Q-Tablici). Proces pronalaženja najboljih vrijednosti hiperparametara naziva se **optimizacija hiperparametara**, i zaslužuje zasebnu temu.

## [Kviz nakon predavanja](https://ff-quizzes.netlify.app/en/ml/)

## Zadatak 
[Realističniji svijet](assignment.md)

---

**Odricanje od odgovornosti**:  
Ovaj dokument je preveden korištenjem AI usluge za prevođenje [Co-op Translator](https://github.com/Azure/co-op-translator). Iako nastojimo osigurati točnost, imajte na umu da automatski prijevodi mogu sadržavati pogreške ili netočnosti. Izvorni dokument na izvornom jeziku treba smatrati mjerodavnim izvorom. Za ključne informacije preporučuje se profesionalni prijevod od strane stručnjaka. Ne preuzimamo odgovornost za bilo kakva nesporazuma ili pogrešna tumačenja koja proizlaze iz korištenja ovog prijevoda.