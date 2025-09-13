<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "107d5bb29da8a562e7ae72262d251a75",
  "translation_date": "2025-09-05T13:46:38+00:00",
  "source_file": "8-Reinforcement/2-Gym/README.md",
  "language_code": "sl"
}
-->
# CartPole Drsanje

Problem, ki smo ga reševali v prejšnji lekciji, se morda zdi kot igrača, ki ni uporabna v resničnih scenarijih. To ni res, saj veliko resničnih problemov deli podobne značilnosti – vključno z igranjem šaha ali Go. Ti problemi so podobni, ker imamo tudi tukaj ploščo z določenimi pravili in **diskretno stanje**.

## [Predhodni kviz](https://ff-quizzes.netlify.app/en/ml/)

## Uvod

V tej lekciji bomo uporabili enaka načela Q-učenja za problem s **kontinuiranim stanjem**, tj. stanjem, ki ga določajo ena ali več realnih števil. Ukvarjali se bomo z naslednjim problemom:

> **Problem**: Če želi Peter pobegniti volku, mora biti sposoben hitreje premikati. Videli bomo, kako se Peter lahko nauči drsati, zlasti ohranjati ravnotežje, z uporabo Q-učenja.

![Veliki pobeg!](../../../../8-Reinforcement/2-Gym/images/escape.png)

> Peter in njegovi prijatelji postanejo ustvarjalni, da pobegnejo volku! Slika: [Jen Looper](https://twitter.com/jenlooper)

Uporabili bomo poenostavljeno različico ohranjanja ravnotežja, znano kot problem **CartPole**. V svetu CartPole imamo horizontalni drsnik, ki se lahko premika levo ali desno, cilj pa je uravnotežiti vertikalno palico na vrhu drsnika.

## Predpogoji

V tej lekciji bomo uporabili knjižnico **OpenAI Gym** za simulacijo različnih **okolij**. Kodo te lekcije lahko zaženete lokalno (npr. iz Visual Studio Code), v tem primeru se bo simulacija odprla v novem oknu. Pri izvajanju kode na spletu boste morda morali narediti nekaj prilagoditev, kot je opisano [tukaj](https://towardsdatascience.com/rendering-openai-gym-envs-on-binder-and-google-colab-536f99391cc7).

## OpenAI Gym

V prejšnji lekciji so bila pravila igre in stanje podana z razredom `Board`, ki smo ga sami definirali. Tukaj bomo uporabili posebno **simulacijsko okolje**, ki bo simuliralo fiziko palice za uravnoteženje. Eno najbolj priljubljenih simulacijskih okolij za treniranje algoritmov za krepitev učenja se imenuje [Gym](https://gym.openai.com/), ki ga vzdržuje [OpenAI](https://openai.com/). Z uporabo tega Gym-a lahko ustvarimo različna **okolja**, od simulacije CartPole do Atari iger.

> **Opomba**: Druge okolja, ki jih ponuja OpenAI Gym, si lahko ogledate [tukaj](https://gym.openai.com/envs/#classic_control).

Najprej namestimo Gym in uvozimo potrebne knjižnice (koda blok 1):

```python
import sys
!{sys.executable} -m pip install gym 

import gym
import matplotlib.pyplot as plt
import numpy as np
import random
```

## Naloga - inicializacija okolja CartPole

Za delo s problemom uravnoteženja CartPole moramo inicializirati ustrezno okolje. Vsako okolje je povezano z:

- **Prostorom opazovanja**, ki določa strukturo informacij, ki jih prejmemo iz okolja. Pri problemu CartPole prejmemo položaj palice, hitrost in nekatere druge vrednosti.

- **Prostorom akcij**, ki določa možne akcije. V našem primeru je prostor akcij diskreten in obsega dve akciji - **levo** in **desno**. (koda blok 2)

1. Za inicializacijo vnesite naslednjo kodo:

    ```python
    env = gym.make("CartPole-v1")
    print(env.action_space)
    print(env.observation_space)
    print(env.action_space.sample())
    ```

Da vidimo, kako okolje deluje, zaženimo kratko simulacijo za 100 korakov. Na vsakem koraku podamo eno od akcij, ki jih je treba izvesti – v tej simulaciji naključno izberemo akcijo iz `action_space`.

1. Zaženite spodnjo kodo in preverite rezultat.

    ✅ Ne pozabite, da je priporočljivo to kodo zagnati na lokalni Python namestitvi! (koda blok 3)

    ```python
    env.reset()
    
    for i in range(100):
       env.render()
       env.step(env.action_space.sample())
    env.close()
    ```

    Videti bi morali nekaj podobnega tej sliki:

    ![ne-uravnotežen CartPole](../../../../8-Reinforcement/2-Gym/images/cartpole-nobalance.gif)

1. Med simulacijo moramo pridobiti opazovanja, da se odločimo, kako ukrepati. Pravzaprav funkcija koraka vrne trenutna opazovanja, funkcijo nagrade in zastavico "done", ki označuje, ali ima smisel nadaljevati simulacijo ali ne: (koda blok 4)

    ```python
    env.reset()
    
    done = False
    while not done:
       env.render()
       obs, rew, done, info = env.step(env.action_space.sample())
       print(f"{obs} -> {rew}")
    env.close()
    ```

    V zvezku boste videli nekaj podobnega temu:

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

    Vektor opazovanja, ki se vrne na vsakem koraku simulacije, vsebuje naslednje vrednosti:
    - Položaj vozička
    - Hitrost vozička
    - Kot palice
    - Hitrost vrtenja palice

1. Pridobite minimalno in maksimalno vrednost teh števil: (koda blok 5)

    ```python
    print(env.observation_space.low)
    print(env.observation_space.high)
    ```

    Opazili boste tudi, da je vrednost nagrade na vsakem koraku simulacije vedno 1. To je zato, ker je naš cilj preživeti čim dlje, tj. ohraniti palico v razumno vertikalnem položaju čim dlje.

    ✅ Pravzaprav se simulacija CartPole šteje za rešeno, če nam uspe doseči povprečno nagrado 195 v 100 zaporednih poskusih.

## Diskretizacija stanja

Pri Q-učenju moramo zgraditi Q-tabelo, ki določa, kaj storiti v vsakem stanju. Da bi to lahko storili, mora biti stanje **diskretno**, natančneje, vsebovati mora končno število diskretnih vrednosti. Zato moramo nekako **diskretizirati** naša opazovanja in jih preslikati v končno množico stanj.

Obstaja nekaj načinov, kako to storiti:

- **Razdelitev na razrede**. Če poznamo interval določene vrednosti, lahko ta interval razdelimo na število **razredov** in nato vrednost zamenjamo s številko razreda, kateremu pripada. To lahko storimo z metodo numpy [`digitize`](https://numpy.org/doc/stable/reference/generated/numpy.digitize.html). V tem primeru bomo natančno poznali velikost stanja, saj bo odvisna od števila razredov, ki jih izberemo za digitalizacijo.

✅ Uporabimo lahko linearno interpolacijo, da vrednosti prenesemo na nek končni interval (recimo od -20 do 20), nato pa številke pretvorimo v cela števila z zaokroževanjem. To nam daje nekoliko manj nadzora nad velikostjo stanja, zlasti če ne poznamo natančnih razponov vhodnih vrednosti. Na primer, v našem primeru 2 od 4 vrednosti nimata zgornjih/spodnjih mej svojih vrednosti, kar lahko povzroči neskončno število stanj.

V našem primeru bomo uporabili drugi pristop. Kot boste morda opazili kasneje, kljub neomejenim zgornjim/spodnjim mejam te vrednosti redko zavzamejo vrednosti zunaj določenih končnih intervalov, zato bodo ta stanja z ekstremnimi vrednostmi zelo redka.

1. Tukaj je funkcija, ki bo vzela opazovanje iz našega modela in ustvarila nabor 4 celih števil: (koda blok 6)

    ```python
    def discretize(x):
        return tuple((x/np.array([0.25, 0.25, 0.01, 0.1])).astype(np.int))
    ```

1. Raziščimo še drugo metodo diskretizacije z uporabo razredov: (koda blok 7)

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

1. Zdaj zaženimo kratko simulacijo in opazujmo te diskretne vrednosti okolja. Preizkusite `discretize` in `discretize_bins` ter preverite, ali obstaja razlika.

    ✅ `discretize_bins` vrne številko razreda, ki je osnovana na 0. Tako za vrednosti vhodne spremenljivke okoli 0 vrne številko iz sredine intervala (10). Pri `discretize` nas ni skrbelo za razpon izhodnih vrednosti, kar omogoča, da so negativne, zato vrednosti stanja niso premaknjene, in 0 ustreza 0. (koda blok 8)

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

    ✅ Odkomentirajte vrstico, ki se začne z `env.render`, če želite videti, kako se okolje izvaja. Sicer pa lahko izvedete v ozadju, kar je hitreje. To "nevidno" izvajanje bomo uporabili med procesom Q-učenja.

## Struktura Q-tabele

V prejšnji lekciji je bilo stanje preprosta dvojica števil od 0 do 8, zato je bilo priročno predstaviti Q-tabelo z numpy tensorjem oblike 8x8x2. Če uporabimo diskretizacijo z razredi, je velikost našega vektorskega stanja prav tako znana, zato lahko uporabimo enak pristop in predstavimo stanje z matriko oblike 20x20x10x10x2 (tu je 2 dimenzija prostora akcij, prve dimenzije pa ustrezajo številu razredov, ki smo jih izbrali za vsako od parametrov v prostoru opazovanja).

Vendar pa včasih natančne dimenzije prostora opazovanja niso znane. V primeru funkcije `discretize` morda nikoli ne bomo prepričani, da naše stanje ostane znotraj določenih omejitev, ker nekatere izvirne vrednosti niso omejene. Zato bomo uporabili nekoliko drugačen pristop in predstavili Q-tabelo z uporabo slovarja.

1. Uporabite par *(state,action)* kot ključ slovarja, vrednost pa bo ustrezala vrednosti vnosa Q-tabele. (koda blok 9)

    ```python
    Q = {}
    actions = (0,1)
    
    def qvalues(state):
        return [Q.get((state,a),0) for a in actions]
    ```

    Tukaj definiramo tudi funkcijo `qvalues()`, ki vrne seznam vrednosti Q-tabele za dano stanje, ki ustreza vsem možnim akcijam. Če vnos ni prisoten v Q-tabeli, bomo privzeto vrnili 0.

## Začnimo Q-učenje

Zdaj smo pripravljeni naučiti Petra, kako ohranjati ravnotežje!

1. Najprej nastavimo nekaj hiperparametrov: (koda blok 10)

    ```python
    # hyperparameters
    alpha = 0.3
    gamma = 0.9
    epsilon = 0.90
    ```

    Tukaj je `alpha` **stopnja učenja**, ki določa, v kolikšni meri naj prilagodimo trenutne vrednosti Q-tabele na vsakem koraku. V prejšnji lekciji smo začeli z 1, nato pa zmanjšali `alpha` na nižje vrednosti med treningom. V tem primeru bomo ohranili konstantno vrednost za preprostost, kasneje pa lahko eksperimentirate z nastavitvijo vrednosti `alpha`.

    `gamma` je **faktor popusta**, ki kaže, v kolikšni meri naj dajemo prednost prihodnji nagradi pred trenutno nagrado.

    `epsilon` je **faktor raziskovanja/izkoriščanja**, ki določa, ali naj raje raziskujemo ali izkoriščamo. V našem algoritmu bomo v `epsilon` odstotkih primerov izbrali naslednjo akcijo glede na vrednosti Q-tabele, v preostalih primerih pa bomo izvedli naključno akcijo. To nam bo omogočilo raziskovanje področij iskalnega prostora, ki jih še nismo videli.

    ✅ Kar zadeva uravnoteženje – izbira naključne akcije (raziskovanje) bi delovala kot naključni udarec v napačno smer, palica pa bi se morala naučiti, kako obnoviti ravnotežje iz teh "napak".

### Izboljšajmo algoritem

Algoritem iz prejšnje lekcije lahko izboljšamo na dva načina:

- **Izračunaj povprečno kumulativno nagrado** skozi več simulacij. Napredek bomo tiskali vsakih 5000 iteracij, povprečno kumulativno nagrado pa bomo izračunali za to obdobje. Če dosežemo več kot 195 točk, lahko problem štejemo za rešen, in to z boljšo kakovostjo, kot je zahtevano.

- **Izračunaj največji povprečni kumulativni rezultat**, `Qmax`, in shranili bomo Q-tabelo, ki ustreza temu rezultatu. Ko zaženete trening, boste opazili, da se povprečni kumulativni rezultat včasih začne zmanjševati, zato želimo obdržati vrednosti Q-tabele, ki ustrezajo najboljšemu modelu, opaženemu med treningom.

1. Zberite vse kumulativne nagrade pri vsaki simulaciji v vektorju `rewards` za nadaljnje risanje. (koda blok 11)

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

Kaj lahko opazite iz teh rezultatov:

- **Blizu našega cilja**. Zelo smo blizu doseganju cilja 195 kumulativnih nagrad v več kot 100 zaporednih simulacijah, ali pa smo ga morda že dosegli! Tudi če dobimo manjše številke, tega ne vemo, ker povprečimo skozi 5000 poskusov, medtem ko je formalno merilo le 100 poskusov.

- **Nagrada začne upadati**. Včasih se nagrada začne zmanjševati, kar pomeni, da lahko "uničimo" že naučene vrednosti v Q-tabeli z novimi, ki poslabšajo situacijo.

To opazovanje je bolj jasno vidno, če narišemo napredek treninga.

## Risanje napredka treninga

Med treningom smo zbirali vrednosti kumulativne nagrade pri vsaki iteraciji v vektorju `rewards`. Tukaj je, kako to izgleda, ko ga narišemo glede na številko iteracije:

```python
plt.plot(rewards)
```

![surov napredek](../../../../8-Reinforcement/2-Gym/images/train_progress_raw.png)

Iz tega grafa ni mogoče razbrati ničesar, saj se zaradi narave stohastičnega procesa treninga dolžina treninga močno razlikuje. Da bi ta graf imel več smisla, lahko izračunamo **tekoče povprečje** skozi serijo poskusov, recimo 100. To lahko priročno storimo z `np.convolve`: (koda blok 12)

```python
def running_average(x,window):
    return np.convolve(x,np.ones(window)/window,mode='valid')

plt.plot(running_average(rewards,100))
```

![napredek treninga](../../../../8-Reinforcement/2-Gym/images/train_progress_runav.png)

## Spreminjanje hiperparametrov

Da bi učenje postalo bolj stabilno, je smiselno med treningom prilagoditi nekatere naše hiperparametre. Zlasti:

- **Za stopnjo učenja**, `alpha`, lahko začnemo z vrednostmi blizu 1, nato pa postopoma zmanjšujemo parameter. Sčasoma bomo dobili dobre verjetnostne vrednosti v Q-tabeli, zato jih moramo le rahlo prilagoditi in ne popolnoma prepisati z novimi vrednostmi.

- **Povečaj epsilon**. Morda želimo počasi povečati `epsilon`, da bi manj raziskovali in bolj izkoriščali. Verjetno ima smisel začeti z nižjo vrednostjo `epsilon` in jo postopoma povečati skoraj do 1.
> **Naloga 1**: Preizkusite različne vrednosti hiperparametrov in preverite, ali lahko dosežete višjo kumulativno nagrado. Ali dosežete več kot 195?
> **Naloga 2**: Da formalno rešite problem, morate doseči povprečno nagrado 195 v 100 zaporednih poskusih. Merite to med treningom in se prepričajte, da ste problem formalno rešili!

## Ogled rezultata v praksi

Zanimivo bi bilo dejansko videti, kako se trenirani model obnaša. Zaženimo simulacijo in sledimo isti strategiji izbire akcij kot med treningom, vzorčimo glede na porazdelitev verjetnosti v Q-tabeli: (koda blok 13)

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

Videti bi morali nekaj takega:

![a balancing cartpole](../../../../8-Reinforcement/2-Gym/images/cartpole-balance.gif)

---

## 🚀Izziv

> **Naloga 3**: Tukaj smo uporabljali končno kopijo Q-tabele, ki morda ni najboljša. Ne pozabite, da smo najboljšo Q-tabelo shranili v spremenljivko `Qbest`! Poskusite isti primer z najboljšo Q-tabelo tako, da kopirate `Qbest` v `Q` in preverite, ali opazite razliko.

> **Naloga 4**: Tukaj nismo izbirali najboljše akcije na vsakem koraku, ampak smo vzorčili glede na ustrezno porazdelitev verjetnosti. Ali bi bilo bolj smiselno vedno izbrati najboljšo akcijo z najvišjo vrednostjo v Q-tabeli? To lahko storite z uporabo funkcije `np.argmax`, da ugotovite številko akcije, ki ustreza najvišji vrednosti v Q-tabeli. Implementirajte to strategijo in preverite, ali izboljša ravnotežje.

## [Kvizi po predavanju](https://ff-quizzes.netlify.app/en/ml/)

## Naloga
[Trenirajte Mountain Car](assignment.md)

## Zaključek

Sedaj smo se naučili, kako trenirati agente, da dosežejo dobre rezultate zgolj z zagotavljanjem funkcije nagrajevanja, ki definira želeno stanje igre, in z omogočanjem inteligentnega raziskovanja iskalnega prostora. Uspešno smo uporabili algoritem Q-Learning v primerih diskretnih in kontinuiranih okolij, vendar z diskretnimi akcijami.

Pomembno je preučiti tudi situacije, kjer je stanje akcij kontinuirano in kjer je opazovalni prostor veliko bolj kompleksen, kot na primer slika zaslona igre Atari. Pri teh problemih pogosto potrebujemo močnejše tehnike strojnega učenja, kot so nevronske mreže, da dosežemo dobre rezultate. Te bolj napredne teme so predmet našega prihajajočega naprednega tečaja umetne inteligence.

---

**Omejitev odgovornosti**:  
Ta dokument je bil preveden z uporabo storitve za strojno prevajanje [Co-op Translator](https://github.com/Azure/co-op-translator). Čeprav si prizadevamo za natančnost, vas opozarjamo, da lahko avtomatizirani prevodi vsebujejo napake ali netočnosti. Izvirni dokument v njegovem izvirnem jeziku je treba obravnavati kot avtoritativni vir. Za ključne informacije priporočamo strokovno človeško prevajanje. Ne prevzemamo odgovornosti za morebitna nesporazumevanja ali napačne razlage, ki izhajajo iz uporabe tega prevoda.