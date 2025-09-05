<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "107d5bb29da8a562e7ae72262d251a75",
  "translation_date": "2025-09-05T13:45:45+00:00",
  "source_file": "8-Reinforcement/2-Gym/README.md",
  "language_code": "hr"
}
-->
# CartPole Klizanje

Problem koji smo rješavali u prethodnoj lekciji može se činiti kao igračka, ne baš primjenjiva u stvarnim životnim situacijama. No, to nije slučaj, jer mnogi stvarni problemi dijele sličan scenarij - uključujući igranje šaha ili Go. Oni su slični jer također imamo ploču s određenim pravilima i **diskretno stanje**.

## [Kviz prije predavanja](https://ff-quizzes.netlify.app/en/ml/)

## Uvod

U ovoj lekciji primijenit ćemo iste principe Q-Learninga na problem s **kontinuiranim stanjem**, tj. stanjem koje je definirano jednim ili više realnih brojeva. Bavimo se sljedećim problemom:

> **Problem**: Ako Peter želi pobjeći od vuka, mora se moći kretati brže. Vidjet ćemo kako Peter može naučiti klizati, posebno održavati ravnotežu, koristeći Q-Learning.

![Veliki bijeg!](../../../../8-Reinforcement/2-Gym/images/escape.png)

> Peter i njegovi prijatelji postaju kreativni kako bi pobjegli od vuka! Slika: [Jen Looper](https://twitter.com/jenlooper)

Koristit ćemo pojednostavljenu verziju održavanja ravnoteže poznatu kao problem **CartPole**. U svijetu CartPole-a imamo horizontalni klizač koji se može kretati lijevo ili desno, a cilj je održati vertikalni stup na vrhu klizača.

## Preduvjeti

U ovoj lekciji koristit ćemo biblioteku pod nazivom **OpenAI Gym** za simulaciju različitih **okruženja**. Kod ove lekcije možete pokrenuti lokalno (npr. iz Visual Studio Code-a), u kojem slučaju će se simulacija otvoriti u novom prozoru. Kada pokrećete kod online, možda ćete morati napraviti neke prilagodbe koda, kao što je opisano [ovdje](https://towardsdatascience.com/rendering-openai-gym-envs-on-binder-and-google-colab-536f99391cc7).

## OpenAI Gym

U prethodnoj lekciji pravila igre i stanje definirali smo pomoću klase `Board` koju smo sami kreirali. Ovdje ćemo koristiti posebno **simulacijsko okruženje**, koje će simulirati fiziku iza balansirajućeg stupa. Jedno od najpopularnijih simulacijskih okruženja za treniranje algoritama za učenje pojačanjem zove se [Gym](https://gym.openai.com/), kojeg održava [OpenAI](https://openai.com/). Koristeći ovaj Gym možemo kreirati različita **okruženja**, od simulacije CartPole-a do Atari igara.

> **Napomena**: Ostala dostupna okruženja iz OpenAI Gym-a možete vidjeti [ovdje](https://gym.openai.com/envs/#classic_control).

Prvo, instalirajmo Gym i uvezimo potrebne biblioteke (blok koda 1):

```python
import sys
!{sys.executable} -m pip install gym 

import gym
import matplotlib.pyplot as plt
import numpy as np
import random
```

## Vježba - inicijalizacija okruženja CartPole

Za rad s problemom balansiranja CartPole-a, moramo inicijalizirati odgovarajuće okruženje. Svako okruženje povezano je s:

- **Prostorom opažanja** koji definira strukturu informacija koje primamo iz okruženja. Za problem CartPole-a primamo poziciju stupa, brzinu i neke druge vrijednosti.

- **Prostorom akcija** koji definira moguće akcije. U našem slučaju prostor akcija je diskretan i sastoji se od dvije akcije - **lijevo** i **desno**. (blok koda 2)

1. Za inicijalizaciju, upišite sljedeći kod:

    ```python
    env = gym.make("CartPole-v1")
    print(env.action_space)
    print(env.observation_space)
    print(env.action_space.sample())
    ```

Da bismo vidjeli kako okruženje funkcionira, pokrenimo kratku simulaciju od 100 koraka. Na svakom koraku pružamo jednu od akcija koje treba poduzeti - u ovoj simulaciji nasumično biramo akciju iz `action_space`.

1. Pokrenite kod ispod i pogledajte rezultat.

    ✅ Zapamtite da je poželjno pokrenuti ovaj kod na lokalnoj Python instalaciji! (blok koda 3)

    ```python
    env.reset()
    
    for i in range(100):
       env.render()
       env.step(env.action_space.sample())
    env.close()
    ```

    Trebali biste vidjeti nešto slično ovoj slici:

    ![CartPole bez ravnoteže](../../../../8-Reinforcement/2-Gym/images/cartpole-nobalance.gif)

1. Tijekom simulacije, trebamo dobiti opažanja kako bismo odlučili što učiniti. Zapravo, funkcija `step` vraća trenutna opažanja, funkciju nagrade i zastavicu `done` koja označava ima li smisla nastaviti simulaciju ili ne: (blok koda 4)

    ```python
    env.reset()
    
    done = False
    while not done:
       env.render()
       obs, rew, done, info = env.step(env.action_space.sample())
       print(f"{obs} -> {rew}")
    env.close()
    ```

    Na kraju ćete vidjeti nešto slično ovome u izlazu bilježnice:

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

    Vektor opažanja koji se vraća na svakom koraku simulacije sadrži sljedeće vrijednosti:
    - Pozicija klizača
    - Brzina klizača
    - Kut stupa
    - Brzina rotacije stupa

1. Dobijte minimalne i maksimalne vrijednosti tih brojeva: (blok koda 5)

    ```python
    print(env.observation_space.low)
    print(env.observation_space.high)
    ```

    Također možete primijetiti da je vrijednost nagrade na svakom koraku simulacije uvijek 1. To je zato što je naš cilj preživjeti što je dulje moguće, tj. održavati stup u razumno vertikalnom položaju što duže.

    ✅ Zapravo, simulacija CartPole-a smatra se riješenom ako uspijemo postići prosječnu nagradu od 195 tijekom 100 uzastopnih pokušaja.

## Diskretizacija stanja

U Q-Learningu, moramo izgraditi Q-Tablicu koja definira što učiniti u svakom stanju. Da bismo to mogli učiniti, stanje mora biti **diskretno**, preciznije, mora sadržavati konačan broj diskretnih vrijednosti. Dakle, moramo nekako **diskretizirati** naša opažanja, mapirajući ih na konačan skup stanja.

Postoji nekoliko načina kako to možemo učiniti:

- **Podjela u binove**. Ako znamo interval određene vrijednosti, možemo podijeliti taj interval u određeni broj **binova**, a zatim zamijeniti vrijednost brojem bina kojem pripada. To se može učiniti pomoću metode numpy [`digitize`](https://numpy.org/doc/stable/reference/generated/numpy.digitize.html). U ovom slučaju, točno ćemo znati veličinu stanja, jer će ovisiti o broju binova koje odaberemo za digitalizaciju.

✅ Možemo koristiti linearnu interpolaciju kako bismo vrijednosti doveli na neki konačan interval (recimo, od -20 do 20), a zatim pretvoriti brojeve u cijele brojeve zaokruživanjem. To nam daje malo manje kontrole nad veličinom stanja, posebno ako ne znamo točne raspone ulaznih vrijednosti. Na primjer, u našem slučaju 2 od 4 vrijednosti nemaju gornje/donje granice svojih vrijednosti, što može rezultirati beskonačnim brojem stanja.

U našem primjeru, koristit ćemo drugi pristup. Kao što ćete kasnije primijetiti, unatoč neodređenim gornjim/donjim granicama, te vrijednosti rijetko uzimaju vrijednosti izvan određenih konačnih intervala, pa će ta stanja s ekstremnim vrijednostima biti vrlo rijetka.

1. Evo funkcije koja će uzeti opažanje iz našeg modela i proizvesti tuple od 4 cijele vrijednosti: (blok koda 6)

    ```python
    def discretize(x):
        return tuple((x/np.array([0.25, 0.25, 0.01, 0.1])).astype(np.int))
    ```

1. Također istražimo drugu metodu diskretizacije koristeći binove: (blok koda 7)

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

1. Sada pokrenimo kratku simulaciju i promatrajmo te diskretne vrijednosti okruženja. Slobodno isprobajte obje funkcije `discretize` i `discretize_bins` i provjerite postoji li razlika.

    ✅ `discretize_bins` vraća broj bina, koji počinje od 0. Dakle, za vrijednosti ulazne varijable oko 0 vraća broj iz sredine intervala (10). U `discretize`, nismo se brinuli o rasponu izlaznih vrijednosti, dopuštajući im da budu negativne, pa vrijednosti stanja nisu pomaknute, i 0 odgovara 0. (blok koda 8)

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

    ✅ Uklonite komentar s linije koja počinje s `env.render` ako želite vidjeti kako se okruženje izvršava. Inače ga možete izvršiti u pozadini, što je brže. Koristit ćemo ovo "nevidljivo" izvršavanje tijekom našeg procesa Q-Learninga.

## Struktura Q-Tablice

U našoj prethodnoj lekciji, stanje je bilo jednostavan par brojeva od 0 do 8, pa je bilo zgodno predstaviti Q-Tablicu pomoću numpy tensor-a oblika 8x8x2. Ako koristimo diskretizaciju binova, veličina našeg vektora stanja također je poznata, pa možemo koristiti isti pristup i predstaviti stanje pomoću niza oblika 20x20x10x10x2 (ovdje 2 predstavlja dimenziju prostora akcija, a prve dimenzije odgovaraju broju binova koje smo odabrali za svaku od parametara u prostoru opažanja).

Međutim, ponekad precizne dimenzije prostora opažanja nisu poznate. U slučaju funkcije `discretize`, nikada ne možemo biti sigurni da naše stanje ostaje unutar određenih granica, jer neke od originalnih vrijednosti nisu ograničene. Stoga ćemo koristiti malo drugačiji pristup i predstaviti Q-Tablicu pomoću rječnika.

1. Koristite par *(state,action)* kao ključ rječnika, a vrijednost bi odgovarala vrijednosti unosa u Q-Tablici. (blok koda 9)

    ```python
    Q = {}
    actions = (0,1)
    
    def qvalues(state):
        return [Q.get((state,a),0) for a in actions]
    ```

    Ovdje također definiramo funkciju `qvalues()`, koja vraća popis vrijednosti Q-Tablice za dano stanje koje odgovara svim mogućim akcijama. Ako unos nije prisutan u Q-Tablici, vratit ćemo 0 kao zadanu vrijednost.

## Započnimo Q-Learning

Sada smo spremni naučiti Petera kako održavati ravnotežu!

1. Prvo, postavimo neke hiperparametre: (blok koda 10)

    ```python
    # hyperparameters
    alpha = 0.3
    gamma = 0.9
    epsilon = 0.90
    ```

    Ovdje, `alpha` je **stopa učenja** koja definira u kojoj mjeri trebamo prilagoditi trenutne vrijednosti Q-Tablice na svakom koraku. U prethodnoj lekciji započeli smo s 1, a zatim smanjili `alpha` na niže vrijednosti tijekom treninga. U ovom primjeru zadržat ćemo ga konstantnim radi jednostavnosti, a vi možete eksperimentirati s prilagodbom vrijednosti `alpha` kasnije.

    `gamma` je **faktor diskontiranja** koji pokazuje u kojoj mjeri trebamo prioritizirati buduću nagradu nad trenutnom nagradom.

    `epsilon` je **faktor istraživanja/iskorištavanja** koji određuje trebamo li preferirati istraživanje ili iskorištavanje. U našem algoritmu, u `epsilon` postotku slučajeva odabrat ćemo sljedeću akciju prema vrijednostima Q-Tablice, a u preostalom broju slučajeva izvršit ćemo nasumičnu akciju. To će nam omogućiti istraživanje područja prostora pretraživanja koja nikada prije nismo vidjeli.

    ✅ U smislu održavanja ravnoteže - odabir nasumične akcije (istraživanje) djelovao bi kao nasumični udarac u pogrešnom smjeru, a stup bi morao naučiti kako povratiti ravnotežu iz tih "pogrešaka".

### Poboljšanje algoritma

Možemo napraviti dva poboljšanja našem algoritmu iz prethodne lekcije:

- **Izračunajte prosječnu kumulativnu nagradu**, tijekom određenog broja simulacija. Ispisivat ćemo napredak svakih 5000 iteracija, i prosječno ćemo izračunati kumulativnu nagradu tijekom tog vremenskog razdoblja. To znači da ako postignemo više od 195 bodova - možemo smatrati problem riješenim, čak i s višom kvalitetom nego što je potrebno.

- **Izračunajte maksimalni prosječni kumulativni rezultat**, `Qmax`, i pohraniti ćemo Q-Tablicu koja odgovara tom rezultatu. Kada pokrenete trening primijetit ćete da ponekad prosječni kumulativni rezultat počinje padati, i želimo zadržati vrijednosti Q-Tablice koje odgovaraju najboljem modelu promatranom tijekom treninga.

1. Prikupite sve kumulativne nagrade na svakoj simulaciji u vektor `rewards` za daljnje crtanje. (blok koda 11)

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

Što možete primijetiti iz tih rezultata:

- **Blizu našeg cilja**. Vrlo smo blizu postizanja cilja od 195 kumulativnih nagrada tijekom 100+ uzastopnih simulacija, ili smo ga možda već postigli! Čak i ako dobijemo manje brojeve, još uvijek ne znamo, jer prosječno računamo tijekom 5000 pokušaja, a samo 100 pokušaja je potrebno prema formalnim kriterijima.

- **Nagrada počinje padati**. Ponekad nagrada počinje padati, što znači da možemo "uništiti" već naučene vrijednosti u Q-Tablici s onima koje pogoršavaju situaciju.

Ovo opažanje je jasnije vidljivo ako nacrtamo napredak treninga.

## Crtanje napretka treninga

Tijekom treninga, prikupili smo vrijednost kumulativne nagrade na svakoj iteraciji u vektor `rewards`. Evo kako izgleda kada ga nacrtamo u odnosu na broj iteracija:

```python
plt.plot(rewards)
```

![sirovi napredak](../../../../8-Reinforcement/2-Gym/images/train_progress_raw.png)

Iz ovog grafikona nije moguće ništa zaključiti, jer zbog prirode stohastičkog procesa treninga duljina sesija treninga jako varira. Da bi grafikon imao više smisla, možemo izračunati **pokretni prosjek** tijekom serije eksperimenata, recimo 100. To se može zgodno učiniti pomoću `np.convolve`: (blok koda 12)

```python
def running_average(x,window):
    return np.convolve(x,np.ones(window)/window,mode='valid')

plt.plot(running_average(rewards,100))
```

![napredak treninga](../../../../8-Reinforcement/2-Gym/images/train_progress_runav.png)

## Variranje hiperparametara

Kako bi učenje bilo stabilnije, ima smisla prilagoditi neke od naših hiperparametara tijekom treninga. Konkretno:

- **Za stopu učenja**, `alpha`, možemo započeti s vrijednostima blizu 1, a zatim postupno smanjivati parametar. S vremenom ćemo dobivati dobre vrijednosti vjerojatnosti u Q-Tablici, pa bismo ih trebali lagano prilagođavati, a ne potpuno prepisivati novim vrijednostima.

- **Povećajte epsilon**. Možda ćemo htjeti postupno povećavati `epsilon`, kako bismo manje istraživali, a više iskorištavali. Vjerojatno ima smisla započeti s nižom vrijednosti `epsilon`, i postupno je povećavati do gotovo 1.
> **Zadatak 1**: Igrajte se s vrijednostima hiperparametara i provjerite možete li postići veći kumulativni nagradni rezultat. Dosežete li iznad 195?
> **Zadatak 2**: Da biste formalno riješili problem, trebate postići prosječnu nagradu od 195 kroz 100 uzastopnih pokretanja. Mjerite to tijekom treninga i uvjerite se da ste formalno riješili problem!

## Promatranje rezultata u praksi

Bilo bi zanimljivo vidjeti kako se trenirani model ponaša. Pokrenimo simulaciju i slijedimo istu strategiju odabira akcija kao tijekom treninga, uzorkujući prema distribuciji vjerojatnosti u Q-Tablici: (blok koda 13)

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

Trebali biste vidjeti nešto poput ovoga:

![balansirajući cartpole](../../../../8-Reinforcement/2-Gym/images/cartpole-balance.gif)

---

## 🚀Izazov

> **Zadatak 3**: Ovdje smo koristili završnu verziju Q-Tablice, koja možda nije najbolja. Zapamtite da smo najbolju Q-Tablicu spremili u varijablu `Qbest`! Isprobajte isti primjer s najboljom Q-Tablicom tako da kopirate `Qbest` u `Q` i provjerite primjećujete li razliku.

> **Zadatak 4**: Ovdje nismo birali najbolju akciju u svakom koraku, već smo uzorkovali prema odgovarajućoj distribuciji vjerojatnosti. Bi li imalo više smisla uvijek birati najbolju akciju, onu s najvišom vrijednošću u Q-Tablici? To se može učiniti korištenjem funkcije `np.argmax` kako biste pronašli broj akcije koji odgovara najvišoj vrijednosti u Q-Tablici. Implementirajte ovu strategiju i provjerite poboljšava li balansiranje.

## [Kviz nakon predavanja](https://ff-quizzes.netlify.app/en/ml/)

## Zadatak
[Trenirajte Mountain Car](assignment.md)

## Zaključak

Sada smo naučili kako trenirati agente da postignu dobre rezultate samo pružanjem funkcije nagrade koja definira željeno stanje igre i omogućavanjem inteligentnog istraživanja prostora pretraživanja. Uspješno smo primijenili algoritam Q-Learning u slučajevima diskretnih i kontinuiranih okruženja, ali s diskretnim akcijama.

Važno je također proučiti situacije u kojima je stanje akcije također kontinuirano, a prostor opažanja mnogo složeniji, poput slike s ekrana Atari igre. U tim problemima često trebamo koristiti moćnije tehnike strojnog učenja, poput neuronskih mreža, kako bismo postigli dobre rezultate. Ti napredniji koncepti bit će tema našeg nadolazećeg naprednog AI tečaja.

---

**Odricanje od odgovornosti**:  
Ovaj dokument je preveden pomoću AI usluge za prevođenje [Co-op Translator](https://github.com/Azure/co-op-translator). Iako nastojimo osigurati točnost, imajte na umu da automatski prijevodi mogu sadržavati pogreške ili netočnosti. Izvorni dokument na izvornom jeziku treba smatrati mjerodavnim izvorom. Za ključne informacije preporučuje se profesionalni prijevod od strane stručnjaka. Ne preuzimamo odgovornost za bilo kakva nesporazuma ili pogrešna tumačenja koja proizlaze iz korištenja ovog prijevoda.