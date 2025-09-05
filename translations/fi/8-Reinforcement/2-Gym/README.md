<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "107d5bb29da8a562e7ae72262d251a75",
  "translation_date": "2025-09-05T01:17:30+00:00",
  "source_file": "8-Reinforcement/2-Gym/README.md",
  "language_code": "fi"
}
-->
## Esivaatimukset

Tässä oppitunnissa käytämme kirjastoa nimeltä **OpenAI Gym** simuloimaan erilaisia **ympäristöjä**. Voit ajaa oppitunnin koodin paikallisesti (esim. Visual Studio Codessa), jolloin simulaatio avautuu uuteen ikkunaan. Jos suoritat koodin verkossa, sinun täytyy ehkä tehdä joitakin muutoksia koodiin, kuten kuvataan [tässä](https://towardsdatascience.com/rendering-openai-gym-envs-on-binder-and-google-colab-536f99391cc7).

## OpenAI Gym

Edellisessä oppitunnissa pelin säännöt ja tila määriteltiin itse luomassamme `Board`-luokassa. Tässä käytämme erityistä **simulaatioympäristöä**, joka simuloi tasapainottavan tangon fysiikkaa. Yksi suosituimmista simulaatioympäristöistä vahvistusoppimisalgoritmien kouluttamiseen on nimeltään [Gym](https://gym.openai.com/), jota ylläpitää [OpenAI](https://openai.com/). Tämän Gymin avulla voimme luoda erilaisia **ympäristöjä**, kuten cartpole-simulaation tai Atari-pelejä.

> **Huomio**: Voit nähdä muita OpenAI Gymin tarjoamia ympäristöjä [täältä](https://gym.openai.com/envs/#classic_control).

Aloitetaan asentamalla Gym ja tuomalla tarvittavat kirjastot (koodilohko 1):

```python
import sys
!{sys.executable} -m pip install gym 

import gym
import matplotlib.pyplot as plt
import numpy as np
import random
```

## Harjoitus - alustetaan cartpole-ympäristö

Cartpole-tasapainotusongelman kanssa työskentelyä varten meidän täytyy alustaa vastaava ympäristö. Jokainen ympäristö liittyy:

- **Havaintotilaan**, joka määrittää rakenteen tiedoille, joita saamme ympäristöstä. Cartpole-ongelmassa saamme tangon sijainnin, nopeuden ja joitakin muita arvoja.

- **Toimintatilaan**, joka määrittää mahdolliset toiminnot. Meidän tapauksessamme toimintatila on diskreetti ja sisältää kaksi toimintoa - **vasemmalle** ja **oikealle**. (koodilohko 2)

1. Alustamiseen kirjoita seuraava koodi:

    ```python
    env = gym.make("CartPole-v1")
    print(env.action_space)
    print(env.observation_space)
    print(env.action_space.sample())
    ```

Näemme, miten ympäristö toimii, suorittamalla lyhyen simulaation 100 askeleen ajan. Jokaisella askeleella annamme yhden toiminnon suoritettavaksi - tässä simulaatiossa valitsemme toiminnon satunnaisesti `action_space`-tilasta.

1. Suorita alla oleva koodi ja katso, mihin se johtaa.

    ✅ Muista, että on suositeltavaa ajaa tämä koodi paikallisessa Python-asennuksessa! (koodilohko 3)

    ```python
    env.reset()
    
    for i in range(100):
       env.render()
       env.step(env.action_space.sample())
    env.close()
    ```

    Sinun pitäisi nähdä jotain tämän kuvan kaltaista:

    ![ei tasapainottava cartpole](../../../../8-Reinforcement/2-Gym/images/cartpole-nobalance.gif)

1. Simulaation aikana meidän täytyy saada havaintoja päättääksemme, miten toimia. Itse asiassa `step`-funktio palauttaa nykyiset havainnot, palkintofunktion ja `done`-lipun, joka osoittaa, onko järkevää jatkaa simulaatiota vai ei: (koodilohko 4)

    ```python
    env.reset()
    
    done = False
    while not done:
       env.render()
       obs, rew, done, info = env.step(env.action_space.sample())
       print(f"{obs} -> {rew}")
    env.close()
    ```

    Näet jotain tällaista notebookin tulosteessa:

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

    Simulaation jokaisella askeleella palautettu havaintovektori sisältää seuraavat arvot:
    - Kärryn sijainti
    - Kärryn nopeus
    - Tangon kulma
    - Tangon pyörimisnopeus

1. Hanki näiden lukujen minimi- ja maksimiarvot: (koodilohko 5)

    ```python
    print(env.observation_space.low)
    print(env.observation_space.high)
    ```

    Saatat myös huomata, että palkintoarvo jokaisella simulaatioaskeleella on aina 1. Tämä johtuu siitä, että tavoitteemme on selviytyä mahdollisimman pitkään, eli pitää tanko kohtuullisen pystysuorassa mahdollisimman pitkään.

    ✅ Itse asiassa CartPole-simulaatio katsotaan ratkaistuksi, jos onnistumme saamaan keskimääräisen palkinnon 195 yli 100 peräkkäisen kokeilun aikana.

## Tilojen diskretisointi

Q-Learningissa meidän täytyy rakentaa Q-taulukko, joka määrittää, mitä tehdä kussakin tilassa. Jotta voimme tehdä tämän, tilan täytyy olla **diskreetti**, tarkemmin sanottuna sen täytyy sisältää rajallinen määrä diskreettejä arvoja. Siksi meidän täytyy jollain tavalla **diskretisoida** havaintomme, kartoittaen ne rajalliseen joukkoon tiloja.

Tähän on muutamia tapoja:

- **Jakaminen osiin**. Jos tiedämme tietyn arvon välin, voimme jakaa tämän välin useisiin **osiin** ja korvata arvon sen osan numerolla, johon se kuuluu. Tämä voidaan tehdä numpy-kirjaston [`digitize`](https://numpy.org/doc/stable/reference/generated/numpy.digitize.html)-menetelmällä. Tässä tapauksessa tiedämme tarkasti tilan koon, koska se riippuu valitsemastamme osien määrästä digitalisointia varten.

✅ Voimme käyttää lineaarista interpolointia tuodaksemme arvot rajalliseen väliin (esim. -20:stä 20:een) ja sitten muuntaa numerot kokonaisluvuiksi pyöristämällä. Tämä antaa meille hieman vähemmän kontrollia tilan koosta, erityisesti jos emme tiedä tarkkoja syötearvojen rajoja. Esimerkiksi meidän tapauksessamme 2 neljästä arvosta ei ole ylä-/alarajoja, mikä voi johtaa äärettömään määrään tiloja.

Esimerkissämme käytämme toista lähestymistapaa. Kuten saatat myöhemmin huomata, huolimatta määrittelemättömistä ylä-/alarajoista, nämä arvot harvoin saavuttavat tiettyjen rajallisten väliarvojen ulkopuolisia arvoja, joten tilat, joilla on äärimmäisiä arvoja, ovat hyvin harvinaisia.

1. Tässä on funktio, joka ottaa mallimme havainnon ja tuottaa 4 kokonaislukuarvon tuplen: (koodilohko 6)

    ```python
    def discretize(x):
        return tuple((x/np.array([0.25, 0.25, 0.01, 0.1])).astype(np.int))
    ```

1. Tutkitaan myös toinen diskretisointimenetelmä käyttäen osia: (koodilohko 7)

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

1. Suoritetaan lyhyt simulaatio ja tarkkaillaan näitä diskreettejä ympäristöarvoja. Voit kokeilla sekä `discretize`- että `discretize_bins`-funktioita ja nähdä, onko niissä eroa.

    ✅ `discretize_bins` palauttaa osan numeron, joka alkaa nollasta. Näin ollen syötearvojen ollessa lähellä 0 se palauttaa numeron välin keskeltä (10). `discretize`-funktiossa emme välittäneet lähtöarvojen välistä, jolloin arvot voivat olla negatiivisia, ja 0 vastaa 0:aa. (koodilohko 8)

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

    ✅ Poista kommentti riviltä, joka alkaa `env.render`, jos haluat nähdä, miten ympäristö suoritetaan. Muutoin voit suorittaa sen taustalla, mikä on nopeampaa. Käytämme tätä "näkymätöntä" suoritusta Q-Learning-prosessimme aikana.

## Q-taulukon rakenne

Edellisessä oppitunnissa tila oli yksinkertainen pariluku välillä 0–8, ja siksi oli kätevää esittää Q-taulukko numpy-tensorina, jonka muoto oli 8x8x2. Jos käytämme osien diskretisointia, tilavektorimme koko on myös tiedossa, joten voimme käyttää samaa lähestymistapaa ja esittää tilan taulukolla, jonka muoto on 20x20x10x10x2 (tässä 2 on toimintatilan ulottuvuus, ja ensimmäiset ulottuvuudet vastaavat osien määrää, joita olemme valinneet kullekin havaintotilan parametrille).

Joskus havaintotilan tarkat ulottuvuudet eivät kuitenkaan ole tiedossa. `discretize`-funktion tapauksessa emme voi koskaan olla varmoja, että tilamme pysyy tiettyjen rajojen sisällä, koska jotkut alkuperäisistä arvoista eivät ole rajattuja. Siksi käytämme hieman erilaista lähestymistapaa ja esittelemme Q-taulukon sanakirjana.

1. Käytä paria *(tila, toiminto)* sanakirjan avaimena, ja arvo vastaisi Q-taulukon arvoa. (koodilohko 9)

    ```python
    Q = {}
    actions = (0,1)
    
    def qvalues(state):
        return [Q.get((state,a),0) for a in actions]
    ```

    Tässä määrittelemme myös funktion `qvalues()`, joka palauttaa listan Q-taulukon arvoista tietylle tilalle, joka vastaa kaikkia mahdollisia toimintoja. Jos merkintää ei ole Q-taulukossa, palautamme oletusarvona 0.

## Aloitetaan Q-Learning

Nyt olemme valmiita opettamaan Peteriä tasapainottamaan!

1. Asetetaan ensin joitakin hyperparametreja: (koodilohko 10)

    ```python
    # hyperparameters
    alpha = 0.3
    gamma = 0.9
    epsilon = 0.90
    ```

    Tässä `alpha` on **oppimisnopeus**, joka määrittää, missä määrin meidän pitäisi säätää Q-taulukon nykyisiä arvoja jokaisella askeleella. Edellisessä oppitunnissa aloitimme arvolla 1 ja sitten vähensimme `alpha`-arvoa koulutuksen aikana. Tässä esimerkissä pidämme sen vakiona yksinkertaisuuden vuoksi, ja voit kokeilla `alpha`-arvojen säätämistä myöhemmin.

    `gamma` on **diskonttaustekijä**, joka osoittaa, missä määrin meidän pitäisi priorisoida tulevaa palkintoa nykyisen palkinnon yli.

    `epsilon` on **tutkimus/hyödyntämistekijä**, joka määrittää, pitäisikö meidän suosia tutkimista vai hyödyntämistä. Algoritmissamme valitsemme `epsilon`-prosentissa tapauksista seuraavan toiminnon Q-taulukon arvojen mukaan, ja jäljellä olevissa tapauksissa suoritamme satunnaisen toiminnon. Tämä antaa meille mahdollisuuden tutkia hakutilan alueita, joita emme ole koskaan nähneet.

    ✅ Tasapainottamisen kannalta satunnaisen toiminnon valitseminen (tutkiminen) toimisi kuin satunnainen isku väärään suuntaan, ja tangon täytyisi oppia, miten palauttaa tasapaino näistä "virheistä".

### Paranna algoritmia

Voimme myös tehdä kaksi parannusta algoritmiimme edellisestä oppitunnista:

- **Laske keskimääräinen kumulatiivinen palkinto** useiden simulaatioiden aikana. Tulostamme edistymisen joka 5000 iteraation jälkeen ja keskiarvoistamme kumulatiivisen palkinnon tuolta ajalta. Tämä tarkoittaa, että jos saamme yli 195 pistettä, voimme pitää ongelman ratkaistuna, jopa vaadittua korkeammalla laadulla.

- **Laske maksimaalinen keskimääräinen kumulatiivinen tulos**, `Qmax`, ja tallennamme Q-taulukon, joka vastaa kyseistä tulosta. Kun suoritat koulutuksen, huomaat, että joskus keskimääräinen kumulatiivinen tulos alkaa laskea, ja haluamme säilyttää Q-taulukon arvot, jotka vastaavat parasta mallia, joka havaittiin koulutuksen aikana.

1. Kerää kaikki kumulatiiviset palkinnot jokaisessa simulaatiossa `rewards`-vektoriin myöhempää visualisointia varten. (koodilohko 11)

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

Mitä voit huomata näistä tuloksista:

- **Lähellä tavoitettamme**. Olemme hyvin lähellä tavoitteen saavuttamista, eli 195 kumulatiivista palkintoa yli 100+ peräkkäisen simulaation aikana, tai olemme ehkä jo saavuttaneet sen! Vaikka saisimme pienempiä lukuja, emme silti tiedä, koska keskiarvoistamme 5000 suorituksen aikana, ja virallinen kriteeri vaatii vain 100 suoritusta.

- **Palkinto alkaa laskea**. Joskus palkinto alkaa laskea, mikä tarkoittaa, että voimme "tuhota" jo opitut arvot Q-taulukossa arvoilla, jotka pahentavat tilannetta.

Tämä havainto näkyy selkeämmin, jos piirrämme koulutuksen edistymisen.

## Koulutuksen edistymisen visualisointi

Koulutuksen aikana olemme keränneet kumulatiivisen palkintoarvon jokaisessa iteraatiossa `rewards`-vektoriin. Tässä on, miltä se näyttää, kun piirrämme sen iteraation numeroa vastaan:

```python
plt.plot(rewards)
```

![raaka edistyminen](../../../../8-Reinforcement/2-Gym/images/train_progress_raw.png)

Tästä graafista ei ole mahdollista päätellä mitään, koska stokastisen koulutusprosessin luonteen vuoksi koulutussessioiden pituus vaihtelee suuresti. Jotta graafi olisi ymmärrettävämpi, voimme laskea **juoksevan keskiarvon** sarjasta kokeita, esimerkiksi 100. Tämä voidaan tehdä kätevästi `np.convolve`-menetelmällä: (koodilohko 12)

```python
def running_average(x,window):
    return np.convolve(x,np.ones(window)/window,mode='valid')

plt.plot(running_average(rewards,100))
```

![koulutuksen edistyminen](../../../../8-Reinforcement/2-Gym/images/train_progress_runav.png)

## Hyperparametrien säätäminen

Jotta oppiminen olisi vakaampaa, kannattaa säätää joitakin hyperparametreja koulutuksen aikana. Erityisesti:

- **Oppimisnopeuden**, `alpha`, osalta voimme aloittaa arvoilla lähellä 1 ja sitten vähentää parametria. Ajan myötä saamme hyviä todennäköisyysarvoja Q-taulukkoon, ja siksi meidän pitäisi säätää niitä hieman, eikä korvata kokonaan uusilla arvoilla.

- **Lisää epsilonia**. Voimme haluta lisätä `epsilon`-arvoa hitaasti, jotta tutkimme vähemmän ja hyödynnämme enemmän. Todennäköisesti kannattaa aloittaa pienemmällä `epsilon`-arvolla ja nostaa se lähes 1:een.
> **Tehtävä 1**: Kokeile hyperparametrien arvoja ja katso, voitko saavuttaa suuremman kumulatiivisen palkkion. Pääsetkö yli 195?
> **Tehtävä 2**: Jotta ongelma ratkaistaan virallisesti, sinun täytyy saavuttaa 195 keskimääräinen palkinto 100 peräkkäisen ajon aikana. Mittaa tämä koulutuksen aikana ja varmista, että olet ratkaissut ongelman virallisesti!

## Tulosten tarkastelu käytännössä

Olisi mielenkiintoista nähdä, miten koulutettu malli käyttäytyy. Suoritetaan simulaatio ja käytetään samaa toimintojen valintastrategiaa kuin koulutuksen aikana, näytteistämällä Q-taulukon todennäköisyysjakauman mukaan: (koodilohko 13)

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

Näet jotain tällaista:

![tasapainottava cartpole](../../../../8-Reinforcement/2-Gym/images/cartpole-balance.gif)

---

## 🚀Haaste

> **Tehtävä 3**: Tässä käytimme Q-taulukon lopullista versiota, joka ei välttämättä ole paras. Muista, että olemme tallentaneet parhaiten toimivan Q-taulukon `Qbest`-muuttujaan! Kokeile samaa esimerkkiä parhaiten toimivalla Q-taulukolla kopioimalla `Qbest` `Q`:n tilalle ja katso, huomaatko eroa.

> **Tehtävä 4**: Tässä emme valinneet parasta toimintoa jokaisella askeleella, vaan näytteistimme vastaavan todennäköisyysjakauman mukaan. Olisiko järkevämpää aina valita paras toiminto, jolla on korkein Q-taulukon arvo? Tämä voidaan tehdä käyttämällä `np.argmax`-funktiota löytääkseen toimintojen numeron, joka vastaa korkeinta Q-taulukon arvoa. Toteuta tämä strategia ja katso, parantaako se tasapainottamista.

## [Luentojälkeinen kysely](https://ff-quizzes.netlify.app/en/ml/)

## Tehtävä
[Harjoittele Mountain Car](assignment.md)

## Yhteenveto

Olemme nyt oppineet, kuinka kouluttaa agentteja saavuttamaan hyviä tuloksia pelkästään tarjoamalla heille palkintofunktion, joka määrittelee pelin halutun tilan, ja antamalla heille mahdollisuuden älykkäästi tutkia hakutilaa. Olemme onnistuneesti soveltaneet Q-Learning-algoritmia sekä diskreeteissä että jatkuvissa ympäristöissä, mutta diskreeteillä toiminnoilla.

On myös tärkeää tutkia tilanteita, joissa toimintotila on jatkuva ja havaintotila paljon monimutkaisempi, kuten kuva Atari-pelin näytöstä. Näissä ongelmissa tarvitsemme usein tehokkaampia koneoppimistekniikoita, kuten neuroverkkoja, saavuttaaksemme hyviä tuloksia. Nämä edistyneemmät aiheet ovat tulevan kehittyneemmän tekoälykurssimme aiheena.

---

**Vastuuvapauslauseke**:  
Tämä asiakirja on käännetty käyttämällä tekoälypohjaista käännöspalvelua [Co-op Translator](https://github.com/Azure/co-op-translator). Vaikka pyrimme tarkkuuteen, huomioithan, että automaattiset käännökset voivat sisältää virheitä tai epätarkkuuksia. Alkuperäinen asiakirja sen alkuperäisellä kielellä tulisi pitää ensisijaisena lähteenä. Kriittisen tiedon osalta suositellaan ammattimaista ihmiskäännöstä. Emme ole vastuussa väärinkäsityksistä tai virhetulkinnoista, jotka johtuvat tämän käännöksen käytöstä.