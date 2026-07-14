# CartPole-luistelu

Ongelma, jota olemme ratkaisseet edellisessä oppitunnissa, saattaa vaikuttaa lelulmaalta, ei oikeastaan sovellettavalta todellisissa tilanteissa. Näin ei kuitenkaan ole, koska monet todellisen maailman ongelmat jakavat tämän tilanteen — mukaan lukien shakin ja Gon pelaaminen. Ne ovat samankaltaisia, koska meillä on lauta tietyin säännöin ja **diskreetti tila**.

## [Ennakkotehtävä](https://ff-quizzes.netlify.app/en/ml/)

## Johdanto

Tässä oppitunnissa sovellamme samoja Q-Learningin periaatteita ongelmaan, jossa on **jatkuva tila**, eli tila, joka esitetään yhdellä tai useammalla reaaliluvulla. Käsittelemme seuraavaa ongelmaa:

> **Ongelma**: Jos Peter haluaa paeta susia, hänen täytyy pystyä liikkumaan nopeammin. Näemme, miten Peter voi oppia luistelemaan, erityisesti pitämään tasapainoa, käyttämällä Q-Learningia.

![Suuri pako!](../../../../translated_images/fi/escape.18862db9930337e3.webp)

> Peter ja hänen ystävänsä keksivät keinoja paeta susia! Kuva: [Jen Looper](https://twitter.com/jenlooper)

Käytämme yksinkertaistettua versiota tasapainoilusta, joka tunnetaan **CartPole**-ongelmana. CartPole-maailmassa on vaakasuora liukukisko, joka voi liikkua vasemmalle tai oikealle, ja tavoitteena on pitää pystysuora tanko tasapainossa liukukiskon päällä.

<img alt="cartpole" src="../../../../translated_images/fi/cartpole.b5609cc0494a14f7.webp" width="200"/>

## Esivaatimukset

Tässä oppitunnissa käytämme **OpenAI Gym** -kirjastoa simuloimaan erilaisia **ympäristöjä**. Voit ajaa tämän oppitunnin koodin paikallisesti (esim. Visual Studio Codessa), jolloin simulointi avautuu uuteen ikkunaan. Kun ajat koodia verkossa, saatat joutua tekemään joitain muutoksia koodiin, kuten [tässä](https://towardsdatascience.com/rendering-openai-gym-envs-on-binder-and-google-colab-536f99391cc7) kuvataan.

## OpenAI Gym

Edellisessä oppitunnissa pelin säännöt ja tila määriteltiin `Board`-luokalla, jonka loimme itse. Tässä käytämme erikoista **simulaatioympäristöä**, joka simuloi tasapainotasangon fysiikkaa. Yksi suosituimmista simulaatioympäristöistä vahvistusoppimisen algoritmien harjoitteluun on nimeltään [Gym](https://gym.openai.com/), jota ylläpitää [OpenAI](https://openai.com/). Käyttämällä tätä gymiä voimme luoda erilaisia **ympäristöjä** cartpole-simulaatiosta Atari-peleihin.

> **Huomaa**: Voit nähdä muita OpenAI Gymin saatavilla olevia ympäristöjä [tästä](https://gym.openai.com/envs/#classic_control).

Ensin asennetaan gym ja tuodaan tarvittavat kirjastot (koodilohko 1):

```python
import sys
!{sys.executable} -m pip install gym 

import gym
import matplotlib.pyplot as plt
import numpy as np
import random
```

## Harjoitus – alustetaan cartpole-ympäristö

Työskennellessäsi cartpole-tasapainon ongelman kanssa, sinun täytyy alustaa vastaava ympäristö. Jokaisella ympäristöllä on:

- **Havainnointitila**, joka määrittelee tiedon rakenteen, jonka saamme ympäristöstä. Cartpole-ongelmassa saamme tangon sijainnin, nopeuden ja muita arvoja.

- **Toimintatila**, joka määrittelee mahdolliset toiminnot. Tapauksessamme toimintatila on diskreetti ja sisältää kaksi toimintoa – **vasen** ja **oikea**. (koodilohko 2)

1. Alusta kirjoittamalla seuraava koodi:

    ```python
    env = gym.make("CartPole-v1")
    print(env.action_space)
    print(env.observation_space)
    print(env.action_space.sample())
    ```

Tarkastellaksemme, miten ympäristö toimii, ajetaan lyhyt 100 askeleen simulaatio. Jokaisessa askeleessa annamme yhden toteutettavan toiminnon – tässä simulaatiossa valitsemme toiminnon sattumanvaraisesti `action_space`-joukosta.

1. Aja alla oleva koodi ja katso, mihin se johtaa.

    ✅ Muista, että kaiken järjen mukaan tämä koodi kannattaa ajaa paikallisessa Python-asennuksessa! (koodilohko 3)

    ```python
    env.reset()
    
    for i in range(100):
       env.render()
       env.step(env.action_space.sample())
    env.close()
    ```

    Näet jotain tällaista kuin tämä kuva:

    ![tasapainottamaton cartpole](../../../../8-Reinforcement/2-Gym/images/cartpole-nobalance.gif)

1. Simulaation aikana tarvitsemme havaintoja päättääksemme, miten toimia. Itse asiassa step-funktio palauttaa nykyiset havainnot, palkkiofunktion ja done-lipun, joka kertoo, onko järkevää jatkaa simulaatiota vai ei: (koodilohko 4)

    ```python
    env.reset()
    
    done = False
    while not done:
       env.render()
       obs, rew, done, info = env.step(env.action_space.sample())
       print(f"{obs} -> {rew}")
    env.close()
    ```

    Näet jotakin tällaista muistikirjan tulosteissa:

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

    Simulaation jokaisessa askeleessa palautettava havaintovektori sisältää seuraavat arvot:
    - Kärryn sijainti
    - Kärryn nopeus
    - Tangon kulma
    - Tangon kiertonopeus

1. Hanki näiden lukujen minimi- ja maksimiarvot: (koodilohko 5)

    ```python
    print(env.observation_space.low)
    print(env.observation_space.high)
    ```

    Saatat myös huomata, että palkkio jokaisella simulaation askeleella on aina 1. Tämä johtuu siitä, että tavoitteemme on selviytyä mahdollisimman pitkään, eli pitää tanko kohtuullisen pystyasennossa mahdollisimman pitkään.

    ✅ CartPole-simulaatiota pidetään ratkaistuna, jos onnistumme saamaan keskimääräisen palkkion 195 sadan peräkkäisen kokeilun yli.

## Tilojen diskretointi

Q-Learningissa meidän täytyy rakentaa Q-Taulukko, joka määrittelee mitä tehdä kussakin tilassa. Jotta voimme tehdä tämän, tilan täytyy olla **diskreetti**, tarkemmin sanottuna sen täytyy sisältää äärellinen määrä diskreettejä arvoja. Näin ollen meidän täytyy jotenkin **diskretoida** havainnot, kartoittamalla ne äärelliseen tilojen joukkoon.

Tämä voidaan tehdä muutamalla tavalla:

- **Jakamalla lokeroihin**. Jos tiedämme tietyn arvon vaihteluvälin, voimme jakaa tämän välin useaan **lokeroon** ja korvata arvon sillä lokeron numerolla, johon arvo kuuluu. Tämä voidaan tehdä numpy-kirjaston [`digitize`](https://numpy.org/doc/stable/reference/generated/numpy.digitize.html)-metodilla. Näin tiedämme täsmällisesti tilan koon, koska se riippuu valitsemistamme lokeroiden lukumäärästä.
  
✅ Voimme käyttää lineaarista interpolaatiota tuomaan arvot tietylle rajatulle välille (esim. -20:sta 20:een), ja sitten muuntaa luvut kokonaisluvuiksi pyöristämällä. Tämä antaa meille hieman vähemmän hallintaa tilan koosta varsinkin silloin, kun tuloarvojen tarkkoja vaihteluvälejä ei tiedetä. Esimerkiksi meidän tapauksessamme 2 neljästä arvosta ei omaa ylä- tai alarajoja, mikä voi johtaa äärettömään tilojen määrään.

Esimerkissämme käytämme toista lähestymistapaa. Kuten myöhemmin huomaat, vaikka ylä- ja alarajat eivät ole määriteltyjä, arvot esiintyvät harvoin tiettyjen äärellisten välien ulkopuolella, joten äärimmäisiin arvoihin liittyvät tilat ovat hyvin harvinaisia.

1. Tässä on funktio, joka ottaa mallimme havainnon ja tuottaa 4 kokonaisluvun tupleen: (koodilohko 6)

    ```python
    def discretize(x):
        return tuple((x/np.array([0.25, 0.25, 0.01, 0.1])).astype(np.int))
    ```

1. Tutustutaan myös toiseen diskretointimenetelmään, joka perustuu lokeroihin: (koodilohko 7)

    ```python
    def create_bins(i,num):
        return np.arange(num+1)*(i[1]-i[0])/num+i[0]
    
    print("Sample bins for interval (-5,5) with 10 bins\n",create_bins((-5,5),10))
    
    ints = [(-5,5),(-2,2),(-0.5,0.5),(-2,2)] # arvojen välit jokaiselle parametrille
    nbins = [20,20,10,10] # kunkin parametrin laatikoiden määrä
    bins = [create_bins(ints[i],nbins[i]) for i in range(4)]
    
    def discretize_bins(x):
        return tuple(np.digitize(x[i],bins[i]) for i in range(4))
    ```

1. Ajetaan nyt lyhyt simulaatio ja tarkastellaan näitä diskreettejä ympäristöarvoja. Voit kokeilla molempia `discretize` ja `discretize_bins` ja katsoa, onko niissä eroa.

    ✅ discretize_bins palauttaa lokeron numeron, joka on nollapohjainen. Näin syötteen arvon ollessa lähellä 0 se palauttaa välin keskivaiheen numeron (10). Discretize ei huomioi tulosarvojen vaihteluväliä, sallien negatiiviset arvot, joten tilan arvot eivät ole siirtyneet ja 0 vastaa 0:aa. (koodilohko 8)

    ```python
    env.reset()
    
    done = False
    while not done:
       #env.renderöi()
       obs, rew, done, info = env.step(env.action_space.sample())
       #tulosta(kotoutetut_astiat(obs))
       print(discretize(obs))
    env.close()
    ```

    ✅ Poista rivin alusta merkintä `env.render` jos haluat nähdä, miten ympäristö toimii. Muussa tapauksessa voit ajaa sen taustalla, mikä on nopeampaa. Tämä näkymätön suoritus käytetään Q-Learning-prosessimme aikana.

## Q-Taulukon rakenne

Edellisessä oppitunnissa tila oli yksinkertainen lukupari välillä 0–8, joten Q-Taulukko esitettiin numpy-tensorina, jonka muoto oli 8x8x2. Jos käytämme lokeroihin diskretisointia, tilavektorimme koko tunnetaan myös, joten voimme käyttää samaa lähestymistapaa ja esittää tilan taulukkona, jonka muoto on 20x20x10x10x2 (tässä 2 on toimintojen dimensio, ja ensimmäiset dimensioiden arvot vastaavat lokeromäärää, jonka valitsimme kullekin havainnointitilan parametrille).

Joskus emme kuitenkaan tiedä täsmällisiä havainnointitilan dimensioita. `discretize`-funktion tapauksessa emme voi koskaan olla varmoja, että tilamme pysyy tietyissä rajoissa, koska jotkut alkuperäiset arvot eivät ole rajoitettuja. Näin ollen käytämme hieman erilaista lähestymistapaa ja esitämmme Q-Taulukon sanakirjana.

1. Käytä paria *(tila, toiminto)* sanakirjan avaimena ja arvona olisi vastaava Q-Taulukon arvo. (koodilohko 9)

    ```python
    Q = {}
    actions = (0,1)
    
    def qvalues(state):
        return [Q.get((state,a),0) for a in actions]
    ```

    Määrittelemme myös funktion `qvalues()`, joka palauttaa listan Q-Taulukon arvoista annetulle tilalle, joka vastaa kaikkia mahdollisia toimintoja. Jos merkintää ei ole taulukossa, palautamme oletusarvona 0.

## Aloitetaan Q-Learning

Nyt olemme valmiita opettamaan Peterille tasapainon hallintaa!

1. Ensin asetetaan muutama hyperparametri: (koodilohko 10)

    ```python
    # hyperparametrit
    alpha = 0.3
    gamma = 0.9
    epsilon = 0.90
    ```

    Tässä `alpha` on **oppimisnopeus**, joka määrittää, missä määrin päivitämme Q-Taulukon arvoja jokaisessa askeleessa. Edellisessä oppitunnissa aloitimme arvolla 1 ja pienensimme sitten `alpha` arvoa koulutuksen aikana. Tässä esimerkissä pidämme sitä yksinkertaisuuden vuoksi vakiona, mutta voit kokeilla myöhemmin `alpha`-arvojen säätämistä.

    `gamma` on **diskonttokerroin**, joka kertoo, kuinka paljon painotamme tulevia palkkioita nykyisiin verrattuna.

    `epsilon` on **tutkimis- ja hyväksikäyttökertoimen** suhdeluku, joka määrittää pidämmekö parempana tutkia vai hyödyntää. Algoritmissamme `epsilon` prosentissa tapauksista valitsemme toiminnon Q-Taulukon arvojen perusteella ja muissa tapauksissa valitsemme toiminnon satunnaisesti. Tämä antaa meille mahdollisuuden tutkia hakutilan alueita, joita emme ole aiemmin nähneet.

    ✅ Tasapainon kannalta satunnainen toiminto (tutkiminen) olisi kuin satunnainen isku väärään suuntaan, ja tangon täytyy oppia palauttamaan tasapaino näistä "virheistä".

### Parannetaan algoritmia

Voimme tehdä kaksi parannusta edelliseen algoritmiimme:

- **Laske keskimääräinen kumulatiivinen palkkio** useiden simulaatioiden yli. Tulostamme edistymisen joka 5000 iteraatiossa ja keskiarvoistamme kumulatiivisen palkkion tämän ajanjakson yli. Tämä tarkoittaa, että jos saamme yli 195 pistettä, voimme pitää ongelmaa ratkaistuna jopa vaadittua paremmin.
  
- **Laske suurin keskimääräinen kumulatiivinen tulos**, `Qmax`, ja tallennamme Q-Taulukon, joka vastaa tätä tulosta. Kun ajat koulutusta, huomaat, että keskimääräinen kumulatiivinen tulos joskus laskee, ja haluamme säilyttää ne Q-Taulukon arvot, jotka vastaavat parasta koulutuksen aikana nähtyä mallia.

1. Kerää kaikki kumulatiiviset palkkiot kullekin simulaatiolle vektoriin `rewards` myöhempää kuvaajaa varten. (koodilohko 11)

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
        # == tee simulointi ==
        while not done:
            s = discretize(obs)
            if random.random()<epsilon:
                # hyväksikäyttö - valitse toiminto Q-taulukon todennäköisyyksien mukaan
                v = probs(np.array(qvalues(s)))
                a = random.choices(actions,weights=v)[0]
            else:
                # tutkimus - valitse toiminto satunnaisesti
                a = np.random.randint(env.action_space.n)
    
            obs, rew, done, info = env.step(a)
            cum_reward+=rew
            ns = discretize(obs)
            Q[(s,a)] = (1 - alpha) * Q.get((s,a),0) + alpha * (rew + gamma * max(qvalues(ns)))
        cum_rewards.append(cum_reward)
        rewards.append(cum_reward)
        # == Tulosta tulokset säännöllisesti ja laske keskimääräinen palkkio ==
        if epoch%5000==0:
            print(f"{epoch}: {np.average(cum_rewards)}, alpha={alpha}, epsilon={epsilon}")
            if np.average(cum_rewards) > Qmax:
                Qmax = np.average(cum_rewards)
                Qbest = Q
            cum_rewards=[]
    ```

Mitä voit huomata näistä tuloksista:

- **Lähellä tavoitetta**. Olemme hyvin lähellä tavoitetta saada 195 kumulatiivista palkkiopistettä sadan peräkkäisen simulaation aikana, tai olemme ehkä jopa saavuttaneet sen! Vaikka saisimme pienempiä arvoja, emme tiedä tarkasti, koska laskemme keskiarvon 5000 ajokerran yli ja virallinen raja on 100 ajokertaa.
  
- **Palkkio alkaa laskea**. Joskus palkkio alkaa laskea, mikä tarkoittaa, että voimme "rikkoa" jo opitut arvot Q-Taulukossa arvoilla, jotka pahentavat tilannetta.

Tämä havainto näkyy selkeämmin, jos piirrämme koulutuksen edistymisen kuvaajan.

## Koulutuksen edistymisen kuvaaja

Koulutuksen aikana olemme keränneet kumulatiivisen palkkion arvon jokaisesta iteraatiosta vektoriin `rewards`. Näin se näyttää, kun kuvaamme sen iteraation numeron funktiona:

```python
plt.plot(rewards)
```

![raakakuva](../../../../translated_images/fi/train_progress_raw.2adfdf2daea09c59.webp)

Tästä kuvasta ei voi päätellä juuri mitään, koska stokastisen koulutusprosessin luonteen vuoksi koulutussessioiden pituudet vaihtelevat paljon. Jotta kuvasta olisi enemmän tolkkua, voimme laskea **liukuvan keskiarvon** joukosta kokeita, sanotaan vaikka 100. Tämä voidaan kätevästi tehdä käyttämällä `np.convolve`:a (koodilohko 12)

```python
def running_average(x,window):
    return np.convolve(x,np.ones(window)/window,mode='valid')

plt.plot(running_average(rewards,100))
```

![koulutuksen edistyminen](../../../../translated_images/fi/train_progress_runav.c71694a8fa9ab359.webp)

## Hyperparametrien muuttaminen

Jotta oppiminen olisi vakaampaa, on järkevää säätää joitain hyperparametreja koulutuksen aikana. Erityisesti:

- **Oppimisnopeus**, `alpha`, voi alkaa lähellä arvoa 1 ja pienentyä asteittain. Ajan myötä saamme hyviä todennäköisyysarvoja Q-Taulukkoon, joten niitä tulisi säätää varovasti eikä täysin korvata uusilla arvoilla.

- **Lisää epsilonia**. Saatamme haluta nostaa `epsilon`-arvoa hitaasti, jotta tutkimista vähennetään ja hyväksikäyttöä lisätään. On todennäköisesti järkevää aloittaa matalalla `epsilon`-arvolla ja nousta lähes ykköseen.

> **Tehtävä 1**: Kokeile hyperparametrien arvoja ja katso, voitko saavuttaa suuremman kumulatiivisen palkkion. Saatko yli 195?


> **Tehtävä 2**: Ratkaistaksesi ongelman muodollisesti, sinun tulee saada keskimäärin 195 pistettä 100 peräkkäisen suorituksen ajan. Mittaa tätä harjoittelun aikana ja varmista, että olet muodollisesti ratkaissut ongelman!

## Tuloksen näkeminen toiminnassa

Olisi mielenkiintoista nähdä, miten koulutettu malli käyttäytyy. Suoritetaan simulaatio ja noudatetaan samaa toimintavalintastrategiaa kuin harjoittelun aikana, otetaan näytteitä Q-taulukon todennäköisyysjakauman mukaan: (koodilohko 13)

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

Näet todennäköisesti jotakin tällaista:

![tasapainottava cartpole](../../../../8-Reinforcement/2-Gym/images/cartpole-balance.gif)

---

## 🚀Haaste

> **Tehtävä 3**: Tässä käytimme Q-taulukon viimeisintä kopiota, joka ei välttämättä ole paras mahdollinen. Muistathan, että olemme tallentaneet parhaiten suoriutuneen Q-taulukon `Qbest`-muuttujaan! Kokeile samaa esimerkkiä käyttämällä parhaiten suoriutunutta Q-taulukkoa kopioimalla `Qbest` muuttujasta `Q`:hun ja katso, huomaatko eroa.

> **Tehtävä 4**: Tässä emme valinneet parasta toimintoa jokaisella askeleella, vaan otimme näytteitä vastaavan todennäköisyysjakauman mukaan. Olisiko järkevämpää aina valita paras toiminto, eli se, jolla on korkein Q-taulukon arvo? Tämä voidaan tehdä käyttämällä `np.argmax`-funktiota, jolla saa selville toimintonumeron, johon korkein Q-taulukon arvo liittyy. Tuo strategia käyttöön ja katso, parantaako se tasapainottelua.

## [Luennon jälkeinen tietovisa](https://ff-quizzes.netlify.app/en/ml/)

## Tehtävä
[Harjoittele Mountain Car](assignment.md)

## Yhteenveto

Olemme nyt oppineet, miten agentteja koulutetaan saavuttamaan hyviä tuloksia yksinkertaisesti tarjoamalla heille palkkiofunktio, joka määrittelee halutun pelitilan, ja antamalla heille mahdollisuus älykkääseen tilan tutkimiseen. Olemme onnistuneesti käyttäneet Q-Learning-algoritmia sekä diskreeteissä että jatkuvissa ympäristöissä, tosin diskreettien toimintojen kanssa.

On tärkeää myös tutkia tilanteita, joissa toiminta-avaruus on jatkuva ja havaintoavaruus paljon monimutkaisempi, kuten kuva Atari-pelin näytöltä. Näissä ongelmissa tarvitsemme usein tehokkaampia koneoppimistekniikoita, kuten neuroverkkoja, hyvän suorituskyvyn saavuttamiseksi. Näistä kehittyneemmistä aiheista käsitellään tulevassa edistyneemmässä tekoälykurssissamme.

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Vastuuvapauslauseke**:
Tämä asiakirja on käännetty käyttämällä tekoälypohjaista käännöspalvelua [Co-op Translator](https://github.com/Azure/co-op-translator). Vaikka pyrimme tarkkuuteen, otathan huomioon, että automaattiset käännökset saattavat sisältää virheitä tai epätarkkuuksia. Alkuperäinen asiakirja sen alkuperäiskielellä on virallinen lähde. Tärkeissä asioissa suositellaan ammattimaista ihmiskäännöstä. Emme ole vastuussa tämän käännöksen käytöstä aiheutuvista väärinymmärryksistä tai tulkinnoista.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->