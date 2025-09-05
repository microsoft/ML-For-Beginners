<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "911efd5e595089000cb3c16fce1beab8",
  "translation_date": "2025-09-05T01:11:06+00:00",
  "source_file": "8-Reinforcement/1-QLearning/README.md",
  "language_code": "fi"
}
-->
# Johdanto vahvistusoppimiseen ja Q-oppimiseen

![Yhteenveto vahvistusoppimisesta koneoppimisessa sketchnotena](../../../../sketchnotes/ml-reinforcement.png)
> Sketchnote by [Tomomi Imura](https://www.twitter.com/girlie_mac)

Vahvistusoppiminen sisältää kolme tärkeää käsitettä: agentti, tilat ja joukko toimintoja per tila. Kun agentti suorittaa toiminnon tietyssä tilassa, se saa palkkion. Kuvittele tietokonepeli Super Mario. Sinä olet Mario, olet pelitasolla, seisot kallion reunalla. Yläpuolellasi on kolikko. Sinä, Mario, pelitasolla, tietyssä sijainnissa... se on tilasi. Yksi askel oikealle (toiminto) vie sinut reunalta alas, ja se antaisi sinulle matalan numeerisen pisteen. Kuitenkin hyppynapin painaminen antaisi sinulle pisteen ja pysyisit hengissä. Se on positiivinen lopputulos, ja sen pitäisi palkita sinut positiivisella numeerisella pisteellä.

Käyttämällä vahvistusoppimista ja simulaattoria (peliä) voit oppia pelaamaan peliä maksimoidaksesi palkkion, joka on pysyä hengissä ja kerätä mahdollisimman paljon pisteitä.

[![Johdanto vahvistusoppimiseen](https://img.youtube.com/vi/lDq_en8RNOo/0.jpg)](https://www.youtube.com/watch?v=lDq_en8RNOo)

> 🎥 Klikkaa yllä olevaa kuvaa kuullaksesi Dmitryn keskustelua vahvistusoppimisesta

## [Ennakkokysely](https://ff-quizzes.netlify.app/en/ml/)

## Esivaatimukset ja asennus

Tässä oppitunnissa kokeilemme Python-koodia. Sinun pitäisi pystyä suorittamaan tämän oppitunnin Jupyter Notebook -koodi joko omalla tietokoneellasi tai pilvessä.

Voit avata [oppitunnin notebookin](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/notebook.ipynb) ja käydä läpi tämän oppitunnin rakentaaksesi.

> **Huom:** Jos avaat tämän koodin pilvestä, sinun täytyy myös hakea [`rlboard.py`](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/rlboard.py) -tiedosto, jota käytetään notebook-koodissa. Lisää se samaan hakemistoon kuin notebook.

## Johdanto

Tässä oppitunnissa tutkimme **[Pekka ja susi](https://en.wikipedia.org/wiki/Peter_and_the_Wolf)** -maailmaa, joka on saanut inspiraationsa venäläisen säveltäjän [Sergei Prokofjevin](https://en.wikipedia.org/wiki/Sergei_Prokofiev) musiikillisesta sadusta. Käytämme **vahvistusoppimista** antaaksemme Pekalle mahdollisuuden tutkia ympäristöään, kerätä herkullisia omenoita ja välttää kohtaamista suden kanssa.

**Vahvistusoppiminen** (RL) on oppimistekniikka, joka mahdollistaa optimaalisen käyttäytymisen oppimisen **agentille** jossain **ympäristössä** suorittamalla lukuisia kokeita. Agentilla tässä ympäristössä tulisi olla jokin **tavoite**, joka määritellään **palkkiofunktiolla**.

## Ympäristö

Yksinkertaisuuden vuoksi kuvitelkaamme Pekan maailma neliötaulukoksi, jonka koko on `leveys` x `korkeus`, kuten tässä:

![Pekan ympäristö](../../../../8-Reinforcement/1-QLearning/images/environment.png)

Jokainen solu tässä taulukossa voi olla:

* **maa**, jolla Pekka ja muut olennot voivat kävellä.
* **vesi**, jolla ei tietenkään voi kävellä.
* **puu** tai **ruoho**, paikka, jossa voi levätä.
* **omena**, joka edustaa jotain, mitä Pekka mielellään löytäisi ravinnokseen.
* **susi**, joka on vaarallinen ja tulisi välttää.

On olemassa erillinen Python-moduuli, [`rlboard.py`](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/rlboard.py), joka sisältää koodin tämän ympäristön kanssa työskentelyyn. Koska tämä koodi ei ole tärkeä käsitteidemme ymmärtämisen kannalta, tuomme moduulin ja käytämme sitä luodaksemme esimerkkitaulukon (koodilohko 1):

```python
from rlboard import *

width, height = 8,8
m = Board(width,height)
m.randomize(seed=13)
m.plot()
```

Tämän koodin pitäisi tulostaa ympäristön kuva, joka on samanlainen kuin yllä.

## Toiminnot ja politiikka

Esimerkissämme Pekan tavoitteena olisi löytää omena samalla välttäen suden ja muut esteet. Tätä varten hän voi käytännössä kävellä ympäriinsä, kunnes löytää omenan.

Siksi missä tahansa sijainnissa hän voi valita yhden seuraavista toiminnoista: ylös, alas, vasemmalle ja oikealle.

Määrittelemme nämä toiminnot sanakirjana ja yhdistämme ne vastaaviin koordinaattimuutoksiin. Esimerkiksi oikealle siirtyminen (`R`) vastaisi paria `(1,0)`. (koodilohko 2):

```python
actions = { "U" : (0,-1), "D" : (0,1), "L" : (-1,0), "R" : (1,0) }
action_idx = { a : i for i,a in enumerate(actions.keys()) }
```

Yhteenvetona tämän skenaarion strategia ja tavoite ovat seuraavat:

- **Strategia**, agenttimme (Pekan) strategia määritellään niin sanotulla **politiikalla**. Politiikka on funktio, joka palauttaa toiminnon missä tahansa tilassa. Meidän tapauksessamme ongelman tila esitetään taulukolla, mukaan lukien pelaajan nykyinen sijainti.

- **Tavoite**, vahvistusoppimisen tavoite on lopulta oppia hyvä politiikka, joka mahdollistaa ongelman tehokkaan ratkaisemisen. Perustana tarkastelemme kuitenkin yksinkertaisinta politiikkaa, jota kutsutaan **satunnaiseksi kävelyksi**.

## Satunnainen kävely

Ratkaistaan ensin ongelmamme toteuttamalla satunnaisen kävelyn strategia. Satunnaisessa kävelyssä valitsemme seuraavan toiminnon satunnaisesti sallituista toiminnoista, kunnes saavumme omenalle (koodilohko 3).

1. Toteuta satunnainen kävely alla olevalla koodilla:

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

    `walk`-kutsun pitäisi palauttaa vastaavan polun pituus, joka voi vaihdella eri suorituskerroilla.

1. Suorita kävelykokeilu useita kertoja (esimerkiksi 100) ja tulosta tuloksena saadut tilastot (koodilohko 4):

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

    Huomaa, että polun keskimääräinen pituus on noin 30–40 askelta, mikä on melko paljon, kun otetaan huomioon, että keskimääräinen etäisyys lähimpään omenaan on noin 5–6 askelta.

    Voit myös nähdä, miltä Pekan liikkuminen näyttää satunnaisen kävelyn aikana:

    ![Pekan satunnainen kävely](../../../../8-Reinforcement/1-QLearning/images/random_walk.gif)

## Palkkiofunktio

Jotta politiikkamme olisi älykkäämpi, meidän täytyy ymmärtää, mitkä siirrot ovat "parempia" kuin toiset. Tätä varten meidän täytyy määritellä tavoitteemme.

Tavoite voidaan määritellä **palkkiofunktion** avulla, joka palauttaa jonkin pistemäärän jokaiselle tilalle. Mitä korkeampi numero, sitä parempi palkkiofunktio. (koodilohko 5)

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

Mielenkiintoinen asia palkkiofunktioissa on, että useimmissa tapauksissa *merkittävä palkkio annetaan vasta pelin lopussa*. Tämä tarkoittaa, että algoritmimme pitäisi jotenkin muistaa "hyvät" askeleet, jotka johtavat positiiviseen palkkioon lopussa, ja lisätä niiden merkitystä. Samoin kaikki siirrot, jotka johtavat huonoihin tuloksiin, tulisi estää.

## Q-oppiminen

Algoritmi, jota käsittelemme tässä, on nimeltään **Q-oppiminen**. Tässä algoritmissa politiikka määritellään funktiolla (tai tietorakenteella), jota kutsutaan **Q-taulukoksi**. Se tallentaa kunkin toiminnon "hyvyyden" tietyssä tilassa.

Sitä kutsutaan Q-taulukoksi, koska se on usein kätevää esittää taulukkona tai monidimensionaalisena matriisina. Koska taulukkomme mitat ovat `leveys` x `korkeus`, voimme esittää Q-taulukon numpy-matriisina, jonka muoto on `leveys` x `korkeus` x `len(toiminnot)`: (koodilohko 6)

```python
Q = np.ones((width,height,len(actions)),dtype=np.float)*1.0/len(actions)
```

Huomaa, että alustamme kaikki Q-taulukon arvot samalla arvolla, tässä tapauksessa - 0.25. Tämä vastaa "satunnaisen kävelyn" politiikkaa, koska kaikki siirrot jokaisessa tilassa ovat yhtä hyviä. Voimme välittää Q-taulukon `plot`-funktiolle visualisoidaksemme taulukon taulukossa: `m.plot(Q)`.

![Pekan ympäristö](../../../../8-Reinforcement/1-QLearning/images/env_init.png)

Jokaisen solun keskellä on "nuoli", joka osoittaa suositellun liikkumissuunnan. Koska kaikki suunnat ovat yhtä hyviä, näytetään piste.

Nyt meidän täytyy suorittaa simulaatio, tutkia ympäristöämme ja oppia parempi Q-taulukon arvojen jakauma, joka mahdollistaa omenan löytämisen paljon nopeammin.

## Q-oppimisen ydin: Bellmanin yhtälö

Kun alamme liikkua, jokaisella toiminnolla on vastaava palkkio, eli voimme teoriassa valita seuraavan toiminnon korkeimman välittömän palkkion perusteella. Useimmissa tiloissa siirto ei kuitenkaan saavuta tavoitettamme, eli omenan löytämistä, joten emme voi heti päättää, mikä suunta on parempi.

> Muista, että välitön tulos ei ole tärkein, vaan lopullinen tulos, jonka saamme simulaation lopussa.

Jotta voimme ottaa huomioon viivästyneen palkkion, meidän täytyy käyttää **[dynaamisen ohjelmoinnin](https://en.wikipedia.org/wiki/Dynamic_programming)** periaatteita, jotka mahdollistavat ongelman tarkastelun rekursiivisesti.

Oletetaan, että olemme nyt tilassa *s*, ja haluamme siirtyä seuraavaan tilaan *s'*. Tekemällä niin saamme välittömän palkkion *r(s,a)*, joka määritellään palkkiofunktiolla, plus jonkin tulevan palkkion. Jos oletamme, että Q-taulukkomme heijastaa oikein kunkin toiminnon "houkuttelevuuden", niin tilassa *s'* valitsemme toiminnon *a*, joka vastaa maksimiarvoa *Q(s',a')*. Näin ollen paras mahdollinen tuleva palkkio, jonka voimme saada tilassa *s*, määritellään `max`

## Politiikan tarkistaminen

Koska Q-taulukko listaa kunkin toiminnon "houkuttelevuuden" kussakin tilassa, sen avulla on melko helppoa määritellä tehokas navigointi maailmassamme. Yksinkertaisimmassa tapauksessa voimme valita toiminnon, joka vastaa korkeinta Q-taulukon arvoa: (koodilohko 9)

```python
def qpolicy_strict(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = list(actions)[np.argmax(v)]
        return a

walk(m,qpolicy_strict)
```

> Jos kokeilet yllä olevaa koodia useita kertoja, saatat huomata, että se joskus "jumittuu", ja sinun täytyy painaa STOP-painiketta keskeyttääksesi sen. Tämä johtuu siitä, että voi olla tilanteita, joissa kaksi tilaa "osoittavat" toisiaan optimaalisen Q-arvon suhteen, jolloin agentti päätyy liikkumaan näiden tilojen välillä loputtomasti.

## 🚀Haaste

> **Tehtävä 1:** Muokkaa `walk`-funktiota rajoittamaan polun maksimipituus tiettyyn askelmäärään (esimerkiksi 100), ja katso, kuinka yllä oleva koodi palauttaa tämän arvon ajoittain.

> **Tehtävä 2:** Muokkaa `walk`-funktiota niin, ettei se palaa paikkoihin, joissa se on jo aiemmin käynyt. Tämä estää `walk`-toiminnon silmukoinnin, mutta agentti voi silti päätyä "jumiin" paikkaan, josta se ei pääse pois.

## Navigointi

Parempi navigointipolitiikka olisi se, jota käytimme harjoittelun aikana, ja joka yhdistää hyödyntämisen ja tutkimisen. Tässä politiikassa valitsemme kunkin toiminnon tietyllä todennäköisyydellä, suhteessa Q-taulukon arvoihin. Tämä strategia voi silti johtaa siihen, että agentti palaa jo tutkittuun paikkaan, mutta kuten alla olevasta koodista näet, se johtaa hyvin lyhyeen keskimääräiseen polkuun haluttuun sijaintiin (muista, että `print_statistics` suorittaa simulaation 100 kertaa): (koodilohko 10)

```python
def qpolicy(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = random.choices(list(actions),weights=v)[0]
        return a

print_statistics(qpolicy)
```

Kun suoritat tämän koodin, saat paljon lyhyemmän keskimääräisen polun pituuden kuin aiemmin, noin 3-6 välillä.

## Oppimisprosessin tutkiminen

Kuten mainitsimme, oppimisprosessi on tasapaino tutkimisen ja ongelmatilan rakenteesta saadun tiedon hyödyntämisen välillä. Olemme nähneet, että oppimisen tulokset (kyky auttaa agenttia löytämään lyhyt polku tavoitteeseen) ovat parantuneet, mutta on myös mielenkiintoista tarkastella, miten keskimääräinen polun pituus käyttäytyy oppimisprosessin aikana:

## Oppimisen yhteenveto:

- **Keskimääräinen polun pituus kasvaa**. Aluksi keskimääräinen polun pituus kasvaa. Tämä johtuu todennäköisesti siitä, että kun emme tiedä ympäristöstä mitään, olemme todennäköisesti jumissa huonoissa tiloissa, kuten vedessä tai suden luona. Kun opimme lisää ja alamme käyttää tätä tietoa, voimme tutkia ympäristöä pidempään, mutta emme silti tiedä kovin hyvin, missä omenat ovat.

- **Polun pituus lyhenee oppimisen edetessä**. Kun opimme tarpeeksi, agentin on helpompi saavuttaa tavoite, ja polun pituus alkaa lyhentyä. Olemme kuitenkin edelleen avoimia tutkimiselle, joten poikkeamme usein parhaasta polusta ja tutkimme uusia vaihtoehtoja, mikä tekee polusta pidemmän kuin optimaalinen.

- **Pituus kasvaa äkillisesti**. Graafista näemme myös, että jossain vaiheessa pituus kasvaa äkillisesti. Tämä osoittaa prosessin satunnaisen luonteen, ja että voimme jossain vaiheessa "pilata" Q-taulukon kertoimet korvaamalla ne uusilla arvoilla. Tämä tulisi ihanteellisesti minimoida pienentämällä oppimisnopeutta (esimerkiksi harjoittelun loppuvaiheessa säädämme Q-taulukon arvoja vain pienellä arvolla).

Kaiken kaikkiaan on tärkeää muistaa, että oppimisprosessin onnistuminen ja laatu riippuvat merkittävästi parametreista, kuten oppimisnopeudesta, oppimisnopeuden vähenemisestä ja diskonttauskerroimesta. Näitä kutsutaan usein **hyperparametreiksi**, jotta ne erotetaan **parametreista**, joita optimoimme harjoittelun aikana (esimerkiksi Q-taulukon kertoimet). Parhaiden hyperparametriarvojen löytämistä kutsutaan **hyperparametrien optimoinniksi**, ja se ansaitsee oman aiheensa.

## [Luennon jälkeinen kysely](https://ff-quizzes.netlify.app/en/ml/)

## Tehtävä 
[Realistisempi maailma](assignment.md)

---

**Vastuuvapauslauseke**:  
Tämä asiakirja on käännetty käyttämällä tekoälypohjaista käännöspalvelua [Co-op Translator](https://github.com/Azure/co-op-translator). Vaikka pyrimme tarkkuuteen, huomioithan, että automaattiset käännökset voivat sisältää virheitä tai epätarkkuuksia. Alkuperäistä asiakirjaa sen alkuperäisellä kielellä tulisi pitää ensisijaisena lähteenä. Kriittisen tiedon osalta suositellaan ammattimaista ihmiskäännöstä. Emme ole vastuussa väärinkäsityksistä tai virhetulkinnoista, jotka johtuvat tämän käännöksen käytöstä.