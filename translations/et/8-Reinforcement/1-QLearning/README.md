<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "911efd5e595089000cb3c16fce1beab8",
  "translation_date": "2025-10-11T11:19:54+00:00",
  "source_file": "8-Reinforcement/1-QLearning/README.md",
  "language_code": "et"
}
-->
# Sissejuhatus tugevdusõppesse ja Q-õppesse

![Tugevdusõppe kokkuvõte masinõppes sketchnote'is](../../../../translated_images/ml-reinforcement.94024374d63348db.et.png)
> Sketchnote autor: [Tomomi Imura](https://www.twitter.com/girlie_mac)

Tugevdusõpe hõlmab kolme olulist mõistet: agent, teatud seisundid ja tegevuste kogum iga seisundi kohta. Kui agent sooritab kindlas seisundis tegevuse, saab ta tasu. Kujutle näiteks arvutimängu Super Mario. Sina oled Mario, oled mängutasemel ja seisad kaljuserval. Sinu kohal on münt. Sina, olles Mario, mängutasemel kindlas asukohas ... see on sinu seisund. Kui liigud ühe sammu paremale (tegevus), kukud kaljult alla ja saad madala punktisumma. Kui aga vajutad hüppenuppu, saad punkti ja jääd ellu. See on positiivne tulemus ja selle eest peaksid saama positiivse punktisumma.

Kasutades tugevdusõpet ja simulaatorit (mängu), saad õppida, kuidas mängu mängida, et maksimeerida tasu, mis tähendab ellu jäämist ja võimalikult paljude punktide kogumist.

[![Sissejuhatus tugevdusõppesse](https://img.youtube.com/vi/lDq_en8RNOo/0.jpg)](https://www.youtube.com/watch?v=lDq_en8RNOo)

> 🎥 Klõpsa ülaloleval pildil, et kuulata Dmitryt rääkimas tugevdusõppest

## [Eelloengu viktoriin](https://ff-quizzes.netlify.app/en/ml/)

## Eeltingimused ja seadistamine

Selles õppetükis katsetame mõningaid Pythonis kirjutatud koode. Sa peaksid suutma käivitada selle õppetüki Jupyter Notebooki koodi kas oma arvutis või pilves.

Sa saad avada [õppetüki märkmiku](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/notebook.ipynb) ja järgida seda õppetükki, et ehitada.

> **Märkus:** Kui avad selle koodi pilvest, pead alla laadima ka faili [`rlboard.py`](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/rlboard.py), mida kasutatakse märkmiku koodis. Lisa see samasse kausta, kus märkmik asub.

## Sissejuhatus

Selles õppetükis uurime **[Peeter ja hunt](https://en.wikipedia.org/wiki/Peter_and_the_Wolf)** maailma, mis on inspireeritud vene helilooja [Sergei Prokofjevi](https://en.wikipedia.org/wiki/Sergei_Prokofiev) muusikalisest muinasjutust. Kasutame **tugevdusõpet**, et lasta Peetril oma keskkonda avastada, maitsvaid õunu korjata ja hundiga kohtumist vältida.

**Tugevdusõpe** (RL) on õppetehnika, mis võimaldab meil õppida **agendi** optimaalset käitumist mingis **keskkonnas**, viies läbi palju katseid. Agent peaks selles keskkonnas omama mingit **eesmärki**, mis on määratletud **tasufunktsiooni** kaudu.

## Keskkond

Lihtsuse huvides kujutame ette, et Peetri maailm on ruudukujuline laud mõõtmetega `laius` x `kõrgus`, mis näeb välja selline:

![Peetri keskkond](../../../../translated_images/environment.40ba3cb66256c93f.et.png)

Iga laua ruut võib olla:

* **maa**, millel Peeter ja teised olendid saavad kõndida.
* **vesi**, millel ilmselgelt ei saa kõndida.
* **puu** või **rohi**, koht, kus saab puhata.
* **õun**, mida Peeter oleks rõõmus leida, et end toita.
* **hunt**, mis on ohtlik ja mida tuleks vältida.

On olemas eraldi Python moodul, [`rlboard.py`](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/rlboard.py), mis sisaldab koodi selle keskkonnaga töötamiseks. Kuna see kood ei ole meie kontseptsioonide mõistmiseks oluline, impordime mooduli ja kasutame seda näidislaua loomiseks (koodilõik 1):

```python
from rlboard import *

width, height = 8,8
m = Board(width,height)
m.randomize(seed=13)
m.plot()
```

See kood peaks printima keskkonna pildi, mis sarnaneb ülaltooduga.

## Tegevused ja poliitika

Meie näites on Peetri eesmärk leida õun, vältides samal ajal hunti ja muid takistusi. Selleks saab ta põhimõtteliselt ringi liikuda, kuni leiab õuna.

Seega võib ta igas asukohas valida ühe järgmistest tegevustest: üles, alla, vasakule ja paremale.

Määratleme need tegevused sõnastikuna ja seome need vastavate koordinaatide muutustega. Näiteks paremale liikumine (`R`) vastab paarile `(1,0)`. (koodilõik 2):

```python
actions = { "U" : (0,-1), "D" : (0,1), "L" : (-1,0), "R" : (1,0) }
action_idx = { a : i for i,a in enumerate(actions.keys()) }
```

Kokkuvõtteks on selle stsenaariumi strateegia ja eesmärk järgmised:

- **Strateegia**, meie agendi (Peetri) jaoks on määratletud nn **poliitikaga**. Poliitika on funktsioon, mis tagastab tegevuse igas antud seisundis. Meie puhul on probleemi seisund esindatud laua ja mängija praeguse asukohaga.

- **Eesmärk**, tugevdusõppe eesmärk on lõpuks õppida hea poliitika, mis võimaldab meil probleemi tõhusalt lahendada. Kuid alustuseks kaalume kõige lihtsamat poliitikat, mida nimetatakse **juhuslikuks kõndimiseks**.

## Juhuslik kõndimine

Lahendame oma probleemi esmalt juhusliku kõndimise strateegia abil. Juhusliku kõndimise korral valime lubatud tegevuste hulgast juhuslikult järgmise tegevuse, kuni jõuame õunani (koodilõik 3).

1. Rakenda juhuslik kõndimine alloleva koodiga:

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

   `walk`-funktsiooni väljakutse peaks tagastama vastava tee pikkuse, mis võib ühest käivituskorrast teise varieeruda.

1. Käivita kõndimiskatse mitu korda (näiteks 100) ja prindi tulemuste statistika (koodilõik 4):

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

   Pane tähele, et tee keskmine pikkus on umbes 30-40 sammu, mis on üsna palju, arvestades, et keskmine kaugus lähima õunani on umbes 5-6 sammu.

   Samuti saad vaadata, kuidas Peetri liikumine juhusliku kõndimise ajal välja näeb:

   ![Peetri juhuslik kõndimine](../../../../8-Reinforcement/1-QLearning/images/random_walk.gif)

## Tasufunktsioon

Selleks, et meie poliitika oleks intelligentsem, peame mõistma, millised liigutused on "paremad" kui teised. Selleks peame määratlema oma eesmärgi.

Eesmärki saab määratleda **tasufunktsiooni** kaudu, mis tagastab iga seisundi kohta mingi punktisumma. Mida suurem on number, seda parem on tasufunktsioon. (koodilõik 5)

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

Huvitav on see, et tasufunktsioonide puhul antakse enamikul juhtudel *oluline tasu alles mängu lõpus*. See tähendab, et meie algoritm peaks kuidagi meelde jätma "head" sammud, mis viivad positiivse tasuni lõpus, ja suurendama nende tähtsust. Samamoodi tuleks kõik liigutused, mis viivad halva tulemuseni, maha suruda.

## Q-õpe

Algoritmi, mida siin arutame, nimetatakse **Q-õppeks**. Selles algoritmis on poliitika määratletud funktsiooni (või andmestruktuuri) kaudu, mida nimetatakse **Q-tabeliks**. See salvestab iga tegevuse "headuse" antud seisundis.

Seda nimetatakse Q-tabeliks, kuna seda on sageli mugav esitada tabelina või mitmemõõtmelise massiivina. Kuna meie laual on mõõtmed `laius` x `kõrgus`, saame Q-tabelit esitada numpy massiivina kujuga `laius` x `kõrgus` x `len(actions)`: (koodilõik 6)

```python
Q = np.ones((width,height,len(actions)),dtype=np.float)*1.0/len(actions)
```

Pange tähele, et algväärtustame kõik Q-tabeli väärtused võrdse väärtusega, meie puhul - 0.25. See vastab "juhusliku kõndimise" poliitikale, kuna kõik liigutused igas seisundis on võrdselt head. Saame Q-tabeli edastada `plot` funktsioonile, et visualiseerida tabelit laual: `m.plot(Q)`.

![Peetri keskkond](../../../../translated_images/env_init.04e8f26d2d60089e.et.png)

Iga ruudu keskel on "nooleke", mis näitab eelistatud liikumissuunda. Kuna kõik suunad on võrdsed, kuvatakse punkt.

Nüüd peame käivitama simulatsiooni, uurima oma keskkonda ja õppima paremat Q-tabeli väärtuste jaotust, mis võimaldab meil õunani jõuda palju kiiremini.

## Q-õppe olemus: Bellmani võrrand

Kui hakkame liikuma, on igal tegevusel vastav tasu, st me saame teoreetiliselt valida järgmise tegevuse, lähtudes kõrgeimast kohesest tasust. Kuid enamikus seisundites ei saavuta liigutus meie eesmärki jõuda õunani, mistõttu ei saa me kohe otsustada, milline suund on parem.

> Pea meeles, et oluline pole kohene tulemus, vaid lõpptulemus, mille saame simulatsiooni lõpus.

Selleks, et arvestada viivitusega tasu, peame kasutama **[dünaamilise programmeerimise](https://en.wikipedia.org/wiki/Dynamic_programming)** põhimõtteid, mis võimaldavad meil probleemi käsitleda rekursiivselt.

Oletame, et oleme nüüd seisundis *s* ja tahame liikuda järgmisesse seisundisse *s'*. Seda tehes saame kohese tasu *r(s,a)*, mis on määratletud tasufunktsiooni kaudu, pluss mingi tulevane tasu. Kui eeldame, et meie Q-tabel kajastab õigesti iga tegevuse "atraktiivsust", siis seisundis *s'* valime tegevuse *a*, mis vastab maksimaalsele väärtusele *Q(s',a')*. Seega on parim võimalik tulevane tasu, mida me seisundis *s* saame, määratletud kui `max`<sub>a'</sub>*Q(s',a')* (maksimum arvutatakse siin kõigi võimalike tegevuste *a'* üle seisundis *s'*).

See annab **Bellmani valemi**, mille abil arvutada Q-tabeli väärtust seisundis *s*, arvestades tegevust *a*:

<img src="../../../../translated_images/bellman-equation.7c0c4c722e5a6b7c.et.png"/>

Siin γ on nn **diskonteerimistegur**, mis määrab, mil määral peaks eelistama praegust tasu tulevase tasu ees ja vastupidi.

## Õppealgoritm

Eeltoodud valemi põhjal saame nüüd kirjutada oma õppealgoritmi pseudokoodi:

* Algväärtusta Q-tabel Q võrdsete numbritega kõigi seisundite ja tegevuste jaoks
* Määra õppemäär α ← 1
* Korda simulatsiooni mitu korda
   1. Alusta juhuslikust asukohast
   1. Korda
        1. Vali tegevus *a* seisundis *s*
        2. Soorita tegevus, liikudes uude seisundisse *s'*
        3. Kui kohtame mängu lõpu tingimust või kogutud tasu on liiga väike - lõpeta simulatsioon  
        4. Arvuta tasu *r* uues seisundis
        5. Uuenda Q-funktsiooni vastavalt Bellmani valemile: *Q(s,a)* ← *(1-α)Q(s,a)+α(r+γ max<sub>a'</sub>Q(s',a'))*
        6. *s* ← *s'*
        7. Uuenda kogutud tasu ja vähenda α.

## Kasutamine vs. avastamine

Ülaltoodud algoritmis ei täpsustanud me, kuidas täpselt peaksime valima tegevuse sammus 2.1. Kui valime tegevuse juhuslikult, **avastame** me keskkonda juhuslikult ja tõenäoliselt sureme sageli ning uurime ka piirkondi, kuhu me tavaliselt ei läheks. Alternatiivne lähenemine oleks **kasutada** Q-tabeli väärtusi, mida me juba teame, ja valida seega parim tegevus (kõrgema Q-tabeli väärtusega) seisundis *s*. See aga takistab meil uurida teisi seisundeid ja tõenäoliselt ei leia me optimaalset lahendust.

Seega on parim lähenemine leida tasakaal avastamise ja kasutamise vahel. Seda saab teha, valides tegevuse seisundis *s* tõenäosustega, mis on proportsionaalsed Q-tabeli väärtustega. Alguses, kui Q-tabeli väärtused on kõik ühesugused, vastab see juhuslikule valikule, kuid mida rohkem me oma keskkonna kohta õpime, seda tõenäolisemalt järgime optimaalset teed, lubades agendil aeg-ajalt valida ka uurimata tee.

## Pythoni implementatsioon

Nüüd oleme valmis õppealgoritmi rakendama. Enne seda vajame ka mõnda funktsiooni, mis teisendab Q-tabeli suvalised numbrid vastavate tegevuste tõenäosusvektoriks.

1. Loo funktsioon `probs()`:

    ```python
    def probs(v,eps=1e-4):
        v = v-v.min()+eps
        v = v/v.sum()
        return v
    ```

   Lisame algsele vektorile mõned `eps`, et vältida jagamist nulliga algses olukorras, kui kõik vektori komponendid on identsed.

Käivita õppealgoritm läbi 5000 eksperimendi, mida nimetatakse ka **epohhideks**: (koodilõik 8)
```python
    for epoch in range(5000):
    
        # Pick initial point
        m.random_start()
        
        # Start travelling
        n=0
        cum_reward = 0
        while True:
            x,y = m.human
            v = probs(Q[x,y])
            a = random.choices(list(actions),weights=v)[0]
            dpos = actions[a]
            m.move(dpos,check_correctness=False) # we allow player to move outside the board, which terminates episode
            r = reward(m)
            cum_reward += r
            if r==end_reward or cum_reward < -1000:
                lpath.append(n)
                break
            alpha = np.exp(-n / 10e5)
            gamma = 0.5
            ai = action_idx[a]
            Q[x,y,ai] = (1 - alpha) * Q[x,y,ai] + alpha * (r + gamma * Q[x+dpos[0], y+dpos[1]].max())
            n+=1
```

Pärast selle algoritmi täitmist peaks Q-tabel olema uuendatud väärtustega, mis määratlevad erinevate tegevuste atraktiivsuse igas etapis. Saame proovida Q-tabelit visualiseerida, joonistades igasse ruutu vektori, mis osutab soovitud liikumissuunda. Lihtsuse huvides joonistame noolepea asemel väikese ringi.

<img src="../../../../translated_images/learned.ed28bcd8484b5287.et.png"/>

## Poliitika kontrollimine

Kuna Q-tabel loetleb iga tegevuse "atraktiivsuse" igas seisundis, on seda üsna lihtne kasutada tõhusa navigeerimise määratlemiseks meie maailmas. Lihtsaimas variandis saame valida tegevuse, mis vastab kõrgeimale Q-tabeli väärtusele: (koodilõik 9)

```python
def qpolicy_strict(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = list(actions)[np.argmax(v)]
        return a

walk(m,qpolicy_strict)
```

> Kui proovite ülaltoodud koodi mitu korda, võite märgata, et see mõnikord "hangub" ja peate vajutama märkmikus STOP-nuppu, et see katkestada. See juhtub, kuna võivad tekkida olukorrad, kus kaks olekut "osutavad" üksteisele optimaalse Q-väärtuse osas, mille tulemusel liigub agent nende olekute vahel lõputult edasi-tagasi.

## 🚀Väljakutse

> **Ülesanne 1:** Muutke funktsiooni `walk`, et piirata maksimaalset teekonna pikkust teatud arvu sammudega (näiteks 100) ja vaadake, kuidas ülaltoodud kood aeg-ajalt selle väärtuse tagastab.

> **Ülesanne 2:** Muutke funktsiooni `walk`, et see ei läheks tagasi kohtadesse, kus ta on juba varem olnud. See takistab `walk` funktsiooni tsüklisse jäämist, kuid agent võib siiski sattuda olukorda, kus ta ei suuda asukohast välja pääseda.

## Navigeerimine

Parem navigeerimispoliitika oleks see, mida kasutasime treeningu ajal, mis ühendab ekspluateerimise ja uurimise. Selle poliitika puhul valime iga tegevuse teatud tõenäosusega, mis on proportsionaalne Q-tabeli väärtustega. See strateegia võib siiski viia agendi tagasi positsioonile, mida ta on juba uurinud, kuid nagu näete allolevast koodist, viib see väga lühikese keskmise teekonna pikkuseni soovitud asukohta (pidage meeles, et `print_statistics` käivitab simulatsiooni 100 korda): (koodiplokk 10)

```python
def qpolicy(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = random.choices(list(actions),weights=v)[0]
        return a

print_statistics(qpolicy)
```

Pärast selle koodi käivitamist peaksite saama palju väiksema keskmise teekonna pikkuse kui varem, vahemikus 3–6.

## Õppimisprotsessi uurimine

Nagu mainitud, on õppimisprotsess tasakaal uurimise ja olemasoleva teadmise rakendamise vahel probleemiruumi struktuuri kohta. Oleme näinud, et õppimise tulemused (võime aidata agenti leida lühike tee eesmärgini) on paranenud, kuid huvitav on ka jälgida, kuidas keskmine teekonna pikkus käitub õppimisprotsessi ajal:

<img src="../../../../translated_images/lpathlen1.0534784add58d4eb.et.png"/>

Õppimisprotsessi saab kokku võtta järgmiselt:

- **Keskmine teekonna pikkus suureneb**. Alguses suureneb keskmine teekonna pikkus. Tõenäoliselt on see tingitud asjaolust, et kui me ei tea keskkonnast midagi, jääme tõenäoliselt kinni halbadest olekutest, nagu vesi või hunt. Kui õpime rohkem ja hakkame seda teadmist kasutama, saame keskkonda kauem uurida, kuid me ei tea veel hästi, kus õunad asuvad.

- **Teekonna pikkus väheneb, kui õpime rohkem**. Kui oleme piisavalt õppinud, muutub agendil eesmärgi saavutamine lihtsamaks ja teekonna pikkus hakkab vähenema. Kuid oleme endiselt avatud uurimisele, mistõttu kaldume sageli parimast teest kõrvale ja uurime uusi võimalusi, muutes teekonna pikemaks kui optimaalne.

- **Pikkus suureneb järsult**. Graafikul näeme ka, et mingil hetkel pikkus suurenes järsult. See näitab protsessi juhuslikku olemust ja seda, et võime mingil hetkel "rikkuda" Q-tabeli koefitsiendid, kirjutades neile uued väärtused üle. Seda tuleks ideaalis minimeerida õppemäära vähendamisega (näiteks treeningu lõpus kohandame Q-tabeli väärtusi ainult väikese väärtusega).

Üldiselt on oluline meeles pidada, et õppimisprotsessi edu ja kvaliteet sõltuvad oluliselt parameetritest, nagu õppemäär, õppemäära kahanemine ja diskontomäär. Neid nimetatakse sageli **hüperparameetriteks**, et eristada neid **parameetritest**, mida optimeerime treeningu ajal (näiteks Q-tabeli koefitsiendid). Parimate hüperparameetrite väärtuste leidmise protsessi nimetatakse **hüperparameetrite optimeerimiseks** ja see väärib eraldi teemat.

## [Loengu järgne viktoriin](https://ff-quizzes.netlify.app/en/ml/)

## Ülesanne 
[Realistlikum maailm](assignment.md)

---

**Lahtiütlus**:  
See dokument on tõlgitud AI tõlketeenuse [Co-op Translator](https://github.com/Azure/co-op-translator) abil. Kuigi püüame tagada täpsust, palume arvestada, et automaatsed tõlked võivad sisaldada vigu või ebatäpsusi. Algne dokument selle algses keeles tuleks pidada autoriteetseks allikaks. Olulise teabe puhul soovitame kasutada professionaalset inimtõlget. Me ei vastuta selle tõlke kasutamisest tulenevate arusaamatuste või valesti tõlgenduste eest.