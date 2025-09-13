<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "911efd5e595089000cb3c16fce1beab8",
  "translation_date": "2025-09-05T08:02:11+00:00",
  "source_file": "8-Reinforcement/1-QLearning/README.md",
  "language_code": "lt"
}
-->
# Įvadas į stiprinamąjį mokymą ir Q-mokymą

![Stiprinamojo mokymosi santrauka sketchnote](../../../../sketchnotes/ml-reinforcement.png)
> Sketchnote sukūrė [Tomomi Imura](https://www.twitter.com/girlie_mac)

Stiprinamasis mokymasis apima tris svarbias sąvokas: agentą, tam tikras būsenas ir veiksmų rinkinį kiekvienai būsenai. Atlikdamas veiksmą tam tikroje būsenoje, agentas gauna atlygį. Įsivaizduokite kompiuterinį žaidimą „Super Mario“. Jūs esate Mario, esate žaidimo lygyje, stovite šalia uolos krašto. Virš jūsų yra moneta. Jūs, būdamas Mario, tam tikroje žaidimo lygio pozicijoje... tai yra jūsų būsena. Žengus žingsnį į dešinę (veiksmas), nukristumėte nuo uolos, ir tai suteiktų jums mažą skaitinį rezultatą. Tačiau paspaudus šuolio mygtuką, pelnytumėte tašką ir išliktumėte gyvas. Tai yra teigiamas rezultatas, kuris turėtų suteikti jums teigiamą skaitinį rezultatą.

Naudodami stiprinamąjį mokymą ir simuliatorių (žaidimą), galite išmokti žaisti žaidimą taip, kad maksimaliai padidintumėte atlygį, t. y. išliktumėte gyvas ir surinktumėte kuo daugiau taškų.

[![Įvadas į stiprinamąjį mokymą](https://img.youtube.com/vi/lDq_en8RNOo/0.jpg)](https://www.youtube.com/watch?v=lDq_en8RNOo)

> 🎥 Spustelėkite aukščiau esančią nuotrauką, kad išgirstumėte Dmitrijų kalbant apie stiprinamąjį mokymą

## [Prieš paskaitos testas](https://ff-quizzes.netlify.app/en/ml/)

## Reikalavimai ir paruošimas

Šioje pamokoje eksperimentuosime su Python kodu. Turėtumėte sugebėti paleisti Jupyter Notebook kodą iš šios pamokos, arba savo kompiuteryje, arba debesyje.

Galite atidaryti [pamokos užrašų knygelę](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/notebook.ipynb) ir peržiūrėti šią pamoką, kad ją sukurtumėte.

> **Pastaba:** Jei atidarote šį kodą iš debesies, taip pat turite gauti [`rlboard.py`](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/rlboard.py) failą, kuris naudojamas užrašų knygelės kode. Įdėkite jį į tą patį katalogą kaip ir užrašų knygelę.

## Įvadas

Šioje pamokoje tyrinėsime **[Petro ir vilko](https://en.wikipedia.org/wiki/Peter_and_the_Wolf)** pasaulį, įkvėptą muzikinės pasakos, sukurtos rusų kompozitoriaus [Sergejaus Prokofjevo](https://en.wikipedia.org/wiki/Sergei_Prokofiev). Naudosime **stiprinamąjį mokymą**, kad Petras galėtų tyrinėti savo aplinką, rinkti skanius obuolius ir vengti susitikimo su vilku.

**Stiprinamasis mokymasis** (RL) yra mokymosi technika, leidžianti išmokti optimalaus **agento** elgesio tam tikroje **aplinkoje**, atliekant daugybę eksperimentų. Agentas šioje aplinkoje turėtų turėti tam tikrą **tikslą**, apibrėžtą **atlygio funkcija**.

## Aplinka

Paprasčiausiai, įsivaizduokime Petro pasaulį kaip kvadratinę lentą, kurios dydis yra `plotis` x `aukštis`, panašiai kaip ši:

![Petro aplinka](../../../../8-Reinforcement/1-QLearning/images/environment.png)

Kiekviena ląstelė šioje lentoje gali būti:

* **žemė**, ant kurios Petras ir kiti padarai gali vaikščioti.
* **vanduo**, ant kurio, akivaizdu, negalima vaikščioti.
* **medis** arba **žolė**, vieta, kur galima pailsėti.
* **obuolys**, kurį Petras norėtų rasti, kad pasimaitintų.
* **vilkas**, kuris yra pavojingas ir kurio reikėtų vengti.

Yra atskiras Python modulis, [`rlboard.py`](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/rlboard.py), kuriame yra kodas, skirtas dirbti su šia aplinka. Kadangi šis kodas nėra svarbus mūsų koncepcijoms suprasti, mes importuosime modulį ir naudosime jį, kad sukurtume pavyzdinę lentą (kodo blokas 1):

```python
from rlboard import *

width, height = 8,8
m = Board(width,height)
m.randomize(seed=13)
m.plot()
```

Šis kodas turėtų atspausdinti aplinkos vaizdą, panašų į aukščiau pateiktą.

## Veiksmai ir politika

Mūsų pavyzdyje Petro tikslas būtų rasti obuolį, vengiant vilko ir kitų kliūčių. Tam jis iš esmės gali vaikščioti, kol suras obuolį.

Todėl bet kurioje pozicijoje jis gali pasirinkti vieną iš šių veiksmų: aukštyn, žemyn, kairėn ir dešinėn.

Šiuos veiksmus apibrėšime kaip žodyną ir susiesime juos su atitinkamais koordinačių pokyčių poromis. Pavyzdžiui, judėjimas dešinėn (`R`) atitiktų porą `(1,0)`. (kodo blokas 2):

```python
actions = { "U" : (0,-1), "D" : (0,1), "L" : (-1,0), "R" : (1,0) }
action_idx = { a : i for i,a in enumerate(actions.keys()) }
```

Apibendrinant, šio scenarijaus strategija ir tikslas yra tokie:

- **Strategija**, mūsų agento (Petro) yra apibrėžta vadinamąja **politika**. Politika yra funkcija, kuri grąžina veiksmą bet kurioje būsenoje. Mūsų atveju problemos būsena yra lentos vaizdas, įskaitant žaidėjo dabartinę poziciją.

- **Tikslas**, stiprinamojo mokymosi yra galiausiai išmokti gerą politiką, kuri leis efektyviai išspręsti problemą. Tačiau kaip pagrindą, apsvarstykime paprasčiausią politiką, vadinamą **atsitiktiniu vaikščiojimu**.

## Atsitiktinis vaikščiojimas

Pirmiausia išspręskime mūsų problemą įgyvendindami atsitiktinio vaikščiojimo strategiją. Naudodami atsitiktinį vaikščiojimą, atsitiktinai pasirinksime kitą veiksmą iš leidžiamų veiksmų, kol pasieksime obuolį (kodo blokas 3).

1. Įgyvendinkite atsitiktinį vaikščiojimą naudodami žemiau pateiktą kodą:

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

    Funkcijos `walk` iškvietimas turėtų grąžinti atitinkamo kelio ilgį, kuris gali skirtis kiekvieno paleidimo metu.

1. Paleiskite vaikščiojimo eksperimentą kelis kartus (pvz., 100) ir atspausdinkite gautą statistiką (kodo blokas 4):

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

    Atkreipkite dėmesį, kad vidutinis kelio ilgis yra apie 30–40 žingsnių, o tai yra gana daug, atsižvelgiant į tai, kad vidutinis atstumas iki artimiausio obuolio yra apie 5–6 žingsnius.

    Taip pat galite pamatyti, kaip atrodo Petro judėjimas atsitiktinio vaikščiojimo metu:

    ![Petro atsitiktinis vaikščiojimas](../../../../8-Reinforcement/1-QLearning/images/random_walk.gif)

## Atlygio funkcija

Kad mūsų politika būtų protingesnė, turime suprasti, kurie judesiai yra „geresni“ už kitus. Tam reikia apibrėžti mūsų tikslą.

Tikslas gali būti apibrėžtas **atlygio funkcijos** forma, kuri grąžins tam tikrą balo reikšmę kiekvienai būsenai. Kuo didesnis skaičius, tuo geresnė atlygio funkcija. (kodo blokas 5)

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

Įdomus dalykas apie atlygio funkcijas yra tas, kad daugeliu atvejų *mes gauname reikšmingą atlygį tik žaidimo pabaigoje*. Tai reiškia, kad mūsų algoritmas turėtų kažkaip prisiminti „gerus“ žingsnius, kurie veda į teigiamą atlygį pabaigoje, ir padidinti jų svarbą. Panašiai visi judesiai, kurie veda į blogus rezultatus, turėtų būti atgrasomi.

## Q-mokymasis

Algoritmas, kurį aptarsime čia, vadinamas **Q-mokymu**. Šiame algoritme politika apibrėžiama funkcija (arba duomenų struktūra), vadinama **Q-lentele**. Ji registruoja kiekvieno veiksmo „gerumą“ tam tikroje būsenoje.

Ji vadinama Q-lentele, nes dažnai patogu ją vaizduoti kaip lentelę arba daugiamačio masyvo formą. Kadangi mūsų lenta turi matmenis `plotis` x `aukštis`, Q-lentelę galime vaizduoti naudodami numpy masyvą, kurio forma yra `plotis` x `aukštis` x `len(veiksmai)`: (kodo blokas 6)

```python
Q = np.ones((width,height,len(actions)),dtype=np.float)*1.0/len(actions)
```

Atkreipkite dėmesį, kad mes inicializuojame visas Q-lentelės reikšmes lygiomis reikšmėmis, mūsų atveju - 0.25. Tai atitinka „atsitiktinio vaikščiojimo“ politiką, nes visi judesiai kiekvienoje būsenoje yra vienodai geri. Q-lentelę galime perduoti funkcijai `plot`, kad vizualizuotume lentelę lentoje: `m.plot(Q)`.

![Petro aplinka](../../../../8-Reinforcement/1-QLearning/images/env_init.png)

Kiekvienos ląstelės centre yra „rodyklė“, rodanti pageidaujamą judėjimo kryptį. Kadangi visos kryptys yra vienodos, rodomas taškas.

Dabar turime paleisti simuliaciją, tyrinėti savo aplinką ir išmokti geresnį Q-lentelės reikšmių pasiskirstymą, kuris leis mums daug greičiau rasti kelią iki obuolio.

## Q-mokymosi esmė: Bellmano lygtis

Kai pradedame judėti, kiekvienas veiksmas turės atitinkamą atlygį, t. y. teoriškai galime pasirinkti kitą veiksmą pagal didžiausią tiesioginį atlygį. Tačiau daugumoje būsenų judesys nepasieks mūsų tikslo pasiekti obuolį, todėl negalime iš karto nuspręsti, kuri kryptis yra geresnė.

> Atminkite, kad svarbus ne tiesioginis rezultatas, o galutinis rezultatas, kurį gausime simuliacijos pabaigoje.

Kad atsižvelgtume į šį uždelstą atlygį, turime naudoti **[dinaminio programavimo](https://en.wikipedia.org/wiki/Dynamic_programming)** principus, kurie leidžia apie problemą galvoti rekursyviai.

Tarkime, kad dabar esame būsenoje *s*, ir norime pereiti į kitą būseną *s'*. Tai darydami gausime tiesioginį atlygį *r(s,a)*, apibrėžtą atlygio funkcija, plius tam tikrą būsimą atlygį. Jei manome, kad mūsų Q-lentelė teisingai atspindi kiekvieno veiksmo „patrauklumą“, tada būsenoje *s'* pasirinksime veiksmą *a*, kuris atitinka didžiausią *Q(s',a')* reikšmę. Taigi, geriausias galimas būsimas atlygis, kurį galėtume gauti būsenoje *s*, bus apibrėžtas kaip `max`

## Tikriname politiką

Kadangi Q-lentelėje pateikiamas kiekvieno veiksmo "patrauklumas" kiekvienoje būsenoje, ją gana lengva naudoti efektyviam navigavimui mūsų pasaulyje apibrėžti. Paprasčiausiu atveju galime pasirinkti veiksmą, atitinkantį didžiausią Q-lentelės reikšmę: (kodo blokas 9)

```python
def qpolicy_strict(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = list(actions)[np.argmax(v)]
        return a

walk(m,qpolicy_strict)
```

> Jei kelis kartus paleisite aukščiau pateiktą kodą, galite pastebėti, kad kartais jis "užstringa", ir jums reikia paspausti STOP mygtuką užrašinėje, kad jį nutrauktumėte. Taip nutinka, nes gali būti situacijų, kai dvi būsenos "rodo" viena į kitą pagal optimalią Q-reikšmę, tokiu atveju agentas nuolat juda tarp tų būsenų.

## 🚀Iššūkis

> **Užduotis 1:** Pakeiskite `walk` funkciją taip, kad kelio ilgis būtų apribotas tam tikru žingsnių skaičiumi (pvz., 100), ir stebėkite, kaip aukščiau pateiktas kodas kartais grąžina šią reikšmę.

> **Užduotis 2:** Pakeiskite `walk` funkciją taip, kad ji negrįžtų į vietas, kuriose jau buvo. Tai neleis `walk` funkcijai užstrigti cikle, tačiau agentas vis tiek gali "įstrigti" vietoje, iš kurios negali pabėgti.

## Navigacija

Geresnė navigacijos politika būtų ta, kurią naudojome mokymo metu, derinant išnaudojimą ir tyrinėjimą. Pagal šią politiką kiekvienas veiksmas pasirenkamas su tam tikra tikimybe, proporcinga Q-lentelės reikšmėms. Ši strategija vis dar gali lemti, kad agentas grįš į jau ištirtą vietą, tačiau, kaip matote iš žemiau pateikto kodo, ji lemia labai trumpą vidutinį kelią iki norimos vietos (prisiminkite, kad `print_statistics` paleidžia simuliaciją 100 kartų): (kodo blokas 10)

```python
def qpolicy(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = random.choices(list(actions),weights=v)[0]
        return a

print_statistics(qpolicy)
```

Paleidus šį kodą, turėtumėte gauti daug mažesnį vidutinį kelio ilgį nei anksčiau, maždaug 3-6 diapazone.

## Mokymosi proceso tyrinėjimas

Kaip jau minėjome, mokymosi procesas yra pusiausvyra tarp tyrinėjimo ir sukauptų žinių apie problemos erdvės struktūrą išnaudojimo. Matėme, kad mokymosi rezultatai (gebėjimas padėti agentui rasti trumpą kelią iki tikslo) pagerėjo, tačiau taip pat įdomu stebėti, kaip vidutinis kelio ilgis keičiasi mokymosi proceso metu:

## Mokymosi apibendrinimas:

- **Vidutinis kelio ilgis didėja.** Iš pradžių matome, kad vidutinis kelio ilgis didėja. Taip tikriausiai yra todėl, kad, kai nieko nežinome apie aplinką, esame linkę įstrigti blogose būsenose, tokiose kaip vanduo ar vilkas. Kai daugiau sužinome ir pradedame naudoti šias žinias, galime ilgiau tyrinėti aplinką, tačiau vis dar nelabai žinome, kur yra obuoliai.

- **Kelio ilgis mažėja, kai daugiau išmokstame.** Kai pakankamai išmokstame, agentui tampa lengviau pasiekti tikslą, ir kelio ilgis pradeda mažėti. Tačiau mes vis dar atviri tyrinėjimui, todėl dažnai nukrypstame nuo geriausio kelio ir tyrinėjame naujas galimybes, dėl ko kelias tampa ilgesnis nei optimalus.

- **Ilgis staiga padidėja.** Grafike taip pat pastebime, kad tam tikru momentu ilgis staiga padidėja. Tai rodo proceso stochastiškumą ir tai, kad tam tikru momentu galime "sugadinti" Q-lentelės koeficientus, perrašydami juos naujomis reikšmėmis. Tai idealiai turėtų būti sumažinta mažinant mokymosi greitį (pavyzdžiui, mokymo pabaigoje Q-lentelės reikšmes koreguojame tik nedidele verte).

Apskritai svarbu prisiminti, kad mokymosi proceso sėkmė ir kokybė labai priklauso nuo parametrų, tokių kaip mokymosi greitis, mokymosi greičio mažėjimas ir diskonto faktorius. Šie parametrai dažnai vadinami **hiperparametrais**, kad būtų atskirti nuo **parametrų**, kuriuos optimizuojame mokymo metu (pavyzdžiui, Q-lentelės koeficientai). Geriausių hiperparametrų reikšmių paieškos procesas vadinamas **hiperparametrų optimizavimu**, ir tai yra atskira tema.

## [Po paskaitos testas](https://ff-quizzes.netlify.app/en/ml/)

## Užduotis 
[Daugiau realistiškas pasaulis](assignment.md)

---

**Atsakomybės apribojimas**:  
Šis dokumentas buvo išverstas naudojant AI vertimo paslaugą [Co-op Translator](https://github.com/Azure/co-op-translator). Nors siekiame tikslumo, prašome atkreipti dėmesį, kad automatiniai vertimai gali turėti klaidų ar netikslumų. Originalus dokumentas jo gimtąja kalba turėtų būti laikomas autoritetingu šaltiniu. Kritinei informacijai rekomenduojama naudoti profesionalų žmogaus vertimą. Mes neprisiimame atsakomybės už nesusipratimus ar klaidingus interpretavimus, atsiradusius dėl šio vertimo naudojimo.