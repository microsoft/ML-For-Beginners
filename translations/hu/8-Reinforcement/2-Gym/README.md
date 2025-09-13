<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "107d5bb29da8a562e7ae72262d251a75",
  "translation_date": "2025-09-05T16:45:21+00:00",
  "source_file": "8-Reinforcement/2-Gym/README.md",
  "language_code": "hu"
}
-->
# CartPole Gördeszkázás

Az előző leckében megoldott probléma talán játékszerűnek tűnhet, és nem igazán alkalmazhatónak a valós életben. Ez azonban nem így van, mivel sok valós probléma is hasonló helyzetet mutat – például sakk vagy Go játék közben. Ezek hasonlóak, mert van egy táblánk adott szabályokkal és egy **diszkrét állapot**.

## [Előadás előtti kvíz](https://ff-quizzes.netlify.app/en/ml/)

## Bevezetés

Ebben a leckében ugyanazokat az elveket alkalmazzuk a Q-Learning során egy **folytonos állapotú** problémára, azaz egy olyan állapotra, amelyet egy vagy több valós szám határoz meg. A következő problémával foglalkozunk:

> **Probléma**: Ha Péter el akar menekülni a farkas elől, gyorsabban kell tudnia mozogni. Megnézzük, hogyan tanulhat meg Péter gördeszkázni, különösen egyensúlyozni, a Q-Learning segítségével.

![A nagy szökés!](../../../../8-Reinforcement/2-Gym/images/escape.png)

> Péter és barátai kreatívak lesznek, hogy elmeneküljenek a farkas elől! Kép: [Jen Looper](https://twitter.com/jenlooper)

Egy egyszerűsített egyensúlyozási problémát fogunk használni, amelyet **CartPole** problémának neveznek. A CartPole világában van egy vízszintes csúszka, amely balra vagy jobbra mozoghat, és a cél az, hogy egy függőleges rudat egyensúlyban tartsunk a csúszka tetején.

## Előfeltételek

Ebben a leckében az **OpenAI Gym** nevű könyvtárat fogjuk használni különböző **környezetek** szimulálására. A lecke kódját futtathatod helyben (például Visual Studio Code-ban), ebben az esetben a szimuláció egy új ablakban nyílik meg. Ha online futtatod a kódot, néhány módosítást kell végezned, ahogy azt [itt](https://towardsdatascience.com/rendering-openai-gym-envs-on-binder-and-google-colab-536f99391cc7) leírták.

## OpenAI Gym

Az előző leckében a játék szabályait és az állapotot a saját magunk által definiált `Board` osztály határozta meg. Itt egy speciális **szimulációs környezetet** fogunk használni, amely szimulálja az egyensúlyozó rúd mögötti fizikát. Az egyik legnépszerűbb szimulációs környezet a megerősítéses tanulási algoritmusokhoz a [Gym](https://gym.openai.com/), amelyet az [OpenAI](https://openai.com/) tart fenn. Ezzel a Gymmel különböző **környezeteket** hozhatunk létre, a CartPole szimulációtól kezdve az Atari játékokig.

> **Megjegyzés**: Az OpenAI Gym által elérhető egyéb környezeteket [itt](https://gym.openai.com/envs/#classic_control) találod.

Először telepítsük a Gymet és importáljuk a szükséges könyvtárakat (kódblokk 1):

```python
import sys
!{sys.executable} -m pip install gym 

import gym
import matplotlib.pyplot as plt
import numpy as np
import random
```

## Gyakorlat - CartPole környezet inicializálása

Ahhoz, hogy a CartPole egyensúlyozási problémával dolgozzunk, inicializálnunk kell a megfelelő környezetet. Minden környezethez tartozik:

- **Megfigyelési tér**, amely meghatározza az információ szerkezetét, amelyet a környezettől kapunk. A CartPole problémában a rúd helyzetét, sebességét és néhány más értéket kapunk.

- **Akciótér**, amely meghatározza a lehetséges akciókat. Esetünkben az akciótér diszkrét, és két akcióból áll - **balra** és **jobbra**. (kódblokk 2)

1. Az inicializáláshoz írd be a következő kódot:

    ```python
    env = gym.make("CartPole-v1")
    print(env.action_space)
    print(env.observation_space)
    print(env.action_space.sample())
    ```

Ahhoz, hogy lássuk, hogyan működik a környezet, futtassunk egy rövid szimulációt 100 lépésig. Minden lépésnél megadunk egy akciót, amelyet végre kell hajtani – ebben a szimulációban véletlenszerűen választunk egy akciót az `action_space`-ből.

1. Futtasd az alábbi kódot, és nézd meg, mi történik.

    ✅ Ne feledd, hogy ezt a kódot helyi Python telepítésen futtatni előnyösebb! (kódblokk 3)

    ```python
    env.reset()
    
    for i in range(100):
       env.render()
       env.step(env.action_space.sample())
    env.close()
    ```

    Valami hasonlót kell látnod, mint ez a kép:

    ![nem egyensúlyozó CartPole](../../../../8-Reinforcement/2-Gym/images/cartpole-nobalance.gif)

1. A szimuláció során megfigyeléseket kell kapnunk, hogy eldönthessük, hogyan cselekedjünk. Valójában a step függvény visszaadja az aktuális megfigyeléseket, egy jutalomfüggvényt és egy done jelzőt, amely jelzi, hogy van-e értelme folytatni a szimulációt: (kódblokk 4)

    ```python
    env.reset()
    
    done = False
    while not done:
       env.render()
       obs, rew, done, info = env.step(env.action_space.sample())
       print(f"{obs} -> {rew}")
    env.close()
    ```

    A notebook kimenetében valami ilyesmit fogsz látni:

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

    A szimuláció minden lépésénél visszaadott megfigyelési vektor a következő értékeket tartalmazza:
    - A kocsi helyzete
    - A kocsi sebessége
    - A rúd szöge
    - A rúd forgási sebessége

1. Szerezd meg ezeknek a számoknak a minimum és maximum értékét: (kódblokk 5)

    ```python
    print(env.observation_space.low)
    print(env.observation_space.high)
    ```

    Azt is észreveheted, hogy a jutalomérték minden szimulációs lépésnél mindig 1. Ennek oka, hogy célunk minél tovább életben maradni, azaz a rudat ésszerűen függőleges helyzetben tartani a lehető leghosszabb ideig.

    ✅ Valójában a CartPole szimulációt akkor tekintjük megoldottnak, ha sikerül 195 átlagos jutalmat elérni 100 egymást követő próbálkozás során.

## Állapot diszkretizálása

A Q-Learning során létre kell hoznunk egy Q-táblát, amely meghatározza, mit kell tenni minden állapotban. Ehhez az állapotnak **diszkrétnek** kell lennie, pontosabban véges számú diszkrét értéket kell tartalmaznia. Ezért valahogy **diszkretizálnunk** kell a megfigyeléseinket, és azokat egy véges állapothalmazhoz kell hozzárendelni.

Néhány módon megtehetjük ezt:

- **Felosztás bin-ekre**. Ha ismerjük egy adott érték intervallumát, feloszthatjuk ezt az intervallumot egy bizonyos számú **binre**, majd az értéket lecserélhetjük arra a bin számra, amelyhez tartozik. Ez a numpy [`digitize`](https://numpy.org/doc/stable/reference/generated/numpy.digitize.html) metódusával végezhető el. Ebben az esetben pontosan ismerni fogjuk az állapot méretét, mivel az a digitalizáláshoz kiválasztott bin-ek számától függ.

✅ Használhatunk lineáris interpolációt, hogy az értékeket egy véges intervallumra (például -20-tól 20-ig) hozzuk, majd a számokat kerekítéssel egész számokká alakíthatjuk. Ez valamivel kevesebb kontrollt ad az állapot méretére, különösen, ha nem ismerjük a bemeneti értékek pontos tartományait. Például esetünkben 4 értékből 2-nek nincs felső/alsó határa, ami végtelen számú állapotot eredményezhet.

Példánkban a második megközelítést fogjuk alkalmazni. Ahogy később észreveheted, a meghatározatlan felső/alsó határok ellenére ezek az értékek ritkán vesznek fel bizonyos véges intervallumokon kívüli értékeket, így azok az állapotok, amelyek szélsőséges értékeket tartalmaznak, nagyon ritkák lesznek.

1. Íme egy függvény, amely a modellünk megfigyelését veszi, és egy 4 egész értékű tuple-t állít elő: (kódblokk 6)

    ```python
    def discretize(x):
        return tuple((x/np.array([0.25, 0.25, 0.01, 0.1])).astype(np.int))
    ```

1. Nézzük meg egy másik diszkretizálási módszert bin-ek használatával: (kódblokk 7)

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

1. Most futtassunk egy rövid szimulációt, és figyeljük meg ezeket a diszkrét környezeti értékeket. Nyugodtan próbáld ki a `discretize` és `discretize_bins` függvényeket, és nézd meg, van-e különbség.

    ✅ A `discretize_bins` a bin számát adja vissza, amely 0-alapú. Így a bemeneti változó körüli értékek esetén 0 körül az intervallum közepéből (10) ad vissza számot. A `discretize` esetében nem törődtünk a kimeneti értékek tartományával, lehetővé téve, hogy negatívak legyenek, így az állapotértékek nem tolódnak el, és 0 megfelel 0-nak. (kódblokk 8)

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

    ✅ Kommenteld ki az env.render-rel kezdődő sort, ha látni szeretnéd, hogyan hajtja végre a környezet. Ellenkező esetben a háttérben is futtathatod, ami gyorsabb. Ezt a "láthatatlan" végrehajtást fogjuk használni a Q-Learning folyamat során.

## A Q-tábla szerkezete

Az előző leckében az állapot egy egyszerű számpár volt 0-tól 8-ig, így kényelmes volt a Q-táblát egy 8x8x2 alakú numpy tensorral ábrázolni. Ha bin-ek diszkretizálását használjuk, az állapotvektor mérete is ismert, így ugyanazt a megközelítést alkalmazhatjuk, és az állapotot egy 20x20x10x10x2 alakú tömbbel ábrázolhatjuk (itt 2 az akciótér dimenziója, az első dimenziók pedig az egyes paraméterekhez kiválasztott bin-ek számát jelölik a megfigyelési térben).

Azonban néha a megfigyelési tér pontos dimenziói nem ismertek. A `discretize` függvény esetében soha nem lehetünk biztosak abban, hogy az állapot bizonyos határokon belül marad, mivel néhány eredeti érték nincs korlátozva. Ezért kissé eltérő megközelítést alkalmazunk, és a Q-táblát szótárként ábrázoljuk.

1. Használjuk az *(állapot, akció)* párost a szótár kulcsaként, és az érték a Q-tábla bejegyzésének értékét jelöli. (kódblokk 9)

    ```python
    Q = {}
    actions = (0,1)
    
    def qvalues(state):
        return [Q.get((state,a),0) for a in actions]
    ```

    Itt definiálunk egy `qvalues()` függvényt is, amely visszaadja a Q-tábla értékeinek listáját egy adott állapothoz, amely az összes lehetséges akcióhoz tartozik. Ha a bejegyzés nem szerepel a Q-táblában, alapértelmezés szerint 0-t adunk vissza.

## Kezdjük a Q-Learninget

Most készen állunk arra, hogy megtanítsuk Pétert egyensúlyozni!

1. Először állítsunk be néhány hiperparamétert: (kódblokk 10)

    ```python
    # hyperparameters
    alpha = 0.3
    gamma = 0.9
    epsilon = 0.90
    ```

    Itt az `alpha` a **tanulási ráta**, amely meghatározza, hogy milyen mértékben kell módosítanunk a Q-tábla aktuális értékeit minden lépésnél. Az előző leckében 1-ről indultunk, majd a tanulás során csökkentettük az `alpha` értékét. Ebben a példában egyszerűség kedvéért állandó értéken tartjuk, de később kísérletezhetsz az `alpha` értékek módosításával.

    A `gamma` a **diszkontfaktor**, amely megmutatja, hogy milyen mértékben kell előnyben részesítenünk a jövőbeli jutalmat a jelenlegi jutalommal szemben.

    Az `epsilon` az **exploráció/hasznosítás tényező**, amely meghatározza, hogy az explorációt vagy a hasznosítást kell-e előnyben részesítenünk. Algoritmusunkban az esetek `epsilon` százalékában a következő akciót a Q-tábla értékei alapján választjuk ki, a fennmaradó esetekben pedig véletlenszerű akciót hajtunk végre. Ez lehetővé teszi számunkra, hogy felfedezzük a keresési tér olyan területeit, amelyeket korábban nem láttunk.

    ✅ Az egyensúlyozás szempontjából – véletlenszerű akció választása (exploráció) olyan, mintha véletlenszerű ütést kapnánk rossz irányba, és a rúdnak meg kell tanulnia, hogyan állítsa vissza az egyensúlyt ezekből a "hibákból".

### Az algoritmus fejlesztése

Két fejlesztést is végezhetünk az előző lecke algoritmusán:

- **Átlagos kumulatív jutalom kiszámítása** több szimuláció során. 5000 iterációnként kinyomtatjuk az előrehaladást, és az átlagos kumulatív jutalmat számítjuk ki ezen időszak alatt. Ez azt jelenti, hogy ha több mint 195 pontot érünk el, akkor a problémát megoldottnak tekinthetjük, még a szükségesnél is jobb minőségben.

- **Maximális átlagos kumulatív eredmény kiszámítása**, `Qmax`, és elmentjük a Q-táblát, amely ehhez az eredményhez tartozik. Amikor futtatod a tanulást, észre fogod venni, hogy néha az átlagos kumulatív eredmény csökkenni kezd, és meg akarjuk őrizni a Q-tábla azon értékeit, amelyek a legjobb modellhez tartoznak a tanulás során.

1. Gyűjtsd össze az összes kumulatív jutalmat minden szimulációnál a `rewards` vektorban további ábrázoláshoz. (kódblokk 11)

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

A következőket figyelheted meg az eredményekből:

- **Közel a célhoz**. Nagyon közel vagyunk ahhoz, hogy elérjük a 195 kumulatív jutalmat 100+ egymást követő szimuláció futtatása során, vagy akár el is érhettük! Még ha kisebb számokat kapunk is, nem tudhatjuk biztosan, mert 5000 futtatás átlagát számítjuk, és a hivatalos kritériumhoz csak 100 futtatás szükséges.

- **A jutalom csökkenni kezd**. Néha a jutalom csökkenni kezd, ami azt jelenti, hogy "tönkretehetjük" a Q-táblában már megtanult értékeket olyanokkal, amelyek rosszabbá teszik a helyzetet.

Ez a megfigyelés egyértelműbben látható, ha ábrázoljuk a tanulási folyamatot.

## A tanulási folyamat ábrázolása

A tanulás során az iterációk során összegyűjtöttük a kumulatív jutalom értékét a `rewards` vektorba. Így néz ki, amikor ábrázoljuk az iterációk száma ellenében:

```python
plt.plot(rewards)
```

![nyers előrehaladás](../../../../8-Reinforcement/2-Gym/images/train_progress_raw.png)

Ebből a grafikonból nem lehet semmit megállapítani, mivel a sztochasztikus tanulási folyamat természetéből adódóan a tanulási szakaszok hossza nagyon változó. Hogy értelmesebbé tegyük ezt a grafikont, kiszámíthatjuk a **futó átlagot** egy sor kísérlet során, mondjuk 100. Ezt kényelmesen elvégezhetjük az `np.convolve` segítségével: (kódblokk 12)

```python
def running_average(x,window):
    return np.convolve(x,np.ones(window)/window,mode='valid')

plt.plot(running_average(rewards,100))
```

![tanulási folyamat](../../../../8-Reinforcement/2-Gym/images/train_progress_runav.png)

## Hiperparaméterek változtatása

A tanulás stabilabbá tétele érdekében érdemes
> **Feladat 1**: Játssz a hiperparaméterek értékeivel, és nézd meg, hogy el tudsz-e érni magasabb kumulatív jutalmat. Sikerül 195 fölé jutnod?
> **2. feladat**: Ahhoz, hogy hivatalosan megoldjuk a problémát, 195-ös átlagos jutalmat kell elérni 100 egymást követő futtatás során. Mérd ezt az edzés alatt, és győződj meg róla, hogy hivatalosan megoldottad a problémát!

## Az eredmény megtekintése működés közben

Érdekes lenne látni, hogyan viselkedik a betanított modell. Futtassuk le a szimulációt, és kövessük ugyanazt az akcióválasztási stratégiát, mint az edzés során, a Q-táblában lévő valószínűségi eloszlás alapján mintázva: (kód blokk 13)

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

Valami ilyesmit kellene látnod:

![egy egyensúlyozó cartpole](../../../../8-Reinforcement/2-Gym/images/cartpole-balance.gif)

---

## 🚀Kihívás

> **3. feladat**: Itt a Q-tábla végső verzióját használtuk, ami lehet, hogy nem a legjobb. Ne feledd, hogy a legjobban teljesítő Q-táblát elmentettük a `Qbest` változóba! Próbáld ki ugyanazt a példát a legjobban teljesítő Q-táblával, úgy, hogy átmásolod a `Qbest`-et a `Q`-ba, és nézd meg, észreveszel-e különbséget.

> **4. feladat**: Itt nem a legjobb akciót választottuk minden lépésnél, hanem a megfelelő valószínűségi eloszlás alapján mintáztunk. Ésszerűbb lenne mindig a legjobb akciót választani, amelynek a legmagasabb Q-tábla értéke van? Ezt megteheted az `np.argmax` függvény használatával, amely megadja a legmagasabb Q-tábla értékhez tartozó akció számát. Valósítsd meg ezt a stratégiát, és nézd meg, javítja-e az egyensúlyozást.

## [Utó-előadás kvíz](https://ff-quizzes.netlify.app/en/ml/)

## Feladat
[Edz egy Mountain Car modellt](assignment.md)

## Összegzés

Mostanra megtanultuk, hogyan lehet ügynököket betanítani arra, hogy jó eredményeket érjenek el pusztán azáltal, hogy egy jutalomfüggvényt biztosítunk számukra, amely meghatározza a játék kívánt állapotát, és lehetőséget adunk nekik arra, hogy intelligensen feltérképezzék a keresési teret. Sikeresen alkalmaztuk a Q-Learning algoritmust diszkrét és folytonos környezetek esetében, de diszkrét akciókkal.

Fontos tanulmányozni azokat a helyzeteket is, ahol az akcióállapot szintén folytonos, és amikor a megfigyelési tér sokkal összetettebb, például az Atari játék képernyőjének képe. Ezekben a problémákban gyakran erősebb gépi tanulási technikákra, például neurális hálókra van szükség ahhoz, hogy jó eredményeket érjünk el. Ezek a fejlettebb témák a következő, haladó AI kurzusunk tárgyát képezik.

---

**Felelősség kizárása**:  
Ez a dokumentum az AI fordítási szolgáltatás [Co-op Translator](https://github.com/Azure/co-op-translator) segítségével lett lefordítva. Bár törekszünk a pontosságra, kérjük, vegye figyelembe, hogy az automatikus fordítások hibákat vagy pontatlanságokat tartalmazhatnak. Az eredeti dokumentum az eredeti nyelvén tekintendő hiteles forrásnak. Kritikus információk esetén javasolt professzionális emberi fordítást igénybe venni. Nem vállalunk felelősséget semmilyen félreértésért vagy téves értelmezésért, amely a fordítás használatából eredhet.