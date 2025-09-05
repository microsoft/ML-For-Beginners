<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "107d5bb29da8a562e7ae72262d251a75",
  "translation_date": "2025-09-05T20:19:02+00:00",
  "source_file": "8-Reinforcement/2-Gym/README.md",
  "language_code": "nl"
}
-->
# CartPole Schaatsen

Het probleem dat we in de vorige les hebben opgelost, lijkt misschien een speelgoedprobleem, niet echt toepasbaar in echte scenario's. Dit is echter niet het geval, omdat veel echte problemen ook dit scenario delen - zoals het spelen van schaken of Go. Ze zijn vergelijkbaar omdat we ook een bord hebben met gegeven regels en een **discrete toestand**.

## [Quiz voorafgaand aan de les](https://ff-quizzes.netlify.app/en/ml/)

## Introductie

In deze les passen we dezelfde principes van Q-Learning toe op een probleem met een **continue toestand**, d.w.z. een toestand die wordt weergegeven door een of meer reële getallen. We gaan het volgende probleem aanpakken:

> **Probleem**: Als Peter wil ontsnappen aan de wolf, moet hij sneller kunnen bewegen. We zullen zien hoe Peter kan leren schaatsen, en in het bijzonder hoe hij balans kan houden, met behulp van Q-Learning.

![De grote ontsnapping!](../../../../8-Reinforcement/2-Gym/images/escape.png)

> Peter en zijn vrienden worden creatief om aan de wolf te ontsnappen! Afbeelding door [Jen Looper](https://twitter.com/jenlooper)

We gebruiken een vereenvoudigde versie van balanceren, bekend als het **CartPole**-probleem. In de CartPole-wereld hebben we een horizontale slider die naar links of rechts kan bewegen, en het doel is om een verticale paal bovenop de slider in balans te houden.

## Vereisten

In deze les gebruiken we een bibliotheek genaamd **OpenAI Gym** om verschillende **omgevingen** te simuleren. Je kunt de code van deze les lokaal uitvoeren (bijvoorbeeld vanuit Visual Studio Code), in welk geval de simulatie in een nieuw venster wordt geopend. Bij het online uitvoeren van de code moet je mogelijk enkele aanpassingen maken, zoals beschreven [hier](https://towardsdatascience.com/rendering-openai-gym-envs-on-binder-and-google-colab-536f99391cc7).

## OpenAI Gym

In de vorige les werden de spelregels en de toestand gegeven door de `Board`-klasse die we zelf hebben gedefinieerd. Hier gebruiken we een speciale **simulatieomgeving**, die de fysica achter de balancerende paal simuleert. Een van de meest populaire simulatieomgevingen voor het trainen van reinforcement learning-algoritmen heet een [Gym](https://gym.openai.com/), die wordt onderhouden door [OpenAI](https://openai.com/). Met deze gym kunnen we verschillende **omgevingen** creëren, van een CartPole-simulatie tot Atari-spellen.

> **Let op**: Je kunt andere beschikbare omgevingen van OpenAI Gym bekijken [hier](https://gym.openai.com/envs/#classic_control).

Laten we eerst de gym installeren en de benodigde bibliotheken importeren (codeblok 1):

```python
import sys
!{sys.executable} -m pip install gym 

import gym
import matplotlib.pyplot as plt
import numpy as np
import random
```

## Oefening - een CartPole-omgeving initialiseren

Om te werken met een CartPole-balanceringsprobleem, moeten we de bijbehorende omgeving initialiseren. Elke omgeving is gekoppeld aan een:

- **Observatieruimte** die de structuur definieert van de informatie die we van de omgeving ontvangen. Voor het CartPole-probleem ontvangen we de positie van de paal, snelheid en enkele andere waarden.

- **Actieruimte** die mogelijke acties definieert. In ons geval is de actieruimte discreet en bestaat uit twee acties - **links** en **rechts**. (codeblok 2)

1. Om te initialiseren, typ de volgende code:

    ```python
    env = gym.make("CartPole-v1")
    print(env.action_space)
    print(env.observation_space)
    print(env.action_space.sample())
    ```

Om te zien hoe de omgeving werkt, laten we een korte simulatie uitvoeren van 100 stappen. Bij elke stap geven we een van de acties op die moet worden uitgevoerd - in deze simulatie selecteren we willekeurig een actie uit `action_space`.

1. Voer de onderstaande code uit en kijk wat het oplevert.

    ✅ Onthoud dat het aanbevolen is om deze code uit te voeren op een lokale Python-installatie! (codeblok 3)

    ```python
    env.reset()
    
    for i in range(100):
       env.render()
       env.step(env.action_space.sample())
    env.close()
    ```

    Je zou iets moeten zien dat lijkt op deze afbeelding:

    ![niet-balancerende CartPole](../../../../8-Reinforcement/2-Gym/images/cartpole-nobalance.gif)

1. Tijdens de simulatie moeten we observaties verkrijgen om te beslissen hoe te handelen. De stapfunctie retourneert namelijk de huidige observaties, een beloningsfunctie en de vlag `done` die aangeeft of het zinvol is om de simulatie voort te zetten of niet: (codeblok 4)

    ```python
    env.reset()
    
    done = False
    while not done:
       env.render()
       obs, rew, done, info = env.step(env.action_space.sample())
       print(f"{obs} -> {rew}")
    env.close()
    ```

    Je zult iets zien zoals dit in de notebook-uitvoer:

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

    De observatievector die bij elke stap van de simulatie wordt geretourneerd, bevat de volgende waarden:
    - Positie van de kar
    - Snelheid van de kar
    - Hoek van de paal
    - Rotatiesnelheid van de paal

1. Verkrijg de minimale en maximale waarde van deze getallen: (codeblok 5)

    ```python
    print(env.observation_space.low)
    print(env.observation_space.high)
    ```

    Je zult ook merken dat de beloningswaarde bij elke simulatiestap altijd 1 is. Dit komt omdat ons doel is om zo lang mogelijk te overleven, d.w.z. de paal zo lang mogelijk in een redelijk verticale positie te houden.

    ✅ In feite wordt de CartPole-simulatie als opgelost beschouwd als we erin slagen een gemiddelde beloning van 195 te behalen over 100 opeenvolgende pogingen.

## Toestand discretiseren

Bij Q-Learning moeten we een Q-Tabel bouwen die definieert wat te doen in elke toestand. Om dit te kunnen doen, moet de toestand **discreet** zijn, meer precies, het moet een eindig aantal discrete waarden bevatten. Daarom moeten we onze observaties op de een of andere manier **discretiseren**, en ze koppelen aan een eindige set toestanden.

Er zijn een paar manieren waarop we dit kunnen doen:

- **Verdelen in intervallen**. Als we het interval van een bepaalde waarde kennen, kunnen we dit interval verdelen in een aantal **intervallen**, en vervolgens de waarde vervangen door het intervalnummer waartoe het behoort. Dit kan worden gedaan met behulp van de numpy-methode [`digitize`](https://numpy.org/doc/stable/reference/generated/numpy.digitize.html). In dit geval kennen we de grootte van de toestand precies, omdat deze afhankelijk is van het aantal intervallen dat we selecteren voor digitalisering.

✅ We kunnen lineaire interpolatie gebruiken om waarden naar een eindig interval te brengen (bijvoorbeeld van -20 tot 20), en vervolgens getallen naar gehele getallen converteren door ze af te ronden. Dit geeft ons iets minder controle over de grootte van de toestand, vooral als we de exacte bereiken van invoerwaarden niet kennen. In ons geval hebben bijvoorbeeld 2 van de 4 waarden geen boven-/ondergrenzen, wat kan resulteren in een oneindig aantal toestanden.

In ons voorbeeld gaan we met de tweede aanpak. Zoals je later zult merken, nemen deze waarden ondanks de ongedefinieerde boven-/ondergrenzen zelden extreme waarden buiten bepaalde eindige intervallen aan, waardoor toestanden met extreme waarden zeer zeldzaam zullen zijn.

1. Hier is de functie die de observatie van ons model neemt en een tuple van 4 gehele waarden produceert: (codeblok 6)

    ```python
    def discretize(x):
        return tuple((x/np.array([0.25, 0.25, 0.01, 0.1])).astype(np.int))
    ```

1. Laten we ook een andere discretisatiemethode verkennen met behulp van intervallen: (codeblok 7)

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

1. Laten we nu een korte simulatie uitvoeren en die discrete omgevingswaarden observeren. Voel je vrij om zowel `discretize` als `discretize_bins` te proberen en te kijken of er een verschil is.

    ✅ `discretize_bins` retourneert het intervalnummer, dat 0-gebaseerd is. Voor waarden van de invoervariabele rond 0 retourneert het het nummer uit het midden van het interval (10). Bij `discretize` hebben we ons niet bekommerd om het bereik van uitvoerwaarden, waardoor ze negatief konden zijn, en 0 komt overeen met 0. (codeblok 8)

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

    ✅ Haal de regel die begint met `env.render` uit commentaar als je wilt zien hoe de omgeving wordt uitgevoerd. Anders kun je het op de achtergrond uitvoeren, wat sneller is. We zullen deze "onzichtbare" uitvoering gebruiken tijdens ons Q-Learning-proces.

## De structuur van de Q-Tabel

In onze vorige les was de toestand een eenvoudige paar getallen van 0 tot 8, en daarom was het handig om de Q-Tabel te representeren met een numpy-tensor met een vorm van 8x8x2. Als we intervallen-discretisatie gebruiken, is de grootte van onze toestandsvector ook bekend, dus we kunnen dezelfde aanpak gebruiken en de toestand representeren door een array met een vorm van 20x20x10x10x2 (hier is 2 de dimensie van de actieruimte, en de eerste dimensies komen overeen met het aantal intervallen dat we hebben geselecteerd voor elk van de parameters in de observatieruimte).

Soms zijn de exacte dimensies van de observatieruimte echter niet bekend. In het geval van de `discretize`-functie kunnen we nooit zeker weten dat onze toestand binnen bepaalde grenzen blijft, omdat sommige van de oorspronkelijke waarden niet begrensd zijn. Daarom gebruiken we een iets andere aanpak en representeren we de Q-Tabel met een dictionary.

1. Gebruik het paar *(toestand, actie)* als de sleutel van de dictionary, en de waarde zou overeenkomen met de waarde van de Q-Tabel. (codeblok 9)

    ```python
    Q = {}
    actions = (0,1)
    
    def qvalues(state):
        return [Q.get((state,a),0) for a in actions]
    ```

    Hier definiëren we ook een functie `qvalues()`, die een lijst retourneert van Q-Tabelwaarden voor een gegeven toestand die overeenkomt met alle mogelijke acties. Als de invoer niet aanwezig is in de Q-Tabel, retourneren we standaard 0.

## Laten we beginnen met Q-Learning

Nu zijn we klaar om Peter te leren balanceren!

1. Laten we eerst enkele hyperparameters instellen: (codeblok 10)

    ```python
    # hyperparameters
    alpha = 0.3
    gamma = 0.9
    epsilon = 0.90
    ```

    Hier is `alpha` de **leersnelheid** die bepaalt in welke mate we de huidige waarden van de Q-Tabel bij elke stap moeten aanpassen. In de vorige les begonnen we met 1 en verlaagden we vervolgens `alpha` naar lagere waarden tijdens de training. In dit voorbeeld houden we het constant voor de eenvoud, en je kunt later experimenteren met het aanpassen van `alpha`-waarden.

    `gamma` is de **kortingsfactor** die aangeeft in welke mate we toekomstige beloning boven huidige beloning moeten prioriteren.

    `epsilon` is de **exploratie/exploitatie-factor** die bepaalt of we exploratie boven exploitatie moeten verkiezen of andersom. In ons algoritme selecteren we in `epsilon` procent van de gevallen de volgende actie op basis van Q-Tabelwaarden, en in de resterende gevallen voeren we een willekeurige actie uit. Dit stelt ons in staat om gebieden van de zoekruimte te verkennen die we nog nooit eerder hebben gezien.

    ✅ In termen van balanceren - het kiezen van een willekeurige actie (exploratie) zou werken als een willekeurige duw in de verkeerde richting, en de paal zou moeten leren hoe de balans te herstellen van die "fouten".

### Verbeter het algoritme

We kunnen ook twee verbeteringen aanbrengen in ons algoritme van de vorige les:

- **Gemiddelde cumulatieve beloning berekenen**, over een aantal simulaties. We printen de voortgang elke 5000 iteraties en middelen onze cumulatieve beloning over die periode. Dit betekent dat als we meer dan 195 punten behalen, we het probleem als opgelost kunnen beschouwen, met een nog hogere kwaliteit dan vereist.

- **Maximale gemiddelde cumulatieve beloning berekenen**, `Qmax`, en we slaan de Q-Tabel op die overeenkomt met dat resultaat. Wanneer je de training uitvoert, zul je merken dat soms de gemiddelde cumulatieve beloning begint te dalen, en we willen de waarden van de Q-Tabel behouden die overeenkomen met het beste model dat tijdens de training is waargenomen.

1. Verzamel alle cumulatieve beloningen bij elke simulatie in de vector `rewards` voor verdere plotting. (codeblok 11)

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

Wat je kunt opmerken uit die resultaten:

- **Dicht bij ons doel**. We zijn heel dicht bij het bereiken van het doel van 195 cumulatieve beloningen over 100+ opeenvolgende runs van de simulatie, of we hebben het misschien zelfs bereikt! Zelfs als we kleinere aantallen behalen, weten we het nog niet, omdat we middelen over 5000 runs, en slechts 100 runs zijn vereist volgens de formele criteria.

- **Beloning begint te dalen**. Soms begint de beloning te dalen, wat betekent dat we de al geleerde waarden in de Q-Tabel kunnen "vernietigen" met de waarden die de situatie verslechteren.

Deze observatie is duidelijker zichtbaar als we de trainingsvoortgang plotten.

## Trainingsvoortgang plotten

Tijdens de training hebben we de cumulatieve beloningswaarde bij elke iteratie verzameld in de vector `rewards`. Hier is hoe het eruit ziet wanneer we het plotten tegen het iteratienummer:

```python
plt.plot(rewards)
```

![ruwe voortgang](../../../../8-Reinforcement/2-Gym/images/train_progress_raw.png)

Van deze grafiek is het niet mogelijk om iets te zeggen, omdat door de aard van het stochastische trainingsproces de lengte van trainingssessies sterk varieert. Om meer betekenis te geven aan deze grafiek, kunnen we het **lopende gemiddelde** berekenen over een reeks experimenten, laten we zeggen 100. Dit kan handig worden gedaan met `np.convolve`: (codeblok 12)

```python
def running_average(x,window):
    return np.convolve(x,np.ones(window)/window,mode='valid')

plt.plot(running_average(rewards,100))
```

![trainingsvoortgang](../../../../8-Reinforcement/2-Gym/images/train_progress_runav.png)

## Hyperparameters variëren

Om het leren stabieler te maken, is het zinvol om enkele van onze hyperparameters tijdens de training aan te passen. In het bijzonder:

- **Voor leersnelheid**, `alpha`, kunnen we beginnen met waarden dicht bij 1 en vervolgens de parameter blijven verlagen. Na verloop van tijd krijgen we goede waarschijnlijkheidswaarden in de Q-Tabel, en dus moeten we ze licht aanpassen en niet volledig overschrijven met nieuwe waarden.

- **Epsilon verhogen**. We kunnen `epsilon` langzaam verhogen, zodat we minder verkennen en meer exploiteren. Het is waarschijnlijk zinvol om te beginnen met een lagere waarde van `epsilon` en deze op te voeren tot bijna 1.
> **Taak 1**: Speel met de waarden van de hyperparameters en kijk of je een hogere cumulatieve beloning kunt behalen. Kom je boven de 195?
> **Taak 2**: Om het probleem formeel op te lossen, moet je een gemiddelde beloning van 195 behalen over 100 opeenvolgende runs. Meet dit tijdens de training en zorg ervoor dat je het probleem formeel hebt opgelost!

## Het resultaat in actie zien

Het zou interessant zijn om daadwerkelijk te zien hoe het getrainde model zich gedraagt. Laten we de simulatie uitvoeren en dezelfde strategie voor actie-selectie volgen als tijdens de training, waarbij we sampelen volgens de kansverdeling in de Q-Table: (code block 13)

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

Je zou iets moeten zien zoals dit:

![een balancerende cartpole](../../../../8-Reinforcement/2-Gym/images/cartpole-balance.gif)

---

## 🚀Uitdaging

> **Taak 3**: Hier gebruikten we de uiteindelijke versie van de Q-Table, die mogelijk niet de beste is. Onthoud dat we de best presterende Q-Table hebben opgeslagen in de variabele `Qbest`! Probeer hetzelfde voorbeeld met de best presterende Q-Table door `Qbest` over te kopiëren naar `Q` en kijk of je verschil merkt.

> **Taak 4**: Hier selecteerden we niet de beste actie bij elke stap, maar sampelden we met de bijbehorende kansverdeling. Zou het logischer zijn om altijd de beste actie te kiezen, met de hoogste waarde in de Q-Table? Dit kan worden gedaan door de functie `np.argmax` te gebruiken om het actienummer te vinden dat overeenkomt met de hoogste waarde in de Q-Table. Implementeer deze strategie en kijk of het de balans verbetert.

## [Quiz na de les](https://ff-quizzes.netlify.app/en/ml/)

## Opdracht
[Train een Mountain Car](assignment.md)

## Conclusie

We hebben nu geleerd hoe we agenten kunnen trainen om goede resultaten te behalen door hen simpelweg een beloningsfunctie te geven die de gewenste toestand van het spel definieert, en door hen de kans te geven om intelligent de zoekruimte te verkennen. We hebben het Q-Learning-algoritme succesvol toegepast in gevallen van discrete en continue omgevingen, maar met discrete acties.

Het is ook belangrijk om situaties te bestuderen waarin de actiestatus ook continu is, en wanneer de observatieruimte veel complexer is, zoals het beeld van het scherm van een Atari-spel. Bij die problemen moeten we vaak krachtigere machine learning-technieken gebruiken, zoals neurale netwerken, om goede resultaten te behalen. Die meer geavanceerde onderwerpen zijn het onderwerp van onze komende meer gevorderde AI-cursus.

---

**Disclaimer**:  
Dit document is vertaald met behulp van de AI-vertalingsservice [Co-op Translator](https://github.com/Azure/co-op-translator). Hoewel we ons best doen voor nauwkeurigheid, dient u zich ervan bewust te zijn dat geautomatiseerde vertalingen fouten of onnauwkeurigheden kunnen bevatten. Het originele document in zijn oorspronkelijke taal moet worden beschouwd als de gezaghebbende bron. Voor cruciale informatie wordt professionele menselijke vertaling aanbevolen. Wij zijn niet aansprakelijk voor misverstanden of verkeerde interpretaties die voortvloeien uit het gebruik van deze vertaling.