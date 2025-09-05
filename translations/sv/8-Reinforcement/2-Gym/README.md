<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "107d5bb29da8a562e7ae72262d251a75",
  "translation_date": "2025-09-05T22:08:49+00:00",
  "source_file": "8-Reinforcement/2-Gym/README.md",
  "language_code": "sv"
}
-->
## Förkunskaper

I denna lektion kommer vi att använda ett bibliotek som heter **OpenAI Gym** för att simulera olika **miljöer**. Du kan köra kod från denna lektion lokalt (t.ex. från Visual Studio Code), i vilket fall simuleringen öppnas i ett nytt fönster. Om du kör koden online kan du behöva göra vissa justeringar, som beskrivs [här](https://towardsdatascience.com/rendering-openai-gym-envs-on-binder-and-google-colab-536f99391cc7).

## OpenAI Gym

I den föregående lektionen definierades spelets regler och tillstånd av klassen `Board` som vi skapade själva. Här kommer vi att använda en speciell **simuleringsmiljö** som simulerar fysiken bakom den balanserande stången. En av de mest populära simuleringsmiljöerna för att träna förstärkningsinlärningsalgoritmer kallas [Gym](https://gym.openai.com/), som underhålls av [OpenAI](https://openai.com/). Med hjälp av Gym kan vi skapa olika **miljöer**, från cartpole-simuleringar till Atari-spel.

> **Note**: Du kan se andra miljöer som finns tillgängliga från OpenAI Gym [här](https://gym.openai.com/envs/#classic_control).

Först installerar vi Gym och importerar nödvändiga bibliotek (kodblock 1):

```python
import sys
!{sys.executable} -m pip install gym 

import gym
import matplotlib.pyplot as plt
import numpy as np
import random
```

## Övning - initiera en cartpole-miljö

För att arbeta med problemet att balansera en cartpole behöver vi initiera motsvarande miljö. Varje miljö är associerad med:

- **Observationsutrymme** som definierar strukturen för den information vi får från miljön. För cartpole-problemet får vi positionen av stången, hastighet och några andra värden.

- **Handlingsutrymme** som definierar möjliga handlingar. I vårt fall är handlingsutrymmet diskret och består av två handlingar - **vänster** och **höger**. (kodblock 2)

1. För att initiera, skriv följande kod:

    ```python
    env = gym.make("CartPole-v1")
    print(env.action_space)
    print(env.observation_space)
    print(env.action_space.sample())
    ```

För att se hur miljön fungerar, låt oss köra en kort simulering i 100 steg. Vid varje steg tillhandahåller vi en av handlingarna som ska utföras - i denna simulering väljer vi bara slumpmässigt en handling från `action_space`.

1. Kör koden nedan och se vad det leder till.

    ✅ Kom ihåg att det är att föredra att köra denna kod på en lokal Python-installation! (kodblock 3)

    ```python
    env.reset()
    
    for i in range(100):
       env.render()
       env.step(env.action_space.sample())
    env.close()
    ```

    Du bör se något liknande denna bild:

    ![icke-balanserande cartpole](../../../../8-Reinforcement/2-Gym/images/cartpole-nobalance.gif)

1. Under simuleringen behöver vi få observationer för att kunna bestämma hur vi ska agera. Faktum är att funktionen `step` returnerar aktuella observationer, en belöningsfunktion och flaggan `done` som indikerar om det är meningsfullt att fortsätta simuleringen eller inte: (kodblock 4)

    ```python
    env.reset()
    
    done = False
    while not done:
       env.render()
       obs, rew, done, info = env.step(env.action_space.sample())
       print(f"{obs} -> {rew}")
    env.close()
    ```

    Du kommer att se något liknande detta i notebookens output:

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

    Observationsvektorn som returneras vid varje steg av simuleringen innehåller följande värden:
    - Vagnens position
    - Vagnens hastighet
    - Stångens vinkel
    - Stångens rotationshastighet

1. Hämta min- och maxvärden för dessa nummer: (kodblock 5)

    ```python
    print(env.observation_space.low)
    print(env.observation_space.high)
    ```

    Du kanske också märker att belöningsvärdet vid varje simuleringssteg alltid är 1. Detta beror på att vårt mål är att överleva så länge som möjligt, dvs. hålla stången i en rimligt vertikal position under längst möjliga tid.

    ✅ Faktum är att CartPole-simuleringen anses vara löst om vi lyckas få ett genomsnittligt belöningsvärde på 195 över 100 på varandra följande försök.

## Diskretisering av tillstånd

I Q-Learning behöver vi bygga en Q-Tabell som definierar vad vi ska göra vid varje tillstånd. För att kunna göra detta måste tillståndet vara **diskret**, mer specifikt, det bör innehålla ett ändligt antal diskreta värden. Därför måste vi på något sätt **diskretisera** våra observationer och mappa dem till en ändlig uppsättning tillstånd.

Det finns några sätt vi kan göra detta på:

- **Dela upp i intervall**. Om vi känner till intervallet för ett visst värde kan vi dela detta intervall i ett antal **intervall**, och sedan ersätta värdet med det intervallnummer det tillhör. Detta kan göras med hjälp av numpy-metoden [`digitize`](https://numpy.org/doc/stable/reference/generated/numpy.digitize.html). I detta fall kommer vi att veta exakt storleken på tillståndet, eftersom det beror på antalet intervall vi väljer för digitalisering.

✅ Vi kan använda linjär interpolation för att föra värden till ett ändligt intervall (säg, från -20 till 20), och sedan konvertera siffror till heltal genom avrundning. Detta ger oss lite mindre kontroll över storleken på tillståndet, särskilt om vi inte känner till de exakta intervallen för ingångsvärdena. Till exempel, i vårt fall har 2 av 4 värden inga övre/nedre gränser för sina värden, vilket kan resultera i ett oändligt antal tillstånd.

I vårt exempel kommer vi att använda det andra tillvägagångssättet. Som du kanske märker senare, trots obestämda övre/nedre gränser, tar dessa värden sällan värden utanför vissa ändliga intervall, vilket gör att tillstånd med extrema värden blir mycket sällsynta.

1. Här är funktionen som tar observationen från vår modell och producerar en tuple med 4 heltalsvärden: (kodblock 6)

    ```python
    def discretize(x):
        return tuple((x/np.array([0.25, 0.25, 0.01, 0.1])).astype(np.int))
    ```

1. Låt oss också utforska en annan diskretiseringsmetod med hjälp av intervall: (kodblock 7)

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

1. Låt oss nu köra en kort simulering och observera dessa diskreta miljövärden. Testa gärna både `discretize` och `discretize_bins` och se om det finns någon skillnad.

    ✅ `discretize_bins` returnerar intervallnumret, som är 0-baserat. För värden på ingångsvariabeln runt 0 returnerar det numret från mitten av intervallet (10). I `discretize` brydde vi oss inte om intervallet för utgångsvärdena, vilket tillåter dem att vara negativa, så tillståndsvärdena är inte förskjutna och 0 motsvarar 0. (kodblock 8)

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

    ✅ Avkommentera raden som börjar med `env.render` om du vill se hur miljön exekveras. Annars kan du köra den i bakgrunden, vilket är snabbare. Vi kommer att använda denna "osynliga" exekvering under vår Q-Learning-process.

## Q-Tabellens struktur

I vår föregående lektion var tillståndet ett enkelt par av siffror från 0 till 8, och det var därför bekvämt att representera Q-Tabellen med en numpy-tensor med formen 8x8x2. Om vi använder intervall-diskretisering är storleken på vår tillståndsvektor också känd, så vi kan använda samma tillvägagångssätt och representera tillståndet med en array med formen 20x20x10x10x2 (här är 2 dimensionen för handlingsutrymmet, och de första dimensionerna motsvarar antalet intervall vi har valt att använda för varje parameter i observationsutrymmet).

Men ibland är de exakta dimensionerna för observationsutrymmet inte kända. I fallet med funktionen `discretize` kan vi aldrig vara säkra på att vårt tillstånd håller sig inom vissa gränser, eftersom vissa av de ursprungliga värdena inte är begränsade. Därför kommer vi att använda ett något annorlunda tillvägagångssätt och representera Q-Tabellen med en ordbok.

1. Använd paret *(state,action)* som nyckel i ordboken, och värdet skulle motsvara värdet i Q-Tabellen. (kodblock 9)

    ```python
    Q = {}
    actions = (0,1)
    
    def qvalues(state):
        return [Q.get((state,a),0) for a in actions]
    ```

    Här definierar vi också en funktion `qvalues()`, som returnerar en lista med värden från Q-Tabellen för ett givet tillstånd som motsvarar alla möjliga handlingar. Om posten inte finns i Q-Tabellen kommer vi att returnera 0 som standard.

## Låt oss börja med Q-Learning

Nu är vi redo att lära Peter att balansera!

1. Först, låt oss ställa in några hyperparametrar: (kodblock 10)

    ```python
    # hyperparameters
    alpha = 0.3
    gamma = 0.9
    epsilon = 0.90
    ```

    Här är `alpha` **inlärningshastigheten** som definierar i vilken utsträckning vi ska justera de aktuella värdena i Q-Tabellen vid varje steg. I den föregående lektionen började vi med 1 och minskade sedan `alpha` till lägre värden under träningen. I detta exempel kommer vi att hålla den konstant för enkelhetens skull, och du kan experimentera med att justera `alpha`-värden senare.

    `gamma` är **diskonteringsfaktorn** som visar i vilken utsträckning vi ska prioritera framtida belöningar över nuvarande belöningar.

    `epsilon` är **utforsknings-/utnyttjandefaktorn** som avgör om vi ska föredra utforskning framför utnyttjande eller vice versa. I vår algoritm kommer vi i `epsilon` procent av fallen att välja nästa handling enligt Q-Tabellens värden, och i resterande antal fall kommer vi att utföra en slumpmässig handling. Detta gör att vi kan utforska områden i sökutrymmet som vi aldrig har sett tidigare.

    ✅ När det gäller balansering - att välja slumpmässig handling (utforskning) skulle fungera som ett slumpmässigt slag i fel riktning, och stången skulle behöva lära sig att återfå balansen från dessa "misstag".

### Förbättra algoritmen

Vi kan också göra två förbättringar av vår algoritm från den föregående lektionen:

- **Beräkna genomsnittlig kumulativ belöning** över ett antal simuleringar. Vi kommer att skriva ut framstegen var 5000:e iteration och vi kommer att ta genomsnittet av vår kumulativa belöning under den tidsperioden. Det betyder att om vi får mer än 195 poäng kan vi anse problemet löst, med ännu högre kvalitet än vad som krävs.

- **Beräkna maximal genomsnittlig kumulativ belöning**, `Qmax`, och vi kommer att lagra Q-Tabellen som motsvarar det resultatet. När du kör träningen kommer du att märka att ibland börjar det genomsnittliga kumulativa resultatet sjunka, och vi vill behålla värdena i Q-Tabellen som motsvarar den bästa modellen som observerats under träningen.

1. Samla alla kumulativa belöningar vid varje simulering i vektorn `rewards` för vidare plottning. (kodblock 11)

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

Vad du kan märka från dessa resultat:

- **Nära vårt mål**. Vi är mycket nära att uppnå målet att få 195 kumulativa belöningar över 100+ på varandra följande körningar av simuleringen, eller så har vi faktiskt uppnått det! Även om vi får mindre siffror vet vi fortfarande inte, eftersom vi tar genomsnittet över 5000 körningar, och endast 100 körningar krävs enligt de formella kriterierna.

- **Belöningen börjar sjunka**. Ibland börjar belöningen sjunka, vilket betyder att vi kan "förstöra" redan inlärda värden i Q-Tabellen med de som gör situationen sämre.

Denna observation blir tydligare om vi plottar träningsframstegen.

## Plotta träningsframsteg

Under träningen har vi samlat det kumulativa belöningsvärdet vid varje iteration i vektorn `rewards`. Så här ser det ut när vi plottar det mot iterationsnumret:

```python
plt.plot(rewards)
```

![råa framsteg](../../../../8-Reinforcement/2-Gym/images/train_progress_raw.png)

Från denna graf är det inte möjligt att säga något, eftersom längden på träningssessionerna varierar kraftigt på grund av den stokastiska träningsprocessens natur. För att göra grafen mer meningsfull kan vi beräkna **rullande medelvärde** över en serie experiment, låt säga 100. Detta kan göras bekvämt med `np.convolve`: (kodblock 12)

```python
def running_average(x,window):
    return np.convolve(x,np.ones(window)/window,mode='valid')

plt.plot(running_average(rewards,100))
```

![träningsframsteg](../../../../8-Reinforcement/2-Gym/images/train_progress_runav.png)

## Variera hyperparametrar

För att göra inlärningen mer stabil kan det vara vettigt att justera några av våra hyperparametrar under träningen. I synnerhet:

- **För inlärningshastigheten**, `alpha`, kan vi börja med värden nära 1 och sedan fortsätta att minska parametern. Med tiden kommer vi att få bra sannolikhetsvärden i Q-Tabellen, och därför bör vi justera dem försiktigt och inte helt skriva över med nya värden.

- **Öka epsilon**. Vi kanske vill öka `epsilon` långsamt för att utforska mindre och utnyttja mer. Det kan vara vettigt att börja med ett lägre värde för `epsilon` och sedan öka det till nästan 1.
> **Uppgift 1**: Testa att ändra hyperparametervärden och se om du kan uppnå högre kumulativ belöning. Kommer du över 195?
> **Uppgift 2**: För att formellt lösa problemet behöver du uppnå ett genomsnittligt belöningsvärde på 195 över 100 på varandra följande körningar. Mät detta under träningen och säkerställ att du formellt har löst problemet!

## Se resultatet i praktiken

Det skulle vara intressant att faktiskt se hur den tränade modellen beter sig. Låt oss köra simuleringen och följa samma strategi för val av handlingar som under träningen, där vi samplar enligt sannolikhetsfördelningen i Q-Tabellen: (kodblock 13)

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

Du bör se något liknande detta:

![en balanserande cartpole](../../../../8-Reinforcement/2-Gym/images/cartpole-balance.gif)

---

## 🚀Utmaning

> **Uppgift 3**: Här använde vi den slutgiltiga versionen av Q-Tabellen, som kanske inte är den bästa. Kom ihåg att vi har sparat den bäst presterande Q-Tabellen i variabeln `Qbest`! Testa samma exempel med den bäst presterande Q-Tabellen genom att kopiera `Qbest` till `Q` och se om du märker någon skillnad.

> **Uppgift 4**: Här valde vi inte den bästa handlingen vid varje steg, utan samplade istället enligt motsvarande sannolikhetsfördelning. Skulle det vara mer logiskt att alltid välja den bästa handlingen, med det högsta värdet i Q-Tabellen? Detta kan göras genom att använda funktionen `np.argmax` för att hitta handlingsnumret som motsvarar det högsta värdet i Q-Tabellen. Implementera denna strategi och se om det förbättrar balansen.

## [Quiz efter föreläsningen](https://ff-quizzes.netlify.app/en/ml/)

## Uppgift
[Träna en Mountain Car](assignment.md)

## Slutsats

Vi har nu lärt oss hur man tränar agenter för att uppnå bra resultat genom att bara tillhandahålla en belöningsfunktion som definierar önskat tillstånd i spelet, och genom att ge dem möjlighet att intelligent utforska sökutrymmet. Vi har framgångsrikt tillämpat Q-Learning-algoritmen i fall med diskreta och kontinuerliga miljöer, men med diskreta handlingar.

Det är också viktigt att studera situationer där handlingsutrymmet också är kontinuerligt, och när observationsutrymmet är mycket mer komplext, som en bild från skärmen i ett Atari-spel. I dessa problem behöver vi ofta använda mer kraftfulla maskininlärningstekniker, såsom neurala nätverk, för att uppnå bra resultat. Dessa mer avancerade ämnen är föremål för vår kommande mer avancerade AI-kurs.

---

**Ansvarsfriskrivning**:  
Detta dokument har översatts med hjälp av AI-översättningstjänsten [Co-op Translator](https://github.com/Azure/co-op-translator). Även om vi strävar efter noggrannhet, bör det noteras att automatiserade översättningar kan innehålla fel eller brister. Det ursprungliga dokumentet på dess originalspråk bör betraktas som den auktoritativa källan. För kritisk information rekommenderas professionell human översättning. Vi ansvarar inte för eventuella missförstånd eller feltolkningar som uppstår vid användning av denna översättning.