# CartPole Skridskoåkning

Problemet vi har löst i föregående lektion kan verka som ett lekproblem, inte riktigt tillämpligt på verkliga scenarier. Så är dock inte fallet, eftersom många verkliga problem också delar detta scenario – inklusive att spela schack eller Go. De är liknande eftersom vi också har ett bräde med givna regler och ett **diskret tillstånd**.

## [Förföreläsningsquiz](https://ff-quizzes.netlify.app/en/ml/)

## Introduktion

I denna lektion kommer vi att applicera samma principer för Q-Learning på ett problem med **kontinuerligt tillstånd**, dvs ett tillstånd som ges av ett eller flera reella tal. Vi kommer att hantera följande problem:

> **Problem**: Om Peter vill fly från vargen måste han kunna röra sig snabbare. Vi kommer att se hur Peter kan lära sig att åka skridskor, framförallt att hålla balansen, med hjälp av Q-Learning.

![Den stora flykten!](../../../../translated_images/sv/escape.18862db9930337e3.webp)

> Peter och hans vänner blir kreativa för att fly från vargen! Bild av [Jen Looper](https://twitter.com/jenlooper)

Vi kommer att använda en förenklad version av balansering känd som ett **CartPole**-problem. I cartpole-världen har vi en horisontell slider som kan röra sig åt vänster eller höger, och målet är att balansera en vertikal stång ovanpå slidern.

<img alt="a cartpole" src="../../../../translated_images/sv/cartpole.b5609cc0494a14f7.webp" width="200"/>

## Förkunskaper

I denna lektion kommer vi att använda ett bibliotek kallat **OpenAI Gym** för att simulera olika **miljöer**. Du kan köra lektionens kod lokalt (t.ex. från Visual Studio Code), då öppnas simuleringen i ett nytt fönster. När du kör koden online kan du behöva göra vissa justeringar i koden, som beskrivs [här](https://towardsdatascience.com/rendering-openai-gym-envs-on-binder-and-google-colab-536f99391cc7).

## OpenAI Gym

I föregående lektion gavs reglerna för spelet och tillståndet av `Board`-klassen som vi definierade själva. Här kommer vi att använda en speciell **simuleringsmiljö**, som simulerar fysiken bakom balansstången. En av de mest populära simuleringsmiljöerna för träning av förstärkningsinlärningsalgoritmer kallas [Gym](https://gym.openai.com/), som underhålls av [OpenAI](https://openai.com/). Genom att använda detta gym kan vi skapa olika **miljöer** från en cartpole-simulering till Atari-spel.

> **Notis**: Du kan se andra miljöer tillgängliga från OpenAI Gym [här](https://gym.openai.com/envs/#classic_control). 

Först, låt oss installera gym och importera nödvändiga bibliotek (kodblock 1):

```python
import sys
!{sys.executable} -m pip install gym 

import gym
import matplotlib.pyplot as plt
import numpy as np
import random
```

## Övning - initiera en cartpole-miljö

För att arbeta med ett cartpole-balansproblem behöver vi initiera motsvarande miljö. Varje miljö är associerad med en:

- **Observationsutrymme** som definierar strukturen för informationen vi får från miljön. För cartpole-problemet får vi positionen för stången, hastighet och några andra värden.

- **Aktionsutrymme** som definierar möjliga handlingar. I vårt fall är aktionsutrymmet diskret och består av två handlingar – **vänster** och **höger**. (kodblock 2)

1. För att initiera, skriv följande kod:

    ```python
    env = gym.make("CartPole-v1")
    print(env.action_space)
    print(env.observation_space)
    print(env.action_space.sample())
    ```

För att se hur miljön fungerar, kör en kort simulering på 100 steg. Vid varje steg ger vi en av handlingarna som ska utföras – i denna simulering väljer vi helt enkelt en slumpmässig handling från `action_space`. 

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

    ![icke-balans cartpole](../../../../8-Reinforcement/2-Gym/images/cartpole-nobalance.gif)

1. Under simuleringen behöver vi få observationer för att kunna avgöra hur vi ska agera. Faktum är att stegfunktionen returnerar aktuella observationer, en belöningsfunktion och en done-flagga som indikerar om det är meningsfullt att fortsätta simuleringen eller inte: (kodblock 4)

    ```python
    env.reset()
    
    done = False
    while not done:
       env.render()
       obs, rew, done, info = env.step(env.action_space.sample())
       print(f"{obs} -> {rew}")
    env.close()
    ```

    Du kommer att se något liknande detta i notebook-utdata:

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
    - Position för vagnen
    - Hastighet för vagnen
    - Vinkel för stången
    - Rotationshastighet för stången

1. Hämta min- och maxvärde för dessa siffror: (kodblock 5)

    ```python
    print(env.observation_space.low)
    print(env.observation_space.high)
    ```

    Du kommer också att märka att belöningsvärdet vid varje simuleringsteg alltid är 1. Detta beror på att vårt mål är att överleva så länge som möjligt, dvs att hålla stången i en rimligt vertikal position under längst tid.

    ✅ Faktiskt betraktas CartPole-simuleringen som löst om vi lyckas få ett genomsnittligt belöningsvärde på 195 över 100 på varandra följande försök.

## Diskretisering av tillstånd

I Q-Learning behöver vi bygga en Q-tabell som definierar vad vi ska göra i varje tillstånd. För att kunna göra detta behöver tillståndet vara **diskret**, närmare bestämt ska det innehålla ett ändligt antal diskreta värden. Vi behöver alltså på något sätt **diskretisera** våra observationer och mappa dem till ett ändligt antal tillstånd.

Det finns några sätt att göra detta på:

- **Dela in i fack (bins)**. Om vi vet intervallet för ett visst värde kan vi dela in detta intervall i ett antal **bins**, och ersätta värdet med det bin-nummer det tillhör. Detta kan göras med numpy-metoden [`digitize`](https://numpy.org/doc/stable/reference/generated/numpy.digitize.html). I detta fall kommer vi exakt veta storleken på tillståndet, eftersom det beror på antalet bins vi valt för digitaliseringen.
  
✅ Vi kan använda linjär interpolation för att föra värden till ett ändligt intervall (t.ex. från -20 till 20), och sedan konvertera talen till heltal genom att avrunda dem. Detta ger oss mindre kontroll över tillståndets storlek, särskilt om vi inte känner till exakta intervall för indata. Till exempel har 2 av 4 värden i vårt fall inga övre eller nedre gränser, vilket kan leda till ett oändligt antal tillstånd.

I vårt exempel kommer vi att välja det andra tillvägagångssättet. Som du kan märka senare, trots odefinierade övre/nedre gränser, tar dessa värden sällan värden utanför vissa ändliga intervall, så tillstånd med extrema värden kommer att vara mycket sällsynta.

1. Här är funktionen som tar en observation från vår modell och producerar en tupel med 4 heltalsvärden: (kodblock 6)

    ```python
    def discretize(x):
        return tuple((x/np.array([0.25, 0.25, 0.01, 0.1])).astype(np.int))
    ```

1. Låt oss också utforska en annan diskretiseringsmetod med bins: (kodblock 7)

    ```python
    def create_bins(i,num):
        return np.arange(num+1)*(i[1]-i[0])/num+i[0]
    
    print("Sample bins for interval (-5,5) with 10 bins\n",create_bins((-5,5),10))
    
    ints = [(-5,5),(-2,2),(-0.5,0.5),(-2,2)] # intervall av värden för varje parameter
    nbins = [20,20,10,10] # antal fack för varje parameter
    bins = [create_bins(ints[i],nbins[i]) for i in range(4)]
    
    def discretize_bins(x):
        return tuple(np.digitize(x[i],bins[i]) for i in range(4))
    ```

1. Låt oss nu köra en kort simulering och observera dessa diskreta miljövärden. Prova gärna både `discretize` och `discretize_bins` och se om det finns någon skillnad.

    ✅ discretize_bins returnerar bin-numret, som är 0-baserat. Så för värden nära 0 returnerar det ett nummer från mitten av intervallet (10). I discretize brydde vi oss inte om utdata-värdenas intervall, vilket tillåter negativa värden, så tillståndsvärdena är inte förskjutna och 0 motsvarar 0. (kodblock 8)

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

    ✅ Avkommentera raden som börjar med env.render om du vill se hur miljön körs. Annars kan du köra den i bakgrunden, vilket är snabbare. Vi kommer att använda denna "osynliga" körning under vår Q-Learning-process.

## Q-Tabellens struktur

I vår föregående lektion var tillståndet ett enkelt par tal från 0 till 8, och det var därför praktiskt att representera Q-tabellen med en numpy-tensor med formen 8x8x2. Om vi använder bins-diskretisering är storleken på vår tillståndsvektor också känd, så vi kan använda samma metod och representera tillstånd som en array i formen 20x20x10x10x2 (här är 2 dimensionen för aktionsutrymmet, och de första dimensionerna motsvarar antalet bins vi valt att använda för varje parameter i observationsutrymmet).

Men ibland är inte de exakta dimensionerna för observationsutrymmet kända. I fallet med `discretize`-funktionen kan vi aldrig vara säkra på att vårt tillstånd håller sig inom vissa gränser, eftersom några av de ursprungliga värdena är obundna. Därför kommer vi att använda en något annan metod och representera Q-tabellen som en ordlista. 

1. Använd paret *(tillstånd, handling)* som nyckel i ordlistan, och värdet motsvarar värdet i Q-tabellen för denna post. (kodblock 9)

    ```python
    Q = {}
    actions = (0,1)
    
    def qvalues(state):
        return [Q.get((state,a),0) for a in actions]
    ```

    Här definierar vi också en funktion `qvalues()`, som returnerar en lista av Q-tabellvärden för ett givet tillstånd som motsvarar alla möjliga handlingar. Om posten inte finns i Q-tabellen returnerar vi 0 som standard.

## Låt oss börja med Q-Learning

Nu är vi redo att lära Peter att balansera!

1. Först sätter vi några hyperparametrar: (kodblock 10)

    ```python
    # hyperparametrar
    alpha = 0.3
    gamma = 0.9
    epsilon = 0.90
    ```

    Här är `alpha` **inlärningshastigheten** som definierar i vilken utsträckning vi ska justera aktuella Q-tabellvärden vid varje steg. I föregående lektion började vi med 1 och sänkte sedan `alpha` till lägre värden under träningen. I detta exempel håller vi den konstant för enkelhetens skull, och du kan experimentera med att justera `alpha`-värden senare.

    `gamma` är **diskonteringsfaktorn** som visar i vilken grad vi ska prioritera framtida belöning över nuvarande belöning.

    `epsilon` är **utforsknings-/utnyttjandefaktorn** som avgör om vi bör föredra utforskning eller utnyttjande. I vår algoritm kommer vi i `epsilon` procent av fallen att välja nästa handling enligt Q-tabellvärdena, och i återstående fall utför vi en slumpmässig handling. Detta möjliggör utforskning av områden i sökutrymmet vi aldrig tidigare sett. 

    ✅ När det gäller balansen – att välja en slumpmässig handling (utforskning) fungerar som en slumpmässig knuff i fel riktning, och stången måste lära sig hur den ska återhämta balansen från dessa "misstag".

### Förbättra algoritmen

Vi kan också göra två förbättringar jämfört med vår algoritm från föregående lektion:

- **Beräkna genomsnittlig kumulativ belöning** över ett antal simuleringar. Vi skriver ut framsteg var 5000:e iteration och tar genomsnitt av vår kumulativa belöning över denna period. Det betyder att om vi får mer än 195 poäng kan vi betrakta problemet som löst, med högre kvalitet än krävt.
  
- **Beräkna maximalt genomsnittligt kumulativt resultat**, `Qmax`, och spara den Q-tabell som motsvarar detta resultat. Under träningen kommer du märka att det genomsnittliga kumulativa resultatet ibland börjar sjunka, och vi vill behålla Q-tabellens värden som motsvarar den bästa modellen som observerats under träningen.

1. Samla alla kumulativa belöningar från varje simulering i vektorn `rewards` för senare visualisering. (kodblock 11)

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
        # == utför simuleringen ==
        while not done:
            s = discretize(obs)
            if random.random()<epsilon:
                # exploatering - välj åtgärden enligt Q-Tabellens sannolikheter
                v = probs(np.array(qvalues(s)))
                a = random.choices(actions,weights=v)[0]
            else:
                # utforskning - välj åtgärden slumpmässigt
                a = np.random.randint(env.action_space.n)
    
            obs, rew, done, info = env.step(a)
            cum_reward+=rew
            ns = discretize(obs)
            Q[(s,a)] = (1 - alpha) * Q.get((s,a),0) + alpha * (rew + gamma * max(qvalues(ns)))
        cum_rewards.append(cum_reward)
        rewards.append(cum_reward)
        # == Skriv ut resultat periodvis och beräkna genomsnittlig belöning ==
        if epoch%5000==0:
            print(f"{epoch}: {np.average(cum_rewards)}, alpha={alpha}, epsilon={epsilon}")
            if np.average(cum_rewards) > Qmax:
                Qmax = np.average(cum_rewards)
                Qbest = Q
            cum_rewards=[]
    ```

Vad du kan notera från dessa resultat:

- **Nästan vid vårt mål**. Vi är mycket nära att uppnå målet att få 195 kumulativa belöningar över 100+ på varandra följande körningar av simuleringen, eller så har vi faktiskt redan uppnått det! Även om vi får lägre siffror vet vi inte säkert, eftersom vi tar genomsnitt över 5000 körningar och formellt krävs bara 100 körningar.
  
- **Belöningen börjar sjunka**. Ibland börjar belöningen sjunka, vilket betyder att vi kan "förstöra" redan inlärda värden i Q-tabellen med sådana som försämrar situationen.

Denna observation syns tydligare om vi plottar träningsframstegen.

## Plotting av träningsframsteg

Under träningen har vi samlat värdet för kumulativ belöning vid varje iteration i vektorn `rewards`. Så här ser det ut när vi plottar det mot iterationsnumret:

```python
plt.plot(rewards)
```

![råa framsteg](../../../../translated_images/sv/train_progress_raw.2adfdf2daea09c59.webp)

Från denna graf är det svårt att dra slutsatser, eftersom längden på träningssessionerna varierar mycket på grund av den stokastiska träningens natur. För att göra grafen mer meningsfull kan vi beräkna **rullande medelvärde** över en serie experiment, låt säga 100. Detta kan göras smidigt med `np.convolve`: (kodblock 12)

```python
def running_average(x,window):
    return np.convolve(x,np.ones(window)/window,mode='valid')

plt.plot(running_average(rewards,100))
```

![träningsframsteg](../../../../translated_images/sv/train_progress_runav.c71694a8fa9ab359.webp)

## Variera hyperparametrar

För att göra inlärningen mer stabilt är det meningsfullt att justera några hyperparametrar under träningen. Särskilt:

- **För inlärningshastigheten**, `alpha`, kan vi börja med värden nära 1 och sedan sakta minska parametern. Med tiden får vi bra sannolikhetsvärden i Q-tabellen, så vi bör justera dem försiktigt och inte skriva över helt med nya värden.

- **Öka epsilon**. Vi kanske vill successivt öka `epsilon` för att utforska mindre och utnyttja mer. Det är troligen meningsfullt att börja med ett lägre värde på `epsilon` och röra sig uppåt mot nästan 1.

> **Uppgift 1**: Lek med hyperparameter-värdena och se om du kan uppnå högre kumulativ belöning. Får du över 195?


> **Uppgift 2**: För att formellt lösa problemet behöver du få 195 i genomsnittlig belöning över 100 på varandra följande körningar. Mät det under träningen och se till att du formellt har löst problemet!

## Se resultatet i aktion

Det vore intressant att faktiskt se hur den tränade modellen beter sig. Låt oss köra simuleringen och följa samma strategi för val av åtgärd som under träningen, där vi sampelar enligt sannolikhetsfördelningen i Q-Tabellen: (kodblock 13)

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

![en balanserande vagnstång](../../../../8-Reinforcement/2-Gym/images/cartpole-balance.gif)

---

## 🚀Utmaning

> **Uppgift 3**: Här använde vi den slutgiltiga kopian av Q-Tabellen, vilket kanske inte är den bästa. Kom ihåg att vi har sparat den bästa presterande Q-Tabellen i variabeln `Qbest`! Testa samma exempel med den bästa presterande Q-Tabellen genom att kopiera `Qbest` över till `Q` och se om du märker någon skillnad.

> **Uppgift 4**: Här valde vi inte den bästa åtgärden vid varje steg, utan samplade snarare enligt motsvarande sannolikhetsfördelning. Skulle det inte vara mer meningsfullt att alltid välja den bästa åtgärden, med högst Q-Tabellvärde? Detta kan göras med hjälp av funktionen `np.argmax` för att ta reda på åtgärdsnumret som motsvarar det högsta Q-Tabellvärdet. Implementera denna strategi och se om det förbättrar balanseringen.

## [Quiz efter föreläsningen](https://ff-quizzes.netlify.app/en/ml/)

## Uppgift
[Träna en Mountain Car](assignment.md)

## Slutsats

Vi har nu lärt oss hur man tränar agenter att uppnå bra resultat enbart genom att ge dem en belöningsfunktion som definierar det önskade tillståndet i spelet, och genom att ge dem en möjlighet att intelligent utforska sökutrymmet. Vi har framgångsrikt applicerat Q-Learning-algoritmen i fall med diskreta och kontinuerliga miljöer, men med diskreta handlingar.

Det är viktigt att även studera situationer där handlingsutrymmet också är kontinuerligt, och när observationsutrymmet är mycket mer komplext, såsom bilden från en spelskärm för Atari. I de problemen behöver vi ofta använda kraftfullare maskininlärningstekniker, såsom neurala nätverk, för att uppnå bra resultat. Dessa mer avancerade ämnen är fokus i vår kommande mer avancerade AI-kurs.

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Ansvarsfriskrivning**:
Detta dokument har översatts med hjälp av AI-översättningstjänsten [Co-op Translator](https://github.com/Azure/co-op-translator). Även om vi strävar efter noggrannhet, var vänlig notera att automatiska översättningar kan innehålla fel eller brister. Det ursprungliga dokumentet på dess modersmål bör betraktas som den auktoritativa källan. För kritisk information rekommenderas professionell mänsklig översättning. Vi ansvarar inte för några missförstånd eller feltolkningar som uppstår till följd av användningen av denna översättning.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->