<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "107d5bb29da8a562e7ae72262d251a75",
  "translation_date": "2025-09-05T01:15:52+00:00",
  "source_file": "8-Reinforcement/2-Gym/README.md",
  "language_code": "da"
}
-->
# CartPole Skating

Problemet, vi har arbejdet med i den tidligere lektion, kan virke som et legetøjsproblem, der ikke rigtig har relevans for virkelige scenarier. Dette er dog ikke tilfældet, da mange virkelige problemer deler samme karakteristika – herunder at spille skak eller Go. De er ens, fordi vi også har et bræt med givne regler og en **diskret tilstand**.

## [Quiz før lektionen](https://ff-quizzes.netlify.app/en/ml/)

## Introduktion

I denne lektion vil vi anvende de samme principper fra Q-Learning på et problem med en **kontinuerlig tilstand**, dvs. en tilstand, der er givet ved en eller flere reelle tal. Vi vil arbejde med følgende problem:

> **Problem**: Hvis Peter vil undslippe ulven, skal han kunne bevæge sig hurtigere. Vi vil se, hvordan Peter kan lære at skøjte, især hvordan han kan holde balancen, ved hjælp af Q-Learning.

![Den store flugt!](../../../../8-Reinforcement/2-Gym/images/escape.png)

> Peter og hans venner bliver kreative for at undslippe ulven! Billede af [Jen Looper](https://twitter.com/jenlooper)

Vi vil bruge en forenklet version af balancering kendt som **CartPole**-problemet. I CartPole-verdenen har vi en horisontal slider, der kan bevæge sig til venstre eller højre, og målet er at balancere en lodret stang oven på slideren.

## Forudsætninger

I denne lektion vil vi bruge et bibliotek kaldet **OpenAI Gym** til at simulere forskellige **miljøer**. Du kan køre lektionens kode lokalt (f.eks. fra Visual Studio Code), hvor simuleringen åbnes i et nyt vindue. Hvis du kører koden online, skal du muligvis lave nogle justeringer, som beskrevet [her](https://towardsdatascience.com/rendering-openai-gym-envs-on-binder-and-google-colab-536f99391cc7).

## OpenAI Gym

I den tidligere lektion blev spillets regler og tilstand givet af klassen `Board`, som vi selv definerede. Her vil vi bruge et specielt **simuleringsmiljø**, der simulerer fysikken bag den balancerende stang. Et af de mest populære simuleringsmiljøer til træning af reinforcement learning-algoritmer kaldes [Gym](https://gym.openai.com/), som vedligeholdes af [OpenAI](https://openai.com/). Ved at bruge dette Gym kan vi skabe forskellige **miljøer**, fra CartPole-simuleringer til Atari-spil.

> **Note**: Du kan se andre miljøer, der er tilgængelige fra OpenAI Gym [her](https://gym.openai.com/envs/#classic_control).

Først skal vi installere Gym og importere de nødvendige biblioteker (kodeblok 1):

```python
import sys
!{sys.executable} -m pip install gym 

import gym
import matplotlib.pyplot as plt
import numpy as np
import random
```

## Øvelse - initialiser et CartPole-miljø

For at arbejde med CartPole-balanceringsproblemet skal vi initialisere det tilsvarende miljø. Hvert miljø er forbundet med:

- **Observation space**, der definerer strukturen af den information, vi modtager fra miljøet. For CartPole-problemet modtager vi positionen af stangen, hastighed og nogle andre værdier.

- **Action space**, der definerer mulige handlinger. I vores tilfælde er action space diskret og består af to handlinger - **venstre** og **højre**. (kodeblok 2)

1. For at initialisere skal du skrive følgende kode:

    ```python
    env = gym.make("CartPole-v1")
    print(env.action_space)
    print(env.observation_space)
    print(env.action_space.sample())
    ```

For at se, hvordan miljøet fungerer, lad os køre en kort simulering i 100 trin. Ved hvert trin giver vi en af de handlinger, der skal udføres – i denne simulering vælger vi bare tilfældigt en handling fra `action_space`.

1. Kør koden nedenfor og se, hvad det fører til.

    ✅ Husk, at det er bedst at køre denne kode på en lokal Python-installation! (kodeblok 3)

    ```python
    env.reset()
    
    for i in range(100):
       env.render()
       env.step(env.action_space.sample())
    env.close()
    ```

    Du bør se noget, der ligner dette billede:

    ![ikke-balancerende CartPole](../../../../8-Reinforcement/2-Gym/images/cartpole-nobalance.gif)

1. Under simuleringen skal vi få observationer for at beslutte, hvordan vi skal handle. Faktisk returnerer step-funktionen aktuelle observationer, en belønningsfunktion og en "done"-flag, der angiver, om det giver mening at fortsætte simuleringen eller ej: (kodeblok 4)

    ```python
    env.reset()
    
    done = False
    while not done:
       env.render()
       obs, rew, done, info = env.step(env.action_space.sample())
       print(f"{obs} -> {rew}")
    env.close()
    ```

    Du vil ende med at se noget som dette i notebook-outputtet:

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

    Observationsvektoren, der returneres ved hvert trin i simuleringen, indeholder følgende værdier:
    - Position af vognen
    - Hastighed af vognen
    - Vinkel af stangen
    - Rotationshastighed af stangen

1. Få minimums- og maksimumsværdier for disse tal: (kodeblok 5)

    ```python
    print(env.observation_space.low)
    print(env.observation_space.high)
    ```

    Du vil også bemærke, at belønningsværdien ved hvert simuleringstrin altid er 1. Dette skyldes, at vores mål er at overleve så længe som muligt, dvs. holde stangen i en rimelig lodret position i længst mulig tid.

    ✅ Faktisk anses CartPole-simuleringen for at være løst, hvis vi formår at opnå en gennemsnitlig belønning på 195 over 100 på hinanden følgende forsøg.

## Diskretisering af tilstand

I Q-Learning skal vi opbygge en Q-Table, der definerer, hvad vi skal gøre i hver tilstand. For at kunne gøre dette skal tilstanden være **diskret**, mere præcist skal den indeholde et begrænset antal diskrete værdier. Derfor skal vi på en eller anden måde **diskretisere** vores observationer og kortlægge dem til et begrænset sæt af tilstande.

Der er et par måder, vi kan gøre dette på:

- **Opdel i intervaller**. Hvis vi kender intervallet for en bestemt værdi, kan vi opdele dette interval i et antal **intervaller** og derefter erstatte værdien med nummeret på det interval, den tilhører. Dette kan gøres ved hjælp af numpy-metoden [`digitize`](https://numpy.org/doc/stable/reference/generated/numpy.digitize.html). I dette tilfælde vil vi præcist kende størrelsen på tilstanden, da den vil afhænge af antallet af intervaller, vi vælger til digitalisering.

✅ Vi kan bruge lineær interpolation til at bringe værdier til et begrænset interval (f.eks. fra -20 til 20) og derefter konvertere tal til heltal ved at runde dem. Dette giver os lidt mindre kontrol over størrelsen af tilstanden, især hvis vi ikke kender de nøjagtige intervaller for inputværdierne. For eksempel har 2 ud af 4 værdier i vores tilfælde ikke øvre/nedre grænser for deres værdier, hvilket kan resultere i et uendeligt antal tilstande.

I vores eksempel vil vi gå med den anden tilgang. Som du måske bemærker senere, på trods af udefinerede øvre/nedre grænser, tager disse værdier sjældent ekstreme værdier uden for visse begrænsede intervaller, så tilstande med ekstreme værdier vil være meget sjældne.

1. Her er funktionen, der tager observationen fra vores model og producerer en tuple med 4 heltalsværdier: (kodeblok 6)

    ```python
    def discretize(x):
        return tuple((x/np.array([0.25, 0.25, 0.01, 0.1])).astype(np.int))
    ```

1. Lad os også udforske en anden diskretiseringsmetode ved hjælp af intervaller: (kodeblok 7)

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

1. Lad os nu køre en kort simulering og observere disse diskrete miljøværdier. Prøv gerne både `discretize` og `discretize_bins` og se, om der er en forskel.

    ✅ `discretize_bins` returnerer intervalnummeret, som er 0-baseret. For værdier af inputvariablen omkring 0 returnerer den nummeret fra midten af intervallet (10). I `discretize` bekymrede vi os ikke om outputværdiernes interval, hvilket tillod dem at være negative, så tilstandsværdierne er ikke forskudt, og 0 svarer til 0. (kodeblok 8)

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

    ✅ Fjern kommentaren fra linjen, der starter med `env.render`, hvis du vil se, hvordan miljøet udføres. Ellers kan du udføre det i baggrunden, hvilket er hurtigere. Vi vil bruge denne "usynlige" udførelse under vores Q-Learning-proces.

## Q-Tabellens struktur

I vores tidligere lektion var tilstanden et simpelt par af tal fra 0 til 8, og det var derfor bekvemt at repræsentere Q-Table med en numpy-tensor med en form på 8x8x2. Hvis vi bruger intervaller til diskretisering, er størrelsen af vores tilstandsvektor også kendt, så vi kan bruge samme tilgang og repræsentere tilstanden med en array med formen 20x20x10x10x2 (her er 2 dimensionen af action space, og de første dimensioner svarer til antallet af intervaller, vi har valgt at bruge for hver af parametrene i observationsrummet).

Men nogle gange er de præcise dimensioner af observationsrummet ikke kendt. I tilfælde af funktionen `discretize` kan vi aldrig være sikre på, at vores tilstand holder sig inden for visse grænser, fordi nogle af de oprindelige værdier ikke er begrænsede. Derfor vil vi bruge en lidt anderledes tilgang og repræsentere Q-Table med en ordbog.

1. Brug parret *(state,action)* som ordbogs-nøgle, og værdien vil svare til Q-Table-indgangsværdien. (kodeblok 9)

    ```python
    Q = {}
    actions = (0,1)
    
    def qvalues(state):
        return [Q.get((state,a),0) for a in actions]
    ```

    Her definerer vi også en funktion `qvalues()`, som returnerer en liste over Q-Table-værdier for en given tilstand, der svarer til alle mulige handlinger. Hvis indgangen ikke er til stede i Q-Table, returnerer vi 0 som standard.

## Lad os starte Q-Learning

Nu er vi klar til at lære Peter at balancere!

1. Først skal vi sætte nogle hyperparametre: (kodeblok 10)

    ```python
    # hyperparameters
    alpha = 0.3
    gamma = 0.9
    epsilon = 0.90
    ```

    Her er `alpha` **læringsraten**, der definerer, i hvilken grad vi skal justere de aktuelle værdier i Q-Table ved hvert trin. I den tidligere lektion startede vi med 1 og reducerede derefter `alpha` til lavere værdier under træningen. I dette eksempel vil vi holde den konstant for enkelhedens skyld, og du kan eksperimentere med at justere `alpha`-værdier senere.

    `gamma` er **diskonteringsfaktoren**, der viser, i hvilken grad vi skal prioritere fremtidig belønning over nuværende belønning.

    `epsilon` er **udforsknings-/udnyttelsesfaktoren**, der bestemmer, om vi skal foretrække udforskning frem for udnyttelse eller omvendt. I vores algoritme vil vi i `epsilon` procent af tilfældene vælge den næste handling baseret på Q-Table-værdier, og i de resterende tilfælde vil vi udføre en tilfældig handling. Dette vil give os mulighed for at udforske områder af søgefeltet, som vi aldrig har set før.

    ✅ Når det gælder balancering – at vælge en tilfældig handling (udforskning) vil fungere som et tilfældigt skub i den forkerte retning, og stangen skal lære at genvinde balancen fra disse "fejl".

### Forbedr algoritmen

Vi kan også lave to forbedringer af vores algoritme fra den tidligere lektion:

- **Beregn gennemsnitlig kumulativ belønning** over et antal simuleringer. Vi vil udskrive fremskridtet hver 5000 iterationer og gennemsnitliggøre vores kumulative belønning over denne periode. Det betyder, at hvis vi får mere end 195 point, kan vi betragte problemet som løst, med endnu højere kvalitet end krævet.

- **Beregn maksimal gennemsnitlig kumulativ belønning**, `Qmax`, og vi vil gemme Q-Table, der svarer til dette resultat. Når du kører træningen, vil du bemærke, at den gennemsnitlige kumulative belønning nogle gange begynder at falde, og vi ønsker at bevare værdierne i Q-Table, der svarer til den bedste model, der er observeret under træningen.

1. Saml alle kumulative belønninger ved hver simulering i `rewards`-vektoren til senere plotning. (kodeblok 11)

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

Hvad du måske bemærker fra disse resultater:

- **Tæt på vores mål**. Vi er meget tæt på at opnå målet om at få 195 kumulative belønninger over 100+ på hinanden følgende simuleringer, eller vi har måske faktisk opnået det! Selv hvis vi får mindre tal, ved vi stadig ikke, fordi vi gennemsnitliggør over 5000 kørsel, og kun 100 kørsel er krævet i de formelle kriterier.

- **Belønning begynder at falde**. Nogle gange begynder belønningen at falde, hvilket betyder, at vi kan "ødelægge" allerede lærte værdier i Q-Table med dem, der gør situationen værre.

Denne observation er mere tydelig, hvis vi plotter træningsfremskridtet.

## Plotning af træningsfremskridt

Under træningen har vi samlet den kumulative belønningsværdi ved hver af iterationerne i `rewards`-vektoren. Her er, hvordan det ser ud, når vi plotter det mod iterationsnummeret:

```python
plt.plot(rewards)
```

![råt fremskridt](../../../../8-Reinforcement/2-Gym/images/train_progress_raw.png)

Fra denne graf er det ikke muligt at sige noget, fordi længden af træningssessionerne varierer meget på grund af den stokastiske træningsproces. For at give mere mening til denne graf kan vi beregne **løbende gennemsnit** over en række eksperimenter, lad os sige 100. Dette kan gøres bekvemt ved hjælp af `np.convolve`: (kodeblok 12)

```python
def running_average(x,window):
    return np.convolve(x,np.ones(window)/window,mode='valid')

plt.plot(running_average(rewards,100))
```

![træningsfremskridt](../../../../8-Reinforcement/2-Gym/images/train_progress_runav.png)

## Justering af hyperparametre

For at gøre læringen mere stabil giver det mening at justere nogle af vores hyperparametre under træningen. Især:

- **For læringsraten**, `alpha`, kan vi starte med værdier tæt på 1 og derefter gradvist reducere parameteren. Med tiden vil vi få gode sandsynlighedsværdier i Q-Table, og derfor bør vi justere dem lidt og ikke overskrive dem fuldstændigt med nye værdier.

- **Øg epsilon**. Vi kan langsomt øge `epsilon` for at udforske mindre og udnytte mere. Det giver sandsynligvis mening at starte med en lav værdi af `epsilon` og gradvist øge den til næsten 1.
> **Opgave 1**: Prøv at ændre værdierne for hyperparametrene og se, om du kan opnå en højere samlet belønning. Kommer du over 195?
> **Opgave 2**: For at løse problemet formelt, skal du opnå en gennemsnitlig belønning på 195 over 100 på hinanden følgende kørsler. Mål dette under træningen og sørg for, at du har løst problemet formelt!

## Se resultatet i aktion

Det kunne være interessant at se, hvordan den trænede model opfører sig. Lad os køre simuleringen og følge den samme strategi for valg af handlinger som under træningen, hvor vi sampler i henhold til sandsynlighedsfordelingen i Q-Tabellen: (kodeblok 13)

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

Du bør se noget lignende dette:

![en balancerende cartpole](../../../../8-Reinforcement/2-Gym/images/cartpole-balance.gif)

---

## 🚀Udfordring

> **Opgave 3**: Her brugte vi den endelige kopi af Q-Tabellen, som måske ikke er den bedste. Husk, at vi har gemt den bedst præsterende Q-Tabel i variablen `Qbest`! Prøv det samme eksempel med den bedst præsterende Q-Tabel ved at kopiere `Qbest` over til `Q` og se, om du bemærker en forskel.

> **Opgave 4**: Her valgte vi ikke den bedste handling ved hvert trin, men samplede i stedet med den tilsvarende sandsynlighedsfordeling. Ville det give mere mening altid at vælge den bedste handling med den højeste værdi i Q-Tabellen? Dette kan gøres ved at bruge funktionen `np.argmax` til at finde handlingsnummeret, der svarer til den højeste værdi i Q-Tabellen. Implementer denne strategi og se, om det forbedrer balanceringen.

## [Quiz efter forelæsning](https://ff-quizzes.netlify.app/en/ml/)

## Opgave
[Træn en Mountain Car](assignment.md)

## Konklusion

Vi har nu lært, hvordan man træner agenter til at opnå gode resultater blot ved at give dem en belønningsfunktion, der definerer den ønskede tilstand i spillet, og ved at give dem mulighed for intelligent at udforske søgeområdet. Vi har med succes anvendt Q-Learning-algoritmen i tilfælde af diskrete og kontinuerlige miljøer, men med diskrete handlinger.

Det er også vigtigt at studere situationer, hvor handlingsrummet også er kontinuerligt, og hvor observationsrummet er meget mere komplekst, såsom billedet fra skærmen i Atari-spillet. I disse problemer har vi ofte brug for mere kraftfulde maskinlæringsteknikker, såsom neurale netværk, for at opnå gode resultater. Disse mere avancerede emner er genstand for vores kommende mere avancerede AI-kursus.

---

**Ansvarsfraskrivelse**:  
Dette dokument er blevet oversat ved hjælp af AI-oversættelsestjenesten [Co-op Translator](https://github.com/Azure/co-op-translator). Selvom vi bestræber os på nøjagtighed, skal du være opmærksom på, at automatiserede oversættelser kan indeholde fejl eller unøjagtigheder. Det originale dokument på dets oprindelige sprog bør betragtes som den autoritative kilde. For kritisk information anbefales professionel menneskelig oversættelse. Vi er ikke ansvarlige for eventuelle misforståelser eller fejltolkninger, der opstår som følge af brugen af denne oversættelse.