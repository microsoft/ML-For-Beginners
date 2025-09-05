<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "107d5bb29da8a562e7ae72262d251a75",
  "translation_date": "2025-09-05T22:09:29+00:00",
  "source_file": "8-Reinforcement/2-Gym/README.md",
  "language_code": "no"
}
-->
## Forutsetninger

I denne leksjonen skal vi bruke et bibliotek kalt **OpenAI Gym** for å simulere ulike **miljøer**. Du kan kjøre koden fra denne leksjonen lokalt (f.eks. fra Visual Studio Code), i så fall vil simuleringen åpne seg i et nytt vindu. Når du kjører koden online, kan det være nødvendig å gjøre noen justeringer i koden, som beskrevet [her](https://towardsdatascience.com/rendering-openai-gym-envs-on-binder-and-google-colab-536f99391cc7).

## OpenAI Gym

I den forrige leksjonen ble reglene for spillet og tilstanden definert av klassen `Board`, som vi laget selv. Her skal vi bruke et spesielt **simuleringsmiljø** som simulerer fysikken bak den balanserende stangen. Et av de mest populære simuleringsmiljøene for trening av forsterkningslæringsalgoritmer kalles [Gym](https://gym.openai.com/), som vedlikeholdes av [OpenAI](https://openai.com/). Ved å bruke dette gymmet kan vi lage ulike **miljøer**, fra en CartPole-simulering til Atari-spill.

> **Merk**: Du kan se andre tilgjengelige miljøer fra OpenAI Gym [her](https://gym.openai.com/envs/#classic_control).

Først, la oss installere gym og importere nødvendige biblioteker (kodeblokk 1):

```python
import sys
!{sys.executable} -m pip install gym 

import gym
import matplotlib.pyplot as plt
import numpy as np
import random
```

## Oppgave - initialiser et CartPole-miljø

For å jobbe med et CartPole-balanseringsproblem må vi initialisere det tilsvarende miljøet. Hvert miljø er knyttet til:

- **Observasjonsrom** som definerer strukturen til informasjonen vi mottar fra miljøet. For CartPole-problemet mottar vi posisjonen til stangen, hastighet og noen andre verdier.

- **Handlingsrom** som definerer mulige handlinger. I vårt tilfelle er handlingsrommet diskret og består av to handlinger - **venstre** og **høyre**. (kodeblokk 2)

1. For å initialisere, skriv følgende kode:

    ```python
    env = gym.make("CartPole-v1")
    print(env.action_space)
    print(env.observation_space)
    print(env.action_space.sample())
    ```

For å se hvordan miljøet fungerer, la oss kjøre en kort simulering i 100 steg. Ved hvert steg gir vi en av handlingene som skal utføres - i denne simuleringen velger vi bare tilfeldig en handling fra `action_space`.

1. Kjør koden nedenfor og se hva det fører til.

    ✅ Husk at det er foretrukket å kjøre denne koden på en lokal Python-installasjon! (kodeblokk 3)

    ```python
    env.reset()
    
    for i in range(100):
       env.render()
       env.step(env.action_space.sample())
    env.close()
    ```

    Du bør se noe som ligner på dette bildet:

    ![ikke-balanserende CartPole](../../../../8-Reinforcement/2-Gym/images/cartpole-nobalance.gif)

1. Under simuleringen må vi få observasjoner for å avgjøre hvordan vi skal handle. Faktisk returnerer step-funksjonen nåværende observasjoner, en belønningsfunksjon og en flagg som indikerer om det gir mening å fortsette simuleringen eller ikke: (kodeblokk 4)

    ```python
    env.reset()
    
    done = False
    while not done:
       env.render()
       obs, rew, done, info = env.step(env.action_space.sample())
       print(f"{obs} -> {rew}")
    env.close()
    ```

    Du vil ende opp med å se noe som dette i notebook-utgangen:

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

    Observasjonsvektoren som returneres ved hvert steg av simuleringen inneholder følgende verdier:
    - Posisjon til vognen
    - Hastighet til vognen
    - Vinkel til stangen
    - Rotasjonshastighet til stangen

1. Finn minimums- og maksimumsverdien av disse tallene: (kodeblokk 5)

    ```python
    print(env.observation_space.low)
    print(env.observation_space.high)
    ```

    Du vil også legge merke til at belønningsverdien ved hvert simuleringssteg alltid er 1. Dette er fordi målet vårt er å overleve så lenge som mulig, dvs. holde stangen i en rimelig vertikal posisjon i lengst mulig tid.

    ✅ Faktisk anses CartPole-simuleringen som løst hvis vi klarer å oppnå en gjennomsnittlig belønning på 195 over 100 påfølgende forsøk.

## Diskretisering av tilstand

I Q-Læring må vi bygge en Q-Tabell som definerer hva vi skal gjøre i hver tilstand. For å kunne gjøre dette må tilstanden være **diskret**, mer presist, den bør inneholde et begrenset antall diskrete verdier. Dermed må vi på en eller annen måte **diskretisere** observasjonene våre, og kartlegge dem til et begrenset sett med tilstander.

Det finnes noen måter vi kan gjøre dette på:

- **Del inn i intervaller**. Hvis vi kjenner intervallet til en viss verdi, kan vi dele dette intervallet inn i et antall **intervaller**, og deretter erstatte verdien med nummeret på intervallet den tilhører. Dette kan gjøres ved hjelp av numpy-metoden [`digitize`](https://numpy.org/doc/stable/reference/generated/numpy.digitize.html). I dette tilfellet vil vi nøyaktig vite størrelsen på tilstanden, fordi den vil avhenge av antall intervaller vi velger for digitalisering.

✅ Vi kan bruke lineær interpolasjon for å bringe verdier til et begrenset intervall (for eksempel fra -20 til 20), og deretter konvertere tallene til heltall ved å runde dem av. Dette gir oss litt mindre kontroll over størrelsen på tilstanden, spesielt hvis vi ikke kjenner de nøyaktige grensene for inngangsverdiene. For eksempel, i vårt tilfelle har 2 av 4 verdier ikke øvre/nedre grenser for verdiene sine, noe som kan resultere i et uendelig antall tilstander.

I vårt eksempel vil vi gå for den andre tilnærmingen. Som du kanskje legger merke til senere, til tross for udefinerte øvre/nedre grenser, tar disse verdiene sjelden verdier utenfor visse begrensede intervaller, og dermed vil tilstander med ekstreme verdier være svært sjeldne.

1. Her er funksjonen som vil ta observasjonen fra modellen vår og produsere en tuple med 4 heltallsverdier: (kodeblokk 6)

    ```python
    def discretize(x):
        return tuple((x/np.array([0.25, 0.25, 0.01, 0.1])).astype(np.int))
    ```

1. La oss også utforske en annen diskretiseringsmetode ved bruk av intervaller: (kodeblokk 7)

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

1. La oss nå kjøre en kort simulering og observere disse diskrete miljøverdiene. Prøv gjerne både `discretize` og `discretize_bins` og se om det er noen forskjell.

    ✅ `discretize_bins` returnerer nummeret på intervallet, som er 0-basert. Dermed returnerer den for verdier av inngangsvariabelen rundt 0 nummeret fra midten av intervallet (10). I `discretize` brydde vi oss ikke om området for utgangsverdiene, og tillot dem å være negative, slik at tilstandsverdiene ikke er forskjøvet, og 0 tilsvarer 0. (kodeblokk 8)

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

    ✅ Fjern kommentaren på linjen som starter med `env.render` hvis du vil se hvordan miljøet utføres. Ellers kan du utføre det i bakgrunnen, som er raskere. Vi vil bruke denne "usynlige" utførelsen under vår Q-Læringsprosess.

## Strukturen til Q-Tabellen

I vår forrige leksjon var tilstanden et enkelt par med tall fra 0 til 8, og dermed var det praktisk å representere Q-Tabellen med en numpy-tensor med formen 8x8x2. Hvis vi bruker intervall-diskretisering, er størrelsen på tilstandsvektoren vår også kjent, så vi kan bruke samme tilnærming og representere tilstanden med en array med formen 20x20x10x10x2 (her er 2 dimensjonen til handlingsrommet, og de første dimensjonene tilsvarer antall intervaller vi har valgt å bruke for hver av parameterne i observasjonsrommet).

Imidlertid er det noen ganger ikke kjent de nøyaktige dimensjonene til observasjonsrommet. I tilfelle av funksjonen `discretize`, kan vi aldri være sikre på at tilstanden vår holder seg innenfor visse grenser, fordi noen av de opprinnelige verdiene ikke er begrenset. Dermed vil vi bruke en litt annen tilnærming og representere Q-Tabellen med en ordbok.

1. Bruk paret *(state,action)* som nøkkelen i ordboken, og verdien vil tilsvare verdien i Q-Tabellen. (kodeblokk 9)

    ```python
    Q = {}
    actions = (0,1)
    
    def qvalues(state):
        return [Q.get((state,a),0) for a in actions]
    ```

    Her definerer vi også en funksjon `qvalues()`, som returnerer en liste med verdier fra Q-Tabellen for en gitt tilstand som tilsvarer alle mulige handlinger. Hvis oppføringen ikke er til stede i Q-Tabellen, vil vi returnere 0 som standard.

## La oss starte Q-Læring

Nå er vi klare til å lære Peter å balansere!

1. Først, la oss sette noen hyperparametere: (kodeblokk 10)

    ```python
    # hyperparameters
    alpha = 0.3
    gamma = 0.9
    epsilon = 0.90
    ```

    Her er `alpha` **læringsraten** som definerer i hvilken grad vi skal justere de nåværende verdiene i Q-Tabellen ved hvert steg. I den forrige leksjonen startet vi med 1, og deretter reduserte vi `alpha` til lavere verdier under treningen. I dette eksemplet vil vi holde den konstant bare for enkelhets skyld, og du kan eksperimentere med å justere `alpha`-verdiene senere.

    `gamma` er **diskonteringsfaktoren** som viser i hvilken grad vi skal prioritere fremtidig belønning over nåværende belønning.

    `epsilon` er **utforsknings-/utnyttelsesfaktoren** som avgjør om vi skal foretrekke utforskning fremfor utnyttelse eller omvendt. I algoritmen vår vil vi i `epsilon` prosent av tilfellene velge neste handling i henhold til verdiene i Q-Tabellen, og i de resterende tilfellene vil vi utføre en tilfeldig handling. Dette vil tillate oss å utforske områder av søkeområdet som vi aldri har sett før.

    ✅ Når det gjelder balansering - å velge en tilfeldig handling (utforskning) vil fungere som et tilfeldig dytt i feil retning, og stangen må lære seg å gjenopprette balansen fra disse "feilene".

### Forbedre algoritmen

Vi kan også gjøre to forbedringer i algoritmen vår fra forrige leksjon:

- **Beregn gjennomsnittlig kumulativ belønning** over et antall simuleringer. Vi vil skrive ut fremgangen hver 5000 iterasjoner, og vi vil beregne gjennomsnittet av vår kumulative belønning over den perioden. Det betyr at hvis vi får mer enn 195 poeng - kan vi anse problemet som løst, med enda høyere kvalitet enn nødvendig.

- **Beregn maksimal gjennomsnittlig kumulativ belønning**, `Qmax`, og vi vil lagre Q-Tabellen som tilsvarer det resultatet. Når du kjører treningen, vil du legge merke til at noen ganger begynner den gjennomsnittlige kumulative belønningen å synke, og vi ønsker å beholde verdiene i Q-Tabellen som tilsvarer den beste modellen observert under treningen.

1. Samle alle kumulative belønninger ved hver simulering i `rewards`-vektoren for videre plotting. (kodeblokk 11)

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

Hva du kan legge merke til fra disse resultatene:

- **Nær målet vårt**. Vi er veldig nær å oppnå målet om å få 195 kumulative belønninger over 100+ påfølgende simuleringer, eller vi kan faktisk ha oppnådd det! Selv om vi får mindre tall, vet vi fortsatt ikke, fordi vi beregner gjennomsnittet over 5000 kjøringer, og bare 100 kjøringer er nødvendig i de formelle kriteriene.

- **Belønningen begynner å synke**. Noen ganger begynner belønningen å synke, noe som betyr at vi kan "ødelegge" allerede lærte verdier i Q-Tabellen med de som gjør situasjonen verre.

Denne observasjonen er mer tydelig synlig hvis vi plotter treningsfremgangen.

## Plotting av treningsfremgang

Under treningen har vi samlet den kumulative belønningsverdien ved hver iterasjon i `rewards`-vektoren. Slik ser det ut når vi plotter det mot iterasjonsnummeret:

```python
plt.plot(rewards)
```

![rå fremgang](../../../../8-Reinforcement/2-Gym/images/train_progress_raw.png)

Fra denne grafen er det ikke mulig å si noe, fordi på grunn av den stokastiske treningsprosessen varierer lengden på treningsøktene sterkt. For å gi mer mening til denne grafen kan vi beregne **glidende gjennomsnitt** over en serie eksperimenter, la oss si 100. Dette kan gjøres praktisk ved hjelp av `np.convolve`: (kodeblokk 12)

```python
def running_average(x,window):
    return np.convolve(x,np.ones(window)/window,mode='valid')

plt.plot(running_average(rewards,100))
```

![treningsfremgang](../../../../8-Reinforcement/2-Gym/images/train_progress_runav.png)

## Variasjon av hyperparametere

For å gjøre læringen mer stabil gir det mening å justere noen av hyperparameterne våre under treningen. Spesielt:

- **For læringsraten**, `alpha`, kan vi starte med verdier nær 1, og deretter fortsette å redusere parameteren. Med tiden vil vi få gode sannsynlighetsverdier i Q-Tabellen, og dermed bør vi justere dem litt, og ikke overskrive dem helt med nye verdier.

- **Øk epsilon**. Vi kan ønske å øke `epsilon` sakte, for å utforske mindre og utnytte mer. Det gir sannsynligvis mening å starte med en lav verdi av `epsilon`, og bevege seg opp mot nesten 1.
> **Oppgave 1**: Prøv deg frem med verdier for hyperparametere og se om du kan oppnå høyere kumulativ belønning. Får du over 195?
> **Oppgave 2**: For å løse problemet formelt, må du oppnå en gjennomsnittlig belønning på 195 over 100 sammenhengende kjøringer. Mål dette under treningen og sørg for at du har løst problemet formelt!

## Se resultatet i praksis

Det kan være interessant å faktisk se hvordan den trente modellen oppfører seg. La oss kjøre simuleringen og følge samme strategi for valg av handlinger som under treningen, ved å prøve ut fra sannsynlighetsfordelingen i Q-Tabellen: (kodeblokk 13)

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

Du bør se noe som dette:

![en balanserende cartpole](../../../../8-Reinforcement/2-Gym/images/cartpole-balance.gif)

---

## 🚀Utfordring

> **Oppgave 3**: Her brukte vi den endelige kopien av Q-Tabellen, som kanskje ikke er den beste. Husk at vi har lagret den best-presterende Q-Tabellen i variabelen `Qbest`! Prøv det samme eksempelet med den best-presterende Q-Tabellen ved å kopiere `Qbest` over til `Q` og se om du merker noen forskjell.

> **Oppgave 4**: Her valgte vi ikke den beste handlingen på hvert steg, men prøvde heller ut fra den tilsvarende sannsynlighetsfordelingen. Ville det gi mer mening å alltid velge den beste handlingen, med den høyeste verdien i Q-Tabellen? Dette kan gjøres ved å bruke funksjonen `np.argmax` for å finne handlingsnummeret som tilsvarer den høyeste verdien i Q-Tabellen. Implementer denne strategien og se om det forbedrer balansen.

## [Quiz etter forelesning](https://ff-quizzes.netlify.app/en/ml/)

## Oppgave
[Tren en Mountain Car](assignment.md)

## Konklusjon

Vi har nå lært hvordan vi kan trene agenter til å oppnå gode resultater bare ved å gi dem en belønningsfunksjon som definerer ønsket tilstand i spillet, og ved å gi dem muligheten til å utforske søkeområdet intelligent. Vi har med suksess anvendt Q-Learning-algoritmen i tilfeller med diskrete og kontinuerlige miljøer, men med diskrete handlinger.

Det er også viktig å studere situasjoner der handlingsrommet også er kontinuerlig, og når observasjonsrommet er mye mer komplekst, som bildet fra skjermen i et Atari-spill. I slike problemer trenger vi ofte å bruke mer kraftige maskinlæringsteknikker, som nevrale nettverk, for å oppnå gode resultater. Disse mer avanserte temaene er emnet for vårt kommende mer avanserte AI-kurs.

---

**Ansvarsfraskrivelse**:  
Dette dokumentet er oversatt ved hjelp av AI-oversettelsestjenesten [Co-op Translator](https://github.com/Azure/co-op-translator). Selv om vi streber etter nøyaktighet, vær oppmerksom på at automatiserte oversettelser kan inneholde feil eller unøyaktigheter. Det originale dokumentet på sitt opprinnelige språk bør anses som den autoritative kilden. For kritisk informasjon anbefales profesjonell menneskelig oversettelse. Vi er ikke ansvarlige for misforståelser eller feiltolkninger som oppstår ved bruk av denne oversettelsen.