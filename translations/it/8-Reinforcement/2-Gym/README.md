<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9660fbd80845c59c15715cb418cd6e23",
  "translation_date": "2025-08-29T22:15:12+00:00",
  "source_file": "8-Reinforcement/2-Gym/README.md",
  "language_code": "it"
}
-->
## Prerequisiti

In questa lezione utilizzeremo una libreria chiamata **OpenAI Gym** per simulare diversi **ambienti**. Puoi eseguire il codice di questa lezione localmente (ad esempio da Visual Studio Code), nel qual caso la simulazione si aprirà in una nuova finestra. Quando esegui il codice online, potrebbe essere necessario apportare alcune modifiche al codice, come descritto [qui](https://towardsdatascience.com/rendering-openai-gym-envs-on-binder-and-google-colab-536f99391cc7).

## OpenAI Gym

Nella lezione precedente, le regole del gioco e lo stato erano definiti dalla classe `Board` che abbiamo creato noi stessi. Qui utilizzeremo un **ambiente di simulazione** speciale, che simulerà la fisica dietro il bilanciamento del palo. Uno degli ambienti di simulazione più popolari per l'addestramento di algoritmi di apprendimento per rinforzo è chiamato [Gym](https://gym.openai.com/), mantenuto da [OpenAI](https://openai.com/). Utilizzando questo Gym possiamo creare diversi **ambienti**, dalla simulazione del cartpole ai giochi Atari.

> **Nota**: Puoi vedere altri ambienti disponibili su OpenAI Gym [qui](https://gym.openai.com/envs/#classic_control).

Per prima cosa, installiamo Gym e importiamo le librerie necessarie (blocco di codice 1):

```python
import sys
!{sys.executable} -m pip install gym 

import gym
import matplotlib.pyplot as plt
import numpy as np
import random
```

## Esercizio - inizializzare un ambiente cartpole

Per lavorare con il problema del bilanciamento del cartpole, dobbiamo inizializzare l'ambiente corrispondente. Ogni ambiente è associato a:

- **Observation space** che definisce la struttura delle informazioni che riceviamo dall'ambiente. Per il problema del cartpole, riceviamo la posizione del palo, la velocità e altri valori.

- **Action space** che definisce le azioni possibili. Nel nostro caso, lo spazio delle azioni è discreto e consiste in due azioni: **sinistra** e **destra**. (blocco di codice 2)

1. Per inizializzare, digita il seguente codice:

    ```python
    env = gym.make("CartPole-v1")
    print(env.action_space)
    print(env.observation_space)
    print(env.action_space.sample())
    ```

Per vedere come funziona l'ambiente, eseguiamo una breve simulazione di 100 passi. Ad ogni passo, forniamo una delle azioni da intraprendere: in questa simulazione selezioniamo casualmente un'azione da `action_space`.

1. Esegui il codice qui sotto e osserva cosa succede.

    ✅ Ricorda che è preferibile eseguire questo codice su un'installazione locale di Python! (blocco di codice 3)

    ```python
    env.reset()
    
    for i in range(100):
       env.render()
       env.step(env.action_space.sample())
    env.close()
    ```

    Dovresti vedere qualcosa di simile a questa immagine:

    ![cartpole senza bilanciamento](../../../../8-Reinforcement/2-Gym/images/cartpole-nobalance.gif)

1. Durante la simulazione, dobbiamo ottenere osservazioni per decidere come agire. Infatti, la funzione step restituisce le osservazioni attuali, una funzione di ricompensa e il flag "done" che indica se ha senso continuare la simulazione o meno: (blocco di codice 4)

    ```python
    env.reset()
    
    done = False
    while not done:
       env.render()
       obs, rew, done, info = env.step(env.action_space.sample())
       print(f"{obs} -> {rew}")
    env.close()
    ```

    Vedrai qualcosa di simile a questo nell'output del notebook:

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

    Il vettore di osservazione restituito ad ogni passo della simulazione contiene i seguenti valori:
    - Posizione del carrello
    - Velocità del carrello
    - Angolo del palo
    - Velocità di rotazione del palo

1. Ottieni il valore minimo e massimo di questi numeri: (blocco di codice 5)

    ```python
    print(env.observation_space.low)
    print(env.observation_space.high)
    ```

    Potresti anche notare che il valore della ricompensa ad ogni passo della simulazione è sempre 1. Questo perché il nostro obiettivo è sopravvivere il più a lungo possibile, ovvero mantenere il palo in una posizione ragionevolmente verticale per il periodo di tempo più lungo.

    ✅ Infatti, la simulazione del CartPole è considerata risolta se riusciamo a ottenere una ricompensa media di 195 su 100 prove consecutive.

## Discretizzazione dello stato

Nel Q-Learning, dobbiamo costruire una Q-Table che definisca cosa fare in ogni stato. Per poterlo fare, lo stato deve essere **discreto**, più precisamente, deve contenere un numero finito di valori discreti. Pertanto, dobbiamo in qualche modo **discretizzare** le nostre osservazioni, mappandole a un insieme finito di stati.

Ci sono alcuni modi per farlo:

- **Dividere in intervalli**. Se conosciamo l'intervallo di un determinato valore, possiamo dividere questo intervallo in un numero di **intervalli**, e poi sostituire il valore con il numero dell'intervallo a cui appartiene. Questo può essere fatto utilizzando il metodo [`digitize`](https://numpy.org/doc/stable/reference/generated/numpy.digitize.html) di numpy. In questo caso, conosceremo esattamente la dimensione dello stato, poiché dipenderà dal numero di intervalli che selezioniamo per la digitalizzazione.

✅ Possiamo utilizzare l'interpolazione lineare per portare i valori a un intervallo finito (ad esempio, da -20 a 20), e poi convertire i numeri in interi arrotondandoli. Questo ci dà un po' meno controllo sulla dimensione dello stato, soprattutto se non conosciamo gli intervalli esatti dei valori di input. Ad esempio, nel nostro caso 2 dei 4 valori non hanno limiti superiori/inferiori, il che potrebbe portare a un numero infinito di stati.

Nel nostro esempio, utilizzeremo il secondo approccio. Come potrai notare più avanti, nonostante i limiti superiori/inferiori non definiti, quei valori raramente assumono valori al di fuori di certi intervalli finiti, quindi quegli stati con valori estremi saranno molto rari.

1. Ecco la funzione che prenderà l'osservazione dal nostro modello e produrrà una tupla di 4 valori interi: (blocco di codice 6)

    ```python
    def discretize(x):
        return tuple((x/np.array([0.25, 0.25, 0.01, 0.1])).astype(np.int))
    ```

1. Esploriamo anche un altro metodo di discretizzazione utilizzando gli intervalli: (blocco di codice 7)

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

1. Ora eseguiamo una breve simulazione e osserviamo quei valori discreti dell'ambiente. Sentiti libero di provare sia `discretize` che `discretize_bins` e vedere se c'è una differenza.

    ✅ `discretize_bins` restituisce il numero dell'intervallo, che è basato su 0. Quindi per valori della variabile di input intorno a 0 restituisce il numero dal centro dell'intervallo (10). In `discretize`, non ci siamo preoccupati dell'intervallo dei valori di output, permettendo loro di essere negativi, quindi i valori dello stato non sono spostati e 0 corrisponde a 0. (blocco di codice 8)

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

    ✅ Decommenta la riga che inizia con `env.render` se vuoi vedere come l'ambiente viene eseguito. Altrimenti puoi eseguirlo in background, il che è più veloce. Utilizzeremo questa esecuzione "invisibile" durante il nostro processo di Q-Learning.

## La struttura della Q-Table

Nella lezione precedente, lo stato era una semplice coppia di numeri da 0 a 8, e quindi era conveniente rappresentare la Q-Table con un tensore numpy con una forma di 8x8x2. Se utilizziamo la discretizzazione con intervalli, la dimensione del nostro vettore di stato è anche nota, quindi possiamo utilizzare lo stesso approccio e rappresentare lo stato con un array di forma 20x20x10x10x2 (qui 2 è la dimensione dello spazio delle azioni, e le prime dimensioni corrispondono al numero di intervalli che abbiamo selezionato per ciascuno dei parametri nello spazio delle osservazioni).

Tuttavia, a volte le dimensioni precise dello spazio delle osservazioni non sono note. Nel caso della funzione `discretize`, non possiamo mai essere sicuri che il nostro stato rimanga entro certi limiti, perché alcuni dei valori originali non sono limitati. Pertanto, utilizzeremo un approccio leggermente diverso e rappresenteremo la Q-Table con un dizionario.

1. Usa la coppia *(state,action)* come chiave del dizionario, e il valore corrisponderà al valore della Q-Table. (blocco di codice 9)

    ```python
    Q = {}
    actions = (0,1)
    
    def qvalues(state):
        return [Q.get((state,a),0) for a in actions]
    ```

    Qui definiamo anche una funzione `qvalues()`, che restituisce un elenco di valori della Q-Table per un determinato stato che corrisponde a tutte le azioni possibili. Se la voce non è presente nella Q-Table, restituiremo 0 come valore predefinito.

## Iniziamo il Q-Learning

Ora siamo pronti per insegnare a Peter a bilanciarsi!

1. Per prima cosa, impostiamo alcuni iperparametri: (blocco di codice 10)

    ```python
    # hyperparameters
    alpha = 0.3
    gamma = 0.9
    epsilon = 0.90
    ```

    Qui, `alpha` è il **learning rate** che definisce in che misura dovremmo regolare i valori attuali della Q-Table ad ogni passo. Nella lezione precedente abbiamo iniziato con 1, e poi abbiamo ridotto `alpha` a valori più bassi durante l'addestramento. In questo esempio lo manterremo costante per semplicità, e potrai sperimentare con l'aggiustamento dei valori di `alpha` più tardi.

    `gamma` è il **discount factor** che mostra in che misura dovremmo dare priorità alla ricompensa futura rispetto a quella attuale.

    `epsilon` è il **exploration/exploitation factor** che determina se dovremmo preferire l'esplorazione o lo sfruttamento. Nel nostro algoritmo, in una percentuale di casi determinata da `epsilon` selezioneremo la prossima azione in base ai valori della Q-Table, e nel restante numero di casi eseguiremo un'azione casuale. Questo ci permetterà di esplorare aree dello spazio di ricerca che non abbiamo mai visto prima.

    ✅ In termini di bilanciamento - scegliere un'azione casuale (esplorazione) agirebbe come un colpo casuale nella direzione sbagliata, e il palo dovrebbe imparare a recuperare l'equilibrio da questi "errori".

### Migliorare l'algoritmo

Possiamo anche apportare due miglioramenti al nostro algoritmo rispetto alla lezione precedente:

- **Calcolare la ricompensa cumulativa media**, su un numero di simulazioni. Stampiamo i progressi ogni 5000 iterazioni e calcoliamo la media della ricompensa cumulativa su quel periodo di tempo. Significa che se otteniamo più di 195 punti, possiamo considerare il problema risolto, con una qualità anche superiore a quella richiesta.

- **Calcolare il massimo risultato cumulativo medio**, `Qmax`, e memorizzeremo la Q-Table corrispondente a quel risultato. Quando esegui l'addestramento noterai che a volte il risultato cumulativo medio inizia a diminuire, e vogliamo mantenere i valori della Q-Table che corrispondono al miglior modello osservato durante l'addestramento.

1. Raccogli tutte le ricompense cumulative ad ogni simulazione nel vettore `rewards` per ulteriori grafici. (blocco di codice 11)

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

Cosa puoi notare da questi risultati:

- **Vicini al nostro obiettivo**. Siamo molto vicini a raggiungere l'obiettivo di ottenere 195 ricompense cumulative su 100+ esecuzioni consecutive della simulazione, o potremmo averlo effettivamente raggiunto! Anche se otteniamo numeri più piccoli, non lo sappiamo con certezza, perché facciamo la media su 5000 esecuzioni, e solo 100 esecuzioni sono richieste nei criteri formali.

- **La ricompensa inizia a diminuire**. A volte la ricompensa inizia a diminuire, il che significa che possiamo "distruggere" i valori già appresi nella Q-Table con quelli che peggiorano la situazione.

Questa osservazione è più chiaramente visibile se tracciamo i progressi dell'addestramento.

## Tracciare i progressi dell'addestramento

Durante l'addestramento, abbiamo raccolto il valore della ricompensa cumulativa ad ogni iterazione nel vettore `rewards`. Ecco come appare quando lo tracciamo rispetto al numero di iterazioni:

```python
plt.plot(rewards)
```

![progressi grezzi](../../../../translated_images/train_progress_raw.2adfdf2daea09c596fc786fa347a23e9aceffe1b463e2257d20a9505794823ec.it.png)

Da questo grafico, non è possibile dedurre molto, perché a causa della natura del processo di addestramento stocastico la lunghezza delle sessioni di addestramento varia notevolmente. Per dare più senso a questo grafico, possiamo calcolare la **media mobile** su una serie di esperimenti, diciamo 100. Questo può essere fatto comodamente usando `np.convolve`: (blocco di codice 12)

```python
def running_average(x,window):
    return np.convolve(x,np.ones(window)/window,mode='valid')

plt.plot(running_average(rewards,100))
```

![progressi dell'addestramento](../../../../translated_images/train_progress_runav.c71694a8fa9ab35935aff6f109e5ecdfdbdf1b0ae265da49479a81b5fae8f0aa.it.png)

## Variazione degli iperparametri

Per rendere l'apprendimento più stabile, ha senso regolare alcuni dei nostri iperparametri durante l'addestramento. In particolare:

- **Per il learning rate**, `alpha`, possiamo iniziare con valori vicini a 1 e poi continuare a diminuire il parametro. Con il tempo, otterremo buone probabilità nella Q-Table, e quindi dovremmo regolarle leggermente, senza sovrascrivere completamente con nuovi valori.

- **Aumentare epsilon**. Potremmo voler aumentare lentamente `epsilon`, per esplorare meno e sfruttare di più. Probabilmente ha senso iniziare con un valore basso di `epsilon` e aumentarlo fino a quasi 1.
> **Attività 1**: Prova a modificare i valori degli iperparametri e verifica se riesci a ottenere una ricompensa cumulativa più alta. Riesci a superare 195?
> **Compito 2**: Per risolvere formalmente il problema, è necessario ottenere una ricompensa media di 195 in 100 esecuzioni consecutive. Misura questo valore durante l'addestramento e assicurati di aver risolto formalmente il problema!

## Osservare il risultato in azione

Sarebbe interessante vedere effettivamente come si comporta il modello addestrato. Eseguiamo la simulazione e seguiamo la stessa strategia di selezione delle azioni utilizzata durante l'addestramento, campionando in base alla distribuzione di probabilità nella Q-Table: (blocco di codice 13)

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

Dovresti vedere qualcosa del genere:

![un carrello in equilibrio](../../../../8-Reinforcement/2-Gym/images/cartpole-balance.gif)

---

## 🚀Sfida

> **Compito 3**: Qui abbiamo utilizzato la copia finale della Q-Table, che potrebbe non essere la migliore. Ricorda che abbiamo salvato la Q-Table con le migliori prestazioni nella variabile `Qbest`! Prova lo stesso esempio con la Q-Table con le migliori prestazioni copiando `Qbest` su `Q` e verifica se noti differenze.

> **Compito 4**: Qui non stavamo selezionando la migliore azione a ogni passo, ma piuttosto campionando in base alla distribuzione di probabilità corrispondente. Avrebbe più senso selezionare sempre l'azione migliore, con il valore più alto nella Q-Table? Questo può essere fatto utilizzando la funzione `np.argmax` per trovare il numero dell'azione corrispondente al valore più alto nella Q-Table. Implementa questa strategia e verifica se migliora l'equilibrio.

## [Quiz post-lezione](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/48/)

## Compito
[Addestra una Mountain Car](assignment.md)

## Conclusione

Abbiamo ora imparato come addestrare agenti per ottenere buoni risultati semplicemente fornendo loro una funzione di ricompensa che definisce lo stato desiderato del gioco e dando loro l'opportunità di esplorare in modo intelligente lo spazio di ricerca. Abbiamo applicato con successo l'algoritmo di Q-Learning nei casi di ambienti discreti e continui, ma con azioni discrete.

È importante anche studiare situazioni in cui lo stato delle azioni è continuo e quando lo spazio di osservazione è molto più complesso, come un'immagine dello schermo di un gioco Atari. In questi problemi spesso è necessario utilizzare tecniche di machine learning più potenti, come le reti neurali, per ottenere buoni risultati. Questi argomenti più avanzati saranno trattati nel nostro prossimo corso avanzato di intelligenza artificiale.

---

**Disclaimer**:  
Questo documento è stato tradotto utilizzando il servizio di traduzione automatica [Co-op Translator](https://github.com/Azure/co-op-translator). Sebbene ci impegniamo per garantire l'accuratezza, si prega di notare che le traduzioni automatiche possono contenere errori o imprecisioni. Il documento originale nella sua lingua nativa dovrebbe essere considerato la fonte autorevole. Per informazioni critiche, si raccomanda una traduzione professionale effettuata da un traduttore umano. Non siamo responsabili per eventuali fraintendimenti o interpretazioni errate derivanti dall'uso di questa traduzione.