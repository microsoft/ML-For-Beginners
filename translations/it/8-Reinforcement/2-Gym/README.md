# CartPole Skating

Il problema che abbiamo risolto nella lezione precedente potrebbe sembrare un problema giocattolo, non veramente applicabile a scenari della vita reale. Non è così, perché molti problemi del mondo reale condividono questo scenario - incluso giocare a Scacchi o Go. Sono simili perché abbiamo anche una scacchiera con regole date e uno **stato discreto**.

## [Quiz pre-lezione](https://ff-quizzes.netlify.app/en/ml/)

## Introduzione

In questa lezione applicheremo gli stessi principi del Q-Learning a un problema con **stato continuo**, cioè uno stato dato da uno o più numeri reali. Affronteremo il seguente problema:

> **Problema**: Se Peter vuole sfuggire al lupo, deve essere in grado di muoversi più velocemente. Vedremo come Peter può imparare a pattinare, in particolare a mantenere l'equilibrio, usando il Q-Learning.

![La grande fuga!](../../../../translated_images/it/escape.18862db9930337e3.webp)

> Peter e i suoi amici diventano creativi per scappare dal lupo! Immagine di [Jen Looper](https://twitter.com/jenlooper)

Useremo una versione semplificata dell'equilibrio nota come problema **CartPole**. Nel mondo del cartpole abbiamo uno slider orizzontale che può muoversi a sinistra o a destra, e l'obiettivo è bilanciare un palo verticale sopra lo slider.

<img alt="un cartpole" src="../../../../translated_images/it/cartpole.b5609cc0494a14f7.webp" width="200"/>

## Prerequisiti

In questa lezione useremo una libreria chiamata **OpenAI Gym** per simulare diversi **ambienti**. Puoi eseguire il codice di questa lezione localmente (ad esempio da Visual Studio Code), nel qual caso la simulazione si aprirà in una nuova finestra. Quando esegui il codice online, potresti dover apportare alcune modifiche al codice, come descritto [qui](https://towardsdatascience.com/rendering-openai-gym-envs-on-binder-and-google-colab-536f99391cc7).

## OpenAI Gym

Nella lezione precedente, le regole del gioco e lo stato erano definiti dalla classe `Board` che abbiamo definito noi stessi. Qui useremo un **ambiente di simulazione** speciale, che simula la fisica dietro il palo in equilibrio. Uno degli ambienti di simulazione più popolari per l'addestramento di algoritmi di apprendimento per rinforzo si chiama [Gym](https://gym.openai.com/), mantenuto da [OpenAI](https://openai.com/). Usando questo gym possiamo creare diversi **ambienti** da una simulazione di cartpole a giochi Atari.

> **Nota**: Puoi vedere altri ambienti disponibili in OpenAI Gym [qui](https://gym.openai.com/envs/#classic_control). 

Per prima cosa, installiamo il gym e importiamo le librerie richieste (blocco di codice 1):

```python
import sys
!{sys.executable} -m pip install gym 

import gym
import matplotlib.pyplot as plt
import numpy as np
import random
```

## Esercizio - inizializzare un ambiente cartpole

Per lavorare con un problema di bilanciamento cartpole, dobbiamo inizializzare l'ambiente corrispondente. Ogni ambiente è associato a:

- **Spazio di osservazione** che definisce la struttura dell'informazione che riceviamo dall'ambiente. Per il problema cartpole, riceviamo la posizione del palo, la velocità e alcuni altri valori.

- **Spazio di azione** che definisce le azioni possibili. Nel nostro caso, lo spazio delle azioni è discreto e consiste di due azioni - **sinistra** e **destra**. (blocco di codice 2)

1. Per inizializzare, digita il seguente codice:

    ```python
    env = gym.make("CartPole-v1")
    print(env.action_space)
    print(env.observation_space)
    print(env.action_space.sample())
    ```

Per vedere come funziona l'ambiente, eseguiamo una breve simulazione di 100 passi. Ad ogni passo, forniamo una delle azioni da eseguire - in questa simulazione scegliamo casualmente un'azione da `action_space`.

1. Esegui il codice qui sotto e guarda cosa succede.

    ✅ Ricorda che è preferibile eseguire questo codice su un'installazione Python locale! (blocco di codice 3)

    ```python
    env.reset()
    
    for i in range(100):
       env.render()
       env.step(env.action_space.sample())
    env.close()
    ```

    Dovresti vedere qualcosa di simile a questa immagine:

    ![cartpole non bilanciato](../../../../8-Reinforcement/2-Gym/images/cartpole-nobalance.gif)

1. Durante la simulazione, dobbiamo ottenere osservazioni per decidere come agire. Infatti, la funzione step restituisce osservazioni correnti, una funzione di ricompensa e il flag done che indica se ha senso continuare la simulazione o no: (blocco di codice 4)

    ```python
    env.reset()
    
    done = False
    while not done:
       env.render()
       obs, rew, done, info = env.step(env.action_space.sample())
       print(f"{obs} -> {rew}")
    env.close()
    ```

    Vedrai qualcosa di simile nell'output del notebook:

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

    Il vettore di osservazione restituito a ogni passo della simulazione contiene i seguenti valori:
    - Posizione del carrello
    - Velocità del carrello
    - Angolo del palo
    - Velocità di rotazione del palo

1. Ottieni il valore minimo e massimo di questi numeri: (blocco di codice 5)

    ```python
    print(env.observation_space.low)
    print(env.observation_space.high)
    ```

    Potresti anche notare che il valore della ricompensa ad ogni passo della simulazione è sempre 1. Questo perché il nostro obiettivo è sopravvivere il più a lungo possibile, cioè mantenere il palo in posizione ragionevolmente verticale per il massimo tempo.

    ✅ Infatti, la simulazione CartPole è considerata risolta se riusciamo a ottenere una ricompensa media di 195 su 100 prove consecutive.

## Discretizzazione dello stato

Nel Q-Learning, dobbiamo costruire una Q-Table che definisca cosa fare in ogni stato. Per poterlo fare, lo stato deve essere **discreto**, più precisamente deve contenere un numero finito di valori discreti. Quindi, dobbiamo in qualche modo **discretizzare** le nostre osservazioni, mappandole a un insieme finito di stati.

Ci sono diversi modi per farlo:

- **Dividere in bins**. Se conosciamo l'intervallo di un certo valore, possiamo dividere questo intervallo in un numero di **bins**, e poi sostituire il valore con il numero del bin a cui appartiene. Questo può essere fatto usando il metodo numpy [`digitize`](https://numpy.org/doc/stable/reference/generated/numpy.digitize.html). In questo caso, conosceremo precisamente la dimensione dello stato, perché dipenderà dal numero di bins scelti per la digitalizzazione.
  
✅ Possiamo usare l'interpolazione lineare per portare i valori in un intervallo finito (ad esempio da -20 a 20), e poi convertire i numeri in interi arrotondandoli. Questo ci dà un controllo un po' minore sulla dimensione dello stato, specialmente se non conosciamo gli intervalli esatti dei valori in input. Per esempio, nel nostro caso 2 su 4 valori non hanno limiti superiori o inferiori, il che può risultare in un numero infinito di stati.

Nel nostro esempio, useremo il secondo approccio. Come noterai più avanti, nonostante i limiti superiori/inferiori indefiniti, questi valori raramente prendono valori al di fuori di certi intervalli finiti, quindi quegli stati con valori estremi saranno molto rari.

1. Ecco la funzione che prende l'osservazione dal nostro modello e produce una tupla di 4 valori interi: (blocco di codice 6)

    ```python
    def discretize(x):
        return tuple((x/np.array([0.25, 0.25, 0.01, 0.1])).astype(np.int))
    ```

1. Esploriamo anche un altro metodo di discretizzazione usando i bins: (blocco di codice 7)

    ```python
    def create_bins(i,num):
        return np.arange(num+1)*(i[1]-i[0])/num+i[0]
    
    print("Sample bins for interval (-5,5) with 10 bins\n",create_bins((-5,5),10))
    
    ints = [(-5,5),(-2,2),(-0.5,0.5),(-2,2)] # intervalli di valori per ogni parametro
    nbins = [20,20,10,10] # numero di intervalli per ogni parametro
    bins = [create_bins(ints[i],nbins[i]) for i in range(4)]
    
    def discretize_bins(x):
        return tuple(np.digitize(x[i],bins[i]) for i in range(4))
    ```

1. Ora eseguiamo una breve simulazione e osserviamo quei valori discreti dell'ambiente. Sentiti libero di provare sia `discretize` che `discretize_bins` e vedere se c'è differenza.

    ✅ discretize_bins restituisce il numero del bin, che è basato su zero. Quindi per valori della variabile di input intorno a 0 restituisce il numero dal centro dell'intervallo (10). In discretize, non abbiamo considerato l'intervallo dei valori di output, permettendo che siano negativi, quindi i valori dello stato non sono spostati, e 0 corrisponde a 0. (blocco di codice 8)

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

    ✅ Decommenta la riga che inizia con env.render se vuoi vedere come l'ambiente esegue. Altrimenti puoi eseguirlo in background, che è più veloce. Useremo questa esecuzione "invisibile" durante il nostro processo di Q-Learning.

## La struttura della Q-Table

Nella nostra lezione precedente, lo stato era una semplice coppia di numeri da 0 a 8, quindi era comodo rappresentare la Q-Table con un tensore numpy di forma 8x8x2. Se usiamo la discretizzazione a bins, la dimensione del nostro vettore stato è nota, quindi possiamo usare lo stesso approccio e rappresentare lo stato con un array di forma 20x20x10x10x2 (qui 2 è la dimensione dello spazio azioni, e le prime dimensioni corrispondono al numero di bins selezionati per ciascun parametro nello spazio di osservazione).

Tuttavia, a volte le dimensioni precise dello spazio di osservazione non sono note. Nel caso della funzione `discretize`, non possiamo mai essere sicuri che il nostro stato rimanga entro certi limiti, perché alcuni dei valori originali non sono vincolati. Quindi useremo un approccio leggermente diverso e rappresenteremo la Q-Table con un dizionario.

1. Usa la coppia *(stato, azione)* come chiave del dizionario, e il valore corrisponderà a una voce della Q-Table. (blocco di codice 9)

    ```python
    Q = {}
    actions = (0,1)
    
    def qvalues(state):
        return [Q.get((state,a),0) for a in actions]
    ```

    Qui definiamo anche una funzione `qvalues()`, che restituisce una lista di valori della Q-Table per uno stato dato, corrispondenti a tutte le azioni possibili. Se la voce non è presente nella Q-Table, ritorneremo 0 come valore predefinito.

## Iniziamo Q-Learning

Ora siamo pronti per insegnare a Peter a mantenere l'equilibrio!

1. Prima, impostiamo alcuni iperparametri: (blocco di codice 10)

    ```python
    # iperparametri
    alpha = 0.3
    gamma = 0.9
    epsilon = 0.90
    ```

    Qui, `alpha` è il **tasso di apprendimento** che definisce in quale misura dobbiamo aggiornare i valori attuali della Q-Table a ogni passo. Nella lezione precedente abbiamo iniziato con 1 e poi diminuito `alpha` durante l'addestramento. In questo esempio lo manterremo costante per semplicità, e potrai sperimentare con la regolazione di `alpha` più tardi.

    `gamma` è il **fattore di sconto** che mostra in quale misura dobbiamo dare priorità alla ricompensa futura rispetto a quella attuale.

    `epsilon` è il **fattore esplorazione/sfruttamento** che determina se preferiamo l'esplorazione allo sfruttamento o viceversa. Nel nostro algoritmo, nel `epsilon` percento dei casi selezioneremo la prossima azione in base ai valori della Q-Table, e nel restante selezioneremo un'azione casuale. Questo ci permetterà di esplorare aree dello spazio di ricerca mai viste prima.

    ✅ In termini di bilanciamento - scegliere un'azione casuale (esplorazione) corrisponderebbe a un colpo casuale nella direzione sbagliata, e il palo dovrà imparare a recuperare l'equilibrio da quegli "errori".

### Migliorare l'algoritmo

Possiamo anche fare due miglioramenti al nostro algoritmo dalla lezione precedente:

- **Calcolare la ricompensa cumulativa media**, su un numero di simulazioni. Stamperemo i progressi ogni 5000 iterazioni, e faremo la media della ricompensa cumulativa in quel periodo. Significa che se otteniamo più di 195 punti possiamo considerare il problema risolto, con qualità anche superiore a quella richiesta.
  
- **Calcolare il massimo risultato medio cumulativo**, `Qmax`, e memorizzeremo la Q-Table corrispondente a quel risultato. Quando alleniamo, noterai che a volte il risultato medio cumulativo inizia a diminuire, e vogliamo mantenere i valori della Q-Table corrispondenti al miglior modello osservato durante l'addestramento.

1. Raccogli tutte le ricompense cumulative a ogni simulazione nel vettore `rewards` per eventuali grafici futuri. (blocco di codice  11)

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
        # == esegui la simulazione ==
        while not done:
            s = discretize(obs)
            if random.random()<epsilon:
                # sfruttamento - scegli l'azione secondo le probabilità della tabella Q
                v = probs(np.array(qvalues(s)))
                a = random.choices(actions,weights=v)[0]
            else:
                # esplorazione - scegli casualmente l'azione
                a = np.random.randint(env.action_space.n)
    
            obs, rew, done, info = env.step(a)
            cum_reward+=rew
            ns = discretize(obs)
            Q[(s,a)] = (1 - alpha) * Q.get((s,a),0) + alpha * (rew + gamma * max(qvalues(ns)))
        cum_rewards.append(cum_reward)
        rewards.append(cum_reward)
        # == Periodicamente stampa i risultati e calcola la ricompensa media ==
        if epoch%5000==0:
            print(f"{epoch}: {np.average(cum_rewards)}, alpha={alpha}, epsilon={epsilon}")
            if np.average(cum_rewards) > Qmax:
                Qmax = np.average(cum_rewards)
                Qbest = Q
            cum_rewards=[]
    ```

Ciò che potresti notare da questi risultati:

- **Vicino al nostro obiettivo**. Siamo molto vicini a raggiungere l'obiettivo di ottenere 195 ricompense cumulative su 100+ esecuzioni consecutive della simulazione, o potremmo averlo effettivamente raggiunto! Anche se otteniamo numeri più piccoli, non lo sappiamo ancora, perché facciamo la media su 5000 esecuzioni e solo 100 sono richieste nel criterio formale.
  
- **La ricompensa inizia a diminuire**. A volte la ricompensa inizia a diminuire, il che significa che possiamo "distruggere" i valori già appresi nella Q-Table con quelli che peggiorano la situazione.

Questa osservazione è più visibile se tracciamo i progressi dell'addestramento.

## Grafico del progresso nell'addestramento

Durante l'addestramento, abbiamo raccolto il valore della ricompensa cumulativa a ogni iterazione nel vettore `rewards`. Ecco come appare se lo tracciamo rispetto al numero iterazione:

```python
plt.plot(rewards)
```

![progresso grezzo](../../../../translated_images/it/train_progress_raw.2adfdf2daea09c59.webp)

Da questo grafico non è possibile trarre conclusioni, perché a causa della natura stocastica del processo di addestramento la durata delle sessioni varia molto. Per dare più senso a questo grafico, possiamo calcolare la **media mobile** su una serie di esperimenti, diciamo 100. Questo può essere fatto comodamente usando `np.convolve`: (blocco di codice 12)

```python
def running_average(x,window):
    return np.convolve(x,np.ones(window)/window,mode='valid')

plt.plot(running_average(rewards,100))
```

![progresso nell'addestramento](../../../../translated_images/it/train_progress_runav.c71694a8fa9ab359.webp)

## Variazione degli iperparametri

Per rendere l'apprendimento più stabile, ha senso regolare alcuni dei nostri iperparametri durante l'addestramento. In particolare:

- **Per il tasso di apprendimento**, `alpha`, possiamo iniziare con valori vicini a 1 e poi diminuire gradualmente il parametro. Con il tempo otterremo buone probabilità nella Q-Table, quindi dovremmo aggiustarle leggermente, non sovrascriverle completamente con nuovi valori.

- **Aumentare epsilon**. Potremmo voler aumentare lentamente `epsilon`, per esplorare di meno e sfruttare di più. Probabilmente conviene partire da un valore basso di `epsilon` e salire quasi a 1.

> **Compito 1**: Gioca con i valori degli iperparametri e vedi se riesci a ottenere una ricompensa cumulativa più alta. Riuscirai a superare 195?


> **Compito 2**: Per risolvere formalmente il problema, devi ottenere una ricompensa media di 195 su 100 esecuzioni consecutive. Misura questo durante l’addestramento e assicurati di aver risolto formalmente il problema!

## Vedere il risultato in azione

Sarebbe interessante vedere effettivamente come si comporta il modello addestrato. Eseguiamo la simulazione e seguiamo la stessa strategia di selezione delle azioni usata durante l’addestramento, campionando secondo la distribuzione di probabilità nella Q-Table: (blocco codice 13)

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

Dovresti vedere qualcosa di simile a questo:

![un carrello in equilibrio su un palo](../../../../8-Reinforcement/2-Gym/images/cartpole-balance.gif)

---

## 🚀Sfida

> **Compito 3**: Qui abbiamo usato la copia finale della Q-Table, che potrebbe non essere la migliore. Ricorda che abbiamo salvato la Q-Table con le migliori prestazioni nella variabile `Qbest`! Prova lo stesso esempio con la Q-Table con le migliori prestazioni copiando `Qbest` su `Q` e verifica se noti la differenza.

> **Compito 4**: Qui non abbiamo selezionato l’azione migliore a ogni passo, ma abbiamo campionato secondo la corrispondente distribuzione di probabilità. Ha senso selezionare sempre l’azione migliore, con il valore più alto della Q-Table? Questo può essere fatto usando la funzione `np.argmax` per trovare il numero dell’azione corrispondente al valore più alto nella Q-Table. Implementa questa strategia e verifica se migliora l’equilibrio.

## [Quiz post-lezione](https://ff-quizzes.netlify.app/en/ml/)

## Assegnazione
[Addestra una Mountain Car](assignment.md)

## Conclusione

Ora abbiamo imparato come addestrare agenti a ottenere buoni risultati fornendo loro una funzione di ricompensa che definisce lo stato desiderato del gioco, e dando loro l’opportunità di esplorare in modo intelligente lo spazio di ricerca. Abbiamo applicato con successo l’algoritmo Q-Learning in ambienti discreti e continui, ma con azioni discrete.

È importante studiare anche situazioni in cui lo stato dell’azione è continuo e quando lo spazio di osservazione è molto più complesso, come l’immagine dello schermo di un gioco Atari. In questi problemi spesso è necessario usare tecniche di apprendimento automatico più potenti, come le reti neurali, per ottenere buoni risultati. Questi argomenti più avanzati sono oggetto del nostro prossimo corso avanzato di AI.

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Disclaimer**:
Questo documento è stato tradotto utilizzando il servizio di traduzione AI [Co-op Translator](https://github.com/Azure/co-op-translator). Sebbene ci impegniamo per garantire la precisione, si prega di notare che le traduzioni automatizzate possono contenere errori o imprecisioni. Il documento originale nella sua lingua nativa deve essere considerato la fonte autorevole. Per informazioni critiche, si raccomanda una traduzione professionale effettuata da un essere umano. Non siamo responsabili per eventuali malintesi o interpretazioni errate derivanti dall’uso di questa traduzione.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->