# CartPole Skating

Il problema risolto nella lezione precedente potrebbe sembrare un problema giocattolo, non propriamente applicabile a scenari di vita reale. Questo non è il caso, perché anche molti problemi del mondo reale condividono questo scenario, incluso Scacchi o Go. Sono simili, perché anche in quei casi si ha una tavolo di gioco con regole date e uno **stato discreto**.

## [Quiz pre-lezione](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/47/?loc=it)

## Introduzione

In questa lezione si applicheranno gli stessi principi di Q-Learning ad un problema con **stato continuo**, cioè uno stato dato da uno o più numeri reali. Ci si occuperà del seguente problema:

> **Problema**: se Pierino vuole scappare dal lupo, deve essere in grado di muoversi più velocemente. Si vedrà come Pierino può imparare a pattinare, in particolare, a mantenere l'equilibrio, utilizzando Q-Learning.

![La grande fuga!](../images/escape.png)

> Pierino e i suoi amici diventano creativi per sfuggire al lupo! Immagine di [Jen Looper](https://twitter.com/jenlooper)

Si userà una versione semplificata del bilanciamento noto come **problema** CartPole. Nel mondo cartpole, c'è un cursore orizzontale che può spostarsi a sinistra o a destra, e l'obiettivo è bilanciare un palo verticale sopra il cursore.

<img alt="un cartpole" src="../images/cartpole.png" width="200"/>

## Prerequisiti

In questa lezione si utilizzerà una libreria chiamata **OpenAI Gym per** simulare **ambienti** diversi. Si può eseguire il codice di questa lezione localmente (es. da Visual Studio Code), nel qual caso la simulazione si aprirà in una nuova finestra. Quando si esegue il codice online, potrebbe essere necessario apportare alcune modifiche, come descritto [qui](https://towardsdatascience.com/rendering-openai-gym-envs-on-binder-and-google-colab-536f99391cc7).

## OpenAI Gym

Nella lezione precedente, le regole del gioco e lo stato sono state date dalla classe `Board` sviluppata nel codice. Qui si utilizzerà uno speciale **ambiente di simulazione**, che simulerà la fisica dietro il palo di bilanciamento. Uno degli ambienti di simulazione più popolari per addestrare gli algoritmi di reinforcement learning è chiamato a [Gym](https://gym.openai.com/), mantenuto da [OpenAI](https://openai.com/). Con questo gym è possibile creare **ambienti** diversi da una simulazione cartpole a giochi Atari.

> **Nota**: si possono vedere altri ambienti disponibili da OpenAI Gym [qui](https://gym.openai.com/envs/#classic_control).

Innanzitutto, si installa gym e si importano le librerie richieste (blocco di codice 1):

```python
import sys
!{sys.executable} -m pip install gym

import gym
import matplotlib.pyplot as plt
import numpy as np
import random
```

## Esercizio: inizializzare un ambiente cartpole

Per lavorare con un problema di bilanciamento del cartpole, è necessario inizializzare l'ambiente corrispondente. Ad ogni ambiente è associato uno:

- **Spazio di osservazione** che definisce la struttura delle informazioni ricevute dall'ambiente. Per il problema del cartpole, si riceve la posizione del palo, la velocità e alcuni altri valori.

- **Spazio di azione** che definisce le possibili azioni. In questo caso lo spazio delle azioni è discreto e consiste di due azioni: **sinistra** e **destra**. (blocco di codice 2)

1. Per inizializzare, digitare il seguente codice:

   ```python
   env = gym.make("CartPole-v1")
   print(env.action_space)
   print(env.observation_space)
   print(env.action_space.sample())
   ```

Per vedere come funziona l'ambiente, si esegue una breve simulazione di 100 passaggi. Ad ogni passaggio, si fornisce una delle azioni da intraprendere: in questa simulazione si seleziona casualmente un'azione da `action_space`.

1. Eseguire il codice qui sotto e guardare a cosa porta.

   ✅ Ricordare che è preferibile eseguire questo codice sull'installazione locale di Python! (blocco di codice 3)

   ```python
   env.reset()

   for i in range(100):
      env.render()
      env.step(env.action_space.sample())
   env.close()
   ```

   Si dovrebbe vedere qualcosa di simile a questa immagine:

   ![carrello non in equilibrio](../images/cartpole-nobalance.gif)

1. Durante la simulazione, sono necessarie osservazioni per decidere come agire. Infatti, la funzione step restituisce le osservazioni correnti, una funzione di ricompensa e il flag done che indica se ha senso continuare o meno la simulazione: (blocco di codice 4)

   ```python
   env.reset()

   done = False
   while not done:
      env.render()
      obs, rew, done, info = env.step(env.action_space.sample())
      print(f"{obs} -> {rew}")
   env.close()
   ```

   Si finirà per vedere qualcosa di simile nell'output del notebook:

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
   - Tasso di rotazione del palo

1. Ottenere il valore minimo e massimo di quei numeri: (blocco di codice 5)

   ```python
   print(env.observation_space.low)
   print(env.observation_space.high)
   ```

   Si potrebbe anche notare che il valore della ricompensa in ogni fase della simulazione è sempre 1. Questo perché l'obiettivo è sopravvivere il più a lungo possibile, ovvero mantenere il palo in una posizione ragionevolmente verticale per il periodo di tempo più lungo.

   ✅ Infatti la simulazione CartPole si considera risolta se si riesce a ottenere la ricompensa media di 195 su 100 prove consecutive.

## Discretizzazione dello stato

In Q-Learning, occorre costruire una Q-Table che definisce cosa fare in ogni stato. Per poterlo fare, è necessario che lo stato sia **discreto**, più precisamente, dovrebbe contenere un numero finito di valori discreti. Quindi, serve un qualche modo per **discretizzare** le osservazioni, mappandole su un insieme finito di stati.

Ci sono alcuni modi in cui si può fare:

- **Dividere in contenitori**. Se è noto l'intervallo di un certo valore, si può dividere questo intervallo in un numero di **bin** (contenitori) e quindi sostituire il valore con il numero di contenitore a cui appartiene. Questo può essere fatto usando il metodo di numpy [`digitize`](https://numpy.org/doc/stable/reference/generated/numpy.digitize.html). In questo caso, si conoscerà con precisione la dimensione dello stato, perché dipenderà dal numero di contenitori selezionati per la digitalizzazione.

✅ Si può usare l'interpolazione lineare per portare i valori a qualche intervallo finito (ad esempio, da -20 a 20), e poi convertire i numeri in interi arrotondandoli. Questo dà un po' meno controllo sulla dimensione dello stato, specialmente se non sono noti gli intervalli esatti dei valori di input. Ad esempio, in questo caso 2 valori su 4 non hanno limiti superiore/inferiore sui loro valori, il che può comportare un numero infinito di stati.

In questo esempio, si andrà con il secondo approccio. Come si potrà notare in seguito, nonostante i limiti superiore/inferiore non definiti, quei valori raramente assumono valori al di fuori di determinati intervalli finiti, quindi quegli stati con valori estremi saranno molto rari.

1. Ecco la funzione che prenderà l'osservazione dal modello e produrrà una tupla di 4 valori interi: (blocco di codice 6)

   ```python
   def discretize(x):
       return tuple((x/np.array([0.25, 0.25, 0.01, 0.1])).astype(np.int))
   ```

1. Si esplora anche un altro metodo di discretizzazione utilizzando i contenitori: (blocco di codice 7)

   ```python
   def create_bins(i,num):
       return np.arange(num+1)*(i[1]-i[0])/num+i[0]

   print("Sample bins for interval (-5,5) with 10 bins\n",create_bins((-5,5),10))

   ints = [(-5,5),(-2,2),(-0.5,0.5),(-2,2)] # Intervallo di valori per ogni parametro
   nbins = [20,20,10,10] # numero di contenitori per ogni parametro
   bins = [create_bins(ints[i],nbins[i]) for i in range(4)]

   def discretize_bins(x):
       return tuple(np.digitize(x[i],bins[i]) for i in range(4))
   ```

1. Si esegue ora una breve simulazione e si osservano quei valori discreti dell'ambiente. Si può provare `discretize` e `discretize_bins` e vedere se c'è una differenza.

   ✅ discretize_bins restituisce il numero del contenitore, che è in base 0. Quindi per i valori della variabile di input intorno a 0 restituisce il numero dalla metà dell'intervallo (10). In discretize, non interessava l'intervallo dei valori di uscita, consentendo loro di essere negativi, quindi i valori di stato non vengono spostati e 0 corrisponde a 0. (blocco di codice 8)

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

   ✅ Decommentare la riga che inizia con env.render se si vuole vedere come viene eseguito l'ambiente. Altrimenti si può eseguirlo in background, che è più veloce. Si userà questa esecuzione "invisibile" durante il processo di Q-Learning.

## La struttura di Q-Table

Nella lezione precedente, lo stato era una semplice coppia di numeri da 0 a 8, e quindi era conveniente rappresentare Q-Table con un tensore numpy con una forma di 8x8x2. Se si usa la discretizzazione dei contenitori, è nota anche la dimensione del vettore di stato, quindi si può usare lo stesso approccio e rappresentare lo stato con un array di forma 20x20x10x10x2 (qui 2 è la dimensione dello spazio delle azioni e le prime dimensioni corrispondono al numero di contenitori che si è scelto di utilizzare per ciascuno dei parametri nello spazio di osservazione).

Tuttavia, a volte non sono note dimensioni precise dello spazio di osservazione. Nel caso della funzione `discretize`, si potrebbe non essere mai sicuri che lo stato rimanga entro certi limiti, perché alcuni dei valori originali non sono vincolati. Pertanto, si utilizzerà un approccio leggermente diverso e si rappresenterà Q-Table con un dizionario.

1. Si usa la coppia *(state, action)* come chiave del dizionario e il valore corrisponderà al valore della voce Q-Table. (blocco di codice 9)

   ```python
   Q = {}
   actions = (0,1)

   def qvalues(state):
       return [Q.get((state,a),0) for a in actions]
   ```

   Qui si definisce anche una funzione `qvalues()`, che restituisce un elenco di valori di Q-Table per un dato stato che corrisponde a tutte le azioni possibili. Se la voce non è presente nella Q-Table, si restituirà 0 come predefinito.

## Far partire Q-Learning

Ora si è pronti per insegnare a Pierino a bilanciare!

1. Per prima cosa, si impostano alcuni iperparametri: (blocco di codice 10)

   ```python
   # iperparametri
   alpha = 0.3
   gamma = 0.9
   epsilon = 0.90
   ```

   Qui, `alfa` è il **tasso di apprendimento** che definisce fino a che punto si dovranno regolare i valori correnti di Q-Table ad ogni passaggio. Nella lezione precedente si è iniziato con 1, quindi si è ridotto `alfa` per abbassare i valori durante l'allenamento. In questo esempio lo si manterrà costante solo per semplicità e si potrà sperimentare con la regolazione dei valori `alfa` in un secondo momento.

   `gamma` è il **fattore di sconto** che mostra fino a che punto si dovrà dare la priorità alla ricompensa futura rispetto alla ricompensa attuale.

   `epsilon` è il **fattore di esplorazione/sfruttamento** che determina se preferire l'esplorazione allo sfruttamento o viceversa. In questo algoritmo, nella percentuale `epsilon` dei casi si selezionerà l'azione successiva in base ai valori della Q-Table e nel restante numero di casi verrà eseguita un'azione casuale. Questo  permetterà di esplorare aree dello spazio di ricerca che non sono mai state viste prima.

   ✅ In termini di bilanciamento - la scelta di un'azione casuale (esplorazione) agirebbe come un pugno casuale nella direzione sbagliata e il palo dovrebbe imparare a recuperare l'equilibrio da quegli "errori"

### Migliorare l'algoritmo

E' possibile anche apportare due miglioramenti all'algoritmo rispetto alla lezione precedente:

- **Calcolare la ricompensa cumulativa media**, su una serie di simulazioni. Si stamperanno i progressi ogni 5000 iterazioni e si farà la media della ricompensa cumulativa in quel periodo di tempo. Significa che se si ottengono più di 195 punti, si può considerare il problema risolto, con una qualità ancora superiore a quella richiesta.

- **Calcolare il risultato cumulativo medio massimo**, `Qmax`, e si memorizzerà la Q-Table corrispondente a quel risultato. Quando si esegue l'allenamento si noterà che a volte il risultato cumulativo medio inizia a diminuire e si vuole mantenere i valori di Q-Table che corrispondono al miglior modello osservato durante l'allenamento.

1. Raccogliere tutte le ricompense cumulative ad ogni simulazione nel vettore `rewards` per ulteriori grafici. (blocco di codice 11)

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
       # == esegue la simulazione ==
       while not done:
           s = discretize(obs)
           if random.random()<epsilon:
               # sfruttamento - sceglie l'azione in accordo alle probabilità di Q-Table
               v = probs(np.array(qvalues(s)))
               a = random.choices(actions,weights=v)[0]
           else:
               # esplorazione - sceglie causalmente l'azione
               a = np.random.randint(env.action_space.n)

           obs, rew, done, info = env.step(a)
           cum_reward+=rew
           ns = discretize(obs)
           Q[(s,a)] = (1 - alpha) * Q.get((s,a),0) + alpha * (rew + gamma * max(qvalues(ns)))
       cum_rewards.append(cum_reward)
       rewards.append(cum_reward)
       # == Stampa periodicamente i risultati a calcola la ricompensa media ==
       if epoch%5000==0:
           print(f"{epoch}: {np.average(cum_rewards)}, alpha={alpha}, epsilon={epsilon}")
           if np.average(cum_rewards) > Qmax:
               Qmax = np.average(cum_rewards)
               Qbest = Q
           cum_rewards=[]
   ```

Cosa si potrebbe notare da questi risultati:

- **Vicino all'obiettivo**. Si è molto vicini al raggiungimento dell'obiettivo di ottenere 195 ricompense cumulative in oltre 100 esecuzioni consecutive della simulazione, o si potrebbe averlo effettivamente raggiunto! Anche se si ottengono numeri più piccoli, non si sa ancora, perché si ha una media di oltre 5000 esecuzioni e nei criteri formali sono richieste solo 100 esecuzioni.

- **La ricompensa inizia a diminuire**. A volte la ricompensa inizia a diminuire, il che significa che si possono "distruggere" i valori già appresi nella Q-Table con quelli che peggiorano la situazione.

Questa osservazione è più chiaramente visibile se si tracciano i progressi dell'allenamento.

## Tracciare i progressi dell'allenamento

Durante l'addestramento, si è raccolto il valore cumulativo della ricompensa a ciascuna delle iterazioni nel vettore delle ricompense `reward` . Ecco come appare quando viene riportato al numero di iterazione:

```python
plt.plot(rewards)
```

![progresso grezzo](../images/train_progress_raw.png)

Da questo grafico non è possibile dire nulla, perché a causa della natura del processo di allenamento stocastico la durata delle sessioni di allenamento varia notevolmente. Per dare più senso a questo grafico, si può calcolare la **media mobile su** una serie di esperimenti, ad esempio 100. Questo può essere fatto comodamente usando `np.convolve` : (blocco di codice 12)

```python
def running_average(x,window):
    return np.convolve(x,np.ones(window)/window,mode='valid')

plt.plot(running_average(rewards,100))
```

![Progressi dell'allenamento](../images/train_progress_runav.png)

## Variare gli iperparametri

Per rendere l'apprendimento più stabile, ha senso regolare alcuni degli iperparametri durante l'allenamento. In particolare:

- **Per il tasso di apprendimento**, `alfa`, si può iniziare con valori vicini a 1 e poi continuare a diminuire il parametro. Con il tempo, si otterranno buoni valori di probabilità nella Q-Table, e quindi si dovranno modificare leggermente e non sovrascrivere completamente con nuovi valori.

- **Aumentare epsilon**. Si potrebbe voler aumentare lentamente `epsilon`, in modo da esplorare di meno e sfruttare di più. Probabilmente ha senso iniziare con un valore inferiore di `epsilone` e salire fino a quasi 1.

> **Compito 1**: giocare con i valori degli iperparametri e vedere se si riesce a ottenere una ricompensa cumulativa più alta. Si stanno superando i 195?

> **Compito 2**: per risolvere formalmente il problema, si devono ottenere 195 ricompense medie in 100 esecuzioni consecutive. Misurare questo durante l'allenamento e assicurarsi di aver risolto formalmente il problema!

## Vedere il risultato in azione

Sarebbe interessante vedere effettivamente come si comporta il modello addestrato. Si esegue la simulazione e si segue la stessa strategia di selezione dell'azione utilizzata durante l'addestramento, campionando secondo la distribuzione di probabilità in Q-Table: (blocco di codice 13)

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

Si dovrebbe vedere qualcosa del genere:

![un cartpole equilibratore](../images/cartpole-balance.gif)

---

## 🚀 Sfida

> **Compito 3**: qui si stava usando la copia finale di Q-Table, che potrebbe non essere la migliore. Ricordare che si è memorizzato la Q-Table con le migliori prestazioni nella variabile `Qbest`! Provare lo stesso esempio con la Q-Table di migliori prestazioni copiando `Qbest` su `Q` e vedere se si nota la differenza.

> **Compito 4**: Qui non si stava selezionando l'azione migliore per ogni passaggio, ma piuttosto campionando con la corrispondente distribuzione di probabilità. Avrebbe più senso selezionare sempre l'azione migliore, con il valore Q-Table più alto? Questo può essere fatto usando la funzione `np.argmax` per trovare il numero dell'azione corrispondente al valore della Q-Table più alto. Implementare questa strategia e vedere se migliora il bilanciamento.

## [Quiz post-lezione](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/48/?loc=it)

## Compito: [addestrare un'auto di montagna](assignment.it.md)

## Conclusione

Ora si è imparato come addestrare gli agenti a ottenere buoni risultati semplicemente fornendo loro una funzione di ricompensa che definisce lo stato desiderato del gioco e dando loro l'opportunità di esplorare in modo intelligente lo spazio di ricerca. E' stato applicato con successo l'algoritmo di Q-Learning nei casi di ambienti discreti e continui, ma con azioni discrete.

È importante studiare anche situazioni in cui anche lo stato di azione è continuo e quando lo spazio di osservazione è molto più complesso, come l'immagine dalla schermata di gioco dell'Atari. In questi problemi spesso è necessario utilizzare tecniche di apprendimento automatico più potenti, come le reti neurali, per ottenere buoni risultati. Questi argomenti più avanzati sono l'oggetto del prossimo corso di intelligenza artificiale più avanzato.
