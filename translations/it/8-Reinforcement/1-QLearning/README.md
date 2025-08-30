<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "0ffe994d1cc881bdeb49226a064116e5",
  "translation_date": "2025-08-29T22:08:57+00:00",
  "source_file": "8-Reinforcement/1-QLearning/README.md",
  "language_code": "it"
}
-->
# Introduzione al Reinforcement Learning e al Q-Learning

![Riassunto del reinforcement learning in machine learning in uno sketchnote](../../../../translated_images/ml-reinforcement.94024374d63348dbb3571c343ca7ddabef72adac0b8086d47164b769ba3a8a1d.it.png)
> Sketchnote di [Tomomi Imura](https://www.twitter.com/girlie_mac)

Il reinforcement learning coinvolge tre concetti fondamentali: l'agente, alcuni stati e un insieme di azioni per stato. Eseguendo un'azione in uno stato specifico, l'agente riceve una ricompensa. Immagina di nuovo il videogioco Super Mario. Tu sei Mario, ti trovi in un livello di gioco, vicino al bordo di un precipizio. Sopra di te c'è una moneta. Essere Mario, in un livello di gioco, in una posizione specifica... quello è il tuo stato. Muoversi di un passo a destra (un'azione) ti porterebbe oltre il bordo, e ciò ti darebbe un punteggio numerico basso. Tuttavia, premendo il pulsante di salto, guadagneresti un punto e rimarresti vivo. Questo è un risultato positivo e dovrebbe assegnarti un punteggio numerico positivo.

Utilizzando il reinforcement learning e un simulatore (il gioco), puoi imparare a giocare per massimizzare la ricompensa, che consiste nel rimanere vivo e accumulare il maggior numero di punti possibile.

[![Introduzione al Reinforcement Learning](https://img.youtube.com/vi/lDq_en8RNOo/0.jpg)](https://www.youtube.com/watch?v=lDq_en8RNOo)

> 🎥 Clicca sull'immagine sopra per ascoltare Dmitry parlare del Reinforcement Learning

## [Quiz pre-lezione](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/45/)

## Prerequisiti e Configurazione

In questa lezione, sperimenteremo del codice in Python. Dovresti essere in grado di eseguire il codice del Jupyter Notebook di questa lezione, sia sul tuo computer che in cloud.

Puoi aprire [il notebook della lezione](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/notebook.ipynb) e seguire questa lezione per costruire.

> **Nota:** Se stai aprendo questo codice dal cloud, devi anche recuperare il file [`rlboard.py`](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/rlboard.py), che viene utilizzato nel codice del notebook. Aggiungilo alla stessa directory del notebook.

## Introduzione

In questa lezione, esploreremo il mondo di **[Pierino e il Lupo](https://it.wikipedia.org/wiki/Pierino_e_il_lupo)**, ispirato a una fiaba musicale di un compositore russo, [Sergei Prokofiev](https://it.wikipedia.org/wiki/Sergej_Prokof%27ev). Utilizzeremo il **Reinforcement Learning** per permettere a Pierino di esplorare il suo ambiente, raccogliere mele gustose ed evitare di incontrare il lupo.

Il **Reinforcement Learning** (RL) è una tecnica di apprendimento che ci consente di apprendere un comportamento ottimale di un **agente** in un determinato **ambiente** eseguendo molti esperimenti. Un agente in questo ambiente dovrebbe avere un **obiettivo**, definito da una **funzione di ricompensa**.

## L'ambiente

Per semplicità, consideriamo il mondo di Pierino come una scacchiera di dimensioni `larghezza` x `altezza`, come questa:

![Ambiente di Pierino](../../../../translated_images/environment.40ba3cb66256c93fa7e92f6f7214e1d1f588aafa97d266c11d108c5c5d101b6c.it.png)

Ogni cella di questa scacchiera può essere:

* **terra**, su cui Pierino e altre creature possono camminare.
* **acqua**, su cui ovviamente non si può camminare.
* un **albero** o **erba**, un luogo dove puoi riposarti.
* una **mela**, che rappresenta qualcosa che Pierino sarebbe felice di trovare per nutrirsi.
* un **lupo**, che è pericoloso e dovrebbe essere evitato.

Esiste un modulo Python separato, [`rlboard.py`](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/rlboard.py), che contiene il codice per lavorare con questo ambiente. Poiché questo codice non è importante per comprendere i nostri concetti, importeremo il modulo e lo utilizzeremo per creare la scacchiera di esempio (blocco di codice 1):

```python
from rlboard import *

width, height = 8,8
m = Board(width,height)
m.randomize(seed=13)
m.plot()
```

Questo codice dovrebbe stampare un'immagine dell'ambiente simile a quella sopra.

## Azioni e politica

Nel nostro esempio, l'obiettivo di Pierino sarebbe trovare una mela, evitando il lupo e altri ostacoli. Per fare ciò, può essenzialmente camminare fino a trovare una mela.

Pertanto, in qualsiasi posizione, può scegliere tra una delle seguenti azioni: su, giù, sinistra e destra.

Definiremo queste azioni come un dizionario e le mapperemo a coppie di modifiche delle coordinate corrispondenti. Ad esempio, muoversi a destra (`R`) corrisponderebbe a una coppia `(1,0)`. (blocco di codice 2):

```python
actions = { "U" : (0,-1), "D" : (0,1), "L" : (-1,0), "R" : (1,0) }
action_idx = { a : i for i,a in enumerate(actions.keys()) }
```

Riassumendo, la strategia e l'obiettivo di questo scenario sono i seguenti:

- **La strategia**, del nostro agente (Pierino) è definita da una cosiddetta **politica**. Una politica è una funzione che restituisce l'azione in uno stato dato. Nel nostro caso, lo stato del problema è rappresentato dalla scacchiera, inclusa la posizione attuale del giocatore.

- **L'obiettivo**, del reinforcement learning è imparare alla fine una buona politica che ci permetta di risolvere il problema in modo efficiente. Tuttavia, come base di partenza, consideriamo la politica più semplice chiamata **camminata casuale**.

## Camminata casuale

Per prima cosa risolviamo il nostro problema implementando una strategia di camminata casuale. Con la camminata casuale, sceglieremo casualmente la prossima azione tra quelle consentite, fino a raggiungere la mela (blocco di codice 3).

1. Implementa la camminata casuale con il codice seguente:

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

    La chiamata a `walk` dovrebbe restituire la lunghezza del percorso corrispondente, che può variare da un'esecuzione all'altra.

1. Esegui l'esperimento di camminata un certo numero di volte (ad esempio, 100) e stampa le statistiche risultanti (blocco di codice 4):

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

    Nota che la lunghezza media di un percorso è di circa 30-40 passi, che è piuttosto elevata, considerando che la distanza media dalla mela più vicina è di circa 5-6 passi.

    Puoi anche vedere come si muove Pierino durante la camminata casuale:

    ![Camminata casuale di Pierino](../../../../8-Reinforcement/1-QLearning/images/random_walk.gif)

## Funzione di ricompensa

Per rendere la nostra politica più intelligente, dobbiamo capire quali mosse sono "migliori" rispetto ad altre. Per fare ciò, dobbiamo definire il nostro obiettivo.

L'obiettivo può essere definito in termini di una **funzione di ricompensa**, che restituirà un valore di punteggio per ogni stato. Più alto è il numero, migliore è la funzione di ricompensa. (blocco di codice 5)

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

Una cosa interessante delle funzioni di ricompensa è che nella maggior parte dei casi, *riceviamo una ricompensa significativa solo alla fine del gioco*. Ciò significa che il nostro algoritmo dovrebbe in qualche modo ricordare i "passi buoni" che portano a una ricompensa positiva alla fine e aumentarne l'importanza. Allo stesso modo, tutte le mosse che portano a risultati negativi dovrebbero essere scoraggiate.

## Q-Learning

L'algoritmo che discuteremo qui si chiama **Q-Learning**. In questo algoritmo, la politica è definita da una funzione (o una struttura dati) chiamata **Q-Table**. Essa registra la "bontà" di ciascuna delle azioni in uno stato dato.

Si chiama Q-Table perché è spesso conveniente rappresentarla come una tabella o un array multidimensionale. Poiché la nostra scacchiera ha dimensioni `larghezza` x `altezza`, possiamo rappresentare la Q-Table utilizzando un array numpy con forma `larghezza` x `altezza` x `len(actions)`: (blocco di codice 6)

```python
Q = np.ones((width,height,len(actions)),dtype=np.float)*1.0/len(actions)
```

Nota che inizializziamo tutti i valori della Q-Table con un valore uguale, nel nostro caso - 0.25. Questo corrisponde alla politica di "camminata casuale", poiché tutte le mosse in ogni stato sono ugualmente buone. Possiamo passare la Q-Table alla funzione `plot` per visualizzare la tabella sulla scacchiera: `m.plot(Q)`.

![Ambiente di Pierino](../../../../translated_images/env_init.04e8f26d2d60089e128f21d22e5fef57d580e559f0d5937b06c689e5e7cdd438.it.png)

Al centro di ogni cella c'è una "freccia" che indica la direzione preferita di movimento. Poiché tutte le direzioni sono uguali, viene visualizzato un punto.

Ora dobbiamo eseguire la simulazione, esplorare il nostro ambiente e apprendere una migliore distribuzione dei valori della Q-Table, che ci permetterà di trovare il percorso verso la mela molto più velocemente.

## Essenza del Q-Learning: Equazione di Bellman

Una volta che iniziamo a muoverci, ogni azione avrà una ricompensa corrispondente, cioè teoricamente possiamo selezionare la prossima azione basandoci sulla ricompensa immediata più alta. Tuttavia, nella maggior parte degli stati, la mossa non raggiungerà il nostro obiettivo di trovare la mela, e quindi non possiamo decidere immediatamente quale direzione sia migliore.

> Ricorda che non è il risultato immediato che conta, ma piuttosto il risultato finale, che otterremo alla fine della simulazione.

Per tenere conto di questa ricompensa ritardata, dobbiamo utilizzare i principi della **[programmazione dinamica](https://it.wikipedia.org/wiki/Programmazione_dinamica)**, che ci permettono di pensare al nostro problema in modo ricorsivo.

Supponiamo di trovarci ora nello stato *s*, e vogliamo passare al prossimo stato *s'*. Facendo ciò, riceveremo la ricompensa immediata *r(s,a)*, definita dalla funzione di ricompensa, più una ricompensa futura. Se supponiamo che la nostra Q-Table rifletta correttamente l'"attrattività" di ciascuna azione, allora nello stato *s'* sceglieremo un'azione *a* che corrisponde al valore massimo di *Q(s',a')*. Pertanto, la migliore ricompensa futura possibile che potremmo ottenere nello stato *s* sarà definita come `max`

## Verifica della politica

Poiché la Q-Table elenca l'"attrattività" di ogni azione in ogni stato, è abbastanza semplice utilizzarla per definire la navigazione efficiente nel nostro mondo. Nel caso più semplice, possiamo selezionare l'azione corrispondente al valore più alto della Q-Table: (blocco di codice 9)

```python
def qpolicy_strict(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = list(actions)[np.argmax(v)]
        return a

walk(m,qpolicy_strict)
```

> Se provi il codice sopra più volte, potresti notare che a volte "si blocca" e devi premere il pulsante STOP nel notebook per interromperlo. Questo accade perché potrebbero esserci situazioni in cui due stati "puntano" l'uno all'altro in termini di valore Q ottimale, nel qual caso l'agente finisce per muoversi tra quegli stati indefinitamente.

## 🚀Sfida

> **Compito 1:** Modifica la funzione `walk` per limitare la lunghezza massima del percorso a un certo numero di passi (ad esempio, 100), e osserva il codice sopra restituire questo valore di tanto in tanto.

> **Compito 2:** Modifica la funzione `walk` in modo che non torni nei luoghi in cui è già stato in precedenza. Questo impedirà a `walk` di entrare in un ciclo, tuttavia, l'agente potrebbe comunque finire "intrappolato" in una posizione da cui non riesce a uscire.

## Navigazione

Una politica di navigazione migliore sarebbe quella che abbiamo utilizzato durante l'addestramento, che combina sfruttamento ed esplorazione. In questa politica, selezioneremo ogni azione con una certa probabilità, proporzionale ai valori nella Q-Table. Questa strategia potrebbe comunque portare l'agente a tornare in una posizione già esplorata, ma, come puoi vedere dal codice qui sotto, risulta in un percorso medio molto breve verso la posizione desiderata (ricorda che `print_statistics` esegue la simulazione 100 volte): (blocco di codice 10)

```python
def qpolicy(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = random.choices(list(actions),weights=v)[0]
        return a

print_statistics(qpolicy)
```

Dopo aver eseguito questo codice, dovresti ottenere una lunghezza media del percorso molto più breve rispetto a prima, nell'intervallo di 3-6.

## Indagare il processo di apprendimento

Come abbiamo menzionato, il processo di apprendimento è un equilibrio tra esplorazione e sfruttamento delle conoscenze acquisite sulla struttura dello spazio del problema. Abbiamo visto che i risultati dell'apprendimento (la capacità di aiutare un agente a trovare un percorso breve verso l'obiettivo) sono migliorati, ma è anche interessante osservare come si comporta la lunghezza media del percorso durante il processo di apprendimento:

## Le osservazioni possono essere riassunte come:

- **La lunghezza media del percorso aumenta**. Quello che vediamo qui è che inizialmente la lunghezza media del percorso aumenta. Questo probabilmente è dovuto al fatto che, quando non sappiamo nulla dell'ambiente, è probabile che rimaniamo intrappolati in stati sfavorevoli, come acqua o lupo. Man mano che apprendiamo di più e iniziamo a utilizzare queste conoscenze, possiamo esplorare l'ambiente più a lungo, ma ancora non sappiamo molto bene dove si trovano le mele.

- **La lunghezza del percorso diminuisce, man mano che apprendiamo di più**. Una volta che apprendiamo abbastanza, diventa più facile per l'agente raggiungere l'obiettivo, e la lunghezza del percorso inizia a diminuire. Tuttavia, siamo ancora aperti all'esplorazione, quindi spesso ci allontaniamo dal percorso migliore ed esploriamo nuove opzioni, rendendo il percorso più lungo del necessario.

- **La lunghezza aumenta improvvisamente**. Quello che osserviamo anche in questo grafico è che a un certo punto la lunghezza aumenta improvvisamente. Questo indica la natura stocastica del processo, e che possiamo a un certo punto "rovinare" i coefficienti della Q-Table sovrascrivendoli con nuovi valori. Questo idealmente dovrebbe essere minimizzato diminuendo il tasso di apprendimento (ad esempio, verso la fine dell'addestramento, regoliamo i valori della Q-Table solo di una piccola quantità).

In generale, è importante ricordare che il successo e la qualità del processo di apprendimento dipendono significativamente dai parametri, come il tasso di apprendimento, la decadenza del tasso di apprendimento e il fattore di sconto. Questi sono spesso chiamati **iperparametri**, per distinguerli dai **parametri**, che ottimizziamo durante l'addestramento (ad esempio, i coefficienti della Q-Table). Il processo di trovare i migliori valori degli iperparametri si chiama **ottimizzazione degli iperparametri**, e merita un argomento a parte.

## [Quiz post-lezione](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/46/)

## Compito 
[Un mondo più realistico](assignment.md)

---

**Disclaimer**:  
Questo documento è stato tradotto utilizzando il servizio di traduzione automatica [Co-op Translator](https://github.com/Azure/co-op-translator). Sebbene ci impegniamo per garantire l'accuratezza, si prega di notare che le traduzioni automatiche possono contenere errori o imprecisioni. Il documento originale nella sua lingua nativa dovrebbe essere considerato la fonte autorevole. Per informazioni critiche, si raccomanda una traduzione professionale effettuata da un esperto umano. Non siamo responsabili per eventuali fraintendimenti o interpretazioni errate derivanti dall'uso di questa traduzione.