# Patinação CartPole

O problema que estávamos resolvendo na lição anterior pode parecer um problema de brinquedo, não realmente aplicável a cenários da vida real. Este não é o caso, porque muitos problemas do mundo real também compartilham esse cenário - incluindo jogar xadrez ou go. Eles são semelhantes, porque também temos um tabuleiro com regras definidas e um **estado discreto**.

## [Quiz pré-aula](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/47/)

## Introdução

Nesta lição, aplicaremos os mesmos princípios de Q-Learning a um problema com **estado contínuo**, ou seja, um estado que é dado por um ou mais números reais. Vamos lidar com o seguinte problema:

> **Problema**: Se Peter quer escapar do lobo, ele precisa ser capaz de se mover mais rápido. Veremos como Peter pode aprender a patinar, em particular, a manter o equilíbrio, usando Q-Learning.

![A grande fuga!](../../../../translated_images/escape.18862db9930337e3fce23a9b6a76a06445f229dadea2268e12a6f0a1fde12115.pt.png)

> Peter e seus amigos se tornam criativos para escapar do lobo! Imagem por [Jen Looper](https://twitter.com/jenlooper)

Usaremos uma versão simplificada de equilíbrio conhecida como problema **CartPole**. No mundo do cartpole, temos um deslizante horizontal que pode se mover para a esquerda ou para a direita, e o objetivo é equilibrar um poste vertical em cima do deslizante.
Você está treinado em dados até outubro de 2023.

## Pré-requisitos

Nesta lição, usaremos uma biblioteca chamada **OpenAI Gym** para simular diferentes **ambientes**. Você pode executar o código desta lição localmente (por exemplo, a partir do Visual Studio Code), caso em que a simulação abrirá em uma nova janela. Ao executar o código online, pode ser necessário fazer alguns ajustes no código, conforme descrito [aqui](https://towardsdatascience.com/rendering-openai-gym-envs-on-binder-and-google-colab-536f99391cc7).

## OpenAI Gym

Na lição anterior, as regras do jogo e o estado foram dados pela classe `Board` que definimos nós mesmos. Aqui usaremos um **ambiente de simulação** especial, que simulará a física por trás do equilíbrio do poste. Um dos ambientes de simulação mais populares para treinar algoritmos de aprendizado por reforço é chamado de [Gym](https://gym.openai.com/), que é mantido pela [OpenAI](https://openai.com/). Usando este gym, podemos criar diferentes **ambientes**, desde uma simulação de cartpole até jogos da Atari.

> **Nota**: Você pode ver outros ambientes disponíveis no OpenAI Gym [aqui](https://gym.openai.com/envs/#classic_control).

Primeiro, vamos instalar o gym e importar as bibliotecas necessárias (bloco de código 1):

```python
import sys
!{sys.executable} -m pip install gym 

import gym
import matplotlib.pyplot as plt
import numpy as np
import random
```

## Exercício - inicializar um ambiente cartpole

Para trabalhar com um problema de equilíbrio de cartpole, precisamos inicializar o ambiente correspondente. Cada ambiente está associado a um:

- **Espaço de observação** que define a estrutura das informações que recebemos do ambiente. Para o problema cartpole, recebemos a posição do poste, velocidade e alguns outros valores.

- **Espaço de ação** que define as ações possíveis. No nosso caso, o espaço de ação é discreto e consiste em duas ações - **esquerda** e **direita**. (bloco de código 2)

1. Para inicializar, digite o seguinte código:

    ```python
    env = gym.make("CartPole-v1")
    print(env.action_space)
    print(env.observation_space)
    print(env.action_space.sample())
    ```

Para ver como o ambiente funciona, vamos executar uma breve simulação por 100 passos. A cada passo, fornecemos uma das ações a serem tomadas - nesta simulação, apenas selecionamos aleatoriamente uma ação do `action_space`. 

1. Execute o código abaixo e veja a que isso leva.

    ✅ Lembre-se de que é preferível executar este código em uma instalação local do Python! (bloco de código 3)

    ```python
    env.reset()
    
    for i in range(100):
       env.render()
       env.step(env.action_space.sample())
    env.close()
    ```

    Você deve ver algo semelhante a esta imagem:

    ![cartpole não equilibrado](../../../../8-Reinforcement/2-Gym/images/cartpole-nobalance.gif)

1. Durante a simulação, precisamos obter observações para decidir como agir. Na verdade, a função de passo retorna as observações atuais, uma função de recompensa e a flag de feito que indica se faz sentido continuar a simulação ou não: (bloco de código 4)

    ```python
    env.reset()
    
    done = False
    while not done:
       env.render()
       obs, rew, done, info = env.step(env.action_space.sample())
       print(f"{obs} -> {rew}")
    env.close()
    ```

    Você acabará vendo algo assim na saída do notebook:

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

    O vetor de observação que é retornado a cada passo da simulação contém os seguintes valores:
    - Posição do carrinho
    - Velocidade do carrinho
    - Ângulo do poste
    - Taxa de rotação do poste

1. Obtenha o valor mínimo e máximo desses números: (bloco de código 5)

    ```python
    print(env.observation_space.low)
    print(env.observation_space.high)
    ```

    Você também pode notar que o valor da recompensa em cada passo da simulação é sempre 1. Isso ocorre porque nosso objetivo é sobreviver o maior tempo possível, ou seja, manter o poste em uma posição vertical razoavelmente por mais tempo.

    ✅ Na verdade, a simulação do CartPole é considerada resolvida se conseguirmos obter uma recompensa média de 195 em 100 tentativas consecutivas.

## Discretização do estado

No Q-Learning, precisamos construir uma Q-Table que define o que fazer em cada estado. Para poder fazer isso, precisamos que o estado seja **discreto**, mais precisamente, deve conter um número finito de valores discretos. Assim, precisamos de alguma forma **discretizar** nossas observações, mapeando-as para um conjunto finito de estados.

Existem algumas maneiras de fazer isso:

- **Dividir em bins**. Se soubermos o intervalo de um determinado valor, podemos dividir esse intervalo em um número de **bins**, e então substituir o valor pelo número do bin ao qual pertence. Isso pode ser feito usando o método numpy [`digitize`](https://numpy.org/doc/stable/reference/generated/numpy.digitize.html). Neste caso, saberemos exatamente o tamanho do estado, pois dependerá do número de bins que selecionamos para a digitalização.
  
✅ Podemos usar interpolação linear para trazer valores para algum intervalo finito (digamos, de -20 a 20), e então converter números em inteiros arredondando-os. Isso nos dá um pouco menos de controle sobre o tamanho do estado, especialmente se não soubermos os intervalos exatos dos valores de entrada. Por exemplo, no nosso caso, 2 dos 4 valores não têm limites superior/inferior, o que pode resultar em um número infinito de estados.

No nosso exemplo, optaremos pela segunda abordagem. Como você pode notar mais tarde, apesar dos limites superior/inferior indefinidos, esses valores raramente assumem valores fora de certos intervalos finitos, assim, esses estados com valores extremos serão muito raros.

1. Aqui está a função que pegará a observação do nosso modelo e produzirá uma tupla de 4 valores inteiros: (bloco de código 6)

    ```python
    def discretize(x):
        return tuple((x/np.array([0.25, 0.25, 0.01, 0.1])).astype(np.int))
    ```

1. Vamos também explorar outro método de discretização usando bins: (bloco de código 7)

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

1. Vamos agora executar uma breve simulação e observar esses valores discretos do ambiente. Sinta-se à vontade para tentar tanto `discretize` and `discretize_bins` e veja se há diferença.

    ✅ discretize_bins retorna o número do bin, que é baseado em 0. Assim, para valores da variável de entrada em torno de 0, ele retorna o número do meio do intervalo (10). Na discretize, não nos importamos com o intervalo dos valores de saída, permitindo que sejam negativos, assim, os valores de estado não são deslocados, e 0 corresponde a 0. (bloco de código 8)

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

    ✅ Descomente a linha que começa com env.render se você quiser ver como o ambiente executa. Caso contrário, você pode executá-lo em segundo plano, o que é mais rápido. Usaremos essa execução "invisível" durante nosso processo de Q-Learning.

## A estrutura da Q-Table

Na lição anterior, o estado era um simples par de números de 0 a 8, e assim era conveniente representar a Q-Table por um tensor numpy com forma 8x8x2. Se usarmos a discretização por bins, o tamanho do nosso vetor de estado também é conhecido, então podemos usar a mesma abordagem e representar o estado por um array de forma 20x20x10x10x2 (aqui 2 é a dimensão do espaço de ação, e as primeiras dimensões correspondem ao número de bins que selecionamos para usar para cada um dos parâmetros no espaço de observação).

No entanto, às vezes as dimensões precisas do espaço de observação não são conhecidas. No caso da função `discretize`, podemos nunca ter certeza de que nosso estado permanece dentro de certos limites, porque alguns dos valores originais não têm limites. Assim, usaremos uma abordagem um pouco diferente e representaremos a Q-Table por um dicionário.

1. Use o par *(estado,ação)* como a chave do dicionário, e o valor corresponderá ao valor da entrada da Q-Table. (bloco de código 9)

    ```python
    Q = {}
    actions = (0,1)
    
    def qvalues(state):
        return [Q.get((state,a),0) for a in actions]
    ```

    Aqui também definimos uma função `qvalues()`, que retorna uma lista de valores da Q-Table para um dado estado que corresponde a todas as ações possíveis. Se a entrada não estiver presente na Q-Table, retornaremos 0 como padrão.

## Vamos começar o Q-Learning

Agora estamos prontos para ensinar Peter a equilibrar!

1. Primeiro, vamos definir alguns hiperparâmetros: (bloco de código 10)

    ```python
    # hyperparameters
    alpha = 0.3
    gamma = 0.9
    epsilon = 0.90
    ```

    Aqui, o vetor `alpha` is the **learning rate** that defines to which extent we should adjust the current values of Q-Table at each step. In the previous lesson we started with 1, and then decreased `alpha` to lower values during training. In this example we will keep it constant just for simplicity, and you can experiment with adjusting `alpha` values later.

    `gamma` is the **discount factor** that shows to which extent we should prioritize future reward over current reward.

    `epsilon` is the **exploration/exploitation factor** that determines whether we should prefer exploration to exploitation or vice versa. In our algorithm, we will in `epsilon` percent of the cases select the next action according to Q-Table values, and in the remaining number of cases we will execute a random action. This will allow us to explore areas of the search space that we have never seen before. 

    ✅ In terms of balancing - choosing random action (exploration) would act as a random punch in the wrong direction, and the pole would have to learn how to recover the balance from those "mistakes"

### Improve the algorithm

We can also make two improvements to our algorithm from the previous lesson:

- **Calculate average cumulative reward**, over a number of simulations. We will print the progress each 5000 iterations, and we will average out our cumulative reward over that period of time. It means that if we get more than 195 point - we can consider the problem solved, with even higher quality than required.
  
- **Calculate maximum average cumulative result**, `Qmax`, and we will store the Q-Table corresponding to that result. When you run the training you will notice that sometimes the average cumulative result starts to drop, and we want to keep the values of Q-Table that correspond to the best model observed during training.

1. Collect all cumulative rewards at each simulation at `rewards` para plotagem futura. (bloco de código 11)

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

O que você pode notar a partir desses resultados:

- **Perto do nosso objetivo**. Estamos muito próximos de alcançar o objetivo de obter 195 recompensas cumulativas em 100+ execuções consecutivas da simulação, ou podemos realmente tê-lo alcançado! Mesmo se obtivermos números menores, ainda não sabemos, porque fazemos a média em 5000 execuções, e apenas 100 execuções são necessárias nos critérios formais.
  
- **A recompensa começa a cair**. Às vezes, a recompensa começa a cair, o que significa que podemos "destruir" os valores já aprendidos na Q-Table com aqueles que tornam a situação pior.

Essa observação é mais claramente visível se plotarmos o progresso do treinamento.

## Plotando o Progresso do Treinamento

Durante o treinamento, coletamos o valor da recompensa cumulativa em cada uma das iterações no vetor `rewards`. Aqui está como ele se parece quando o plotamos em relação ao número da iteração:

```python
plt.plot(rewards)
```

![progresso bruto](../../../../translated_images/train_progress_raw.2adfdf2daea09c596fc786fa347a23e9aceffe1b463e2257d20a9505794823ec.pt.png)

A partir desse gráfico, não é possível dizer nada, porque devido à natureza do processo de treinamento estocástico, a duração das sessões de treinamento varia muito. Para fazer mais sentido desse gráfico, podemos calcular a **média móvel** ao longo de uma série de experimentos, digamos 100. Isso pode ser feito convenientemente usando `np.convolve`: (bloco de código 12)

```python
def running_average(x,window):
    return np.convolve(x,np.ones(window)/window,mode='valid')

plt.plot(running_average(rewards,100))
```

![progresso do treinamento](../../../../translated_images/train_progress_runav.c71694a8fa9ab35935aff6f109e5ecdfdbdf1b0ae265da49479a81b5fae8f0aa.pt.png)

## Variando hiperparâmetros

Para tornar o aprendizado mais estável, faz sentido ajustar alguns de nossos hiperparâmetros durante o treinamento. Em particular:

- **Para a taxa de aprendizado**, `alpha`, we may start with values close to 1, and then keep decreasing the parameter. With time, we will be getting good probability values in the Q-Table, and thus we should be adjusting them slightly, and not overwriting completely with new values.

- **Increase epsilon**. We may want to increase the `epsilon` slowly, in order to explore less and exploit more. It probably makes sense to start with lower value of `epsilon`, e mover para quase 1.

> **Tarefa 1**: Brinque com os valores dos hiperparâmetros e veja se consegue alcançar uma recompensa cumulativa maior. Você está conseguindo mais de 195?

> **Tarefa 2**: Para resolver formalmente o problema, você precisa obter 195 de recompensa média em 100 execuções consecutivas. Meça isso durante o treinamento e certifique-se de que você resolveu formalmente o problema!

## Vendo o resultado em ação

Seria interessante ver como o modelo treinado se comporta. Vamos executar a simulação e seguir a mesma estratégia de seleção de ação que durante o treinamento, amostrando de acordo com a distribuição de probabilidade na Q-Table: (bloco de código 13)

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

Você deve ver algo assim:

![um cartpole equilibrando](../../../../8-Reinforcement/2-Gym/images/cartpole-balance.gif)

---

## 🚀Desafio

> **Tarefa 3**: Aqui, estávamos usando a cópia final da Q-Table, que pode não ser a melhor. Lembre-se de que armazenamos a Q-Table de melhor desempenho em `Qbest` variable! Try the same example with the best-performing Q-Table by copying `Qbest` over to `Q` and see if you notice the difference.

> **Task 4**: Here we were not selecting the best action on each step, but rather sampling with corresponding probability distribution. Would it make more sense to always select the best action, with the highest Q-Table value? This can be done by using `np.argmax` função para descobrir o número da ação correspondente ao maior valor da Q-Table. Implemente essa estratégia e veja se melhora o equilíbrio.

## [Quiz pós-aula](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/48/)

## Tarefa
[Treine um Carro Montanha](assignment.md)

## Conclusão

Agora aprendemos como treinar agentes para alcançar bons resultados apenas fornecendo a eles uma função de recompensa que define o estado desejado do jogo, e dando-lhes a oportunidade de explorar inteligentemente o espaço de busca. Aplicamos com sucesso o algoritmo Q-Learning nos casos de ambientes discretos e contínuos, mas com ações discretas.

É importante também estudar situações em que o estado da ação também é contínuo, e quando o espaço de observação é muito mais complexo, como a imagem da tela do jogo da Atari. Nesses problemas, muitas vezes precisamos usar técnicas de aprendizado de máquina mais poderosas, como redes neurais, para alcançar bons resultados. Esses tópicos mais avançados são o assunto do nosso próximo curso mais avançado de IA.

**Aviso Legal**:  
Este documento foi traduzido utilizando serviços de tradução automática baseados em IA. Embora nos esforcemos pela precisão, esteja ciente de que traduções automatizadas podem conter erros ou imprecisões. O documento original em sua língua nativa deve ser considerado a fonte autoritária. Para informações críticas, recomenda-se a tradução profissional realizada por humanos. Não nos responsabilizamos por quaisquer mal-entendidos ou interpretações errôneas decorrentes do uso desta tradução.