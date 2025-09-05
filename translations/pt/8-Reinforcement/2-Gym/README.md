<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "107d5bb29da8a562e7ae72262d251a75",
  "translation_date": "2025-09-05T08:49:47+00:00",
  "source_file": "8-Reinforcement/2-Gym/README.md",
  "language_code": "pt"
}
-->
## Pré-requisitos

Nesta lição, utilizaremos uma biblioteca chamada **OpenAI Gym** para simular diferentes **ambientes**. Pode executar o código desta lição localmente (por exemplo, no Visual Studio Code), caso em que a simulação será aberta numa nova janela. Ao executar o código online, poderá precisar de fazer alguns ajustes, conforme descrito [aqui](https://towardsdatascience.com/rendering-openai-gym-envs-on-binder-and-google-colab-536f99391cc7).

## OpenAI Gym

Na lição anterior, as regras do jogo e o estado foram definidos pela classe `Board`, que criámos nós mesmos. Aqui, utilizaremos um **ambiente de simulação** especial, que simulará a física por trás do equilíbrio do poste. Um dos ambientes de simulação mais populares para treinar algoritmos de aprendizagem por reforço é chamado de [Gym](https://gym.openai.com/), mantido pela [OpenAI](https://openai.com/). Com este Gym, podemos criar diferentes **ambientes**, desde simulações de CartPole até jogos de Atari.

> **Nota**: Pode ver outros ambientes disponíveis no OpenAI Gym [aqui](https://gym.openai.com/envs/#classic_control).

Primeiro, vamos instalar o Gym e importar as bibliotecas necessárias (bloco de código 1):

```python
import sys
!{sys.executable} -m pip install gym 

import gym
import matplotlib.pyplot as plt
import numpy as np
import random
```

## Exercício - inicializar um ambiente de CartPole

Para trabalhar com o problema de equilíbrio do CartPole, precisamos de inicializar o ambiente correspondente. Cada ambiente está associado a:

- **Espaço de observação**, que define a estrutura da informação que recebemos do ambiente. No problema do CartPole, recebemos a posição do poste, a velocidade e outros valores.

- **Espaço de ação**, que define as ações possíveis. No nosso caso, o espaço de ação é discreto e consiste em duas ações: **esquerda** e **direita**. (bloco de código 2)

1. Para inicializar, escreva o seguinte código:

    ```python
    env = gym.make("CartPole-v1")
    print(env.action_space)
    print(env.observation_space)
    print(env.action_space.sample())
    ```

Para ver como o ambiente funciona, vamos executar uma simulação curta de 100 passos. Em cada passo, fornecemos uma das ações a serem realizadas - nesta simulação, selecionamos aleatoriamente uma ação do `action_space`.

1. Execute o código abaixo e veja o resultado.

    ✅ Lembre-se de que é preferível executar este código numa instalação local de Python! (bloco de código 3)

    ```python
    env.reset()
    
    for i in range(100):
       env.render()
       env.step(env.action_space.sample())
    env.close()
    ```

    Deve ver algo semelhante a esta imagem:

    ![CartPole sem equilíbrio](../../../../8-Reinforcement/2-Gym/images/cartpole-nobalance.gif)

1. Durante a simulação, precisamos de obter observações para decidir como agir. Na verdade, a função step retorna as observações atuais, uma função de recompensa e um indicador `done` que indica se faz sentido continuar a simulação ou não: (bloco de código 4)

    ```python
    env.reset()
    
    done = False
    while not done:
       env.render()
       obs, rew, done, info = env.step(env.action_space.sample())
       print(f"{obs} -> {rew}")
    env.close()
    ```

    No notebook, verá algo semelhante a isto:

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

    O vetor de observação retornado em cada passo da simulação contém os seguintes valores:
    - Posição do carrinho
    - Velocidade do carrinho
    - Ângulo do poste
    - Taxa de rotação do poste

1. Obtenha os valores mínimos e máximos desses números: (bloco de código 5)

    ```python
    print(env.observation_space.low)
    print(env.observation_space.high)
    ```

    Também pode notar que o valor de recompensa em cada passo da simulação é sempre 1. Isto acontece porque o nosso objetivo é sobreviver o maior tempo possível, ou seja, manter o poste numa posição razoavelmente vertical pelo maior período de tempo.

    ✅ Na verdade, a simulação do CartPole é considerada resolvida se conseguirmos obter uma recompensa média de 195 ao longo de 100 tentativas consecutivas.

## Discretização do estado

No Q-Learning, precisamos de construir uma Q-Table que define o que fazer em cada estado. Para isso, o estado precisa de ser **discreto**, ou seja, deve conter um número finito de valores discretos. Assim, precisamos de alguma forma **discretizar** as nossas observações, mapeando-as para um conjunto finito de estados.

Existem algumas formas de fazer isto:

- **Dividir em intervalos**. Se conhecemos o intervalo de um determinado valor, podemos dividir este intervalo em vários **bins** e substituir o valor pelo número do bin ao qual pertence. Isto pode ser feito usando o método [`digitize`](https://numpy.org/doc/stable/reference/generated/numpy.digitize.html) do numpy. Neste caso, saberemos exatamente o tamanho do estado, pois dependerá do número de bins que selecionarmos para a digitalização.

✅ Podemos usar interpolação linear para trazer os valores para um intervalo finito (por exemplo, de -20 a 20) e, em seguida, converter os números em inteiros arredondando-os. Isto dá-nos um pouco menos de controlo sobre o tamanho do estado, especialmente se não conhecermos os intervalos exatos dos valores de entrada. Por exemplo, no nosso caso, 2 dos 4 valores não têm limites superiores/inferiores, o que pode resultar num número infinito de estados.

No nosso exemplo, utilizaremos a segunda abordagem. Como poderá notar mais tarde, apesar de os limites superiores/inferiores não estarem definidos, esses valores raramente assumem valores fora de certos intervalos finitos, tornando os estados com valores extremos muito raros.

1. Aqui está a função que irá receber a observação do nosso modelo e produzir um tuplo de 4 valores inteiros: (bloco de código 6)

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

1. Agora, execute uma simulação curta e observe esses valores discretos do ambiente. Sinta-se à vontade para experimentar tanto `discretize` quanto `discretize_bins` e veja se há alguma diferença.

    ✅ `discretize_bins` retorna o número do bin, que é baseado em 0. Assim, para valores da variável de entrada próximos de 0, retorna o número do meio do intervalo (10). Em `discretize`, não nos preocupámos com o intervalo dos valores de saída, permitindo que sejam negativos, e assim os valores do estado não são deslocados, e 0 corresponde a 0. (bloco de código 8)

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

    ✅ Descomente a linha que começa com `env.render` se quiser ver como o ambiente é executado. Caso contrário, pode executá-lo em segundo plano, o que é mais rápido. Utilizaremos esta execução "invisível" durante o nosso processo de Q-Learning.

## Estrutura da Q-Table

Na lição anterior, o estado era um simples par de números de 0 a 8, e assim era conveniente representar a Q-Table por um tensor numpy com uma forma de 8x8x2. Se utilizarmos a discretização por bins, o tamanho do nosso vetor de estado também será conhecido, então podemos usar a mesma abordagem e representar o estado por um array com a forma 20x20x10x10x2 (aqui 2 é a dimensão do espaço de ação, e as primeiras dimensões correspondem ao número de bins que selecionámos para cada um dos parâmetros no espaço de observação).

No entanto, às vezes as dimensões precisas do espaço de observação não são conhecidas. No caso da função `discretize`, nunca podemos ter certeza de que o nosso estado permanece dentro de certos limites, porque alguns dos valores originais não têm limites. Assim, utilizaremos uma abordagem ligeiramente diferente e representaremos a Q-Table por um dicionário.

1. Use o par *(state,action)* como chave do dicionário, e o valor corresponderá ao valor da entrada na Q-Table. (bloco de código 9)

    ```python
    Q = {}
    actions = (0,1)
    
    def qvalues(state):
        return [Q.get((state,a),0) for a in actions]
    ```

    Aqui também definimos uma função `qvalues()`, que retorna uma lista de valores da Q-Table para um determinado estado que corresponde a todas as ações possíveis. Se a entrada não estiver presente na Q-Table, retornaremos 0 como padrão.

## Vamos começar o Q-Learning

Agora estamos prontos para ensinar o Peter a equilibrar!

1. Primeiro, vamos definir alguns hiperparâmetros: (bloco de código 10)

    ```python
    # hyperparameters
    alpha = 0.3
    gamma = 0.9
    epsilon = 0.90
    ```

    Aqui, `alpha` é a **taxa de aprendizagem**, que define até que ponto devemos ajustar os valores atuais da Q-Table em cada passo. Na lição anterior, começámos com 1 e depois reduzimos `alpha` para valores mais baixos durante o treino. Neste exemplo, manteremos constante apenas por simplicidade, e pode experimentar ajustar os valores de `alpha` mais tarde.

    `gamma` é o **fator de desconto**, que mostra até que ponto devemos priorizar a recompensa futura em relação à recompensa atual.

    `epsilon` é o **fator de exploração/exploração**, que determina se devemos preferir explorar ou explorar. No nosso algoritmo, em `epsilon` por cento dos casos, selecionaremos a próxima ação de acordo com os valores da Q-Table, e no restante dos casos executaremos uma ação aleatória. Isto permitirá explorar áreas do espaço de busca que nunca vimos antes.

    ✅ Em termos de equilíbrio - escolher uma ação aleatória (exploração) seria como um empurrão aleatório na direção errada, e o poste teria de aprender a recuperar o equilíbrio desses "erros".

### Melhorar o algoritmo

Podemos também fazer duas melhorias ao nosso algoritmo da lição anterior:

- **Calcular a recompensa cumulativa média**, ao longo de várias simulações. Iremos imprimir o progresso a cada 5000 iterações e calcularemos a média da recompensa cumulativa nesse período de tempo. Isto significa que, se obtivermos mais de 195 pontos, podemos considerar o problema resolvido, com qualidade ainda maior do que o necessário.

- **Calcular o resultado cumulativo médio máximo**, `Qmax`, e armazenaremos a Q-Table correspondente a esse resultado. Quando executar o treino, notará que, às vezes, o resultado cumulativo médio começa a cair, e queremos manter os valores da Q-Table que correspondem ao melhor modelo observado durante o treino.

1. Colete todas as recompensas cumulativas em cada simulação no vetor `rewards` para posterior plotagem. (bloco de código 11)

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

O que pode notar a partir desses resultados:

- **Perto do nosso objetivo**. Estamos muito próximos de alcançar o objetivo de obter 195 recompensas cumulativas ao longo de 100+ execuções consecutivas da simulação, ou podemos até ter alcançado! Mesmo que obtenhamos números menores, ainda não sabemos, porque calculamos a média ao longo de 5000 execuções, e apenas 100 execuções são necessárias no critério formal.

- **Recompensa começa a cair**. Às vezes, a recompensa começa a cair, o que significa que podemos "destruir" valores já aprendidos na Q-Table com os que pioram a situação.

Esta observação é mais claramente visível se plotarmos o progresso do treino.

## Plotar o progresso do treino

Durante o treino, coletámos o valor da recompensa cumulativa em cada uma das iterações no vetor `rewards`. Aqui está como fica quando o plotamos contra o número de iterações:

```python
plt.plot(rewards)
```

![progresso bruto](../../../../8-Reinforcement/2-Gym/images/train_progress_raw.png)

A partir deste gráfico, não é possível dizer nada, porque, devido à natureza do processo de treino estocástico, a duração das sessões de treino varia muito. Para dar mais sentido a este gráfico, podemos calcular a **média móvel** ao longo de uma série de experimentos, digamos 100. Isto pode ser feito convenientemente usando `np.convolve`: (bloco de código 12)

```python
def running_average(x,window):
    return np.convolve(x,np.ones(window)/window,mode='valid')

plt.plot(running_average(rewards,100))
```

![progresso do treino](../../../../8-Reinforcement/2-Gym/images/train_progress_runav.png)

## Variar os hiperparâmetros

Para tornar o treino mais estável, faz sentido ajustar alguns dos nossos hiperparâmetros durante o treino. Em particular:

- **Para a taxa de aprendizagem**, `alpha`, podemos começar com valores próximos de 1 e depois ir reduzindo o parâmetro. Com o tempo, obteremos boas probabilidades na Q-Table e, assim, devemos ajustá-las ligeiramente, e não sobrescrever completamente com novos valores.

- **Aumentar epsilon**. Podemos querer aumentar o `epsilon` lentamente, para explorar menos e explorar mais. Provavelmente faz sentido começar com um valor mais baixo de `epsilon` e aumentar até quase 1.
> **Tarefa 1**: Experimente alterar os valores dos hiperparâmetros e veja se consegue obter uma recompensa cumulativa mais alta. Está a conseguir ultrapassar 195?
> **Tarefa 2**: Para resolver formalmente o problema, é necessário alcançar uma recompensa média de 195 ao longo de 100 execuções consecutivas. Meça isso durante o treino e certifique-se de que o problema foi resolvido formalmente!

## Ver o resultado em ação

Seria interessante ver como o modelo treinado se comporta. Vamos executar a simulação e seguir a mesma estratégia de seleção de ações usada durante o treino, amostrando de acordo com a distribuição de probabilidade na Q-Table: (bloco de código 13)

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

Deverá aparecer algo como isto:

![um cartpole equilibrado](../../../../8-Reinforcement/2-Gym/images/cartpole-balance.gif)

---

## 🚀Desafio

> **Tarefa 3**: Aqui, estávamos a usar a cópia final da Q-Table, que pode não ser a melhor. Lembre-se de que armazenámos a Q-Table com melhor desempenho na variável `Qbest`! Experimente o mesmo exemplo com a Q-Table de melhor desempenho, copiando `Qbest` para `Q`, e veja se nota alguma diferença.

> **Tarefa 4**: Aqui, não estávamos a selecionar a melhor ação em cada passo, mas sim a amostrar com a correspondente distribuição de probabilidade. Faria mais sentido selecionar sempre a melhor ação, com o valor mais alto na Q-Table? Isto pode ser feito utilizando a função `np.argmax` para descobrir o número da ação correspondente ao maior valor na Q-Table. Implemente esta estratégia e veja se melhora o equilíbrio.

## [Questionário pós-aula](https://ff-quizzes.netlify.app/en/ml/)

## Tarefa
[Treinar um Mountain Car](assignment.md)

## Conclusão

Aprendemos agora como treinar agentes para alcançar bons resultados apenas fornecendo-lhes uma função de recompensa que define o estado desejado do jogo e dando-lhes a oportunidade de explorar inteligentemente o espaço de busca. Aplicámos com sucesso o algoritmo de Q-Learning em casos de ambientes discretos e contínuos, mas com ações discretas.

É importante também estudar situações em que o estado das ações é contínuo e quando o espaço de observação é muito mais complexo, como a imagem do ecrã de um jogo Atari. Nestes problemas, muitas vezes é necessário usar técnicas de aprendizagem automática mais avançadas, como redes neuronais, para alcançar bons resultados. Esses tópicos mais avançados serão abordados no nosso próximo curso de IA mais avançado.

---

**Aviso Legal**:  
Este documento foi traduzido utilizando o serviço de tradução por IA [Co-op Translator](https://github.com/Azure/co-op-translator). Embora nos esforcemos para garantir a precisão, esteja ciente de que traduções automáticas podem conter erros ou imprecisões. O documento original no seu idioma nativo deve ser considerado a fonte oficial. Para informações críticas, recomenda-se uma tradução profissional realizada por humanos. Não nos responsabilizamos por quaisquer mal-entendidos ou interpretações incorretas resultantes do uso desta tradução.