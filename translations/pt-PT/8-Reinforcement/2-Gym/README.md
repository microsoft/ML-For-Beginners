# CartPole Skating

O problema que resolvemos na lição anterior pode parecer um problema de brinquedo, não realmente aplicável a cenários da vida real. Isso não é o caso, porque muitos problemas do mundo real também partilham este cenário - incluindo jogar Xadrez ou Go. São semelhantes porque também temos um tabuleiro com regras dadas e um **estado discreto**.

## [Quiz pré-aula](https://ff-quizzes.netlify.app/en/ml/)

## Introdução

Nesta lição aplicaremos os mesmos princípios do Q-Learning a um problema com **estado contínuo**, ou seja, um estado dado por um ou mais números reais. Vamos lidar com o seguinte problema:

> **Problema**: Se o Peter quiser escapar do lobo, ele precisa ser capaz de se mover mais rapidamente. Veremos como o Peter pode aprender a patinar, em particular, a manter o equilíbrio, usando Q-Learning.

![A grande fuga!](../../../../translated_images/pt-PT/escape.18862db9930337e3.webp)

> O Peter e os seus amigos tornam-se criativos para escapar ao lobo! Imagem de [Jen Looper](https://twitter.com/jenlooper)

Vamos usar uma versão simplificada do equilíbrio conhecido como problema **CartPole**. No mundo do cartpole, temos um deslizador horizontal que pode mover-se para a esquerda ou para a direita, e o objetivo é equilibrar uma vara vertical em cima do deslizador.

<img alt="a cartpole" src="../../../../translated_images/pt-PT/cartpole.b5609cc0494a14f7.webp" width="200"/>

## Pré-requisitos

Nesta lição, usaremos uma biblioteca chamada **OpenAI Gym** para simular diferentes **ambientes**. Pode executar o código desta lição localmente (por exemplo, no Visual Studio Code), caso em que a simulação abrirá numa nova janela. Ao executar o código online, pode ser necessário fazer alguns ajustes no código, conforme descrito [aqui](https://towardsdatascience.com/rendering-openai-gym-envs-on-binder-and-google-colab-536f99391cc7).

## OpenAI Gym

Na lição anterior, as regras do jogo e o estado foram definidos pela classe `Board` que criámos. Aqui usaremos um **ambiente de simulação** especial, que simulará a física por trás da vara de equilíbrio. Um dos ambientes de simulação mais populares para treinar algoritmos de aprendizagem por reforço chama-se [Gym](https://gym.openai.com/), mantido pela [OpenAI](https://openai.com/). Usando este gym podemos criar diversos **ambientes**, desde uma simulação de cartpole até jogos de Atari.

> **Nota**: Pode consultar outros ambientes disponíveis no OpenAI Gym [aqui](https://gym.openai.com/envs/#classic_control). 

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

Para trabalhar com um problema de equilíbrio cartpole, precisamos inicializar o ambiente correspondente. Cada ambiente está associado a:

- **Espaço de observação** que define a estrutura da informação que recebemos do ambiente. Para o problema cartpole, recebemos a posição da vara, a velocidade e alguns outros valores.

- **Espaço de ação** que define as ações possíveis. No nosso caso, o espaço de ação é discreto, e consiste em duas ações - **esquerda** e **direita**. (bloco de código 2)

1. Para inicializar, digite o seguinte código:

    ```python
    env = gym.make("CartPole-v1")
    print(env.action_space)
    print(env.observation_space)
    print(env.action_space.sample())
    ```

Para ver como o ambiente funciona, vamos executar uma curta simulação de 100 passos. A cada passo, fornecemos uma das ações a realizar - nesta simulação selecionamos uma ação aleatória do `action_space`. 

1. Execute o código abaixo e veja o resultado.

    ✅ Lembre-se que é preferível executar este código numa instalação local do Python! (bloco de código 3)

    ```python
    env.reset()
    
    for i in range(100):
       env.render()
       env.step(env.action_space.sample())
    env.close()
    ```

    Deve ver algo parecido com esta imagem:

    ![cartpole sem equilíbrio](../../../../8-Reinforcement/2-Gym/images/cartpole-nobalance.gif)

1. Durante a simulação, precisamos de obter observações para decidir como agir. De facto, a função step retorna observações atuais, uma função de recompensa e a flag done que indica se faz sentido continuar a simulação ou não: (bloco de código 4)

    ```python
    env.reset()
    
    done = False
    while not done:
       env.render()
       obs, rew, done, info = env.step(env.action_space.sample())
       print(f"{obs} -> {rew}")
    env.close()
    ```

    Eventualmente verá algo deste género na saída do notebook:

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

    O vetor de observação retornado a cada passo da simulação contém os seguintes valores:
    - Posição do carrinho
    - Velocidade do carrinho
    - Ângulo da vara
    - Taxa de rotação da vara

1. Obtenha o valor mínimo e máximo desses números: (bloco de código 5)

    ```python
    print(env.observation_space.low)
    print(env.observation_space.high)
    ```

    Poderá também notar que o valor da recompensa em cada passo da simulação é sempre 1. Isto acontece porque o nosso objetivo é sobreviver o maior tempo possível, ou seja, manter a vara numa posição razoavelmente vertical pelo maior período de tempo.

    ✅ De facto, a simulação CartPole é considerada resolvida se conseguirmos obter uma recompensa média de 195 em 100 tentativas consecutivas.

## Discretização do estado

No Q-Learning, precisamos construir uma Q-Table que defina o que fazer em cada estado. Para isso, o estado precisa ser **discreto**, mais precisamente, deve conter um número finito de valores discretos. Assim, precisamos de alguma forma **discretizar** as nossas observações, mapeando-as para um conjunto finito de estados.

Existem algumas formas de o fazer:

- **Dividir em binários**. Se soubermos o intervalo de um certo valor, podemos dividi-lo num número de **bins**, e então substituir o valor pelo número do bin a que pertence. Isto pode ser feito usando o método do numpy [`digitize`](https://numpy.org/doc/stable/reference/generated/numpy.digitize.html). Neste caso, saberemos exatamente o tamanho do estado, porque ele depende do número de bins selecionados para digitalização.
  
✅ Podemos usar interpolação linear para trazer os valores para algum intervalo finito (por exemplo, de -20 a 20), e depois converter os números para inteiros arredondando-os. Isto dá-nos menos controlo sobre o tamanho do estado, especialmente se não conhecermos os intervalos exatos dos valores de entrada. Por exemplo, no nosso caso 2 dos 4 valores não têm limites superiores/inferiores, o que pode resultar num número infinito de estados.

No nosso exemplo, usaremos a segunda abordagem. Como poderá notar mais tarde, apesar de não existirem limites superiores/inferiores definidos, esses valores raramente tomam valores fora de certos intervalos finitos, portanto esses estados com valores extremos serão muito raros.

1. Aqui está a função que recebe a observação do nosso modelo e produz uma tupla de 4 valores inteiros: (bloco de código 6)

    ```python
    def discretize(x):
        return tuple((x/np.array([0.25, 0.25, 0.01, 0.1])).astype(np.int))
    ```

1. Vamos também explorar outro método de discretização usando bins: (bloco de código 7)

    ```python
    def create_bins(i,num):
        return np.arange(num+1)*(i[1]-i[0])/num+i[0]
    
    print("Sample bins for interval (-5,5) with 10 bins\n",create_bins((-5,5),10))
    
    ints = [(-5,5),(-2,2),(-0.5,0.5),(-2,2)] # intervalos de valores para cada parâmetro
    nbins = [20,20,10,10] # número de intervalos para cada parâmetro
    bins = [create_bins(ints[i],nbins[i]) for i in range(4)]
    
    def discretize_bins(x):
        return tuple(np.digitize(x[i],bins[i]) for i in range(4))
    ```

1. Agora vamos executar uma curta simulação e observar esses valores discretos do ambiente. Sinta-se à vontade para experimentar tanto `discretize` como `discretize_bins` e verificar se existe alguma diferença.

    ✅ discretize_bins retorna o número do bin, que é 0-based. Assim, para valores da variável de entrada próximos de 0, retorna o número do meio do intervalo (10). No discretize, não nos preocupámos com o intervalo dos valores de saída, permitindo valores negativos, portanto os valores do estado não são deslocados, e 0 corresponde a 0. (bloco de código 8)

    ```python
    env.reset()
    
    done = False
    while not done:
       #env.render()
       obs, rew, done, info = env.step(env.action_space.sample())
       #print(discretizar_bins(obs))
       print(discretize(obs))
    env.close()
    ```

    ✅ Descomente a linha começada por env.render se quiser ver como o ambiente executa. De outra forma, pode executar em segundo plano, o que é mais rápido. Usaremos esta execução "invisível" durante o processo de Q-Learning.

## A estrutura da Q-Table

Na nossa lição anterior, o estado era um par simples de números de 0 a 8, e assim era conveniente representar a Q-Table por um tensor numpy com forma 8x8x2. Se usarmos a discretização em bins, o tamanho do vetor estado é também conhecido, então podemos utilizar a mesma abordagem e representar o estado por um array da forma 20x20x10x10x2 (aqui 2 é a dimensão do espaço de ação, e as primeiras dimensões correspondem ao número de bins selecionados para cada um dos parâmetros no espaço de observação).

No entanto, por vezes as dimensões precisas do espaço de observação não são conhecidas. No caso da função `discretize`, nunca poderemos ter a certeza que o nosso estado ficará dentro de certos limites, porque alguns dos valores originais não estão limitados. Assim, usaremos uma abordagem ligeiramente diferente e representaremos a Q-Table por um dicionário. 

1. Use o par *(estado, ação)* como a chave do dicionário, e o valor corresponderá à entrada na Q-Table. (bloco de código 9)

    ```python
    Q = {}
    actions = (0,1)
    
    def qvalues(state):
        return [Q.get((state,a),0) for a in actions]
    ```

    Aqui definimos também uma função `qvalues()`, que retorna uma lista de valores da Q-Table para um dado estado correspondendo a todas as ações possíveis. Se a entrada não estiver presente na Q-Table, retornaremos 0 como valor padrão.

## Vamos começar o Q-Learning

Agora estamos prontos para ensinar o Peter a equilibrar-se!

1. Primeiro, vamos definir alguns hiperparâmetros: (bloco de código 10)

    ```python
    # hiperparâmetros
    alpha = 0.3
    gamma = 0.9
    epsilon = 0.90
    ```

    Aqui, `alpha` é a **taxa de aprendizagem** que define em que medida devemos ajustar os valores atuais da Q-Table a cada passo. Na lição anterior começámos com 1 e depois diminuímos `alpha` para valores mais baixos durante o treino. Neste exemplo manteremos constante só para simplificar, e poderá experimentar ajustar os valores de `alpha` mais tarde.

    `gamma` é o **fator de desconto** que mostra até que ponto devemos priorizar a recompensa futura sobre a recompensa atual.

    `epsilon` é o **fator de exploração/exploração** que determina se devemos preferir exploração à exploração ou vice-versa. No nosso algoritmo, em `epsilon` por cento dos casos selecionaremos a próxima ação segundo os valores da Q-Table, e no número de casos restantes executaremos uma ação aleatória. Isto permitir-nos-á explorar áreas do espaço de pesquisa que nunca vimos antes. 

    ✅ Em termos de equilíbrio - escolher ação aleatória (exploração) seria como um empurrão aleatório na direção errada, e a vara teria de aprender a recuperar o equilíbrio desses "erros".

### Melhorar o algoritmo

Podemos também fazer duas melhorias ao nosso algoritmo da lição anterior:

- **Calcular recompensa cumulativa média**, ao longo de várias simulações. Imprimiremos o progresso a cada 5000 iterações, e fizemos a média da recompensa cumulativa ao longo desse período. Isto significa que se obtivermos mais de 195 pontos - podemos considerar o problema resolvido, com qualidade até superior ao requerido.
  
- **Calcular resultado médio máximo cumulativo**, `Qmax`, e guardaremos a Q-Table correspondente a esse resultado. Quando executar o treino vai notar que às vezes o resultado médio cumulativo começa a cair, e queremos manter os valores da Q-Table correspondentes ao melhor modelo observado durante o treino.

1. Colete todas as recompensas cumulativas em cada simulação no vetor `rewards` para posterior representação gráfica. (bloco de código 11)

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
        # == fazer a simulação ==
        while not done:
            s = discretize(obs)
            if random.random()<epsilon:
                # exploração - escolher a ação de acordo com as probabilidades da Tabela Q
                v = probs(np.array(qvalues(s)))
                a = random.choices(actions,weights=v)[0]
            else:
                # exploração - escolher a ação aleatoriamente
                a = np.random.randint(env.action_space.n)
    
            obs, rew, done, info = env.step(a)
            cum_reward+=rew
            ns = discretize(obs)
            Q[(s,a)] = (1 - alpha) * Q.get((s,a),0) + alpha * (rew + gamma * max(qvalues(ns)))
        cum_rewards.append(cum_reward)
        rewards.append(cum_reward)
        # == Imprimir resultados periodicamente e calcular a recompensa média ==
        if epoch%5000==0:
            print(f"{epoch}: {np.average(cum_rewards)}, alpha={alpha}, epsilon={epsilon}")
            if np.average(cum_rewards) > Qmax:
                Qmax = np.average(cum_rewards)
                Qbest = Q
            cum_rewards=[]
    ```

O que poderá notar a partir desses resultados:

- **Próximo do nosso objetivo**. Estamos muito perto de atingir o objetivo de obter 195 recompensas cumulativas em mais de 100 execuções consecutivas da simulação, ou podemos até já ter alcançado! Mesmo que fiquemos com números menores, ainda assim não sabemos, porque fazemos a média sobre 5000 execuções, enquanto o critério formal requer apenas 100 execuções.
  
- **A recompensa começa a cair**. Por vezes a recompensa começa a diminuir, o que significa que podemos "destruir" valores já aprendidos na Q-Table com outros que pioram a situação.

Esta observação é mais claramente visível se representarmos graficamente o progresso do treino.

## Representação gráfica do progresso do treino

Durante o treino, recolhemos o valor da recompensa cumulativa em cada iteração no vetor `rewards`. Aqui está como isso fica quando o representamos contra o número de iteração:

```python
plt.plot(rewards)
```

![progresso cru](../../../../translated_images/pt-PT/train_progress_raw.2adfdf2daea09c59.webp)

A partir deste gráfico, não é possível concluir nada, porque devido à natureza do processo de treino estocástico, a duração das sessões de treino varia muito. Para fazer mais sentido deste gráfico, podemos calcular a **média móvel** ao longo de uma série de experiências, digamos 100. Isto pode ser feito comodamente usando `np.convolve`: (bloco de código 12)

```python
def running_average(x,window):
    return np.convolve(x,np.ones(window)/window,mode='valid')

plt.plot(running_average(rewards,100))
```

![progresso do treino](../../../../translated_images/pt-PT/train_progress_runav.c71694a8fa9ab359.webp)

## Variar os hiperparâmetros

Para tornar a aprendizagem mais estável, faz sentido ajustar alguns dos nossos hiperparâmetros durante o treino. Em particular:

- **Para a taxa de aprendizagem**, `alpha`, podemos começar com valores próximos de 1 e depois continuar a diminuir esse parâmetro. Com o tempo, teremos bons valores de probabilidade na Q-Table e, portanto, devemos ajustá-los ligeiramente, sem sobrescrever completamente com novos valores.

- **Aumentar epsilon**. Poderemos querer aumentar o `epsilon` lentamente, para explorar menos e explorar mais. Provavelmente faz sentido começar com um valor baixo de `epsilon` e subir até quase 1.

> **Tarefa 1**: Brinque com os valores dos hiperparâmetros e veja se consegue alcançar uma recompensa cumulativa mais alta. Está a conseguir ultrapassar 195?


> **Tarefa 2**: Para resolver formalmente o problema, precisa de obter uma recompensa média de 195 ao longo de 100 execuções consecutivas. Meça isso durante o treino e certifique-se de que resolveu formalmente o problema!

## Ver o resultado em ação

Seria interessante ver realmente como o modelo treinado se comporta. Vamos correr a simulação e seguir a mesma estratégia de seleção de ação utilizada durante o treino, amostrando conforme a distribuição de probabilidade na Q-Table: (bloco de código 13)

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

Deve ver algo como isto:

![a balancing cartpole](../../../../8-Reinforcement/2-Gym/images/cartpole-balance.gif)

---

## 🚀Desafio

> **Tarefa 3**: Aqui, estávamos a usar a cópia final da Q-Table, que pode não ser a melhor. Lembre-se que armazenámos a Q-Table com melhor desempenho na variável `Qbest`! Experimente o mesmo exemplo com a Q-Table com melhor desempenho copiando `Qbest` para `Q` e veja se nota a diferença.

> **Tarefa 4**: Aqui não selecionámos a melhor ação em cada passo, mas sim amostramos com a correspondente distribuição de probabilidade. Faz mais sentido selecionar sempre a melhor ação, com o maior valor da Q-Table? Isto pode ser feito usando a função `np.argmax` para descobrir o número da ação correspondente ao maior valor da Q-Table. Implemente esta estratégia e veja se melhora o equilíbrio.

## [Questionário pós-aula](https://ff-quizzes.netlify.app/en/ml/)

## Tarefa
[Treinar um Mountain Car](assignment.md)

## Conclusão

Agora aprendemos como treinar agentes para obter bons resultados apenas fornecendo-lhes uma função de recompensa que define o estado desejado do jogo, e dando-lhes oportunidade de explorar inteligentemente o espaço de busca. Aplicámos com sucesso o algoritmo Q-Learning nos casos de ambientes discretos e contínuos, mas com ações discretas.

É importante também estudar situações onde o estado da ação é também contínuo, e quando o espaço de observação é muito mais complexo, como a imagem do ecrã do jogo Atari. Nesses problemas, frequentemente precisamos usar técnicas de aprendizagem automática mais poderosas, como redes neurais, para alcançar bons resultados. Esses tópicos mais avançados são o tema do nosso próximo curso mais avançado de IA.

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Aviso Legal**:
Este documento foi traduzido utilizando o serviço de tradução automática [Co-op Translator](https://github.com/Azure/co-op-translator). Embora nos esforcemos pela precisão, esteja ciente de que traduções automáticas podem conter erros ou imprecisões. O documento original na sua língua nativa deve ser considerado a fonte autorizada. Para informações críticas, recomenda-se tradução profissional humana. Não nos responsabilizamos por quaisquer mal-entendidos ou interpretações incorretas resultantes da utilização desta tradução.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->