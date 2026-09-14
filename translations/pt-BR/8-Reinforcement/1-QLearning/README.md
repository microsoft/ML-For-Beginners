# Introdução ao Aprendizado por Reforço e Q-Learning

![Resumo do reforço em aprendizado de máquina em um sketchnote](../../../../translated_images/pt-BR/ml-reinforcement.94024374d63348db.webp)
> Sketchnote por [Tomomi Imura](https://www.twitter.com/girlie_mac)

O aprendizado por reforço envolve três conceitos importantes: o agente, alguns estados e um conjunto de ações por estado. Ao executar uma ação em um estado especificado, o agente recebe uma recompensa. Imagine novamente o jogo de computador Super Mario. Você é o Mario, está em um nível do jogo, parado ao lado de um penhasco. Acima de você há uma moeda. Você sendo o Mario, em um nível do jogo, em uma posição específica... esse é o seu estado. Mover um passo para a direita (uma ação) o fará cair do penhasco, e isso lhe daria uma pontuação numérica baixa. No entanto, apertar o botão de pular permitiria que você marcasse um ponto e continuasse vivo. Esse é um resultado positivo e isso deve recompensá-lo com uma pontuação numérica positiva.

Usando o aprendizado por reforço e um simulador (o jogo), você pode aprender a jogar o jogo para maximizar a recompensa, que é permanecer vivo e marcar o máximo de pontos possível.

[![Introdução ao Aprendizado por Reforço](https://img.youtube.com/vi/lDq_en8RNOo/0.jpg)](https://www.youtube.com/watch?v=lDq_en8RNOo)

> 🎥 Clique na imagem acima para ouvir Dmitry falar sobre Aprendizado por Reforço

## [Quiz pré-aula](https://ff-quizzes.netlify.app/en/ml/)

## Pré-requisitos e Configuração

Nesta lição, experimentaremos alguns códigos em Python. Você deve ser capaz de executar o código do Jupyter Notebook desta lição, seja no seu computador ou em algum lugar na nuvem.

Você pode abrir [o notebook da lição](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/notebook.ipynb) e acompanhar esta lição para construir.

> **Nota:** Se você estiver abrindo este código na nuvem, também precisa obter o arquivo [`rlboard.py`](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/rlboard.py), que é usado no código do notebook. Adicione-o ao mesmo diretório do notebook.

## Introdução

Nesta lição, exploraremos o mundo de **[Pedro e o Lobo](https://en.wikipedia.org/wiki/Peter_and_the_Wolf)**, inspirado em um conto musical de um compositor russo, [Sergei Prokofiev](https://en.wikipedia.org/wiki/Sergei_Prokofiev). Usaremos **Aprendizado por Reforço** para permitir que Pedro explore seu ambiente, colete maçãs saborosas e evite encontrar o lobo.

**Aprendizado por Reforço** (RL) é uma técnica de aprendizado que nos permite aprender um comportamento ideal de um **agente** em algum **ambiente** realizando muitos experimentos. Um agente nesse ambiente deve ter algum **objetivo**, definido por uma **função de recompensa**.

## O ambiente

Para simplificar, vamos considerar o mundo de Pedro como um tabuleiro quadrado de tamanho `width` x `height`, assim:

![Ambiente de Pedro](../../../../translated_images/pt-BR/environment.40ba3cb66256c93f.webp)

Cada célula neste tabuleiro pode ser:

* **terra**, onde Pedro e outras criaturas podem andar.
* **água**, onde obviamente não se pode andar.
* uma **árvore** ou **grama**, lugar para descansar.
* uma **maçã**, que representa algo que Pedro ficaria feliz em encontrar para se alimentar.
* um **lobo**, que é perigoso e deve ser evitado.

Há um módulo Python separado, [`rlboard.py`](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/rlboard.py), que contém o código para trabalhar com esse ambiente. Como esse código não é importante para compreendermos os conceitos, importaremos o módulo e usaremos para criar o tabuleiro exemplo (bloco de código 1):

```python
from rlboard import *

width, height = 8,8
m = Board(width,height)
m.randomize(seed=13)
m.plot()
```

Este código deve imprimir uma imagem do ambiente semelhante à acima.

## Ações e política

No nosso exemplo, o objetivo de Pedro seria encontrar uma maçã, evitando o lobo e outros obstáculos. Para isso, ele pode basicamente andar por aí até encontrar uma maçã.

Portanto, em qualquer posição, ele pode escolher entre uma das seguintes ações: para cima, para baixo, para a esquerda e para a direita.

Definiremos essas ações como um dicionário, mapeando-as para pares correspondentes de mudanças nas coordenadas. Por exemplo, mover para a direita (`R`) corresponderia ao par `(1,0)`. (bloco de código 2):

```python
actions = { "U" : (0,-1), "D" : (0,1), "L" : (-1,0), "R" : (1,0) }
action_idx = { a : i for i,a in enumerate(actions.keys()) }
```

Para resumir, a estratégia e o objetivo deste cenário são os seguintes:

- **A estratégia**, do nosso agente (Pedro) é definida por uma chamada **política**. Uma política é uma função que retorna a ação em qualquer estado dado. No nosso caso, o estado do problema é representado pelo tabuleiro, incluindo a posição atual do jogador.

- **O objetivo**, do aprendizado por reforço é eventualmente aprender uma boa política que nos permita resolver o problema eficientemente. Entretanto, como referência, consideremos a política mais simples, chamada **random walk** (caminhada aleatória).

## Caminhada aleatória

Vamos primeiro resolver nosso problema implementando uma estratégia de caminhada aleatória. Com a caminhada aleatória, escolheremos aleatoriamente a próxima ação entre as ações permitidas, até alcançar a maçã (bloco de código 3).

1. Implemente a caminhada aleatória com o código abaixo:

    ```python
    def random_policy(m):
        return random.choice(list(actions))
    
    def walk(m,policy,start_position=None):
        n = 0 # número de etapas
        # definir posição inicial
        if start_position:
            m.human = start_position 
        else:
            m.random_start()
        while True:
            if m.at() == Board.Cell.apple:
                return n # sucesso!
            if m.at() in [Board.Cell.wolf, Board.Cell.water]:
                return -1 # comido pelo lobo ou afogado
            while True:
                a = actions[policy(m)]
                new_pos = m.move_pos(m.human,a)
                if m.is_valid(new_pos) and m.at(new_pos)!=Board.Cell.water:
                    m.move(a) # fazer o movimento real
                    break
            n+=1
    
    walk(m,random_policy)
    ```

    A chamada para `walk` deve retornar o comprimento do caminho correspondente, que pode variar de uma execução para outra.

1. Execute o experimento da caminhada várias vezes (digamos, 100), e imprima as estatísticas resultantes (bloco de código 4):

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

    Note que o comprimento médio de um caminho fica em torno de 30-40 passos, o que é bastante, dado que a distância média até a maçã mais próxima é de cerca de 5-6 passos.

    Você também pode ver como é o movimento de Pedro durante a caminhada aleatória:

    ![Caminhada Aleatória de Pedro](../../../../8-Reinforcement/1-QLearning/images/random_walk.gif)

## Função de recompensa

Para tornar nossa política mais inteligente, precisamos entender quais movimentos são "melhores" que outros. Para isso, precisamos definir nosso objetivo.

O objetivo pode ser definido em termos de uma **função de recompensa**, que retornará um valor de pontuação para cada estado. Quanto maior o número, melhor a função de recompensa. (bloco de código 5)

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

Uma coisa interessante sobre funções de recompensa é que na maioria dos casos, *somos recompensados substancialmente apenas ao final do jogo*. Isso significa que nosso algoritmo deve de alguma forma lembrar dos passos "bons" que levam a uma recompensa positiva no final, e aumentar sua importância. Similarmente, todos os movimentos que levam a resultados ruins devem ser desencorajados.

## Q-Learning

Um algoritmo que discutiremos aqui chama-se **Q-Learning**. Nesse algoritmo, a política é definida por uma função (ou uma estrutura de dados) chamada **Q-Table**. Ela registra a "qualidade" de cada ação em um estado dado.

Ela é chamada de Q-Table porque é conveniente representá-la como uma tabela ou matriz multidimensional. Como nosso tabuleiro tem dimensões `width` x `height`, podemos representar a Q-Table usando um array numpy com formato `width` x `height` x `len(actions)`: (bloco de código 6)

```python
Q = np.ones((width,height,len(actions)),dtype=np.float)*1.0/len(actions)
```

Note que inicializamos todos os valores da Q-Table com um valor igual, no nosso caso - 0,25. Isso corresponde à política de "caminhada aleatória", porque todos os movimentos em cada estado são igualmente bons. Podemos passar a Q-Table para a função `plot` para visualizar a tabela no tabuleiro: `m.plot(Q)`.

![Ambiente de Pedro](../../../../translated_images/pt-BR/env_init.04e8f26d2d60089e.webp)

No centro de cada célula há uma "seta" que indica a direção preferida de movimento. Como todas as direções são iguais, um ponto é exibido.

Agora precisamos executar a simulação, explorar nosso ambiente e aprender uma melhor distribuição dos valores da Q-Table, o que nos permitirá encontrar o caminho para a maçã muito mais rápido.

## Essência do Q-Learning: Equação de Bellman

Uma vez que começamos a nos mover, cada ação terá uma recompensa correspondente, ou seja, teoricamente podemos selecionar a próxima ação com base na recompensa imediata maior. Contudo, na maioria dos estados, o movimento não alcançará nosso objetivo de chegar à maçã, e portanto não podemos decidir imediatamente qual direção é melhor.

> Lembre-se que não é o resultado imediato que importa, mas sim o resultado final, que obteremos ao final da simulação.

Para levar em conta essa recompensa retardada, precisamos usar os princípios da **[programação dinâmica](https://en.wikipedia.org/wiki/Dynamic_programming)**, que nos permitem pensar em nosso problema recursivamente.

Suponha que estamos agora no estado *s*, e queremos mover para o próximo estado *s'*. Fazendo isso, receberemos a recompensa imediata *r(s,a)*, definida pela função de recompensa, além de alguma recompensa futura. Se supusermos que nossa Q-Table reflete corretamente a "atratividade" de cada ação, então no estado *s'* escolheremos uma ação *a* que corresponde ao valor máximo de *Q(s',a')*. Assim, a melhor recompensa futura possível que poderíamos obter no estado *s* será definida como `max`<sub>a'</sub>*Q(s',a')* (o máximo aqui é calculado sobre todas as ações possíveis *a'* no estado *s'*).

Isso dá a **fórmula de Bellman** para calcular o valor da Q-Table no estado *s*, dada a ação *a*:

<img src="../../../../translated_images/pt-BR/bellman-equation.7c0c4c722e5a6b7c.webp"/>

Aqui γ é o chamado **fator de desconto** que determina em que medida você deve preferir a recompensa atual sobre a recompensa futura e vice-versa.

## Algoritmo de Aprendizado

Dada a equação acima, podemos agora escrever o pseudo-código para nosso algoritmo de aprendizado:

* Inicialize a Q-Table Q com números iguais para todos os estados e ações
* Defina a taxa de aprendizado α ← 1
* Repita a simulação várias vezes
   1. Comece em uma posição aleatória
   1. Repita
        1. Selecione uma ação *a* no estado *s*
        2. Execute a ação movendo para um novo estado *s'*
        3. Se encontrarmos condição de fim de jogo, ou a recompensa total for muito pequena - saia da simulação  
        4. Calcule a recompensa *r* no novo estado
        5. Atualize a função Q segundo a equação de Bellman: *Q(s,a)* ← *(1-α)Q(s,a)+α(r+γ max<sub>a'</sub>Q(s',a'))*
        6. *s* ← *s'*
        7. Atualize a recompensa total e diminua α.

## Explorar vs. Explorar

No algoritmo acima, não especificamos exatamente como devemos escolher uma ação no passo 2.1. Se escolhermos a ação aleatoriamente, exploraremos o ambiente aleatoriamente, e provavelmente morreremos com frequência, além de explorar áreas onde normalmente não iríamos. Uma abordagem alternativa seria **explorar** os valores da Q-Table que já conhecemos, escolhendo assim a melhor ação (com maior valor na Q-Table) no estado *s*. Isso, entretanto, evitará que exploremos outros estados, e provavelmente não encontraremos a solução ótima.

Assim, a melhor abordagem é encontrar um equilíbrio entre exploração e exploração. Isso pode ser feito escolhendo a ação no estado *s* com probabilidades proporcionais aos valores na Q-Table. No começo, quando os valores da Q-Table são todos iguais, isso corresponderia a uma seleção aleatória, mas à medida que aprendemos mais sobre nosso ambiente, teremos mais chances de seguir a rota ótima, permitindo que o agente escolha o caminho não explorado de vez em quando.

## Implementação em Python

Estamos agora prontos para implementar o algoritmo de aprendizado. Antes disso, também precisamos de uma função que converta números arbitrários na Q-Table em um vetor de probabilidades para as ações correspondentes.

1. Crie uma função `probs()`:

    ```python
    def probs(v,eps=1e-4):
        v = v-v.min()+eps
        v = v/v.sum()
        return v
    ```

    Adicionamos alguns `eps` ao vetor original para evitar divisão por 0 no caso inicial, quando todos os componentes do vetor são idênticos.

Execute o algoritmo de aprendizado por 5000 experimentos, também chamados de **épocas**: (bloco de código 8)
```python
    for epoch in range(5000):
    
        # Escolher ponto inicial
        m.random_start()
        
        # Começar a viajar
        n=0
        cum_reward = 0
        while True:
            x,y = m.human
            v = probs(Q[x,y])
            a = random.choices(list(actions),weights=v)[0]
            dpos = actions[a]
            m.move(dpos,check_correctness=False) # permitimos que o jogador se mova fora do tabuleiro, o que termina o episódio
            r = reward(m)
            cum_reward += r
            if r==end_reward or cum_reward < -1000:
                lpath.append(n)
                break
            alpha = np.exp(-n / 10e5)
            gamma = 0.5
            ai = action_idx[a]
            Q[x,y,ai] = (1 - alpha) * Q[x,y,ai] + alpha * (r + gamma * Q[x+dpos[0], y+dpos[1]].max())
            n+=1
```

Depois de executar este algoritmo, a Q-Table deve estar atualizada com valores que definem a atratividade das diferentes ações em cada passo. Podemos tentar visualizar a Q-Table desenhando um vetor em cada célula que apontará na direção desejada de movimento. Para simplificar, desenhamos um pequeno círculo ao invés da ponta da seta.

<img src="../../../../translated_images/pt-BR/learned.ed28bcd8484b5287.webp"/>

## Verificando a política

Como a Q-Table lista a "atratividade" de cada ação em cada estado, é bastante fácil usá-la para definir a navegação eficiente em nosso mundo. No caso mais simples, podemos selecionar a ação que corresponde ao maior valor na Q-Table: (bloco de código 9)

```python
def qpolicy_strict(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = list(actions)[np.argmax(v)]
        return a

walk(m,qpolicy_strict)
```


> Se você tentar o código acima várias vezes, pode notar que às vezes ele "trava", e você precisa pressionar o botão PARAR no notebook para interrompê-lo. Isso acontece porque podem haver situações em que dois estados "apontam" um para o outro em termos de valor Q ótimo, e, nesse caso, o agente acaba se movendo entre esses estados indefinidamente.

## 🚀Desafio

> **Tarefa 1:** Modifique a função `walk` para limitar o comprimento máximo do caminho por um certo número de passos (digamos, 100), e observe o código acima retornar este valor de tempos em tempos.

> **Tarefa 2:** Modifique a função `walk` para que ela não volte aos lugares onde já esteve anteriormente. Isso evitará que `walk` entre em loop, porém, o agente ainda pode acabar "preso" em um local do qual não consegue escapar.

## Navegação

Uma política de navegação melhor seria aquela que usamos durante o treinamento, que combina exploração e aproveitamento. Nessa política, selecionaremos cada ação com uma certa probabilidade, proporcional aos valores na Q-Table. Essa estratégia ainda pode resultar no agente retornar a uma posição que já explorou, mas, como você pode ver no código abaixo, resulta em um caminho médio muito curto até o local desejado (lembre-se que `print_statistics` executa a simulação 100 vezes): (bloco de código 10)

```python
def qpolicy(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = random.choices(list(actions),weights=v)[0]
        return a

print_statistics(qpolicy)
```

Após executar este código, você deve obter um comprimento médio de caminho muito menor do que antes, na faixa de 3 a 6.

## Investigando o processo de aprendizagem

Como mencionamos, o processo de aprendizagem é um equilíbrio entre exploração e aproveitamento do conhecimento adquirido sobre a estrutura do espaço de problema. Vimos que os resultados da aprendizagem (a habilidade de ajudar um agente a encontrar um caminho curto até o objetivo) melhoraram, mas também é interessante observar como o comprimento médio do caminho se comporta durante o processo de aprendizagem:

<img src="../../../../translated_images/pt-BR/lpathlen1.0534784add58d4eb.webp"/>

O aprendizado pode ser resumido em:

- **O comprimento médio do caminho aumenta**. O que vemos aqui é que, no começo, o comprimento médio do caminho aumenta. Isso provavelmente ocorre porque, quando não sabemos nada sobre o ambiente, é provável que fiquemos presos em estados ruins, água ou lobo. Conforme aprendemos mais e começamos a usar esse conhecimento, podemos explorar o ambiente por mais tempo, mas ainda não sabemos muito bem onde estão as maçãs.

- **O comprimento do caminho diminui, conforme aprendemos mais**. Quando aprendemos o suficiente, torna-se mais fácil para o agente atingir o objetivo, e o comprimento do caminho começa a diminuir. No entanto, ainda estamos abertos à exploração, então frequentemente nos desviamos do melhor caminho e exploramos novas opções, deixando o caminho mais longo do que o ideal.

- **O comprimento aumenta abruptamente**. O que também observamos nesse gráfico é que, em algum momento, o comprimento aumentou abruptamente. Isso indica a natureza estocástica do processo, e que em algum ponto podemos "estragar" os coeficientes da Q-Table sobrescrevendo-os com novos valores. Isso idealmente deveria ser minimizado diminuindo a taxa de aprendizado (por exemplo, próximo ao fim do treinamento, ajustamos os valores da Q-Table por um valor pequeno).

No geral, é importante lembrar que o sucesso e a qualidade do processo de aprendizagem dependem significativamente dos parâmetros, como taxa de aprendizado, decaimento da taxa de aprendizado e fator de desconto. Eles são frequentemente chamados de **hiperparâmetros**, para distingui-los dos **parâmetros**, que otimizamos durante o treinamento (por exemplo, coeficientes da Q-Table). O processo de encontrar os melhores valores de hiperparâmetros é chamado de **otimização de hiperparâmetros**, e merece um tópico separado.

## [Quiz pós-aula](https://ff-quizzes.netlify.app/en/ml/)

## Exercício
[Um Mundo Mais Realista](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Aviso Legal**:
Este documento foi traduzido usando o serviço de tradução por IA [Co-op Translator](https://github.com/Azure/co-op-translator). Embora nos esforcemos pela precisão, por favor, esteja ciente de que traduções automatizadas podem conter erros ou imprecisões. O documento original em seu idioma nativo deve ser considerado a fonte autorizada. Para informações críticas, recomenda-se tradução profissional humana. Não nos responsabilizamos por quaisquer mal-entendidos ou interpretações incorretas decorrentes do uso desta tradução.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->