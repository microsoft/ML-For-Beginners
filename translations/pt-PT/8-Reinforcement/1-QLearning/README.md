# Introdução ao Aprendizagem por Reforço e Q-Learning

![Resumo do reforço em aprendizagem de máquina numa sketchnote](../../../../translated_images/pt-PT/ml-reinforcement.94024374d63348db.webp)
> Sketchnote por [Tomomi Imura](https://www.twitter.com/girlie_mac)

A aprendizagem por reforço envolve três conceitos importantes: o agente, alguns estados, e um conjunto de ações por estado. Ao executar uma ação num estado especificado, o agente recebe uma recompensa. Imagine novamente o jogo de computador Super Mario. Você é o Mario, está num nível do jogo, parado junto a um penhasco. Acima de si está uma moeda. Você sendo o Mario, num nível de jogo, numa posição específica ... esse é o seu estado. Mover-se um passo para a direita (uma ação) levará você para além do penhasco, e isso dar-lhe-ia uma pontuação numérica baixa. No entanto, premir o botão de pular permitiria que ganhou um ponto e que continuasse vivo. Esse é um resultado positivo e deveria atribuir-lhe uma pontuação numérica positiva.

Usando aprendizagem por reforço e um simulador (o jogo), pode aprender a jogar para maximizar a recompensa, que é manter-se vivo e marcar o maior número de pontos possível.

[![Introdução à Aprendizagem por Reforço](https://img.youtube.com/vi/lDq_en8RNOo/0.jpg)](https://www.youtube.com/watch?v=lDq_en8RNOo)

> 🎥 Clique na imagem acima para ouvir Dmitry discutir Aprendizagem por Reforço

## [Quiz pré-aula](https://ff-quizzes.netlify.app/en/ml/)

## Pré-requisitos e Configuração

Nesta lição, vamos experimentar algum código em Python. Deverá ser capaz de executar o código do Jupyter Notebook desta lição, seja no seu computador ou em algum lugar na nuvem.

Pode abrir [o notebook da lição](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/notebook.ipynb) e seguir esta lição para desenvolver.

> **Nota:** Se estiver a abrir este código a partir da nuvem, também precisa de obter o ficheiro [`rlboard.py`](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/rlboard.py), que é usado no código do notebook. Adicione-o ao mesmo diretório do notebook.

## Introdução

Nesta lição, vamos explorar o mundo de **[Pedro e o Lobo](https://en.wikipedia.org/wiki/Peter_and_the_Wolf)**, inspirado por um conto musical de um compositor russo, [Sergei Prokofiev](https://en.wikipedia.org/wiki/Sergei_Prokofiev). Usaremos **Aprendizagem por Reforço** para deixar que Pedro explore o seu ambiente, recolha maçãs saborosas e evite encontrar o lobo.

**Aprendizagem por Reforço** (RL) é uma técnica de aprendizagem que nos permite aprender um comportamento ótimo de um **agente** num **ambiente** ao realizar muitos experimentos. Um agente neste ambiente deve ter algum **objetivo**, definido por uma **função de recompensa**.

## O ambiente

Para simplificar, consideremos que o mundo de Pedro é um tabuleiro quadrado com tamanho `width` x `height`, como este:

![Ambiente de Pedro](../../../../translated_images/pt-PT/environment.40ba3cb66256c93f.webp)

Cada célula deste tabuleiro pode ser:

* **terra**, sobre a qual Pedro e outras criaturas podem andar.
* **água**, sobre a qual obviamente não se pode andar.
* uma **árvore** ou **relva**, um local onde se pode descansar.
* uma **maçã**, que representa algo que Pedro ficaria feliz por encontrar para se alimentar.
* um **lobo**, que é perigoso e deve ser evitado.

Existe um módulo Python separado, [`rlboard.py`](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/rlboard.py), que contém o código para trabalhar com este ambiente. Como este código não é importante para entender os nossos conceitos, vamos importar o módulo e usá-lo para criar o tabuleiro de exemplo (bloco de código 1):

```python
from rlboard import *

width, height = 8,8
m = Board(width,height)
m.randomize(seed=13)
m.plot()
```

Este código deverá imprimir uma imagem do ambiente semelhante à acima.

## Ações e política

No nosso exemplo, o objetivo de Pedro seria conseguir encontrar uma maçã, enquanto evita o lobo e outros obstáculos. Para isso, ele pode essencialmente andar até encontrar uma maçã.

Portanto, em qualquer posição, ele pode escolher entre uma das seguintes ações: subir, descer, esquerda e direita.

Vamos definir essas ações como um dicionário, e mapear-las para pares de alterações correspondentes nas coordenadas. Por exemplo, mover para a direita (`R`) corresponderia ao par `(1,0)`. (bloco de código 2):

```python
actions = { "U" : (0,-1), "D" : (0,1), "L" : (-1,0), "R" : (1,0) }
action_idx = { a : i for i,a in enumerate(actions.keys()) }
```

Para resumir, a estratégia e objetivo deste cenário são os seguintes:

- **A estratégia**, do nosso agente (Pedro) é definida por uma chamada **política**. Uma política é uma função que retorna a ação em qualquer estado dado. No nosso caso, o estado do problema é representado pelo tabuleiro, incluindo a posição atual do jogador.

- **O objetivo**, da aprendizagem por reforço é finalmente aprender uma boa política que nos permita resolver o problema eficientemente. No entanto, como linha base, vamos considerar a política mais simples chamada **passeio aleatório**.

## Passeio aleatório

Primeiro vamos resolver o nosso problema implementando a estratégia do passeio aleatório. Com passeio aleatório, escolhemos aleatoriamente a próxima ação entre as ações permitidas, até atingirmos a maçã (bloco de código 3).

1. Implemente o passeio aleatório com o código abaixo:

    ```python
    def random_policy(m):
        return random.choice(list(actions))
    
    def walk(m,policy,start_position=None):
        n = 0 # número de passos
        # definir posição inicial
        if start_position:
            m.human = start_position 
        else:
            m.random_start()
        while True:
            if m.at() == Board.Cell.apple:
                return n # sucesso!
            if m.at() in [Board.Cell.wolf, Board.Cell.water]:
                return -1 # comido por lobo ou afogado
            while True:
                a = actions[policy(m)]
                new_pos = m.move_pos(m.human,a)
                if m.is_valid(new_pos) and m.at(new_pos)!=Board.Cell.water:
                    m.move(a) # executar o movimento real
                    break
            n+=1
    
    walk(m,random_policy)
    ```

    A chamada a `walk` deve retornar o comprimento do percurso correspondente, que pode variar de uma execução para outra.

1. Execute o experimento do passeio várias vezes (digamos, 100), e imprima as estatísticas resultantes (bloco de código 4):

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

    Note que o comprimento médio de um percurso é cerca de 30-40 passos, o que é bastante, dado o facto de que a distância média até à maçã mais próxima é cerca de 5-6 passos.

    Pode também ver como é o movimento de Pedro durante o passeio aleatório:

    ![Passeio aleatório de Pedro](../../../../8-Reinforcement/1-QLearning/images/random_walk.gif)

## Função de recompensa

Para tornar a nossa política mais inteligente, precisamos de entender quais os movimentos que são "melhores" do que outros. Para isso, precisamos definir o nosso objetivo.

O objetivo pode ser definido em termos de uma **função de recompensa**, que vai retornar um valor de pontuação para cada estado. Quanto maior o número, melhor a função de recompensa. (bloco de código 5)

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

Uma coisa interessante sobre funções de recompensa é que na maioria dos casos, *só recebemos uma recompensa substancial no fim do jogo*. Isto significa que o nosso algoritmo deve de alguma forma lembrar-se dos passos "bons" que levam a uma recompensa positiva no fim, e aumentar sua importância. Da mesma forma, todos os movimentos que levam a resultados maus devem ser desencorajados.

## Q-Learning

Um algoritmo que vamos discutir aqui chama-se **Q-Learning**. Neste algoritmo, a política é definida por uma função (ou estrutura de dados) chamada **Q-Table**. Ela regista a "qualidade" de cada uma das ações num dado estado.

É chamada Q-Table porque é frequentemente conveniente representá-la como uma tabela, ou array multidimensional. Como o nosso tabuleiro tem dimensões `width` x `height`, podemos representar a Q-Table usando um array numpy com formato `width` x `height` x `len(actions)`: (bloco de código 6)

```python
Q = np.ones((width,height,len(actions)),dtype=np.float)*1.0/len(actions)
```

Note que inicializamos todos os valores da Q-Table com um valor igual, no nosso caso - 0.25. Isto corresponde à política "passeio aleatório", porque todos os movimentos em cada estado são igualmente bons. Podemos passar a Q-Table para a função `plot` para visualizar a tabela no tabuleiro: `m.plot(Q)`.

![Ambiente de Pedro](../../../../translated_images/pt-PT/env_init.04e8f26d2d60089e.webp)

No centro de cada célula há uma "seta" que indica a direção preferida do movimento. Como todas as direções são iguais, é exibido um ponto.

Agora precisamos executar a simulação, explorar o nosso ambiente, e aprender uma melhor distribuição dos valores da Q-Table, o que nos permitirá encontrar o caminho para a maçã muito mais rápido.

## Essência do Q-Learning: Equação de Bellman

Assim que começamos a mover, cada ação terá uma recompensa correspondente, isto é, teoricamente podemos selecionar a próxima ação com base na recompensa imediata mais alta. No entanto, na maioria dos estados, o movimento não alcançará o nosso objetivo de chegar à maçã, e portanto não podemos decidir imediatamente qual direção é melhor.

> Lembre-se que não é o resultado imediato que importa, mas sim o resultado final, que será obtido no fim da simulação.

Para contabilizar essa recompensa retardada, precisamos usar os princípios da **[programação dinâmica](https://en.wikipedia.org/wiki/Dynamic_programming)**, que nos permitem pensar no nosso problema recursivamente.

Suponha que estamos agora no estado *s*, e queremos mover para o próximo estado *s'*. Ao fazê-lo, receberemos a recompensa imediata *r(s,a)*, definida pela função de recompensa, mais alguma recompensa futura. Se supusermos que a nossa Q-Table reflete corretamente a "atratividade" de cada ação, então no estado *s'* escolheremos uma ação *a* que corresponda ao valor máximo de *Q(s',a')*. Assim, a melhor recompensa futura possível que poderíamos obter no estado *s* será definida como `max`<sub>a'</sub>*Q(s',a')* (o máximo aqui é calculado sobre todas as possíveis ações *a'* no estado *s'*).

Isso dá a **fórmula de Bellman** para calcular o valor da Q-Table no estado *s*, dada a ação *a*:

<img src="../../../../translated_images/pt-PT/bellman-equation.7c0c4c722e5a6b7c.webp"/>

Aqui γ é o chamado **fator de desconto** que determina em que medida deve preferir a recompensa actual à recompensa futura e vice-versa.

## Algoritmo de aprendizagem

Dada a equação acima, agora podemos escrever pseudocódigo para o nosso algoritmo de aprendizagem:

* Inicialize a Q-Table Q com números iguais para todos os estados e ações
* Defina a taxa de aprendizagem α ← 1
* Repita a simulação muitas vezes
   1. Comece numa posição aleatória
   1. Repita
        1. Selecione uma ação *a* no estado *s*
        2. Execute ação movendo para um novo estado *s'*
        3. Se encontrarmos a condição de fim do jogo, ou a recompensa total for demasiado pequena - saia da simulação  
        4. Calcule a recompensa *r* no novo estado
        5. Atualize a função Q de acordo com a equação de Bellman: *Q(s,a)* ← *(1-α)Q(s,a)+α(r+γ max<sub>a'</sub>Q(s',a'))*
        6. *s* ← *s'*
        7. Atualize a recompensa total e diminua α.

## Explorar vs. Explorar

No algoritmo acima, não especificamos como exatamente devemos escolher uma ação no passo 2.1. Se escolhermos a ação aleatoriamente, iremos aleatoriamente **explorar** o ambiente, e é bastante provável que morramos frequentemente assim como exploremos áreas onde normalmente não iríamos. Uma abordagem alternativa seria **explorar** os valores da Q-Table que já conhecemos, e assim escolher a melhor ação (com maior valor na Q-Table) no estado *s*. Isso, no entanto, irá impedir-nos de explorar outros estados, e é provável que não encontremos a solução ótima.

Portanto, a melhor abordagem é encontrar um equilíbrio entre exploração e aproveitamento. Isto pode ser feito escolhendo a ação no estado *s* com probabilidades proporcionais aos valores na Q-Table. No início, quando os valores da Q-Table são todos iguais, isso corresponderia a uma seleção aleatória, mas à medida que aprendemos mais sobre o nosso ambiente, seremos mais propensos a seguir a rota ótima permitindo, contudo, que o agente escolha o caminho inexplorado de vez em quando.

## Implementação em Python

Estamos agora prontos para implementar o algoritmo de aprendizagem. Antes disso, também precisamos de uma função que converta números arbitrários na Q-Table num vetor de probabilidades para as ações correspondentes.

1. Crie uma função `probs()`:

    ```python
    def probs(v,eps=1e-4):
        v = v-v.min()+eps
        v = v/v.sum()
        return v
    ```

    Adicionamos alguns `eps` ao vetor original para evitar a divisão por 0 no caso inicial, quando todos os componentes do vetor são idênticos.

Execute o algoritmo de aprendizagem através de 5000 experiências, também chamadas de **épocas**: (bloco de código 8)
```python
    for epoch in range(5000):
    
        # Escolher ponto inicial
        m.random_start()
        
        # Começar a viagem
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

Depois de executar este algoritmo, a Q-Table deverá estar atualizada com valores que definem a atratividade das diferentes ações em cada passo. Podemos tentar visualizar a Q-Table ao desenhar um vetor em cada célula que aponta na direção desejada do movimento. Para simplificar, desenhamos um pequeno círculo em vez de uma ponta de seta.

<img src="../../../../translated_images/pt-PT/learned.ed28bcd8484b5287.webp"/>

## Verificação da política

Como a Q-Table lista a "atratividade" de cada ação em cada estado, é bastante fácil usá-la para definir a navegação eficiente no nosso mundo. No caso mais simples, podemos selecionar a ação correspondente ao maior valor na Q-Table: (bloco de código 9)

```python
def qpolicy_strict(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = list(actions)[np.argmax(v)]
        return a

walk(m,qpolicy_strict)
```


> Se experimentares o código acima várias vezes, poderás notar que às vezes ele "trava", e precisas de pressionar o botão STOP no notebook para o interromper. Isto acontece porque podem existir situações em que dois estados "apontam" um para o outro em termos do valor Q ótimo, caso em que o agente acaba a mover-se entre esses estados indefinidamente.

## 🚀Desafio

> **Tarefa 1:** Modifica a função `walk` para limitar o comprimento máximo do caminho a um certo número de passos (por exemplo, 100), e observa o código acima a devolver este valor de vez em quando.

> **Tarefa 2:** Modifica a função `walk` para que não volte aos locais onde já passou anteriormente. Isto impedirá que `walk` fique em loop, no entanto, o agente ainda pode acabar por ficar "preso" num local do qual não consiga escapar.

## Navegação

Uma política de navegação melhor seria aquela que usamos durante o treino, que combina exploração e aproveitamento. Nessa política, selecionaremos cada ação com uma certa probabilidade, proporcional aos valores na tabela Q. Esta estratégia ainda pode resultar no agente a voltar a uma posição que já tenha explorado, mas, como podes ver pelo código abaixo, resulta num caminho médio muito curto até ao local desejado (lembra-te que `print_statistics` executa a simulação 100 vezes): (code block 10)

```python
def qpolicy(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = random.choices(list(actions),weights=v)[0]
        return a

print_statistics(qpolicy)
```

Depois de executares este código, deverás obter um comprimento médio de caminho muito menor do que antes, na ordem dos 3-6.

## Investigando o processo de aprendizagem

Como mencionámos, o processo de aprendizagem é um equilíbrio entre a exploração e o aproveitamento do conhecimento adquirido sobre a estrutura do espaço do problema. Vimos que os resultados da aprendizagem (a capacidade de ajudar um agente a encontrar um caminho curto até ao objetivo) melhoraram, mas também é interessante observar como o comprimento médio do caminho se comporta durante o processo de aprendizagem:

<img src="../../../../translated_images/pt-PT/lpathlen1.0534784add58d4eb.webp"/>

As aprendizagens podem ser resumidas como:

- **O comprimento médio do caminho aumenta**. O que vemos aqui é que, inicialmente, o comprimento médio do caminho aumenta. Isto deve-se provavelmente ao facto de, quando nada sabemos sobre o ambiente, ser provável que fiquemos presos em estados maus, como água ou lobo. À medida que aprendemos mais e começamos a usar este conhecimento, podemos explorar o ambiente por mais tempo, mas ainda não sabemos muito bem onde estão as maçãs.

- **O comprimento do caminho diminui à medida que aprendemos mais**. Quando aprendemos o suficiente, torna-se mais fácil para o agente alcançar o objetivo, e o comprimento do caminho começa a diminuir. Contudo, ainda estamos abertos à exploração, por isso frequentemente desviamo-nos do melhor caminho e exploramos novas opções, tornando o caminho mais longo do que o ótimo.

- **O comprimento aumenta abruptamente**. O que também observamos neste gráfico é que, em determinado ponto, o comprimento aumentou abruptamente. Isto indica a natureza estocástica do processo, e que em algum momento podemos "estragar" os coeficientes da tabela Q ao sobrescrevê-los com novos valores. Idealmente, isto deveria ser minimizado ao diminuir a taxa de aprendizagem (por exemplo, para o final do treino, devemos ajustar os valores da tabela Q apenas por um valor pequeno).

No geral, é importante lembrar que o sucesso e a qualidade do processo de aprendizagem dependem significativamente dos parâmetros, como a taxa de aprendizagem, a diminuição da taxa de aprendizagem, e o fator de desconto. Eles são frequentemente chamados **hiperparâmetros**, para os distinguir dos **parâmetros**, que otimizamos durante o treino (por exemplo, coeficientes da tabela Q). O processo de encontrar os melhores valores para os hiperparâmetros chama-se **otimização de hiperparâmetros**, e merece um tópico separado.

## [Questionário pós-aula](https://ff-quizzes.netlify.app/en/ml/)

## Trabalho
[Um Mundo Mais Realista](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Aviso Legal**:
Este documento foi traduzido utilizando o serviço de tradução automática [Co-op Translator](https://github.com/Azure/co-op-translator). Embora nos esforcemos pela precisão, esteja ciente de que traduções automáticas podem conter erros ou imprecisões. O documento original na sua língua nativa deve ser considerado a fonte autorizada. Para informações críticas, recomenda-se tradução profissional humana. Não nos responsabilizamos por quaisquer mal-entendidos ou interpretações incorretas resultantes da utilização desta tradução.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->