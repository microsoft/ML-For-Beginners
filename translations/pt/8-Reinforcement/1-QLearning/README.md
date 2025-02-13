## Verificando a política

Como a Q-Table lista a "atratividade" de cada ação em cada estado, é bastante fácil usá-la para definir a navegação eficiente em nosso mundo. No caso mais simples, podemos selecionar a ação correspondente ao maior valor da Q-Table: (código bloco 9)

```python
def qpolicy_strict(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = list(actions)[np.argmax(v)]
        return a

walk(m,qpolicy_strict)
```

> Se você tentar o código acima várias vezes, pode notar que às vezes ele "trava", e você precisa pressionar o botão PARAR no notebook para interrompê-lo. Isso acontece porque pode haver situações em que dois estados "apontam" um para o outro em termos de Q-Valor ótimo, nesse caso, o agente acaba se movendo entre esses estados indefinidamente.

## 🚀Desafio

> **Tarefa 1:** Modifique o `walk` function to limit the maximum length of path by a certain number of steps (say, 100), and watch the code above return this value from time to time.

> **Task 2:** Modify the `walk` function so that it does not go back to the places where it has already been previously. This will prevent `walk` from looping, however, the agent can still end up being "trapped" in a location from which it is unable to escape.

## Navigation

A better navigation policy would be the one that we used during training, which combines exploitation and exploration. In this policy, we will select each action with a certain probability, proportional to the values in the Q-Table. This strategy may still result in the agent returning back to a position it has already explored, but, as you can see from the code below, it results in a very short average path to the desired location (remember that `print_statistics` para executar a simulação 100 vezes): (código bloco 10)

```python
def qpolicy(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = random.choices(list(actions),weights=v)[0]
        return a

print_statistics(qpolicy)
```

Após executar este código, você deve obter um comprimento médio de caminho muito menor do que antes, na faixa de 3-6.

## Investigando o processo de aprendizado

Como mencionamos, o processo de aprendizado é um equilíbrio entre exploração e exploração do conhecimento adquirido sobre a estrutura do espaço do problema. Vimos que os resultados do aprendizado (a capacidade de ajudar um agente a encontrar um caminho curto para o objetivo) melhoraram, mas também é interessante observar como o comprimento médio do caminho se comporta durante o processo de aprendizado:

Os aprendizados podem ser resumidos como:

- **O comprimento médio do caminho aumenta**. O que vemos aqui é que, a princípio, o comprimento médio do caminho aumenta. Isso provavelmente se deve ao fato de que, quando não sabemos nada sobre o ambiente, é provável que fiquemos presos em estados ruins, água ou lobo. À medida que aprendemos mais e começamos a usar esse conhecimento, podemos explorar o ambiente por mais tempo, mas ainda não sabemos muito bem onde estão as maçãs.

- **O comprimento do caminho diminui, à medida que aprendemos mais**. Uma vez que aprendemos o suficiente, torna-se mais fácil para o agente alcançar o objetivo, e o comprimento do caminho começa a diminuir. No entanto, ainda estamos abertos à exploração, então muitas vezes nos afastamos do melhor caminho e exploramos novas opções, tornando o caminho mais longo do que o ideal.

- **Aumento abrupto do comprimento**. O que também observamos neste gráfico é que, em algum momento, o comprimento aumentou abruptamente. Isso indica a natureza estocástica do processo e que, em algum ponto, podemos "estragar" os coeficientes da Q-Table ao sobrescrevê-los com novos valores. Isso deve ser minimizado idealmente, diminuindo a taxa de aprendizado (por exemplo, no final do treinamento, ajustamos os valores da Q-Table apenas por um pequeno valor).

No geral, é importante lembrar que o sucesso e a qualidade do processo de aprendizado dependem significativamente de parâmetros, como taxa de aprendizado, decaimento da taxa de aprendizado e fator de desconto. Esses parâmetros são frequentemente chamados de **hiperparâmetros**, para distingui-los dos **parâmetros**, que otimizamos durante o treinamento (por exemplo, coeficientes da Q-Table). O processo de encontrar os melhores valores de hiperparâmetros é chamado de **otimização de hiperparâmetros**, e merece um tópico separado.

## [Quiz pós-aula](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/46/)

## Tarefa 
[Um Mundo Mais Realista](assignment.md)

**Aviso Legal**:  
Este documento foi traduzido utilizando serviços de tradução automática baseados em IA. Embora nos esforcemos pela precisão, esteja ciente de que traduções automatizadas podem conter erros ou imprecisões. O documento original em seu idioma nativo deve ser considerado a fonte autoritativa. Para informações críticas, recomenda-se a tradução profissional por humanos. Não nos responsabilizamos por quaisquer mal-entendidos ou interpretações errôneas decorrentes do uso desta tradução.