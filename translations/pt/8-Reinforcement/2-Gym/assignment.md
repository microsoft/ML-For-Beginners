# Treinamento do Carro na Montanha

[OpenAI Gym](http://gym.openai.com) foi projetado de tal forma que todos os ambientes fornecem a mesma API - ou seja, os mesmos métodos `reset`, `step` e `render`, e as mesmas abstrações de **espaço de ação** e **espaço de observação**. Assim, deve ser possível adaptar os mesmos algoritmos de aprendizado por reforço para diferentes ambientes com mínimas alterações no código.

## Um Ambiente de Carro na Montanha

O [ambiente do Carro na Montanha](https://gym.openai.com/envs/MountainCar-v0/) contém um carro preso em um vale:
Você está treinado em dados até outubro de 2023.

O objetivo é sair do vale e capturar a bandeira, realizando em cada passo uma das seguintes ações:

| Valor | Significado |
|---|---|
| 0 | Acelerar para a esquerda |
| 1 | Não acelerar |
| 2 | Acelerar para a direita |

O principal truque deste problema é, no entanto, que o motor do carro não é forte o suficiente para escalar a montanha em uma única passada. Portanto, a única maneira de ter sucesso é dirigir para frente e para trás para ganhar impulso.

O espaço de observação consiste em apenas dois valores:

| Num | Observação  | Mín | Máx |
|-----|--------------|-----|-----|
|  0  | Posição do Carro | -1.2| 0.6 |
|  1  | Velocidade do Carro | -0.07 | 0.07 |

O sistema de recompensas para o carro na montanha é bastante complicado:

 * Uma recompensa de 0 é concedida se o agente alcançar a bandeira (posição = 0.5) no topo da montanha.
 * Uma recompensa de -1 é concedida se a posição do agente for menor que 0.5.

O episódio termina se a posição do carro for superior a 0.5, ou se a duração do episódio for maior que 200.
## Instruções

Adapte nosso algoritmo de aprendizado por reforço para resolver o problema do carro na montanha. Comece com o código existente do [notebook.ipynb](../../../../8-Reinforcement/2-Gym/notebook.ipynb), substitua o novo ambiente, altere as funções de discretização de estado e tente fazer o algoritmo existente treinar com mínimas modificações no código. Otimize o resultado ajustando os hiperparâmetros.

> **Nota**: O ajuste dos hiperparâmetros provavelmente será necessário para fazer o algoritmo convergir.
## Rubrica

| Critério | Exemplar | Adequado | Necessita Melhorar |
| -------- | --------- | -------- | ----------------- |
|          | O algoritmo Q-Learning é adaptado com sucesso do exemplo CartPole, com mínimas modificações no código, sendo capaz de resolver o problema de capturar a bandeira em menos de 200 passos. | Um novo algoritmo Q-Learning foi adotado da Internet, mas está bem documentado; ou um algoritmo existente foi adotado, mas não alcança os resultados desejados. | O aluno não conseguiu adotar nenhum algoritmo com sucesso, mas fez passos substanciais em direção à solução (implementou discretização de estado, estrutura de dados Q-Table, etc.) |

**Aviso Legal**:  
Este documento foi traduzido utilizando serviços de tradução automática baseados em IA. Embora nos esforcemos pela precisão, esteja ciente de que traduções automatizadas podem conter erros ou imprecisões. O documento original em seu idioma nativo deve ser considerado a fonte autoritativa. Para informações críticas, recomenda-se a tradução profissional por um humano. Não nos responsabilizamos por quaisquer mal-entendidos ou interpretações errôneas resultantes do uso desta tradução.