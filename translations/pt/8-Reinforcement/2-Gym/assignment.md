<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1f2b7441745eb52e25745423b247016b",
  "translation_date": "2025-09-03T18:44:00+00:00",
  "source_file": "8-Reinforcement/2-Gym/assignment.md",
  "language_code": "pt"
}
-->
# Treinar Carro na Montanha

[OpenAI Gym](http://gym.openai.com) foi projetado de forma que todos os ambientes fornecem a mesma API - ou seja, os mesmos métodos `reset`, `step` e `render`, e as mesmas abstrações de **espaço de ação** e **espaço de observação**. Assim, deve ser possível adaptar os mesmos algoritmos de aprendizagem por reforço a diferentes ambientes com mudanças mínimas no código.

## Um Ambiente de Carro na Montanha

O [ambiente Mountain Car](https://gym.openai.com/envs/MountainCar-v0/) contém um carro preso num vale:

O objetivo é sair do vale e capturar a bandeira, realizando em cada passo uma das seguintes ações:

| Valor | Significado |
|---|---|
| 0 | Acelerar para a esquerda |
| 1 | Não acelerar |
| 2 | Acelerar para a direita |

O principal truque deste problema, no entanto, é que o motor do carro não é forte o suficiente para escalar a montanha numa única tentativa. Portanto, a única maneira de ter sucesso é dirigir para frente e para trás para ganhar impulso.

O espaço de observação consiste em apenas dois valores:

| Num | Observação    | Min | Max |
|-----|--------------|-----|-----|
|  0  | Posição do carro | -1.2| 0.6 |
|  1  | Velocidade do carro | -0.07 | 0.07 |

O sistema de recompensa para o carro na montanha é bastante peculiar:

 * Uma recompensa de 0 é atribuída se o agente alcançar a bandeira (posição = 0.5) no topo da montanha.
 * Uma recompensa de -1 é atribuída se a posição do agente for menor que 0.5.

O episódio termina se a posição do carro for maior que 0.5 ou se o comprimento do episódio for superior a 200.

## Instruções

Adapte o nosso algoritmo de aprendizagem por reforço para resolver o problema do carro na montanha. Comece com o código existente no [notebook.ipynb](notebook.ipynb), substitua o ambiente, altere as funções de discretização de estado e tente fazer o algoritmo existente treinar com modificações mínimas no código. Otimize o resultado ajustando os hiperparâmetros.

> **Nota**: É provável que seja necessário ajustar os hiperparâmetros para que o algoritmo converja.

## Rubrica

| Critério | Exemplar | Adequado | Necessita Melhorias |
| -------- | --------- | -------- | ------------------- |
|          | O algoritmo de Q-Learning foi adaptado com sucesso a partir do exemplo CartPole, com modificações mínimas no código, sendo capaz de resolver o problema de capturar a bandeira em menos de 200 passos. | Um novo algoritmo de Q-Learning foi adotado da Internet, mas está bem documentado; ou o algoritmo existente foi adotado, mas não alcança os resultados desejados. | O estudante não conseguiu adotar nenhum algoritmo com sucesso, mas deu passos substanciais em direção à solução (implementou discretização de estado, estrutura de dados da Q-Table, etc.) |

---

**Aviso Legal**:  
Este documento foi traduzido utilizando o serviço de tradução por IA [Co-op Translator](https://github.com/Azure/co-op-translator). Embora nos esforcemos para garantir a precisão, esteja ciente de que traduções automáticas podem conter erros ou imprecisões. O documento original no seu idioma nativo deve ser considerado a fonte oficial. Para informações críticas, recomenda-se uma tradução profissional realizada por humanos. Não nos responsabilizamos por quaisquer mal-entendidos ou interpretações incorretas resultantes do uso desta tradução.