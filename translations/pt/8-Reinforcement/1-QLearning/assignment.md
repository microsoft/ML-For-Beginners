<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "68394b2102d3503882e5e914bd0ff5c1",
  "translation_date": "2025-09-03T18:36:57+00:00",
  "source_file": "8-Reinforcement/1-QLearning/assignment.md",
  "language_code": "pt"
}
-->
# Um Mundo Mais Realista

Na nossa situação, Peter conseguia mover-se quase sem se cansar ou sentir fome. Num mundo mais realista, ele teria de sentar-se e descansar de vez em quando, além de alimentar-se. Vamos tornar o nosso mundo mais realista, implementando as seguintes regras:

1. Ao mover-se de um lugar para outro, Peter perde **energia** e ganha **fadiga**.
2. Peter pode recuperar energia ao comer maçãs.
3. Peter pode livrar-se da fadiga descansando debaixo de uma árvore ou na relva (ou seja, ao caminhar para uma localização no tabuleiro com uma árvore ou relva - campo verde).
4. Peter precisa encontrar e matar o lobo.
5. Para matar o lobo, Peter precisa ter certos níveis de energia e fadiga; caso contrário, ele perde a batalha.

## Instruções

Use o [notebook.ipynb](notebook.ipynb) original como ponto de partida para a sua solução.

Modifique a função de recompensa acima de acordo com as regras do jogo, execute o algoritmo de aprendizagem por reforço para aprender a melhor estratégia para vencer o jogo e compare os resultados da caminhada aleatória com o seu algoritmo em termos de número de jogos ganhos e perdidos.

> **Note**: No seu novo mundo, o estado é mais complexo e, além da posição do humano, também inclui os níveis de fadiga e energia. Pode optar por representar o estado como uma tupla (Tabuleiro, energia, fadiga), ou definir uma classe para o estado (também pode derivá-la de `Board`), ou até modificar a classe original `Board` dentro de [rlboard.py](../../../../8-Reinforcement/1-QLearning/rlboard.py).

Na sua solução, mantenha o código responsável pela estratégia de caminhada aleatória e compare os resultados do seu algoritmo com a caminhada aleatória no final.

> **Note**: Pode ser necessário ajustar os hiperparâmetros para que funcione, especialmente o número de épocas. Como o sucesso no jogo (lutar contra o lobo) é um evento raro, pode esperar um tempo de treino muito mais longo.

## Rubrica

| Critérios | Exemplar                                                                                                                                                                                             | Adequado                                                                                                                                                                                | Necessita Melhorar                                                                                                                          |
| --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
|           | Um notebook é apresentado com a definição das novas regras do mundo, o algoritmo de Q-Learning e algumas explicações textuais. O Q-Learning consegue melhorar significativamente os resultados em comparação com a caminhada aleatória. | Um notebook é apresentado, o Q-Learning é implementado e melhora os resultados em comparação com a caminhada aleatória, mas não significativamente; ou o notebook está mal documentado e o código não está bem estruturado. | Alguma tentativa de redefinir as regras do mundo foi feita, mas o algoritmo de Q-Learning não funciona, ou a função de recompensa não está totalmente definida. |

---

**Aviso Legal**:  
Este documento foi traduzido utilizando o serviço de tradução por IA [Co-op Translator](https://github.com/Azure/co-op-translator). Embora nos esforcemos para garantir a precisão, é importante notar que traduções automáticas podem conter erros ou imprecisões. O documento original na sua língua nativa deve ser considerado a fonte autoritária. Para informações críticas, recomenda-se a tradução profissional realizada por humanos. Não nos responsabilizamos por quaisquer mal-entendidos ou interpretações incorretas decorrentes da utilização desta tradução.