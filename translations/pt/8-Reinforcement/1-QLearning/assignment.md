# Um Mundo Mais Realista

Na nossa situação, Peter conseguia se mover quase sem ficar cansado ou com fome. Em um mundo mais realista, ele precisaria sentar e descansar de tempos em tempos, além de se alimentar. Vamos tornar nosso mundo mais realista, implementando as seguintes regras:

1. Ao se mover de um lugar para outro, Peter perde **energia** e ganha um pouco de **fadiga**.
2. Peter pode ganhar mais energia comendo maçãs.
3. Peter pode se livrar da fadiga descansando debaixo da árvore ou na grama (ou seja, caminhando para um local com uma árvore ou grama - campo verde).
4. Peter precisa encontrar e matar o lobo.
5. Para matar o lobo, Peter precisa ter certos níveis de energia e fadiga; caso contrário, ele perde a batalha.

## Instruções

Use o notebook original [notebook.ipynb](../../../../8-Reinforcement/1-QLearning/notebook.ipynb) como ponto de partida para sua solução.

Modifique a função de recompensa acima de acordo com as regras do jogo, execute o algoritmo de aprendizado por reforço para aprender a melhor estratégia para vencer o jogo e compare os resultados do passeio aleatório com seu algoritmo em termos de número de jogos ganhos e perdidos.

> **Nota**: Em seu novo mundo, o estado é mais complexo e, além da posição humana, também inclui níveis de fadiga e energia. Você pode optar por representar o estado como uma tupla (Board, energia, fadiga), ou definir uma classe para o estado (você também pode querer derivá-la de `Board`), ou até mesmo modificar a classe original `Board` dentro de [rlboard.py](../../../../8-Reinforcement/1-QLearning/rlboard.py).

Em sua solução, mantenha o código responsável pela estratégia de passeio aleatório e compare os resultados do seu algoritmo com o passeio aleatório no final.

> **Nota**: Você pode precisar ajustar os hiperparâmetros para que funcione, especialmente o número de épocas. Como o sucesso do jogo (lutando contra o lobo) é um evento raro, você pode esperar um tempo de treinamento muito mais longo.

## Rubrica

| Critérios | Exemplar                                                                                                                                                                                               | Adequado                                                                                                                                                                                 | Necessita Melhorias                                                                                                                        |
| --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
|           | Um notebook é apresentado com a definição das novas regras do mundo, algoritmo Q-Learning e algumas explicações textuais. O Q-Learning consegue melhorar significativamente os resultados em comparação ao passeio aleatório. | O notebook é apresentado, o Q-Learning é implementado e melhora os resultados em comparação ao passeio aleatório, mas não de forma significativa; ou o notebook é mal documentado e o código não é bem estruturado. | Alguma tentativa de redefinir as regras do mundo foi feita, mas o algoritmo Q-Learning não funciona, ou a função de recompensa não está totalmente definida. |

**Isenção de responsabilidade**:  
Este documento foi traduzido utilizando serviços de tradução automática baseados em IA. Embora nos esforcemos pela precisão, esteja ciente de que traduções automatizadas podem conter erros ou imprecisões. O documento original em sua língua nativa deve ser considerado a fonte autoritativa. Para informações críticas, recomenda-se a tradução profissional por um humano. Não nos responsabilizamos por quaisquer mal-entendidos ou interpretações errôneas decorrentes do uso desta tradução.