<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "68394b2102d3503882e5e914bd0ff5c1",
  "translation_date": "2025-08-29T22:10:51+00:00",
  "source_file": "8-Reinforcement/1-QLearning/assignment.md",
  "language_code": "br"
}
-->
# Um Mundo Mais Realista

Na nossa situação, Peter conseguia se mover quase sem se cansar ou sentir fome. Em um mundo mais realista, ele precisa sentar e descansar de tempos em tempos, além de se alimentar. Vamos tornar nosso mundo mais realista, implementando as seguintes regras:

1. Ao se mover de um lugar para outro, Peter perde **energia** e ganha **fadiga**.
2. Peter pode recuperar energia comendo maçãs.
3. Peter pode se livrar da fadiga descansando sob uma árvore ou na grama (ou seja, caminhando para uma posição no tabuleiro com uma árvore ou grama - campo verde).
4. Peter precisa encontrar e matar o lobo.
5. Para matar o lobo, Peter precisa ter certos níveis de energia e fadiga; caso contrário, ele perde a batalha.

## Instruções

Use o [notebook.ipynb](notebook.ipynb) original como ponto de partida para sua solução.

Modifique a função de recompensa acima de acordo com as regras do jogo, execute o algoritmo de aprendizado por reforço para aprender a melhor estratégia para vencer o jogo e compare os resultados do passeio aleatório com o seu algoritmo em termos de número de jogos ganhos e perdidos.

> **Note**: No seu novo mundo, o estado é mais complexo e, além da posição do humano, também inclui os níveis de fadiga e energia. Você pode optar por representar o estado como uma tupla (Tabuleiro, energia, fadiga), ou definir uma classe para o estado (você também pode querer derivá-la de `Board`), ou até mesmo modificar a classe `Board` original dentro de [rlboard.py](../../../../8-Reinforcement/1-QLearning/rlboard.py).

Na sua solução, mantenha o código responsável pela estratégia de passeio aleatório e compare os resultados do seu algoritmo com o passeio aleatório ao final.

> **Note**: Pode ser necessário ajustar os hiperparâmetros para fazer o algoritmo funcionar, especialmente o número de épocas. Como o sucesso no jogo (derrotar o lobo) é um evento raro, você pode esperar um tempo de treinamento significativamente maior.

## Rubrica

| Critério  | Exemplary                                                                                                                                                                                             | Adequate                                                                                                                                                                                | Needs Improvement                                                                                                                          |
| --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
|           | Um notebook é apresentado com a definição das novas regras do mundo, algoritmo de Q-Learning e algumas explicações textuais. O Q-Learning consegue melhorar significativamente os resultados em comparação ao passeio aleatório. | O notebook é apresentado, o Q-Learning é implementado e melhora os resultados em comparação ao passeio aleatório, mas não de forma significativa; ou o notebook está mal documentado e o código não está bem estruturado. | Alguma tentativa de redefinir as regras do mundo foi feita, mas o algoritmo de Q-Learning não funciona, ou a função de recompensa não está totalmente definida. |

---

**Aviso Legal**:  
Este documento foi traduzido utilizando o serviço de tradução por IA [Co-op Translator](https://github.com/Azure/co-op-translator). Embora nos esforcemos para garantir a precisão, esteja ciente de que traduções automáticas podem conter erros ou imprecisões. O documento original em seu idioma nativo deve ser considerado a fonte oficial. Para informações críticas, recomenda-se a tradução profissional feita por humanos. Não nos responsabilizamos por quaisquer mal-entendidos ou interpretações equivocadas decorrentes do uso desta tradução.