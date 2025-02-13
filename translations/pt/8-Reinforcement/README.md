# Introdução ao aprendizado por reforço

O aprendizado por reforço, RL, é visto como um dos paradigmas básicos de aprendizado de máquina, ao lado do aprendizado supervisionado e do aprendizado não supervisionado. O RL gira em torno de decisões: tomar as decisões corretas ou, pelo menos, aprender com elas.

Imagine que você tem um ambiente simulado, como o mercado de ações. O que acontece se você impuser uma determinada regulamentação? Isso tem um efeito positivo ou negativo? Se algo negativo acontecer, você precisa aceitar esse _reforço negativo_, aprender com isso e mudar de rumo. Se for um resultado positivo, você precisa se basear nesse _reforço positivo_.

![peter and the wolf](../../../translated_images/peter.779730f9ba3a8a8d9290600dcf55f2e491c0640c785af7ac0d64f583c49b8864.pt.png)

> Peter e seus amigos precisam escapar do lobo faminto! Imagem por [Jen Looper](https://twitter.com/jenlooper)

## Tópico regional: Pedro e o Lobo (Rússia)

[Pedro e o Lobo](https://en.wikipedia.org/wiki/Peter_and_the_Wolf) é um conto musical escrito pelo compositor russo [Sergei Prokofiev](https://en.wikipedia.org/wiki/Sergei_Prokofiev). É uma história sobre o jovem pioneiro Pedro, que bravamente sai de casa em direção a uma clareira na floresta para perseguir o lobo. Nesta seção, treinaremos algoritmos de aprendizado de máquina que ajudarão Pedro:

- **Explorar** a área ao redor e construir um mapa de navegação otimizado
- **Aprender** a usar um skate e se equilibrar nele, para se mover mais rápido.

[![Pedro e o Lobo](https://img.youtube.com/vi/Fmi5zHg4QSM/0.jpg)](https://www.youtube.com/watch?v=Fmi5zHg4QSM)

> 🎥 Clique na imagem acima para ouvir Pedro e o Lobo de Prokofiev

## Aprendizado por reforço

Nas seções anteriores, você viu dois exemplos de problemas de aprendizado de máquina:

- **Supervisionado**, onde temos conjuntos de dados que sugerem soluções de exemplo para o problema que queremos resolver. [Classificação](../4-Classification/README.md) e [regressão](../2-Regression/README.md) são tarefas de aprendizado supervisionado.
- **Não supervisionado**, no qual não temos dados de treinamento rotulados. O principal exemplo de aprendizado não supervisionado é [Agrupamento](../5-Clustering/README.md).

Nesta seção, vamos apresentar um novo tipo de problema de aprendizado que não requer dados de treinamento rotulados. Existem vários tipos de tais problemas:

- **[Aprendizado semi-supervisionado](https://wikipedia.org/wiki/Semi-supervised_learning)**, onde temos muitos dados não rotulados que podem ser usados para pré-treinar o modelo.
- **[Aprendizado por reforço](https://wikipedia.org/wiki/Reinforcement_learning)**, no qual um agente aprende a se comportar realizando experimentos em algum ambiente simulado.

### Exemplo - jogo de computador

Suponha que você queira ensinar um computador a jogar um jogo, como xadrez ou [Super Mario](https://wikipedia.org/wiki/Super_Mario). Para que o computador jogue um jogo, precisamos que ele preveja qual movimento fazer em cada um dos estados do jogo. Embora isso possa parecer um problema de classificação, não é - porque não temos um conjunto de dados com estados e ações correspondentes. Embora possamos ter alguns dados, como partidas de xadrez existentes ou gravações de jogadores jogando Super Mario, é provável que esses dados não cubram um número grande o suficiente de estados possíveis.

Em vez de procurar dados de jogos existentes, **Aprendizado por Reforço** (RL) baseia-se na ideia de *fazer o computador jogar* muitas vezes e observar o resultado. Assim, para aplicar o Aprendizado por Reforço, precisamos de duas coisas:

- **Um ambiente** e **um simulador** que nos permita jogar um jogo muitas vezes. Esse simulador definiria todas as regras do jogo, bem como os possíveis estados e ações.

- **Uma função de recompensa**, que nos diria quão bem nos saímos durante cada movimento ou jogo.

A principal diferença entre outros tipos de aprendizado de máquina e RL é que, no RL, geralmente não sabemos se ganhamos ou perdemos até terminarmos o jogo. Assim, não podemos dizer se um determinado movimento é bom ou não - recebemos uma recompensa apenas ao final do jogo. E nosso objetivo é projetar algoritmos que nos permitam treinar um modelo em condições incertas. Vamos aprender sobre um algoritmo de RL chamado **Q-learning**.

## Aulas

1. [Introdução ao aprendizado por reforço e Q-Learning](1-QLearning/README.md)
2. [Usando um ambiente de simulação de ginásio](2-Gym/README.md)

## Créditos

"Introdução ao Aprendizado por Reforço" foi escrito com ♥️ por [Dmitry Soshnikov](http://soshnikov.com)

**Isenção de responsabilidade**:  
Este documento foi traduzido usando serviços de tradução automática baseados em IA. Embora nos esforcemos pela precisão, esteja ciente de que as traduções automatizadas podem conter erros ou imprecisões. O documento original em seu idioma nativo deve ser considerado a fonte autorizada. Para informações críticas, recomenda-se a tradução profissional por um humano. Não nos responsabilizamos por quaisquer mal-entendidos ou interpretações erradas decorrentes do uso desta tradução.