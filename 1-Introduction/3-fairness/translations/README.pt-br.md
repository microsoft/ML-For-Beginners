# Equidade no Machine Learning

![Resumo de imparcialidade no Machine Learning em um sketchnote](../../../sketchnotes/ml-fairness.png)
> Sketchnote por [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Teste pré-aula](https://white-water-09ec41f0f.azurestaticapps.net/quiz/5?loc=ptbr)

## Introdução

Neste curso, você começará a descobrir como o machine learning pode e está impactando nosso dia a dia. Mesmo agora, os sistemas e modelos estão envolvidos nas tarefas diárias de tomada de decisão, como diagnósticos de saúde ou detecção de fraudes. Portanto, é importante que esses modelos funcionem bem para fornecer resultados justos para todos.

Imagine o que pode acontecer quando os dados que você está usando para construir esses modelos não têm certos dados demográficos, como raça, gênero, visão política, religião ou representam desproporcionalmente esses dados demográficos. E quando a saída do modelo é interpretada para favorecer alguns dados demográficos? Qual é a consequência para a aplicação?

Nesta lição, você irá:

- Aumentar sua consciência sobre a importância da imparcialidade no machine learning.
- Aprender sobre danos relacionados à justiça.
- Aprender sobre avaliação e mitigação de injustiças.

## Pré-requisito

Como pré-requisito, siga o Caminho de aprendizagem "Princípios de AI responsável" e assista ao vídeo abaixo sobre o tópico:

Saiba mais sobre a AI responsável seguindo este [Caminho de aprendizagem](https://docs.microsoft.com/learn/modules/responsible-ai-principles/?WT.mc_id=academic-15963-cxa)

[![Abordagem da Microsoft para AI responsável](https://img.youtube.com/vi/dnC8-uUZXSc/0.jpg)](https://youtu.be/dnC8-uUZXSc "Abordagem da Microsoft para AI responsável")

> 🎥 Clique na imagem acima para ver um vídeo: Abordagem da Microsoft para AI responsável

## Injustiça em dados e algoritmos

> "Se você torturar os dados por tempo suficiente, eles confessarão qualquer coisa" - Ronald Coase

Essa afirmação parece extrema, mas é verdade que os dados podem ser manipulados para apoiar qualquer conclusão. Essa manipulação às vezes pode acontecer de forma não intencional. Como humanos, todos nós temos preconceitos e muitas vezes é difícil saber conscientemente quando você está introduzindo preconceitos nos dados.

Garantir a justiça na AI e no machine learning continua sendo um desafio sociotécnico complexo. O que significa que não pode ser abordado de perspectivas puramente sociais ou técnicas.

### Danos relacionados à justiça

O que você quer dizer com injustiça? “Injustiça” abrange impactos negativos, ou “danos”, para um grupo de pessoas, tais como aqueles definidos em termos de raça, sexo, idade ou condição de deficiência.  

Os principais danos relacionados à justiça podem ser classificados como:

- **Alocação**, se um gênero ou etnia, por exemplo, for favorecido em relação a outro.
- **Qualidade de serviço**. Se você treinar os dados para um cenário específico, mas a realidade for muito mais complexa, isso levará a um serviço de baixo desempenho.
- **Estereotipagem**. Associar um determinado grupo a atributos pré-atribuídos.
- **Difamação**. Criticar e rotular injustamente algo ou alguém..
- **Excesso ou falta de representação**. A ideia é que determinado grupo não seja visto em determinada profissão, e qualquer serviço ou função que continue promovendo isso está contribuindo para o mal.

Vamos dar uma olhada nos exemplos.

### Alocação

Considere um sistema hipotético para examinar os pedidos de empréstimo. O sistema tende a escolher homens brancos como melhores candidatos em relação a outros grupos. Como resultado, os empréstimos são negados a certos candidatos.

Outro exemplo seria uma ferramenta de contratação experimental desenvolvida por uma grande empresa para selecionar candidatos. A ferramenta discriminou sistematicamente um gênero por meio dos modelos foram treinados para preferir palavras associadas a outro. Isso resultou na penalização de candidatos cujos currículos continham palavras como "time feminino de rúgbi".

✅ Faça uma pequena pesquisa para encontrar um exemplo do mundo real de algo assim

### Qualidade de serviço

Os pesquisadores descobriram que vários classificadores comerciais de gênero apresentavam taxas de erro mais altas em imagens de mulheres com tons de pele mais escuros, em oposição a imagens de homens com tons de pele mais claros. [Referência](https://www.media.mit.edu/publications/gender-shades-intersectional-accuracy-disparities-in-commercial-gender-classification/)

Outro exemplo infame é um distribuidor de sabonete para as mãos que parecia não ser capaz de detectar pessoas com pele escura. [Referência](https://gizmodo.com/why-cant-this-soap-dispenser-identify-dark-skin-1797931773)

### Estereotipagem

Visão de gênero estereotípica foi encontrada na tradução automática. Ao traduzir “ele é enfermeiro e ela médica” para o turco, foram encontrados problemas. Turco é uma língua sem gênero que tem um pronome, “o” para transmitir uma terceira pessoa do singular, mas traduzir a frase de volta do turco para o inglês resulta no estereótipo e incorreto como “ela é uma enfermeira e ele é um médico”.

![translation to Turkish](../images/gender-bias-translate-en-tr.png)

![translation back to English](../images/gender-bias-translate-tr-en.png)

### Difamação

Uma tecnologia de rotulagem de imagens erroneamente rotulou imagens de pessoas de pele escura como gorilas. Rotulagem incorreta é prejudicial não apenas porque o sistema cometeu um erro, pois aplicou especificamente um rótulo que tem uma longa história de ser usado propositalmente para difamar os negros.

[![AI: Não sou uma mulher?](https://img.youtube.com/vi/QxuyfWoVV98/0.jpg)](https://www.youtube.com/watch?v=QxuyfWoVV98 "AI, Não sou uma mulher?")
> 🎥 Clique na imagem acima para ver o vídeo: AI, Não sou uma mulher - uma performance que mostra os danos causados pela difamação racista da AI

### Excesso ou falta de representação

Resultados de pesquisa de imagens distorcidos podem ser um bom exemplo desse dano. Ao pesquisar imagens de profissões com uma porcentagem igual ou maior de homens do que mulheres, como engenharia ou CEO, observe os resultados que são mais inclinados para um determinado gênero.

![Pesquisa de CEO do Bing](../images/ceos.png)
> Esta pesquisa no Bing por 'CEO' produz resultados bastante inclusivos

Esses cinco tipos principais de danos não são mutuamente exclusivos e um único sistema pode exibir mais de um tipo de dano. Além disso, cada caso varia em sua gravidade. Por exemplo, rotular injustamente alguém como criminoso é um dano muito mais grave do que rotular erroneamente uma imagem. É importante, no entanto, lembrar que mesmo danos relativamente não graves podem fazer as pessoas se sentirem alienadas ou isoladas e o impacto cumulativo pode ser extremamente opressor.

✅ **Discussão**: Reveja alguns dos exemplos e veja se eles mostram danos diferentes.  

|                         |  Alocação  | Qualidade de serviço | Estereótipo | Difamação | Excesso ou falta de representação |
| ----------------------- | :--------: | :----------------: | :----------: | :---------: | :----------------------------: |
| Sistema de contratação automatizado |     x      |         x          |      x       |             |               x                |
| Maquina de tradução     |            |                    |              |             |                                |
| Rotulagem de fotos          |            |                    |              |             |                                |

## Detectando injustiça

Existem muitas razões pelas quais um determinado sistema se comporta de maneira injusta. Vieses sociais, por exemplo, podem ser refletidos nos conjuntos de dados usados ​​para treiná-los. Por exemplo, a injustiça na contratação pode ter sido exacerbada pela dependência excessiva de dados históricos. Ao usar os padrões em currículos enviados à empresa ao longo de um período de 10 anos, o modelo determinou que os homens eram mais qualificados porque a maioria dos currículos vinha de homens, um reflexo do domínio masculino anterior na indústria de tecnologia.

Dados inadequados sobre um determinado grupo de pessoas podem ser motivo de injustiça. Por exemplo, os classificadores de imagem têm maior taxa de erro para imagens de pessoas de pele escura porque os tons de pele mais escuros estavam subrepresentados nos dados.

Suposições erradas feitas durante o desenvolvimento também causam injustiça. Por exemplo, um sistema de análise facial destinado a prever quem vai cometer um crime com base em imagens de rostos de pessoas pode levar a suposições prejudiciais. Isso pode levar a danos substanciais para as pessoas classificadas incorretamente.

## Entenda seus modelos e construa com justiça

Embora muitos aspectos de justiça não sejam capturados em métricas de justiça quantitativas e não seja possível remover totalmente o preconceito de um sistema para garantir a justiça, você ainda é responsável por detectar e mitigar os problemas de justiça tanto quanto possível.

Quando você está trabalhando com modelos de machine learning, é importante entender seus modelos por meio de garantir sua interpretabilidade e avaliar e mitigar injustiças.

Vamos usar o exemplo de seleção de empréstimo para isolar o caso e descobrir o nível de impacto de cada fator na previsão.

## Métodos de avaliação

1. **Identifique os danos (e benefícios)**. O primeiro passo é identificar danos e benefícios. Pense em como as ações e decisões podem afetar os clientes em potencial e a própria empresa.
  
2. **Identifique os grupos afetados**. Depois de entender que tipo de danos ou benefícios podem ocorrer, identifique os grupos que podem ser afetados. Esses grupos são definidos por gênero, etnia ou grupo social?

3. **Defina métricas de justiça**. Por fim, defina uma métrica para que você tenha algo para comparar em seu trabalho para melhorar a situação.

### Identificar danos (e benefícios)

Quais são os danos e benefícios associados aos empréstimos? Pense em falsos negativos e cenários de falsos positivos:

**Falsos negativos** (rejeitar, mas Y=1) - neste caso, um candidato que será capaz de reembolsar um empréstimo é rejeitado. Este é um evento adverso porque os recursos dos empréstimos são retidos de candidatos qualificados.

**Falsos positivos** ((aceitar, mas Y=0) - neste caso, o requerente obtém um empréstimo, mas acaba inadimplente. Como resultado, o caso do requerente será enviado a uma agência de cobrança de dívidas que pode afetar seus futuros pedidos de empréstimo.

### Identificar grupos afetados

A próxima etapa é determinar quais grupos provavelmente serão afetados. Por exemplo, no caso de um pedido de cartão de crédito, um modelo pode determinar que as mulheres devem receber limites de crédito muito mais baixos em comparação com seus cônjuges que compartilham bens domésticos. Todo um grupo demográfico, definido por gênero, é assim afetado.

### Definir métricas de justiça

Você identificou danos e um grupo afetado, neste caso, delineado por gênero. Agora, use os fatores quantificados para desagregar suas métricas. Por exemplo, usando os dados abaixo, você pode ver que as mulheres têm a maior taxa de falsos positivos e os homens a menor, e que o oposto é verdadeiro para falsos negativos.

✅ Em uma lição futura sobre Clustering, você verá como construir esta 'matriz de confusão' no código

|            | Taxa de falsos positivos | Taxa de falsos negativos | contagem |
| ---------- | ------------------- | ------------------- | ----- |
| Mulheres   | 0.37                | 0.27                | 54032 |
| Homens     | 0.31                | 0.35                | 28620 |
| Não binário | 0.33                | 0.31                | 1266  |

Esta tabela nos diz várias coisas. Primeiro, notamos que existem comparativamente poucas pessoas não binárias nos dados. Os dados estão distorcidos, então você precisa ter cuidado ao interpretar esses números.

Nesse caso, temos 3 grupos e 2 métricas. Quando estamos pensando em como nosso sistema afeta o grupo de clientes com seus solicitantes de empréstimos, isso pode ser suficiente, mas quando você deseja definir um número maior de grupos, pode destilar isso em conjuntos menores de resumos. Para fazer isso, você pode adicionar mais métricas, como a maior diferença ou menor proporção de cada falso negativo e falso positivo.

✅ Pare e pense: Que outros grupos provavelmente serão afetados pelo pedido de empréstimo?

## Mitigando a injustiça

Para mitigar a injustiça, explore o modelo para gerar vários modelos mitigados e compare as compensações que ele faz entre precisão e justiça para selecionar o modelo mais justo.

Esta lição introdutória não se aprofunda nos detalhes da mitigação de injustiça algorítmica, como pós-processamento e abordagem de reduções, mas aqui está uma ferramenta que você pode querer experimentar.

### Fairlearn

[Fairlearn](https://fairlearn.github.io/) is an open-source Python package that allows you to assess your systems' fairness and mitigate unfairness.  

The tool helps you to assesses how a model's predictions affect different groups, enabling you to compare multiple models by using fairness and performance metrics, and supplying a set of algorithms to mitigate unfairness in binary classification and regression. 

- Learn how to use the different components by checking out the Fairlearn's [GitHub](https://github.com/fairlearn/fairlearn/)

- Explore the [user guide](https://fairlearn.github.io/main/user_guide/index.html), [examples](https://fairlearn.github.io/main/auto_examples/index.html)

- Try some [sample notebooks](https://github.com/fairlearn/fairlearn/tree/master/notebooks).
  
- Learn [how to enable fairness assessments](https://docs.microsoft.com/azure/machine-learning/how-to-machine-learning-fairness-aml?WT.mc_id=academic-15963-cxa) of machine learning models in Azure Machine Learning.
  
- Check out these [sample notebooks](https://github.com/Azure/MachineLearningNotebooks/tree/master/contrib/fairness) for more fairness assessment scenarios in Azure Machine Learning.

---
## 🚀 Desafio

Para evitar que preconceitos sejam introduzidos em primeiro lugar, devemos:

- têm uma diversidade de experiências e perspectivas entre as pessoas que trabalham em sistemas
- investir em conjuntos de dados que reflitam a diversidade de nossa sociedade
- desenvolver melhores métodos para detectar e corrigir preconceitos quando eles ocorrem

Pense em cenários da vida real onde a injustiça é evidente na construção e uso de modelos. O que mais devemos considerar?

## [Questionário pós-aula](https://white-water-09ec41f0f.azurestaticapps.net/quiz/6?loc=ptbr)

## Revisão e Autoestudo

Nesta lição, você aprendeu alguns conceitos básicos de justiça e injustiça no aprendizado de máquina.

Assista a este workshop para se aprofundar nos tópicos:

- YouTube: danos relacionados à imparcialidade em sistemas de AI:  exemplos, avaliação e mitigação por Hanna Wallach e Miro Dudik  [Danos relacionados à imparcialidade em sistemas de AI: exemplos, avaliação e mitigação - YouTube](https://www.youtube.com/watch?v=1RptHwfkx_k)

Além disso, leia:

- Centro de recursos RAI da Microsoft: [Responsible AI Resources – Microsoft AI](https://www.microsoft.com/ai/responsible-ai-resources?activetab=pivot1%3aprimaryr4)

- Grupo de pesquisa FATE da Microsoft: [FATE: Equidade, Responsabilidade, Transparência e Ética em IA - Microsoft Research](https://www.microsoft.com/research/theme/fate/)

Explore o kit de ferramentas Fairlearn

[Fairlearn](https://fairlearn.org/)

Leia sobre as ferramentas do Azure Machine Learning para garantir justiça

- [Azure Machine Learning](https://docs.microsoft.com/azure/machine-learning/concept-fairness-ml?WT.mc_id=academic-15963-cxa)

## Tarefa

[Explore Fairlearn](assignment.pt-br.md)