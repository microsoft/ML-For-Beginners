# Pós-escrito: Depuração de Modelos de Machine Learning usando componentes do painel de IA Responsável

## [Quiz pré-aula](https://ff-quizzes.netlify.app/en/ml/)

## Introdução

O aprendizado de máquina impacta nossas vidas cotidianas. A IA está se inserindo em alguns dos sistemas mais importantes que nos afetam como indivíduos e como sociedade, desde saúde, finanças, educação e emprego. Por exemplo, sistemas e modelos estão envolvidos em tarefas diárias de tomada de decisão, como diagnósticos médicos ou detecção de fraudes. Consequentemente, os avanços na IA, juntamente com sua adoção acelerada, estão sendo acompanhados por expectativas sociais em evolução e regulamentações crescentes em resposta. Constantemente vemos áreas onde os sistemas de IA continuam a não atender às expectativas; eles expõem novos desafios; e os governos estão começando a regulamentar soluções de IA. Por isso, é importante que esses modelos sejam analisados para fornecer resultados justos, confiáveis, inclusivos, transparentes e responsáveis para todos.

Neste currículo, exploraremos ferramentas práticas que podem ser usadas para avaliar se um modelo apresenta problemas relacionados à IA responsável. Técnicas tradicionais de depuração de aprendizado de máquina tendem a ser baseadas em cálculos quantitativos, como precisão agregada ou perda média de erro. Imagine o que pode acontecer quando os dados que você está usando para construir esses modelos carecem de certos grupos demográficos, como raça, gênero, visão política, religião, ou representam esses grupos de forma desproporcional. E se a saída do modelo for interpretada de forma a favorecer algum grupo demográfico? Isso pode introduzir uma super ou sub-representação desses grupos sensíveis, resultando em problemas de justiça, inclusão ou confiabilidade no modelo. Outro fator é que os modelos de aprendizado de máquina são considerados "caixas-pretas", o que dificulta entender e explicar o que impulsiona as previsões de um modelo. Todos esses são desafios enfrentados por cientistas de dados e desenvolvedores de IA quando não possuem ferramentas adequadas para depurar e avaliar a justiça ou confiabilidade de um modelo.

Nesta lição, você aprenderá a depurar seus modelos usando:

- **Análise de Erros**: identificar onde na distribuição de dados o modelo apresenta altas taxas de erro.
- **Visão Geral do Modelo**: realizar análises comparativas entre diferentes coortes de dados para descobrir disparidades nas métricas de desempenho do modelo.
- **Análise de Dados**: investigar onde pode haver super ou sub-representação nos dados que podem enviesar o modelo para favorecer um grupo demográfico em detrimento de outro.
- **Importância das Features**: entender quais características estão impulsionando as previsões do modelo em nível global ou local.

## Pré-requisito

Como pré-requisito, revise [Ferramentas de IA Responsável para desenvolvedores](https://www.microsoft.com/ai/ai-lab-responsible-ai-dashboard)

> ![Gif sobre Ferramentas de IA Responsável](../../../../9-Real-World/2-Debugging-ML-Models/images/rai-overview.gif)

## Análise de Erros

Métricas tradicionais de desempenho de modelos usadas para medir precisão são, na maioria das vezes, cálculos baseados em previsões corretas versus incorretas. Por exemplo, determinar que um modelo é preciso 89% do tempo com uma perda de erro de 0,001 pode ser considerado um bom desempenho. No entanto, os erros geralmente não são distribuídos uniformemente no conjunto de dados subjacente. Você pode obter uma pontuação de precisão de 89%, mas descobrir que existem diferentes regiões nos dados em que o modelo falha 42% do tempo. As consequências desses padrões de falha em certos grupos de dados podem levar a problemas de justiça ou confiabilidade. É essencial entender as áreas onde o modelo está se saindo bem ou não. As regiões de dados com um alto número de imprecisões no modelo podem acabar sendo um grupo demográfico importante.

![Analisar e depurar erros do modelo](../../../../9-Real-World/2-Debugging-ML-Models/images/ea-error-distribution.png)

O componente de Análise de Erros no painel de IA Responsável ilustra como as falhas do modelo estão distribuídas entre vários coortes com uma visualização em árvore. Isso é útil para identificar características ou áreas onde há uma alta taxa de erro no conjunto de dados. Ao ver de onde vêm a maioria das imprecisões do modelo, você pode começar a investigar a causa raiz. Também é possível criar coortes de dados para realizar análises. Esses coortes de dados ajudam no processo de depuração para determinar por que o desempenho do modelo é bom em um coorte, mas apresenta erros em outro.

![Análise de Erros](../../../../9-Real-World/2-Debugging-ML-Models/images/ea-error-cohort.png)

Os indicadores visuais no mapa de árvore ajudam a localizar as áreas problemáticas mais rapidamente. Por exemplo, quanto mais escuro o tom de vermelho em um nó da árvore, maior a taxa de erro.

O mapa de calor é outra funcionalidade de visualização que os usuários podem usar para investigar a taxa de erro usando uma ou duas características para encontrar contribuições para os erros do modelo em todo o conjunto de dados ou coortes.

![Mapa de calor da Análise de Erros](../../../../9-Real-World/2-Debugging-ML-Models/images/ea-heatmap.png)

Use a análise de erros quando você precisar:

* Obter uma compreensão profunda de como as falhas do modelo estão distribuídas em um conjunto de dados e em várias dimensões de entrada e características.
* Dividir as métricas de desempenho agregadas para descobrir automaticamente coortes com erros e informar suas etapas de mitigação direcionadas.

## Visão Geral do Modelo

Avaliar o desempenho de um modelo de aprendizado de máquina requer uma compreensão holística de seu comportamento. Isso pode ser alcançado revisando mais de uma métrica, como taxa de erro, precisão, recall, precisão ou MAE (Erro Absoluto Médio), para encontrar disparidades entre as métricas de desempenho. Uma métrica de desempenho pode parecer ótima, mas imprecisões podem ser expostas em outra métrica. Além disso, comparar as métricas para disparidades em todo o conjunto de dados ou coortes ajuda a esclarecer onde o modelo está se saindo bem ou não. Isso é especialmente importante para observar o desempenho do modelo entre características sensíveis e insensíveis (por exemplo, raça, gênero ou idade de pacientes) para descobrir possíveis injustiças que o modelo possa ter. Por exemplo, descobrir que o modelo é mais impreciso em um coorte que possui características sensíveis pode revelar possíveis injustiças no modelo.

O componente Visão Geral do Modelo do painel de IA Responsável ajuda não apenas na análise das métricas de desempenho da representação de dados em um coorte, mas também oferece aos usuários a capacidade de comparar o comportamento do modelo entre diferentes coortes.

![Coortes de conjunto de dados - visão geral do modelo no painel de IA Responsável](../../../../9-Real-World/2-Debugging-ML-Models/images/model-overview-dataset-cohorts.png)

A funcionalidade de análise baseada em características do componente permite que os usuários delimitem subgrupos de dados dentro de uma característica específica para identificar anomalias em um nível granular. Por exemplo, o painel possui inteligência integrada para gerar automaticamente coortes para uma característica selecionada pelo usuário (ex., *"tempo_no_hospital < 3"* ou *"tempo_no_hospital >= 7"*). Isso permite que o usuário isole uma característica específica de um grupo maior de dados para verificar se ela é um influenciador chave dos resultados errôneos do modelo.

![Coortes de características - visão geral do modelo no painel de IA Responsável](../../../../9-Real-World/2-Debugging-ML-Models/images/model-overview-feature-cohorts.png)

O componente Visão Geral do Modelo suporta duas classes de métricas de disparidade:

**Disparidade no desempenho do modelo**: Esses conjuntos de métricas calculam a disparidade (diferença) nos valores da métrica de desempenho selecionada entre subgrupos de dados. Aqui estão alguns exemplos:

* Disparidade na taxa de precisão
* Disparidade na taxa de erro
* Disparidade na precisão
* Disparidade no recall
* Disparidade no erro absoluto médio (MAE)

**Disparidade na taxa de seleção**: Essa métrica contém a diferença na taxa de seleção (previsão favorável) entre subgrupos. Um exemplo disso é a disparidade nas taxas de aprovação de empréstimos. Taxa de seleção significa a fração de pontos de dados em cada classe classificados como 1 (em classificação binária) ou a distribuição dos valores de previsão (em regressão).

## Análise de Dados

> "Se você torturar os dados por tempo suficiente, eles confessarão qualquer coisa" - Ronald Coase

Essa afirmação soa extrema, mas é verdade que os dados podem ser manipulados para apoiar qualquer conclusão. Tal manipulação às vezes pode acontecer de forma não intencional. Como seres humanos, todos temos preconceitos, e muitas vezes é difícil saber conscientemente quando estamos introduzindo preconceitos nos dados. Garantir justiça na IA e no aprendizado de máquina continua sendo um desafio complexo.

Os dados são um grande ponto cego para métricas tradicionais de desempenho de modelos. Você pode ter altas pontuações de precisão, mas isso nem sempre reflete o viés subjacente que pode estar presente no conjunto de dados. Por exemplo, se um conjunto de dados de funcionários possui 27% de mulheres em cargos executivos em uma empresa e 73% de homens no mesmo nível, um modelo de IA de publicidade de empregos treinado com esses dados pode direcionar principalmente um público masculino para posições de alto nível. Ter esse desequilíbrio nos dados enviesou a previsão do modelo para favorecer um gênero. Isso revela um problema de justiça onde há um viés de gênero no modelo de IA.

O componente de Análise de Dados no painel de IA Responsável ajuda a identificar áreas onde há super ou sub-representação no conjunto de dados. Ele ajuda os usuários a diagnosticar a causa raiz de erros e problemas de justiça introduzidos por desequilíbrios nos dados ou falta de representação de um grupo de dados específico. Isso dá aos usuários a capacidade de visualizar conjuntos de dados com base em resultados previstos e reais, grupos de erros e características específicas. Às vezes, descobrir um grupo de dados sub-representado também pode revelar que o modelo não está aprendendo bem, daí as altas imprecisões. Ter um modelo com viés nos dados não é apenas um problema de justiça, mas mostra que o modelo não é inclusivo ou confiável.

![Componente de Análise de Dados no painel de IA Responsável](../../../../9-Real-World/2-Debugging-ML-Models/images/dataanalysis-cover.png)

Use a análise de dados quando você precisar:

* Explorar as estatísticas do seu conjunto de dados selecionando diferentes filtros para dividir seus dados em diferentes dimensões (também conhecidas como coortes).
* Entender a distribuição do seu conjunto de dados entre diferentes coortes e grupos de características.
* Determinar se suas descobertas relacionadas à justiça, análise de erros e causalidade (derivadas de outros componentes do painel) são resultado da distribuição do seu conjunto de dados.
* Decidir em quais áreas coletar mais dados para mitigar erros que surgem de problemas de representação, ruído de rótulo, ruído de características, viés de rótulo e fatores semelhantes.

## Interpretabilidade do Modelo

Modelos de aprendizado de máquina tendem a ser "caixas-pretas". Entender quais características-chave dos dados impulsionam as previsões de um modelo pode ser desafiador. É importante fornecer transparência sobre por que um modelo faz uma determinada previsão. Por exemplo, se um sistema de IA prevê que um paciente diabético está em risco de ser readmitido em um hospital em menos de 30 dias, ele deve ser capaz de fornecer dados de suporte que levaram à sua previsão. Ter indicadores de dados de suporte traz transparência para ajudar clínicos ou hospitais a tomarem decisões bem informadas. Além disso, ser capaz de explicar por que um modelo fez uma previsão para um paciente individual permite responsabilidade com as regulamentações de saúde. Quando você usa modelos de aprendizado de máquina de maneiras que afetam a vida das pessoas, é crucial entender e explicar o que influencia o comportamento de um modelo. A explicabilidade e interpretabilidade do modelo ajudam a responder perguntas em cenários como:

* Depuração do modelo: Por que meu modelo cometeu esse erro? Como posso melhorar meu modelo?
* Colaboração humano-IA: Como posso entender e confiar nas decisões do modelo?
* Conformidade regulatória: Meu modelo atende aos requisitos legais?

O componente Importância das Features do painel de IA Responsável ajuda você a depurar e obter uma compreensão abrangente de como um modelo faz previsões. Também é uma ferramenta útil para profissionais de aprendizado de máquina e tomadores de decisão explicarem e mostrarem evidências das características que influenciam o comportamento do modelo para conformidade regulatória. Em seguida, os usuários podem explorar explicações globais e locais para validar quais características impulsionam as previsões do modelo. Explicações globais listam as principais características que afetaram a previsão geral do modelo. Explicações locais exibem quais características levaram à previsão do modelo para um caso individual. A capacidade de avaliar explicações locais também é útil na depuração ou auditoria de um caso específico para entender e interpretar melhor por que um modelo fez uma previsão precisa ou imprecisa.

![Componente Importância das Features do painel de IA Responsável](../../../../9-Real-World/2-Debugging-ML-Models/images/9-feature-importance.png)

* Explicações globais: Por exemplo, quais características afetam o comportamento geral de um modelo de readmissão hospitalar para diabetes?
* Explicações locais: Por exemplo, por que um paciente diabético com mais de 60 anos e hospitalizações anteriores foi previsto como readmitido ou não readmitido em menos de 30 dias em um hospital?

No processo de depuração ao examinar o desempenho de um modelo entre diferentes coortes, a Importância das Features mostra o nível de impacto que uma característica tem entre os coortes. Ela ajuda a revelar anomalias ao comparar o nível de influência que a característica tem em impulsionar as previsões errôneas do modelo. O componente Importância das Features pode mostrar quais valores em uma característica influenciaram positivamente ou negativamente o resultado do modelo. Por exemplo, se um modelo fez uma previsão imprecisa, o componente dá a capacidade de detalhar e identificar quais características ou valores de características impulsionaram a previsão. Esse nível de detalhe ajuda não apenas na depuração, mas também fornece transparência e responsabilidade em situações de auditoria. Por fim, o componente pode ajudar a identificar problemas de justiça. Para ilustrar, se uma característica sensível como etnia ou gênero é altamente influente em impulsionar a previsão do modelo, isso pode ser um sinal de viés de raça ou gênero no modelo.

![Importância das Features](../../../../9-Real-World/2-Debugging-ML-Models/images/9-features-influence.png)

Use interpretabilidade quando você precisar:

* Determinar o quão confiáveis são as previsões do seu sistema de IA, entendendo quais características são mais importantes para as previsões.
* Abordar a depuração do seu modelo entendendo-o primeiro e identificando se o modelo está usando características saudáveis ou apenas correlações falsas.
* Descobrir possíveis fontes de injustiça entendendo se o modelo está baseando previsões em características sensíveis ou em características altamente correlacionadas com elas.
* Construir confiança do usuário nas decisões do modelo gerando explicações locais para ilustrar seus resultados.
* Completar uma auditoria regulatória de um sistema de IA para validar modelos e monitorar o impacto das decisões do modelo sobre os seres humanos.

## Conclusão

Todos os componentes do painel de IA Responsável são ferramentas práticas para ajudar você a construir modelos de aprendizado de máquina que sejam menos prejudiciais e mais confiáveis para a sociedade. Eles melhoram a prevenção de ameaças aos direitos humanos; discriminação ou exclusão de certos grupos de oportunidades de vida; e o risco de danos físicos ou psicológicos. Também ajudam a construir confiança nas decisões do modelo, gerando explicações locais para ilustrar seus resultados. Alguns dos possíveis danos podem ser classificados como:

- **Alocação**, se um gênero ou etnia, por exemplo, for favorecido em detrimento de outro.
- **Qualidade do serviço**. Se você treinar os dados para um cenário específico, mas a realidade for muito mais complexa, isso leva a um serviço de desempenho ruim.
- **Estereotipagem**. Associar um determinado grupo a atributos pré-atribuídos.
- **Denigração**. Criticar injustamente e rotular algo ou alguém.
- **Representação excessiva ou insuficiente**. A ideia é que um determinado grupo não seja visto em uma certa profissão, e qualquer serviço ou função que continue promovendo isso está contribuindo para o problema.

### Painel Azure RAI

O [Painel Azure RAI](https://learn.microsoft.com/en-us/azure/machine-learning/concept-responsible-ai-dashboard?WT.mc_id=aiml-90525-ruyakubu) é baseado em ferramentas de código aberto desenvolvidas por instituições acadêmicas e organizações líderes, incluindo a Microsoft. Ele é essencial para cientistas de dados e desenvolvedores de IA entenderem melhor o comportamento dos modelos, identificarem e mitigarem problemas indesejáveis em modelos de IA.

- Aprenda como usar os diferentes componentes consultando a [documentação do painel RAI.](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-responsible-ai-dashboard?WT.mc_id=aiml-90525-ruyakubu)

- Confira alguns [notebooks de exemplo do painel RAI](https://github.com/Azure/RAI-vNext-Preview/tree/main/examples/notebooks) para depurar cenários de IA mais responsáveis no Azure Machine Learning.

---
## 🚀 Desafio

Para evitar que vieses estatísticos ou de dados sejam introduzidos desde o início, devemos:

- ter diversidade de origens e perspectivas entre as pessoas que trabalham nos sistemas
- investir em conjuntos de dados que reflitam a diversidade da nossa sociedade
- desenvolver melhores métodos para detectar e corrigir vieses quando eles ocorrerem

Pense em cenários da vida real onde a injustiça é evidente na construção e uso de modelos. O que mais devemos considerar?

## [Quiz pós-aula](https://ff-quizzes.netlify.app/en/ml/)
## Revisão e Autoestudo

Nesta lição, você aprendeu algumas ferramentas práticas para incorporar IA responsável no aprendizado de máquina.

Assista a este workshop para se aprofundar nos tópicos:

- Painel de IA Responsável: Uma solução completa para operacionalizar IA responsável na prática, por Besmira Nushi e Mehrnoosh Sameki

[![Painel de IA Responsável: Uma solução completa para operacionalizar IA responsável na prática](https://img.youtube.com/vi/f1oaDNl3djg/0.jpg)](https://www.youtube.com/watch?v=f1oaDNl3djg "Painel de IA Responsável: Uma solução completa para operacionalizar IA responsável na prática")


> 🎥 Clique na imagem acima para assistir ao vídeo: Painel de IA Responsável: Uma solução completa para operacionalizar IA responsável na prática, por Besmira Nushi e Mehrnoosh Sameki

Consulte os seguintes materiais para aprender mais sobre IA responsável e como construir modelos mais confiáveis:

- Ferramentas do painel RAI da Microsoft para depuração de modelos de aprendizado de máquina: [Recursos de ferramentas de IA responsável](https://aka.ms/rai-dashboard)

- Explore o kit de ferramentas de IA responsável: [Github](https://github.com/microsoft/responsible-ai-toolbox)

- Centro de recursos de IA responsável da Microsoft: [Recursos de IA Responsável – Microsoft AI](https://www.microsoft.com/en-us/ai/tools-practices)

- Grupo de pesquisa FATE da Microsoft: [FATE: Justiça, Responsabilidade, Transparência e Ética em IA - Microsoft Research](https://www.microsoft.com/research/theme/fate/)

## Tarefa

[Explore o painel RAI](assignment.md)

---

**Aviso Legal**:  
Este documento foi traduzido utilizando o serviço de tradução por IA [Co-op Translator](https://github.com/Azure/co-op-translator). Embora nos esforcemos para garantir a precisão, esteja ciente de que traduções automatizadas podem conter erros ou imprecisões. O documento original em seu idioma nativo deve ser considerado a fonte autoritativa. Para informações críticas, recomenda-se a tradução profissional realizada por humanos. Não nos responsabilizamos por quaisquer mal-entendidos ou interpretações equivocadas decorrentes do uso desta tradução.