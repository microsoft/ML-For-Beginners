# Pós-escrito: Aprendizado de máquina no mundo real

![Resumo do aprendizado de máquina no mundo real em um sketchnote](../../../../translated_images/ml-realworld.26ee2746716155771f8076598b6145e6533fe4a9e2e465ea745f46648cbf1b84.pt.png)
> Sketchnote por [Tomomi Imura](https://www.twitter.com/girlie_mac)

Neste currículo, você aprendeu várias maneiras de preparar dados para treinamento e criar modelos de aprendizado de máquina. Você construiu uma série de modelos clássicos de regressão, agrupamento, classificação, processamento de linguagem natural e séries temporais. Parabéns! Agora, você pode estar se perguntando para que tudo isso serve... quais são as aplicações no mundo real para esses modelos?

Embora muito do interesse na indústria tenha sido despertado pela IA, que geralmente utiliza aprendizado profundo, ainda existem aplicações valiosas para modelos clássicos de aprendizado de máquina. Você pode até usar algumas dessas aplicações hoje! Nesta lição, você explorará como oito indústrias diferentes e áreas de conhecimento utilizam esses tipos de modelos para tornar suas aplicações mais performáticas, confiáveis, inteligentes e valiosas para os usuários.

## [Quiz pré-aula](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/49/)

## 💰 Finanças

O setor financeiro oferece muitas oportunidades para aprendizado de máquina. Muitos problemas nesta área podem ser modelados e resolvidos usando ML.

### Detecção de fraudes com cartão de crédito

Aprendemos sobre [agrupamento k-means](../../5-Clustering/2-K-Means/README.md) anteriormente no curso, mas como ele pode ser usado para resolver problemas relacionados a fraudes com cartão de crédito?

O agrupamento k-means é útil durante uma técnica de detecção de fraudes com cartão de crédito chamada **detecção de outliers**. Outliers, ou desvios nas observações sobre um conjunto de dados, podem nos dizer se um cartão de crédito está sendo usado de forma normal ou se algo incomum está acontecendo. Como mostrado no artigo vinculado abaixo, você pode classificar dados de cartão de crédito usando um algoritmo de agrupamento k-means e atribuir cada transação a um cluster com base em quão fora do normal ela parece estar. Em seguida, você pode avaliar os clusters mais arriscados para transações fraudulentas versus legítimas.
[Referência](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.680.1195&rep=rep1&type=pdf)

### Gestão de patrimônio

Na gestão de patrimônio, um indivíduo ou empresa cuida de investimentos em nome de seus clientes. O trabalho deles é sustentar e crescer a riqueza a longo prazo, portanto, é essencial escolher investimentos que tenham um bom desempenho.

Uma maneira de avaliar como um investimento específico se comporta é através da regressão estatística. A [regressão linear](../../2-Regression/1-Tools/README.md) é uma ferramenta valiosa para entender como um fundo se comporta em relação a algum benchmark. Também podemos deduzir se os resultados da regressão são estatisticamente significativos ou quanto eles afetariam os investimentos de um cliente. Você poderia até expandir ainda mais sua análise usando múltiplas regressões, onde fatores de risco adicionais podem ser levados em conta. Para um exemplo de como isso funcionaria para um fundo específico, confira o artigo abaixo sobre avaliação de desempenho de fundos usando regressão.
[Referência](http://www.brightwoodventures.com/evaluating-fund-performance-using-regression/)

## 🎓 Educação

O setor educacional também é uma área muito interessante onde o ML pode ser aplicado. Existem problemas interessantes a serem enfrentados, como detectar trapaças em testes ou redações ou gerenciar preconceitos, intencionais ou não, no processo de correção.

### Predição do comportamento dos alunos

[Coursera](https://coursera.com), um provedor de cursos online abertos, tem um ótimo blog técnico onde discutem muitas decisões de engenharia. Neste estudo de caso, eles traçaram uma linha de regressão para tentar explorar qualquer correlação entre uma baixa classificação NPS (Net Promoter Score) e retenção ou desistência de cursos.
[Referência](https://medium.com/coursera-engineering/controlled-regression-quantifying-the-impact-of-course-quality-on-learner-retention-31f956bd592a)

### Mitigação de preconceitos

[Grammarly](https://grammarly.com), um assistente de escrita que verifica erros de ortografia e gramática, utiliza sofisticados [sistemas de processamento de linguagem natural](../../6-NLP/README.md) em seus produtos. Eles publicaram um estudo de caso interessante em seu blog técnico sobre como lidaram com preconceitos de gênero no aprendizado de máquina, que você aprendeu em nossa [lição introdutória sobre justiça](../../1-Introduction/3-fairness/README.md).
[Referência](https://www.grammarly.com/blog/engineering/mitigating-gender-bias-in-autocorrect/)

## 👜 Varejo

O setor de varejo pode definitivamente se beneficiar do uso de ML, desde a criação de uma melhor jornada do cliente até o gerenciamento otimizado de estoques.

### Personalizando a jornada do cliente

Na Wayfair, uma empresa que vende produtos para o lar, como móveis, ajudar os clientes a encontrar os produtos certos para seu gosto e necessidades é primordial. Neste artigo, engenheiros da empresa descrevem como utilizam ML e NLP para "exibir os resultados certos para os clientes". Notavelmente, seu Query Intent Engine foi desenvolvido para usar extração de entidades, treinamento de classificadores, extração de ativos e opiniões, e marcação de sentimentos em avaliações de clientes. Este é um caso clássico de como o NLP funciona no varejo online.
[Referência](https://www.aboutwayfair.com/tech-innovation/how-we-use-machine-learning-and-natural-language-processing-to-empower-search)

### Gestão de inventário

Empresas inovadoras e ágeis como [StitchFix](https://stitchfix.com), um serviço de box que envia roupas para os consumidores, dependem fortemente de ML para recomendações e gestão de inventário. Suas equipes de estilo trabalham em conjunto com suas equipes de merchandising, na verdade: "um de nossos cientistas de dados brincou com um algoritmo genético e o aplicou a vestuário para prever qual seria uma peça de roupa de sucesso que não existe hoje. Nós apresentamos isso à equipe de merchandising e agora eles podem usar isso como uma ferramenta."
[Referência](https://www.zdnet.com/article/how-stitch-fix-uses-machine-learning-to-master-the-science-of-styling/)

## 🏥 Cuidados de Saúde

O setor de saúde pode aproveitar o ML para otimizar tarefas de pesquisa e também problemas logísticos, como readmissão de pacientes ou controle da propagação de doenças.

### Gestão de ensaios clínicos

A toxicidade em ensaios clínicos é uma grande preocupação para os fabricantes de medicamentos. Quanta toxicidade é tolerável? Neste estudo, a análise de vários métodos de ensaios clínicos levou ao desenvolvimento de uma nova abordagem para prever as chances de resultados de ensaios clínicos. Especificamente, eles foram capazes de usar florestas aleatórias para produzir um [classificador](../../4-Classification/README.md) que é capaz de distinguir entre grupos de medicamentos.
[Referência](https://www.sciencedirect.com/science/article/pii/S2451945616302914)

### Gestão de readmissão hospitalar

Os cuidados hospitalares são caros, especialmente quando os pacientes precisam ser readmitidos. Este artigo discute uma empresa que utiliza ML para prever o potencial de readmissão usando algoritmos de [agrupamento](../../5-Clustering/README.md). Esses clusters ajudam os analistas a "descobrir grupos de readmissões que podem compartilhar uma causa comum".
[Referência](https://healthmanagement.org/c/healthmanagement/issuearticle/hospital-readmissions-and-machine-learning)

### Gestão de doenças

A recente pandemia destacou as maneiras pelas quais o aprendizado de máquina pode ajudar a interromper a propagação de doenças. Neste artigo, você reconhecerá o uso de ARIMA, curvas logísticas, regressão linear e SARIMA. "Este trabalho é uma tentativa de calcular a taxa de propagação deste vírus e, assim, prever as mortes, recuperações e casos confirmados, para que possamos nos preparar melhor e sobreviver."
[Referência](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7979218/)

## 🌲 Ecologia e Tecnologia Verde

A natureza e a ecologia consistem em muitos sistemas sensíveis onde a interação entre animais e a natureza se destaca. É importante ser capaz de medir esses sistemas com precisão e agir adequadamente se algo acontecer, como um incêndio florestal ou uma queda na população animal.

### Gestão florestal

Você aprendeu sobre [Aprendizado por Reforço](../../8-Reinforcement/README.md) em lições anteriores. Ele pode ser muito útil ao tentar prever padrões na natureza. Em particular, pode ser usado para rastrear problemas ecológicos, como incêndios florestais e a propagação de espécies invasivas. No Canadá, um grupo de pesquisadores usou Aprendizado por Reforço para construir modelos dinâmicos de incêndios florestais a partir de imagens de satélite. Usando um inovador "processo de propagação espacial (SSP)", eles imaginaram um incêndio florestal como "o agente em qualquer célula da paisagem." "O conjunto de ações que o fogo pode tomar de uma localização a qualquer momento inclui se espalhar para o norte, sul, leste ou oeste ou não se espalhar.

Essa abordagem inverte a configuração usual de RL, uma vez que a dinâmica do Processo de Decisão de Markov (MDP) correspondente é uma função conhecida para a propagação imediata do incêndio florestal." Leia mais sobre os algoritmos clássicos usados por este grupo no link abaixo.
[Referência](https://www.frontiersin.org/articles/10.3389/fict.2018.00006/full)

### Sensoriamento de movimentos de animais

Embora o aprendizado profundo tenha criado uma revolução no rastreamento visual de movimentos de animais (você pode construir seu próprio [rastreador de ursos polares](https://docs.microsoft.com/learn/modules/build-ml-model-with-azure-stream-analytics/?WT.mc_id=academic-77952-leestott) aqui), o ML clássico ainda tem seu espaço nessa tarefa.

Sensores para rastrear movimentos de animais de fazenda e IoT utilizam esse tipo de processamento visual, mas técnicas de ML mais básicas são úteis para pré-processar dados. Por exemplo, neste artigo, as posturas das ovelhas foram monitoradas e analisadas usando vários algoritmos classificadores. Você pode reconhecer a curva ROC na página 335.
[Referência](https://druckhaus-hofmann.de/gallery/31-wj-feb-2020.pdf)

### ⚡️ Gestão de Energia

Em nossas lições sobre [previsão de séries temporais](../../7-TimeSeries/README.md), mencionamos o conceito de parquímetros inteligentes para gerar receita para uma cidade com base na compreensão da oferta e da demanda. Este artigo discute em detalhes como agrupamento, regressão e previsão de séries temporais se combinaram para ajudar a prever o uso futuro de energia na Irlanda, com base em medições inteligentes.
[Referência](https://www-cdn.knime.com/sites/default/files/inline-images/knime_bigdata_energy_timeseries_whitepaper.pdf)

## 💼 Seguros

O setor de seguros é outro setor que utiliza ML para construir e otimizar modelos financeiros e atuariais viáveis.

### Gestão de Volatilidade

A MetLife, uma provedora de seguros de vida, é transparente sobre a maneira como analisa e mitiga a volatilidade em seus modelos financeiros. Neste artigo, você notará visualizações de classificação binária e ordinal. Você também descobrirá visualizações de previsão.
[Referência](https://investments.metlife.com/content/dam/metlifecom/us/investments/insights/research-topics/macro-strategy/pdf/MetLifeInvestmentManagement_MachineLearnedRanking_070920.pdf)

## 🎨 Artes, Cultura e Literatura

Nas artes, por exemplo, no jornalismo, existem muitos problemas interessantes. Detectar notícias falsas é um grande problema, pois já foi provado que influencia a opinião das pessoas e até derruba democracias. Museus também podem se beneficiar do uso de ML em tudo, desde encontrar conexões entre artefatos até planejamento de recursos.

### Detecção de notícias falsas

Detectar notícias falsas se tornou um jogo de gato e rato na mídia atual. Neste artigo, pesquisadores sugerem que um sistema combinando várias das técnicas de ML que estudamos pode ser testado e o melhor modelo implantado: "Este sistema é baseado em processamento de linguagem natural para extrair características dos dados e, em seguida, essas características são usadas para o treinamento de classificadores de aprendizado de máquina, como Naive Bayes, Support Vector Machine (SVM), Random Forest (RF), Stochastic Gradient Descent (SGD) e Regressão Logística (LR)."
[Referência](https://www.irjet.net/archives/V7/i6/IRJET-V7I6688.pdf)

Este artigo mostra como combinar diferentes domínios de ML pode produzir resultados interessantes que podem ajudar a impedir a propagação de notícias falsas e causar danos reais; neste caso, o impulso foi a disseminação de rumores sobre tratamentos para COVID que incitaram a violência em massa.

### ML em Museus

Os museus estão à beira de uma revolução da IA em que catalogar e digitalizar coleções e encontrar conexões entre artefatos está se tornando mais fácil à medida que a tecnologia avança. Projetos como [In Codice Ratio](https://www.sciencedirect.com/science/article/abs/pii/S0306457321001035#:~:text=1.,studies%20over%20large%20historical%20sources.) estão ajudando a desvendar os mistérios de coleções inacessíveis, como os Arquivos do Vaticano. Mas, o aspecto comercial dos museus também se beneficia de modelos de ML.

Por exemplo, o Art Institute of Chicago construiu modelos para prever quais públicos estão interessados e quando eles irão às exposições. O objetivo é criar experiências de visita individualizadas e otimizadas toda vez que o usuário visita o museu. "Durante o exercício fiscal de 2017, o modelo previu a participação e as admissões com uma precisão de 1 por cento, diz Andrew Simnick, vice-presidente sênior do Art Institute."
[Reference](https://www.chicagobusiness.com/article/20180518/ISSUE01/180519840/art-institute-of-chicago-uses-data-to-make-exhibit-choices)

## 🏷 Marketing

### Segmentação de clientes

As estratégias de marketing mais eficazes segmentam os clientes de maneiras diferentes com base em vários agrupamentos. Neste artigo, são discutidos os usos de algoritmos de agrupamento para apoiar o marketing diferenciado. O marketing diferenciado ajuda as empresas a melhorar o reconhecimento da marca, alcançar mais clientes e aumentar os lucros.
[Reference](https://ai.inqline.com/machine-learning-for-marketing-customer-segmentation/)

## 🚀 Desafio

Identifique outro setor que se beneficie de algumas das técnicas que você aprendeu neste currículo e descubra como ele utiliza ML.

## [Questionário pós-aula](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/50/)

## Revisão e Autoestudo

A equipe de ciência de dados da Wayfair tem vários vídeos interessantes sobre como eles usam ML em sua empresa. Vale a pena [dar uma olhada](https://www.youtube.com/channel/UCe2PjkQXqOuwkW1gw6Ameuw/videos)!

## Tarefa

[Uma caça ao tesouro de ML](assignment.md)

**Isenção de responsabilidade**:  
Este documento foi traduzido utilizando serviços de tradução automática baseados em IA. Embora nos esforcemos pela precisão, esteja ciente de que traduções automatizadas podem conter erros ou imprecisões. O documento original em seu idioma nativo deve ser considerado a fonte autoritária. Para informações críticas, recomenda-se a tradução profissional por um humano. Não nos responsabilizamos por quaisquer mal-entendidos ou interpretações errôneas decorrentes do uso desta tradução.