<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "83320d6b6994909e35d830cebf214039",
  "translation_date": "2025-09-05T08:42:15+00:00",
  "source_file": "9-Real-World/1-Applications/README.md",
  "language_code": "pt"
}
-->
# Pós-escrito: Aprendizagem automática no mundo real

![Resumo da aprendizagem automática no mundo real em um sketchnote](../../../../sketchnotes/ml-realworld.png)
> Sketchnote por [Tomomi Imura](https://www.twitter.com/girlie_mac)

Neste currículo, aprendeste várias formas de preparar dados para treino e criar modelos de aprendizagem automática. Construíste uma série de modelos clássicos de regressão, clustering, classificação, processamento de linguagem natural e séries temporais. Parabéns! Agora, talvez te perguntes para que serve tudo isto... quais são as aplicações reais destes modelos?

Embora a inteligência artificial (IA), que geralmente utiliza aprendizagem profunda, tenha atraído muita atenção na indústria, ainda existem aplicações valiosas para modelos clássicos de aprendizagem automática. É provável que já utilizes algumas destas aplicações no teu dia a dia! Nesta lição, vais explorar como oito indústrias e domínios de especialização utilizam estes tipos de modelos para tornar as suas aplicações mais eficientes, fiáveis, inteligentes e valiosas para os utilizadores.

## [Questionário pré-aula](https://ff-quizzes.netlify.app/en/ml/)

## 💰 Finanças

O setor financeiro oferece muitas oportunidades para a aprendizagem automática. Muitos problemas nesta área podem ser modelados e resolvidos com ML.

### Deteção de fraude com cartões de crédito

Aprendemos sobre [k-means clustering](../../5-Clustering/2-K-Means/README.md) anteriormente no curso, mas como pode ser usado para resolver problemas relacionados com fraude em cartões de crédito?

O k-means clustering é útil numa técnica de deteção de fraude chamada **deteção de outliers**. Outliers, ou desvios nas observações de um conjunto de dados, podem indicar se um cartão de crédito está a ser usado de forma normal ou se algo incomum está a acontecer. Como mostrado no artigo abaixo, é possível organizar dados de cartões de crédito usando um algoritmo de k-means clustering e atribuir cada transação a um cluster com base no grau de anomalia. Depois, é possível avaliar os clusters mais arriscados para identificar transações fraudulentas versus legítimas.  
[Referência](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.680.1195&rep=rep1&type=pdf)

### Gestão de património

Na gestão de património, um indivíduo ou empresa gere investimentos em nome dos seus clientes. O objetivo é sustentar e aumentar o património a longo prazo, sendo essencial escolher investimentos que tenham um bom desempenho.

Uma forma de avaliar o desempenho de um investimento é através da regressão estatística. [A regressão linear](../../2-Regression/1-Tools/README.md) é uma ferramenta valiosa para compreender como um fundo se comporta em relação a um benchmark. Também é possível determinar se os resultados da regressão são estatisticamente significativos ou o impacto que teriam nos investimentos de um cliente. Podes expandir ainda mais a análise utilizando regressão múltipla, onde fatores de risco adicionais são considerados. Para um exemplo de como isto funciona para um fundo específico, consulta o artigo abaixo sobre avaliação de desempenho de fundos usando regressão.  
[Referência](http://www.brightwoodventures.com/evaluating-fund-performance-using-regression/)

## 🎓 Educação

O setor da educação também é uma área muito interessante para a aplicação de ML. Existem problemas intrigantes a resolver, como detetar fraudes em testes ou ensaios, ou gerir preconceitos, intencionais ou não, no processo de correção.

### Prever o comportamento dos estudantes

A [Coursera](https://coursera.com), uma plataforma de cursos online, tem um excelente blog técnico onde discute várias decisões de engenharia. Neste estudo de caso, traçaram uma linha de regressão para explorar a correlação entre uma baixa classificação NPS (Net Promoter Score) e a retenção ou desistência de um curso.  
[Referência](https://medium.com/coursera-engineering/controlled-regression-quantifying-the-impact-of-course-quality-on-learner-retention-31f956bd592a)

### Mitigação de preconceitos

O [Grammarly](https://grammarly.com), um assistente de escrita que verifica erros ortográficos e gramaticais, utiliza sistemas sofisticados de [processamento de linguagem natural](../../6-NLP/README.md) nos seus produtos. Publicaram um estudo de caso interessante no seu blog técnico sobre como lidaram com preconceitos de género na aprendizagem automática, tema que aprendeste na nossa [lição introdutória sobre justiça](../../1-Introduction/3-fairness/README.md).  
[Referência](https://www.grammarly.com/blog/engineering/mitigating-gender-bias-in-autocorrect/)

## 👜 Retalho

O setor do retalho pode beneficiar imenso do uso de ML, desde a criação de uma melhor jornada do cliente até à gestão otimizada de inventário.

### Personalização da jornada do cliente

Na Wayfair, uma empresa que vende artigos para o lar, ajudar os clientes a encontrar os produtos certos para os seus gostos e necessidades é fundamental. Neste artigo, os engenheiros da empresa descrevem como utilizam ML e NLP para "apresentar os resultados certos aos clientes". Notavelmente, o seu motor de intenção de consulta foi construído para usar extração de entidades, treino de classificadores, extração de ativos e opiniões, e etiquetagem de sentimentos em avaliações de clientes. Este é um caso clássico de como o NLP funciona no retalho online.  
[Referência](https://www.aboutwayfair.com/tech-innovation/how-we-use-machine-learning-and-natural-language-processing-to-empower-search)

### Gestão de inventário

Empresas inovadoras e ágeis como a [StitchFix](https://stitchfix.com), um serviço de caixas de roupa por assinatura, dependem fortemente de ML para recomendações e gestão de inventário. As suas equipas de estilistas trabalham em conjunto com as equipas de merchandising. Por exemplo: "um dos nossos cientistas de dados experimentou um algoritmo genético e aplicou-o ao vestuário para prever o sucesso de uma peça de roupa que ainda não existe. Apresentámos isso à equipa de merchandising, que agora pode usar isso como uma ferramenta."  
[Referência](https://www.zdnet.com/article/how-stitch-fix-uses-machine-learning-to-master-the-science-of-styling/)

## 🏥 Saúde

O setor da saúde pode aproveitar o ML para otimizar tarefas de investigação e também problemas logísticos, como readmissões hospitalares ou a contenção de doenças.

### Gestão de ensaios clínicos

A toxicidade em ensaios clínicos é uma grande preocupação para os fabricantes de medicamentos. Quanta toxicidade é tolerável? Neste estudo, a análise de vários métodos de ensaios clínicos levou ao desenvolvimento de uma nova abordagem para prever os resultados dos ensaios. Especificamente, foi possível usar random forest para produzir um [classificador](../../4-Classification/README.md) capaz de distinguir entre grupos de medicamentos.  
[Referência](https://www.sciencedirect.com/science/article/pii/S2451945616302914)

### Gestão de readmissões hospitalares

Os cuidados hospitalares são dispendiosos, especialmente quando os pacientes precisam de ser readmitidos. Este artigo discute uma empresa que utiliza ML para prever o potencial de readmissão usando [clustering](../../5-Clustering/README.md). Estes clusters ajudam os analistas a "descobrir grupos de readmissões que podem partilhar uma causa comum".  
[Referência](https://healthmanagement.org/c/healthmanagement/issuearticle/hospital-readmissions-and-machine-learning)

### Gestão de doenças

A recente pandemia destacou como a aprendizagem automática pode ajudar a conter a propagação de doenças. Neste artigo, reconheces o uso de ARIMA, curvas logísticas, regressão linear e SARIMA. "Este trabalho é uma tentativa de calcular a taxa de propagação deste vírus e, assim, prever mortes, recuperações e casos confirmados, para que possamos estar melhor preparados e sobreviver."  
[Referência](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7979218/)

## 🌲 Ecologia e Tecnologia Verde

A natureza e a ecologia consistem em muitos sistemas sensíveis onde a interação entre animais e o meio ambiente é crucial. É importante medir estes sistemas com precisão e agir adequadamente se algo acontecer, como um incêndio florestal ou uma queda na população animal.

### Gestão florestal

Aprendeste sobre [Reinforcement Learning](../../8-Reinforcement/README.md) em lições anteriores. Pode ser muito útil para prever padrões na natureza. Em particular, pode ser usado para monitorizar problemas ecológicos como incêndios florestais e a propagação de espécies invasoras. No Canadá, um grupo de investigadores utilizou Reinforcement Learning para construir modelos de dinâmica de incêndios florestais a partir de imagens de satélite. Usando um inovador "processo de propagação espacial (SSP)", imaginaram um incêndio florestal como "o agente em qualquer célula da paisagem". "O conjunto de ações que o fogo pode tomar de uma localização em qualquer momento inclui propagar-se para norte, sul, leste, oeste ou não se propagar."  
[Referência](https://www.frontiersin.org/articles/10.3389/fict.2018.00006/full)

### Monitorização de movimentos de animais

Embora a aprendizagem profunda tenha revolucionado o rastreamento visual de movimentos de animais (podes criar o teu próprio [rastreador de ursos polares](https://docs.microsoft.com/learn/modules/build-ml-model-with-azure-stream-analytics/?WT.mc_id=academic-77952-leestott) aqui), as técnicas clássicas de ML ainda têm o seu lugar nesta tarefa.

Sensores para rastrear movimentos de animais de quinta e IoT utilizam este tipo de processamento visual, mas técnicas mais básicas de ML são úteis para pré-processar dados. Por exemplo, neste artigo, as posturas de ovelhas foram monitorizadas e analisadas usando vários algoritmos de classificação. Podes reconhecer a curva ROC na página 335.  
[Referência](https://druckhaus-hofmann.de/gallery/31-wj-feb-2020.pdf)

### ⚡️ Gestão de Energia

Nas nossas lições sobre [previsão de séries temporais](../../7-TimeSeries/README.md), abordámos o conceito de parquímetros inteligentes para gerar receita para uma cidade com base na compreensão da oferta e da procura. Este artigo discute em detalhe como clustering, regressão e previsão de séries temporais foram combinados para ajudar a prever o uso futuro de energia na Irlanda, com base em medições inteligentes.  
[Referência](https://www-cdn.knime.com/sites/default/files/inline-images/knime_bigdata_energy_timeseries_whitepaper.pdf)

## 💼 Seguros

O setor de seguros é outro setor que utiliza ML para construir e otimizar modelos financeiros e atuariais viáveis.

### Gestão de volatilidade

A MetLife, uma fornecedora de seguros de vida, é transparente sobre como analisa e mitiga a volatilidade nos seus modelos financeiros. Neste artigo, vais notar visualizações de classificação binária e ordinal. Também vais encontrar visualizações de previsão.  
[Referência](https://investments.metlife.com/content/dam/metlifecom/us/investments/insights/research-topics/macro-strategy/pdf/MetLifeInvestmentManagement_MachineLearnedRanking_070920.pdf)

## 🎨 Artes, Cultura e Literatura

Nas artes, por exemplo no jornalismo, existem muitos problemas interessantes. Detetar notícias falsas é um grande desafio, pois já foi provado que influenciam a opinião pública e até derrubam democracias. Os museus também podem beneficiar do uso de ML em tudo, desde encontrar ligações entre artefactos até ao planeamento de recursos.

### Deteção de notícias falsas

Detetar notícias falsas tornou-se um jogo de gato e rato nos meios de comunicação atuais. Neste artigo, os investigadores sugerem que um sistema combinando várias das técnicas de ML que estudámos pode ser testado e o melhor modelo implementado: "Este sistema baseia-se no processamento de linguagem natural para extrair características dos dados e, em seguida, essas características são usadas para treinar classificadores de aprendizagem automática como Naive Bayes, Support Vector Machine (SVM), Random Forest (RF), Stochastic Gradient Descent (SGD) e Logistic Regression (LR)."  
[Referência](https://www.irjet.net/archives/V7/i6/IRJET-V7I6688.pdf)

Este artigo mostra como combinar diferentes domínios de ML pode produzir resultados interessantes que ajudam a impedir a disseminação de notícias falsas e os danos que podem causar; neste caso, o foco foi a propagação de rumores sobre tratamentos para a COVID que incitaram violência.

### ML em Museus

Os museus estão à beira de uma revolução em IA, onde catalogar e digitalizar coleções e encontrar ligações entre artefactos está a tornar-se mais fácil à medida que a tecnologia avança. Projetos como o [In Codice Ratio](https://www.sciencedirect.com/science/article/abs/pii/S0306457321001035#:~:text=1.,studies%20over%20large%20historical%20sources.) estão a ajudar a desvendar os mistérios de coleções inacessíveis, como os Arquivos do Vaticano. Mas o aspeto comercial dos museus também beneficia de modelos de ML.

Por exemplo, o Art Institute of Chicago construiu modelos para prever os interesses do público e quando irão visitar exposições. O objetivo é criar experiências de visita individualizadas e otimizadas sempre que o utilizador visita o museu. "Durante o ano fiscal de 2017, o modelo previu a frequência e as receitas de bilheteira com uma precisão de 1%, diz Andrew Simnick, vice-presidente sénior do Art Institute."  
[Referência](https://www.chicagobusiness.com/article/20180518/ISSUE01/180519840/art-institute-of-chicago-uses-data-to-make-exhibit-choices)

## 🏷 Marketing

### Segmentação de clientes

As estratégias de marketing mais eficazes segmentam os clientes de diferentes formas com base em vários agrupamentos. Neste artigo, são discutidas as utilizações de algoritmos de Clustering para apoiar o marketing diferenciado. O marketing diferenciado ajuda as empresas a melhorar o reconhecimento da marca, alcançar mais clientes e gerar mais receitas.  
[Referência](https://ai.inqline.com/machine-learning-for-marketing-customer-segmentation/)

## 🚀 Desafio

Identifica outro setor que beneficie de algumas das técnicas que aprendeste neste currículo e descobre como utiliza ML.
## [Questionário pós-aula](https://ff-quizzes.netlify.app/en/ml/)

## Revisão e Autoestudo

A equipa de ciência de dados da Wayfair tem vários vídeos interessantes sobre como utilizam ML na sua empresa. Vale a pena [dar uma vista de olhos](https://www.youtube.com/channel/UCe2PjkQXqOuwkW1gw6Ameuw/videos)!

## Tarefa

[Uma caça ao tesouro de ML](assignment.md)

---

**Aviso Legal**:  
Este documento foi traduzido utilizando o serviço de tradução por IA [Co-op Translator](https://github.com/Azure/co-op-translator). Embora nos esforcemos para garantir a precisão, é importante notar que traduções automáticas podem conter erros ou imprecisões. O documento original na sua língua nativa deve ser considerado a fonte autoritária. Para informações críticas, recomenda-se a tradução profissional realizada por humanos. Não nos responsabilizamos por quaisquer mal-entendidos ou interpretações incorretas decorrentes da utilização desta tradução.