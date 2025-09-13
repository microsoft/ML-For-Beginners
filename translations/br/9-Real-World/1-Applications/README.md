<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "83320d6b6994909e35d830cebf214039",
  "translation_date": "2025-09-04T21:31:29+00:00",
  "source_file": "9-Real-World/1-Applications/README.md",
  "language_code": "br"
}
-->
# Pós-escrito: Aprendizado de máquina no mundo real

![Resumo do aprendizado de máquina no mundo real em um sketchnote](../../../../sketchnotes/ml-realworld.png)
> Sketchnote por [Tomomi Imura](https://www.twitter.com/girlie_mac)

Neste currículo, você aprendeu várias maneiras de preparar dados para treinamento e criar modelos de aprendizado de máquina. Você construiu uma série de modelos clássicos de regressão, agrupamento, classificação, processamento de linguagem natural e séries temporais. Parabéns! Agora, você pode estar se perguntando para que tudo isso serve... quais são as aplicações reais desses modelos?

Embora a inteligência artificial, que geralmente utiliza aprendizado profundo, tenha atraído muito interesse na indústria, ainda existem aplicações valiosas para modelos clássicos de aprendizado de máquina. Você pode até estar usando algumas dessas aplicações hoje! Nesta lição, você explorará como oito diferentes indústrias e áreas de conhecimento utilizam esses tipos de modelos para tornar suas aplicações mais eficientes, confiáveis, inteligentes e valiosas para os usuários.

## [Quiz pré-aula](https://ff-quizzes.netlify.app/en/ml/)

## 💰 Finanças

O setor financeiro oferece muitas oportunidades para o aprendizado de máquina. Muitos problemas nessa área podem ser modelados e resolvidos usando ML.

### Detecção de fraude em cartões de crédito

Aprendemos sobre [agrupamento k-means](../../5-Clustering/2-K-Means/README.md) anteriormente no curso, mas como ele pode ser usado para resolver problemas relacionados à fraude em cartões de crédito?

O agrupamento k-means é útil em uma técnica de detecção de fraude chamada **detecção de outliers**. Outliers, ou desvios nas observações de um conjunto de dados, podem nos dizer se um cartão de crédito está sendo usado de forma normal ou se algo incomum está acontecendo. Como mostrado no artigo abaixo, você pode classificar dados de cartões de crédito usando um algoritmo de agrupamento k-means e atribuir cada transação a um grupo com base no quanto ela parece ser um outlier. Em seguida, você pode avaliar os grupos mais arriscados para identificar transações fraudulentas versus legítimas.  
[Referência](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.680.1195&rep=rep1&type=pdf)

### Gestão de patrimônio

Na gestão de patrimônio, um indivíduo ou empresa administra investimentos em nome de seus clientes. O objetivo é sustentar e aumentar a riqueza a longo prazo, sendo essencial escolher investimentos que tenham bom desempenho.

Uma maneira de avaliar o desempenho de um investimento é por meio de regressão estatística. [Regressão linear](../../2-Regression/1-Tools/README.md) é uma ferramenta valiosa para entender como um fundo se comporta em relação a um benchmark. Também podemos deduzir se os resultados da regressão são estatisticamente significativos ou o quanto eles impactariam os investimentos de um cliente. Você pode expandir ainda mais sua análise usando regressão múltipla, onde fatores de risco adicionais podem ser considerados. Para um exemplo de como isso funcionaria para um fundo específico, confira o artigo abaixo sobre avaliação de desempenho de fundos usando regressão.  
[Referência](http://www.brightwoodventures.com/evaluating-fund-performance-using-regression/)

## 🎓 Educação

O setor educacional também é uma área muito interessante onde o ML pode ser aplicado. Existem problemas intrigantes a serem resolvidos, como detectar trapaças em testes ou redações e gerenciar vieses, intencionais ou não, no processo de correção.

### Previsão de comportamento estudantil

[Coursera](https://coursera.com), um provedor de cursos online abertos, tem um ótimo blog técnico onde discutem muitas decisões de engenharia. Neste estudo de caso, eles traçaram uma linha de regressão para explorar qualquer correlação entre uma baixa classificação de NPS (Net Promoter Score) e a retenção ou abandono de cursos.  
[Referência](https://medium.com/coursera-engineering/controlled-regression-quantifying-the-impact-of-course-quality-on-learner-retention-31f956bd592a)

### Mitigação de vieses

[Grammarly](https://grammarly.com), um assistente de escrita que verifica erros de ortografia e gramática, utiliza sistemas sofisticados de [processamento de linguagem natural](../../6-NLP/README.md) em seus produtos. Eles publicaram um estudo de caso interessante em seu blog técnico sobre como lidaram com o viés de gênero no aprendizado de máquina, algo que você aprendeu em nossa [lição introdutória sobre justiça](../../1-Introduction/3-fairness/README.md).  
[Referência](https://www.grammarly.com/blog/engineering/mitigating-gender-bias-in-autocorrect/)

## 👜 Varejo

O setor de varejo pode se beneficiar muito do uso de ML, desde a criação de uma jornada do cliente mais personalizada até o gerenciamento otimizado de estoques.

### Personalização da jornada do cliente

Na Wayfair, uma empresa que vende produtos para casa, como móveis, ajudar os clientes a encontrar os produtos certos para seus gostos e necessidades é fundamental. Neste artigo, engenheiros da empresa descrevem como utilizam ML e NLP para "apresentar os resultados certos para os clientes". Notavelmente, seu motor de intenção de consulta foi construído para usar extração de entidades, treinamento de classificadores, extração de ativos e opiniões, e marcação de sentimentos em avaliações de clientes. Este é um caso clássico de como o NLP funciona no varejo online.  
[Referência](https://www.aboutwayfair.com/tech-innovation/how-we-use-machine-learning-and-natural-language-processing-to-empower-search)

### Gestão de estoque

Empresas inovadoras e ágeis como [StitchFix](https://stitchfix.com), um serviço de assinatura que envia roupas para consumidores, dependem fortemente de ML para recomendações e gestão de estoque. Suas equipes de estilo trabalham em conjunto com suas equipes de merchandising. "Um de nossos cientistas de dados experimentou um algoritmo genético e o aplicou ao vestuário para prever o que seria uma peça de roupa bem-sucedida que ainda não existe. Levamos isso para a equipe de merchandising e agora eles podem usar isso como uma ferramenta."  
[Referência](https://www.zdnet.com/article/how-stitch-fix-uses-machine-learning-to-master-the-science-of-styling/)

## 🏥 Saúde

O setor de saúde pode aproveitar o ML para otimizar tarefas de pesquisa e também problemas logísticos, como readmissão de pacientes ou controle de doenças.

### Gestão de ensaios clínicos

A toxicidade em ensaios clínicos é uma grande preocupação para os fabricantes de medicamentos. Qual é o nível de toxicidade tolerável? Neste estudo, a análise de vários métodos de ensaios clínicos levou ao desenvolvimento de uma nova abordagem para prever as chances de resultados de ensaios clínicos. Especificamente, eles conseguiram usar random forest para produzir um [classificador](../../4-Classification/README.md) capaz de distinguir entre grupos de medicamentos.  
[Referência](https://www.sciencedirect.com/science/article/pii/S2451945616302914)

### Gestão de readmissão hospitalar

O atendimento hospitalar é caro, especialmente quando os pacientes precisam ser readmitidos. Este artigo discute uma empresa que usa ML para prever o potencial de readmissão usando algoritmos de [agrupamento](../../5-Clustering/README.md). Esses grupos ajudam os analistas a "descobrir grupos de readmissões que podem compartilhar uma causa comum".  
[Referência](https://healthmanagement.org/c/healthmanagement/issuearticle/hospital-readmissions-and-machine-learning)

### Gestão de doenças

A recente pandemia destacou como o aprendizado de máquina pode ajudar a conter a disseminação de doenças. Neste artigo, você reconhecerá o uso de ARIMA, curvas logísticas, regressão linear e SARIMA. "Este trabalho é uma tentativa de calcular a taxa de disseminação deste vírus e, assim, prever mortes, recuperações e casos confirmados, para que possamos nos preparar melhor e sobreviver."  
[Referência](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7979218/)

## 🌲 Ecologia e Tecnologia Verde

A natureza e a ecologia consistem em muitos sistemas sensíveis onde a interação entre animais e o meio ambiente entra em foco. É importante medir esses sistemas com precisão e agir adequadamente se algo acontecer, como um incêndio florestal ou uma queda na população animal.

### Gestão florestal

Você aprendeu sobre [Aprendizado por Reforço](../../8-Reinforcement/README.md) em lições anteriores. Ele pode ser muito útil ao tentar prever padrões na natureza. Em particular, pode ser usado para monitorar problemas ecológicos como incêndios florestais e a disseminação de espécies invasoras. No Canadá, um grupo de pesquisadores usou Aprendizado por Reforço para construir modelos de dinâmica de incêndios florestais a partir de imagens de satélite. Usando um inovador "processo de propagação espacial (SSP)", eles imaginaram um incêndio florestal como "o agente em qualquer célula na paisagem". "O conjunto de ações que o fogo pode realizar de um local em qualquer momento inclui se espalhar para o norte, sul, leste ou oeste ou não se espalhar."

Essa abordagem inverte o cenário usual de RL, já que a dinâmica do Processo de Decisão de Markov (MDP) correspondente é uma função conhecida para a propagação imediata do incêndio. Leia mais sobre os algoritmos clássicos usados por este grupo no link abaixo.  
[Referência](https://www.frontiersin.org/articles/10.3389/fict.2018.00006/full)

### Monitoramento de movimentos de animais

Embora o aprendizado profundo tenha revolucionado o rastreamento visual de movimentos de animais (você pode construir seu próprio [rastreador de ursos polares](https://docs.microsoft.com/learn/modules/build-ml-model-with-azure-stream-analytics/?WT.mc_id=academic-77952-leestott) aqui), o ML clássico ainda tem um lugar nessa tarefa.

Sensores para rastrear movimentos de animais de fazenda e IoT utilizam esse tipo de processamento visual, mas técnicas mais básicas de ML são úteis para pré-processar dados. Por exemplo, neste artigo, posturas de ovelhas foram monitoradas e analisadas usando vários algoritmos de classificação. Você pode reconhecer a curva ROC na página 335.  
[Referência](https://druckhaus-hofmann.de/gallery/31-wj-feb-2020.pdf)

### ⚡️ Gestão de Energia

Em nossas lições sobre [previsão de séries temporais](../../7-TimeSeries/README.md), invocamos o conceito de parquímetros inteligentes para gerar receita para uma cidade com base na compreensão de oferta e demanda. Este artigo discute em detalhes como agrupamento, regressão e previsão de séries temporais se combinam para ajudar a prever o uso futuro de energia na Irlanda, com base em medição inteligente.  
[Referência](https://www-cdn.knime.com/sites/default/files/inline-images/knime_bigdata_energy_timeseries_whitepaper.pdf)

## 💼 Seguros

O setor de seguros é outro setor que utiliza ML para construir e otimizar modelos financeiros e atuariais viáveis.

### Gestão de volatilidade

A MetLife, uma provedora de seguros de vida, é transparente sobre como analisa e mitiga a volatilidade em seus modelos financeiros. Neste artigo, você verá visualizações de classificação binária e ordinal. Também encontrará visualizações de previsão.  
[Referência](https://investments.metlife.com/content/dam/metlifecom/us/investments/insights/research-topics/macro-strategy/pdf/MetLifeInvestmentManagement_MachineLearnedRanking_070920.pdf)

## 🎨 Artes, Cultura e Literatura

Nas artes, por exemplo no jornalismo, há muitos problemas interessantes. Detectar notícias falsas é um grande desafio, pois já foi comprovado que elas influenciam a opinião das pessoas e até derrubam democracias. Museus também podem se beneficiar do uso de ML em tudo, desde encontrar conexões entre artefatos até planejamento de recursos.

### Detecção de notícias falsas

Detectar notícias falsas tornou-se um jogo de gato e rato na mídia atual. Neste artigo, pesquisadores sugerem que um sistema combinando várias das técnicas de ML que estudamos pode ser testado e o melhor modelo implantado: "Este sistema é baseado no processamento de linguagem natural para extrair características dos dados e, em seguida, essas características são usadas para o treinamento de classificadores de aprendizado de máquina, como Naive Bayes, Support Vector Machine (SVM), Random Forest (RF), Stochastic Gradient Descent (SGD) e Regressão Logística (LR)."  
[Referência](https://www.irjet.net/archives/V7/i6/IRJET-V7I6688.pdf)

Este artigo mostra como combinar diferentes domínios de ML pode produzir resultados interessantes que ajudam a impedir a disseminação de notícias falsas e a criação de danos reais; neste caso, o impulso foi a disseminação de rumores sobre tratamentos para COVID que incitaram violência em massa.

### ML em museus

Os museus estão na vanguarda de uma revolução da IA, onde catalogar e digitalizar coleções e encontrar conexões entre artefatos está se tornando mais fácil à medida que a tecnologia avança. Projetos como [In Codice Ratio](https://www.sciencedirect.com/science/article/abs/pii/S0306457321001035#:~:text=1.,studies%20over%20large%20historical%20sources.) estão ajudando a desvendar os mistérios de coleções inacessíveis, como os Arquivos do Vaticano. Mas o aspecto comercial dos museus também se beneficia de modelos de ML.

Por exemplo, o Instituto de Arte de Chicago construiu modelos para prever o que os públicos estão interessados e quando irão visitar exposições. O objetivo é criar experiências de visita individualizadas e otimizadas cada vez que o usuário visita o museu. "Durante o ano fiscal de 2017, o modelo previu a frequência e as admissões com uma precisão de 1%, diz Andrew Simnick, vice-presidente sênior do Instituto de Arte."  
[Referência](https://www.chicagobusiness.com/article/20180518/ISSUE01/180519840/art-institute-of-chicago-uses-data-to-make-exhibit-choices)

## 🏷 Marketing

### Segmentação de clientes

As estratégias de marketing mais eficazes segmentam os clientes de diferentes maneiras com base em vários agrupamentos. Neste artigo, os usos de algoritmos de agrupamento são discutidos para apoiar o marketing diferenciado. O marketing diferenciado ajuda as empresas a melhorar o reconhecimento da marca, alcançar mais clientes e gerar mais receita.  
[Referência](https://ai.inqline.com/machine-learning-for-marketing-customer-segmentation/)

## 🚀 Desafio

Identifique outro setor que se beneficie de algumas das técnicas que você aprendeu neste currículo e descubra como ele utiliza ML.
## [Quiz pós-aula](https://ff-quizzes.netlify.app/en/ml/)

## Revisão e Autoestudo

A equipe de ciência de dados da Wayfair tem vários vídeos interessantes sobre como eles utilizam ML na empresa. Vale a pena [dar uma olhada](https://www.youtube.com/channel/UCe2PjkQXqOuwkW1gw6Ameuw/videos)!

## Tarefa

[Uma caça ao tesouro de ML](assignment.md)

---

**Aviso Legal**:  
Este documento foi traduzido utilizando o serviço de tradução por IA [Co-op Translator](https://github.com/Azure/co-op-translator). Embora nos esforcemos para garantir a precisão, esteja ciente de que traduções automatizadas podem conter erros ou imprecisões. O documento original em seu idioma nativo deve ser considerado a fonte autoritativa. Para informações críticas, recomenda-se a tradução profissional realizada por humanos. Não nos responsabilizamos por quaisquer mal-entendidos ou interpretações equivocadas decorrentes do uso desta tradução.