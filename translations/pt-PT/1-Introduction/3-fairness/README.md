# Construir soluções de Machine Learning com IA responsável
 
![Resumo da IA responsável em Machine Learning num sketchnote](../../../../translated_images/pt-PT/ml-fairness.ef296ebec6afc98a.webp)
> Sketchnote por [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Questionário pré-aula](https://ff-quizzes.netlify.app/en/ml/)
 
## Introdução

Neste currículo, irá começar a descobrir como o machine learning pode e está a impactar a nossa vida quotidiana. Mesmo agora, sistemas e modelos estão envolvidos em tarefas diárias de tomada de decisão, como diagnósticos de saúde, aprovações de empréstimos ou deteção de fraude. Por isso, é importante que estes modelos funcionem bem para fornecer resultados em que se possa confiar. Tal como qualquer aplicação de software, os sistemas de IA vão falhar expectativas ou terão um resultado indesejado. É por isso que é essencial conseguir compreender e explicar o comportamento de um modelo de IA.

Imagine o que pode acontecer quando os dados que está a usar para construir estes modelos não contêm certos grupos demográficos, como raça, género, visão política, religião, ou representam desproporcionalmente esses grupos demográficos. E se o resultado do modelo for interpretado para favorecer algum grupo demográfico? Qual a consequência para a aplicação? Além disso, o que acontece quando o modelo tem um resultado adverso e é prejudicial para as pessoas? Quem é responsável pelo comportamento dos sistemas de IA? Estas são algumas das questões que iremos explorar neste currículo.

Nesta lição, irá:

- Aumentar a sua consciência da importância da justiça em machine learning e dos danos relacionados com a justiça.
- Familiarizar-se com a prática de explorar outliers e cenários invulgares para garantir fiabilidade e segurança.
- Compreender a necessidade de capacitar todos ao desenhar sistemas inclusivos.
- Explorar a importância vital de proteger a privacidade e a segurança dos dados e das pessoas.
- Ver a importância de ter uma abordagem de caixa transparente para explicar o comportamento dos modelos de IA.
- Ser consciente de como a responsabilidade é essencial para construir confiança em sistemas de IA.

## Pré-requisitos

Como pré-requisito, por favor faça o Learn Path "Princípios de IA Responsável" e assista ao vídeo abaixo sobre o tema:

Saiba mais sobre IA Responsável seguindo este [Learning Path](https://docs.microsoft.com/learn/modules/responsible-ai-principles/?WT.mc_id=academic-77952-leestott)

[![Abordagem da Microsoft à IA responsável](https://img.youtube.com/vi/dnC8-uUZXSc/0.jpg)](https://youtu.be/dnC8-uUZXSc "Abordagem da Microsoft à IA responsável")

> 🎥 Clique na imagem acima para um vídeo: Abordagem da Microsoft à IA responsável

## Justiça

Os sistemas de IA devem tratar todos com justiça e evitar afetar grupos semelhantes de pessoas de forma diferente. Por exemplo, quando sistemas de IA fornecem orientação sobre tratamentos médicos, candidaturas a empréstimos ou emprego, devem fazer as mesmas recomendações a todos com sintomas semelhantes, circunstâncias financeiras ou qualificações profissionais. Cada um de nós, como humanos, carrega preconceitos herdados que afetam as nossas decisões e ações. Esses preconceitos podem estar evidentes nos dados que usamos para treinar sistemas de IA. Essa manipulação pode ocorrer às vezes involuntariamente. Muitas vezes é difícil saber conscientemente quando estamos a introduzir preconceito nos dados.

**“Injustiça”** engloba impactos negativos, ou “danos”, para um grupo de pessoas, como aqueles definidos em termos de raça, género, idade ou estado de deficiência. Os principais danos relacionados com a justiça podem ser classificados como:

- **Alocação**, se, por exemplo, um género ou etnia é favorecido em detrimento de outro.
- **Qualidade do serviço**. Se treinar os dados para um cenário específico, mas a realidade for muito mais complexa, leva a um serviço com fraco desempenho. Por exemplo, um dispensador de sabão que aparentemente não conseguia detetar pessoas com pele escura. [Referência](https://gizmodo.com/why-cant-this-soap-dispenser-identify-dark-skin-1797931773)
- **Difamação**. Criticar injustamente e rotular algo ou alguém. Por exemplo, uma tecnologia de rotulagem de imagens rotulou infamemente imagens de pessoas de pele escura como gorilas.
- **Sobre ou subrepresentação**. A ideia é que um certo grupo não é visto numa determinada profissão, e qualquer serviço ou função que continue a promover isso está a contribuir para o dano.
- **Estereotipagem**. Associar um dado grupo a atributos pré-atribuídos. Por exemplo, um sistema de tradução linguística entre inglês e turco pode apresentar imprecisões devido a palavras com associações estereotipadas ao género.

![tradução para turco](../../../../translated_images/pt-PT/gender-bias-translate-en-tr.f185fd8822c2d437.webp)
> tradução para turco

![tradução de volta para inglês](../../../../translated_images/pt-PT/gender-bias-translate-tr-en.4eee7e3cecb8c70e.webp)
> tradução de volta para inglês

Ao desenhar e testar sistemas de IA, precisamos garantir que a IA é justa e não programada para tomar decisões enviesadas ou discriminatórias, que os próprios humanos também são proibidos de tomar. Garantir a justiça na IA e no machine learning continua a ser um desafio sociotécnico complexo.

### Fiabilidade e segurança

Para construir confiança, os sistemas de IA precisam de ser fiáveis, seguros e consistentes em condições normais e inesperadas. É importante saber como os sistemas de IA vão comportar-se numa variedade de situações, especialmente quando são outliers. Ao construir soluções de IA, deve haver uma atenção substancial sobre como lidar com uma grande variedade de circunstâncias que as soluções IA possam encontrar. Por exemplo, um carro autónomo tem de colocar a segurança das pessoas como prioridade máxima. Como resultado, a IA que alimenta o carro precisa de considerar todos os possíveis cenários que o carro pode encontrar, tais como de noite, tempestades ou nevascas, crianças a atravessarem a rua, animais de estimação, obras na estrada, etc. Quão bem um sistema de IA consegue lidar com uma ampla gama de condições fiável e seguramente reflete o nível de antecipação que o cientista de dados ou o programador de IA consideraram durante o design ou teste do sistema.

> [🎥 Clique aqui para um vídeo: ](https://www.microsoft.com/videoplayer/embed/RE4vvIl)

### Inclusividade

Os sistemas de IA devem ser desenhados para envolver e capacitar todos. Ao desenhar e implementar sistemas de IA, os cientistas de dados e programadores identificam e resolvem potenciais barreiras no sistema que poderiam excluir pessoas involuntariamente. Por exemplo, existem 1 bilião de pessoas com deficiência em todo o mundo. Com o avanço da IA, podem aceder a uma ampla gama de informações e oportunidades mais facilmente no seu dia a dia. Ao resolver as barreiras, criam-se oportunidades para inovar e desenvolver produtos de IA com melhores experiências que beneficiem todos.

> [🎥 Clique aqui para um vídeo: inclusividade na IA](https://www.microsoft.com/videoplayer/embed/RE4vl9v)

### Segurança e privacidade

Os sistemas de IA devem ser seguros e respeitar a privacidade das pessoas. As pessoas confiam menos em sistemas que colocam a sua privacidade, informação ou vidas em risco. Ao treinar modelos de machine learning, dependemos dos dados para produzir os melhores resultados. Ao fazê-lo, a origem dos dados e a integridade devem ser consideradas. Por exemplo, os dados foram submetidos pelo utilizador ou estavam disponíveis publicamente? Depois, ao trabalhar com os dados, é crucial desenvolver sistemas de IA que possam proteger informações confidenciais e resistir a ataques. À medida que a IA se revela mais comum, proteger a privacidade e garantir a segurança da informação pessoal e empresarial importante torna-se mais crítico e complexo. Questões de privacidade e segurança de dados requerem especial atenção para a IA porque o acesso aos dados é essencial para que os sistemas de IA façam previsões e decisões precisas e informadas sobre as pessoas.

> [🎥 Clique aqui para um vídeo: segurança na IA](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Como indústria, fizemos avanços significativos em Privacidade & segurança, impulsionados significativamente por regulamentos como o GDPR (Regulamento Geral de Proteção de Dados).
- Contudo, com sistemas de IA, devemos reconhecer a tensão entre a necessidade de mais dados pessoais para tornar os sistemas mais pessoais e eficazes – e a privacidade.
- Tal como com o nascimento dos computadores ligados à internet, estamos a ver também um grande aumento no número de problemas de segurança relacionados com a IA.
- Ao mesmo tempo, temos visto a IA a ser usada para melhorar a segurança. Por exemplo, a maioria dos scanners antivírus modernos hoje é guiada por heurísticas de IA.
- Precisamos garantir que os nossos processos de Ciência de Dados se misturam harmoniosamente com as práticas mais recentes de privacidade e segurança.


### Transparência
Os sistemas de IA devem ser compreensíveis. Uma parte crucial da transparência é explicar o comportamento dos sistemas de IA e dos seus componentes. Melhorar a compreensão dos sistemas de IA requer que as partes interessadas percebam como e por que funcionam, para poderem identificar potenciais problemas de desempenho, preocupações de segurança e privacidade, preconceitos, práticas exclusivas ou resultados indesejados. Também acreditamos que quem usa sistemas de IA deve ser honesto e transparente sobre quando, porquê e como decide implementá-los, bem como sobre as limitações dos sistemas utilizados. Por exemplo, se um banco usa um sistema de IA para suportar as suas decisões de crédito ao consumidor, é importante examinar os resultados e perceber quais os dados que influenciam as recomendações do sistema. Os governos estão a começar a regular a IA em várias indústrias, pelo que os cientistas de dados e organizações devem explicar se um sistema de IA cumpre os requisitos regulamentares, especialmente quando ocorre um resultado indesejado.

> [🎥 Clique aqui para um vídeo: transparência na IA](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Porque os sistemas de IA são tão complexos, é difícil compreender como funcionam e interpretar os resultados.
- Esta falta de compreensão afeta a forma como esses sistemas são geridos, operacionalizados e documentados.
- Esta falta de compreensão, mais importante, afeta as decisões tomadas com base nos resultados que esses sistemas produzem.

### Responsabilização
 
As pessoas que desenham e implementam sistemas de IA devem ser responsáveis por como os seus sistemas operam. A necessidade de responsabilização é particularmente crucial com tecnologias de uso sensível como o reconhecimento facial. Recentemente, tem havido uma crescente procura pela tecnologia de reconhecimento facial, especialmente por parte de organizações de aplicação da lei que veem o potencial da tecnologia em usos como encontrar crianças desaparecidas. No entanto, essas tecnologias podem ser potencialmente usadas por um governo para pôr em risco as liberdades fundamentais dos seus cidadãos, por exemplo, permitindo vigilância contínua de indivíduos específicos. Por isso, os cientistas de dados e organizações precisam de ser responsáveis pelo impacto do seu sistema de IA sobre indivíduos ou a sociedade.

[![Investigador principal em IA alerta para vigilância em massa através do reconhecimento facial](../../../../translated_images/pt-PT/accountability.41d8c0f4b85b6231.webp)](https://www.youtube.com/watch?v=Wldt8P5V6D0 "Abordagem da Microsoft à IA responsável")

> 🎥 Clique na imagem acima para um vídeo: Avisos sobre vigilância em massa através do reconhecimento facial

Em última análise, uma das maiores questões para a nossa geração, como a primeira geração que está a trazer a IA para a sociedade, é como garantir que os computadores permanecerão responsáveis perante as pessoas e como garantir que as pessoas que desenham computadores permanecem responsáveis perante todos os outros.

## Avaliação do impacto

Antes de treinar um modelo de machine learning, é importante realizar uma avaliação de impacto para entender o propósito do sistema de IA; qual é o uso pretendido; onde será implementado; e quem interagirá com o sistema. Isto é útil para revisores ou testadores avaliarem o sistema e saberem que fatores considerar ao identificar riscos potenciais e consequências esperadas.

As seguintes são áreas de foco ao conduzir uma avaliação de impacto:

* **Impacto adverso sobre indivíduos**. Estar ciente de quaisquer restrições ou requisitos, uso não suportado ou quaisquer limitações conhecidas que prejudiquem o desempenho do sistema é vital para garantir que o sistema não seja usado de forma a causar dano às pessoas.
* **Requisitos de dados**. Compreender como e onde o sistema usará dados permite aos revisores explorar quaisquer requisitos de dados que deva ter em conta (exemplos: regulamentos GDPR ou HIPAA). Além disso, analisar se a fonte ou quantidade de dados é substancial para o treino.
* **Resumo do impacto**. Reunir uma lista de potenciais danos que podem surgir do uso do sistema. Ao longo do ciclo de vida da ML, rever se os problemas identificados são mitigados ou resolvidos.
* **Objetivos aplicáveis** para cada um dos seis princípios centrais. Avaliar se os objetivos de cada um dos princípios são cumpridos e se existem lacunas.


## Depuração com IA responsável

Tal como depurar uma aplicação de software, depurar um sistema de IA é um processo necessário de identificar e resolver problemas no sistema. Existem muitos fatores que podem fazer com que um modelo não funcione conforme o esperado ou responsável. A maioria das métricas tradicionais de desempenho do modelo são agregados quantitativos do desempenho do modelo, que não são suficientes para analisar como um modelo viola os princípios de IA responsável. Além disso, um modelo de machine learning é uma caixa negra, o que dificulta perceber o que conduz ao seu resultado ou fornecer explicações quando comete um erro. Mais adiante neste curso, aprenderemos como usar o dashboard de IA responsável para ajudar a depurar sistemas de IA. O dashboard oferece uma ferramenta holística para cientistas de dados e programadores de IA realizarem:

* **Análise de erros**. Para identificar a distribuição de erros do modelo que pode afetar a justiça ou fiabilidade do sistema.
* **Visão geral do modelo**. Para descobrir onde há disparidades no desempenho do modelo através de grupos de dados.
* **Análise de dados**. Para entender a distribuição dos dados e identificar qualquer possível viés nos dados que possa levar a problemas de justiça, inclusividade e fiabilidade.
* **Interpretabilidade do modelo**. Para compreender o que afeta ou influencia as previsões do modelo. Isso ajuda a explicar o comportamento do modelo, importante para transparência e responsabilidade.


## 🚀 Desafio
 
Para prevenir que danos sejam introduzidos em primeiro lugar, devemos:

- ter uma diversidade de origens e perspetivas entre as pessoas que trabalham nos sistemas
- investir em conjuntos de dados que reflitam a diversidade da nossa sociedade
- desenvolver melhores métodos ao longo do ciclo de vida do machine learning para detetar e corrigir IA responsável quando esta ocorrer

Pense em cenários da vida real onde a falta de confiança num modelo é evidente na construção e uso do modelo. O que mais deveríamos considerar?

## [Questionário pós-aula](https://ff-quizzes.netlify.app/en/ml/)

## Revisão & Estudo autónomo
 

Nesta lição, aprendeu alguns conceitos básicos sobre justiça e injustiça no machine learning.  
 
Veja este workshop para aprofundar os temas: 

- Na busca por uma IA responsável: Aplicar princípios na prática por Besmira Nushi, Mehrnoosh Sameki e Amit Sharma

[![Responsible AI Toolbox: An open-source framework for building responsible AI](https://img.youtube.com/vi/tGgJCrA-MZU/0.jpg)](https://www.youtube.com/watch?v=tGgJCrA-MZU "RAI Toolbox: An open-source framework for building responsible AI")

> 🎥 Clique na imagem acima para um vídeo: RAI Toolbox: Um framework open-source para construir IA responsável por Besmira Nushi, Mehrnoosh Sameki, e Amit Sharma

Também leia: 

- Centro de recursos de RAI da Microsoft: [Responsible AI Resources – Microsoft AI](https://www.microsoft.com/ai/responsible-ai-resources?activetab=pivot1%3aprimaryr4) 

- Grupo de investigação FATE da Microsoft: [FATE: Fairness, Accountability, Transparency, and Ethics in AI - Microsoft Research](https://www.microsoft.com/research/theme/fate/) 

RAI Toolbox: 

- [Repositório GitHub do Responsible AI Toolbox](https://github.com/microsoft/responsible-ai-toolbox)

Leia sobre as ferramentas do Azure Machine Learning para garantir a justiça:

- [Azure Machine Learning](https://docs.microsoft.com/azure/machine-learning/concept-fairness-ml?WT.mc_id=academic-77952-leestott) 

## Tarefa

[Explore o RAI Toolbox](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Aviso Legal**:
Este documento foi traduzido utilizando o serviço de tradução automática [Co-op Translator](https://github.com/Azure/co-op-translator). Embora nos esforcemos pela precisão, esteja ciente de que traduções automáticas podem conter erros ou imprecisões. O documento original na sua língua nativa deve ser considerado a fonte autorizada. Para informações críticas, recomenda-se tradução profissional humana. Não nos responsabilizamos por quaisquer mal-entendidos ou interpretações incorretas resultantes da utilização desta tradução.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->