# Construindo soluções de Machine Learning com IA responsável
 
![Resumo da IA responsável em Machine Learning em um sketchnote](../../../../translated_images/pt-BR/ml-fairness.ef296ebec6afc98a.webp)
> Sketchnote por [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Quiz pré-aula](https://ff-quizzes.netlify.app/en/ml/)
 
## Introdução

Neste currículo, você começará a descobrir como o machine learning pode e está impactando nosso dia a dia. Mesmo agora, sistemas e modelos estão envolvidos em tarefas diárias de tomada de decisão, como diagnósticos médicos, aprovações de empréstimos ou detecção de fraudes. Portanto, é importante que esses modelos funcionem bem para fornecer resultados confiáveis. Assim como qualquer aplicação de software, sistemas de IA podem falhar em atender expectativas ou produzir resultados indesejados. Por isso, é essencial ser capaz de entender e explicar o comportamento de um modelo de IA.

Imagine o que pode acontecer quando os dados usados para construir esses modelos não incluem certos grupos demográficos, como raça, gênero, visão política, religião, ou representam esses grupos de forma desproporcional. E quando a saída do modelo é interpretada para favorecer algum grupo demográfico? Qual é a consequência para a aplicação? Além disso, o que acontece quando o modelo tem um resultado adverso e prejudica pessoas? Quem é responsável pelo comportamento dos sistemas de IA? Estas são algumas perguntas que exploraremos neste currículo.

Nesta aula, você irá:

- Elevar sua consciência sobre a importância da justiça em machine learning e os danos relacionados à justiça.
- Familiarizar-se com a prática de explorar casos fora do comum e cenários incomuns para garantir confiabilidade e segurança.
- Compreender a necessidade de capacitar todos ao projetar sistemas inclusivos.
- Explorar a importância vital de proteger a privacidade e segurança dos dados e das pessoas.
- Perceber a importância de uma abordagem transparente para explicar o comportamento dos modelos de IA.
- Estar atento a como a responsabilidade é essencial para construir confiança em sistemas de IA.

## Pré-requisito

Como pré-requisito, por favor realize a Jornada de Aprendizagem "Princípios de IA Responsável" e assista o vídeo abaixo sobre o tema:

Saiba mais sobre IA Responsável seguindo esta [Jornada de Aprendizagem](https://docs.microsoft.com/learn/modules/responsible-ai-principles/?WT.mc_id=academic-77952-leestott)

[![Abordagem da Microsoft para IA Responsável](https://img.youtube.com/vi/dnC8-uUZXSc/0.jpg)](https://youtu.be/dnC8-uUZXSc "Abordagem da Microsoft para IA Responsável")

> 🎥 Clique na imagem acima para um vídeo: Abordagem da Microsoft para IA Responsável

## Justiça

Sistemas de IA devem tratar todos com justiça e evitar afetar grupos semelhantes de maneiras diferentes. Por exemplo, quando sistemas de IA fornecem orientações para tratamentos médicos, solicitações de empréstimos ou emprego, eles devem fazer as mesmas recomendações para todos com sintomas, condições financeiras ou qualificações profissionais similares. Cada um de nós carrega preconceitos herdados que afetam nossas decisões e ações. Esses preconceitos podem estar evidentes nos dados usados para treinar sistemas de IA. Essa manipulação pode ocorrer sem intenção. Frequentemente é difícil saber conscientemente quando estamos introduzindo viés nos dados.

**“Injustiça”** engloba impactos negativos, ou “danos”, para um grupo de pessoas, como aqueles definidos por raça, gênero, idade ou status de deficiência. Os principais danos relacionados à justiça podem ser classificados como:

- **Alocação**, se um gênero ou etnia, por exemplo, é favorecido em detrimento de outro.
- **Qualidade de serviço**. Se você treina dados para um cenário específico, mas a realidade é muito mais complexa, isso leva a um serviço de baixa performance. Por exemplo, um dispenser de sabonete que não consegue detectar pessoas com pele escura. [Referência](https://gizmodo.com/why-cant-this-soap-dispenser-identify-dark-skin-1797931773)
- **Denigreção**. Criticar e rotular algo ou alguém injustamente. Por exemplo, uma tecnologia de rotulagem de imagens que infamemente rotulou pessoas de pele escura como gorilas.
- **Super ou sub-representação**. A ideia é que um determinado grupo não é visto em uma profissão, e qualquer serviço ou função que continue promovendo isso contribui para o dano.
- **Estereotipagem**. Associar um grupo com atributos pré-definidos. Por exemplo, um sistema de tradução de inglês para turco pode ter imprecisões devido a palavras com associações estereotipadas de gênero.

![tradução para o turco](../../../../translated_images/pt-BR/gender-bias-translate-en-tr.f185fd8822c2d437.webp)
> tradução para o turco

![tradução de volta para o inglês](../../../../translated_images/pt-BR/gender-bias-translate-tr-en.4eee7e3cecb8c70e.webp)
> tradução de volta para o inglês

Ao projetar e testar sistemas de IA, precisamos garantir que a IA seja justa e não programada para tomar decisões enviesadas ou discriminatórias, o que humanos também são proibidos de fazer. Garantir justiça em IA e machine learning continua um desafio sociotécnico complexo.

### Confiabilidade e segurança

Para construir confiança, sistemas de IA precisam ser confiáveis, seguros e consistentes em condições normais e inesperadas. É importante saber como sistemas de IA se comportam em diferentes situações, especialmente quando são exceções. Ao construir soluções de IA, deve haver grande foco em como lidar com uma variedade ampla de circunstâncias. Por exemplo, um carro autônomo precisa colocar a segurança das pessoas como prioridade máxima. Assim, a IA que o alimenta precisa considerar todos os possíveis cenários que o carro possa encontrar, como noite, tempestades ou nevascas, crianças correndo na rua, animais de estimação, construções na via, etc. Quão bem um sistema de IA consegue lidar com uma gama ampla de condições de forma confiável e segura reflete o nível de antecipação que o cientista de dados ou desenvolvedor de IA considerou durante o design ou teste do sistema.

> [🎥 Clique aqui para um vídeo: Confiabilidade e segurança na IA](https://www.microsoft.com/videoplayer/embed/RE4vvIl)

### Inclusividade

Sistemas de IA devem ser projetados para engajar e capacitar a todos. Ao projetar e implementar sistemas de IA, cientistas de dados e desenvolvedores de IA identificam e eliminam barreiras que possam excluir pessoas sem intenção. Por exemplo, existem 1 bilhão de pessoas com deficiência no mundo. Com o avanço da IA, elas podem acessar uma ampla gama de informações e oportunidades mais facilmente no dia a dia. Ao eliminar barreiras, cria-se oportunidades para inovar e desenvolver produtos de IA com melhores experiências que beneficiem a todos.

> [🎥 Clique aqui para um vídeo: Inclusividade na IA](https://www.microsoft.com/videoplayer/embed/RE4vl9v)

### Segurança e privacidade

Sistemas de IA devem ser seguros e respeitar a privacidade das pessoas. Pessoas confiam menos em sistemas que colocam sua privacidade, informações ou vidas em risco. Ao treinar modelos de machine learning, dependemos dos dados para produzir os melhores resultados. Para isso, a origem dos dados e sua integridade devem ser consideradas. Por exemplo, os dados foram submetidos pelos usuários ou são públicos? Em seguida, ao trabalhar com os dados, é crucial desenvolver sistemas de IA que protejam informações confidenciais e resistam a ataques. À medida que a IA se torna mais difundida, proteger a privacidade e a segurança de informações pessoais e empresariais importantes torna-se mais crítico e complexo. Questões de privacidade e segurança dos dados exigem atenção especial para IA, pois o acesso a dados é essencial para que sistemas de IA façam previsões e decisões precisas e informadas sobre pessoas.

> [🎥 Clique aqui para um vídeo: Segurança na IA](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Como setor, fizemos avanços significativos em Privacidade e segurança, impulsionados principalmente por regulamentações como o GDPR (Regulamento Geral de Proteção de Dados).
- Ainda assim, com sistemas de IA devemos reconhecer a tensão entre a necessidade de mais dados pessoais para tornar sistemas mais pessoais e eficazes – e a privacidade.
- Assim como com o surgimento dos computadores conectados à internet, também estamos vendo grande aumento nas questões de segurança relacionadas à IA.
- Ao mesmo tempo, temos visto a IA ser usada para melhorar a segurança. Por exemplo, os antivírus modernos são impulsionados por heurísticas de IA hoje em dia.
- Precisamos garantir que nossos processos de Ciência de Dados estejam alinhados harmonicamente às práticas mais recentes de privacidade e segurança.


### Transparência
Sistemas de IA devem ser compreensíveis. Parte crucial da transparência é explicar o comportamento dos sistemas de IA e seus componentes. Melhorar o entendimento dos sistemas de IA exige que as partes interessadas compreendam como e por que eles funcionam para identificar possíveis problemas de desempenho, preocupações de segurança e privacidade, vieses, práticas excludentes ou resultados inesperados. Também acreditamos que quem usa sistemas de IA deve ser honesto e transparente sobre quando, por que e como decide implantá-los, assim como sobre as limitações dos sistemas usados. Por exemplo, se um banco usa um sistema de IA para apoiar decisões de concessão de crédito, é importante examinar os resultados e entender quais dados influenciam as recomendações do sistema. Governos estão começando a regulamentar a IA em diferentes setores, portanto cientistas de dados e organizações devem explicar se um sistema de IA atende aos requisitos regulatórios, especialmente quando há um resultado indesejado.

> [🎥 Clique aqui para um vídeo: Transparência na IA](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Como sistemas de IA são tão complexos, é difícil entender como funcionam e interpretar os resultados.
- Essa falta de entendimento afeta a forma como esses sistemas são gerenciados, operacionalizados e documentados.
- Mais importante, essa falta de entendimento afeta as decisões tomadas com base nos resultados produzidos por esses sistemas.

### Responsabilidade
 
As pessoas que projetam e implementam sistemas de IA devem ser responsáveis pelo funcionamento desses sistemas. A necessidade de responsabilidade é especialmente crucial em tecnologias de uso sensível, como reconhecimento facial. Recentemente, houve crescente demanda por essa tecnologia, especialmente por forças policiais que veem potencial em usos como encontrar crianças desaparecidas. Contudo, essas tecnologias podem ser usadas por governos para colocar em risco as liberdades fundamentais dos cidadãos por meio de, por exemplo, vigilância contínua de indivíduos específicos. Portanto, cientistas de dados e organizações precisam ser responsáveis pelo impacto que seus sistemas de IA causam em indivíduos ou na sociedade.

[![Pesquisador líder em IA alerta sobre vigilância em massa por reconhecimento facial](../../../../translated_images/pt-BR/accountability.41d8c0f4b85b6231.webp)](https://www.youtube.com/watch?v=Wldt8P5V6D0 "Abordagem da Microsoft para IA Responsável")

> 🎥 Clique na imagem acima para um vídeo: Avisos sobre vigilância em massa por reconhecimento facial

Em última análise, uma das maiores questões para a nossa geração, como a primeira que traz a IA para a sociedade, é como garantir que computadores permaneçam responsáveis perante as pessoas e que aqueles que projetam computadores permaneçam responsáveis perante todos os outros.

## Avaliação de impacto

Antes de treinar um modelo de machine learning, é importante realizar uma avaliação de impacto para entender o propósito do sistema de IA; o uso pretendido; onde ele será implantado; e quem irá interagir com o sistema. Isso é útil para revisores ou testadores avaliarem fatores a considerar ao identificar riscos potenciais e consequências esperadas.

As seguintes áreas são foco ao conduzir uma avaliação de impacto:

* **Impacto adverso em indivíduos**. Estar ciente de quaisquer restrições ou requisitos, uso não suportado ou limitações conhecidas que prejudiquem o desempenho do sistema é vital para garantir que o sistema não seja usado de forma a causar danos a pessoas.
* **Requisitos de dados**. Compreender como e onde o sistema usará os dados permite aos revisores explorar quaisquer requisitos de dados que precisem ser observados (ex.: regulamentos GDPR ou HIPAA). Além disso, analisar se a fonte ou quantidade de dados é substancial para o treinamento.
* **Resumo do impacto**. Elaborar uma lista de possíveis danos que podem surgir do uso do sistema. Ao longo do ciclo de vida do ML, revisar se os problemas identificados foram mitigados ou tratados.
* **Metas aplicáveis** para cada um dos seis princípios principais. Avaliar se as metas de cada princípio foram atingidas e se há lacunas.


## Depuração com IA responsável

Semelhante a depurar uma aplicação de software, depurar um sistema de IA é um processo necessário de identificar e resolver problemas no sistema. Muitos fatores podem afetar desempenho inadequado ou irresponsável de um modelo. A maioria das métricas tradicionais de desempenho são agregados quantitativos do modelo, insuficientes para analisar como um modelo viola princípios de IA responsável. Além disso, um modelo de machine learning é uma caixa preta que dificulta entender o que motiva seu resultado ou explicar erros. Mais adiante no curso, aprenderemos a usar o painel de IA Responsável para ajudar a depurar sistemas de IA. O painel oferece uma ferramenta holística para cientistas de dados e desenvolvedores de IA realizarem:

* **Análise de erros**. Identificar a distribuição de erro do modelo que pode afetar justiça ou confiabilidade do sistema.
* **Visão geral do modelo**. Descobrir onde há disparidades no desempenho do modelo entre coortes de dados.
* **Análise de dados**. Entender a distribuição dos dados e identificar potenciais vieses que possam levar a problemas de justiça, inclusão e confiabilidade.
* **Interpretabilidade do modelo**. Entender o que afeta ou influencia as previsões do modelo. Isso ajuda a explicar o comportamento do modelo, importante para transparência e responsabilidade.


## 🚀 Desafio
 
Para prevenir danos desde o início, devemos:

- ter diversidade de origens e perspectivas entre as pessoas que trabalham nos sistemas
- investir em conjuntos de dados que reflitam a diversidade da nossa sociedade
- desenvolver melhores métodos ao longo do ciclo de vida do machine learning para detectar e corrigir IA irresponsável quando ocorrer

Pense em cenários reais onde a falta de confiabilidade de um modelo é evidente na construção e uso do modelo. O que mais devemos considerar?

## [Quiz pós-aula](https://ff-quizzes.netlify.app/en/ml/)

## Revisão & Autoestudo
 
Nesta aula, você aprendeu noções básicas sobre os conceitos de justiça e injustiça em machine learning.
 
Assista a este workshop para aprofundar os tópicos:

- Em busca de IA responsável: trazendo princípios à prática por Besmira Nushi, Mehrnoosh Sameki e Amit Sharma

[![Responsible AI Toolbox: Uma estrutura de código aberto para construir IA responsável](https://img.youtube.com/vi/tGgJCrA-MZU/0.jpg)](https://www.youtube.com/watch?v=tGgJCrA-MZU "RAI Toolbox: Uma estrutura de código aberto para construir IA responsável")

> 🎥 Clique na imagem acima para um vídeo: RAI Toolbox: Uma estrutura de código aberto para construir IA responsável por Besmira Nushi, Mehrnoosh Sameki, e Amit Sharma

Também leia:

- Centro de recursos de IA responsável da Microsoft: [Responsible AI Resources – Microsoft AI](https://www.microsoft.com/ai/responsible-ai-resources?activetab=pivot1%3aprimaryr4) 

- Grupo de pesquisa FATE da Microsoft: [FATE: Fairness, Accountability, Transparency, and Ethics in AI - Microsoft Research](https://www.microsoft.com/research/theme/fate/)

RAI Toolbox:

- [Repositório GitHub do Responsible AI Toolbox](https://github.com/microsoft/responsible-ai-toolbox)

Leia sobre as ferramentas do Azure Machine Learning para garantir justiça:

- [Azure Machine Learning](https://docs.microsoft.com/azure/machine-learning/concept-fairness-ml?WT.mc_id=academic-77952-leestott)

## Tarefa

[Explore o RAI Toolbox](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Aviso Legal**:
Este documento foi traduzido usando o serviço de tradução por IA [Co-op Translator](https://github.com/Azure/co-op-translator). Embora nos esforcemos pela precisão, por favor, esteja ciente de que traduções automatizadas podem conter erros ou imprecisões. O documento original em seu idioma nativo deve ser considerado a fonte autorizada. Para informações críticas, recomenda-se tradução profissional humana. Não nos responsabilizamos por quaisquer mal-entendidos ou interpretações incorretas decorrentes do uso desta tradução.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->