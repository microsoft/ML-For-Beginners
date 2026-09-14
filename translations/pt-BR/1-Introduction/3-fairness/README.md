# Construindo soluções de Machine Learning com IA responsável
 
![Resumo de IA responsável em Machine Learning em um sketchnote](../../../../translated_images/pt-BR/ml-fairness.ef296ebec6afc98a.webp)
> Sketchnote por [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Quiz pré-aula](https://ff-quizzes.netlify.app/en/ml/)
 
## Introdução

Neste currículo, você começará a descobrir como o machine learning pode e está impactando nosso dia a dia. Mesmo agora, sistemas e modelos estão envolvidos em tarefas diárias de tomada de decisão, como diagnósticos de saúde, aprovações de empréstimos ou detecção de fraude. Portanto, é importante que esses modelos funcionem bem para fornecer resultados confiáveis. Assim como qualquer aplicativo de software, sistemas de IA podem falhar nas expectativas ou apresentar um resultado indesejado. É por isso que é essencial ser capaz de entender e explicar o comportamento de um modelo de IA. 

Imagine o que pode acontecer quando os dados que você usa para construir esses modelos não incluem determinados grupos demográficos, como raça, gênero, visão política, religião, ou representam desproporcionalmente tais grupos. E quando a saída do modelo é interpretada para favorecer algum grupo demográfico? Qual é a consequência para a aplicação? Além disso, o que acontece quando o modelo tem um resultado adverso e prejudica pessoas? Quem é responsável pelo comportamento dos sistemas de IA? Essas são algumas perguntas que exploraremos neste currículo. 

Nesta lição, você irá: 

- Elevar sua conscientização sobre a importância da justiça no machine learning e danos relacionados à justiça.
- Familiarizar-se com a prática de explorar casos extremos e cenários incomuns para garantir confiabilidade e segurança
- Obter entendimento sobre a necessidade de empoderar todos por meio do design de sistemas inclusivos
- Explorar como é vital proteger a privacidade e a segurança dos dados e das pessoas
- Ver a importância de ter uma abordagem de caixa transparente para explicar o comportamento dos modelos de IA
- Estar atento a como a responsabilidade é essencial para construir confiança nos sistemas de IA

## Pré-requisito

Como pré-requisito, por favor faça o "Princípios de IA Responsável" no Caminho de Aprendizagem e assista ao vídeo abaixo sobre o tema:

Saiba mais sobre IA Responsável seguindo este [Caminho de Aprendizagem](https://docs.microsoft.com/learn/modules/responsible-ai-principles/?WT.mc_id=academic-77952-leestott)

[![Abordagem da Microsoft para IA Responsável](https://img.youtube.com/vi/dnC8-uUZXSc/0.jpg)](https://youtu.be/dnC8-uUZXSc "Abordagem da Microsoft para IA Responsável")

> 🎥 Clique na imagem acima para um vídeo: Abordagem da Microsoft para IA Responsável

## Justiça

Sistemas de IA devem tratar todas as pessoas de forma justa e evitar afetar grupos semelhantes de maneira diferente. Por exemplo, quando sistemas de IA fornecem orientações sobre tratamento médico, solicitações de empréstimos ou emprego, eles devem fazer as mesmas recomendações para todos com sintomas, condições financeiras ou qualificações profissionais semelhantes. Cada um de nós, como humanos, carrega preconceitos herdados que afetam nossas decisões e ações. Esses preconceitos podem ser evidentes nos dados que usamos para treinar sistemas de IA. Tal manipulação às vezes ocorre sem intenção. Muitas vezes é difícil saber conscientemente quando você está introduzindo viés nos dados. 

**“Injustiça”** engloba impactos negativos, ou “danos”, para um grupo de pessoas, como aqueles definidos em termos de raça, gênero, idade ou condição de deficiência. Os principais danos relacionados à justiça podem ser classificados como: 

- **Alocação**, se um gênero ou etnia, por exemplo, é favorecido em relação a outro.
- **Qualidade do serviço**. Se você treina os dados para um cenário específico, mas a realidade for muito mais complexa, isso leva a um serviço com desempenho ruim. Por exemplo, um dispensador de sabonete que parecia não conseguir detectar pessoas com pele escura. [Referência](https://gizmodo.com/why-cant-this-soap-dispenser-identify-dark-skin-1797931773)
- **Denegrimento**. Criticar e rotular algo ou alguém de forma injusta. Por exemplo, uma tecnologia de rotulagem de imagens que famosamente rotulou erroneamente imagens de pessoas de pele escura como gorilas.
- **Super- ou sub-representação**. A ideia é que um determinado grupo não é visto em uma certa profissão, e qualquer serviço ou função que continue promovendo isso contribui para o dano.
- **Estereotipagem**. Associar um grupo dado com atributos pré-atribuídos. Por exemplo, um sistema de tradução de idiomas entre inglês e turco pode ter imprecisões devido a palavras com associações estereotipadas de gênero.

![tradução para turco](../../../../translated_images/pt-BR/gender-bias-translate-en-tr.f185fd8822c2d437.webp)
> tradução para turco

![tradução de volta para inglês](../../../../translated_images/pt-BR/gender-bias-translate-tr-en.4eee7e3cecb8c70e.webp)
> tradução de volta para inglês

Ao projetar e testar sistemas de IA, precisamos garantir que a IA seja justa e não programada para tomar decisões tendenciosas ou discriminatórias, o que os seres humanos também são proibidos de fazer. Garantir justiça em IA e machine learning continua sendo um desafio sociotécnico complexo. 

### Confiabilidade e segurança

Para construir confiança, sistemas de IA precisam ser confiáveis, seguros e consistentes em condições normais e inesperadas. É importante saber como sistemas de IA se comportarão em diversas situações, especialmente quando são casos atípicos. Ao construir soluções de IA, deve haver um foco substancial em como lidar com uma ampla variedade de circunstâncias que as soluções de IA podem encontrar. Por exemplo, um carro autônomo precisa colocar a segurança das pessoas como prioridade máxima. Como resultado, a IA que controla o carro precisa considerar todos os possíveis cenários que o carro pode enfrentar, como noite, tempestades ou nevascas, crianças correndo pela rua, animais de estimação, construções na estrada etc. Quão bem um sistema de IA pode lidar de forma confiável e segura com uma gama tão variada de condições reflete o nível de antecipação que o cientista de dados ou desenvolvedor de IA considerou durante o design ou teste do sistema.  

> [🎥 Clique aqui para um vídeo: ](https://www.microsoft.com/videoplayer/embed/RE4vvIl)

### Inclusividade

Sistemas de IA devem ser projetados para engajar e empoderar todos. Ao projetar e implementar sistemas de IA, cientistas de dados e desenvolvedores de IA identificam e abordam barreiras potenciais no sistema que poderiam, inadvertidamente, excluir pessoas. Por exemplo, há 1 bilhão de pessoas com deficiências no mundo. Com o avanço da IA, eles podem acessar uma ampla gama de informações e oportunidades mais facilmente em suas vidas diárias. Ao abordar essas barreiras, criam-se oportunidades para inovar e desenvolver produtos de IA com melhores experiências que beneficiam a todos. 

> [🎥 Clique aqui para um vídeo: inclusividade em IA](https://www.microsoft.com/videoplayer/embed/RE4vl9v)

### Segurança e privacidade 

Sistemas de IA devem ser seguros e respeitar a privacidade das pessoas. As pessoas confiam menos em sistemas que colocam sua privacidade, informações ou vidas em risco. Ao treinar modelos de machine learning, dependemos dos dados para produzir os melhores resultados. Ao fazer isso, a origem dos dados e a integridade devem ser consideradas. Por exemplo, os dados foram submetidos pelo usuário ou são publicamente disponíveis? Em seguida, ao trabalhar com os dados, é crucial desenvolver sistemas de IA que possam proteger informações confidenciais e resistir a ataques. À medida que a IA se torna mais prevalente, proteger a privacidade e assegurar informações pessoais e comerciais importantes torna-se mais crítico e complexo. Questões de privacidade e segurança de dados exigem atenção especialmente próxima para IA porque o acesso a dados é essencial para que sistemas de IA façam previsões e decisões precisas e informadas sobre pessoas. 

> [🎥 Clique aqui para um vídeo: segurança em IA](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Como indústria, fizemos avanços significativos em Privacidade & segurança, impulsionados significativamente por regulamentações como o GDPR (Regulamento Geral de Proteção de Dados).
- Porém, com sistemas de IA, precisamos reconhecer a tensão entre a necessidade de mais dados pessoais para tornar os sistemas mais personalizados e eficazes – e a privacidade.
- Assim como com o surgimento dos computadores conectados com a internet, também estamos vendo um grande aumento no número de questões de segurança relacionadas à IA.
- Ao mesmo tempo, vimos IA sendo usada para melhorar a segurança. Por exemplo, a maioria dos scanners antivírus modernos é hoje movida por heurísticas de IA.
- Precisamos garantir que nossos processos de Ciência de Dados se integrem harmoniosamente com as práticas mais recentes de privacidade e segurança.


### Transparência
Sistemas de IA devem ser compreensíveis. Uma parte crucial da transparência é explicar o comportamento dos sistemas de IA e seus componentes. Melhorar a compreensão dos sistemas de IA requer que as partes interessadas entendam como e por que eles funcionam para que possam identificar potenciais problemas de desempenho, preocupações de segurança e privacidade, preconceitos, práticas excludentes ou resultados não intencionais. Também acreditamos que aqueles que usam sistemas de IA devem ser honestos e francos sobre quando, por que e como escolhem implementá-los. Assim como sobre as limitações dos sistemas que usam. Por exemplo, se um banco usa um sistema de IA para apoiar suas decisões de empréstimos a consumidores, é importante analisar os resultados e entender quais dados influenciam as recomendações do sistema. Governos estão começando a regulamentar IA em diversos setores, então cientistas de dados e organizações devem explicar se um sistema de IA atende aos requisitos regulatórios, especialmente quando há um resultado indesejado. 

> [🎥 Clique aqui para um vídeo: transparência em IA](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Porque os sistemas de IA são tão complexos, é difícil entender como eles funcionam e interpretar os resultados.
- Essa falta de entendimento afeta a forma como esses sistemas são gerenciados, operacionalizados e documentados.
- Essa falta de entendimento, mais importante, afeta as decisões tomadas usando os resultados que esses sistemas produzem.

### Responsabilidade
 
As pessoas que projetam e implementam sistemas de IA devem ser responsáveis por como seus sistemas operam. A necessidade de responsabilidade é particularmente crucial com tecnologias de uso sensível, como o reconhecimento facial. Recentemente, houve uma crescente demanda por tecnologia de reconhecimento facial, especialmente de organizações de aplicação da lei que veem o potencial da tecnologia em usos como encontrar crianças desaparecidas. No entanto, essas tecnologias poderiam potencialmente ser usadas por um governo para colocar em risco as liberdades fundamentais de seus cidadãos ao, por exemplo, permitir vigilância contínua de indivíduos específicos. Portanto, cientistas de dados e organizações precisam ser responsáveis por como seus sistemas de IA impactam indivíduos ou a sociedade.

[![Pesquisador líder em IA alerta sobre vigilância em massa por reconhecimento facial](../../../../translated_images/pt-BR/accountability.41d8c0f4b85b6231.webp)](https://www.youtube.com/watch?v=Wldt8P5V6D0 "Abordagem da Microsoft para IA Responsável")

> 🎥 Clique na imagem acima para um vídeo: Avisos sobre Vigilância em Massa por Reconhecimento Facial

Em última análise, uma das maiores perguntas para nossa geração, como a primeira geração que está trazendo IA para a sociedade, é como garantir que os computadores permaneçam responsáveis perante as pessoas e como garantir que as pessoas que projetam computadores permaneçam responsáveis perante todos os demais.

## Avaliação de impacto

Antes de treinar um modelo de machine learning, é importante realizar uma avaliação de impacto para entender o propósito do sistema de IA; qual é seu uso pretendido; onde será implementado; e quem interagirá com o sistema. Esses são pontos úteis para revisores ou testadores que avaliam o sistema saberem quais fatores levar em consideração ao identificar riscos potenciais e consequências esperadas.

As seguintes são áreas de foco ao conduzir uma avaliação de impacto:

* **Impacto adverso em indivíduos**. Estar ciente de quaisquer restrições ou requisitos, uso não suportado ou quaisquer limitações conhecidas que prejudiquem o desempenho do sistema é vital para garantir que o sistema não seja usado de maneira que possa causar danos aos indivíduos.
* **Requisitos de dados**. Obter uma compreensão de como e onde o sistema usará os dados permite que os revisores explorem quaisquer requisitos de dados que você deve ter em mente (ex: regulamentos GDPR ou HIPAA). Além disso, examinar se a origem ou quantidade de dados é substancial para o treinamento.
* **Resumo do impacto**. Reunir uma lista de danos potenciais que poderiam surgir do uso do sistema. Ao longo do ciclo de vida do ML, revisar se os problemas identificados são mitigados ou tratados.
* **Objetivos aplicáveis** para cada um dos seis princípios centrais. Avaliar se os objetivos de cada um dos princípios são atendidos e se existem lacunas.


## Depuração com IA responsável  

Semelhante à depuração de um aplicativo de software, depurar um sistema de IA é um processo necessário de identificação e resolução de problemas no sistema. Há muitos fatores que podem afetar um modelo a não performar como esperado ou de forma responsável. A maioria das métricas tradicionais de desempenho do modelo são agregados quantitativos do desempenho do modelo, que não são suficientes para analisar como um modelo viola os princípios de IA responsável. Além disso, um modelo de machine learning é uma caixa preta que dificulta entender o que conduz seu resultado ou fornecer explicação quando ele comete um erro. Mais adiante neste curso, aprenderemos como usar o painel de IA Responsável para ajudar a depurar sistemas de IA. O painel fornece uma ferramenta holística para cientistas de dados e desenvolvedores de IA realizarem:

* **Análise de erros**. Para identificar a distribuição de erros do modelo que pode afetar a justiça ou confiabilidade do sistema.
* **Visão geral do modelo**. Para descobrir onde há disparidades no desempenho do modelo entre coortes de dados.
* **Análise de dados**. Para entender a distribuição dos dados e identificar qualquer viés potencial nos dados que possa levar a problemas de justiça, inclusividade e confiabilidade.
* **Interpretabilidade do modelo**. Para compreender o que afeta ou influencia as previsões do modelo. Isso ajuda a explicar o comportamento do modelo, o que é importante para transparência e responsabilidade.


## 🚀 Desafio
 
Para prevenir que danos sejam introduzidos desde o início, devemos: 

- ter diversidade de origens e perspectivas entre as pessoas que trabalham nos sistemas
- investir em conjuntos de dados que reflitam a diversidade da nossa sociedade
- desenvolver melhores métodos ao longo do ciclo de vida do machine learning para detectar e corrigir IA responsável quando ocorrer

Pense em cenários da vida real onde a falta de confiabilidade de um modelo é evidente na construção e uso do modelo. O que mais deveríamos considerar? 

## [Quiz pós-aula](https://ff-quizzes.netlify.app/en/ml/)

## Revisão & Estudo Autônomo
 

Nesta lição, você aprendeu alguns conceitos básicos sobre justiça e injustiça em aprendizado de máquina.  
 
Assista este workshop para aprofundar nos tópicos: 

- Na busca por IA responsável: Aplicando princípios na prática por Besmira Nushi, Mehrnoosh Sameki e Amit Sharma

[![Responsible AI Toolbox: Um framework open-source para construção de IA responsável](https://img.youtube.com/vi/tGgJCrA-MZU/0.jpg)](https://www.youtube.com/watch?v=tGgJCrA-MZU "RAI Toolbox: Um framework open-source para construção de IA responsável")

> 🎥 Clique na imagem acima para assistir ao vídeo: RAI Toolbox: Um framework open-source para construção de IA responsável por Besmira Nushi, Mehrnoosh Sameki e Amit Sharma

Além disso, leia: 

- Centro de recursos RAI da Microsoft: [Recursos de IA Responsável – Microsoft AI](https://www.microsoft.com/ai/responsible-ai-resources?activetab=pivot1%3aprimaryr4) 

- Grupo de pesquisa FATE da Microsoft: [FATE: Justiça, Responsabilidade, Transparência e Ética em IA - Microsoft Research](https://www.microsoft.com/research/theme/fate/) 

RAI Toolbox: 

- [Repositório GitHub do Responsible AI Toolbox](https://github.com/microsoft/responsible-ai-toolbox)

Saiba mais sobre as ferramentas do Azure Machine Learning para garantir justiça:

- [Azure Machine Learning](https://docs.microsoft.com/azure/machine-learning/concept-fairness-ml?WT.mc_id=academic-77952-leestott) 

## Tarefa

[Explore a RAI Toolbox](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Aviso Legal**:
Este documento foi traduzido usando o serviço de tradução por IA [Co-op Translator](https://github.com/Azure/co-op-translator). Embora nos esforcemos pela precisão, por favor, esteja ciente de que traduções automatizadas podem conter erros ou imprecisões. O documento original em seu idioma nativo deve ser considerado a fonte autorizada. Para informações críticas, recomenda-se tradução profissional humana. Não nos responsabilizamos por quaisquer mal-entendidos ou interpretações incorretas decorrentes do uso desta tradução.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->