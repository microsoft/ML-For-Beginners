<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "8f819813b2ca08ec7b9f60a2c9336045",
  "translation_date": "2025-08-29T21:22:28+00:00",
  "source_file": "1-Introduction/3-fairness/README.md",
  "language_code": "br"
}
-->
# Construindo soluções de Machine Learning com IA responsável

![Resumo de IA responsável em Machine Learning em um sketchnote](../../../../translated_images/ml-fairness.ef296ebec6afc98a44566d7b6c1ed18dc2bf1115c13ec679bb626028e852fa1d.br.png)
> Sketchnote por [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Quiz pré-aula](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/5/)

## Introdução

Neste currículo, você começará a descobrir como o aprendizado de máquina pode impactar e já está impactando nossas vidas cotidianas. Mesmo agora, sistemas e modelos estão envolvidos em tarefas de tomada de decisão diária, como diagnósticos de saúde, aprovações de empréstimos ou detecção de fraudes. Por isso, é importante que esses modelos funcionem bem para fornecer resultados confiáveis. Assim como qualquer aplicação de software, sistemas de IA podem falhar em atender às expectativas ou ter resultados indesejáveis. É por isso que é essencial entender e explicar o comportamento de um modelo de IA.

Imagine o que pode acontecer quando os dados que você usa para construir esses modelos carecem de certos grupos demográficos, como raça, gênero, visão política, religião, ou representam esses grupos de forma desproporcional. E se a saída do modelo for interpretada de forma a favorecer algum grupo demográfico? Qual é a consequência para a aplicação? Além disso, o que acontece quando o modelo tem um resultado adverso e prejudica pessoas? Quem é responsável pelo comportamento dos sistemas de IA? Estas são algumas das questões que exploraremos neste currículo.

Nesta lição, você irá:

- Aumentar sua conscientização sobre a importância da equidade no aprendizado de máquina e os danos relacionados à falta de equidade.
- Familiarizar-se com a prática de explorar outliers e cenários incomuns para garantir confiabilidade e segurança.
- Compreender a necessidade de capacitar todos ao projetar sistemas inclusivos.
- Explorar como é vital proteger a privacidade e a segurança dos dados e das pessoas.
- Ver a importância de ter uma abordagem transparente para explicar o comportamento dos modelos de IA.
- Ter em mente como a responsabilidade é essencial para construir confiança em sistemas de IA.

## Pré-requisito

Como pré-requisito, faça o "Caminho de Aprendizado sobre Princípios de IA Responsável" e assista ao vídeo abaixo sobre o tema:

Saiba mais sobre IA Responsável seguindo este [Caminho de Aprendizado](https://docs.microsoft.com/learn/modules/responsible-ai-principles/?WT.mc_id=academic-77952-leestott)

[![Abordagem da Microsoft para IA Responsável](https://img.youtube.com/vi/dnC8-uUZXSc/0.jpg)](https://youtu.be/dnC8-uUZXSc "Abordagem da Microsoft para IA Responsável")

> 🎥 Clique na imagem acima para assistir ao vídeo: Abordagem da Microsoft para IA Responsável

## Equidade

Sistemas de IA devem tratar todos de forma justa e evitar afetar grupos semelhantes de pessoas de maneiras diferentes. Por exemplo, quando sistemas de IA fornecem orientações sobre tratamento médico, solicitações de empréstimos ou emprego, eles devem fazer as mesmas recomendações para todos com sintomas, circunstâncias financeiras ou qualificações profissionais semelhantes. Cada um de nós, como humanos, carrega preconceitos herdados que afetam nossas decisões e ações. Esses preconceitos podem ser evidentes nos dados que usamos para treinar sistemas de IA. Tal manipulação pode, às vezes, acontecer de forma não intencional. Muitas vezes, é difícil saber conscientemente quando você está introduzindo preconceito nos dados.

**“Injustiça”** abrange impactos negativos, ou “danos”, para um grupo de pessoas, como aqueles definidos em termos de raça, gênero, idade ou status de deficiência. Os principais danos relacionados à equidade podem ser classificados como:

- **Alocação**, se um gênero ou etnia, por exemplo, for favorecido em detrimento de outro.
- **Qualidade do serviço**. Se você treinar os dados para um cenário específico, mas a realidade for muito mais complexa, isso leva a um serviço de desempenho ruim. Por exemplo, um dispensador de sabão que não consegue detectar pessoas com pele escura. [Referência](https://gizmodo.com/why-cant-this-soap-dispenser-identify-dark-skin-1797931773)
- **Denigração**. Criticar ou rotular algo ou alguém de forma injusta. Por exemplo, uma tecnologia de rotulagem de imagens que infamemente rotulou imagens de pessoas de pele escura como gorilas.
- **Super ou sub-representação**. A ideia é que um determinado grupo não seja visto em uma certa profissão, e qualquer serviço ou função que continue promovendo isso está contribuindo para o dano.
- **Estereotipagem**. Associar um grupo específico a atributos pré-atribuídos. Por exemplo, um sistema de tradução de idiomas entre inglês e turco pode ter imprecisões devido a palavras com associações estereotipadas de gênero.

![tradução para turco](../../../../translated_images/gender-bias-translate-en-tr.f185fd8822c2d4372912f2b690f6aaddd306ffbb49d795ad8d12a4bf141e7af0.br.png)
> tradução para turco

![tradução de volta para inglês](../../../../translated_images/gender-bias-translate-tr-en.4eee7e3cecb8c70e13a8abbc379209bc8032714169e585bdeac75af09b1752aa.br.png)
> tradução de volta para inglês

Ao projetar e testar sistemas de IA, precisamos garantir que a IA seja justa e não programada para tomar decisões tendenciosas ou discriminatórias, que também são proibidas para seres humanos. Garantir equidade em IA e aprendizado de máquina continua sendo um desafio sociotécnico complexo.

### Confiabilidade e segurança

Para construir confiança, sistemas de IA precisam ser confiáveis, seguros e consistentes em condições normais e inesperadas. É importante saber como os sistemas de IA se comportarão em uma variedade de situações, especialmente quando são outliers. Ao construir soluções de IA, é necessário um foco substancial em como lidar com uma ampla variedade de circunstâncias que as soluções de IA encontrariam. Por exemplo, um carro autônomo precisa priorizar a segurança das pessoas. Como resultado, a IA que alimenta o carro precisa considerar todos os cenários possíveis que o carro poderia encontrar, como noite, tempestades, nevascas, crianças correndo pela rua, animais de estimação, construções na estrada, etc. Quão bem um sistema de IA pode lidar com uma ampla gama de condições de forma confiável e segura reflete o nível de antecipação que o cientista de dados ou desenvolvedor de IA considerou durante o design ou teste do sistema.

> [🎥 Clique aqui para um vídeo: ](https://www.microsoft.com/videoplayer/embed/RE4vvIl)

### Inclusividade

Sistemas de IA devem ser projetados para engajar e capacitar todos. Ao projetar e implementar sistemas de IA, cientistas de dados e desenvolvedores de IA identificam e abordam barreiras potenciais no sistema que poderiam excluir pessoas de forma não intencional. Por exemplo, há 1 bilhão de pessoas com deficiência ao redor do mundo. Com o avanço da IA, elas podem acessar uma ampla gama de informações e oportunidades mais facilmente em suas vidas diárias. Ao abordar as barreiras, cria-se oportunidades para inovar e desenvolver produtos de IA com melhores experiências que beneficiem a todos.

> [🎥 Clique aqui para um vídeo: inclusividade em IA](https://www.microsoft.com/videoplayer/embed/RE4vl9v)

### Segurança e privacidade

Sistemas de IA devem ser seguros e respeitar a privacidade das pessoas. As pessoas têm menos confiança em sistemas que colocam sua privacidade, informações ou vidas em risco. Ao treinar modelos de aprendizado de máquina, dependemos de dados para produzir os melhores resultados. Ao fazer isso, a origem dos dados e sua integridade devem ser consideradas. Por exemplo, os dados foram enviados por usuários ou estão disponíveis publicamente? Em seguida, ao trabalhar com os dados, é crucial desenvolver sistemas de IA que possam proteger informações confidenciais e resistir a ataques. À medida que a IA se torna mais prevalente, proteger a privacidade e garantir a segurança de informações pessoais e empresariais importantes está se tornando mais crítico e complexo. Questões de privacidade e segurança de dados exigem atenção especial para IA porque o acesso aos dados é essencial para que os sistemas de IA façam previsões e decisões precisas e informadas sobre as pessoas.

> [🎥 Clique aqui para um vídeo: segurança em IA](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Como indústria, fizemos avanços significativos em privacidade e segurança, impulsionados significativamente por regulamentações como o GDPR (Regulamento Geral de Proteção de Dados).
- No entanto, com sistemas de IA, devemos reconhecer a tensão entre a necessidade de mais dados pessoais para tornar os sistemas mais personalizados e eficazes – e a privacidade.
- Assim como com o nascimento de computadores conectados à internet, também estamos vendo um grande aumento no número de problemas de segurança relacionados à IA.
- Ao mesmo tempo, vimos a IA sendo usada para melhorar a segurança. Por exemplo, a maioria dos scanners antivírus modernos é impulsionada por heurísticas de IA hoje.
- Precisamos garantir que nossos processos de Ciência de Dados se harmonizem com as práticas mais recentes de privacidade e segurança.

### Transparência

Sistemas de IA devem ser compreensíveis. Uma parte crucial da transparência é explicar o comportamento dos sistemas de IA e seus componentes. Melhorar a compreensão dos sistemas de IA exige que as partes interessadas compreendam como e por que eles funcionam, para que possam identificar possíveis problemas de desempenho, preocupações de segurança e privacidade, preconceitos, práticas excludentes ou resultados não intencionais. Também acreditamos que aqueles que usam sistemas de IA devem ser honestos e transparentes sobre quando, por que e como escolhem implantá-los, bem como as limitações dos sistemas que utilizam. Por exemplo, se um banco usa um sistema de IA para apoiar suas decisões de concessão de crédito ao consumidor, é importante examinar os resultados e entender quais dados influenciam as recomendações do sistema. Os governos estão começando a regulamentar a IA em diversos setores, então cientistas de dados e organizações devem explicar se um sistema de IA atende aos requisitos regulatórios, especialmente quando há um resultado indesejável.

> [🎥 Clique aqui para um vídeo: transparência em IA](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Como os sistemas de IA são tão complexos, é difícil entender como eles funcionam e interpretar os resultados.
- Essa falta de compreensão afeta a forma como esses sistemas são gerenciados, operacionalizados e documentados.
- Essa falta de compreensão, mais importante, afeta as decisões tomadas com base nos resultados que esses sistemas produzem.

### Responsabilidade

As pessoas que projetam e implantam sistemas de IA devem ser responsáveis pelo funcionamento de seus sistemas. A necessidade de responsabilidade é particularmente crucial com tecnologias sensíveis, como reconhecimento facial. Recentemente, houve uma demanda crescente por tecnologia de reconhecimento facial, especialmente de organizações de aplicação da lei que veem o potencial da tecnologia em usos como encontrar crianças desaparecidas. No entanto, essas tecnologias poderiam ser usadas por um governo para colocar em risco as liberdades fundamentais de seus cidadãos, por exemplo, permitindo a vigilância contínua de indivíduos específicos. Portanto, cientistas de dados e organizações precisam ser responsáveis pelo impacto de seus sistemas de IA sobre indivíduos ou a sociedade.

[![Pesquisador líder em IA alerta sobre vigilância em massa por meio de reconhecimento facial](../../../../translated_images/accountability.41d8c0f4b85b6231301d97f17a450a805b7a07aaeb56b34015d71c757cad142e.br.png)](https://www.youtube.com/watch?v=Wldt8P5V6D0 "Abordagem da Microsoft para IA Responsável")

> 🎥 Clique na imagem acima para assistir ao vídeo: Alertas sobre Vigilância em Massa por Reconhecimento Facial

Em última análise, uma das maiores questões para nossa geração, como a primeira geração que está trazendo IA para a sociedade, é como garantir que os computadores permaneçam responsáveis perante as pessoas e como garantir que as pessoas que projetam computadores permaneçam responsáveis perante todos os outros.

## Avaliação de impacto

Antes de treinar um modelo de aprendizado de máquina, é importante realizar uma avaliação de impacto para entender o propósito do sistema de IA; qual é o uso pretendido; onde será implantado; e quem interagirá com o sistema. Isso é útil para os revisores ou testadores que avaliam o sistema saberem quais fatores considerar ao identificar riscos potenciais e consequências esperadas.

As seguintes áreas são focos ao realizar uma avaliação de impacto:

* **Impacto adverso sobre indivíduos**. Estar ciente de quaisquer restrições ou requisitos, uso não suportado ou quaisquer limitações conhecidas que prejudiquem o desempenho do sistema é vital para garantir que o sistema não seja usado de forma que possa causar danos aos indivíduos.
* **Requisitos de dados**. Compreender como e onde o sistema usará dados permite que os revisores explorem quaisquer requisitos de dados que você precisaria considerar (por exemplo, regulamentações de dados como GDPR ou HIPAA). Além disso, examine se a origem ou quantidade de dados é substancial para o treinamento.
* **Resumo do impacto**. Reúna uma lista de possíveis danos que poderiam surgir do uso do sistema. Ao longo do ciclo de vida do aprendizado de máquina, revise se os problemas identificados foram mitigados ou abordados.
* **Metas aplicáveis** para cada um dos seis princípios fundamentais. Avalie se as metas de cada princípio foram atendidas e se há lacunas.

## Depuração com IA responsável

Semelhante à depuração de uma aplicação de software, depurar um sistema de IA é um processo necessário para identificar e resolver problemas no sistema. Há muitos fatores que podem afetar um modelo que não está funcionando como esperado ou de forma responsável. A maioria das métricas tradicionais de desempenho de modelos são agregados quantitativos do desempenho de um modelo, que não são suficientes para analisar como um modelo viola os princípios de IA responsável. Além disso, um modelo de aprendizado de máquina é uma caixa preta que dificulta entender o que impulsiona seu resultado ou fornecer explicações quando ele comete um erro. Mais adiante neste curso, aprenderemos como usar o painel de IA Responsável para ajudar a depurar sistemas de IA. O painel fornece uma ferramenta holística para cientistas de dados e desenvolvedores de IA realizarem:

* **Análise de erros**. Para identificar a distribuição de erros do modelo que pode afetar a equidade ou confiabilidade do sistema.
* **Visão geral do modelo**. Para descobrir onde há disparidades no desempenho do modelo entre diferentes grupos de dados.
* **Análise de dados**. Para entender a distribuição dos dados e identificar qualquer potencial preconceito nos dados que poderia levar a problemas de equidade, inclusão e confiabilidade.
* **Interpretabilidade do modelo**. Para entender o que afeta ou influencia as previsões do modelo. Isso ajuda a explicar o comportamento do modelo, o que é importante para transparência e responsabilidade.

## 🚀 Desafio

Para evitar que danos sejam introduzidos desde o início, devemos:

- ter diversidade de origens e perspectivas entre as pessoas que trabalham nos sistemas
- investir em conjuntos de dados que reflitam a diversidade de nossa sociedade
- desenvolver melhores métodos ao longo do ciclo de vida do aprendizado de máquina para detectar e corrigir problemas de IA responsável quando eles ocorrerem

Pense em cenários da vida real onde a falta de confiabilidade de um modelo é evidente na construção e uso do modelo. O que mais devemos considerar?

## [Quiz pós-aula](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/6/)
## Revisão e Autoestudo
Nesta lição, você aprendeu alguns conceitos básicos sobre justiça e injustiça em aprendizado de máquina.  

Assista a este workshop para se aprofundar nos tópicos: 

- Em busca de IA responsável: Colocando princípios em prática por Besmira Nushi, Mehrnoosh Sameki e Amit Sharma

[![Responsible AI Toolbox: Um framework de código aberto para construir IA responsável](https://img.youtube.com/vi/tGgJCrA-MZU/0.jpg)](https://www.youtube.com/watch?v=tGgJCrA-MZU "RAI Toolbox: Um framework de código aberto para construir IA responsável")


> 🎥 Clique na imagem acima para assistir ao vídeo: RAI Toolbox: Um framework de código aberto para construir IA responsável por Besmira Nushi, Mehrnoosh Sameki e Amit Sharma

Além disso, leia: 

- Centro de recursos de IA responsável da Microsoft: [Recursos de IA Responsável – Microsoft AI](https://www.microsoft.com/ai/responsible-ai-resources?activetab=pivot1%3aprimaryr4) 

- Grupo de pesquisa FATE da Microsoft: [FATE: Justiça, Responsabilidade, Transparência e Ética em IA - Microsoft Research](https://www.microsoft.com/research/theme/fate/) 

RAI Toolbox: 

- [Repositório GitHub do Responsible AI Toolbox](https://github.com/microsoft/responsible-ai-toolbox)

Leia sobre as ferramentas do Azure Machine Learning para garantir justiça:

- [Azure Machine Learning](https://docs.microsoft.com/azure/machine-learning/concept-fairness-ml?WT.mc_id=academic-77952-leestott) 

## Tarefa

[Explore o RAI Toolbox](assignment.md)

---

**Aviso Legal**:  
Este documento foi traduzido utilizando o serviço de tradução por IA [Co-op Translator](https://github.com/Azure/co-op-translator). Embora nos esforcemos para garantir a precisão, esteja ciente de que traduções automatizadas podem conter erros ou imprecisões. O documento original em seu idioma nativo deve ser considerado a fonte autoritativa. Para informações críticas, recomenda-se a tradução profissional realizada por humanos. Não nos responsabilizamos por quaisquer mal-entendidos ou interpretações equivocadas decorrentes do uso desta tradução.