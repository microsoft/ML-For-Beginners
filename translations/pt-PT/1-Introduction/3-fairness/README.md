# Construir soluções de Aprendizagem Automática com IA responsável
 
![Resumo da IA responsável na Aprendizagem Automática num sketchnote](../../../../translated_images/pt-PT/ml-fairness.ef296ebec6afc98a.webp)
> Sketchnote por [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Questionário pré-aula](https://ff-quizzes.netlify.app/en/ml/)
 
## Introdução

Neste currículo, irá começar a descobrir como a aprendizagem automática pode e está a impactar as nossas vidas quotidianas. Mesmo agora, sistemas e modelos estão envolvidos em tarefas diárias de tomada de decisão, como diagnósticos de saúde, aprovação de empréstimos ou deteção de fraudes. Por isso, é importante que estes modelos funcionem bem para fornecer resultados em que se possa confiar. Tal como qualquer aplicação de software, os sistemas de IA podem não corresponder às expectativas ou ter um resultado indesejado. É por isso que é essencial conseguir compreender e explicar o comportamento de um modelo de IA.

Imagine o que pode acontecer quando os dados que está a utilizar para construir estes modelos carecem de determinados dados demográficos, como raça, género, visão política, religião, ou representam desproporcionalmente tais dados demográficos. E quando o resultado do modelo é interpretado para favorecer algum grupo demográfico? Qual é a consequência para a aplicação? Além disso, o que acontece quando o modelo tem um resultado adverso e prejudica pessoas? Quem é responsável pelo comportamento dos sistemas de IA? Estas são algumas das questões que iremos explorar neste currículo.

Nesta lição, irá:

- Aumentar a sua consciência sobre a importância da equidade na aprendizagem automática e os prejuízos relacionados com a equidade.
- Familiarizar-se com a prática de explorar outliers e cenários invulgares para garantir fiabilidade e segurança
- Compreender a necessidade de capacitar todos através do design de sistemas inclusivos
- Explorar a importância vital de proteger a privacidade e segurança dos dados e das pessoas
- Ver a importância de ter uma abordagem transparente para explicar o comportamento dos modelos de IA
- Estar atento a como a responsabilidade é essencial para construir confiança nos sistemas de IA

## Pré-requisito

Como pré-requisito, por favor faça o "Princípios da IA Responsável" caminho de aprendizagem e veja o vídeo abaixo sobre o tema:

Saiba mais sobre IA Responsável seguindo este [Caminho de Aprendizagem](https://docs.microsoft.com/learn/modules/responsible-ai-principles/?WT.mc_id=academic-77952-leestott)

[![Abordagem da Microsoft para IA Responsável](https://img.youtube.com/vi/dnC8-uUZXSc/0.jpg)](https://youtu.be/dnC8-uUZXSc "Abordagem da Microsoft para IA Responsável")

> 🎥 Clique na imagem acima para ver um vídeo: Abordagem da Microsoft para IA Responsável

## Equidade

Os sistemas de IA devem tratar todas as pessoas de forma justa e evitar afetar grupos semelhantes de maneiras diferentes. Por exemplo, quando sistemas de IA fornecem orientação sobre tratamentos médicos, pedidos de empréstimos ou emprego, devem fazer as mesmas recomendações para todos com sintomas semelhantes, condições financeiras ou qualificações profissionais. Cada um de nós, como humanos, traz inerentes preconceitos que afetam as nossas decisões e ações. Estes preconceitos podem ser evidentes nos dados que usamos para treinar sistemas de IA. Tal manipulação pode ocorrer, por vezes, involuntariamente. Geralmente é difícil saber conscientemente quando se está a introduzir um preconceito nos dados.

**“Injustiça”** abrange impactos negativos, ou “prejuízos”, para um grupo de pessoas, tais como aqueles definidos em termos de raça, género, idade ou condição de incapacidade. Os principais prejuízos relacionados com a equidade podem ser classificados como:

- **Alocação**, se um género ou etnia, por exemplo, é favorecido em relação a outro.
- **Qualidade do serviço**. Se treinar os dados para um cenário específico mas a realidade for muito mais complexa, leva a um serviço de fraco desempenho. Por exemplo, um dispensador de sabão que não conseguia detetar pessoas com pele escura. [Referência](https://gizmodo.com/why-cant-this-soap-dispenser-identify-dark-skin-1797931773)
- **Denigração**. Criticar e rotular injustamente algo ou alguém. Por exemplo, uma tecnologia de etiquetagem de imagens que infamemente rotulou imagens de pessoas de pele escura como gorilas.
- **Sobre- ou sub-representação**. A ideia é que um certo grupo não é visto numa certa profissão, e qualquer serviço ou função que promova isso contribui para o prejuízo.
- **Estereotipagem**. Associar um dado grupo com atributos pré-atribuídos. Por exemplo, um sistema de tradução linguística entre inglês e turco pode ter imprecisões devido a palavras com associações estereotipadas ao género.

![tradução para turco](../../../../translated_images/pt-PT/gender-bias-translate-en-tr.f185fd8822c2d437.webp)
> tradução para turco

![tradução de volta para inglês](../../../../translated_images/pt-PT/gender-bias-translate-tr-en.4eee7e3cecb8c70e.webp)
> tradução de volta para inglês

Ao conceber e testar sistemas de IA, precisamos garantir que a IA seja justa e não esteja programada para tomar decisões tendenciosas ou discriminatórias, algo que também é proibido aos seres humanos. Garantir a equidade na IA e na aprendizagem automática continua a ser um desafio sociotécnico complexo.

### Fiabilidade e segurança

Para construir confiança, os sistemas de IA precisam de ser fiáveis, seguros e consistentes em condições normais e inesperadas. É importante saber como os sistemas de IA se vão comportar em várias situações, especialmente quando são outliers. Ao construir soluções de IA, deve haver um foco substancial em como gerir uma vasta variedade de circunstâncias que as soluções de IA possam encontrar. Por exemplo, um carro autónomo deve colocar a segurança das pessoas como prioridade máxima. Como resultado, a IA que impulsiona o carro precisa considerar todos os cenários possíveis que o carro possa encontrar, como noite, tempestades ou nevascas, crianças a atravessar a rua a correr, animais de estimação, obras na estrada, etc. O quão bem um sistema de IA consegue lidar de forma fiável e segura com uma ampla gama de condições reflete o nível de antecipação que o cientista de dados ou programador de IA considerou durante o desenho ou teste do sistema.

> [🎥 Clique aqui para ver um vídeo: Fiabilidade e segurança na IA](https://www.microsoft.com/videoplayer/embed/RE4vvIl)

### Inclusividade

Os sistemas de IA devem ser concebidos para envolver e capacitar todos. Ao conceber e implementar sistemas de IA, cientistas de dados e programadores de IA identificam e abordam barreiras potenciais no sistema que possam excluir pessoas inadvertidamente. Por exemplo, existem 1 bilião de pessoas com deficiências no mundo. Com o avanço da IA, podem aceder de forma mais fácil a uma vasta gama de informações e oportunidades no dia a dia. Ao eliminar barreiras, cria-se oportunidade para inovar e desenvolver produtos de IA com melhores experiências que beneficiem todos.

> [🎥 Clique aqui para ver um vídeo: Inclusividade na IA](https://www.microsoft.com/videoplayer/embed/RE4vl9v)

### Segurança e privacidade

Os sistemas de IA devem ser seguros e respeitar a privacidade das pessoas. As pessoas confiam menos em sistemas que colocam em risco a sua privacidade, informações ou vidas. Ao treinar modelos de aprendizagem automática, dependemos dos dados para produzir os melhores resultados. Para tal, a origem dos dados e a integridade devem ser consideradas. Por exemplo, foram os dados fornecidos pelo utilizador ou estavam publicamente disponíveis? Depois, ao trabalhar com os dados, é crucial desenvolver sistemas de IA que possam proteger informação confidencial e resistir a ataques. À medida que a IA se torna mais prevalente, proteger a privacidade e assegurar informação pessoal e empresarial importante torna-se cada vez mais crítico e complexo. Questões de privacidade e segurança dos dados requerem especial atenção para a IA porque o acesso aos dados é essencial para que os sistemas de IA façam previsões e decisões precisas e informadas sobre as pessoas.

> [🎥 Clique aqui para ver um vídeo: Segurança na IA](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Como indústria, fizemos avanços significativos em Privacidade e segurança, impulsionados significativamente por regulamentações como o GDPR (Regulamento Geral sobre a Proteção de Dados).
- Ainda assim, com sistemas de IA devemos reconhecer a tensão entre a necessidade de mais dados pessoais para tornar os sistemas mais pessoais e eficazes – e a privacidade.
- Tal como no nascimento dos computadores ligados à internet, também estamos a assistir a um grande aumento no número de questões de segurança relacionadas com a IA.
- Ao mesmo tempo, temos visto a IA ser usada para melhorar a segurança. Por exemplo, a maioria dos antivírus modernos é hoje movida por heurísticas de IA.
- Precisamos garantir que os nossos processos de Ciência de Dados se alinhem harmoniosamente com as mais recentes práticas de privacidade e segurança.


### Transparência
Os sistemas de IA devem ser compreensíveis. Uma parte crucial da transparência é explicar o comportamento dos sistemas de IA e dos seus componentes. Melhorar a compreensão dos sistemas de IA requer que os intervenientes percebam como e porquê funcionam para que possam identificar potenciais problemas de desempenho, preocupações de segurança e privacidade, preconceitos, práticas de exclusão ou resultados indesejados. Acreditamos também que quem usa sistemas de IA deve ser honesto e claro sobre quando, porquê, e como escolhem implementá-los, bem como sobre as limitações dos sistemas que usam. Por exemplo, se um banco usa um sistema de IA para apoiar as suas decisões de concessão de crédito ao consumidor, é importante examinar os resultados e compreender quais dados influenciam as recomendações do sistema. Os governos estão a começar a regulamentar IA em diversas indústrias, por isso cientistas de dados e organizações devem explicar se um sistema de IA cumpre os requisitos regulamentares, especialmente quando há um resultado indesejado.

> [🎥 Clique aqui para ver um vídeo: Transparência na IA](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Porque os sistemas de IA são tão complexos, é difícil entender como funcionam e interpretar os resultados.
- Esta falta de compreensão afeta a forma como estes sistemas são geridos, operacionalizados e documentados.
- Esta falta de compreensão, mais importante ainda, afeta as decisões tomadas com base nos resultados que estes sistemas produzem.

### Responsabilidade
 
As pessoas que concebem e implementam sistemas de IA devem ser responsáveis pelo modo como os seus sistemas operam. A necessidade de responsabilidade é particularmente crucial com tecnologias de uso sensível como o reconhecimento facial. Recentemente, tem havido um crescimento da procura por tecnologia de reconhecimento facial, sobretudo por parte de organizações policiais que veem o potencial da tecnologia em aplicações como encontrar crianças desaparecidas. No entanto, estas tecnologias podem ser potencialmente usadas por um governo para pôr em risco as liberdades fundamentais dos seus cidadãos, por exemplo, permitindo vigilância contínua de indivíduos específicos. Por isso, cientistas de dados e organizações precisam ser responsáveis pelo impacto do seu sistema de IA nos indivíduos ou na sociedade.

[![Investigador líder em IA alerta para vigilância massiva através do reconhecimento facial](../../../../translated_images/pt-PT/accountability.41d8c0f4b85b6231.webp)](https://www.youtube.com/watch?v=Wldt8P5V6D0 "Abordagem da Microsoft para IA Responsável")

> 🎥 Clique na imagem acima para ver um vídeo: Alertas sobre Vigilância Massiva através do Reconhecimento Facial

Em última análise, uma das maiores questões para a nossa geração, como a primeira geração a trazer IA para a sociedade, é como garantir que os computadores continuam responsáveis perante as pessoas e como garantir que as pessoas que concebem computadores permanecem responsáveis perante todos os outros.

## Avaliação de impacto

Antes de treinar um modelo de aprendizagem automática, é importante conduzir uma avaliação de impacto para entender o propósito do sistema de IA; qual o uso previsto; onde será implementado; e quem irá interagir com o sistema. Estas informações são úteis para o(s) revisor(es) ou testador(es) que avaliam o sistema para saber que fatores devem ter em conta ao identificar riscos potenciais e consequências esperadas.

As seguintes são áreas de foco ao conduzir uma avaliação de impacto:

* **Impacto adverso nos indivíduos**. Estar ciente de quaisquer restrições ou requisitos, uso não suportado ou quaisquer limitações conhecidas que prejudiquem o desempenho do sistema é vital para garantir que o sistema não seja usado de forma a causar danos a pessoas.
* **Requisitos de dados**. Compreender como e onde o sistema usará dados permite aos revisores explorar quaisquer requisitos de dados a ter em conta (ex.: regulamentos GDPR ou HIPAA). Além disso, verificar se a fonte ou a quantidade de dados é substancial para o treino.
* **Resumo do impacto**. Recolher uma lista de possíveis prejuízos que podem surgir do uso do sistema. Ao longo do ciclo de vida da aprendizagem automática, rever se as questões identificadas são mitigadas ou resolvidas.
* **Objetivos aplicáveis** para cada um dos seis princípios fundamentais. Avaliar se os objetivos de cada princípio são atingidos e se existem lacunas.


## Depuração com IA responsável

Tal como depurar uma aplicação de software, depurar um sistema de IA é um processo necessário para identificar e resolver problemas no sistema. Existem muitos fatores que podem afetar o desempenho de um modelo, fazendo com que não cumpra as expectativas ou princípios da IA responsável. A maioria das métricas tradicionais de desempenho de modelo são agregados quantitativos de desempenho, que não são suficientes para analisar como um modelo viola os princípios da IA responsável. Além disso, um modelo de aprendizagem automática é uma caixa preta, o que dificulta compreender o que impulsiona o seu resultado ou fornecer explicações quando comete um erro. Mais adiante neste curso, iremos aprender a usar o painel de IA Responsável para ajudar a depurar sistemas de IA. Este painel fornece uma ferramenta holística para cientistas de dados e programadores de IA para realizar:

* **Análise de erros**. Para identificar a distribuição de erros do modelo que podem afetar a equidade ou fiabilidade do sistema.
* **Visão geral do modelo**. Para descobrir onde existem disparidades no desempenho do modelo entre diferentes cohortes de dados.
* **Análise de dados**. Para compreender a distribuição dos dados e identificar quaisquer preconceitos potenciais nos dados que possam levar a problemas de equidade, inclusão e fiabilidade.
* **Interpretabilidade do modelo**. Para entender o que afeta ou influencia as previsões do modelo. Isto ajuda a explicar o comportamento do modelo, importante para transparência e responsabilidade.


## 🚀 Desafio
 
Para prevenir que prejuízos sejam introduzidos desde o início, devemos:

- ter diversidade de origens e perspetivas entre as pessoas que trabalham nos sistemas
- investir em conjuntos de dados que refletem a diversidade da nossa sociedade
- desenvolver melhores métodos ao longo do ciclo de vida da aprendizagem automática para detetar e corrigir IA irresponsável quando ocorre

Pense em cenários reais onde a falta de confiança num modelo é evidente na construção e uso do modelo. O que mais devemos considerar?

## [Questionário pós-aula](https://ff-quizzes.netlify.app/en/ml/)

## Revisão & Estudo autónomo
 
Nesta lição, aprendeu alguns conceitos básicos sobre equidade e injustiça na aprendizagem automática.
 
Veja este workshop para aprofundar os temas:

- À procura de IA responsável: Trazendo princípios para a prática por Besmira Nushi, Mehrnoosh Sameki e Amit Sharma

[![Caixa de Ferramentas de IA Responsável: Um framework open-source para construir IA responsável](https://img.youtube.com/vi/tGgJCrA-MZU/0.jpg)](https://www.youtube.com/watch?v=tGgJCrA-MZU "Caixa de Ferramentas RAI: Um framework open-source para construir IA responsável")

> 🎥 Clique na imagem acima para um vídeo: Caixa de Ferramentas RAI: Um framework open-source para construir IA responsável por Besmira Nushi, Mehrnoosh Sameki, e Amit Sharma

Leia também: 

- Centro de recursos de IA Responsável da Microsoft: [Recursos de IA Responsável – Microsoft AI](https://www.microsoft.com/ai/responsible-ai-resources?activetab=pivot1%3aprimaryr4) 

- Grupo de investigação FATE da Microsoft: [FATE: Justiça, Responsabilidade, Transparência, e Ética em IA - Microsoft Research](https://www.microsoft.com/research/theme/fate/) 

Caixa de Ferramentas RAI: 

- [Repositório GitHub da Caixa de Ferramentas de IA Responsável](https://github.com/microsoft/responsible-ai-toolbox)

Saiba mais sobre as ferramentas do Azure Machine Learning para garantir justiça:

- [Azure Machine Learning](https://docs.microsoft.com/azure/machine-learning/concept-fairness-ml?WT.mc_id=academic-77952-leestott) 

## Exercício

[Explore a Caixa de Ferramentas RAI](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Aviso Legal**:
Este documento foi traduzido utilizando o serviço de tradução automática [Co-op Translator](https://github.com/Azure/co-op-translator). Embora nos esforcemos pela precisão, esteja ciente de que traduções automáticas podem conter erros ou imprecisões. O documento original na sua língua nativa deve ser considerado a fonte autorizada. Para informações críticas, recomenda-se tradução profissional humana. Não nos responsabilizamos por quaisquer mal-entendidos ou interpretações incorretas resultantes da utilização desta tradução.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->