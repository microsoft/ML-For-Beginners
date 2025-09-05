<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9a6b702d1437c0467e3c5c28d763dac2",
  "translation_date": "2025-09-05T08:43:41+00:00",
  "source_file": "1-Introduction/3-fairness/README.md",
  "language_code": "pt"
}
-->
# Construir soluções de Machine Learning com IA responsável

![Resumo da IA responsável em Machine Learning em um sketchnote](../../../../sketchnotes/ml-fairness.png)
> Sketchnote por [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Questionário pré-aula](https://ff-quizzes.netlify.app/en/ml/)

## Introdução

Neste currículo, começará a descobrir como o machine learning pode impactar e já está a impactar as nossas vidas diárias. Mesmo agora, sistemas e modelos estão envolvidos em tarefas de tomada de decisão diária, como diagnósticos de saúde, aprovações de empréstimos ou deteção de fraudes. Por isso, é importante que esses modelos funcionem bem para fornecer resultados confiáveis. Assim como qualquer aplicação de software, os sistemas de IA podem falhar em atender às expectativas ou gerar resultados indesejáveis. É por isso que é essencial compreender e explicar o comportamento de um modelo de IA.

Imagine o que pode acontecer quando os dados que utiliza para construir esses modelos não incluem certos grupos demográficos, como raça, género, visão política, religião, ou representam esses grupos de forma desproporcional. E se a saída do modelo for interpretada de forma a favorecer um grupo demográfico? Qual é a consequência para a aplicação? Além disso, o que acontece quando o modelo gera um resultado adverso e prejudica as pessoas? Quem é responsável pelo comportamento dos sistemas de IA? Estas são algumas questões que exploraremos neste currículo.

Nesta lição, irá:

- Aumentar a sua consciência sobre a importância da equidade no machine learning e os danos relacionados com a falta de equidade.
- Familiarizar-se com a prática de explorar outliers e cenários incomuns para garantir fiabilidade e segurança.
- Compreender a necessidade de capacitar todos ao projetar sistemas inclusivos.
- Explorar como é vital proteger a privacidade e a segurança dos dados e das pessoas.
- Perceber a importância de uma abordagem de "caixa de vidro" para explicar o comportamento dos modelos de IA.
- Ter em mente como a responsabilidade é essencial para construir confiança nos sistemas de IA.

## Pré-requisito

Como pré-requisito, faça o percurso de aprendizagem "Princípios de IA Responsável" e assista ao vídeo abaixo sobre o tema:

Saiba mais sobre IA Responsável seguindo este [Percurso de Aprendizagem](https://docs.microsoft.com/learn/modules/responsible-ai-principles/?WT.mc_id=academic-77952-leestott)

[![Abordagem da Microsoft para IA Responsável](https://img.youtube.com/vi/dnC8-uUZXSc/0.jpg)](https://youtu.be/dnC8-uUZXSc "Abordagem da Microsoft para IA Responsável")

> 🎥 Clique na imagem acima para assistir ao vídeo: Abordagem da Microsoft para IA Responsável

## Equidade

Os sistemas de IA devem tratar todos de forma justa e evitar afetar grupos semelhantes de pessoas de maneiras diferentes. Por exemplo, quando os sistemas de IA fornecem orientações sobre tratamentos médicos, aplicações de empréstimos ou emprego, devem fazer as mesmas recomendações para todos com sintomas, circunstâncias financeiras ou qualificações profissionais semelhantes. Cada um de nós, como seres humanos, carrega preconceitos herdados que afetam as nossas decisões e ações. Esses preconceitos podem ser evidentes nos dados que usamos para treinar sistemas de IA. Tal manipulação pode, por vezes, ocorrer de forma não intencional. Muitas vezes, é difícil perceber conscientemente quando está a introduzir preconceitos nos dados.

**"Injustiça"** abrange impactos negativos, ou "danos", para um grupo de pessoas, como aqueles definidos em termos de raça, género, idade ou deficiência. Os principais danos relacionados com a equidade podem ser classificados como:

- **Alocação**, se, por exemplo, um género ou etnia for favorecido em detrimento de outro.
- **Qualidade do serviço**. Se treinar os dados para um cenário específico, mas a realidade for muito mais complexa, isso leva a um serviço de desempenho inferior. Por exemplo, um dispensador de sabão que não consegue detetar pessoas com pele escura. [Referência](https://gizmodo.com/why-cant-this-soap-dispenser-identify-dark-skin-1797931773)
- **Denigração**. Criticar ou rotular algo ou alguém de forma injusta. Por exemplo, uma tecnologia de rotulagem de imagens que, de forma infame, rotulou imagens de pessoas de pele escura como gorilas.
- **Sobre ou sub-representação**. A ideia de que um determinado grupo não é visto numa certa profissão, e qualquer serviço ou função que continue a promover isso está a contribuir para o dano.
- **Estereotipagem**. Associar um grupo a atributos pré-definidos. Por exemplo, um sistema de tradução entre inglês e turco pode apresentar imprecisões devido a palavras com associações estereotipadas de género.

![tradução para turco](../../../../1-Introduction/3-fairness/images/gender-bias-translate-en-tr.png)
> tradução para turco

![tradução de volta para inglês](../../../../1-Introduction/3-fairness/images/gender-bias-translate-tr-en.png)
> tradução de volta para inglês

Ao projetar e testar sistemas de IA, precisamos garantir que a IA seja justa e não programada para tomar decisões enviesadas ou discriminatórias, que também são proibidas para os seres humanos. Garantir a equidade na IA e no machine learning continua a ser um desafio sociotécnico complexo.

### Fiabilidade e segurança

Para construir confiança, os sistemas de IA precisam ser fiáveis, seguros e consistentes em condições normais e inesperadas. É importante saber como os sistemas de IA se comportarão numa variedade de situações, especialmente quando se trata de outliers. Ao construir soluções de IA, é necessário focar-se substancialmente em como lidar com uma ampla variedade de circunstâncias que as soluções de IA podem encontrar. Por exemplo, um carro autónomo precisa de colocar a segurança das pessoas como prioridade máxima. Como resultado, a IA que alimenta o carro precisa de considerar todos os cenários possíveis que o carro pode encontrar, como noite, tempestades, nevascas, crianças a atravessar a rua, animais de estimação, obras na estrada, etc. O quão bem um sistema de IA consegue lidar com uma ampla gama de condições de forma fiável e segura reflete o nível de antecipação que o cientista de dados ou desenvolvedor de IA considerou durante o design ou teste do sistema.

> [🎥 Clique aqui para um vídeo: ](https://www.microsoft.com/videoplayer/embed/RE4vvIl)

### Inclusividade

Os sistemas de IA devem ser projetados para envolver e capacitar todos. Ao projetar e implementar sistemas de IA, cientistas de dados e desenvolvedores de IA identificam e abordam potenciais barreiras no sistema que poderiam, de forma não intencional, excluir pessoas. Por exemplo, existem 1 bilião de pessoas com deficiência em todo o mundo. Com o avanço da IA, elas podem aceder a uma ampla gama de informações e oportunidades mais facilmente nas suas vidas diárias. Ao abordar essas barreiras, criam-se oportunidades para inovar e desenvolver produtos de IA com melhores experiências que beneficiam todos.

> [🎥 Clique aqui para um vídeo: inclusividade na IA](https://www.microsoft.com/videoplayer/embed/RE4vl9v)

### Segurança e privacidade

Os sistemas de IA devem ser seguros e respeitar a privacidade das pessoas. As pessoas têm menos confiança em sistemas que colocam a sua privacidade, informações ou vidas em risco. Ao treinar modelos de machine learning, confiamos nos dados para produzir os melhores resultados. Ao fazê-lo, a origem e a integridade dos dados devem ser consideradas. Por exemplo, os dados foram submetidos por utilizadores ou estavam disponíveis publicamente? Além disso, ao trabalhar com os dados, é crucial desenvolver sistemas de IA que possam proteger informações confidenciais e resistir a ataques. À medida que a IA se torna mais prevalente, proteger a privacidade e garantir a segurança de informações pessoais e empresariais importantes está a tornar-se mais crítico e complexo. Questões de privacidade e segurança de dados requerem atenção especial para a IA, pois o acesso aos dados é essencial para que os sistemas de IA façam previsões e decisões precisas e informadas sobre as pessoas.

> [🎥 Clique aqui para um vídeo: segurança na IA](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Como indústria, fizemos avanços significativos em privacidade e segurança, impulsionados significativamente por regulamentações como o RGPD (Regulamento Geral de Proteção de Dados).
- No entanto, com os sistemas de IA, devemos reconhecer a tensão entre a necessidade de mais dados pessoais para tornar os sistemas mais personalizados e eficazes – e a privacidade.
- Assim como com o nascimento de computadores conectados à internet, também estamos a assistir a um grande aumento no número de problemas de segurança relacionados com a IA.
- Ao mesmo tempo, vimos a IA ser usada para melhorar a segurança. Por exemplo, a maioria dos scanners antivírus modernos é alimentada por heurísticas de IA.
- Precisamos garantir que os nossos processos de ciência de dados se harmonizem com as práticas mais recentes de privacidade e segurança.

### Transparência

Os sistemas de IA devem ser compreensíveis. Uma parte crucial da transparência é explicar o comportamento dos sistemas de IA e os seus componentes. Melhorar a compreensão dos sistemas de IA exige que as partes interessadas compreendam como e por que funcionam, para que possam identificar potenciais problemas de desempenho, preocupações com segurança e privacidade, preconceitos, práticas de exclusão ou resultados indesejados. Também acreditamos que aqueles que usam sistemas de IA devem ser honestos e claros sobre quando, por que e como escolhem implementá-los, bem como as limitações dos sistemas que utilizam. Por exemplo, se um banco usa um sistema de IA para apoiar as suas decisões de concessão de crédito, é importante examinar os resultados e entender quais dados influenciam as recomendações do sistema. Os governos estão a começar a regulamentar a IA em vários setores, por isso, cientistas de dados e organizações devem explicar se um sistema de IA atende aos requisitos regulamentares, especialmente quando há um resultado indesejável.

> [🎥 Clique aqui para um vídeo: transparência na IA](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Como os sistemas de IA são tão complexos, é difícil entender como funcionam e interpretar os resultados.
- Essa falta de compreensão afeta a forma como esses sistemas são geridos, operacionalizados e documentados.
- Mais importante ainda, essa falta de compreensão afeta as decisões tomadas com base nos resultados produzidos por esses sistemas.

### Responsabilidade

As pessoas que projetam e implementam sistemas de IA devem ser responsáveis pelo funcionamento dos seus sistemas. A necessidade de responsabilidade é particularmente crucial com tecnologias sensíveis, como o reconhecimento facial. Recentemente, tem havido uma crescente procura por tecnologia de reconhecimento facial, especialmente por parte de organizações de aplicação da lei que veem o potencial da tecnologia em usos como encontrar crianças desaparecidas. No entanto, essas tecnologias podem ser usadas por um governo para colocar em risco as liberdades fundamentais dos seus cidadãos, por exemplo, ao permitir a vigilância contínua de indivíduos específicos. Por isso, cientistas de dados e organizações precisam ser responsáveis pelo impacto dos seus sistemas de IA em indivíduos ou na sociedade.

[![Investigador líder em IA alerta para vigilância em massa através do reconhecimento facial](../../../../1-Introduction/3-fairness/images/accountability.png)](https://www.youtube.com/watch?v=Wldt8P5V6D0 "Abordagem da Microsoft para IA Responsável")

> 🎥 Clique na imagem acima para assistir ao vídeo: Alertas sobre Vigilância em Massa através do Reconhecimento Facial

Em última análise, uma das maiores questões para a nossa geração, como a primeira geração a trazer a IA para a sociedade, é como garantir que os computadores permaneçam responsáveis perante as pessoas e como garantir que as pessoas que projetam computadores sejam responsáveis perante todos os outros.

## Avaliação de impacto

Antes de treinar um modelo de machine learning, é importante realizar uma avaliação de impacto para entender o propósito do sistema de IA; qual é o uso pretendido; onde será implementado; e quem interagirá com o sistema. Estas avaliações são úteis para os revisores ou testadores avaliarem o sistema e saberem quais fatores considerar ao identificar potenciais riscos e consequências esperadas.

As seguintes áreas devem ser focadas ao realizar uma avaliação de impacto:

* **Impacto adverso nos indivíduos**. Estar ciente de quaisquer restrições ou requisitos, usos não suportados ou quaisquer limitações conhecidas que prejudiquem o desempenho do sistema é vital para garantir que o sistema não seja usado de forma a causar danos aos indivíduos.
* **Requisitos de dados**. Compreender como e onde o sistema usará os dados permite que os revisores explorem quaisquer requisitos de dados que precisem de ser considerados (por exemplo, regulamentações de dados como RGPD ou HIPAA). Além disso, examine se a origem ou a quantidade de dados é substancial para o treino.
* **Resumo do impacto**. Reúna uma lista de potenciais danos que possam surgir do uso do sistema. Ao longo do ciclo de vida do ML, reveja se os problemas identificados foram mitigados ou resolvidos.
* **Objetivos aplicáveis** para cada um dos seis princípios fundamentais. Avalie se os objetivos de cada princípio foram cumpridos e se existem lacunas.

## Depuração com IA responsável

Semelhante à depuração de uma aplicação de software, a depuração de um sistema de IA é um processo necessário para identificar e resolver problemas no sistema. Existem muitos fatores que podem afetar o desempenho de um modelo ou a sua responsabilidade. A maioria das métricas tradicionais de desempenho de modelos são agregados quantitativos do desempenho do modelo, o que não é suficiente para analisar como um modelo viola os princípios de IA responsável. Além disso, um modelo de machine learning é uma "caixa preta", o que dificulta entender o que motiva os seus resultados ou fornecer explicações quando comete um erro. Mais tarde neste curso, aprenderemos a usar o painel de IA Responsável para ajudar a depurar sistemas de IA. O painel fornece uma ferramenta holística para cientistas de dados e desenvolvedores de IA realizarem:

* **Análise de erros**. Para identificar a distribuição de erros do modelo que pode afetar a equidade ou fiabilidade do sistema.
* **Visão geral do modelo**. Para descobrir onde existem disparidades no desempenho do modelo em diferentes coortes de dados.
* **Análise de dados**. Para compreender a distribuição dos dados e identificar potenciais preconceitos nos dados que possam levar a problemas de equidade, inclusividade e fiabilidade.
* **Interpretabilidade do modelo**. Para entender o que afeta ou influencia as previsões do modelo. Isso ajuda a explicar o comportamento do modelo, o que é importante para transparência e responsabilidade.

## 🚀 Desafio

Para evitar que danos sejam introduzidos desde o início, devemos:

- ter diversidade de origens e perspetivas entre as pessoas que trabalham nos sistemas
- investir em conjuntos de dados que reflitam a diversidade da nossa sociedade
- desenvolver melhores métodos ao longo do ciclo de vida do machine learning para detetar e corrigir problemas de IA responsável quando eles ocorrerem

Pense em cenários da vida real onde a falta de confiança num modelo é evidente na construção e utilização do modelo. O que mais deveríamos considerar?

## [Questionário pós-aula](https://ff-quizzes.netlify.app/en/ml/)

## Revisão e Estudo Individual

Nesta lição, aprendeu alguns conceitos básicos sobre equidade e falta de equidade no machine learning.
Assista a este workshop para aprofundar os tópicos:

- Em busca de IA responsável: Aplicando princípios na prática por Besmira Nushi, Mehrnoosh Sameki e Amit Sharma

[![Responsible AI Toolbox: Uma framework de código aberto para construir IA responsável](https://img.youtube.com/vi/tGgJCrA-MZU/0.jpg)](https://www.youtube.com/watch?v=tGgJCrA-MZU "RAI Toolbox: Uma framework de código aberto para construir IA responsável")

> 🎥 Clique na imagem acima para assistir ao vídeo: RAI Toolbox: Uma framework de código aberto para construir IA responsável por Besmira Nushi, Mehrnoosh Sameki e Amit Sharma

Além disso, leia:

- Centro de recursos de IA responsável da Microsoft: [Responsible AI Resources – Microsoft AI](https://www.microsoft.com/ai/responsible-ai-resources?activetab=pivot1%3aprimaryr4)

- Grupo de pesquisa FATE da Microsoft: [FATE: Fairness, Accountability, Transparency, and Ethics in AI - Microsoft Research](https://www.microsoft.com/research/theme/fate/)

RAI Toolbox:

- [Repositório GitHub do Responsible AI Toolbox](https://github.com/microsoft/responsible-ai-toolbox)

Leia sobre as ferramentas do Azure Machine Learning para garantir equidade:

- [Azure Machine Learning](https://docs.microsoft.com/azure/machine-learning/concept-fairness-ml?WT.mc_id=academic-77952-leestott)

## Tarefa

[Explore o RAI Toolbox](assignment.md)

---

**Aviso Legal**:  
Este documento foi traduzido utilizando o serviço de tradução por IA [Co-op Translator](https://github.com/Azure/co-op-translator). Embora nos esforcemos para garantir a precisão, esteja ciente de que traduções automáticas podem conter erros ou imprecisões. O documento original no seu idioma nativo deve ser considerado a fonte oficial. Para informações críticas, recomenda-se uma tradução profissional realizada por humanos. Não nos responsabilizamos por quaisquer mal-entendidos ou interpretações incorretas resultantes do uso desta tradução.