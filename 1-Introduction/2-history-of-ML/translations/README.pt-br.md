# História do machine learning

![Resumo da história do machine learning no sketchnote](../../../sketchnotes/ml-history.png)
> Sketchnote por [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Teste pré-aula](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/3?loc=ptbr)

Nesta lição, veremos os principais marcos da história do machine learning e da artificial intelligence.

A história da inteligência artificial, IA, como um campo está entrelaçada com a história do machine learning, pois os algoritmos e os avanços computacionais por trás do machine learning contribuíram para o desenvolvimento da inteligência artificial. É útil lembrar que, embora esses campos como áreas distintas de investigação tenham começado a se cristalizar na década de 1950, importantes [descobertas algorítmicas, estatísticas, matemáticas, computacionais e técnicas](https://wikipedia.org/wiki/Timeline_of_machine_learning)
precederam e se sobrepuseram com esta época. Na verdade, as pessoas têm refletido sobre essas questões por [centenas de anos](https://wikipedia.org/wiki/History_of_artificial_intelligence): este artigo discute a base intelectual histórica da ideia de uma 'máquina pensante'.

## Descobertas notáveis

- 1763, 1812 [Teorema de Bayes](https://wikipedia.org/wiki/Bayes%27_theorem) e seus predecessores. Este teorema e suas aplicações fundamentam a inferência, descrevendo a probabilidade de um evento ocorrer com base em conhecimento prévio.
- 1805 [Teoria dos Mínimos Quadrados](https://wikipedia.org/wiki/Least_squares) pelo matemático francês Adrien-Marie Legendre. Esta teoria, que você aprenderá em nossa unidade de regressão, ajuda no ajuste de dados.
- 1913 [Cadeias de Markov](https://wikipedia.org/wiki/Markov_chain) com o nome do matemático russo Andrey Markov é usado para descrever uma sequência de eventos possíveis com base em um estado anterior.
- 1957 [Perceptron](https://wikipedia.org/wiki/Perceptron) é um tipo de classificador linear inventado pelo psicólogo americano Frank Rosenblatt que fundamenta os avanços no aprendizado profundo.
- 1967 [Vizinho mais próximo](https://wikipedia.org/wiki/Nearest_neighbor) é um algoritmo originalmente projetado para mapear rotas. Em um contexto de ML, ele é usado para detectar padrões.
- 1970 [Backpropagation](https://wikipedia.org/wiki/Backpropagation) é usado para treinar [redes neurais feedforward](https://wikipedia.org/wiki/Feedforward_neural_network).
- 1982 [Redes Neurais Recorrentes](https://wikipedia.org/wiki/Recurrent_neural_network) são redes neurais artificiais derivadas de redes neurais feedforward que criam gráficos temporais.

✅ Faça uma pequena pesquisa. Que outras datas se destacam como fundamentais na história do ML e da AI?

## 1950: Máquinas que pensam

Alan Turing, uma pessoa verdadeiramente notável que foi eleita [pelo público em 2019](https://wikipedia.org/wiki/Icons:_The_Greatest_Person_of_the_20th_Century) como o maior cientista do século 20, é creditado por ajudar a lançar as bases para o conceito de uma 'máquina que pode pensar'. Ele lutou contra os pessimistas e sua própria necessidade de evidências empíricas desse conceito, em parte criando o [Teste de Turing](https://www.bbc.com/news/technology-18475646), que você explorará em nossas lições de NPL.

## 1956: Projeto de Pesquisa de Verão de Dartmouth

"O Projeto de Pesquisa de Verão de Dartmouth sobre inteligência artificial foi um evento seminal para a inteligência artificial como um campo", e foi aqui que o termo 'inteligência artificial (AI)' foi cunhado ([fonte](https://250.dartmouth.edu/highlights/artificial-intelligence-ai-coined-dartmouth)).

> Cada aspecto da aprendizagem ou qualquer outra característica da inteligência pode, em princípio, ser descrito com tanta precisão que uma máquina pode ser feita para simulá-lo.

O pesquisador principal, professor de matemática John McCarthy, esperava "proceder com base na conjectura de que cada aspecto do aprendizado ou qualquer outra característica da inteligência pode, em princípio, ser descrito de forma tão precisa que uma máquina pode ser feita para simulá-lo". Os participantes incluíram outro luminar da área, Marvin Minsky.

O workshop é creditado por ter iniciado e encorajado várias discussões, incluindo "o surgimento de métodos simbólicos, sistemas focados em domínios limitados (primeiros sistemas especialistas) e sistemas dedutivos versus sistemas indutivos." ([fonte](https://wikipedia.org/wiki/Dartmouth_workshop)).

## 1956 - 1974: "Os anos dourados"

Dos anos 1950 até meados dos anos 1970, o otimismo era alto na esperança de que a IA pudesse resolver muitos problemas. Em 1967, Marvin Minsky afirmou com segurança que "dentro de uma geração ... o problema de criar 'inteligência artificial' será substancialmente resolvido." (Minsky, Marvin (1967), Computation: Finite and Infinite Machines, Englewood Cliffs, N.J.: Prentice-Hall)

A pesquisa em processamento de linguagem natural floresceu, a pesquisa foi refinada e tornada mais poderosa, e o conceito de "micro-mundos" foi criado, no qual tarefas simples são concluídas usando instruções de linguagem simples.

A pesquisa foi bem financiada por agências governamentais, avanços foram feitos em computação e algoritmos e protótipos de máquinas inteligentes foram construídos. Algumas dessas máquinas incluem:

* [Shakey o robô](https://wikipedia.org/wiki/Shakey_the_robot), quem poderia manobrar e decidir como realizar as tarefas de forma 'inteligente'.

    ![Shakey, o robô inteligente](../images/shakey.jpg)
    > Shakey em 1972

* Eliza, um dos primeiros 'chatterbot', podia conversar com as pessoas e agir como uma 'terapeuta' primitiva. Você aprenderá mais sobre Eliza nas lições de NPL.

    ![Eliza, a bot](../images/eliza.png)
    > Uma versão de Eliza, um chatbot

* O "mundo dos blocos" era um exemplo de micro-mundo onde os blocos podiam ser empilhados e classificados, e experimentos em máquinas de ensino para tomar decisões podiam ser testados. Avanços construídos com bibliotecas como [SHRDLU](https://wikipedia.org/wiki/SHRDLU) ajudaram a impulsionar o processamento de linguagem.

    [![mundo dos blocos com SHRDLU](https://img.youtube.com/vi/QAJz4YKUwqw/0.jpg)](https://www.youtube.com/watch?v=QAJz4YKUwqw "mundo dos blocos com SHRDLU")
    
    > 🎥 Clique na imagem acima para ver um vídeo: Mundo dos blocos com SHRDLU

## 1974 - 1980: "O inverno da AI"

Em meados da década de 1970, ficou claro que a complexidade de criar 'máquinas inteligentes' havia sido subestimada e que sua promessa, dado o poder de computação disponível, havia sido exagerada. O financiamento secou e a confiança no setor desacelerou. Alguns problemas que afetaram a confiança incluem:

- **Limitações**. O poder de computação era muito limitado.
- **Explosão combinatória**. A quantidade de parâmetros que precisavam ser treinados cresceu exponencialmente à medida que mais computadores eram exigidos, sem uma evolução paralela no poder e nas capacidades de computação.
- **Falta de dados**. Havia uma escassez de dados que dificultou o processo de teste, desenvolvimento e refinamento dos algoritmos.
- **Estamos fazendo as perguntas certas?**. As próprias perguntas que estavam sendo feitas começaram a ser questionadas. Os pesquisadores começaram a criticar suas abordagens:
  - Os testes de Turing foram desafiados através, entre outras ideias, da 'teoria da sala chinesa' que postulava que "programar um computador digital pode fazer com que pareça compreender a linguagem, mas não pode produzir uma compreensão verdadeira". ([fonte](https://plato.stanford.edu/entries/chinese-room/))
  - A ética da introdução de inteligências artificiais como o "terapeuta" de ELIZA na sociedade tem sido questionada.

Ao mesmo tempo, várias escolas de pensamento de IA começaram a se formar. Uma dicotomia foi estabelecida entre as práticas ["scruffy" versus "neat AI"](https://wikipedia.org/wiki/Neats_and_scruffies). O laboratório do _Scruffy_ otimizou os programas por horas até obter os resultados desejados. O laboratório do _Neat_ "focava na solução de problemas lógicos e formais". ELIZA e SHRDLU eram sistemas desalinhados bem conhecidos. Na década de 1980, quando surgiu a demanda para tornar os sistemas de ML reproduzíveis, a abordagem _neat_ gradualmente assumiu o controle, à medida que seus resultados eram mais explicáveis.

## Sistemas especialistas de 1980

À medida que o campo cresceu, seus benefícios para os negócios tornaram-se mais claros e, na década de 1980, o mesmo aconteceu com a proliferação de 'sistemas especialistas'. "Os sistemas especialistas estavam entre as primeiras formas verdadeiramente bem-sucedidas de software de inteligência artificial (AI)." ([fonte](https://wikipedia.org/wiki/Expert_system)).

Na verdade, esse tipo de sistema é _híbrido_, consistindo parcialmente em um mecanismo de regras que define os requisitos de negócios e um mecanismo de inferência que potencializa o sistema de regras para deduzir novos fatos.

Essa era também viu uma crescente atenção dada às redes neurais.

## 1987 - 1993: AI 'Chill'

A proliferação de hardware de sistemas especialistas especializados teve o infeliz efeito de se tornar muito especializado. A ascensão dos computadores pessoais também competiu com esses sistemas grandes, especializados e centralizados. A democratização da computação havia começado e, por fim, pavimentou o caminho para a explosão moderna de big data.

## 1993 - 2011

Essa época viu uma nova era para o ML e a AI serem capazes de resolver alguns dos problemas que eram causados anteriormente pela falta de dados e capacidade de computação. A quantidade de dados começou a aumentar rapidamente e se tornar mais amplamente disponível, para melhor e para pior, especialmente com o advento do smartphone por volta de 2007. O poder de computação se expandiu exponencialmente e os algoritmos evoluíram junto. O campo começou a ganhar maturidade à medida que os dias livres do passado começaram a se cristalizar em uma verdadeira disciplina.

## Agora

Hoje, o machine learning e a inteligência artificial afetam quase todas as partes de nossa vida. Esta era requer uma compreensão cuidadosa dos riscos e efeitos potenciais desses algoritmos em vidas humanas. Como disse Brad Smith, da Microsoft, "a tecnologia da informação levanta questões que vão ao cerne das proteções fundamentais dos direitos humanos, como privacidade e liberdade de expressão. Essas questões aumentam a responsabilidade das empresas de tecnologia que criam esses produtos. Observe, elas também exigem um governo cuidadoso regulamentação e o desenvolvimento de padrões sobre usos aceitáveis​​" ([fonte](https://www.technologyreview.com/2019/12/18/102365/the-future-of-ais-impact-on-society/)).

Resta saber o que o futuro reserva, mas é importante entender esses sistemas de computador e o software e algoritmos que eles executam. Esperamos que este curso lhe ajude a obter um melhor entendimento para que você possa decidir por si mesmo.

[![A história do deep learning](https://img.youtube.com/vi/mTtDfKgLm54/0.jpg)](https://www.youtube.com/watch?v=mTtDfKgLm54 "A história do deep learning")
> 🎥 Clique na imagem acima para ver um vídeo: Yann LeCun discute a história do deep learning nesta palestra

---
## 🚀Desafio

Explore um desses momentos históricos e aprenda mais sobre as pessoas por trás deles. Existem personagens fascinantes e nenhuma descoberta científica foi criada em um vácuo cultural. O que você descobriu?

## [Questionário pós-aula](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/4?loc=ptbr)

## Revisão e Autoestudo

Aqui estão os itens para assistir e ouvir:

[Este podcast em que Amy Boyd discute a evolução da AI](http://runasradio.com/Shows/Show/739)

[![A história da AI ​​por Amy Boyd](https://img.youtube.com/vi/EJt3_bFYKss/0.jpg)](https://www.youtube.com/watch?v=EJt3_bFYKss "A história da AI ​​por Amy Boyd")

## Tarefa

[Crie uma linha do tempo](assignment.pt-br.md)
