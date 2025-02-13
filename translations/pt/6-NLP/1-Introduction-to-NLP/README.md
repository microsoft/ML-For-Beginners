# Introdução ao processamento de linguagem natural

Esta lição cobre uma breve história e conceitos importantes de *processamento de linguagem natural*, um subcampo da *linguística computacional*.

## [Quiz pré-aula](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/31/)

## Introdução

O PLN, como é comumente conhecido, é uma das áreas mais conhecidas onde o aprendizado de máquina foi aplicado e utilizado em software de produção.

✅ Você consegue pensar em algum software que usa todos os dias e que provavelmente tem algum PLN embutido? E quanto aos seus programas de processamento de texto ou aplicativos móveis que você usa regularmente?

Você aprenderá sobre:

- **A ideia de idiomas**. Como as línguas se desenvolveram e quais foram as principais áreas de estudo.
- **Definição e conceitos**. Você também aprenderá definições e conceitos sobre como os computadores processam texto, incluindo análise sintática, gramática e identificação de substantivos e verbos. Existem algumas tarefas de codificação nesta lição, e vários conceitos importantes são introduzidos que você aprenderá a codificar mais adiante nas próximas lições.

## Linguística computacional

A linguística computacional é uma área de pesquisa e desenvolvimento ao longo de muitas décadas que estuda como os computadores podem trabalhar com, e até mesmo entender, traduzir e se comunicar em línguas. O processamento de linguagem natural (PLN) é um campo relacionado focado em como os computadores podem processar línguas 'naturais', ou humanas.

### Exemplo - ditado por telefone

Se você já ditou algo para o seu telefone em vez de digitar ou fez uma pergunta a um assistente virtual, sua fala foi convertida em forma de texto e depois processada ou *analisada* a partir da língua que você falou. As palavras-chave detectadas foram então processadas em um formato que o telefone ou assistente poderia entender e agir.

![compreensão](../../../../translated_images/comprehension.619708fc5959b0f6a24ebffba2ad7b0625391a476141df65b43b59de24e45c6f.pt.png)
> A verdadeira compreensão linguística é difícil! Imagem de [Jen Looper](https://twitter.com/jenlooper)

### Como essa tecnologia é possível?

Isso é possível porque alguém escreveu um programa de computador para fazer isso. Algumas décadas atrás, alguns escritores de ficção científica previram que as pessoas falariam principalmente com seus computadores, e os computadores sempre entenderiam exatamente o que elas queriam dizer. Infelizmente, acabou sendo um problema mais difícil do que muitos imaginavam, e embora hoje seja um problema muito melhor compreendido, existem desafios significativos para alcançar um processamento de linguagem natural 'perfeito' quando se trata de entender o significado de uma frase. Este é um problema particularmente difícil quando se trata de entender humor ou detectar emoções, como sarcasmo, em uma frase.

Neste ponto, você pode estar se lembrando das aulas da escola em que o professor abordava as partes da gramática em uma frase. Em alguns países, os alunos aprendem gramática e linguística como uma disciplina dedicada, mas em muitos, esses tópicos estão incluídos como parte do aprendizado de uma língua: seja sua primeira língua na escola primária (aprendendo a ler e escrever) e talvez uma segunda língua no ensino secundário, ou no ensino médio. Não se preocupe se você não é um especialista em diferenciar substantivos de verbos ou advérbios de adjetivos!

Se você tem dificuldades em entender a diferença entre o *presente simples* e o *presente contínuo*, você não está sozinho. Isso é um desafio para muitas pessoas, até mesmo falantes nativos de uma língua. A boa notícia é que os computadores são realmente bons em aplicar regras formais, e você aprenderá a escrever código que pode *analisar* uma frase tão bem quanto um humano. O desafio maior que você examinará mais tarde é entender o *significado* e o *sentimento* de uma frase.

## Pré-requisitos

Para esta lição, o principal pré-requisito é ser capaz de ler e entender a língua desta lição. Não há problemas matemáticos ou equações para resolver. Embora o autor original tenha escrito esta lição em inglês, ela também foi traduzida para outras línguas, então você pode estar lendo uma tradução. Existem exemplos onde um número de línguas diferentes é usado (para comparar as diferentes regras gramaticais de diferentes línguas). Estes *não* são traduzidos, mas o texto explicativo é, então o significado deve estar claro.

Para as tarefas de codificação, você usará Python e os exemplos estão utilizando Python 3.8.

Nesta seção, você precisará, e usará:

- **Compreensão do Python 3**. Compreensão da linguagem de programação em Python 3, esta lição utiliza entrada, loops, leitura de arquivos, arrays.
- **Visual Studio Code + extensão**. Usaremos o Visual Studio Code e sua extensão Python. Você também pode usar um IDE Python de sua escolha.
- **TextBlob**. [TextBlob](https://github.com/sloria/TextBlob) é uma biblioteca de processamento de texto simplificada para Python. Siga as instruções no site do TextBlob para instalá-lo em seu sistema (instale os corpora também, conforme mostrado abaixo):

   ```bash
   pip install -U textblob
   python -m textblob.download_corpora
   ```

> 💡 Dica: Você pode executar Python diretamente em ambientes do VS Code. Consulte a [documentação](https://code.visualstudio.com/docs/languages/python?WT.mc_id=academic-77952-leestott) para mais informações.

## Conversando com máquinas

A história de tentar fazer os computadores entenderem a linguagem humana remonta a décadas, e um dos primeiros cientistas a considerar o processamento de linguagem natural foi *Alan Turing*.

### O 'teste de Turing'

Quando Turing estava pesquisando *inteligência artificial* na década de 1950, ele considerou se um teste de conversa poderia ser dado a um humano e a um computador (por meio de correspondência digitada) onde o humano na conversa não tinha certeza se estava conversando com outro humano ou com um computador.

Se, após um certo tempo de conversa, o humano não pudesse determinar se as respostas eram de um computador ou não, poderia-se dizer que o computador estava *pensando*?

### A inspiração - 'o jogo da imitação'

A ideia para isso veio de um jogo de festa chamado *O Jogo da Imitacão* onde um interrogador está sozinho em uma sala e encarregado de determinar qual das duas pessoas (em outra sala) é do sexo masculino e qual é do sexo feminino, respectivamente. O interrogador pode enviar notas e deve tentar pensar em perguntas onde as respostas escritas revelem o gênero da pessoa misteriosa. É claro que os jogadores na outra sala estão tentando enganar o interrogador, respondendo perguntas de uma forma que possa induzi-lo ao erro ou confundi-lo, enquanto também dão a aparência de responder honestamente.

### Desenvolvendo Eliza

Na década de 1960, um cientista do MIT chamado *Joseph Weizenbaum* desenvolveu [*Eliza*](https://wikipedia.org/wiki/ELIZA), um 'terapeuta' de computador que faria perguntas ao humano e daria a aparência de entender suas respostas. No entanto, embora Eliza pudesse analisar uma frase e identificar certos construtos gramaticais e palavras-chave para dar uma resposta razoável, não se poderia dizer que ela *entendia* a frase. Se Eliza fosse apresentada com uma frase seguindo o formato "**Eu estou** <u>triste</u>", ela poderia reorganizar e substituir palavras na frase para formar a resposta "Há quanto tempo você **está** <u>triste</u>?".

Isso dava a impressão de que Eliza entendia a afirmação e estava fazendo uma pergunta de seguimento, enquanto na realidade, ela estava apenas mudando o tempo verbal e adicionando algumas palavras. Se Eliza não conseguisse identificar uma palavra-chave para a qual tinha uma resposta, ela daria uma resposta aleatória que poderia ser aplicável a muitas afirmações diferentes. Eliza poderia ser facilmente enganada; por exemplo, se um usuário escrevesse "**Você é** uma <u>bicicleta</u>", ela poderia responder com "Há quanto tempo **eu sou** uma <u>bicicleta</u>?", em vez de uma resposta mais razoável.

[![Conversando com Eliza](https://img.youtube.com/vi/RMK9AphfLco/0.jpg)](https://youtu.be/RMK9AphfLco "Conversando com Eliza")

> 🎥 Clique na imagem acima para assistir a um vídeo sobre o programa original ELIZA

> Nota: Você pode ler a descrição original de [Eliza](https://cacm.acm.org/magazines/1966/1/13317-elizaa-computer-program-for-the-study-of-natural-language-communication-between-man-and-machine/abstract) publicada em 1966 se tiver uma conta da ACM. Alternativamente, leia sobre Eliza na [wikipedia](https://wikipedia.org/wiki/ELIZA)

## Exercício - codificando um bot conversacional básico

Um bot conversacional, como Eliza, é um programa que provoca a entrada do usuário e parece entender e responder de forma inteligente. Ao contrário de Eliza, nosso bot não terá várias regras que lhe conferem a aparência de ter uma conversa inteligente. Em vez disso, nosso bot terá apenas uma habilidade, que é manter a conversa com respostas aleatórias que podem funcionar em quase qualquer conversa trivial.

### O plano

Seus passos ao construir um bot conversacional:

1. Imprima instruções aconselhando o usuário sobre como interagir com o bot
2. Inicie um loop
   1. Aceite a entrada do usuário
   2. Se o usuário pediu para sair, então saia
   3. Processem a entrada do usuário e determine a resposta (neste caso, a resposta é uma escolha aleatória de uma lista de possíveis respostas genéricas)
   4. Imprima a resposta
3. volte ao passo 2

### Construindo o bot

Vamos criar o bot a seguir. Começaremos definindo algumas frases.

1. Crie este bot você mesmo em Python com as seguintes respostas aleatórias:

    ```python
    random_responses = ["That is quite interesting, please tell me more.",
                        "I see. Do go on.",
                        "Why do you say that?",
                        "Funny weather we've been having, isn't it?",
                        "Let's change the subject.",
                        "Did you catch the game last night?"]
    ```

    Aqui está uma saída de exemplo para guiá-lo (a entrada do usuário está nas linhas que começam com `>`):

    ```output
    Hello, I am Marvin, the simple robot.
    You can end this conversation at any time by typing 'bye'
    After typing each answer, press 'enter'
    How are you today?
    > I am good thanks
    That is quite interesting, please tell me more.
    > today I went for a walk     
    Did you catch the game last night?
    > I did, but my team lost
    Funny weather we've been having, isn't it?
    > yes but I hope next week is better
    Let's change the subject.
    > ok, lets talk about music
    Why do you say that?
    > because I like music!
    Why do you say that?
    > bye
    It was nice talking to you, goodbye!
    ```

    Uma possível solução para a tarefa está [aqui](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/1-Introduction-to-NLP/solution/bot.py)

    ✅ Pare e considere

    1. Você acha que as respostas aleatórias poderiam 'enganar' alguém a pensar que o bot realmente o entendia?
    2. Que recursos o bot precisaria para ser mais eficaz?
    3. Se um bot realmente pudesse 'entender' o significado de uma frase, ele precisaria 'lembrar' o significado de frases anteriores em uma conversa também?

---

## 🚀Desafio

Escolha um dos elementos "pare e considere" acima e tente implementá-los em código ou escreva uma solução no papel usando pseudocódigo.

Na próxima lição, você aprenderá sobre várias outras abordagens para a análise de linguagem natural e aprendizado de máquina.

## [Quiz pós-aula](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/32/)

## Revisão & Autoestudo

Dê uma olhada nas referências abaixo como oportunidades de leitura adicional.

### Referências

1. Schubert, Lenhart, "Linguística Computacional", *A Enciclopédia de Filosofia de Stanford* (Edição da Primavera de 2020), Edward N. Zalta (ed.), URL = <https://plato.stanford.edu/archives/spr2020/entries/computational-linguistics/>.
2. Universidade de Princeton "Sobre o WordNet." [WordNet](https://wordnet.princeton.edu/). Universidade de Princeton. 2010.

## Tarefa

[Pesquise um bot](assignment.md)

**Isenção de responsabilidade**:  
Este documento foi traduzido utilizando serviços de tradução automática baseados em IA. Embora nos esforcemos pela precisão, esteja ciente de que traduções automatizadas podem conter erros ou imprecisões. O documento original em seu idioma nativo deve ser considerado a fonte autoritativa. Para informações críticas, recomenda-se a tradução profissional feita por humanos. Não nos responsabilizamos por quaisquer mal-entendidos ou interpretações errôneas decorrentes do uso desta tradução.