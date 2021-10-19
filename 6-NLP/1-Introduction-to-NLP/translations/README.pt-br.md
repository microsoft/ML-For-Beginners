# Introdução ao Processamento de Linguagem Natural

Esta aula cobre uma breve história e conceitos importantes do *processamento de linguagem natural*, uma subárea da *Linguística computacional*.

## [Quiz pŕe-aula](https://white-water-09ec41f0f.azurestaticapps.net/quiz/31/)

## Introdução

PLN (ou, em inglês NLP), como é geralmente conhecido, é uma das áreas mais conhecidas onde o aprendizado de máquina (machine learning) tem sido aplicado e usado na produção de software.

✅ Você consegue pensar em algum software que você usa todo dia que provavelmente tem algum PLN integrado? E programas de processamento de palavras e aplicativos mobile que você usa com frequência?

Você vai aprender sobre:

- **A ideia das linguagens**. Como as linguagens se desenvolveram e quais as maiores áreas de estudo têm sido.
- **Definição e conceitos**. Você também vai aprender definições e conceitos sobre como os computadores processam texto, incluindo análise sintática (parsing), gramática, e identificação de substantivos e verbos. Existem algumas tarefas de programação nesta aula, e muitos conceitos importantes serão introduzidos, que você ira aprender a programar nas próximas aulas.

## Linguística computacional

Linguística computacional é uma área de pesquisa e desenvolvimento ao longo de várias décadas que estuda como computadores podem trabalhar com, e até entender, traduzir, e comunicar com linguagens. Processamento de linguagem natural (PLN) é um campo correlato que tem o foco em como computadores podem processar linguagens 'naturais' ou humanas.

### Exemplo - transcrição de voz no celular

Se você já usou o recurso de digitação por voz ao invés de escrever ou fez uma pergunta para uma assistente virtual, sua fala foi convertida para o formato textual e então processou ou *parseou* (analisou a sintaxe) da linguagem que você falou. As palavras-chave detectadas então são processadas em um formato que o celular ou o assistente possa entender e agir.

![compreensão](../images/comprehension.png)
> Compreensão de linguagem de verdade é difícil! Imagem por [Jen Looper](https://twitter.com/jenlooper)
> Tradução:
   > Mulher: Mas o que você quer? Frango? Peixe? Patê?
   > Gato: Miau

### Como essa tecnologia é possível?

Ela é possível porque alguém escreveu um programa de computador para fazer isto. Algumas décadas atrás, escritores de ficção previram que as pessoas iriam falar principalmente com seus computadores, e que computadores sempre iriam entender exatamente o que eles queriam dizer. Infelizmente, isto mostrou-se mais difícil do que muitos imaginavam, e enquanto hoje é um problema muito melhor compreendido, existem desafios significantes em alcançar o processamento de linguagem natural 'perfeito' quando pensamos em entender o significado de uma frase/expressão. Este é um problema particularmente difícil quando é preciso entender humor ou detectar emoções como sarcasmo em uma frase.

Neste momento, você pode estar se lembrando das aulas da escola onde o professor fala sobre a gramática de uma oração. Em alguns países, estudantes aprendem gramática e linguística em uma matéria dedicada, mas, em muitos, estes tópicos são incluídos como parte do aprendizado da linguagem: ou sua primeira linguagem na pré-escola (aprendendo a ler e escrever) e talvez a segunda linguagem no ensino fundamental ou médio. Não se preocupe se você não é experiente em diferenciar substantivos de verbos ou advérbios de adjetivos!

Se você tem dificuldade com a diferença entre o *presente do indicativo* e o *gerúndio*, você não está sozinho(a). Esta é uma tarefa desafiadora para muitas pessoas, mesmo falantes nativos de uma língua. A boa notícia é que computadores são muito bons em aplicar regras formais, e você vai aprender a escrever código que pode *parsear* (analisar a sintaxe) uma frase tão bem quanto um humano. A maior dificuldade que você irá enfrentar mais tarde é entender o *significado* e o *sentimento* de uma frase.

## Pré-requisitos

Para esta aula. o pré-requisito principal é conseguir ler e entender a linguagem desta lição. Não existem equações ou prblemas da matemática para resolver. Enquanto o autor original escreveu essa aula em inglês. ela também será traduzida em outras línguas (como em português!), então você pode estar lendo uma tradução. Existem exemplos onde um número de diferentes linguagens são usadas (como comparar regras gramaticais entre diferentes linguagens). Elas *não* são traduzidas, mas o texto que as explica sim, então o significado fica claro.

Para as tarefas de programação, você irá usar Python. Os exemplos a seguir usam Python 3.8.

Nesta seção, você vai usar:

- **Entendimento de Python 3**.  Entendimento da linguagem Python 3, esta aula usa input (entrada), loops (iteração), leitura de arquivos, arrays (vetores).
- **Visual Studio Code + extenção**. Nós iremos utilizar o Visual Studio Code e sua extensão de Python. Você também pode usar a IDE Python de sua preferência.
- **TextBlob**. [TextBlob](https://github.com/sloria/TextBlob) é uma biblioteca de processamento de texto simplificada para Python. Siga as instruções no site do  TextBlob para instalá-lo no seu sistema (instale o corpora também, como mostrado abaixo):

   ```bash
   pip install -U textblob
   python -m textblob.download_corpora
   ```

> 💡 Dica: Você pode rodar Python diretamente nos ambientes (environments) do VS Code. Veja a [documentação](https://code.visualstudio.com/docs/languages/python?WT.mc_id=academic-15963-cxa) para mais informações.

## Falando com máquinas

A história de tentar fazer computadores entender a linguagem humana é de décadas atrás, e um dos primeiros cientistas a considerar o processamento de linguagem natural foi *Alan Turing*.

### O 'Teste de Turing'

Quando Turing estava pesquisando *inteligência artificial* na década de 1950, ele imaginou um caso onde um teste de conversação que poderia ser dado para um humano e um computador (correspondentemente digitado), onde o humano na conversa não conseguiria ter certeza de que ele estava falando com outro humano ou um computador.

Se, depois de um certo período de conversa, o humano não pudesse determinar se as respostas foram dadas por um computador ou não, então poderíamos dizer que computador está *pensando*?

### A inspiração - 'o jogo da imitação'

A ideia disso veio de um jogo de festa chamado *O Jogo da Imitação* (The Imitation Game) onde um interrogador está sozinho em um cômodo e tem a tarefa de determinar qual de duas pessoas (em outro cômodo) é homem e mulher respectivamente. O interrogador pode mandar notas, e precisa tentar pensar em questões onde as respostas escritas revelam o gênero da pessoa misteriosa. Obviamente, os jogadores na outra sala estão tentando enganar o interrogador ao responder questões de forma confusa/enganosa, ao mesmo tempo em que aparentam ser respostas sinceras.

### Developing Eliza

In the 1960's an MIT scientist called *Joseph Weizenbaum* developed [*Eliza*](https://wikipedia.org/wiki/ELIZA), a computer 'therapist' that would ask the human questions and give the appearance of understanding their answers. However, while Eliza could parse a sentence and identify certain grammatical constructs and keywords so as to give a reasonable answer, it could not be said to *understand* the sentence. If Eliza was presented with a sentence following the format "**I am** <u>sad</u>" it might rearrange and substitute words in the sentence to form the response "How long have **you been** <u>sad</u>". 

This gave the impression that Eliza understood the statement and was asking a follow-on question, whereas in reality, it was changing the tense and adding some words. If Eliza could not identify a keyword that it had a response for, it would instead give a random response that could be applicable to many different statements. Eliza could be easily tricked, for instance if a user wrote "**You are** a <u>bicycle</u>" it might respond with "How long have **I been** a <u>bicycle</u>?", instead of a more reasoned response.

[![Chatting with Eliza](https://img.youtube.com/vi/RMK9AphfLco/0.jpg)](https://youtu.be/RMK9AphfLco "Chatting with Eliza")

> 🎥 Click the image above for a video about original ELIZA program

> Note: You can read the original description of [Eliza](https://cacm.acm.org/magazines/1966/1/13317-elizaa-computer-program-for-the-study-of-natural-language-communication-between-man-and-machine/abstract) published in 1966 if you have an ACM account. Alternately, read about Eliza on [wikipedia](https://wikipedia.org/wiki/ELIZA)

## Exercise - coding a basic conversational bot

A conversational bot, like Eliza, is a program that elicits user input and seems to understand and respond intelligently. Unlike Eliza, our bot will not have several rules giving it the appearance of having an intelligent conversation. Instead, our bot will have one ability only, to keep the conversation going with random responses that might work in almost any trivial conversation.

### The plan

Your steps when building a conversational bot:

1. Print instructions advising the user how to interact with the bot
2. Start a loop
   1. Accept user input
   2. If user has asked to exit, then exit
   3. Process user input and determine response (in this case, the response is a random choice from a list of possible generic responses)
   4. Print response
3. loop back to step 2

### Building the bot

Let's create the bot next. We'll start by defining some phrases.

1. Create this bot yourself in Python with the following random responses:

    ```python
    random_responses = ["That is quite interesting, please tell me more.",
                        "I see. Do go on.",
                        "Why do you say that?",
                        "Funny weather we've been having, isn't it?",
                        "Let's change the subject.",
                        "Did you catch the game last night?"]
    ```

    Here is some sample output to guide you (user input is on the lines starting with `>`):

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

    One possible solution to the task is [here](solution/bot.py)

    ✅ Stop and consider

    1. Do you think the random responses would 'trick' someone into thinking that the bot actually understood them?
    2. What features would the bot need to be more effective?
    3. If a bot could really 'understand' the meaning of a sentence, would it need to 'remember' the meaning of previous sentences in a conversation too?

---

## 🚀Challenge

Choose one of the "stop and consider" elements above and either try to implement them in code or write a solution on paper using pseudocode.

In the next lesson, you'll learn about a number of other approaches to parsing natural language and machine learning.

## [Post-lecture quiz](https://white-water-09ec41f0f.azurestaticapps.net/quiz/32/)

## Review & Self Study

Take a look at the references below as further reading opportunities.

### References

1. Schubert, Lenhart, "Computational Linguistics", *The Stanford Encyclopedia of Philosophy* (Spring 2020 Edition), Edward N. Zalta (ed.), URL = <https://plato.stanford.edu/archives/spr2020/entries/computational-linguistics/>.
2. Princeton University "About WordNet." [WordNet](https://wordnet.princeton.edu/). Princeton University. 2010. 

## Assignment 

[Search for a bot](assignment.md)
