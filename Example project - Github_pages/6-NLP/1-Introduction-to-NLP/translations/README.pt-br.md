# Introdução ao Processamento de Linguagem Natural

Esta aula cobre uma breve história, bem como conceitos importantes do *processamento de linguagem natural*, uma subárea da *Linguística computacional*.

## [Teste pré-aula](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/31?loc=ptbr)

## Introdução

O Processamento de Linguagem Natural (PLN) ou, em inglês, Natural Language Processing (NLP), como é geralmente conhecido, é um dos campos mais conhecidos onde o aprendizado de máquina (machine learning) tem sido aplicado e usado na produção de software.

✅ Você consegue pensar em algum software que você usa todo dia e que provavelmente tem algum PLN integrado? E em programas de processamento de palavras ou aplicativos mobile que você usa com frequência?

Você vai aprender sobre:

- **A noção de linguagens**. Como as linguagens se desenvolveram e quais são as principais áreas de estudo.
- **Definição e conceitos**. Você também vai aprender definições e conceitos relacionados com o modo como os computadores processam texto, incluindo análise sintática (parsing), gramática e identificação de substantivos e verbos. Existem algumas tarefas de programação nesta aula, juntamente com a introdução de muitos conceitos importantes, os quais você irá aprender a programar nas próximas aulas.

## Linguística computacional

Linguística computacional é uma área de pesquisa e desenvolvimento que vem aumentando ao longo das décadas e estuda como computadores podem trabalhar e comunicar com linguagens, traduzir, e até entendê-las. O processamento de linguagem natural (PLN) é um campo relacionado à linguística computacional que foca em como computadores podem processar linguagens 'naturais' ou humanas.

### Exemplo - transcrição de voz no celular

Se você já usou o recurso de transcrição de voz ao invés de escrever ou fez uma pergunta para uma assistente virtual, sua fala foi convertida para o formato textual e então ela foi processada ou *parseada* (teve a sintaxe analisada). As palavras-chave detectadas então são processadas em um formato que o celular ou a assistente possa entender e agir.

![compreensão](../images/comprehension.png)
> Compreensão de linguagem de verdade é difícil! Imagem por [Jen Looper](https://twitter.com/jenlooper)  
  
> Tradução:  
   > Mulher: Mas o que você quer? Frango? Peixe? Patê?  
   > Gato: Miau  

### Como essa tecnologia é possível?

Ela é possível porque alguém escreveu um programa de computador para fazer isto. Algumas décadas atrás, escritores de ficção científica previram que as pessoas iriam falar majoritariamente com seus computadores, e que computadores sempre conseguiriam entender exatamente o que elas queriam dizer. Infelizmente, isto mostrou-se mais difícil do que muitos imaginavam, e apesar de hoje ser um problema muito melhor compreendido, ainda existem desafios significativos para alcançar o processamento de linguagem natural 'perfeito' no que tange a entender o significado de uma frase/oração. Este é um problema particularmente difícil quando é preciso entender humor ou detectar emoções como sarcasmo em uma frase.

Agora, você pode estar se lembrando das aulas da escola onde o professor fala sobre a gramática de uma oração. Em alguns países, estudantes aprendem gramática e linguística em uma matéria dedicada, mas, em muitos, estes tópicos são incluídos como parte do aprendizado da linguagem: ou sua primeira linguagem na pré-escola (aprendendo a ler e escrever) e talvez a segunda linguagem no ensino fundamental ou médio. Contudo, não se preocupe se você não é experiente em diferenciar substantivos de verbos ou advérbios de adjetivos!

Se você tem dificuldade com a diferença entre o *presente do indicativo* e o *gerúndio*, você não está sozinho(a). Esta é uma tarefa desafiadora para muitas pessoas, mesmo falantes nativos de uma língua. A boa notícia é que computadores são ótimos em aplicar regras formais, e você vai aprender a escrever código que pode *parsear* (analisar a sintaxe) uma frase tão bem quanto um humano. A maior dificuldade que você irá encontrar é entender o *significado* e o *sentimento* de uma frase.

## Pré-requisitos

Para esta aula, o pré-requisito principal é conseguir ler e entender a linguagem. Não existem equações ou problemas da matemática para resolver. Enquanto o autor original escreveu esta aula em inglês, ela também será traduzida em outras línguas (como em português!), então você pode estar lendo uma tradução. Existem exemplos onde diferentes linguagens são usadas (como comparar regras gramaticais entre diferentes linguagens). Elas *não* são traduzidas, mas o texto que as explica sim, então o significado fica claro.

Para as tarefas de programação, você irá usar Python. Os exemplos a seguir usam Python 3.8.

Nesta seção, você vai precisar:

- **Entender Python 3**.  Entender a linguagem Python 3, esta aula usa input (entrada), loops (iteração), leitura de arquivos, arrays (vetores).
- **Visual Studio Code + extensão**. Nós iremos utilizar o Visual Studio Code e sua extensão de Python, mas você pode usar a IDE Python de sua preferência.
- **TextBlob**. [TextBlob](https://github.com/sloria/TextBlob) é uma biblioteca de processamento de texto simplificada para Python. Siga as instruções no site do  TextBlob para instalá-lo no seu sistema (instale o corpora também, como mostrado abaixo):

   ```bash
   pip install -U textblob
   python -m textblob.download_corpora
   ```

> 💡 Dica: Você pode rodar Python diretamente nos ambientes (environments) do VS Code. Veja a [documentação](https://code.visualstudio.com/docs/languages/python?WT.mc_id=academic-77952-leestott) para mais informações.

## Falando com máquinas

A história de tentar fazer computadores entender a linguagem humana é de décadas atrás, e um dos primeiros cientistas a considerar o processamento de linguagem natural foi *Alan Turing*.

### O 'Teste de Turing'

Quando Turing estava pesquisando *inteligência artificial* na década de 1950, ele imaginou um caso onde um teste de conversação digitada que poderia ser dado para um humano e um computador, onde o humano na conversa não conseguiria ter certeza de que ele estava falando com outro humano ou um computador.

Se o humano não pudesse determinar se as respostas foram dadas por um computador ou não depois de um certo período de conversa, poderíamos dizer que computador está *pensando*?

### A inspiração - 'o jogo da imitação'

A ideia do teste veio de um jogo de festa chamado *O Jogo da Imitação* (The Imitation Game), onde um interrogador está sozinho em um cômodo e tem a tarefa de determinar qual de duas pessoas (em outro cômodo) é o homem e qual é a mulher. O interrogador pode mandar notas, e precisa tentar pensar em questões onde as respostas escritas revelam o gênero da pessoa misteriosa. Obviamente, os jogadores na outra sala estão tentando enganar o interrogador ao responder questões de forma confusa/enganosa, ao mesmo tempo em que aparentam dar respostas sinceras.

### Desenvolvendo Eliza

Nos anos 1960, um cientista do MIT chamado *Joseph Weizenbaum* desenvolveu [*Eliza*](https://wikipedia.org/wiki/ELIZA), um computador 'terapeuta' que fazia perguntas ao humano e aparentava entender suas respostas. No entanto, enquanto Eliza conseguia parsear e identificar certas construções gramaticais e palavras-chave para conseguir responder de forma razoável, não podemos dizer que ele conseguia *entender* a frase. Se Eliza fosse apresentado com uma sequência de sentenças seguindo o formato "**Eu estou** <u>triste</u>" ele podia rearranjar e substituir palavras na sentença para formar a resposta "Há quanto tempo **você está** <u>triste</u>?". 

Isso dá a impressão de que Eliza entendeu a afirmação e fez uma pergunta subsequente, enquanto na realidade, o computador mudou a conjugação verbal e adicionou algumas palavras. Se Eliza não conseguisse identificar uma palavra-chave que já tem uma resposta pronta, ele daria uma resposta aleatória que pode ser aplicada em diversas afirmações do usuário. Eliza podia ser facilmente enganado, por exemplo, quando um usuário escrevia "**Você é** uma <u>bicicleta</u>", a resposta dada poderia ser "Há quanto tempo **eu sou** uma <u>bicicleta</u>?", ao invés de uma resposta mais razoável.

[![Conversando com Eliza](https://img.youtube.com/vi/RMK9AphfLco/0.jpg)](https://youtu.be/RMK9AphfLco "Chatting with Eliza")

> 🎥 Clique na imagem abaixo para ver um video sobre o programa original ELIZA

> Nota: Você pode ler a descrição original de [Eliza](https://cacm.acm.org/magazines/1966/1/13317-elizaa-computer-program-for-the-study-of-natural-language-communication-between-man-and-machine/abstract) publicada em 1966 se você tem uma conta ACM. Alternativamente, leia sobre Eliza na [wikipedia](https://wikipedia.org/wiki/ELIZA)

## Exercício - programando um bot de conversação básico

Um bot de conversação, como Eliza, é um programa que obtém o input do usuário e parece entender e responder de forma inteligente. Diferentemente de Eliza, nosso bot não vai ter diversas regras dando a aparência de uma conversação inteligente. Ao invés disso, nosso bot tem uma única habilidade, a de continuar a conversação com respostas aleatórias que podem funcionar em qualquer conversação trivial.

### O plano

Seus passos quando estiver construindo um bot de conversação:

1. Imprima instruções indicando como o usuário pode interagir com o bot
2. Comece um loop (laço)
   1. Aceite o input do usuário
   2. Se o usuário pedir para sair, então sair
   3. Processar o input do usuário e determinar resposta (neste caso, a resposta é uma escolha aleatória de uma lista de possíveis respostas genéricas)
   4. Imprima a resposta
3. Voltar para o passo 2 (continuando o loop/laço)

### Construindo o bot

Agora, vamos criar o bot. Iremos começar definindo algumas frases.

> Nota da tradutora: em função da política de contribuição da Microsoft, todos os códigos foram mantidos em inglês. No entanto, é possível encontrar traduções abaixo deles para ajudar no entendimento. Para não estender muito o arquivo, somente algumas partes foram traduzidas, então sintam-se convidados a pesquisar em tradutores/dicionários.

1. Crie este bot em Python com as seguintes respostas:

    ```python
    random_responses = ["That is quite interesting, please tell me more.",
                        "I see. Do go on.",
                        "Why do you say that?",
                        "Funny weather we've been having, isn't it?",
                        "Let's change the subject.",
                        "Did you catch the game last night?"]
    ```
   > A lista de respostas genéricas inclui frases como "Isso é bem interessante, por favor me conte mais."  e "O tempo esses dias está bem doido, né?"

    Aqui estão alguns outputs de exemplo para te guiar (as entradas do usuário se iniciam com `>`):

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
   > O bot se apresenta e dá instruções de como o usuário deve interagir. A conversa é iniciada pelo bot, que pergunta "Como você está hoje?". O usuário diz "Estou bem, valeu", ao que o bot responde "Isso é bem interessante, por favor me conte mais.". A conversa continua por mais alguns diálogos.

    Uma solução possível para a tarefa está [aqui](../solution/bot.py)

    ✅ Pare e pense

    1. Você acha que respostas aleatórias seriam capazes de fazer uma pessoa achar que o bot realmente entendeu que ela disse?
    2. Quais recursos/funções o bot precisaria ter para ser mais convincente?
    3. Se um bot pudesse 'entender' facilmente o significado de uma frase, ele também precisaria se 'lembrar' do significado de frases anteriores?

---

## 🚀Desafio

Escolha um dos elementos do "pare e considere" acima e tente implementá-lo em código ou escreva uma solução no papel usando pseudocódigo.

Na próxima aula, você irá aprender sobre algumas outras abordagens de análise sintática de linguagem natural e de aprendizado de máquina.

## [Teste pós-aula](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/32?loc=ptbr)

## Revisão & Autoestudo

Dê uma olhada nas referências abaixo e talvez até as considere como oportunidade de leitura futura.

### Referências

1. Schubert, Lenhart, "Computational Linguistics", *The Stanford Encyclopedia of Philosophy* (Spring 2020 Edition), Edward N. Zalta (ed.), URL = <https://plato.stanford.edu/archives/spr2020/entries/computational-linguistics/>.
2. Princeton University "About WordNet." [WordNet](https://wordnet.princeton.edu/). Princeton University. 2010. 

## Tarefa 

[Procure por um bot](assignment.pt-br.md)
