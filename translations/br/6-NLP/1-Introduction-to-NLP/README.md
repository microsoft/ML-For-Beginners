<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "89e923cf3e8bdff9662536e8bf9516e6",
  "translation_date": "2025-08-29T22:32:06+00:00",
  "source_file": "6-NLP/1-Introduction-to-NLP/README.md",
  "language_code": "br"
}
-->
# Introdução ao processamento de linguagem natural

Esta lição aborda uma breve história e conceitos importantes do *processamento de linguagem natural*, um subcampo da *linguística computacional*.

## [Quiz pré-aula](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/31/)

## Introdução

NLP, como é comumente conhecido, é uma das áreas mais conhecidas onde o aprendizado de máquina foi aplicado e utilizado em softwares de produção.

✅ Você consegue pensar em algum software que usa diariamente e que provavelmente tem algum NLP embutido? E os programas de processamento de texto ou aplicativos móveis que você utiliza regularmente?

Você aprenderá sobre:

- **A ideia de linguagens**. Como as linguagens se desenvolveram e quais foram as principais áreas de estudo.
- **Definição e conceitos**. Você também aprenderá definições e conceitos sobre como os computadores processam texto, incluindo análise sintática, gramática e identificação de substantivos e verbos. Há algumas tarefas de codificação nesta lição, e vários conceitos importantes são introduzidos, que você aprenderá a codificar nas próximas lições.

## Linguística computacional

Linguística computacional é uma área de pesquisa e desenvolvimento que, ao longo de muitas décadas, estuda como os computadores podem trabalhar com, e até mesmo entender, traduzir e se comunicar com linguagens. Processamento de linguagem natural (NLP) é um campo relacionado que foca em como os computadores podem processar linguagens 'naturais', ou humanas.

### Exemplo - ditado no telefone

Se você já ditou algo para o seu telefone em vez de digitar ou perguntou algo a um assistente virtual, sua fala foi convertida em texto e então processada ou *analisada* a partir da linguagem que você falou. As palavras-chave detectadas foram então processadas em um formato que o telefone ou assistente pudesse entender e agir.

![compreensão](../../../../translated_images/comprehension.619708fc5959b0f6a24ebffba2ad7b0625391a476141df65b43b59de24e45c6f.br.png)
> Compreensão linguística real é difícil! Imagem por [Jen Looper](https://twitter.com/jenlooper)

### Como essa tecnologia é possível?

Isso é possível porque alguém escreveu um programa de computador para fazer isso. Algumas décadas atrás, alguns escritores de ficção científica previram que as pessoas falariam principalmente com seus computadores, e os computadores sempre entenderiam exatamente o que elas queriam dizer. Infelizmente, isso se revelou um problema mais difícil do que muitos imaginavam, e embora seja um problema muito mais compreendido hoje, ainda há desafios significativos para alcançar um processamento de linguagem natural 'perfeito' quando se trata de entender o significado de uma frase. Este é um problema particularmente difícil quando se trata de entender humor ou detectar emoções como sarcasmo em uma frase.

Neste momento, você pode estar se lembrando das aulas escolares onde o professor abordava as partes da gramática em uma frase. Em alguns países, os alunos aprendem gramática e linguística como uma matéria dedicada, mas em muitos, esses tópicos são incluídos como parte do aprendizado de uma língua: seja sua primeira língua na escola primária (aprendendo a ler e escrever) e talvez uma segunda língua no ensino médio. Não se preocupe se você não é um especialista em diferenciar substantivos de verbos ou advérbios de adjetivos!

Se você tem dificuldade com a diferença entre o *presente simples* e o *presente contínuo*, você não está sozinho. Isso é algo desafiador para muitas pessoas, até mesmo falantes nativos de uma língua. A boa notícia é que os computadores são muito bons em aplicar regras formais, e você aprenderá a escrever código que pode *analisar* uma frase tão bem quanto um humano. O maior desafio que você examinará mais tarde é entender o *significado* e o *sentimento* de uma frase.

## Pré-requisitos

Para esta lição, o principal pré-requisito é ser capaz de ler e entender o idioma desta lição. Não há problemas matemáticos ou equações para resolver. Embora o autor original tenha escrito esta lição em inglês, ela também foi traduzida para outros idiomas, então você pode estar lendo uma tradução. Há exemplos onde vários idiomas diferentes são usados (para comparar as diferentes regras gramaticais de diferentes idiomas). Estes *não* são traduzidos, mas o texto explicativo é, então o significado deve ser claro.

Para as tarefas de codificação, você usará Python e os exemplos utilizam Python 3.8.

Nesta seção, você precisará e usará:

- **Compreensão de Python 3**. Compreensão da linguagem de programação Python 3, esta lição utiliza entrada, loops, leitura de arquivos, arrays.
- **Visual Studio Code + extensão**. Usaremos o Visual Studio Code e sua extensão Python. Você também pode usar um IDE de Python de sua escolha.
- **TextBlob**. [TextBlob](https://github.com/sloria/TextBlob) é uma biblioteca simplificada de processamento de texto para Python. Siga as instruções no site do TextBlob para instalá-lo em seu sistema (instale também os corpora, conforme mostrado abaixo):

   ```bash
   pip install -U textblob
   python -m textblob.download_corpora
   ```

> 💡 Dica: Você pode executar Python diretamente em ambientes do VS Code. Confira os [documentos](https://code.visualstudio.com/docs/languages/python?WT.mc_id=academic-77952-leestott) para mais informações.

## Conversando com máquinas

A história de tentar fazer os computadores entenderem a linguagem humana remonta a décadas, e um dos primeiros cientistas a considerar o processamento de linguagem natural foi *Alan Turing*.

### O 'teste de Turing'

Quando Turing estava pesquisando *inteligência artificial* na década de 1950, ele considerou se um teste de conversação poderia ser dado a um humano e a um computador (via correspondência escrita) onde o humano na conversa não tivesse certeza se estava conversando com outro humano ou com um computador.

Se, após um certo tempo de conversa, o humano não conseguisse determinar se as respostas vinham de um computador ou não, então o computador poderia ser considerado como *pensando*?

### A inspiração - 'o jogo da imitação'

A ideia para isso veio de um jogo de festa chamado *O Jogo da Imitação*, onde um interrogador está sozinho em uma sala e tem a tarefa de determinar quais das duas pessoas (em outra sala) são homem e mulher, respectivamente. O interrogador pode enviar notas e deve tentar pensar em perguntas cujas respostas escritas revelem o gênero da pessoa misteriosa. Claro, os jogadores na outra sala estão tentando enganar o interrogador respondendo às perguntas de forma a confundir ou enganar o interrogador, enquanto também dão a impressão de responder honestamente.

### Desenvolvendo Eliza

Na década de 1960, um cientista do MIT chamado *Joseph Weizenbaum* desenvolveu [*Eliza*](https://wikipedia.org/wiki/ELIZA), uma 'terapeuta' computadorizada que fazia perguntas ao humano e dava a impressão de entender suas respostas. No entanto, enquanto Eliza podia analisar uma frase e identificar certos construtos gramaticais e palavras-chave para dar uma resposta razoável, não se podia dizer que ela *entendia* a frase. Se Eliza fosse apresentada a uma frase no formato "**Eu estou** <u>triste</u>", ela poderia reorganizar e substituir palavras na frase para formar a resposta "Há quanto tempo **você está** <u>triste</u>". 

Isso dava a impressão de que Eliza entendia a declaração e estava fazendo uma pergunta de acompanhamento, enquanto na realidade, ela estava apenas mudando o tempo verbal e adicionando algumas palavras. Se Eliza não conseguisse identificar uma palavra-chave para a qual tivesse uma resposta, ela daria uma resposta aleatória que poderia ser aplicável a muitas declarações diferentes. Eliza podia ser facilmente enganada, por exemplo, se um usuário escrevesse "**Você é** uma <u>bicicleta</u>", ela poderia responder com "Há quanto tempo **eu sou** uma <u>bicicleta</u>?", em vez de uma resposta mais razoável.

[![Conversando com Eliza](https://img.youtube.com/vi/RMK9AphfLco/0.jpg)](https://youtu.be/RMK9AphfLco "Conversando com Eliza")

> 🎥 Clique na imagem acima para um vídeo sobre o programa original ELIZA

> Nota: Você pode ler a descrição original de [Eliza](https://cacm.acm.org/magazines/1966/1/13317-elizaa-computer-program-for-the-study-of-natural-language-communication-between-man-and-machine/abstract) publicada em 1966 se tiver uma conta ACM. Alternativamente, leia sobre Eliza na [wikipedia](https://wikipedia.org/wiki/ELIZA)

## Exercício - codificando um bot de conversação básico

Um bot de conversação, como Eliza, é um programa que solicita entrada do usuário e parece entender e responder de forma inteligente. Diferentemente de Eliza, nosso bot não terá várias regras que dão a aparência de uma conversa inteligente. Em vez disso, nosso bot terá apenas uma habilidade: manter a conversa com respostas aleatórias que podem funcionar em quase qualquer conversa trivial.

### O plano

Seus passos ao construir um bot de conversação:

1. Imprimir instruções aconselhando o usuário sobre como interagir com o bot
2. Iniciar um loop
   1. Aceitar entrada do usuário
   2. Se o usuário pedir para sair, então sair
   3. Processar a entrada do usuário e determinar a resposta (neste caso, a resposta é uma escolha aleatória de uma lista de possíveis respostas genéricas)
   4. Imprimir a resposta
3. Voltar ao passo 2

### Construindo o bot

Vamos criar o bot agora. Começaremos definindo algumas frases.

1. Crie este bot você mesmo em Python com as seguintes respostas aleatórias:

    ```python
    random_responses = ["That is quite interesting, please tell me more.",
                        "I see. Do go on.",
                        "Why do you say that?",
                        "Funny weather we've been having, isn't it?",
                        "Let's change the subject.",
                        "Did you catch the game last night?"]
    ```

    Aqui está um exemplo de saída para orientá-lo (entrada do usuário está nas linhas começando com `>`):

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

    ✅ Pare e reflita

    1. Você acha que as respostas aleatórias poderiam 'enganar' alguém a pensar que o bot realmente entendeu?
    2. Quais recursos o bot precisaria para ser mais eficaz?
    3. Se um bot pudesse realmente 'entender' o significado de uma frase, ele precisaria 'lembrar' o significado de frases anteriores em uma conversa também?

---

## 🚀Desafio

Escolha um dos elementos "pare e reflita" acima e tente implementá-lo em código ou escreva uma solução no papel usando pseudocódigo.

Na próxima lição, você aprenderá sobre várias outras abordagens para analisar linguagem natural e aprendizado de máquina.

## [Quiz pós-aula](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/32/)

## Revisão e Autoestudo

Confira as referências abaixo como oportunidades de leitura adicional.

### Referências

1. Schubert, Lenhart, "Computational Linguistics", *The Stanford Encyclopedia of Philosophy* (Spring 2020 Edition), Edward N. Zalta (ed.), URL = <https://plato.stanford.edu/archives/spr2020/entries/computational-linguistics/>.
2. Princeton University "About WordNet." [WordNet](https://wordnet.princeton.edu/). Princeton University. 2010. 

## Tarefa 

[Procure um bot](assignment.md)

---

**Aviso Legal**:  
Este documento foi traduzido utilizando o serviço de tradução por IA [Co-op Translator](https://github.com/Azure/co-op-translator). Embora nos esforcemos para garantir a precisão, esteja ciente de que traduções automatizadas podem conter erros ou imprecisões. O documento original em seu idioma nativo deve ser considerado a fonte autoritativa. Para informações críticas, recomenda-se a tradução profissional realizada por humanos. Não nos responsabilizamos por quaisquer mal-entendidos ou interpretações equivocadas decorrentes do uso desta tradução.