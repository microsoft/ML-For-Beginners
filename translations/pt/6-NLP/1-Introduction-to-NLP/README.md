<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1c2ec40cf55c98a028a359c27ef7e45a",
  "translation_date": "2025-09-05T08:52:17+00:00",
  "source_file": "6-NLP/1-Introduction-to-NLP/README.md",
  "language_code": "pt"
}
-->
# Introdução ao processamento de linguagem natural

Esta lição aborda uma breve história e conceitos importantes do *processamento de linguagem natural* (PLN), um subcampo da *linguística computacional*.

## [Questionário pré-aula](https://ff-quizzes.netlify.app/en/ml/)

## Introdução

O PLN, como é comumente conhecido, é uma das áreas mais conhecidas onde o aprendizado de máquina foi aplicado e utilizado em softwares de produção.

✅ Consegue pensar em algum software que utiliza diariamente e que provavelmente tem algum PLN integrado? E os seus programas de processamento de texto ou as aplicações móveis que usa regularmente?

Você aprenderá sobre:

- **A ideia de línguas**. Como as línguas se desenvolveram e quais foram as principais áreas de estudo.
- **Definição e conceitos**. Também aprenderá definições e conceitos sobre como os computadores processam texto, incluindo análise sintática, gramática e identificação de substantivos e verbos. Há algumas tarefas de codificação nesta lição, e vários conceitos importantes são introduzidos, que você aprenderá a programar nas próximas lições.

## Linguística computacional

A linguística computacional é uma área de pesquisa e desenvolvimento que, ao longo de muitas décadas, estuda como os computadores podem trabalhar com línguas, compreendê-las, traduzi-las e até mesmo se comunicar com elas. O processamento de linguagem natural (PLN) é um campo relacionado, focado em como os computadores podem processar línguas "naturais", ou seja, humanas.

### Exemplo - ditado no telemóvel

Se alguma vez ditou algo para o seu telemóvel em vez de escrever ou fez uma pergunta a um assistente virtual, a sua fala foi convertida em texto e depois processada ou *analisada* a partir da língua que falou. As palavras-chave detectadas foram então processadas num formato que o telemóvel ou assistente pudesse compreender e agir.

![compreensão](../../../../6-NLP/1-Introduction-to-NLP/images/comprehension.png)
> Compreensão linguística real é difícil! Imagem por [Jen Looper](https://twitter.com/jenlooper)

### Como é que esta tecnologia é possível?

Isto é possível porque alguém escreveu um programa de computador para o fazer. Há algumas décadas, alguns escritores de ficção científica previram que as pessoas falariam principalmente com os seus computadores, e que os computadores entenderiam sempre exatamente o que elas queriam dizer. Infelizmente, revelou-se um problema mais difícil do que muitos imaginavam e, embora hoje seja um problema muito mais bem compreendido, ainda existem desafios significativos para alcançar um processamento de linguagem natural "perfeito" no que diz respeito a compreender o significado de uma frase. Este é um problema particularmente difícil quando se trata de entender humor ou detectar emoções como sarcasmo numa frase.

Neste momento, talvez se lembre das aulas escolares em que o professor abordava as partes da gramática numa frase. Em alguns países, os alunos aprendem gramática e linguística como uma disciplina dedicada, mas em muitos, esses tópicos estão incluídos como parte do aprendizado de uma língua: seja a sua primeira língua na escola primária (aprendendo a ler e escrever) ou talvez uma segunda língua no ensino secundário. Não se preocupe se não for um especialista em diferenciar substantivos de verbos ou advérbios de adjetivos!

Se tem dificuldade em distinguir entre o *presente simples* e o *presente contínuo*, não está sozinho. Isso é um desafio para muitas pessoas, mesmo falantes nativos de uma língua. A boa notícia é que os computadores são muito bons em aplicar regras formais, e você aprenderá a escrever código que pode *analisar* uma frase tão bem quanto um humano. O maior desafio que examinará mais tarde é compreender o *significado* e o *sentimento* de uma frase.

## Pré-requisitos

Para esta lição, o principal pré-requisito é ser capaz de ler e compreender a língua desta lição. Não há problemas matemáticos ou equações para resolver. Embora o autor original tenha escrito esta lição em inglês, ela também foi traduzida para outras línguas, então pode estar a ler uma tradução. Há exemplos onde são usados vários idiomas diferentes (para comparar as diferentes regras gramaticais de diferentes línguas). Estes *não* são traduzidos, mas o texto explicativo é, então o significado deve ser claro.

Para as tarefas de codificação, usará Python e os exemplos utilizam Python 3.8.

Nesta seção, precisará e usará:

- **Compreensão de Python 3**. Compreensão da linguagem de programação Python 3, esta lição utiliza entrada, loops, leitura de ficheiros e arrays.
- **Visual Studio Code + extensão**. Usaremos o Visual Studio Code e sua extensão para Python. Também pode usar um IDE de Python à sua escolha.
- **TextBlob**. [TextBlob](https://github.com/sloria/TextBlob) é uma biblioteca simplificada de processamento de texto para Python. Siga as instruções no site do TextBlob para instalá-lo no seu sistema (instale também os corpora, conforme mostrado abaixo):

   ```bash
   pip install -U textblob
   python -m textblob.download_corpora
   ```

> 💡 Dica: Pode executar Python diretamente em ambientes do VS Code. Consulte a [documentação](https://code.visualstudio.com/docs/languages/python?WT.mc_id=academic-77952-leestott) para mais informações.

## Conversando com máquinas

A história de tentar fazer os computadores entenderem a linguagem humana remonta a décadas, e um dos primeiros cientistas a considerar o processamento de linguagem natural foi *Alan Turing*.

### O 'teste de Turing'

Quando Turing estava a pesquisar *inteligência artificial* na década de 1950, ele considerou se um teste de conversação poderia ser dado a um humano e a um computador (via correspondência escrita) onde o humano na conversa não tivesse certeza se estava a conversar com outro humano ou com um computador.

Se, após um certo tempo de conversa, o humano não conseguisse determinar se as respostas vinham de um computador ou não, então poderia dizer-se que o computador estava *a pensar*?

### A inspiração - 'o jogo da imitação'

A ideia para isso veio de um jogo de festa chamado *O Jogo da Imitação*, onde um interrogador está sozinho numa sala e tem a tarefa de determinar quais das duas pessoas (noutra sala) são homem e mulher, respetivamente. O interrogador pode enviar notas e deve tentar pensar em perguntas cujas respostas escritas revelem o género da pessoa misteriosa. Claro, os jogadores na outra sala tentam enganar o interrogador, respondendo de forma a confundir ou induzir em erro, enquanto dão a aparência de responder honestamente.

### Desenvolvendo Eliza

Na década de 1960, um cientista do MIT chamado *Joseph Weizenbaum* desenvolveu [*Eliza*](https://wikipedia.org/wiki/ELIZA), uma "terapeuta" computorizada que fazia perguntas ao humano e dava a impressão de entender as suas respostas. No entanto, embora Eliza pudesse analisar uma frase e identificar certos construtos gramaticais e palavras-chave para dar uma resposta razoável, não se podia dizer que *entendia* a frase. Se Eliza recebesse uma frase no formato "**Eu estou** <u>triste</u>", poderia reorganizar e substituir palavras na frase para formar a resposta "Há quanto tempo **você está** <u>triste</u>". 

Isso dava a impressão de que Eliza entendia a declaração e estava a fazer uma pergunta de seguimento, enquanto na realidade estava apenas a mudar o tempo verbal e a adicionar algumas palavras. Se Eliza não conseguisse identificar uma palavra-chave para a qual tivesse uma resposta, daria uma resposta aleatória que poderia ser aplicável a muitas declarações diferentes. Eliza podia ser facilmente enganada, por exemplo, se um utilizador escrevesse "**Você é** uma <u>bicicleta</u>", ela poderia responder "Há quanto tempo **eu sou** uma <u>bicicleta</u>?", em vez de uma resposta mais razoável.

[![Conversando com Eliza](https://img.youtube.com/vi/RMK9AphfLco/0.jpg)](https://youtu.be/RMK9AphfLco "Conversando com Eliza")

> 🎥 Clique na imagem acima para um vídeo sobre o programa original ELIZA

> Nota: Pode ler a descrição original de [Eliza](https://cacm.acm.org/magazines/1966/1/13317-elizaa-computer-program-for-the-study-of-natural-language-communication-between-man-and-machine/abstract) publicada em 1966 se tiver uma conta ACM. Alternativamente, leia sobre Eliza na [wikipedia](https://wikipedia.org/wiki/ELIZA)

## Exercício - programar um bot de conversação básico

Um bot de conversação, como Eliza, é um programa que solicita a entrada do utilizador e parece entender e responder de forma inteligente. Ao contrário de Eliza, o nosso bot não terá várias regras que lhe dão a aparência de uma conversa inteligente. Em vez disso, o nosso bot terá apenas uma habilidade: manter a conversa com respostas aleatórias que possam funcionar em quase qualquer conversa trivial.

### O plano

Os seus passos ao construir um bot de conversação:

1. Imprimir instruções a aconselhar o utilizador sobre como interagir com o bot
2. Iniciar um loop
   1. Aceitar a entrada do utilizador
   2. Se o utilizador pedir para sair, então sair
   3. Processar a entrada do utilizador e determinar a resposta (neste caso, a resposta é uma escolha aleatória de uma lista de possíveis respostas genéricas)
   4. Imprimir a resposta
3. Voltar ao passo 2

### Construindo o bot

Vamos criar o bot a seguir. Começaremos por definir algumas frases.

1. Crie este bot em Python com as seguintes respostas aleatórias:

    ```python
    random_responses = ["That is quite interesting, please tell me more.",
                        "I see. Do go on.",
                        "Why do you say that?",
                        "Funny weather we've been having, isn't it?",
                        "Let's change the subject.",
                        "Did you catch the game last night?"]
    ```

    Aqui está um exemplo de saída para orientar (a entrada do utilizador está nas linhas que começam com `>`):

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

    1. Acha que as respostas aleatórias poderiam "enganar" alguém a pensar que o bot realmente o entende?
    2. Que recursos o bot precisaria para ser mais eficaz?
    3. Se um bot pudesse realmente "entender" o significado de uma frase, precisaria "lembrar-se" do significado das frases anteriores numa conversa também?

---

## 🚀Desafio

Escolha um dos elementos de "pare e reflita" acima e tente implementá-lo em código ou escreva uma solução no papel usando pseudocódigo.

Na próxima lição, aprenderá sobre várias outras abordagens para analisar linguagem natural e aprendizado de máquina.

## [Questionário pós-aula](https://ff-quizzes.netlify.app/en/ml/)

## Revisão e autoestudo

Consulte as referências abaixo como oportunidades de leitura adicional.

### Referências

1. Schubert, Lenhart, "Computational Linguistics", *The Stanford Encyclopedia of Philosophy* (Spring 2020 Edition), Edward N. Zalta (ed.), URL = <https://plato.stanford.edu/archives/spr2020/entries/computational-linguistics/>.
2. Princeton University "About WordNet." [WordNet](https://wordnet.princeton.edu/). Princeton University. 2010. 

## Tarefa 

[Procure um bot](assignment.md)

---

**Aviso Legal**:  
Este documento foi traduzido utilizando o serviço de tradução por IA [Co-op Translator](https://github.com/Azure/co-op-translator). Embora nos esforcemos para garantir a precisão, esteja ciente de que traduções automáticas podem conter erros ou imprecisões. O documento original no seu idioma nativo deve ser considerado a fonte oficial. Para informações críticas, recomenda-se uma tradução profissional realizada por humanos. Não nos responsabilizamos por quaisquer mal-entendidos ou interpretações incorretas resultantes do uso desta tradução.