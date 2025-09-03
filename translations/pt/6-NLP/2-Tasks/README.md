<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "6534e145d52a3890590d27be75386e5d",
  "translation_date": "2025-09-03T18:47:40+00:00",
  "source_file": "6-NLP/2-Tasks/README.md",
  "language_code": "pt"
}
-->
# Tarefas e técnicas comuns de processamento de linguagem natural

Para a maioria das tarefas de *processamento de linguagem natural*, o texto a ser processado deve ser dividido, examinado e os resultados armazenados ou cruzados com regras e conjuntos de dados. Essas tarefas permitem ao programador derivar o _significado_, a _intenção_ ou apenas a _frequência_ de termos e palavras em um texto.

## [Questionário pré-aula](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/33/)

Vamos explorar técnicas comuns usadas no processamento de texto. Combinadas com aprendizado de máquina, essas técnicas ajudam a analisar grandes volumes de texto de forma eficiente. Antes de aplicar ML a essas tarefas, no entanto, vamos entender os problemas enfrentados por um especialista em NLP.

## Tarefas comuns em NLP

Existem diferentes maneiras de analisar um texto com o qual você está trabalhando. Há tarefas que você pode realizar e, por meio delas, é possível compreender o texto e tirar conclusões. Normalmente, essas tarefas são realizadas em sequência.

### Tokenização

Provavelmente, a primeira coisa que a maioria dos algoritmos de NLP precisa fazer é dividir o texto em tokens ou palavras. Embora isso pareça simples, lidar com pontuação e delimitadores de palavras e frases em diferentes idiomas pode tornar o processo complicado. Pode ser necessário usar vários métodos para determinar as demarcações.

![tokenização](../../../../translated_images/tokenization.1641a160c66cd2d93d4524e8114e93158a9ce0eba3ecf117bae318e8a6ad3487.pt.png)
> Tokenizando uma frase de **Orgulho e Preconceito**. Infográfico por [Jen Looper](https://twitter.com/jenlooper)

### Embeddings

[Word embeddings](https://wikipedia.org/wiki/Word_embedding) são uma forma de converter seus dados textuais em valores numéricos. Os embeddings são feitos de maneira que palavras com significados semelhantes ou usadas juntas fiquem agrupadas.

![word embeddings](../../../../translated_images/embedding.2cf8953c4b3101d188c2f61a5de5b6f53caaa5ad4ed99236d42bc3b6bd6a1fe2.pt.png)
> "Tenho o maior respeito pelos seus nervos, eles são meus velhos amigos." - Word embeddings para uma frase de **Orgulho e Preconceito**. Infográfico por [Jen Looper](https://twitter.com/jenlooper)

✅ Experimente [esta ferramenta interessante](https://projector.tensorflow.org/) para explorar word embeddings. Ao clicar em uma palavra, aparecem clusters de palavras semelhantes: 'toy' agrupa-se com 'disney', 'lego', 'playstation' e 'console'.

### Parsing e Marcação de Partes do Discurso

Cada palavra que foi tokenizada pode ser marcada como uma parte do discurso - um substantivo, verbo ou adjetivo. A frase `the quick red fox jumped over the lazy brown dog` pode ser marcada como POS, por exemplo, fox = substantivo, jumped = verbo.

![parsing](../../../../translated_images/parse.d0c5bbe1106eae8fe7d60a183cd1736c8b6cec907f38000366535f84f3036101.pt.png)

> Parsing de uma frase de **Orgulho e Preconceito**. Infográfico por [Jen Looper](https://twitter.com/jenlooper)

Parsing é o reconhecimento de quais palavras estão relacionadas umas às outras em uma frase - por exemplo, `the quick red fox jumped` é uma sequência de adjetivo-substantivo-verbo que é separada da sequência `lazy brown dog`.

### Frequência de Palavras e Frases

Um procedimento útil ao analisar um grande corpo de texto é construir um dicionário de cada palavra ou frase de interesse e quantas vezes ela aparece. A frase `the quick red fox jumped over the lazy brown dog` tem uma frequência de palavras de 2 para "the".

Vamos analisar um texto de exemplo onde contamos a frequência das palavras. O poema "The Winners" de Rudyard Kipling contém o seguinte verso:

```output
What the moral? Who rides may read.
When the night is thick and the tracks are blind
A friend at a pinch is a friend, indeed,
But a fool to wait for the laggard behind.
Down to Gehenna or up to the Throne,
He travels the fastest who travels alone.
```

Como as frequências de frases podem ser sensíveis ou não a maiúsculas, a frase `a friend` tem uma frequência de 2, `the` tem uma frequência de 6 e `travels` tem uma frequência de 2.

### N-grams

Um texto pode ser dividido em sequências de palavras de um comprimento definido: uma única palavra (unigrama), duas palavras (bigramas), três palavras (trigramas) ou qualquer número de palavras (n-grams).

Por exemplo, `the quick red fox jumped over the lazy brown dog` com um valor de n-gram de 2 produz os seguintes n-grams:

1. the quick  
2. quick red  
3. red fox  
4. fox jumped  
5. jumped over  
6. over the  
7. the lazy  
8. lazy brown  
9. brown dog  

Pode ser mais fácil visualizar isso como uma janela deslizante sobre a frase. Aqui está para n-grams de 3 palavras, o n-gram está em negrito em cada frase:

1.   <u>**the quick red**</u> fox jumped over the lazy brown dog  
2.   the **<u>quick red fox</u>** jumped over the lazy brown dog  
3.   the quick **<u>red fox jumped</u>** over the lazy brown dog  
4.   the quick red **<u>fox jumped over</u>** the lazy brown dog  
5.   the quick red fox **<u>jumped over the</u>** lazy brown dog  
6.   the quick red fox jumped **<u>over the lazy</u>** brown dog  
7.   the quick red fox jumped over <u>**the lazy brown**</u> dog  
8.   the quick red fox jumped over the **<u>lazy brown dog</u>**  

![janela deslizante de n-grams](../../../../6-NLP/2-Tasks/images/n-grams.gif)

> Valor de n-gram de 3: Infográfico por [Jen Looper](https://twitter.com/jenlooper)

### Extração de Frases Nominais

Na maioria das frases, há um substantivo que é o sujeito ou objeto da frase. Em inglês, ele geralmente pode ser identificado por ter 'a', 'an' ou 'the' antes dele. Identificar o sujeito ou objeto de uma frase por meio da 'extração da frase nominal' é uma tarefa comum em NLP ao tentar entender o significado de uma frase.

✅ Na frase "I cannot fix on the hour, or the spot, or the look or the words, which laid the foundation. It is too long ago. I was in the middle before I knew that I had begun.", consegue identificar as frases nominais?

Na frase `the quick red fox jumped over the lazy brown dog` há 2 frases nominais: **quick red fox** e **lazy brown dog**.

### Análise de Sentimento

Uma frase ou texto pode ser analisado para determinar o sentimento, ou quão *positivo* ou *negativo* ele é. O sentimento é medido em *polaridade* e *objetividade/subjetividade*. A polaridade é medida de -1.0 a 1.0 (negativo a positivo) e de 0.0 a 1.0 (mais objetivo a mais subjetivo).

✅ Mais tarde, você aprenderá que existem diferentes maneiras de determinar o sentimento usando aprendizado de máquina, mas uma delas é ter uma lista de palavras e frases categorizadas como positivas ou negativas por um especialista humano e aplicar esse modelo ao texto para calcular um escore de polaridade. Consegue perceber como isso funcionaria em algumas circunstâncias e menos em outras?

### Flexão

A flexão permite que você pegue uma palavra e obtenha sua forma singular ou plural.

### Lematização

Um *lema* é a raiz ou palavra principal de um conjunto de palavras, por exemplo, *flew*, *flies*, *flying* têm como lema o verbo *fly*.

Existem também bancos de dados úteis disponíveis para pesquisadores de NLP, como:

### WordNet

[WordNet](https://wordnet.princeton.edu/) é um banco de dados de palavras, sinônimos, antônimos e muitos outros detalhes para cada palavra em vários idiomas. É incrivelmente útil ao tentar construir traduções, verificadores ortográficos ou ferramentas de linguagem de qualquer tipo.

## Bibliotecas de NLP

Felizmente, você não precisa construir todas essas técnicas sozinho, pois existem excelentes bibliotecas Python disponíveis que tornam o NLP muito mais acessível para desenvolvedores que não são especializados em processamento de linguagem natural ou aprendizado de máquina. As próximas lições incluem mais exemplos dessas bibliotecas, mas aqui você aprenderá alguns exemplos úteis para ajudá-lo na próxima tarefa.

### Exercício - usando a biblioteca `TextBlob`

Vamos usar uma biblioteca chamada TextBlob, que contém APIs úteis para lidar com esses tipos de tarefas. TextBlob "se baseia nos ombros gigantes do [NLTK](https://nltk.org) e [pattern](https://github.com/clips/pattern), e funciona bem com ambos." Ela possui uma quantidade considerável de ML embutida em sua API.

> Nota: Um [Guia de Início Rápido](https://textblob.readthedocs.io/en/dev/quickstart.html#quickstart) útil está disponível para TextBlob e é recomendado para desenvolvedores Python experientes.

Ao tentar identificar *frases nominais*, TextBlob oferece várias opções de extratores para encontrar frases nominais.

1. Veja o `ConllExtractor`.

    ```python
    from textblob import TextBlob
    from textblob.np_extractors import ConllExtractor
    # import and create a Conll extractor to use later 
    extractor = ConllExtractor()
    
    # later when you need a noun phrase extractor:
    user_input = input("> ")
    user_input_blob = TextBlob(user_input, np_extractor=extractor)  # note non-default extractor specified
    np = user_input_blob.noun_phrases                                    
    ```

    > O que está acontecendo aqui? [ConllExtractor](https://textblob.readthedocs.io/en/dev/api_reference.html?highlight=Conll#textblob.en.np_extractors.ConllExtractor) é "Um extrator de frases nominais que usa chunk parsing treinado com o corpus de treinamento ConLL-2000." ConLL-2000 refere-se à Conferência de Aprendizado Computacional de Linguagem Natural de 2000. Cada ano, a conferência hospedava um workshop para abordar um problema difícil de NLP, e em 2000 foi o chunking de frases nominais. Um modelo foi treinado no Wall Street Journal, com "seções 15-18 como dados de treinamento (211727 tokens) e seção 20 como dados de teste (47377 tokens)". Você pode conferir os procedimentos usados [aqui](https://www.clips.uantwerpen.be/conll2000/chunking/) e os [resultados](https://ifarm.nl/erikt/research/np-chunking.html).

### Desafio - melhorando seu bot com NLP

Na lição anterior, você construiu um bot de perguntas e respostas muito simples. Agora, você tornará Marvin um pouco mais simpático ao analisar sua entrada para determinar o sentimento e imprimir uma resposta que corresponda ao sentimento. Você também precisará identificar uma `noun_phrase` e perguntar sobre ela.

Seus passos ao construir um bot de conversação melhor:

1. Imprimir instruções orientando o usuário sobre como interagir com o bot  
2. Iniciar o loop  
   1. Aceitar entrada do usuário  
   2. Se o usuário pedir para sair, então sair  
   3. Processar a entrada do usuário e determinar uma resposta de sentimento apropriada  
   4. Se uma frase nominal for detectada no sentimento, pluralizá-la e pedir mais informações sobre esse tópico  
   5. Imprimir resposta  
3. Voltar ao passo 2  

Aqui está o trecho de código para determinar o sentimento usando TextBlob. Note que há apenas quatro *gradientes* de resposta de sentimento (você pode ter mais, se quiser):

```python
if user_input_blob.polarity <= -0.5:
  response = "Oh dear, that sounds bad. "
elif user_input_blob.polarity <= 0:
  response = "Hmm, that's not great. "
elif user_input_blob.polarity <= 0.5:
  response = "Well, that sounds positive. "
elif user_input_blob.polarity <= 1:
  response = "Wow, that sounds great. "
```

Aqui está um exemplo de saída para orientá-lo (entrada do usuário está nas linhas começando com >):

```output
Hello, I am Marvin, the friendly robot.
You can end this conversation at any time by typing 'bye'
After typing each answer, press 'enter'
How are you today?
> I am ok
Well, that sounds positive. Can you tell me more?
> I went for a walk and saw a lovely cat
Well, that sounds positive. Can you tell me more about lovely cats?
> cats are the best. But I also have a cool dog
Wow, that sounds great. Can you tell me more about cool dogs?
> I have an old hounddog but he is sick
Hmm, that's not great. Can you tell me more about old hounddogs?
> bye
It was nice talking to you, goodbye!
```

Uma possível solução para a tarefa está [aqui](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/2-Tasks/solution/bot.py)

✅ Verificação de Conhecimento

1. Você acha que as respostas simpáticas poderiam 'enganar' alguém a pensar que o bot realmente os entende?  
2. Identificar a frase nominal torna o bot mais 'crível'?  
3. Por que extrair uma 'frase nominal' de uma frase seria algo útil?  

---

Implemente o bot na verificação de conhecimento anterior e teste-o com um amigo. Ele consegue enganá-lo? Você consegue tornar seu bot mais 'crível'?

## 🚀Desafio

Escolha uma tarefa na verificação de conhecimento anterior e tente implementá-la. Teste o bot com um amigo. Ele consegue enganá-lo? Você consegue tornar seu bot mais 'crível'?

## [Questionário pós-aula](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/34/)

## Revisão e Autoestudo

Nas próximas lições, você aprenderá mais sobre análise de sentimento. Pesquise essa técnica interessante em artigos como estes no [KDNuggets](https://www.kdnuggets.com/tag/nlp)

## Tarefa 

[Faça um bot responder](assignment.md)

---

**Aviso Legal**:  
Este documento foi traduzido utilizando o serviço de tradução por IA [Co-op Translator](https://github.com/Azure/co-op-translator). Embora nos esforcemos para garantir a precisão, esteja ciente de que traduções automáticas podem conter erros ou imprecisões. O documento original na sua língua nativa deve ser considerado a fonte oficial. Para informações críticas, recomenda-se uma tradução profissional realizada por humanos. Não nos responsabilizamos por quaisquer mal-entendidos ou interpretações incorretas resultantes do uso desta tradução.