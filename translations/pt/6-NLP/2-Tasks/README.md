<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "5f3cb462e3122e1afe7ab0050ccf2bd3",
  "translation_date": "2025-09-05T08:50:27+00:00",
  "source_file": "6-NLP/2-Tasks/README.md",
  "language_code": "pt"
}
-->
# Tarefas e técnicas comuns de processamento de linguagem natural

Para a maioria das tarefas de *processamento de linguagem natural*, o texto a ser processado deve ser dividido, examinado e os resultados armazenados ou cruzados com regras e conjuntos de dados. Essas tarefas permitem ao programador derivar o _significado_, a _intenção_ ou apenas a _frequência_ de termos e palavras em um texto.

## [Questionário pré-aula](https://ff-quizzes.netlify.app/en/ml/)

Vamos explorar técnicas comuns usadas no processamento de texto. Combinadas com aprendizagem automática, essas técnicas ajudam a analisar grandes volumes de texto de forma eficiente. Antes de aplicar ML a essas tarefas, no entanto, vamos entender os problemas enfrentados por um especialista em NLP.

## Tarefas comuns em NLP

Existem diferentes maneiras de analisar um texto com o qual você está trabalhando. Há tarefas que você pode realizar e, através delas, é possível compreender o texto e tirar conclusões. Normalmente, essas tarefas são realizadas em sequência.

### Tokenização

Provavelmente, a primeira coisa que a maioria dos algoritmos de NLP precisa fazer é dividir o texto em tokens ou palavras. Embora isso pareça simples, lidar com pontuação e delimitadores de palavras e frases em diferentes idiomas pode tornar o processo complicado. Pode ser necessário usar vários métodos para determinar as demarcações.

![tokenização](../../../../6-NLP/2-Tasks/images/tokenization.png)
> Tokenizando uma frase de **Orgulho e Preconceito**. Infográfico por [Jen Looper](https://twitter.com/jenlooper)

### Embeddings

[Word embeddings](https://wikipedia.org/wiki/Word_embedding) são uma forma de converter seus dados textuais em valores numéricos. Os embeddings são feitos de maneira que palavras com significados semelhantes ou usadas juntas fiquem agrupadas.

![word embeddings](../../../../6-NLP/2-Tasks/images/embedding.png)
> "Tenho o maior respeito pelos seus nervos, eles são meus velhos amigos." - Word embeddings para uma frase de **Orgulho e Preconceito**. Infográfico por [Jen Looper](https://twitter.com/jenlooper)

✅ Experimente [esta ferramenta interessante](https://projector.tensorflow.org/) para explorar word embeddings. Ao clicar em uma palavra, aparecem clusters de palavras semelhantes: 'brinquedo' agrupa-se com 'disney', 'lego', 'playstation' e 'console'.

### Parsing & Marcação de Partes do Discurso

Cada palavra que foi tokenizada pode ser marcada como uma parte do discurso - substantivo, verbo ou adjetivo. A frase `a rápida raposa vermelha saltou sobre o cão castanho preguiçoso` pode ser marcada como POS, por exemplo, raposa = substantivo, saltou = verbo.

![parsing](../../../../6-NLP/2-Tasks/images/parse.png)

> Parsing de uma frase de **Orgulho e Preconceito**. Infográfico por [Jen Looper](https://twitter.com/jenlooper)

Parsing é o reconhecimento de quais palavras estão relacionadas umas às outras em uma frase - por exemplo, `a rápida raposa vermelha saltou` é uma sequência de adjetivo-substantivo-verbo que é separada da sequência `cão castanho preguiçoso`.

### Frequência de Palavras e Frases

Um procedimento útil ao analisar um grande corpo de texto é construir um dicionário de cada palavra ou frase de interesse e quantas vezes ela aparece. A frase `a rápida raposa vermelha saltou sobre o cão castanho preguiçoso` tem uma frequência de palavras de 2 para "a".

Vamos analisar um texto de exemplo onde contamos a frequência de palavras. O poema The Winners de Rudyard Kipling contém o seguinte verso:

```output
What the moral? Who rides may read.
When the night is thick and the tracks are blind
A friend at a pinch is a friend, indeed,
But a fool to wait for the laggard behind.
Down to Gehenna or up to the Throne,
He travels the fastest who travels alone.
```

Como as frequências de frases podem ser sensíveis ou não a maiúsculas, a frase `um amigo` tem uma frequência de 2, `o` tem uma frequência de 6 e `viaja` tem uma frequência de 2.

### N-grams

Um texto pode ser dividido em sequências de palavras de um comprimento definido, uma única palavra (unigrama), duas palavras (bigramas), três palavras (trigramas) ou qualquer número de palavras (n-grams).

Por exemplo, `a rápida raposa vermelha saltou sobre o cão castanho preguiçoso` com um valor de n-gram de 2 produz os seguintes n-grams:

1. a rápida  
2. rápida raposa  
3. raposa vermelha  
4. vermelha saltou  
5. saltou sobre  
6. sobre o  
7. o cão  
8. cão castanho  
9. castanho preguiçoso  

Pode ser mais fácil visualizar isso como uma janela deslizante sobre a frase. Aqui está para n-grams de 3 palavras, o n-gram está em negrito em cada frase:

1.   <u>**a rápida raposa**</u> vermelha saltou sobre o cão castanho preguiçoso  
2.   a **<u>rápida raposa vermelha</u>** saltou sobre o cão castanho preguiçoso  
3.   a rápida **<u>raposa vermelha saltou</u>** sobre o cão castanho preguiçoso  
4.   a rápida raposa **<u>vermelha saltou sobre</u>** o cão castanho preguiçoso  
5.   a rápida raposa vermelha **<u>saltou sobre o</u>** cão castanho preguiçoso  
6.   a rápida raposa vermelha saltou **<u>sobre o cão</u>** castanho preguiçoso  
7.   a rápida raposa vermelha saltou sobre <u>**o cão castanho**</u> preguiçoso  
8.   a rápida raposa vermelha saltou sobre o **<u>cão castanho preguiçoso</u>**

![janela deslizante de n-grams](../../../../6-NLP/2-Tasks/images/n-grams.gif)

> Valor de n-gram de 3: Infográfico por [Jen Looper](https://twitter.com/jenlooper)

### Extração de Frases Nominais

Na maioria das frases, há um substantivo que é o sujeito ou objeto da frase. Em inglês, muitas vezes é identificável por ter 'a', 'an' ou 'the' antes dele. Identificar o sujeito ou objeto de uma frase através da 'extração da frase nominal' é uma tarefa comum em NLP ao tentar entender o significado de uma frase.

✅ Na frase "Não consigo fixar a hora, ou o local, ou o olhar ou as palavras, que lançaram a base. Faz muito tempo. Eu estava no meio antes de perceber que tinha começado.", consegue identificar as frases nominais?

Na frase `a rápida raposa vermelha saltou sobre o cão castanho preguiçoso` há 2 frases nominais: **rápida raposa vermelha** e **cão castanho preguiçoso**.

### Análise de Sentimento

Uma frase ou texto pode ser analisado para determinar o sentimento, ou quão *positivo* ou *negativo* ele é. O sentimento é medido em *polaridade* e *objetividade/subjetividade*. A polaridade é medida de -1.0 a 1.0 (negativo a positivo) e de 0.0 a 1.0 (mais objetivo a mais subjetivo).

✅ Mais tarde, aprenderá que existem diferentes maneiras de determinar o sentimento usando aprendizagem automática, mas uma delas é ter uma lista de palavras e frases categorizadas como positivas ou negativas por um especialista humano e aplicar esse modelo ao texto para calcular um score de polaridade. Consegue perceber como isso funcionaria em algumas circunstâncias e menos em outras?

### Flexão

A flexão permite que você pegue uma palavra e obtenha o singular ou plural dela.

### Lematização

Um *lema* é a raiz ou palavra principal de um conjunto de palavras, por exemplo, *voou*, *voa*, *voando* têm como lema o verbo *voar*.

Existem também bases de dados úteis disponíveis para o pesquisador de NLP, como:

### WordNet

[WordNet](https://wordnet.princeton.edu/) é uma base de dados de palavras, sinônimos, antônimos e muitos outros detalhes para cada palavra em vários idiomas. É incrivelmente útil ao tentar construir traduções, verificadores ortográficos ou ferramentas de linguagem de qualquer tipo.

## Bibliotecas de NLP

Felizmente, você não precisa construir todas essas técnicas sozinho, pois existem excelentes bibliotecas Python disponíveis que tornam o NLP muito mais acessível para desenvolvedores que não são especializados em processamento de linguagem natural ou aprendizagem automática. As próximas lições incluem mais exemplos dessas bibliotecas, mas aqui aprenderá alguns exemplos úteis para ajudá-lo na próxima tarefa.

### Exercício - usando a biblioteca `TextBlob`

Vamos usar uma biblioteca chamada TextBlob, pois ela contém APIs úteis para lidar com esses tipos de tarefas. TextBlob "baseia-se nos ombros gigantes do [NLTK](https://nltk.org) e [pattern](https://github.com/clips/pattern), e funciona bem com ambos." Ela possui uma quantidade considerável de ML embutida em sua API.

> Nota: Um [Guia de Introdução](https://textblob.readthedocs.io/en/dev/quickstart.html#quickstart) útil está disponível para TextBlob e é recomendado para desenvolvedores Python experientes.

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

    > O que está acontecendo aqui? [ConllExtractor](https://textblob.readthedocs.io/en/dev/api_reference.html?highlight=Conll#textblob.en.np_extractors.ConllExtractor) é "Um extrator de frases nominais que usa chunk parsing treinado com o corpus de treinamento ConLL-2000." ConLL-2000 refere-se à Conferência de Aprendizagem Computacional de Linguagem Natural de 2000. Cada ano a conferência hospedava um workshop para resolver um problema difícil de NLP, e em 2000 foi chunking de frases nominais. Um modelo foi treinado no Wall Street Journal, com "as seções 15-18 como dados de treinamento (211727 tokens) e a seção 20 como dados de teste (47377 tokens)". Pode consultar os procedimentos usados [aqui](https://www.clips.uantwerpen.be/conll2000/chunking/) e os [resultados](https://ifarm.nl/erikt/research/np-chunking.html).

### Desafio - melhorando seu bot com NLP

Na lição anterior, você construiu um bot de perguntas e respostas muito simples. Agora, tornará Marvin um pouco mais simpático ao analisar sua entrada para sentimento e imprimir uma resposta que corresponda ao sentimento. Também precisará identificar uma `noun_phrase` e perguntar sobre ela.

Os passos para construir um bot conversacional melhor:

1. Imprimir instruções aconselhando o utilizador sobre como interagir com o bot  
2. Iniciar loop  
   1. Aceitar entrada do utilizador  
   2. Se o utilizador pedir para sair, então sair  
   3. Processar a entrada do utilizador e determinar a resposta de sentimento apropriada  
   4. Se uma frase nominal for detectada no sentimento, pluralizá-la e pedir mais informações sobre esse tópico  
   5. Imprimir resposta  
3. Voltar ao passo 2  

Aqui está o trecho de código para determinar o sentimento usando TextBlob. Note que há apenas quatro *gradientes* de resposta de sentimento (poderia haver mais, se desejar):

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

Aqui está um exemplo de saída para orientá-lo (entrada do utilizador está nas linhas que começam com >):

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

1. Acha que as respostas simpáticas poderiam 'enganar' alguém a pensar que o bot realmente os compreendeu?  
2. Identificar a frase nominal torna o bot mais 'crível'?  
3. Por que extrair uma 'frase nominal' de uma frase seria algo útil?

---

Implemente o bot na verificação de conhecimento anterior e teste-o com um amigo. Ele consegue enganá-lo? Consegue tornar seu bot mais 'crível'?

## 🚀Desafio

Escolha uma tarefa na verificação de conhecimento anterior e tente implementá-la. Teste o bot com um amigo. Ele consegue enganá-lo? Consegue tornar seu bot mais 'crível'?

## [Questionário pós-aula](https://ff-quizzes.netlify.app/en/ml/)

## Revisão & Autoestudo

Nas próximas lições, aprenderá mais sobre análise de sentimento. Pesquise esta técnica interessante em artigos como estes no [KDNuggets](https://www.kdnuggets.com/tag/nlp)

## Tarefa 

[Fazer um bot responder](assignment.md)

---

**Aviso Legal**:  
Este documento foi traduzido utilizando o serviço de tradução por IA [Co-op Translator](https://github.com/Azure/co-op-translator). Embora nos esforcemos para garantir a precisão, esteja ciente de que traduções automáticas podem conter erros ou imprecisões. O documento original no seu idioma nativo deve ser considerado a fonte oficial. Para informações críticas, recomenda-se uma tradução profissional realizada por humanos. Não nos responsabilizamos por quaisquer mal-entendidos ou interpretações incorretas resultantes do uso desta tradução.