<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "be03c8182982b87ced155e4e9d1438e8",
  "translation_date": "2025-09-04T21:46:33+00:00",
  "source_file": "6-NLP/3-Translation-Sentiment/README.md",
  "language_code": "br"
}
-->
# Tradução e análise de sentimentos com ML

Nas lições anteriores, você aprendeu como construir um bot básico usando `TextBlob`, uma biblioteca que incorpora aprendizado de máquina nos bastidores para realizar tarefas básicas de PLN, como extração de frases nominais. Outro desafio importante na linguística computacional é a _tradução_ precisa de uma frase de um idioma falado ou escrito para outro.

## [Quiz pré-aula](https://ff-quizzes.netlify.app/en/ml/)

A tradução é um problema muito difícil, agravado pelo fato de que existem milhares de idiomas, cada um com regras gramaticais muito diferentes. Uma abordagem é converter as regras gramaticais formais de um idioma, como o inglês, em uma estrutura independente de idioma e, em seguida, traduzi-la convertendo de volta para outro idioma. Essa abordagem envolve as seguintes etapas:

1. **Identificação**. Identificar ou marcar as palavras no idioma de entrada como substantivos, verbos, etc.
2. **Criar tradução**. Produzir uma tradução direta de cada palavra no formato do idioma de destino.

### Exemplo de frase, Inglês para Irlandês

Em 'inglês', a frase _I feel happy_ tem três palavras na seguinte ordem:

- **sujeito** (I)
- **verbo** (feel)
- **adjetivo** (happy)

No entanto, no idioma 'irlandês', a mesma frase tem uma estrutura gramatical muito diferente - emoções como "*feliz*" ou "*triste*" são expressas como estando *sobre* você.

A frase em inglês `I feel happy` em irlandês seria `Tá athas orm`. Uma tradução *literal* seria `Felicidade está sobre mim`.

Um falante de irlandês traduzindo para o inglês diria `I feel happy`, e não `Happy is upon me`, porque ele entende o significado da frase, mesmo que as palavras e a estrutura da frase sejam diferentes.

A ordem formal da frase em irlandês é:

- **verbo** (Tá ou é)
- **adjetivo** (athas, ou feliz)
- **sujeito** (orm, ou sobre mim)

## Tradução

Um programa de tradução ingênuo pode traduzir apenas palavras, ignorando a estrutura da frase.

✅ Se você aprendeu um segundo (ou terceiro ou mais) idioma como adulto, pode ter começado pensando no seu idioma nativo, traduzindo um conceito palavra por palavra na sua cabeça para o segundo idioma e, em seguida, falando sua tradução. Isso é semelhante ao que programas de tradução ingênuos fazem. É importante superar essa fase para alcançar a fluência!

A tradução ingênua leva a traduções ruins (e às vezes hilárias): `I feel happy` traduz literalmente para `Mise bhraitheann athas` em irlandês. Isso significa (literalmente) `eu sinto feliz` e não é uma frase válida em irlandês. Mesmo que o inglês e o irlandês sejam idiomas falados em duas ilhas vizinhas, eles são muito diferentes, com estruturas gramaticais distintas.

> Você pode assistir a alguns vídeos sobre as tradições linguísticas irlandesas, como [este aqui](https://www.youtube.com/watch?v=mRIaLSdRMMs)

### Abordagens de aprendizado de máquina

Até agora, você aprendeu sobre a abordagem de regras formais para o processamento de linguagem natural. Outra abordagem é ignorar o significado das palavras e _usar aprendizado de máquina para detectar padrões_. Isso pode funcionar na tradução se você tiver muitos textos (um *corpus*) ou textos (*corpora*) nos idiomas de origem e de destino.

Por exemplo, considere o caso de *Orgulho e Preconceito*, um romance inglês bem conhecido escrito por Jane Austen em 1813. Se você consultar o livro em inglês e uma tradução humana do livro em *francês*, poderá detectar frases em um idioma que são traduzidas _idiomaticamente_ para o outro. Você fará isso em breve.

Por exemplo, quando uma frase em inglês como `I have no money` é traduzida literalmente para o francês, pode se tornar `Je n'ai pas de monnaie`. "Monnaie" é um falso cognato complicado em francês, pois 'money' e 'monnaie' não são sinônimos. Uma tradução melhor que um humano poderia fazer seria `Je n'ai pas d'argent`, porque transmite melhor o significado de que você não tem dinheiro (em vez de 'troco', que é o significado de 'monnaie').

![monnaie](../../../../6-NLP/3-Translation-Sentiment/images/monnaie.png)

> Imagem por [Jen Looper](https://twitter.com/jenlooper)

Se um modelo de ML tiver traduções humanas suficientes para construir um modelo, ele pode melhorar a precisão das traduções identificando padrões comuns em textos que foram previamente traduzidos por falantes humanos especialistas em ambos os idiomas.

### Exercício - tradução

Você pode usar `TextBlob` para traduzir frases. Experimente a famosa primeira linha de **Orgulho e Preconceito**:

```python
from textblob import TextBlob

blob = TextBlob(
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife!"
)
print(blob.translate(to="fr"))

```

`TextBlob` faz um trabalho muito bom na tradução: "C'est une vérité universellement reconnue, qu'un homme célibataire en possession d'une bonne fortune doit avoir besoin d'une femme!". 

Pode-se argumentar que a tradução do TextBlob é muito mais precisa, de fato, do que a tradução francesa de 1932 do livro por V. Leconte e Ch. Pressoir:

"C'est une vérité universelle qu'un célibataire pourvu d'une belle fortune doit avoir envie de se marier, et, si peu que l'on sache de son sentiment à cet egard, lorsqu'il arrive dans une nouvelle résidence, cette idée est si bien fixée dans l'esprit de ses voisins qu'ils le considèrent sur-le-champ comme la propriété légitime de l'une ou l'autre de leurs filles."

Nesse caso, a tradução informada por ML faz um trabalho melhor do que o tradutor humano, que adicionou palavras desnecessárias à boca do autor original para 'clareza'.

> O que está acontecendo aqui? E por que o TextBlob é tão bom em tradução? Bem, nos bastidores, ele está usando o Google Translate, uma IA sofisticada capaz de analisar milhões de frases para prever as melhores sequências para a tarefa em questão. Não há nada manual acontecendo aqui, e você precisa de uma conexão com a internet para usar `blob.translate`.

✅ Experimente mais frases. Qual é melhor, ML ou tradução humana? Em quais casos?

## Análise de sentimentos

Outra área onde o aprendizado de máquina pode funcionar muito bem é a análise de sentimentos. Uma abordagem não baseada em ML para sentimentos é identificar palavras e frases que são 'positivas' e 'negativas'. Em seguida, dado um novo texto, calcular o valor total das palavras positivas, negativas e neutras para identificar o sentimento geral.

Essa abordagem é facilmente enganada, como você pode ter visto na tarefa do Marvin - a frase `Great, that was a wonderful waste of time, I'm glad we are lost on this dark road` é uma frase sarcástica e de sentimento negativo, mas o algoritmo simples detecta 'great', 'wonderful', 'glad' como positivas e 'waste', 'lost' e 'dark' como negativas. O sentimento geral é influenciado por essas palavras conflitantes.

✅ Pare um segundo e pense em como transmitimos sarcasmo como falantes humanos. A inflexão do tom desempenha um papel importante. Tente dizer a frase "Well, that film was awesome" de diferentes maneiras para descobrir como sua voz transmite significado.

### Abordagens de ML

A abordagem de ML seria reunir manualmente textos negativos e positivos - tweets, resenhas de filmes ou qualquer coisa onde o humano tenha dado uma pontuação *e* uma opinião escrita. Em seguida, técnicas de PLN podem ser aplicadas às opiniões e pontuações, para que padrões surjam (por exemplo, resenhas de filmes positivas tendem a ter a frase 'Oscar worthy' mais do que resenhas negativas, ou resenhas positivas de restaurantes dizem 'gourmet' muito mais do que 'disgusting').

> ⚖️ **Exemplo**: Se você trabalhasse no escritório de um político e houvesse uma nova lei sendo debatida, os eleitores poderiam escrever para o escritório com e-mails a favor ou contra a nova lei. Digamos que você seja encarregado de ler os e-mails e classificá-los em 2 pilhas, *a favor* e *contra*. Se houvesse muitos e-mails, você poderia se sentir sobrecarregado tentando ler todos. Não seria ótimo se um bot pudesse ler todos para você, entendê-los e dizer em qual pilha cada e-mail deveria estar? 
> 
> Uma maneira de conseguir isso é usar aprendizado de máquina. Você treinaria o modelo com uma parte dos e-mails *contra* e uma parte dos e-mails *a favor*. O modelo tenderia a associar frases e palavras ao lado contra e ao lado a favor, *mas não entenderia nenhum conteúdo*, apenas que certas palavras e padrões eram mais propensos a aparecer em um e-mail *contra* ou *a favor*. Você poderia testá-lo com alguns e-mails que não usou para treinar o modelo e ver se ele chegava à mesma conclusão que você. Então, uma vez satisfeito com a precisão do modelo, você poderia processar e-mails futuros sem precisar ler cada um.

✅ Esse processo soa como processos que você usou em lições anteriores?

## Exercício - frases sentimentais

O sentimento é medido com uma *polaridade* de -1 a 1, onde -1 é o sentimento mais negativo e 1 é o mais positivo. O sentimento também é medido com uma pontuação de 0 - 1 para objetividade (0) e subjetividade (1).

Dê outra olhada em *Orgulho e Preconceito* de Jane Austen. O texto está disponível aqui no [Project Gutenberg](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm). O exemplo abaixo mostra um programa curto que analisa o sentimento das primeiras e últimas frases do livro e exibe sua polaridade de sentimento e pontuação de subjetividade/objetividade.

Você deve usar a biblioteca `TextBlob` (descrita acima) para determinar o `sentimento` (você não precisa escrever seu próprio calculador de sentimentos) na seguinte tarefa.

```python
from textblob import TextBlob

quote1 = """It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife."""

quote2 = """Darcy, as well as Elizabeth, really loved them; and they were both ever sensible of the warmest gratitude towards the persons who, by bringing her into Derbyshire, had been the means of uniting them."""

sentiment1 = TextBlob(quote1).sentiment
sentiment2 = TextBlob(quote2).sentiment

print(quote1 + " has a sentiment of " + str(sentiment1))
print(quote2 + " has a sentiment of " + str(sentiment2))
```

Você verá a seguinte saída:

```output
It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want # of a wife. has a sentiment of Sentiment(polarity=0.20952380952380953, subjectivity=0.27142857142857146)

Darcy, as well as Elizabeth, really loved them; and they were
     both ever sensible of the warmest gratitude towards the persons
      who, by bringing her into Derbyshire, had been the means of
      uniting them. has a sentiment of Sentiment(polarity=0.7, subjectivity=0.8)
```

## Desafio - verificar polaridade de sentimentos

Sua tarefa é determinar, usando a polaridade de sentimentos, se *Orgulho e Preconceito* tem mais frases absolutamente positivas do que absolutamente negativas. Para esta tarefa, você pode assumir que uma pontuação de polaridade de 1 ou -1 é absolutamente positiva ou negativa, respectivamente.

**Passos:**

1. Baixe uma [cópia de Orgulho e Preconceito](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm) do Project Gutenberg como um arquivo .txt. Remova os metadados no início e no final do arquivo, deixando apenas o texto original.
2. Abra o arquivo em Python e extraia o conteúdo como uma string.
3. Crie um TextBlob usando a string do livro.
4. Analise cada frase do livro em um loop.
   1. Se a polaridade for 1 ou -1, armazene a frase em um array ou lista de mensagens positivas ou negativas.
5. No final, imprima todas as frases positivas e negativas (separadamente) e o número de cada uma.

Aqui está uma [solução de exemplo](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/3-Translation-Sentiment/solution/notebook.ipynb).

✅ Verificação de Conhecimento

1. O sentimento é baseado nas palavras usadas na frase, mas o código *entende* as palavras?
2. Você acha que a polaridade de sentimento é precisa ou, em outras palavras, você *concorda* com as pontuações?
   1. Em particular, você concorda ou discorda da polaridade **positiva** absoluta das seguintes frases?
      * “What an excellent father you have, girls!” said she, when the door was shut.
      * “Your examination of Mr. Darcy is over, I presume,” said Miss Bingley; “and pray what is the result?” “I am perfectly convinced by it that Mr. Darcy has no defect.
      * How wonderfully these sort of things occur!
      * I have the greatest dislike in the world to that sort of thing.
      * Charlotte is an excellent manager, I dare say.
      * “This is delightful indeed!
      * I am so happy!
      * Your idea of the ponies is delightful.
   2. As próximas 3 frases foram pontuadas com um sentimento positivo absoluto, mas, ao ler com atenção, elas não são frases positivas. Por que a análise de sentimentos achou que eram frases positivas?
      * Happy shall I be, when his stay at Netherfield is over!” “I wish I could say anything to comfort you,” replied Elizabeth; “but it is wholly out of my power.
      * If I could but see you as happy!
      * Our distress, my dear Lizzy, is very great.
   3. Você concorda ou discorda da polaridade **negativa** absoluta das seguintes frases?
      - Everybody is disgusted with his pride.
      - “I should like to know how he behaves among strangers.” “You shall hear then—but prepare yourself for something very dreadful.
      - The pause was to Elizabeth’s feelings dreadful.
      - It would be dreadful!

✅ Qualquer aficionado por Jane Austen entenderá que ela frequentemente usa seus livros para criticar os aspectos mais ridículos da sociedade da Regência Inglesa. Elizabeth Bennett, a personagem principal de *Orgulho e Preconceito*, é uma observadora social perspicaz (como a autora), e sua linguagem é frequentemente muito sutil. Até mesmo Mr. Darcy (o interesse amoroso na história) observa o uso brincalhão e provocador da linguagem por Elizabeth: "Eu tive o prazer de sua companhia por tempo suficiente para saber que você encontra grande diversão em ocasionalmente professar opiniões que, na verdade, não são suas."

---

## 🚀Desafio

Você pode melhorar o Marvin extraindo outros recursos da entrada do usuário?

## [Quiz pós-aula](https://ff-quizzes.netlify.app/en/ml/)

## Revisão e Autoestudo
Existem muitas maneiras de extrair sentimentos de um texto. Pense nas aplicações empresariais que podem fazer uso dessa técnica. Reflita sobre como isso pode dar errado. Leia mais sobre sistemas sofisticados e prontos para empresas que analisam sentimentos, como o [Azure Text Analysis](https://docs.microsoft.com/azure/cognitive-services/Text-Analytics/how-tos/text-analytics-how-to-sentiment-analysis?tabs=version-3-1?WT.mc_id=academic-77952-leestott). Teste algumas das frases de Orgulho e Preconceito mencionadas acima e veja se ele consegue detectar nuances.

## Tarefa

[Licença poética](assignment.md)

---

**Aviso Legal**:  
Este documento foi traduzido utilizando o serviço de tradução por IA [Co-op Translator](https://github.com/Azure/co-op-translator). Embora nos esforcemos para garantir a precisão, esteja ciente de que traduções automáticas podem conter erros ou imprecisões. O documento original em seu idioma nativo deve ser considerado a fonte oficial. Para informações críticas, recomenda-se a tradução profissional feita por humanos. Não nos responsabilizamos por quaisquer mal-entendidos ou interpretações equivocadas decorrentes do uso desta tradução.