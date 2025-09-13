<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "be03c8182982b87ced155e4e9d1438e8",
  "translation_date": "2025-09-05T08:52:47+00:00",
  "source_file": "6-NLP/3-Translation-Sentiment/README.md",
  "language_code": "pt"
}
-->
# Tradução e análise de sentimento com ML

Nas lições anteriores, aprendeste a construir um bot básico utilizando `TextBlob`, uma biblioteca que incorpora ML nos bastidores para realizar tarefas básicas de NLP, como a extração de frases nominais. Outro desafio importante na linguística computacional é a tradução _precisa_ de uma frase de uma língua falada ou escrita para outra.

## [Questionário pré-aula](https://ff-quizzes.netlify.app/en/ml/)

A tradução é um problema muito difícil, agravado pelo facto de existirem milhares de línguas, cada uma com regras gramaticais muito diferentes. Uma abordagem é converter as regras gramaticais formais de uma língua, como o inglês, numa estrutura independente de língua e, em seguida, traduzi-la convertendo-a novamente para outra língua. Esta abordagem implica os seguintes passos:

1. **Identificação**. Identificar ou etiquetar as palavras na língua de entrada como substantivos, verbos, etc.
2. **Criar tradução**. Produzir uma tradução direta de cada palavra no formato da língua de destino.

### Exemplo de frase, Inglês para Irlandês

Em 'Inglês', a frase _I feel happy_ tem três palavras na seguinte ordem:

- **sujeito** (I)
- **verbo** (feel)
- **adjetivo** (happy)

No entanto, na língua 'Irlandesa', a mesma frase tem uma estrutura gramatical muito diferente - emoções como "*happy*" ou "*sad*" são expressas como estando *sobre* ti.

A frase em inglês `I feel happy` em irlandês seria `Tá athas orm`. Uma tradução *literal* seria `Happy is upon me`.

Um falante de irlandês que traduzisse para inglês diria `I feel happy`, não `Happy is upon me`, porque compreende o significado da frase, mesmo que as palavras e a estrutura da frase sejam diferentes.

A ordem formal da frase em irlandês é:

- **verbo** (Tá ou is)
- **adjetivo** (athas, ou happy)
- **sujeito** (orm, ou upon me)

## Tradução

Um programa de tradução ingênuo poderia traduzir apenas palavras, ignorando a estrutura da frase.

✅ Se aprendeste uma segunda (ou terceira ou mais) língua como adulto, provavelmente começaste por pensar na tua língua nativa, traduzindo um conceito palavra por palavra na tua cabeça para a segunda língua e, em seguida, falando a tradução. Isto é semelhante ao que os programas de tradução ingênuos fazem. É importante ultrapassar esta fase para alcançar fluência!

A tradução ingênua leva a traduções erradas (e às vezes hilariantes): `I feel happy` traduzido literalmente para irlandês seria `Mise bhraitheann athas`. Isso significa (literalmente) `me feel happy` e não é uma frase válida em irlandês. Mesmo que o inglês e o irlandês sejam línguas faladas em duas ilhas vizinhas, são línguas muito diferentes com estruturas gramaticais distintas.

> Podes assistir a alguns vídeos sobre tradições linguísticas irlandesas, como [este](https://www.youtube.com/watch?v=mRIaLSdRMMs)

### Abordagens de aprendizagem automática

Até agora, aprendeste sobre a abordagem de regras formais para o processamento de linguagem natural. Outra abordagem é ignorar o significado das palavras e _usar aprendizagem automática para detetar padrões_. Isto pode funcionar na tradução se tiveres muitos textos (um *corpus*) ou textos (*corpora*) na língua de origem e na língua de destino.

Por exemplo, considera o caso de *Orgulho e Preconceito*, um romance inglês bem conhecido escrito por Jane Austen em 1813. Se consultares o livro em inglês e uma tradução humana do livro em *francês*, poderias detetar frases em um que são traduzidas _idiomaticamente_ para o outro. Vais fazer isso em breve.

Por exemplo, quando uma frase em inglês como `I have no money` é traduzida literalmente para francês, pode tornar-se `Je n'ai pas de monnaie`. "Monnaie" é um falso cognato francês complicado, pois 'money' e 'monnaie' não são sinónimos. Uma tradução melhor que um humano poderia fazer seria `Je n'ai pas d'argent`, porque transmite melhor o significado de que não tens dinheiro (em vez de 'troco', que é o significado de 'monnaie').

![monnaie](../../../../6-NLP/3-Translation-Sentiment/images/monnaie.png)

> Imagem por [Jen Looper](https://twitter.com/jenlooper)

Se um modelo de ML tiver traduções humanas suficientes para construir um modelo, pode melhorar a precisão das traduções ao identificar padrões comuns em textos que foram previamente traduzidos por falantes humanos especializados em ambas as línguas.

### Exercício - tradução

Podes usar `TextBlob` para traduzir frases. Experimenta a famosa primeira linha de **Orgulho e Preconceito**:

```python
from textblob import TextBlob

blob = TextBlob(
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife!"
)
print(blob.translate(to="fr"))

```

`TextBlob` faz um bom trabalho na tradução: "C'est une vérité universellement reconnue, qu'un homme célibataire en possession d'une bonne fortune doit avoir besoin d'une femme!". 

Pode-se argumentar que a tradução do TextBlob é muito mais precisa, de facto, do que a tradução francesa de 1932 do livro por V. Leconte e Ch. Pressoir:

"C'est une vérité universelle qu'un célibataire pourvu d'une belle fortune doit avoir envie de se marier, et, si peu que l'on sache de son sentiment à cet egard, lorsqu'il arrive dans une nouvelle résidence, cette idée est si bien fixée dans l'esprit de ses voisins qu'ils le considèrent sur-le-champ comme la propriété légitime de l'une ou l'autre de leurs filles."

Neste caso, a tradução informada por ML faz um trabalho melhor do que o tradutor humano que está a colocar palavras desnecessárias na boca do autor original para 'clareza'.

> O que está a acontecer aqui? E por que o TextBlob é tão bom na tradução? Bem, nos bastidores, está a usar o Google Translate, uma IA sofisticada capaz de analisar milhões de frases para prever as melhores sequências para a tarefa em questão. Não há nada manual aqui e precisas de uma conexão à internet para usar `blob.translate`.

✅ Experimenta mais frases. Qual é melhor, ML ou tradução humana? Em que casos?

## Análise de sentimento

Outra área onde a aprendizagem automática pode funcionar muito bem é na análise de sentimento. Uma abordagem não baseada em ML para sentimento é identificar palavras e frases que são 'positivas' e 'negativas'. Depois, dado um novo texto, calcular o valor total das palavras positivas, negativas e neutras para identificar o sentimento geral. 

Esta abordagem é facilmente enganada, como podes ter visto na tarefa do Marvin - a frase `Great, that was a wonderful waste of time, I'm glad we are lost on this dark road` é uma frase sarcástica, de sentimento negativo, mas o algoritmo simples deteta 'great', 'wonderful', 'glad' como positivas e 'waste', 'lost' e 'dark' como negativas. O sentimento geral é influenciado por estas palavras contraditórias.

✅ Para um momento e pensa sobre como transmitimos sarcasmo como falantes humanos. A inflexão do tom desempenha um papel importante. Experimenta dizer a frase "Well, that film was awesome" de diferentes maneiras para descobrir como a tua voz transmite significado.

### Abordagens de ML

A abordagem de ML seria reunir manualmente corpos de texto negativos e positivos - tweets, ou críticas de filmes, ou qualquer coisa onde o humano tenha dado uma pontuação *e* uma opinião escrita. Depois, técnicas de NLP podem ser aplicadas às opiniões e pontuações, para que padrões emerjam (por exemplo, críticas positivas de filmes tendem a ter a frase 'Oscar worthy' mais do que críticas negativas de filmes, ou críticas positivas de restaurantes dizem 'gourmet' muito mais do que 'disgusting').

> ⚖️ **Exemplo**: Se trabalhares no gabinete de um político e houver uma nova lei a ser debatida, os eleitores podem escrever para o gabinete com emails a favor ou contra a nova lei. Digamos que és encarregado de ler os emails e classificá-los em 2 pilhas, *a favor* e *contra*. Se houver muitos emails, podes sentir-te sobrecarregado ao tentar lê-los todos. Não seria ótimo se um bot pudesse lê-los todos por ti, compreendê-los e dizer-te em que pilha cada email pertence? 
> 
> Uma maneira de conseguir isso é usar Aprendizagem Automática. Treinarias o modelo com uma parte dos emails *contra* e uma parte dos emails *a favor*. O modelo tenderia a associar frases e palavras ao lado contra e ao lado a favor, *mas não entenderia nenhum dos conteúdos*, apenas que certas palavras e padrões eram mais prováveis de aparecer num email *contra* ou *a favor*. Poderias testá-lo com alguns emails que não usaste para treinar o modelo e ver se chegava à mesma conclusão que tu. Depois, uma vez satisfeito com a precisão do modelo, poderias processar emails futuros sem ter de ler cada um.

✅ Este processo parece semelhante a processos que usaste em lições anteriores?

## Exercício - frases sentimentais

O sentimento é medido com uma *polaridade* de -1 a 1, significando que -1 é o sentimento mais negativo e 1 é o mais positivo. O sentimento também é medido com uma pontuação de 0 - 1 para objetividade (0) e subjetividade (1).

Dá outra olhada em *Orgulho e Preconceito* de Jane Austen. O texto está disponível aqui no [Project Gutenberg](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm). O exemplo abaixo mostra um programa curto que analisa o sentimento das primeiras e últimas frases do livro e exibe a sua polaridade de sentimento e pontuação de subjetividade/objetividade.

Deves usar a biblioteca `TextBlob` (descrita acima) para determinar `sentiment` (não precisas de escrever o teu próprio calculador de sentimento) na seguinte tarefa.

```python
from textblob import TextBlob

quote1 = """It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife."""

quote2 = """Darcy, as well as Elizabeth, really loved them; and they were both ever sensible of the warmest gratitude towards the persons who, by bringing her into Derbyshire, had been the means of uniting them."""

sentiment1 = TextBlob(quote1).sentiment
sentiment2 = TextBlob(quote2).sentiment

print(quote1 + " has a sentiment of " + str(sentiment1))
print(quote2 + " has a sentiment of " + str(sentiment2))
```

Vês o seguinte output:

```output
It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want # of a wife. has a sentiment of Sentiment(polarity=0.20952380952380953, subjectivity=0.27142857142857146)

Darcy, as well as Elizabeth, really loved them; and they were
     both ever sensible of the warmest gratitude towards the persons
      who, by bringing her into Derbyshire, had been the means of
      uniting them. has a sentiment of Sentiment(polarity=0.7, subjectivity=0.8)
```

## Desafio - verificar polaridade de sentimento

A tua tarefa é determinar, usando polaridade de sentimento, se *Orgulho e Preconceito* tem mais frases absolutamente positivas do que absolutamente negativas. Para esta tarefa, podes assumir que uma pontuação de polaridade de 1 ou -1 é absolutamente positiva ou negativa, respetivamente.

**Passos:**

1. Faz o download de uma [cópia de Orgulho e Preconceito](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm) do Project Gutenberg como um ficheiro .txt. Remove os metadados no início e no fim do ficheiro, deixando apenas o texto original.
2. Abre o ficheiro em Python e extrai o conteúdo como uma string.
3. Cria um TextBlob usando a string do livro.
4. Analisa cada frase do livro num loop.
   1. Se a polaridade for 1 ou -1, armazena a frase numa lista de mensagens positivas ou negativas.
5. No final, imprime todas as frases positivas e negativas (separadamente) e o número de cada uma.

Aqui está uma [solução de exemplo](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/3-Translation-Sentiment/solution/notebook.ipynb).

✅ Verificação de Conhecimento

1. O sentimento é baseado nas palavras usadas na frase, mas o código *entende* as palavras?
2. Achas que a polaridade de sentimento é precisa, ou seja, *concordas* com as pontuações?
   1. Em particular, concordas ou discordas da polaridade **positiva** absoluta das seguintes frases?
      * “What an excellent father you have, girls!” said she, when the door was shut.
      * “Your examination of Mr. Darcy is over, I presume,” said Miss Bingley; “and pray what is the result?” “I am perfectly convinced by it that Mr. Darcy has no defect.
      * How wonderfully these sort of things occur!
      * I have the greatest dislike in the world to that sort of thing.
      * Charlotte is an excellent manager, I dare say.
      * “This is delightful indeed!
      * I am so happy!
      * Your idea of the ponies is delightful.
   2. As próximas 3 frases foram pontuadas com um sentimento absolutamente positivo, mas, ao ler atentamente, não são frases positivas. Por que a análise de sentimento pensou que eram frases positivas?
      * Happy shall I be, when his stay at Netherfield is over!” “I wish I could say anything to comfort you,” replied Elizabeth; “but it is wholly out of my power.
      * If I could but see you as happy!
      * Our distress, my dear Lizzy, is very great.
   3. Concordas ou discordas da polaridade **negativa** absoluta das seguintes frases?
      - Everybody is disgusted with his pride.
      - “I should like to know how he behaves among strangers.” “You shall hear then—but prepare yourself for something very dreadful.
      - The pause was to Elizabeth’s feelings dreadful.
      - It would be dreadful!

✅ Qualquer aficionado de Jane Austen entenderá que ela frequentemente usa os seus livros para criticar os aspetos mais ridículos da sociedade da Regência Inglesa. Elizabeth Bennett, a personagem principal em *Orgulho e Preconceito*, é uma observadora social perspicaz (como a autora) e a sua linguagem é frequentemente muito subtil. Até mesmo Mr. Darcy (o interesse amoroso na história) nota o uso brincalhão e provocador da linguagem por parte de Elizabeth: "I have had the pleasure of your acquaintance long enough to know that you find great enjoyment in occasionally professing opinions which in fact are not your own."

---

## 🚀Desafio

Consegues melhorar o Marvin ainda mais ao extrair outras características da entrada do utilizador?

## [Questionário pós-aula](https://ff-quizzes.netlify.app/en/ml/)

## Revisão & Autoestudo
Existem muitas formas de extrair sentimentos de texto. Pense nas aplicações empresariais que podem utilizar esta técnica. Reflita sobre como isso pode dar errado. Leia mais sobre sistemas empresariais sofisticados prontos para analisar sentimentos, como [Azure Text Analysis](https://docs.microsoft.com/azure/cognitive-services/Text-Analytics/how-tos/text-analytics-how-to-sentiment-analysis?tabs=version-3-1?WT.mc_id=academic-77952-leestott). Teste algumas das frases de Orgulho e Preconceito acima e veja se consegue detetar nuances.

## Tarefa

[Licença poética](assignment.md)

---

**Aviso Legal**:  
Este documento foi traduzido utilizando o serviço de tradução por IA [Co-op Translator](https://github.com/Azure/co-op-translator). Embora nos esforcemos para garantir a precisão, esteja ciente de que traduções automáticas podem conter erros ou imprecisões. O documento original no seu idioma nativo deve ser considerado a fonte oficial. Para informações críticas, recomenda-se uma tradução profissional realizada por humanos. Não nos responsabilizamos por quaisquer mal-entendidos ou interpretações incorretas resultantes do uso desta tradução.