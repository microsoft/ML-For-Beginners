<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "a2aa4e9b91b9640db2c15363c4299d8b",
  "translation_date": "2025-08-29T22:41:42+00:00",
  "source_file": "6-NLP/5-Hotel-Reviews-2/README.md",
  "language_code": "br"
}
-->
# Análise de sentimento com avaliações de hotéis

Agora que você explorou o conjunto de dados em detalhes, é hora de filtrar as colunas e usar técnicas de NLP no conjunto de dados para obter novos insights sobre os hotéis.

## [Questionário pré-aula](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/39/)

### Operações de Filtragem e Análise de Sentimento

Como você provavelmente percebeu, o conjunto de dados apresenta alguns problemas. Algumas colunas estão preenchidas com informações inúteis, outras parecem incorretas. Mesmo que estejam corretas, não está claro como foram calculadas, e as respostas não podem ser verificadas de forma independente por seus próprios cálculos.

## Exercício: um pouco mais de processamento de dados

Limpe os dados um pouco mais. Adicione colunas que serão úteis mais tarde, altere os valores em outras colunas e exclua certas colunas completamente.

1. Processamento inicial de colunas

   1. Exclua `lat` e `lng`

   2. Substitua os valores de `Hotel_Address` pelos seguintes valores (se o endereço contiver o nome da cidade e do país, altere para apenas a cidade e o país).

      Estas são as únicas cidades e países no conjunto de dados:

      Amsterdam, Netherlands

      Barcelona, Spain

      London, United Kingdom

      Milan, Italy

      Paris, France

      Vienna, Austria 

      ```python
      def replace_address(row):
          if "Netherlands" in row["Hotel_Address"]:
              return "Amsterdam, Netherlands"
          elif "Barcelona" in row["Hotel_Address"]:
              return "Barcelona, Spain"
          elif "United Kingdom" in row["Hotel_Address"]:
              return "London, United Kingdom"
          elif "Milan" in row["Hotel_Address"]:        
              return "Milan, Italy"
          elif "France" in row["Hotel_Address"]:
              return "Paris, France"
          elif "Vienna" in row["Hotel_Address"]:
              return "Vienna, Austria" 
      
      # Replace all the addresses with a shortened, more useful form
      df["Hotel_Address"] = df.apply(replace_address, axis = 1)
      # The sum of the value_counts() should add up to the total number of reviews
      print(df["Hotel_Address"].value_counts())
      ```

      Agora você pode consultar dados em nível de país:

      ```python
      display(df.groupby("Hotel_Address").agg({"Hotel_Name": "nunique"}))
      ```

      | Hotel_Address          | Hotel_Name |
      | :--------------------- | :--------: |
      | Amsterdam, Netherlands |    105     |
      | Barcelona, Spain       |    211     |
      | London, United Kingdom |    400     |
      | Milan, Italy           |    162     |
      | Paris, France          |    458     |
      | Vienna, Austria        |    158     |

2. Processar colunas de Meta-avaliação do Hotel

   1. Exclua `Additional_Number_of_Scoring`

   2. Substitua `Total_Number_of_Reviews` pelo número total de avaliações para aquele hotel que estão realmente no conjunto de dados 

   3. Substitua `Average_Score` pela nossa própria pontuação calculada

   ```python
  # Drop `Additional_Number_of_Scoring`
  df.drop(["Additional_Number_of_Scoring"], axis = 1, inplace=True)
  # Replace `Total_Number_of_Reviews` and `Average_Score` with our own calculated values
  df.Total_Number_of_Reviews = df.groupby('Hotel_Name').transform('count')
  df.Average_Score = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
  ```

3. Processar colunas de avaliação

   1. Exclua `Review_Total_Negative_Word_Counts`, `Review_Total_Positive_Word_Counts`, `Review_Date` e `days_since_review`

   2. Mantenha `Reviewer_Score`, `Negative_Review` e `Positive_Review` como estão
     
   3. Mantenha `Tags` por enquanto

     - Faremos algumas operações adicionais de filtragem nas tags na próxima seção e, em seguida, as tags serão excluídas

4. Processar colunas do avaliador

   1. Exclua `Total_Number_of_Reviews_Reviewer_Has_Given`
  
   2. Mantenha `Reviewer_Nationality`

### Colunas de Tags

A coluna `Tag` é problemática, pois é uma lista (em formato de texto) armazenada na coluna. Infelizmente, a ordem e o número de subseções nesta coluna nem sempre são os mesmos. É difícil para um humano identificar as frases corretas de interesse, porque há 515.000 linhas e 1.427 hotéis, e cada um tem opções ligeiramente diferentes que um avaliador poderia escolher. É aqui que o NLP se destaca. Você pode escanear o texto, encontrar as frases mais comuns e contá-las.

Infelizmente, não estamos interessados em palavras únicas, mas em frases de várias palavras (por exemplo, *Viagem de negócios*). Executar um algoritmo de distribuição de frequência de frases em tantos dados (6.762.646 palavras) pode levar um tempo extraordinário, mas, sem olhar para os dados, parece que isso é uma despesa necessária. É aqui que a análise exploratória de dados é útil, porque você viu uma amostra das tags, como `[' Viagem de negócios  ', ' Viajante solo ', ' Quarto individual ', ' Ficou 5 noites ', ' Enviado de um dispositivo móvel ']`, e pode começar a perguntar se é possível reduzir significativamente o processamento necessário. Felizmente, é - mas primeiro você precisa seguir algumas etapas para determinar as tags de interesse.

### Filtrando tags

Lembre-se de que o objetivo do conjunto de dados é adicionar sentimento e colunas que o ajudarão a escolher o melhor hotel (para você ou talvez para um cliente que o encarregue de criar um bot de recomendação de hotéis). Você precisa se perguntar se as tags são úteis ou não no conjunto de dados final. Aqui está uma interpretação (se você precisasse do conjunto de dados para outros motivos, diferentes tags poderiam permanecer/ser removidas da seleção):

1. O tipo de viagem é relevante e deve permanecer
2. O tipo de grupo de hóspedes é importante e deve permanecer
3. O tipo de quarto, suíte ou estúdio em que o hóspede ficou é irrelevante (todos os hotéis têm basicamente os mesmos quartos)
4. O dispositivo no qual a avaliação foi enviada é irrelevante
5. O número de noites que o avaliador ficou *poderia* ser relevante se você atribuísse estadias mais longas a uma maior satisfação com o hotel, mas é improvável e provavelmente irrelevante

Em resumo, **mantenha 2 tipos de tags e remova as outras**.

Primeiro, você não quer contar as tags até que estejam em um formato melhor, o que significa remover os colchetes e aspas. Você pode fazer isso de várias maneiras, mas deseja a mais rápida, pois pode levar muito tempo para processar muitos dados. Felizmente, o pandas tem uma maneira fácil de realizar cada uma dessas etapas.

```Python
# Remove opening and closing brackets
df.Tags = df.Tags.str.strip("[']")
# remove all quotes too
df.Tags = df.Tags.str.replace(" ', '", ",", regex = False)
```

Cada tag se torna algo como: `Viagem de negócios, Viajante solo, Quarto individual, Ficou 5 noites, Enviado de um dispositivo móvel`. 

Em seguida, encontramos um problema. Algumas avaliações, ou linhas, têm 5 colunas, outras 3, outras 6. Isso é resultado de como o conjunto de dados foi criado e é difícil de corrigir. Você quer obter uma contagem de frequência de cada frase, mas elas estão em ordens diferentes em cada avaliação, então a contagem pode estar errada, e um hotel pode não receber uma tag que merecia.

Em vez disso, você usará a ordem diferente a seu favor, porque cada tag é uma frase de várias palavras, mas também separada por uma vírgula! A maneira mais simples de fazer isso é criar 6 colunas temporárias com cada tag inserida na coluna correspondente à sua ordem na tag. Você pode então mesclar as 6 colunas em uma grande coluna e executar o método `value_counts()` na coluna resultante. Ao imprimir isso, você verá que havia 2.428 tags únicas. Aqui está uma pequena amostra:

| Tag                            | Contagem |
| ------------------------------ | -------- |
| Viagem de lazer                | 417778   |
| Enviado de um dispositivo móvel| 307640   |
| Casal                          | 252294   |
| Ficou 1 noite                  | 193645   |
| Ficou 2 noites                 | 133937   |
| Viajante solo                  | 108545   |
| Ficou 3 noites                 | 95821    |
| Viagem de negócios             | 82939    |
| Grupo                          | 65392    |
| Família com crianças pequenas  | 61015    |
| Ficou 4 noites                 | 47817    |
| Quarto Duplo                   | 35207    |
| Quarto Duplo Standard          | 32248    |
| Quarto Duplo Superior          | 31393    |
| Família com crianças mais velhas| 26349   |
| Quarto Duplo Deluxe            | 24823    |
| Quarto Duplo ou Twin           | 22393    |
| Ficou 5 noites                 | 20845    |
| Quarto Duplo ou Twin Standard  | 17483    |
| Quarto Duplo Clássico          | 16989    |
| Quarto Duplo ou Twin Superior  | 13570    |
| 2 quartos                      | 12393    |

Algumas das tags comuns, como `Enviado de um dispositivo móvel`, não são úteis para nós, então pode ser inteligente removê-las antes de contar a ocorrência das frases, mas é uma operação tão rápida que você pode deixá-las e ignorá-las.

### Removendo as tags de duração da estadia

Remover essas tags é o passo 1, reduzindo ligeiramente o número total de tags a serem consideradas. Observe que você não as remove do conjunto de dados, apenas escolhe não considerá-las como valores a contar/manter no conjunto de avaliações.

| Duração da estadia | Contagem |
| ------------------ | -------- |
| Ficou 1 noite      | 193645   |
| Ficou 2 noites     | 133937   |
| Ficou 3 noites     | 95821    |
| Ficou 4 noites     | 47817    |
| Ficou 5 noites     | 20845    |
| Ficou 6 noites     | 9776     |
| Ficou 7 noites     | 7399     |
| Ficou 8 noites     | 2502     |
| Ficou 9 noites     | 1293     |
| ...                | ...      |

Há uma enorme variedade de quartos, suítes, estúdios, apartamentos e assim por diante. Todos significam basicamente a mesma coisa e não são relevantes para você, então remova-os da consideração.

| Tipo de quarto                | Contagem |
| ----------------------------- | -------- |
| Quarto Duplo                  | 35207    |
| Quarto Duplo Standard         | 32248    |
| Quarto Duplo Superior         | 31393    |
| Quarto Duplo Deluxe           | 24823    |
| Quarto Duplo ou Twin          | 22393    |
| Quarto Duplo ou Twin Standard | 17483    |
| Quarto Duplo Clássico         | 16989    |
| Quarto Duplo ou Twin Superior | 13570    |

Finalmente, e isso é ótimo (porque não exigiu muito processamento), você ficará com as seguintes tags *úteis*:

| Tag                                           | Contagem |
| --------------------------------------------- | -------- |
| Viagem de lazer                               | 417778   |
| Casal                                         | 252294   |
| Viajante solo                                 | 108545   |
| Viagem de negócios                            | 82939    |
| Grupo (combinado com Viajantes com amigos)    | 67535    |
| Família com crianças pequenas                 | 61015    |
| Família com crianças mais velhas              | 26349    |
| Com um animal de estimação                    | 1405     |

Você poderia argumentar que `Viajantes com amigos` é praticamente o mesmo que `Grupo`, e seria justo combinar os dois como acima. O código para identificar as tags corretas está no [notebook de Tags](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb).

O passo final é criar novas colunas para cada uma dessas tags. Então, para cada linha de avaliação, se a coluna `Tag` corresponder a uma das novas colunas, adicione um 1, caso contrário, adicione um 0. O resultado final será uma contagem de quantos avaliadores escolheram este hotel (em agregado) para, por exemplo, negócios vs lazer, ou para trazer um animal de estimação, e isso é uma informação útil ao recomendar um hotel.

```python
# Process the Tags into new columns
# The file Hotel_Reviews_Tags.py, identifies the most important tags
# Leisure trip, Couple, Solo traveler, Business trip, Group combined with Travelers with friends, 
# Family with young children, Family with older children, With a pet
df["Leisure_trip"] = df.Tags.apply(lambda tag: 1 if "Leisure trip" in tag else 0)
df["Couple"] = df.Tags.apply(lambda tag: 1 if "Couple" in tag else 0)
df["Solo_traveler"] = df.Tags.apply(lambda tag: 1 if "Solo traveler" in tag else 0)
df["Business_trip"] = df.Tags.apply(lambda tag: 1 if "Business trip" in tag else 0)
df["Group"] = df.Tags.apply(lambda tag: 1 if "Group" in tag or "Travelers with friends" in tag else 0)
df["Family_with_young_children"] = df.Tags.apply(lambda tag: 1 if "Family with young children" in tag else 0)
df["Family_with_older_children"] = df.Tags.apply(lambda tag: 1 if "Family with older children" in tag else 0)
df["With_a_pet"] = df.Tags.apply(lambda tag: 1 if "With a pet" in tag else 0)

```

### Salve seu arquivo

Finalmente, salve o conjunto de dados como está agora com um novo nome.

```python
df.drop(["Review_Total_Negative_Word_Counts", "Review_Total_Positive_Word_Counts", "days_since_review", "Total_Number_of_Reviews_Reviewer_Has_Given"], axis = 1, inplace=True)

# Saving new data file with calculated columns
print("Saving results to Hotel_Reviews_Filtered.csv")
df.to_csv(r'../data/Hotel_Reviews_Filtered.csv', index = False)
```

## Operações de Análise de Sentimento

Nesta seção final, você aplicará análise de sentimento às colunas de avaliação e salvará os resultados em um conjunto de dados.

## Exercício: carregar e salvar os dados filtrados

Observe que agora você está carregando o conjunto de dados filtrado que foi salvo na seção anterior, **não** o conjunto de dados original.

```python
import time
import pandas as pd
import nltk as nltk
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

# Load the filtered hotel reviews from CSV
df = pd.read_csv('../../data/Hotel_Reviews_Filtered.csv')

# You code will be added here


# Finally remember to save the hotel reviews with new NLP data added
print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r'../data/Hotel_Reviews_NLP.csv', index = False)
```

### Removendo palavras de parada

Se você fosse executar a Análise de Sentimento nas colunas de Avaliações Negativas e Positivas, isso poderia levar muito tempo. Testado em um laptop de teste poderoso com CPU rápida, levou de 12 a 14 minutos, dependendo da biblioteca de sentimento usada. Esse é um tempo (relativamente) longo, então vale a pena investigar se isso pode ser acelerado.

Remover palavras de parada, ou palavras comuns em inglês que não alteram o sentimento de uma frase, é o primeiro passo. Ao removê-las, a análise de sentimento deve ser mais rápida, mas não menos precisa (já que as palavras de parada não afetam o sentimento, mas desaceleram a análise).

A avaliação negativa mais longa tinha 395 palavras, mas após remover as palavras de parada, ficou com 195 palavras.

Remover as palavras de parada também é uma operação rápida, levando 3,3 segundos para remover as palavras de parada de 2 colunas de avaliação em 515.000 linhas no dispositivo de teste. Pode levar um pouco mais ou menos tempo para você, dependendo da velocidade da CPU do seu dispositivo, RAM, se você tem um SSD ou não, e outros fatores. A relativa rapidez da operação significa que, se ela melhorar o tempo de análise de sentimento, vale a pena fazê-la.

```python
from nltk.corpus import stopwords

# Load the hotel reviews from CSV
df = pd.read_csv("../../data/Hotel_Reviews_Filtered.csv")

# Remove stop words - can be slow for a lot of text!
# Ryan Han (ryanxjhan on Kaggle) has a great post measuring performance of different stop words removal approaches
# https://www.kaggle.com/ryanxjhan/fast-stop-words-removal # using the approach that Ryan recommends
start = time.time()
cache = set(stopwords.words("english"))
def remove_stopwords(review):
    text = " ".join([word for word in review.split() if word not in cache])
    return text

# Remove the stop words from both columns
df.Negative_Review = df.Negative_Review.apply(remove_stopwords)   
df.Positive_Review = df.Positive_Review.apply(remove_stopwords)
```

### Realizando análise de sentimento
Agora você deve calcular a análise de sentimento para as colunas de avaliações negativas e positivas, e armazenar o resultado em 2 novas colunas. O teste do sentimento será compará-lo à pontuação dada pelo avaliador para a mesma avaliação. Por exemplo, se a análise de sentimento indicar que a avaliação negativa teve um sentimento de 1 (extremamente positivo) e a avaliação positiva também teve um sentimento de 1, mas o avaliador deu a menor pontuação possível ao hotel, então ou o texto da avaliação não corresponde à pontuação, ou o analisador de sentimentos não conseguiu reconhecer o sentimento corretamente. Você deve esperar que algumas pontuações de sentimento estejam completamente erradas, e muitas vezes isso será explicável, por exemplo, a avaliação pode ser extremamente sarcástica: "Claro que eu AMEI dormir em um quarto sem aquecimento", e o analisador de sentimentos interpreta isso como um sentimento positivo, mesmo que um humano lendo saiba que é sarcasmo.

O NLTK fornece diferentes analisadores de sentimentos para aprender, e você pode substituí-los e ver se o sentimento é mais ou menos preciso. A análise de sentimento VADER é usada aqui.

> Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.

```python
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Create the vader sentiment analyser (there are others in NLTK you can try too)
vader_sentiment = SentimentIntensityAnalyzer()
# Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.

# There are 3 possibilities of input for a review:
# It could be "No Negative", in which case, return 0
# It could be "No Positive", in which case, return 0
# It could be a review, in which case calculate the sentiment
def calc_sentiment(review):    
    if review == "No Negative" or review == "No Positive":
        return 0
    return vader_sentiment.polarity_scores(review)["compound"]    
```

Mais tarde, no seu programa, quando estiver pronto para calcular o sentimento, você pode aplicá-lo a cada avaliação da seguinte forma:

```python
# Add a negative sentiment and positive sentiment column
print("Calculating sentiment columns for both positive and negative reviews")
start = time.time()
df["Negative_Sentiment"] = df.Negative_Review.apply(calc_sentiment)
df["Positive_Sentiment"] = df.Positive_Review.apply(calc_sentiment)
end = time.time()
print("Calculating sentiment took " + str(round(end - start, 2)) + " seconds")
```

Isso leva aproximadamente 120 segundos no meu computador, mas pode variar em cada máquina. Se você quiser imprimir os resultados e verificar se o sentimento corresponde à avaliação:

```python
df = df.sort_values(by=["Negative_Sentiment"], ascending=True)
print(df[["Negative_Review", "Negative_Sentiment"]])
df = df.sort_values(by=["Positive_Sentiment"], ascending=True)
print(df[["Positive_Review", "Positive_Sentiment"]])
```

A última coisa a fazer com o arquivo antes de usá-lo no desafio é salvá-lo! Você também deve considerar reorganizar todas as suas novas colunas para que sejam fáceis de trabalhar (para um humano, é uma mudança cosmética).

```python
# Reorder the columns (This is cosmetic, but to make it easier to explore the data later)
df = df.reindex(["Hotel_Name", "Hotel_Address", "Total_Number_of_Reviews", "Average_Score", "Reviewer_Score", "Negative_Sentiment", "Positive_Sentiment", "Reviewer_Nationality", "Leisure_trip", "Couple", "Solo_traveler", "Business_trip", "Group", "Family_with_young_children", "Family_with_older_children", "With_a_pet", "Negative_Review", "Positive_Review"], axis=1)

print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r"../data/Hotel_Reviews_NLP.csv", index = False)
```

Você deve executar todo o código do [notebook de análise](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) (depois de executar o [notebook de filtragem](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) para gerar o arquivo Hotel_Reviews_Filtered.csv).

Para revisar, os passos são:

1. O arquivo do conjunto de dados original **Hotel_Reviews.csv** foi explorado na lição anterior com [o notebook explorador](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/4-Hotel-Reviews-1/solution/notebook.ipynb)
2. Hotel_Reviews.csv é filtrado pelo [notebook de filtragem](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb), resultando em **Hotel_Reviews_Filtered.csv**
3. Hotel_Reviews_Filtered.csv é processado pelo [notebook de análise de sentimento](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb), resultando em **Hotel_Reviews_NLP.csv**
4. Use Hotel_Reviews_NLP.csv no Desafio de NLP abaixo

### Conclusão

Quando você começou, tinha um conjunto de dados com colunas e informações, mas nem tudo podia ser verificado ou utilizado. Você explorou os dados, filtrou o que não era necessário, converteu tags em algo útil, calculou suas próprias médias, adicionou algumas colunas de sentimento e, com sorte, aprendeu coisas interessantes sobre o processamento de texto natural.

## [Questionário pós-aula](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/40/)

## Desafio

Agora que você analisou o sentimento do seu conjunto de dados, veja se consegue usar estratégias que aprendeu neste curso (talvez agrupamento?) para determinar padrões relacionados ao sentimento.

## Revisão e Autoestudo

Faça [este módulo do Learn](https://docs.microsoft.com/en-us/learn/modules/classify-user-feedback-with-the-text-analytics-api/?WT.mc_id=academic-77952-leestott) para aprender mais e usar diferentes ferramentas para explorar sentimentos em textos.

## Tarefa

[Experimente um conjunto de dados diferente](assignment.md)

---

**Aviso Legal**:  
Este documento foi traduzido utilizando o serviço de tradução por IA [Co-op Translator](https://github.com/Azure/co-op-translator). Embora nos esforcemos para garantir a precisão, esteja ciente de que traduções automatizadas podem conter erros ou imprecisões. O documento original em seu idioma nativo deve ser considerado a fonte autoritativa. Para informações críticas, recomenda-se a tradução profissional realizada por humanos. Não nos responsabilizamos por quaisquer mal-entendidos ou interpretações equivocadas decorrentes do uso desta tradução.