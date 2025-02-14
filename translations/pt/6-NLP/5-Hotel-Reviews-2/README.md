# Análise de sentimento com avaliações de hotéis

Agora que você explorou o conjunto de dados em detalhes, é hora de filtrar as colunas e usar técnicas de PNL no conjunto de dados para obter novas percepções sobre os hotéis.
## [Quiz pré-aula](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/39/)

### Operações de Filtragem e Análise de Sentimento

Como você provavelmente notou, o conjunto de dados possui alguns problemas. Algumas colunas estão preenchidas com informações inúteis, outras parecem incorretas. Se estiverem corretas, não está claro como foram calculadas, e as respostas não podem ser verificadas de forma independente por seus próprios cálculos.

## Exercício: um pouco mais de processamento de dados

Limpe os dados um pouco mais. Adicione colunas que serão úteis mais tarde, altere os valores em outras colunas e remova certas colunas completamente.

1. Processamento inicial das colunas

   1. Remova `lat` e `lng`

   2. Substitua os valores de `Hotel_Address` pelos seguintes valores (se o endereço contiver o nome da cidade e do país, altere para apenas a cidade e o país).

      Estas são as únicas cidades e países no conjunto de dados:

      Amsterdã, Países Baixos

      Barcelona, Espanha

      Londres, Reino Unido

      Milão, Itália

      Paris, França

      Viena, Áustria 

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
      | Amsterdã, Países Baixos |    105     |
      | Barcelona, Espanha       |    211     |
      | Londres, Reino Unido |    400     |
      | Milão, Itália           |    162     |
      | Paris, França          |    458     |
      | Viena, Áustria        |    158     |

2. Processar colunas de Meta-avaliação do Hotel

  1. Remova `Additional_Number_of_Scoring`

  1. Replace `Total_Number_of_Reviews` with the total number of reviews for that hotel that are actually in the dataset 

  1. Replace `Average_Score` com nossa própria pontuação calculada

  ```python
  # Drop `Additional_Number_of_Scoring`
  df.drop(["Additional_Number_of_Scoring"], axis = 1, inplace=True)
  # Replace `Total_Number_of_Reviews` and `Average_Score` with our own calculated values
  df.Total_Number_of_Reviews = df.groupby('Hotel_Name').transform('count')
  df.Average_Score = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
  ```

3. Processar colunas de avaliação

   1. Remova `Review_Total_Negative_Word_Counts`, `Review_Total_Positive_Word_Counts`, `Review_Date` and `days_since_review`

   2. Keep `Reviewer_Score`, `Negative_Review`, and `Positive_Review` as they are,
     
   3. Keep `Tags` for now

     - We'll be doing some additional filtering operations on the tags in the next section and then tags will be dropped

4. Process reviewer columns

  1. Drop `Total_Number_of_Reviews_Reviewer_Has_Given`
  
  2. Keep `Reviewer_Nationality`

### Tag columns

The `Tag` column is problematic as it is a list (in text form) stored in the column. Unfortunately the order and number of sub sections in this column are not always the same. It's hard for a human to identify the correct phrases to be interested in, because there are 515,000 rows, and 1427 hotels, and each has slightly different options a reviewer could choose. This is where NLP shines. You can scan the text and find the most common phrases, and count them.

Unfortunately, we are not interested in single words, but multi-word phrases (e.g. *Business trip*). Running a multi-word frequency distribution algorithm on that much data (6762646 words) could take an extraordinary amount of time, but without looking at the data, it would seem that is a necessary expense. This is where exploratory data analysis comes in useful, because you've seen a sample of the tags such as `[' Business trip  ', ' Solo traveler ', ' Single Room ', ' Stayed 5 nights ', ' Submitted from  a mobile device ']`, você pode começar a perguntar se é possível reduzir bastante o processamento que você tem que fazer. Felizmente, é - mas primeiro você precisa seguir alguns passos para determinar as tags de interesse.

### Filtrando tags

Lembre-se de que o objetivo do conjunto de dados é adicionar sentimento e colunas que ajudarão você a escolher o melhor hotel (para você ou talvez para um cliente que lhe pediu para criar um bot de recomendação de hotéis). Você precisa se perguntar se as tags são úteis ou não no conjunto de dados final. Aqui está uma interpretação (se você precisasse do conjunto de dados por outros motivos, diferentes tags poderiam ser mantidas ou removidas da seleção):

1. O tipo de viagem é relevante, e isso deve ser mantido
2. O tipo de grupo de hóspedes é importante, e isso deve ser mantido
3. O tipo de quarto, suíte ou estúdio em que o hóspede se hospedou é irrelevante (todos os hotéis têm basicamente os mesmos quartos)
4. O dispositivo em que a avaliação foi enviada é irrelevante
5. O número de noites que o avaliador ficou *poderia* ser relevante se você atribuísse estadias mais longas a uma maior satisfação com o hotel, mas é uma suposição, e provavelmente irrelevante

Em resumo, **mantenha 2 tipos de tags e remova as outras**.

Primeiro, você não quer contar as tags até que elas estejam em um formato melhor, então isso significa remover os colchetes e aspas. Você pode fazer isso de várias maneiras, mas deseja a mais rápida, pois pode levar muito tempo para processar muitos dados. Felizmente, o pandas tem uma maneira fácil de fazer cada um desses passos.

```Python
# Remove opening and closing brackets
df.Tags = df.Tags.str.strip("[']")
# remove all quotes too
df.Tags = df.Tags.str.replace(" ', '", ",", regex = False)
```

Cada tag se torna algo como: `Business trip, Solo traveler, Single Room, Stayed 5 nights, Submitted from a mobile device`. 

Next we find a problem. Some reviews, or rows, have 5 columns, some 3, some 6. This is a result of how the dataset was created, and hard to fix. You want to get a frequency count of each phrase, but they are in different order in each review, so the count might be off, and a hotel might not get a tag assigned to it that it deserved.

Instead you will use the different order to our advantage, because each tag is multi-word but also separated by a comma! The simplest way to do this is to create 6 temporary columns with each tag inserted in to the column corresponding to its order in the tag. You can then merge the 6 columns into one big column and run the `value_counts()` method on the resulting column. Printing that out, you'll see there was 2428 unique tags. Here is a small sample:

| Tag                            | Count  |
| ------------------------------ | ------ |
| Leisure trip                   | 417778 |
| Submitted from a mobile device | 307640 |
| Couple                         | 252294 |
| Stayed 1 night                 | 193645 |
| Stayed 2 nights                | 133937 |
| Solo traveler                  | 108545 |
| Stayed 3 nights                | 95821  |
| Business trip                  | 82939  |
| Group                          | 65392  |
| Family with young children     | 61015  |
| Stayed 4 nights                | 47817  |
| Double Room                    | 35207  |
| Standard Double Room           | 32248  |
| Superior Double Room           | 31393  |
| Family with older children     | 26349  |
| Deluxe Double Room             | 24823  |
| Double or Twin Room            | 22393  |
| Stayed 5 nights                | 20845  |
| Standard Double or Twin Room   | 17483  |
| Classic Double Room            | 16989  |
| Superior Double or Twin Room   | 13570  |
| 2 rooms                        | 12393  |

Some of the common tags like `Submitted from a mobile device` are of no use to us, so it might be a smart thing to remove them before counting phrase occurrence, but it is such a fast operation you can leave them in and ignore them.

### Removing the length of stay tags

Removing these tags is step 1, it reduces the total number of tags to be considered slightly. Note you do not remove them from the dataset, just choose to remove them from consideration as values to  count/keep in the reviews dataset.

| Length of stay   | Count  |
| ---------------- | ------ |
| Stayed 1 night   | 193645 |
| Stayed  2 nights | 133937 |
| Stayed 3 nights  | 95821  |
| Stayed  4 nights | 47817  |
| Stayed 5 nights  | 20845  |
| Stayed  6 nights | 9776   |
| Stayed 7 nights  | 7399   |
| Stayed  8 nights | 2502   |
| Stayed 9 nights  | 1293   |
| ...              | ...    |

There are a huge variety of rooms, suites, studios, apartments and so on. They all mean roughly the same thing and not relevant to you, so remove them from consideration.

| Type of room                  | Count |
| ----------------------------- | ----- |
| Double Room                   | 35207 |
| Standard  Double Room         | 32248 |
| Superior Double Room          | 31393 |
| Deluxe  Double Room           | 24823 |
| Double or Twin Room           | 22393 |
| Standard  Double or Twin Room | 17483 |
| Classic Double Room           | 16989 |
| Superior  Double or Twin Room | 13570 |

Finally, and this is delightful (because it didn't take much processing at all), you will be left with the following *useful* tags:

| Tag                                           | Count  |
| --------------------------------------------- | ------ |
| Leisure trip                                  | 417778 |
| Couple                                        | 252294 |
| Solo  traveler                                | 108545 |
| Business trip                                 | 82939  |
| Group (combined with Travellers with friends) | 67535  |
| Family with young children                    | 61015  |
| Family  with older children                   | 26349  |
| With a  pet                                   | 1405   |

You could argue that `Travellers with friends` is the same as `Group` more or less, and that would be fair to combine the two as above. The code for identifying the correct tags is [the Tags notebook](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb).

The final step is to create new columns for each of these tags. Then, for every review row, if the `Tag` a coluna corresponde a uma das novas colunas, adicione um 1, se não, adicione um 0. O resultado final será uma contagem de quantos avaliadores escolheram este hotel (em agregados) para, digamos, negócios versus lazer, ou para levar um animal de estimação, e isso é uma informação útil ao recomendar um hotel.

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

Nesta seção final, você aplicará a análise de sentimento às colunas de avaliação e salvará os resultados em um conjunto de dados.

## Exercício: carregar e salvar os dados filtrados

Note que agora você está carregando o conjunto de dados filtrado que foi salvo na seção anterior, **não** o conjunto de dados original.

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

Se você fosse executar a Análise de Sentimento nas colunas de avaliação negativa e positiva, isso poderia levar muito tempo. Testado em um laptop de teste poderoso com um CPU rápido, levou de 12 a 14 minutos, dependendo da biblioteca de sentimento utilizada. Esse é um tempo (relativamente) longo, então vale a pena investigar se isso pode ser acelerado.

Remover palavras de parada, ou palavras comuns em inglês que não alteram o sentimento de uma frase, é o primeiro passo. Ao removê-las, a análise de sentimento deve ser executada mais rapidamente, mas não menos precisa (já que as palavras de parada não afetam o sentimento, mas desaceleram a análise).

A avaliação negativa mais longa tinha 395 palavras, mas após a remoção das palavras de parada, ficou com 195 palavras.

Remover as palavras de parada também é uma operação rápida; remover as palavras de parada de 2 colunas de avaliação com mais de 515.000 linhas levou 3,3 segundos no dispositivo de teste. Pode levar um pouco mais ou menos tempo para você, dependendo da velocidade do CPU do seu dispositivo, RAM, se você tem um SSD ou não, e alguns outros fatores. A relativa brevidade da operação significa que, se isso melhorar o tempo da análise de sentimento, vale a pena fazer.

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

### Realizando a análise de sentimento

Agora você deve calcular a análise de sentimento para ambas as colunas de avaliação negativa e positiva e armazenar o resultado em 2 novas colunas. O teste do sentimento será compará-lo à pontuação do avaliador para a mesma avaliação. Por exemplo, se o sentimento acha que a avaliação negativa teve um sentimento de 1 (sentimento extremamente positivo) e um sentimento de avaliação positiva de 1, mas o avaliador deu ao hotel a pontuação mais baixa possível, então ou o texto da avaliação não corresponde à pontuação, ou o analisador de sentimento não conseguiu reconhecer o sentimento corretamente. Você deve esperar que algumas pontuações de sentimento estejam completamente erradas, e muitas vezes isso será explicável, por exemplo, a avaliação pode ser extremamente sarcástica "Claro que ADORO dormir em um quarto sem aquecimento" e o analisador de sentimento acha que isso é um sentimento positivo, mesmo que um humano ao lê-lo saiba que é sarcasmo.

NLTK fornece diferentes analisadores de sentimento para aprender, e você pode substituí-los e ver se o sentimento é mais ou menos preciso. A análise de sentimento VADER é utilizada aqui.

> Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, junho de 2014.

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

Mais tarde em seu programa, quando você estiver pronto para calcular o sentimento, você pode aplicá-lo a cada avaliação da seguinte forma:

```python
# Add a negative sentiment and positive sentiment column
print("Calculating sentiment columns for both positive and negative reviews")
start = time.time()
df["Negative_Sentiment"] = df.Negative_Review.apply(calc_sentiment)
df["Positive_Sentiment"] = df.Positive_Review.apply(calc_sentiment)
end = time.time()
print("Calculating sentiment took " + str(round(end - start, 2)) + " seconds")
```

Isso leva aproximadamente 120 segundos no meu computador, mas varia em cada computador. Se você quiser imprimir os resultados e ver se o sentimento corresponde à avaliação:

```python
df = df.sort_values(by=["Negative_Sentiment"], ascending=True)
print(df[["Negative_Review", "Negative_Sentiment"]])
df = df.sort_values(by=["Positive_Sentiment"], ascending=True)
print(df[["Positive_Review", "Positive_Sentiment"]])
```

A última coisa a fazer com o arquivo antes de usá-lo no desafio é salvá-lo! Você também deve considerar reorganizar todas as suas novas colunas para que sejam fáceis de trabalhar (para um humano, é uma mudança estética).

```python
# Reorder the columns (This is cosmetic, but to make it easier to explore the data later)
df = df.reindex(["Hotel_Name", "Hotel_Address", "Total_Number_of_Reviews", "Average_Score", "Reviewer_Score", "Negative_Sentiment", "Positive_Sentiment", "Reviewer_Nationality", "Leisure_trip", "Couple", "Solo_traveler", "Business_trip", "Group", "Family_with_young_children", "Family_with_older_children", "With_a_pet", "Negative_Review", "Positive_Review"], axis=1)

print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r"../data/Hotel_Reviews_NLP.csv", index = False)
```

Você deve executar todo o código para [o notebook de análise](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) (depois de ter executado [seu notebook de filtragem](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) para gerar o arquivo Hotel_Reviews_Filtered.csv).

Para revisar, os passos são:

1. O arquivo do conjunto de dados original **Hotel_Reviews.csv** foi explorado na lição anterior com [o notebook explorador](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/4-Hotel-Reviews-1/solution/notebook.ipynb)
2. Hotel_Reviews.csv foi filtrado pelo [notebook de filtragem](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb), resultando em **Hotel_Reviews_Filtered.csv**
3. Hotel_Reviews_Filtered.csv foi processado pelo [notebook de análise de sentimento](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb), resultando em **Hotel_Reviews_NLP.csv**
4. Use Hotel_Reviews_NLP.csv no Desafio de PNL abaixo

### Conclusão

Quando você começou, tinha um conjunto de dados com colunas e dados, mas nem todos podiam ser verificados ou utilizados. Você explorou os dados, filtrou o que não precisava, converteu tags em algo útil, calculou suas próprias médias, adicionou algumas colunas de sentimento e, com sorte, aprendeu algumas coisas interessantes sobre o processamento de texto natural.

## [Quiz pós-aula](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/40/)

## Desafio

Agora que você analisou seu conjunto de dados para sentimento, veja se consegue usar estratégias que aprendeu neste currículo (aglomeração, talvez?) para determinar padrões em torno do sentimento.

## Revisão e Estudo Autônomo

Faça [este módulo Learn](https://docs.microsoft.com/en-us/learn/modules/classify-user-feedback-with-the-text-analytics-api/?WT.mc_id=academic-77952-leestott) para aprender mais e usar diferentes ferramentas para explorar o sentimento em texto.
## Tarefa 

[Experimente um conjunto de dados diferente](assignment.md)

**Isenção de responsabilidade**:  
Este documento foi traduzido utilizando serviços de tradução automática baseados em IA. Embora nos esforcemos pela precisão, esteja ciente de que traduções automatizadas podem conter erros ou imprecisões. O documento original em seu idioma nativo deve ser considerado a fonte autoritativa. Para informações críticas, recomenda-se a tradução profissional por um humano. Não nos responsabilizamos por quaisquer mal-entendidos ou interpretações errôneas decorrentes do uso desta tradução.