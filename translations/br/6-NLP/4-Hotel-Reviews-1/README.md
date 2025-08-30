<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "3c4738bb0836dd838c552ab9cab7e09d",
  "translation_date": "2025-08-29T22:26:20+00:00",
  "source_file": "6-NLP/4-Hotel-Reviews-1/README.md",
  "language_code": "br"
}
-->
# Análise de sentimentos com avaliações de hotéis - processando os dados

Nesta seção, você usará as técnicas das lições anteriores para realizar uma análise exploratória de dados em um grande conjunto de dados. Assim que tiver uma boa compreensão da utilidade das várias colunas, você aprenderá:

- como remover colunas desnecessárias
- como calcular novos dados com base nas colunas existentes
- como salvar o conjunto de dados resultante para uso no desafio final

## [Questionário pré-aula](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/37/)

### Introdução

Até agora, você aprendeu que dados textuais são bem diferentes de dados numéricos. Se o texto foi escrito ou falado por um humano, ele pode ser analisado para encontrar padrões, frequências, sentimentos e significados. Esta lição apresenta um conjunto de dados real com um desafio real: **[515K Hotel Reviews Data in Europe](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)**, que inclui uma [licença CC0: Domínio Público](https://creativecommons.org/publicdomain/zero/1.0/). Ele foi extraído do Booking.com a partir de fontes públicas. O criador do conjunto de dados foi Jiashen Liu.

### Preparação

Você vai precisar de:

* Capacidade de executar notebooks .ipynb usando Python 3
* pandas
* NLTK, [que você deve instalar localmente](https://www.nltk.org/install.html)
* O conjunto de dados disponível no Kaggle [515K Hotel Reviews Data in Europe](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe). Ele tem cerca de 230 MB descompactado. Baixe-o para a pasta raiz `/data` associada a estas lições de PLN.

## Análise exploratória de dados

Este desafio assume que você está construindo um bot de recomendação de hotéis usando análise de sentimentos e pontuações de avaliações de hóspedes. O conjunto de dados que você usará inclui avaliações de 1493 hotéis diferentes em 6 cidades.

Usando Python, um conjunto de dados de avaliações de hotéis e a análise de sentimentos do NLTK, você poderia descobrir:

* Quais são as palavras e frases mais frequentemente usadas nas avaliações?
* As *tags* oficiais que descrevem um hotel têm correlação com as pontuações das avaliações (por exemplo, há mais avaliações negativas para um hotel específico por *Famílias com crianças pequenas* do que por *Viajantes solo*, talvez indicando que ele é melhor para *Viajantes solo*)?
* As pontuações de sentimento do NLTK "concordam" com a pontuação numérica do avaliador?

#### Conjunto de dados

Vamos explorar o conjunto de dados que você baixou e salvou localmente. Abra o arquivo em um editor como o VS Code ou até mesmo no Excel.

Os cabeçalhos no conjunto de dados são os seguintes:

*Hotel_Address, Additional_Number_of_Scoring, Review_Date, Average_Score, Hotel_Name, Reviewer_Nationality, Negative_Review, Review_Total_Negative_Word_Counts, Total_Number_of_Reviews, Positive_Review, Review_Total_Positive_Word_Counts, Total_Number_of_Reviews_Reviewer_Has_Given, Reviewer_Score, Tags, days_since_review, lat, lng*

Aqui estão agrupados de uma forma que pode ser mais fácil de examinar: 
##### Colunas do hotel

* `Hotel_Name`, `Hotel_Address`, `lat` (latitude), `lng` (longitude)
  * Usando *lat* e *lng*, você poderia plotar um mapa com Python mostrando as localizações dos hotéis (talvez codificadas por cores para avaliações negativas e positivas)
  * Hotel_Address não parece ser obviamente útil para nós, e provavelmente substituiremos isso por um país para facilitar a classificação e a busca

**Colunas de meta-avaliação do hotel**

* `Average_Score`
  * De acordo com o criador do conjunto de dados, esta coluna é a *Pontuação Média do hotel, calculada com base no comentário mais recente do último ano*. Este parece ser um método incomum para calcular a pontuação, mas é o dado extraído, então podemos aceitá-lo como está por enquanto.
  
  ✅ Com base nas outras colunas deste conjunto de dados, você consegue pensar em outra maneira de calcular a pontuação média?

* `Total_Number_of_Reviews`
  * O número total de avaliações que este hotel recebeu - não está claro (sem escrever algum código) se isso se refere às avaliações no conjunto de dados.
* `Additional_Number_of_Scoring`
  * Isso significa que uma pontuação foi dada, mas nenhuma avaliação positiva ou negativa foi escrita pelo avaliador.

**Colunas de avaliação**

- `Reviewer_Score`
  - Este é um valor numérico com no máximo 1 casa decimal entre os valores mínimos e máximos de 2.5 e 10
  - Não é explicado por que 2.5 é a menor pontuação possível
- `Negative_Review`
  - Se um avaliador não escreveu nada, este campo terá "**No Negative**"
  - Note que um avaliador pode escrever uma avaliação positiva na coluna de avaliação negativa (por exemplo, "não há nada de ruim neste hotel")
- `Review_Total_Negative_Word_Counts`
  - Contagens mais altas de palavras negativas indicam uma pontuação mais baixa (sem verificar a sentimentalidade)
- `Positive_Review`
  - Se um avaliador não escreveu nada, este campo terá "**No Positive**"
  - Note que um avaliador pode escrever uma avaliação negativa na coluna de avaliação positiva (por exemplo, "não há nada de bom neste hotel")
- `Review_Total_Positive_Word_Counts`
  - Contagens mais altas de palavras positivas indicam uma pontuação mais alta (sem verificar a sentimentalidade)
- `Review_Date` e `days_since_review`
  - Uma medida de frescor ou desatualização pode ser aplicada a uma avaliação (avaliações mais antigas podem não ser tão precisas quanto as mais recentes porque a administração do hotel mudou, ou reformas foram feitas, ou uma piscina foi adicionada, etc.)
- `Tags`
  - Estas são descrições curtas que um avaliador pode selecionar para descrever o tipo de hóspede que era (por exemplo, solo ou família), o tipo de quarto que teve, a duração da estadia e como a avaliação foi enviada.
  - Infelizmente, usar essas tags é problemático, veja a seção abaixo que discute sua utilidade.

**Colunas do avaliador**

- `Total_Number_of_Reviews_Reviewer_Has_Given`
  - Isso pode ser um fator em um modelo de recomendação, por exemplo, se você puder determinar que avaliadores mais prolíficos com centenas de avaliações eram mais propensos a serem negativos do que positivos. No entanto, o avaliador de qualquer avaliação específica não é identificado com um código único e, portanto, não pode ser vinculado a um conjunto de avaliações. Há 30 avaliadores com 100 ou mais avaliações, mas é difícil ver como isso pode ajudar no modelo de recomendação.
- `Reviewer_Nationality`
  - Algumas pessoas podem pensar que certas nacionalidades são mais propensas a dar uma avaliação positiva ou negativa devido a uma inclinação nacional. Tenha cuidado ao construir tais visões anedóticas em seus modelos. Estes são estereótipos nacionais (e às vezes raciais), e cada avaliador foi um indivíduo que escreveu uma avaliação com base em sua experiência. Isso pode ter sido filtrado por muitas lentes, como suas estadias anteriores em hotéis, a distância percorrida e seu temperamento pessoal. Pensar que a nacionalidade foi a razão para uma pontuação de avaliação é difícil de justificar.

##### Exemplos

| Average  Score | Total Number   Reviews | Reviewer   Score | Negative <br />Review                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Positive   Review                 | Tags                                                                                      |
| -------------- | ---------------------- | ---------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- | ----------------------------------------------------------------------------------------- |
| 7.8            | 1945                   | 2.5              | Este não é atualmente um hotel, mas um canteiro de obras. Fui atormentado desde cedo pela manhã e durante todo o dia com ruídos inaceitáveis de construção enquanto descansava após uma longa viagem e trabalhava no quarto. Pessoas estavam trabalhando o dia todo, por exemplo, com martelos pneumáticos nos quartos adjacentes. Pedi para trocar de quarto, mas nenhum quarto silencioso estava disponível. Para piorar, fui cobrado a mais. Fiz o check-out à noite, pois tinha um voo muito cedo, e recebi uma conta apropriada. Um dia depois, o hotel fez outra cobrança sem meu consentimento, excedendo o preço reservado. É um lugar terrível. Não se puna reservando aqui. | Nada. Lugar terrível. Fique longe. | Viagem de negócios, Casal, Quarto Duplo Padrão, Ficou 2 noites |

Como você pode ver, este hóspede não teve uma estadia feliz neste hotel. O hotel tem uma boa pontuação média de 7.8 e 1945 avaliações, mas este avaliador deu 2.5 e escreveu 115 palavras sobre como sua estadia foi negativa. Se ele não tivesse escrito nada na coluna Positive_Review, você poderia supor que não havia nada positivo, mas, ainda assim, ele escreveu 7 palavras de aviso. Se contássemos apenas as palavras em vez do significado ou sentimento das palavras, poderíamos ter uma visão distorcida da intenção do avaliador. Estranhamente, sua pontuação de 2.5 é confusa, porque se a estadia no hotel foi tão ruim, por que dar qualquer ponto? Investigando o conjunto de dados de perto, você verá que a menor pontuação possível é 2.5, não 0. A maior pontuação possível é 10.

##### Tags

Como mencionado acima, à primeira vista, a ideia de usar `Tags` para categorizar os dados faz sentido. Infelizmente, essas tags não são padronizadas, o que significa que, em um hotel, as opções podem ser *Quarto Individual*, *Quarto Duplo*, e *Quarto Twin*, mas no próximo hotel, elas são *Quarto Individual Deluxe*, *Quarto Queen Clássico*, e *Quarto King Executivo*. Esses podem ser os mesmos tipos de quarto, mas há tantas variações que a escolha se torna:

1. Tentar alterar todos os termos para um único padrão, o que é muito difícil, porque não está claro qual seria o caminho de conversão em cada caso (por exemplo, *Quarto Individual Clássico* mapeia para *Quarto Individual*, mas *Quarto Queen Superior com Vista para o Jardim ou Cidade* é muito mais difícil de mapear)

2. Podemos adotar uma abordagem de PLN e medir a frequência de certos termos como *Solo*, *Viajante a Negócios* ou *Família com crianças pequenas* conforme se aplicam a cada hotel, e incluir isso no modelo de recomendação.

As tags geralmente (mas nem sempre) são um único campo contendo uma lista de 5 a 6 valores separados por vírgulas, alinhando-se a *Tipo de viagem*, *Tipo de hóspede*, *Tipo de quarto*, *Número de noites* e *Tipo de dispositivo usado para enviar a avaliação*. No entanto, como alguns avaliadores não preenchem cada campo (podem deixar um em branco), os valores nem sempre estão na mesma ordem.

Como exemplo, pegue *Tipo de grupo*. Há 1025 possibilidades únicas neste campo na coluna `Tags`, e, infelizmente, apenas algumas delas se referem a um grupo (algumas são o tipo de quarto, etc.). Se você filtrar apenas as que mencionam família, os resultados contêm muitos tipos de *Quarto Familiar*. Se você incluir o termo *com*, ou seja, contar os valores *Família com*, os resultados são melhores, com mais de 80.000 dos 515.000 resultados contendo a frase "Família com crianças pequenas" ou "Família com crianças mais velhas".

Isso significa que a coluna tags não é completamente inútil para nós, mas será necessário algum trabalho para torná-la útil.

##### Pontuação média do hotel

Há uma série de peculiaridades ou discrepâncias no conjunto de dados que não consigo entender, mas que estão ilustradas aqui para que você esteja ciente delas ao construir seus modelos. Se você descobrir, por favor, nos avise na seção de discussão!

O conjunto de dados possui as seguintes colunas relacionadas à pontuação média e ao número de avaliações:

1. Hotel_Name
2. Additional_Number_of_Scoring
3. Average_Score
4. Total_Number_of_Reviews
5. Reviewer_Score  

O único hotel com mais avaliações neste conjunto de dados é o *Britannia International Hotel Canary Wharf*, com 4789 avaliações de um total de 515.000. Mas se olharmos para o valor de `Total_Number_of_Reviews` para este hotel, ele é 9086. Você poderia supor que há muitas outras pontuações sem avaliações, então talvez devêssemos adicionar o valor da coluna `Additional_Number_of_Scoring`. Esse valor é 2682, e adicioná-lo a 4789 nos dá 7471, o que ainda está 1615 abaixo de `Total_Number_of_Reviews`.

Se você pegar a coluna `Average_Score`, poderia supor que é a média das avaliações no conjunto de dados, mas a descrição do Kaggle é "*Pontuação Média do hotel, calculada com base no comentário mais recente do último ano*". Isso não parece muito útil, mas podemos calcular nossa própria média com base nas pontuações das avaliações no conjunto de dados. Usando o mesmo hotel como exemplo, a pontuação média do hotel é dada como 7.1, mas a pontuação calculada (média das pontuações dos avaliadores *no* conjunto de dados) é 6.8. Isso é próximo, mas não o mesmo valor, e só podemos supor que as pontuações dadas nas avaliações de `Additional_Number_of_Scoring` aumentaram a média para 7.1. Infelizmente, sem uma maneira de testar ou provar essa suposição, é difícil usar ou confiar em `Average_Score`, `Additional_Number_of_Scoring` e `Total_Number_of_Reviews` quando eles são baseados em, ou se referem a, dados que não temos.

Para complicar ainda mais, o hotel com o segundo maior número de avaliações tem uma pontuação média calculada de 8.12, e a `Average_Score` do conjunto de dados é 8.1. Essa pontuação correta é uma coincidência ou o primeiro hotel é uma discrepância?
Na possibilidade de que este hotel seja um caso atípico, e que talvez a maioria dos valores estejam corretos (mas alguns não, por algum motivo), escreveremos um pequeno programa a seguir para explorar os valores no conjunto de dados e determinar o uso correto (ou não uso) dos valores.

> 🚨 Uma nota de cautela
>
> Ao trabalhar com este conjunto de dados, você escreverá código que calcula algo a partir do texto sem precisar ler ou analisar o texto você mesmo. Esta é a essência do PLN (Processamento de Linguagem Natural): interpretar significado ou sentimento sem que um humano precise fazê-lo. No entanto, é possível que você leia algumas das avaliações negativas. Eu recomendo que não o faça, porque não é necessário. Algumas delas são bobas ou irrelevantes, como avaliações negativas de hotéis do tipo "O tempo não estava bom", algo fora do controle do hotel ou de qualquer pessoa. Mas também há um lado sombrio em algumas avaliações. Às vezes, as avaliações negativas são racistas, sexistas ou preconceituosas com relação à idade. Isso é lamentável, mas esperado em um conjunto de dados extraído de um site público. Alguns avaliadores deixam comentários que você pode achar desagradáveis, desconfortáveis ou perturbadores. É melhor deixar o código medir o sentimento do que lê-los você mesmo e se aborrecer. Dito isso, é uma minoria que escreve tais coisas, mas eles existem.

## Exercício - Exploração de Dados
### Carregar os dados

Já chega de examinar os dados visualmente, agora você escreverá algum código e obterá algumas respostas! Esta seção utiliza a biblioteca pandas. Sua primeira tarefa é garantir que você pode carregar e ler os dados CSV. A biblioteca pandas possui um carregador de CSV rápido, e o resultado é colocado em um dataframe, como nas lições anteriores. O CSV que estamos carregando tem mais de meio milhão de linhas, mas apenas 17 colunas. O pandas oferece muitas maneiras poderosas de interagir com um dataframe, incluindo a capacidade de realizar operações em cada linha.

A partir daqui, nesta lição, haverá trechos de código, algumas explicações sobre o código e discussões sobre o que os resultados significam. Use o _notebook.ipynb_ incluído para o seu código.

Vamos começar carregando o arquivo de dados que você usará:

```python
# Load the hotel reviews from CSV
import pandas as pd
import time
# importing time so the start and end time can be used to calculate file loading time
print("Loading data file now, this could take a while depending on file size")
start = time.time()
# df is 'DataFrame' - make sure you downloaded the file to the data folder
df = pd.read_csv('../../data/Hotel_Reviews.csv')
end = time.time()
print("Loading took " + str(round(end - start, 2)) + " seconds")
```

Agora que os dados estão carregados, podemos realizar algumas operações com eles. Mantenha este código no início do seu programa para a próxima parte.

## Explorar os dados

Neste caso, os dados já estão *limpos*, o que significa que estão prontos para serem trabalhados e não possuem caracteres em outros idiomas que possam atrapalhar algoritmos que esperam apenas caracteres em inglês.

✅ Você pode ter que trabalhar com dados que exigem algum processamento inicial para formatá-los antes de aplicar técnicas de PLN, mas não desta vez. Se tivesse que lidar com isso, como você trataria caracteres que não estão em inglês?

Reserve um momento para garantir que, uma vez que os dados estejam carregados, você possa explorá-los com código. É muito fácil querer focar nas colunas `Negative_Review` e `Positive_Review`. Elas estão preenchidas com texto natural para seus algoritmos de PLN processarem. Mas espere! Antes de mergulhar no PLN e no sentimento, você deve seguir o código abaixo para verificar se os valores fornecidos no conjunto de dados correspondem aos valores que você calcula com pandas.

## Operações com Dataframe

A primeira tarefa nesta lição é verificar se as seguintes afirmações estão corretas, escrevendo algum código que examine o dataframe (sem alterá-lo).

> Como em muitas tarefas de programação, existem várias maneiras de completar isso, mas um bom conselho é fazê-lo da maneira mais simples e fácil possível, especialmente se isso for mais fácil de entender quando você voltar a este código no futuro. Com dataframes, há uma API abrangente que frequentemente terá uma maneira eficiente de fazer o que você deseja.

Trate as seguintes perguntas como tarefas de codificação e tente respondê-las sem olhar a solução.

1. Imprima o *shape* do dataframe que você acabou de carregar (o shape é o número de linhas e colunas).
2. Calcule a contagem de frequência para as nacionalidades dos avaliadores:
   1. Quantos valores distintos existem na coluna `Reviewer_Nationality` e quais são eles?
   2. Qual nacionalidade de avaliador é a mais comum no conjunto de dados (imprima o país e o número de avaliações)?
   3. Quais são as 10 nacionalidades mais frequentes e suas contagens de frequência?
3. Qual foi o hotel mais avaliado para cada uma das 10 nacionalidades mais frequentes?
4. Quantas avaliações existem por hotel (contagem de frequência de hotel) no conjunto de dados?
5. Embora haja uma coluna `Average_Score` para cada hotel no conjunto de dados, você também pode calcular uma pontuação média (obtendo a média de todas as pontuações dos avaliadores no conjunto de dados para cada hotel). Adicione uma nova coluna ao seu dataframe com o cabeçalho `Calc_Average_Score` que contenha essa média calculada.
6. Algum hotel tem o mesmo valor (arredondado para 1 casa decimal) em `Average_Score` e `Calc_Average_Score`?
   1. Tente escrever uma função Python que receba uma Series (linha) como argumento e compare os valores, imprimindo uma mensagem quando os valores não forem iguais. Em seguida, use o método `.apply()` para processar cada linha com a função.
7. Calcule e imprima quantas linhas têm valores "No Negative" na coluna `Negative_Review`.
8. Calcule e imprima quantas linhas têm valores "No Positive" na coluna `Positive_Review`.
9. Calcule e imprima quantas linhas têm valores "No Positive" na coluna `Positive_Review` **e** "No Negative" na coluna `Negative_Review`.

### Respostas em código

1. Imprima o *shape* do dataframe que você acabou de carregar (o shape é o número de linhas e colunas).

   ```python
   print("The shape of the data (rows, cols) is " + str(df.shape))
   > The shape of the data (rows, cols) is (515738, 17)
   ```

2. Calcule a contagem de frequência para as nacionalidades dos avaliadores:

   1. Quantos valores distintos existem na coluna `Reviewer_Nationality` e quais são eles?
   2. Qual nacionalidade de avaliador é a mais comum no conjunto de dados (imprima o país e o número de avaliações)?

   ```python
   # value_counts() creates a Series object that has index and values in this case, the country and the frequency they occur in reviewer nationality
   nationality_freq = df["Reviewer_Nationality"].value_counts()
   print("There are " + str(nationality_freq.size) + " different nationalities")
   # print first and last rows of the Series. Change to nationality_freq.to_string() to print all of the data
   print(nationality_freq) 
   
   There are 227 different nationalities
    United Kingdom               245246
    United States of America      35437
    Australia                     21686
    Ireland                       14827
    United Arab Emirates          10235
                                  ...  
    Comoros                           1
    Palau                             1
    Northern Mariana Islands          1
    Cape Verde                        1
    Guinea                            1
   Name: Reviewer_Nationality, Length: 227, dtype: int64
   ```

   3. Quais são as 10 nacionalidades mais frequentes e suas contagens de frequência?

      ```python
      print("The highest frequency reviewer nationality is " + str(nationality_freq.index[0]).strip() + " with " + str(nationality_freq[0]) + " reviews.")
      # Notice there is a leading space on the values, strip() removes that for printing
      # What is the top 10 most common nationalities and their frequencies?
      print("The next 10 highest frequency reviewer nationalities are:")
      print(nationality_freq[1:11].to_string())
      
      The highest frequency reviewer nationality is United Kingdom with 245246 reviews.
      The next 10 highest frequency reviewer nationalities are:
       United States of America     35437
       Australia                    21686
       Ireland                      14827
       United Arab Emirates         10235
       Saudi Arabia                  8951
       Netherlands                   8772
       Switzerland                   8678
       Germany                       7941
       Canada                        7894
       France                        7296
      ```

3. Qual foi o hotel mais avaliado para cada uma das 10 nacionalidades mais frequentes?

   ```python
   # What was the most frequently reviewed hotel for the top 10 nationalities
   # Normally with pandas you will avoid an explicit loop, but wanted to show creating a new dataframe using criteria (don't do this with large amounts of data because it could be very slow)
   for nat in nationality_freq[:10].index:
      # First, extract all the rows that match the criteria into a new dataframe
      nat_df = df[df["Reviewer_Nationality"] == nat]   
      # Now get the hotel freq
      freq = nat_df["Hotel_Name"].value_counts()
      print("The most reviewed hotel for " + str(nat).strip() + " was " + str(freq.index[0]) + " with " + str(freq[0]) + " reviews.") 
      
   The most reviewed hotel for United Kingdom was Britannia International Hotel Canary Wharf with 3833 reviews.
   The most reviewed hotel for United States of America was Hotel Esther a with 423 reviews.
   The most reviewed hotel for Australia was Park Plaza Westminster Bridge London with 167 reviews.
   The most reviewed hotel for Ireland was Copthorne Tara Hotel London Kensington with 239 reviews.
   The most reviewed hotel for United Arab Emirates was Millennium Hotel London Knightsbridge with 129 reviews.
   The most reviewed hotel for Saudi Arabia was The Cumberland A Guoman Hotel with 142 reviews.
   The most reviewed hotel for Netherlands was Jaz Amsterdam with 97 reviews.
   The most reviewed hotel for Switzerland was Hotel Da Vinci with 97 reviews.
   The most reviewed hotel for Germany was Hotel Da Vinci with 86 reviews.
   The most reviewed hotel for Canada was St James Court A Taj Hotel London with 61 reviews.
   ```

4. Quantas avaliações existem por hotel (contagem de frequência de hotel) no conjunto de dados?

   ```python
   # First create a new dataframe based on the old one, removing the uneeded columns
   hotel_freq_df = df.drop(["Hotel_Address", "Additional_Number_of_Scoring", "Review_Date", "Average_Score", "Reviewer_Nationality", "Negative_Review", "Review_Total_Negative_Word_Counts", "Positive_Review", "Review_Total_Positive_Word_Counts", "Total_Number_of_Reviews_Reviewer_Has_Given", "Reviewer_Score", "Tags", "days_since_review", "lat", "lng"], axis = 1)
   
   # Group the rows by Hotel_Name, count them and put the result in a new column Total_Reviews_Found
   hotel_freq_df['Total_Reviews_Found'] = hotel_freq_df.groupby('Hotel_Name').transform('count')
   
   # Get rid of all the duplicated rows
   hotel_freq_df = hotel_freq_df.drop_duplicates(subset = ["Hotel_Name"])
   display(hotel_freq_df) 
   ```
   |                 Hotel_Name                 | Total_Number_of_Reviews | Total_Reviews_Found |
   | :----------------------------------------: | :---------------------: | :-----------------: |
   | Britannia International Hotel Canary Wharf |          9086           |        4789         |
   |    Park Plaza Westminster Bridge London    |          12158          |        4169         |
   |   Copthorne Tara Hotel London Kensington   |          7105           |        3578         |
   |                    ...                     |           ...           |         ...         |
   |       Mercure Paris Porte d Orleans        |           110           |         10          |
   |                Hotel Wagner                |           135           |         10          |
   |            Hotel Gallitzinberg             |           173           |          8          |

   Você pode notar que os resultados *contados no conjunto de dados* não correspondem ao valor em `Total_Number_of_Reviews`. Não está claro se este valor no conjunto de dados representa o número total de avaliações que o hotel teve, mas nem todas foram extraídas, ou algum outro cálculo. `Total_Number_of_Reviews` não é usado no modelo devido a essa falta de clareza.

5. Embora haja uma coluna `Average_Score` para cada hotel no conjunto de dados, você também pode calcular uma pontuação média (obtendo a média de todas as pontuações dos avaliadores no conjunto de dados para cada hotel). Adicione uma nova coluna ao seu dataframe com o cabeçalho `Calc_Average_Score` que contenha essa média calculada. Imprima as colunas `Hotel_Name`, `Average_Score` e `Calc_Average_Score`.

   ```python
   # define a function that takes a row and performs some calculation with it
   def get_difference_review_avg(row):
     return row["Average_Score"] - row["Calc_Average_Score"]
   
   # 'mean' is mathematical word for 'average'
   df['Calc_Average_Score'] = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
   
   # Add a new column with the difference between the two average scores
   df["Average_Score_Difference"] = df.apply(get_difference_review_avg, axis = 1)
   
   # Create a df without all the duplicates of Hotel_Name (so only 1 row per hotel)
   review_scores_df = df.drop_duplicates(subset = ["Hotel_Name"])
   
   # Sort the dataframe to find the lowest and highest average score difference
   review_scores_df = review_scores_df.sort_values(by=["Average_Score_Difference"])
   
   display(review_scores_df[["Average_Score_Difference", "Average_Score", "Calc_Average_Score", "Hotel_Name"]])
   ```

   Você também pode se perguntar sobre o valor de `Average_Score` e por que ele às vezes é diferente da pontuação média calculada. Como não podemos saber por que alguns valores correspondem, mas outros têm uma diferença, é mais seguro, neste caso, usar as pontuações dos avaliadores que temos para calcular a média nós mesmos. Dito isso, as diferenças geralmente são muito pequenas. Aqui estão os hotéis com a maior diferença entre a média do conjunto de dados e a média calculada:

   | Average_Score_Difference | Average_Score | Calc_Average_Score |                                  Hotel_Name |
   | :----------------------: | :-----------: | :----------------: | ------------------------------------------: |
   |           -0.8           |      7.7      |        8.5         |                  Best Western Hotel Astoria |
   |           -0.7           |      8.8      |        9.5         | Hotel Stendhal Place Vend me Paris MGallery |
   |           -0.7           |      7.5      |        8.2         |               Mercure Paris Porte d Orleans |
   |           -0.7           |      7.9      |        8.6         |             Renaissance Paris Vendome Hotel |
   |           -0.5           |      7.0      |        7.5         |                         Hotel Royal Elys es |
   |           ...            |      ...      |        ...         |                                         ... |
   |           0.7            |      7.5      |        6.8         |     Mercure Paris Op ra Faubourg Montmartre |
   |           0.8            |      7.1      |        6.3         |      Holiday Inn Paris Montparnasse Pasteur |
   |           0.9            |      6.8      |        5.9         |                               Villa Eugenie |
   |           0.9            |      8.6      |        7.7         |   MARQUIS Faubourg St Honor Relais Ch teaux |
   |           1.3            |      7.2      |        5.9         |                          Kube Hotel Ice Bar |

   Com apenas 1 hotel tendo uma diferença de pontuação maior que 1, isso significa que provavelmente podemos ignorar a diferença e usar a pontuação média calculada.

6. Calcule e imprima quantas linhas têm valores "No Negative" na coluna `Negative_Review`.

7. Calcule e imprima quantas linhas têm valores "No Positive" na coluna `Positive_Review`.

8. Calcule e imprima quantas linhas têm valores "No Positive" na coluna `Positive_Review` **e** "No Negative" na coluna `Negative_Review`.

   ```python
   # with lambdas:
   start = time.time()
   no_negative_reviews = df.apply(lambda x: True if x['Negative_Review'] == "No Negative" else False , axis=1)
   print("Number of No Negative reviews: " + str(len(no_negative_reviews[no_negative_reviews == True].index)))
   
   no_positive_reviews = df.apply(lambda x: True if x['Positive_Review'] == "No Positive" else False , axis=1)
   print("Number of No Positive reviews: " + str(len(no_positive_reviews[no_positive_reviews == True].index)))
   
   both_no_reviews = df.apply(lambda x: True if x['Negative_Review'] == "No Negative" and x['Positive_Review'] == "No Positive" else False , axis=1)
   print("Number of both No Negative and No Positive reviews: " + str(len(both_no_reviews[both_no_reviews == True].index)))
   end = time.time()
   print("Lambdas took " + str(round(end - start, 2)) + " seconds")
   
   Number of No Negative reviews: 127890
   Number of No Positive reviews: 35946
   Number of both No Negative and No Positive reviews: 127
   Lambdas took 9.64 seconds
   ```

## Outra maneira

Outra maneira de contar itens sem Lambdas, e usar sum para contar as linhas:

   ```python
   # without lambdas (using a mixture of notations to show you can use both)
   start = time.time()
   no_negative_reviews = sum(df.Negative_Review == "No Negative")
   print("Number of No Negative reviews: " + str(no_negative_reviews))
   
   no_positive_reviews = sum(df["Positive_Review"] == "No Positive")
   print("Number of No Positive reviews: " + str(no_positive_reviews))
   
   both_no_reviews = sum((df.Negative_Review == "No Negative") & (df.Positive_Review == "No Positive"))
   print("Number of both No Negative and No Positive reviews: " + str(both_no_reviews))
   
   end = time.time()
   print("Sum took " + str(round(end - start, 2)) + " seconds")
   
   Number of No Negative reviews: 127890
   Number of No Positive reviews: 35946
   Number of both No Negative and No Positive reviews: 127
   Sum took 0.19 seconds
   ```

   Você pode ter notado que há 127 linhas que possuem "No Negative" e "No Positive" como valores nas colunas `Negative_Review` e `Positive_Review`, respectivamente. Isso significa que o avaliador deu ao hotel uma pontuação numérica, mas optou por não escrever uma avaliação positiva ou negativa. Felizmente, isso é uma pequena quantidade de linhas (127 de 515738, ou 0,02%), então provavelmente não distorcerá nosso modelo ou resultados em nenhuma direção específica, mas você pode não ter esperado que um conjunto de dados de avaliações tivesse linhas sem avaliações. Portanto, vale a pena explorar os dados para descobrir linhas como essa.

Agora que você explorou o conjunto de dados, na próxima lição você filtrará os dados e adicionará alguma análise de sentimento.

---
## 🚀Desafio

Esta lição demonstra, como vimos em lições anteriores, o quão importante é entender seus dados e suas peculiaridades antes de realizar operações sobre eles. Dados baseados em texto, em particular, exigem uma análise cuidadosa. Explore vários conjuntos de dados ricos em texto e veja se você consegue descobrir áreas que poderiam introduzir viés ou sentimento distorcido em um modelo.

## [Questionário pós-aula](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/38/)

## Revisão e Autoestudo

Faça [este Caminho de Aprendizado sobre PLN](https://docs.microsoft.com/learn/paths/explore-natural-language-processing/?WT.mc_id=academic-77952-leestott) para descobrir ferramentas para experimentar ao construir modelos baseados em fala e texto.

## Tarefa

[NLTK](assignment.md)

---

**Aviso Legal**:  
Este documento foi traduzido utilizando o serviço de tradução por IA [Co-op Translator](https://github.com/Azure/co-op-translator). Embora nos esforcemos para garantir a precisão, esteja ciente de que traduções automatizadas podem conter erros ou imprecisões. O documento original em seu idioma nativo deve ser considerado a fonte autoritativa. Para informações críticas, recomenda-se a tradução profissional realizada por humanos. Não nos responsabilizamos por quaisquer mal-entendidos ou interpretações equivocadas decorrentes do uso desta tradução.