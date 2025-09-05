<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "8d32dadeda93c6fb5c43619854882ab1",
  "translation_date": "2025-09-05T08:50:55+00:00",
  "source_file": "6-NLP/4-Hotel-Reviews-1/README.md",
  "language_code": "pt"
}
-->
# Análise de sentimentos com avaliações de hotéis - processamento de dados

Nesta seção, vais utilizar as técnicas das lições anteriores para realizar uma análise exploratória de dados num conjunto de dados extenso. Depois de compreender bem a utilidade das várias colunas, vais aprender:

- como remover as colunas desnecessárias
- como calcular novos dados com base nas colunas existentes
- como guardar o conjunto de dados resultante para uso no desafio final

## [Questionário pré-aula](https://ff-quizzes.netlify.app/en/ml/)

### Introdução

Até agora aprendeste que os dados textuais são bastante diferentes dos dados numéricos. Quando o texto é escrito ou falado por humanos, pode ser analisado para encontrar padrões, frequências, sentimentos e significados. Esta lição leva-te a um conjunto de dados real com um desafio real: **[515K Hotel Reviews Data in Europe](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)**, que inclui uma [licença CC0: Domínio Público](https://creativecommons.org/publicdomain/zero/1.0/). Os dados foram extraídos de Booking.com a partir de fontes públicas. O criador do conjunto de dados foi Jiashen Liu.

### Preparação

Vais precisar de:

* Capacidade de executar notebooks .ipynb usando Python 3
* pandas
* NLTK, [que deves instalar localmente](https://www.nltk.org/install.html)
* O conjunto de dados disponível no Kaggle [515K Hotel Reviews Data in Europe](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe). Tem cerca de 230 MB descompactado. Faz o download para a pasta raiz `/data` associada a estas lições de NLP.

## Análise exploratória de dados

Este desafio assume que estás a construir um bot de recomendação de hotéis usando análise de sentimentos e pontuações de avaliações de hóspedes. O conjunto de dados que vais utilizar inclui avaliações de 1493 hotéis diferentes em 6 cidades.

Usando Python, um conjunto de dados de avaliações de hotéis e a análise de sentimentos do NLTK, podes descobrir:

* Quais são as palavras e frases mais frequentemente usadas nas avaliações?
* As *tags* oficiais que descrevem um hotel correlacionam-se com as pontuações das avaliações (por exemplo, há mais avaliações negativas para um hotel específico por *Famílias com crianças pequenas* do que por *Viajantes solitários*, talvez indicando que é mais adequado para *Viajantes solitários*)?
* As pontuações de sentimentos do NLTK "concordam" com a pontuação numérica do avaliador?

#### Conjunto de dados

Vamos explorar o conjunto de dados que descarregaste e guardaste localmente. Abre o ficheiro num editor como o VS Code ou até mesmo no Excel.

Os cabeçalhos no conjunto de dados são os seguintes:

*Hotel_Address, Additional_Number_of_Scoring, Review_Date, Average_Score, Hotel_Name, Reviewer_Nationality, Negative_Review, Review_Total_Negative_Word_Counts, Total_Number_of_Reviews, Positive_Review, Review_Total_Positive_Word_Counts, Total_Number_of_Reviews_Reviewer_Has_Given, Reviewer_Score, Tags, days_since_review, lat, lng*

Aqui estão agrupados de forma que possam ser mais fáceis de examinar: 
##### Colunas do hotel

* `Hotel_Name`, `Hotel_Address`, `lat` (latitude), `lng` (longitude)
  * Usando *lat* e *lng* podes criar um mapa com Python mostrando as localizações dos hotéis (talvez codificado por cores para avaliações negativas e positivas)
  * Hotel_Address não parece ser muito útil para nós, e provavelmente será substituído por um país para facilitar a ordenação e pesquisa

**Colunas de meta-avaliação do hotel**

* `Average_Score`
  * De acordo com o criador do conjunto de dados, esta coluna é a *Pontuação média do hotel, calculada com base no último comentário do último ano*. Parece uma forma incomum de calcular a pontuação, mas são os dados extraídos, então podemos aceitá-los como estão por agora.
  
  ✅ Com base nas outras colunas deste conjunto de dados, consegues pensar noutra forma de calcular a pontuação média?

* `Total_Number_of_Reviews`
  * O número total de avaliações que este hotel recebeu - não está claro (sem escrever algum código) se isto se refere às avaliações no conjunto de dados.
* `Additional_Number_of_Scoring`
  * Isto significa que foi dada uma pontuação de avaliação, mas o avaliador não escreveu uma avaliação positiva ou negativa.

**Colunas de avaliação**

- `Reviewer_Score`
  - Este é um valor numérico com, no máximo, 1 casa decimal entre os valores mínimos e máximos de 2.5 e 10
  - Não é explicado porque 2.5 é a pontuação mínima possível
- `Negative_Review`
  - Se um avaliador não escreveu nada, este campo terá "**No Negative**"
  - Nota que um avaliador pode escrever uma avaliação positiva na coluna de avaliação negativa (por exemplo, "não há nada de mau neste hotel")
- `Review_Total_Negative_Word_Counts`
  - Contagens mais altas de palavras negativas indicam uma pontuação mais baixa (sem verificar a sentimentalidade)
- `Positive_Review`
  - Se um avaliador não escreveu nada, este campo terá "**No Positive**"
  - Nota que um avaliador pode escrever uma avaliação negativa na coluna de avaliação positiva (por exemplo, "não há nada de bom neste hotel")
- `Review_Total_Positive_Word_Counts`
  - Contagens mais altas de palavras positivas indicam uma pontuação mais alta (sem verificar a sentimentalidade)
- `Review_Date` e `days_since_review`
  - Pode ser aplicada uma medida de frescura ou desatualização a uma avaliação (avaliações mais antigas podem não ser tão precisas quanto as mais recentes porque a gestão do hotel mudou, ou foram feitas renovações, ou foi adicionada uma piscina, etc.)
- `Tags`
  - Estas são descritores curtos que um avaliador pode selecionar para descrever o tipo de hóspede que era (por exemplo, solitário ou família), o tipo de quarto que teve, a duração da estadia e como a avaliação foi submetida.
  - Infelizmente, usar estas tags é problemático, verifica a seção abaixo que discute a sua utilidade.

**Colunas do avaliador**

- `Total_Number_of_Reviews_Reviewer_Has_Given`
  - Isto pode ser um fator num modelo de recomendação, por exemplo, se conseguires determinar que avaliadores mais prolíficos, com centenas de avaliações, eram mais propensos a serem negativos do que positivos. No entanto, o avaliador de qualquer avaliação específica não é identificado com um código único e, portanto, não pode ser ligado a um conjunto de avaliações. Existem 30 avaliadores com 100 ou mais avaliações, mas é difícil ver como isto pode ajudar o modelo de recomendação.
- `Reviewer_Nationality`
  - Algumas pessoas podem pensar que certas nacionalidades são mais propensas a dar uma avaliação positiva ou negativa devido a uma inclinação nacional. Tem cuidado ao construir tais visões anedóticas nos teus modelos. Estes são estereótipos nacionais (e às vezes raciais), e cada avaliador foi um indivíduo que escreveu uma avaliação com base na sua experiência. Esta pode ter sido filtrada por muitas lentes, como as suas estadias anteriores em hotéis, a distância percorrida e o seu temperamento pessoal. Pensar que a nacionalidade foi a razão para uma pontuação de avaliação é difícil de justificar.

##### Exemplos

| Average  Score | Total Number   Reviews | Reviewer   Score | Negative <br />Review                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Positive   Review                 | Tags                                                                                      |
| -------------- | ---------------------- | ---------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- | ----------------------------------------------------------------------------------------- |
| 7.8            | 1945                   | 2.5              | Este não é atualmente um hotel, mas um local de construção. Fui aterrorizado desde cedo pela manhã e durante todo o dia com ruídos de construção inaceitáveis enquanto descansava após uma longa viagem e trabalhava no quarto. Pessoas estavam a trabalhar o dia todo, ou seja, com martelos pneumáticos nos quartos adjacentes. Pedi para mudar de quarto, mas não havia nenhum quarto silencioso disponível. Para piorar, fui cobrado em excesso. Fiz o check-out à noite, pois tinha um voo muito cedo e recebi uma fatura apropriada. Um dia depois, o hotel fez outra cobrança sem o meu consentimento, acima do preço reservado. É um lugar terrível. Não te castigues reservando aqui. | Nada. Lugar terrível. Fica longe. | Viagem de negócios. Casal. Quarto duplo standard. Ficou 2 noites. |

Como podes ver, este hóspede não teve uma estadia feliz neste hotel. O hotel tem uma boa pontuação média de 7.8 e 1945 avaliações, mas este avaliador deu-lhe 2.5 e escreveu 115 palavras sobre como a sua estadia foi negativa. Se não escreveu nada na coluna Positive_Review, podes deduzir que não houve nada positivo, mas ainda assim escreveu 7 palavras de aviso. Se apenas contássemos palavras em vez do significado ou sentimento das palavras, poderíamos ter uma visão distorcida da intenção do avaliador. Estranhamente, a sua pontuação de 2.5 é confusa, porque se a estadia no hotel foi tão má, por que dar qualquer ponto? Investigando o conjunto de dados de perto, vais ver que a pontuação mínima possível é 2.5, não 0. A pontuação máxima possível é 10.

##### Tags

Como mencionado acima, à primeira vista, a ideia de usar `Tags` para categorizar os dados faz sentido. Infelizmente, estas tags não são padronizadas, o que significa que num dado hotel, as opções podem ser *Single room*, *Twin room* e *Double room*, mas no próximo hotel, são *Deluxe Single Room*, *Classic Queen Room* e *Executive King Room*. Podem ser a mesma coisa, mas há tantas variações que a escolha torna-se:

1. Tentar alterar todos os termos para um único padrão, o que é muito difícil, porque não está claro qual seria o caminho de conversão em cada caso (por exemplo, *Classic single room* mapeia para *Single room*, mas *Superior Queen Room with Courtyard Garden or City View* é muito mais difícil de mapear)

1. Podemos adotar uma abordagem de NLP e medir a frequência de certos termos como *Solo*, *Business Traveller* ou *Family with young kids* conforme se aplicam a cada hotel, e incluir isso na recomendação  

As tags geralmente (mas nem sempre) são um único campo contendo uma lista de 5 a 6 valores separados por vírgulas alinhados a *Tipo de viagem*, *Tipo de hóspedes*, *Tipo de quarto*, *Número de noites* e *Tipo de dispositivo em que a avaliação foi submetida*. No entanto, como alguns avaliadores não preenchem cada campo (podem deixar um em branco), os valores nem sempre estão na mesma ordem.

Como exemplo, considera *Tipo de grupo*. Existem 1025 possibilidades únicas neste campo na coluna `Tags`, e infelizmente apenas algumas delas referem-se a um grupo (algumas são o tipo de quarto, etc.). Se filtrares apenas as que mencionam família, os resultados contêm muitos resultados do tipo *Family room*. Se incluíres o termo *with*, ou seja, contares os valores *Family with*, os resultados são melhores, com mais de 80.000 dos 515.000 resultados contendo a frase "Family with young children" ou "Family with older children".

Isto significa que a coluna tags não é completamente inútil para nós, mas será necessário algum trabalho para torná-la útil.

##### Pontuação média do hotel

Existem algumas peculiaridades ou discrepâncias no conjunto de dados que não consigo entender, mas são ilustradas aqui para que estejas ciente delas ao construir os teus modelos. Se conseguires descobrir, por favor, avisa-nos na seção de discussão!

O conjunto de dados tem as seguintes colunas relacionadas à pontuação média e ao número de avaliações:

1. Hotel_Name
2. Additional_Number_of_Scoring
3. Average_Score
4. Total_Number_of_Reviews
5. Reviewer_Score  

O único hotel com mais avaliações neste conjunto de dados é *Britannia International Hotel Canary Wharf* com 4789 avaliações de um total de 515.000. Mas se olharmos para o valor `Total_Number_of_Reviews` deste hotel, é 9086. Podes deduzir que há muitas mais pontuações sem avaliações, então talvez devêssemos adicionar o valor da coluna `Additional_Number_of_Scoring`. Esse valor é 2682, e adicioná-lo a 4789 dá-nos 7471, que ainda está 1615 abaixo do `Total_Number_of_Reviews`.

Se considerares a coluna `Average_Score`, podes deduzir que é a média das avaliações no conjunto de dados, mas a descrição do Kaggle é "*Pontuação média do hotel, calculada com base no último comentário do último ano*". Isso não parece muito útil, mas podemos calcular a nossa própria média com base nas pontuações das avaliações no conjunto de dados. Usando o mesmo hotel como exemplo, a pontuação média do hotel é dada como 7.1, mas a pontuação calculada (pontuação média do avaliador *no* conjunto de dados) é 6.8. Isto é próximo, mas não o mesmo valor, e só podemos supor que as pontuações dadas nas avaliações `Additional_Number_of_Scoring` aumentaram a média para 7.1. Infelizmente, sem forma de testar ou provar essa suposição, é difícil usar ou confiar em `Average_Score`, `Additional_Number_of_Scoring` e `Total_Number_of_Reviews` quando são baseados em, ou referem-se a, dados que não temos.

Para complicar ainda mais, o hotel com o segundo maior número de avaliações tem uma pontuação média calculada de 8.12 e a `Average_Score` do conjunto de dados é 8.1. Esta pontuação correta é uma coincidência ou o primeiro hotel é uma discrepância?

Na possibilidade de que estes hotéis possam ser um caso isolado, e que talvez a maioria dos valores se alinhem (mas alguns não por alguma razão), vamos escrever um pequeno programa a seguir para explorar os valores no conjunto de dados e determinar o uso correto (ou não uso) dos valores.
> 🚨 Uma nota de precaução  
>  
> Ao trabalhar com este conjunto de dados, irá escrever código que calcula algo a partir do texto sem ter de ler ou analisar o texto diretamente. Esta é a essência do NLP, interpretar significado ou sentimento sem que seja necessário um humano fazê-lo. No entanto, é possível que acabe por ler algumas das críticas negativas. Recomendo que não o faça, porque não é necessário. Algumas delas são disparatadas ou irrelevantes, como críticas negativas a hotéis do tipo "O tempo não estava bom", algo que está fora do controlo do hotel, ou de qualquer pessoa. Mas há também um lado sombrio em algumas críticas. Por vezes, as críticas negativas são racistas, sexistas ou preconceituosas em relação à idade. Isto é lamentável, mas esperado num conjunto de dados extraído de um site público. Alguns utilizadores deixam críticas que podem ser consideradas de mau gosto, desconfortáveis ou perturbadoras. É melhor deixar que o código avalie o sentimento do que lê-las e ficar incomodado. Dito isto, é uma minoria que escreve tais coisas, mas elas existem na mesma.
## Exercício - Exploração de dados
### Carregar os dados

Já chega de examinar os dados visualmente, agora vais escrever algum código e obter respostas! Esta secção utiliza a biblioteca pandas. A tua primeira tarefa é garantir que consegues carregar e ler os dados CSV. A biblioteca pandas tem um carregador rápido de CSV, e o resultado é colocado num dataframe, como nas lições anteriores. O CSV que estamos a carregar tem mais de meio milhão de linhas, mas apenas 17 colunas. O pandas oferece muitas formas poderosas de interagir com um dataframe, incluindo a capacidade de realizar operações em cada linha.

A partir daqui, nesta lição, haverá trechos de código e algumas explicações sobre o código, bem como discussões sobre o significado dos resultados. Usa o _notebook.ipynb_ incluído para o teu código.

Vamos começar por carregar o ficheiro de dados que vais utilizar:

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

Agora que os dados estão carregados, podemos realizar algumas operações sobre eles. Mantém este código no topo do teu programa para a próxima parte.

## Explorar os dados

Neste caso, os dados já estão *limpos*, o que significa que estão prontos para trabalhar e não têm caracteres noutras línguas que possam causar problemas a algoritmos que esperam apenas caracteres em inglês.

✅ Poderás ter de trabalhar com dados que requerem algum processamento inicial para os formatar antes de aplicar técnicas de NLP, mas não desta vez. Se tivesses de o fazer, como lidarias com caracteres não ingleses?

Tira um momento para garantir que, uma vez carregados os dados, consegues explorá-los com código. É muito fácil querer focar nas colunas `Negative_Review` e `Positive_Review`. Elas estão preenchidas com texto natural para os teus algoritmos de NLP processarem. Mas espera! Antes de mergulhares no NLP e na análise de sentimentos, deves seguir o código abaixo para verificar se os valores fornecidos no conjunto de dados correspondem aos valores que calculas com pandas.

## Operações no dataframe

A primeira tarefa nesta lição é verificar se as seguintes afirmações estão corretas, escrevendo algum código que examine o dataframe (sem o alterar).

> Tal como em muitas tarefas de programação, há várias formas de completar isto, mas um bom conselho é fazê-lo da forma mais simples e fácil possível, especialmente se for mais fácil de entender quando voltares a este código no futuro. Com dataframes, há uma API abrangente que frequentemente terá uma forma eficiente de fazer o que precisas.

Trata as seguintes perguntas como tarefas de codificação e tenta respondê-las sem olhar para a solução.

1. Imprime a *forma* do dataframe que acabaste de carregar (a forma é o número de linhas e colunas).
2. Calcula a contagem de frequência para as nacionalidades dos revisores:
   1. Quantos valores distintos existem na coluna `Reviewer_Nationality` e quais são eles?
   2. Qual é a nacionalidade de revisor mais comum no conjunto de dados (imprime o país e o número de revisões)?
   3. Quais são as 10 nacionalidades mais frequentes e a sua contagem de frequência?
3. Qual foi o hotel mais revisado para cada uma das 10 nacionalidades de revisores mais frequentes?
4. Quantas revisões existem por hotel (contagem de frequência de hotel) no conjunto de dados?
5. Embora exista uma coluna `Average_Score` para cada hotel no conjunto de dados, também podes calcular uma pontuação média (obtendo a média de todas as pontuações dos revisores no conjunto de dados para cada hotel). Adiciona uma nova coluna ao teu dataframe com o cabeçalho `Calc_Average_Score` que contém essa média calculada.
6. Algum hotel tem o mesmo valor (arredondado a 1 casa decimal) para `Average_Score` e `Calc_Average_Score`?
   1. Tenta escrever uma função Python que receba uma Série (linha) como argumento e compare os valores, imprimindo uma mensagem quando os valores não forem iguais. Depois usa o método `.apply()` para processar cada linha com a função.
7. Calcula e imprime quantas linhas têm valores "No Negative" na coluna `Negative_Review`.
8. Calcula e imprime quantas linhas têm valores "No Positive" na coluna `Positive_Review`.
9. Calcula e imprime quantas linhas têm valores "No Positive" na coluna `Positive_Review` **e** valores "No Negative" na coluna `Negative_Review`.

### Respostas em código

1. Imprime a *forma* do dataframe que acabaste de carregar (a forma é o número de linhas e colunas).

   ```python
   print("The shape of the data (rows, cols) is " + str(df.shape))
   > The shape of the data (rows, cols) is (515738, 17)
   ```

2. Calcula a contagem de frequência para as nacionalidades dos revisores:

   1. Quantos valores distintos existem na coluna `Reviewer_Nationality` e quais são eles?
   2. Qual é a nacionalidade de revisor mais comum no conjunto de dados (imprime o país e o número de revisões)?

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

   3. Quais são as 10 nacionalidades mais frequentes e a sua contagem de frequência?

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

3. Qual foi o hotel mais revisado para cada uma das 10 nacionalidades de revisores mais frequentes?

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

4. Quantas revisões existem por hotel (contagem de frequência de hotel) no conjunto de dados?

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
   
   Podes notar que os resultados *contados no conjunto de dados* não correspondem ao valor em `Total_Number_of_Reviews`. Não está claro se este valor no conjunto de dados representa o número total de revisões que o hotel teve, mas nem todas foram recolhidas, ou algum outro cálculo. `Total_Number_of_Reviews` não é usado no modelo devido a esta falta de clareza.

5. Embora exista uma coluna `Average_Score` para cada hotel no conjunto de dados, também podes calcular uma pontuação média (obtendo a média de todas as pontuações dos revisores no conjunto de dados para cada hotel). Adiciona uma nova coluna ao teu dataframe com o cabeçalho `Calc_Average_Score` que contém essa média calculada. Imprime as colunas `Hotel_Name`, `Average_Score` e `Calc_Average_Score`.

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

   Podes também questionar-te sobre o valor de `Average_Score` e porque é que às vezes é diferente da pontuação média calculada. Como não podemos saber porque é que alguns dos valores coincidem, mas outros têm uma diferença, é mais seguro, neste caso, usar as pontuações das revisões que temos para calcular a média nós mesmos. Dito isto, as diferenças são geralmente muito pequenas. Aqui estão os hotéis com a maior diferença entre a média do conjunto de dados e a média calculada:

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

   Com apenas 1 hotel tendo uma diferença de pontuação maior que 1, significa que provavelmente podemos ignorar a diferença e usar a pontuação média calculada.

6. Calcula e imprime quantas linhas têm valores "No Negative" na coluna `Negative_Review`.

7. Calcula e imprime quantas linhas têm valores "No Positive" na coluna `Positive_Review`.

8. Calcula e imprime quantas linhas têm valores "No Positive" na coluna `Positive_Review` **e** valores "No Negative" na coluna `Negative_Review`.

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

## Outra forma

Outra forma de contar itens sem Lambdas, e usar sum para contar as linhas:

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

   Podes ter notado que há 127 linhas que têm ambos os valores "No Negative" e "No Positive" nas colunas `Negative_Review` e `Positive_Review`, respetivamente. Isso significa que o revisor deu ao hotel uma pontuação numérica, mas optou por não escrever uma revisão positiva ou negativa. Felizmente, isto é uma pequena quantidade de linhas (127 de 515738, ou 0,02%), então provavelmente não vai enviesar o nosso modelo ou resultados numa direção particular, mas talvez não esperasses que um conjunto de dados de revisões tivesse linhas sem revisões, por isso vale a pena explorar os dados para descobrir linhas como esta.

Agora que exploraste o conjunto de dados, na próxima lição vais filtrar os dados e adicionar alguma análise de sentimentos.

---
## 🚀Desafio

Esta lição demonstra, como vimos em lições anteriores, quão importante é entender os teus dados e as suas peculiaridades antes de realizar operações sobre eles. Dados baseados em texto, em particular, requerem uma análise cuidadosa. Explora vários conjuntos de dados ricos em texto e vê se consegues descobrir áreas que poderiam introduzir enviesamento ou sentimentos distorcidos num modelo.

## [Questionário pós-aula](https://ff-quizzes.netlify.app/en/ml/)

## Revisão & Autoestudo

Segue [este percurso de aprendizagem sobre NLP](https://docs.microsoft.com/learn/paths/explore-natural-language-processing/?WT.mc_id=academic-77952-leestott) para descobrir ferramentas que podes experimentar ao construir modelos baseados em fala e texto.

## Tarefa 

[NLTK](assignment.md)

---

**Aviso Legal**:  
Este documento foi traduzido utilizando o serviço de tradução por IA [Co-op Translator](https://github.com/Azure/co-op-translator). Embora nos esforcemos para garantir a precisão, esteja ciente de que traduções automáticas podem conter erros ou imprecisões. O documento original no seu idioma nativo deve ser considerado a fonte oficial. Para informações críticas, recomenda-se uma tradução profissional realizada por humanos. Não nos responsabilizamos por quaisquer mal-entendidos ou interpretações incorretas resultantes do uso desta tradução.