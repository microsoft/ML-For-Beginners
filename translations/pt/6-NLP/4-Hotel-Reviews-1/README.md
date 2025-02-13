# Análise de sentimento com avaliações de hotéis - processando os dados

Nesta seção, você usará as técnicas das lições anteriores para realizar uma análise exploratória de dados de um grande conjunto de dados. Assim que você tiver uma boa compreensão da utilidade das várias colunas, você aprenderá:

- como remover as colunas desnecessárias
- como calcular novos dados com base nas colunas existentes
- como salvar o conjunto de dados resultante para uso no desafio final

## [Quiz pré-aula](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/37/)

### Introdução

Até agora, você aprendeu que os dados textuais são bastante diferentes dos tipos de dados numéricos. Se é um texto escrito ou falado por um humano, ele pode ser analisado para encontrar padrões e frequências, sentimentos e significados. Esta lição o leva a um conjunto de dados real com um desafio real: **[Dados de Avaliações de Hotéis de 515K na Europa](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)** e inclui uma [licença CC0: Domínio Público](https://creativecommons.org/publicdomain/zero/1.0/). Os dados foram extraídos do Booking.com de fontes públicas. O criador do conjunto de dados foi Jiashen Liu.

### Preparação

Você precisará de:

* A capacidade de executar notebooks .ipynb usando Python 3
* pandas
* NLTK, [que você deve instalar localmente](https://www.nltk.org/install.html)
* O conjunto de dados que está disponível no Kaggle [Dados de Avaliações de Hotéis de 515K na Europa](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe). Ele tem cerca de 230 MB descompactado. Baixe-o para a pasta raiz `/data` associada a estas lições de NLP.

## Análise exploratória de dados

Este desafio assume que você está construindo um bot de recomendação de hotéis usando análise de sentimento e pontuações de avaliações de hóspedes. O conjunto de dados que você usará inclui avaliações de 1493 hotéis diferentes em 6 cidades.

Usando Python, um conjunto de dados de avaliações de hotéis e a análise de sentimento do NLTK, você pode descobrir:

* Quais são as palavras e frases mais frequentemente usadas nas avaliações?
* Os *tags* oficiais que descrevem um hotel estão correlacionados com as pontuações das avaliações (por exemplo, as avaliações mais negativas para um determinado hotel são para *Família com crianças pequenas* em comparação com *Viajante solitário*, talvez indicando que é melhor para *Viajantes solitários*)?
* As pontuações de sentimento do NLTK 'concordam' com a pontuação numérica do avaliador do hotel?

#### Conjunto de dados

Vamos explorar o conjunto de dados que você baixou e salvou localmente. Abra o arquivo em um editor como o VS Code ou até mesmo o Excel.

Os cabeçalhos no conjunto de dados são os seguintes:

*Hotel_Address, Additional_Number_of_Scoring, Review_Date, Average_Score, Hotel_Name, Reviewer_Nationality, Negative_Review, Review_Total_Negative_Word_Counts, Total_Number_of_Reviews, Positive_Review, Review_Total_Positive_Word_Counts, Total_Number_of_Reviews_Reviewer_Has_Given, Reviewer_Score, Tags, days_since_review, lat, lng*

Aqui estão agrupados de uma maneira que pode ser mais fácil de examinar: 
##### Colunas do hotel

* `Hotel_Name`, `Hotel_Address`, `lat` (latitude), `lng` (longitude)
  * Usando *lat* e *lng* você poderia plotar um mapa com Python mostrando as localizações dos hotéis (talvez colorido para avaliações negativas e positivas)
  * Hotel_Address não é obviamente útil para nós, e provavelmente o substituiremos por um país para facilitar a classificação e a busca

**Colunas de meta-avaliação do hotel**

* `Average_Score`
  * De acordo com o criador do conjunto de dados, esta coluna é a *Pontuação Média do hotel, calculada com base no último comentário no último ano*. Esta parece ser uma maneira incomum de calcular a pontuação, mas é o dado extraído, então podemos aceitá-lo como está por enquanto.
  
  ✅ Com base nas outras colunas desses dados, você consegue pensar em outra maneira de calcular a pontuação média?

* `Total_Number_of_Reviews`
  * O número total de avaliações que este hotel recebeu - não está claro (sem escrever algum código) se isso se refere às avaliações no conjunto de dados.
* `Additional_Number_of_Scoring`
  * Isso significa que uma pontuação de avaliação foi dada, mas nenhuma avaliação positiva ou negativa foi escrita pelo avaliador.

**Colunas de avaliação**

- `Reviewer_Score`
  - Este é um valor numérico com no máximo 1 casa decimal entre os valores mínimos e máximos de 2.5 e 10
  - Não está explicado por que 2.5 é a menor pontuação possível
- `Negative_Review`
  - Se um avaliador não escreveu nada, este campo terá "**No Negative**"
  - Note que um avaliador pode escrever uma avaliação positiva na coluna de avaliação negativa (por exemplo, "não há nada de ruim neste hotel")
- `Review_Total_Negative_Word_Counts`
  - Contagens de palavras negativas mais altas indicam uma pontuação mais baixa (sem verificar a sentimentalidade)
- `Positive_Review`
  - Se um avaliador não escreveu nada, este campo terá "**No Positive**"
  - Note que um avaliador pode escrever uma avaliação negativa na coluna de avaliação positiva (por exemplo, "não há nada de bom neste hotel")
- `Review_Total_Positive_Word_Counts`
  - Contagens de palavras positivas mais altas indicam uma pontuação mais alta (sem verificar a sentimentalidade)
- `Review_Date` e `days_since_review`
  - Uma medida de frescor ou obsolescência pode ser aplicada a uma avaliação (avaliações mais antigas podem não ser tão precisas quanto as mais novas porque a administração do hotel mudou, ou renovações foram feitas, ou uma piscina foi adicionada etc.)
- `Tags`
  - Estes são descritores curtos que um avaliador pode selecionar para descrever o tipo de hóspede que eram (por exemplo, solteiro ou família), o tipo de quarto que tiveram, a duração da estadia e como a avaliação foi submetida. 
  - Infelizmente, usar essas tags é problemático, confira a seção abaixo que discute sua utilidade.

**Colunas do avaliador**

- `Total_Number_of_Reviews_Reviewer_Has_Given`
  - Isso pode ser um fator em um modelo de recomendação, por exemplo, se você puder determinar que avaliadores mais prolíficos com centenas de avaliações eram mais propensos a serem negativos em vez de positivos. No entanto, o avaliador de qualquer avaliação específica não é identificado com um código único e, portanto, não pode ser vinculado a um conjunto de avaliações. Existem 30 avaliadores com 100 ou mais avaliações, mas é difícil ver como isso pode ajudar o modelo de recomendação.
- `Reviewer_Nationality`
  - Algumas pessoas podem pensar que certas nacionalidades são mais propensas a dar uma avaliação positiva ou negativa por causa de uma inclinação nacional. Tenha cuidado ao construir tais visões anedóticas em seus modelos. Esses são estereótipos nacionais (e às vezes raciais), e cada avaliador era um indivíduo que escreveu uma avaliação com base em sua experiência. Isso pode ter sido filtrado por muitas lentes, como suas estadias em hotéis anteriores, a distância percorrida e seu temperamento pessoal. Pensar que a nacionalidade deles foi a razão para uma pontuação de avaliação é difícil de justificar.

##### Exemplos

| Pontuação Média | Total de Avaliações | Pontuação do Avaliador | Avaliação Negativa                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Avaliação Positiva                 | Tags                                                                                      |
| ---------------- | -------------------- | ---------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- | ----------------------------------------------------------------------------------------- |
| 7.8              | 1945                 | 2.5                    | Atualmente, não é um hotel, mas um canteiro de obras. Fui aterrorizado desde cedo pela manhã e o dia todo com barulho de construção inaceitável enquanto descansava após uma longa viagem e trabalhava no quarto. Pessoas estavam trabalhando o dia todo, ou seja, com martelos pneumáticos nos quartos adjacentes. Pedi uma troca de quarto, mas nenhum quarto silencioso estava disponível. Para piorar as coisas, fui cobrado a mais. Fiz o check-out à noite, pois tinha que sair muito cedo para um voo e recebi uma conta apropriada. Um dia depois, o hotel fez outra cobrança sem meu consentimento, além do preço reservado. É um lugar terrível. Não se puna fazendo uma reserva aqui | Nada. Terrível lugar. Fique longe | Viagem de negócios                                Casal. Quarto Duplo Padrão. Fiquei 2 noites. |

Como você pode ver, este hóspede não teve uma estadia feliz neste hotel. O hotel tem uma boa pontuação média de 7.8 e 1945 avaliações, mas este avaliador deu 2.5 e escreveu 115 palavras sobre como foi negativa sua estadia. Se ele não tivesse escrito nada na coluna Positive_Review, você poderia supor que não havia nada positivo, mas, infelizmente, ele escreveu 7 palavras de aviso. Se apenas contássemos palavras em vez do significado ou sentimento das palavras, poderíamos ter uma visão distorcida da intenção do avaliador. Estranhamente, sua pontuação de 2.5 é confusa, porque se a estadia no hotel foi tão ruim, por que dar qualquer ponto? Investigando o conjunto de dados de perto, você verá que a menor pontuação possível é 2.5, não 0. A maior pontuação possível é 10.

##### Tags

Como mencionado acima, à primeira vista, a ideia de usar `Tags` para categorizar os dados faz sentido. Infelizmente, essas tags não são padronizadas, o que significa que em um determinado hotel, as opções podem ser *Quarto individual*, *Quarto duplo*, e *Quarto de casal*, mas no próximo hotel, eles são *Quarto Individual Deluxe*, *Quarto Clássico Queen* e *Quarto Executivo King*. Esses podem ser a mesma coisa, mas há tantas variações que a escolha se torna:

1. Tentar mudar todos os termos para um único padrão, o que é muito difícil, porque não está claro qual seria o caminho de conversão em cada caso (por exemplo, *Quarto individual clássico* se mapeia para *Quarto individual*, mas *Quarto Superior Queen com Jardim de Courtyard ou Vista da Cidade* é muito mais difícil de mapear)

2. Podemos adotar uma abordagem de NLP e medir a frequência de certos termos como *Solteiro*, *Viajante de Negócios*, ou *Família com crianças pequenas* conforme se aplicam a cada hotel e levar isso em consideração na recomendação  

As tags geralmente (mas nem sempre) são um único campo contendo uma lista de 5 a 6 valores separados por vírgulas que se alinham a *Tipo de viagem*, *Tipo de hóspedes*, *Tipo de quarto*, *Número de noites*, e *Tipo de dispositivo em que a avaliação foi submetida*. No entanto, porque alguns avaliadores não preenchem cada campo (eles podem deixar um em branco), os valores nem sempre estão na mesma ordem.

Como exemplo, pegue *Tipo de grupo*. Existem 1025 possibilidades únicas neste campo na coluna `Tags`, e, infelizmente, apenas algumas delas se referem a um grupo (algumas são do tipo de quarto etc.). Se você filtrar apenas os que mencionam família, os resultados contêm muitos resultados do tipo *Quarto para família*. Se você incluir o termo *com*, ou seja, contar os valores *Família com*, os resultados são melhores, com mais de 80.000 dos 515.000 resultados contendo a frase "Família com crianças pequenas" ou "Família com crianças mais velhas".

Isso significa que a coluna de tags não é completamente inútil para nós, mas levará algum trabalho para torná-la útil.

##### Pontuação média do hotel

Existem várias peculiaridades ou discrepâncias com o conjunto de dados que não consigo entender, mas são ilustradas aqui para que você esteja ciente delas ao construir seus modelos. Se você descobrir, por favor, nos avise na seção de discussão!

O conjunto de dados possui as seguintes colunas relacionadas à pontuação média e ao número de avaliações:

1. Hotel_Name
2. Additional_Number_of_Scoring
3. Average_Score
4. Total_Number_of_Reviews
5. Reviewer_Score  

O único hotel com mais avaliações neste conjunto de dados é o *Britannia International Hotel Canary Wharf* com 4789 avaliações de 515.000. Mas se olharmos o valor de `Total_Number_of_Reviews` para este hotel, ele é 9086. Você pode supor que há muitas mais pontuações sem avaliações, então talvez devêssemos adicionar o valor da coluna `Additional_Number_of_Scoring`. Esse valor é 2682, e adicioná-lo a 4789 nos dá 7471, o que ainda está 1615 abaixo do `Total_Number_of_Reviews`. 

Se você pegar as colunas `Average_Score`, pode supor que é a média das avaliações no conjunto de dados, mas a descrição do Kaggle é "*Pontuação Média do hotel, calculada com base no último comentário no último ano*". Isso não parece muito útil, mas podemos calcular nossa própria média com base nas pontuações das avaliações no conjunto de dados. Usando o mesmo hotel como exemplo, a pontuação média do hotel é dada como 7.1, mas a pontuação calculada (pontuação média do avaliador *no* conjunto de dados) é 6.8. Isso é próximo, mas não é o mesmo valor, e só podemos supor que as pontuações dadas nas avaliações `Additional_Number_of_Scoring` aumentaram a média para 7.1. Infelizmente, sem uma maneira de testar ou provar essa afirmação, é difícil usar ou confiar em `Average_Score`, `Additional_Number_of_Scoring` e `Total_Number_of_Reviews` quando eles se baseiam, ou se referem a, dados que não temos.

Para complicar ainda mais as coisas, o hotel com o segundo maior número de avaliações tem uma pontuação média calculada de 8.12 e a `Average_Score` do conjunto de dados é 8.1. Essa pontuação correta é uma coincidência ou o primeiro hotel é uma discrepância? 

Na possibilidade de que esses hotéis possam ser um outlier, e que talvez a maioria dos valores se somem (mas alguns não por algum motivo), escreveremos um programa curto a seguir para explorar os valores no conjunto de dados e determinar o uso correto (ou não uso) dos valores.

> 🚨 Uma nota de cautela
>
> Ao trabalhar com este conjunto de dados, você escreverá um código que calcula algo a partir do texto sem precisar ler ou analisar o texto você mesmo. Essa é a essência do NLP, interpretar significado ou sentimento sem que um humano tenha que fazê-lo. No entanto, é possível que você leia algumas das avaliações negativas. Eu recomendaria que você não fizesse isso, porque você não precisa. Algumas delas são tolas ou irrelevantes, como "O tempo não estava bom", algo além do controle do hotel, ou de fato, de qualquer um. Mas há um lado sombrio em algumas avaliações também. Às vezes, as avaliações negativas são racistas, sexistas ou idadistas. Isso é lamentável, mas esperado em um conjunto de dados extraído de um site público. Alguns avaliadores deixam avaliações que você acharia de mau gosto, desconfortáveis ou perturbadoras. Melhor deixar o código medir o sentimento do que lê-las você mesmo e ficar chateado. Dito isso, é uma minoria que escreve tais coisas, mas elas existem.

## Exercício - Exploração de dados
### Carregar os dados

Isso é o suficiente para examinar os dados visualmente, agora você escreverá algum código e obterá algumas respostas! Esta seção usa a biblioteca pandas. Sua primeira tarefa é garantir que você pode carregar e ler os dados CSV. A biblioteca pandas tem um carregador CSV rápido, e o resultado é colocado em um dataframe, como nas lições anteriores. O CSV que estamos carregando tem mais de meio milhão de linhas, mas apenas 17 colunas. O pandas oferece muitas maneiras poderosas de interagir com um dataframe, incluindo a capacidade de realizar operações em cada linha.

A partir daqui, nesta lição, haverá trechos de código e algumas explicações do código e algumas discussões sobre o que os resultados significam. Use o _notebook.ipynb_ incluído para seu código.

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

Agora que os dados estão carregados, podemos realizar algumas operações sobre eles. Mantenha este código no topo do seu programa para a próxima parte.

## Explorar os dados

Neste caso, os dados já estão *limpos*, o que significa que estão prontos para trabalhar, e não têm caracteres em outros idiomas que possam confundir algoritmos que esperam apenas caracteres em inglês.

✅ Você pode ter que trabalhar com dados que exigem algum processamento inicial para formatá-los antes de aplicar técnicas de NLP, mas não desta vez. Se você tivesse que fazer isso, como lidaria com caracteres não ingleses?

Reserve um momento para garantir que, uma vez que os dados estejam carregados, você possa explorá-los com código. É muito fácil querer se concentrar nas colunas `Negative_Review` e `Positive_Review`. Elas estão preenchidas com texto natural para seus algoritmos de NLP processarem. Mas espere! Antes de você mergulhar no NLP e no sentimento, você deve seguir o código abaixo para verificar se os valores dados no conjunto de dados correspondem aos valores que você calcula com pandas.

## Operações no dataframe

A primeira tarefa nesta lição é verificar se as seguintes afirmações estão corretas escrevendo algum código que examine o dataframe (sem alterá-lo).

> Como muitas tarefas de programação, existem várias maneiras de concluir isso, mas um bom conselho é fazê-lo da maneira mais simples e fácil que você puder, especialmente se for mais fácil de entender quando você voltar a esse código no futuro. Com dataframes, existe uma API abrangente que muitas vezes terá uma maneira de fazer o que você deseja de forma eficiente.
Trate as seguintes perguntas como tarefas de codificação e tente respondê-las sem olhar para a solução. 1. Imprima a *forma* do dataframe que você acabou de carregar (a forma é o número de linhas e colunas) 2. Calcule a contagem de frequência para nacionalidades dos avaliadores: 1. Quantos valores distintos existem para a coluna `Reviewer_Nationality` e quais são eles? 2
as linhas têm valores da coluna `Positive_Review` de "No Positive" 9. Calcule e imprima quantas linhas têm valores da coluna `Positive_Review` de "No Positive" **e** valores da `Negative_Review` de "No Negative" ### Respostas do código 1. Imprima a *forma* do dataframe que você acabou de carregar (a forma é o número de linhas e colunas) ```python
   print("The shape of the data (rows, cols) is " + str(df.shape))
   > The shape of the data (rows, cols) is (515738, 17)
   ``` 2. Calcule a contagem de frequência para nacionalidades de revisores: 1. Quantos valores distintos existem para a coluna `Reviewer_Nationality` e quais são? 2. Qual nacionalidade de revisor é a mais comum no conjunto de dados (imprima o país e o número de avaliações)? ```python
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
   ``` 3. Quais são as próximas 10 nacionalidades mais frequentemente encontradas e suas contagens de frequência? ```python
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
      ``` 3. Qual foi o hotel mais frequentemente avaliado para cada uma das 10 nacionalidades de revisores mais comuns? ```python
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
   ``` 4. Quantas avaliações existem por hotel (contagem de frequência de hotel) no conjunto de dados? ```python
   # First create a new dataframe based on the old one, removing the uneeded columns
   hotel_freq_df = df.drop(["Hotel_Address", "Additional_Number_of_Scoring", "Review_Date", "Average_Score", "Reviewer_Nationality", "Negative_Review", "Review_Total_Negative_Word_Counts", "Positive_Review", "Review_Total_Positive_Word_Counts", "Total_Number_of_Reviews_Reviewer_Has_Given", "Reviewer_Score", "Tags", "days_since_review", "lat", "lng"], axis = 1)
   
   # Group the rows by Hotel_Name, count them and put the result in a new column Total_Reviews_Found
   hotel_freq_df['Total_Reviews_Found'] = hotel_freq_df.groupby('Hotel_Name').transform('count')
   
   # Get rid of all the duplicated rows
   hotel_freq_df = hotel_freq_df.drop_duplicates(subset = ["Hotel_Name"])
   display(hotel_freq_df) 
   ``` | Hotel_Name | Total_Number_of_Reviews | Total_Reviews_Found | | :----------------------------------------: | :---------------------: | :-----------------: | | Britannia International Hotel Canary Wharf | 9086 | 4789 | | Park Plaza Westminster Bridge London | 12158 | 4169 | | Copthorne Tara Hotel London Kensington | 7105 | 3578 | | ... | ... | ... | | Mercure Paris Porte d Orleans | 110 | 10 | | Hotel Wagner | 135 | 10 | | Hotel Gallitzinberg | 173 | 8 | Você pode notar que os resultados *contados no conjunto de dados* não correspondem ao valor em `Total_Number_of_Reviews`. Não está claro se esse valor no conjunto de dados representava o número total de avaliações que o hotel teve, mas nem todas foram coletadas, ou algum outro cálculo. `Total_Number_of_Reviews` não é usado no modelo por causa dessa falta de clareza. 5. Embora haja uma coluna `Average_Score` para cada hotel no conjunto de dados, você também pode calcular uma pontuação média (obtendo a média de todas as pontuações dos revisores no conjunto de dados para cada hotel). Adicione uma nova coluna ao seu dataframe com o cabeçalho da coluna `Calc_Average_Score` que contenha essa média calculada. Imprima as colunas `Hotel_Name`, `Average_Score`, e `Calc_Average_Score`. ```python
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
   ``` Você também pode se perguntar sobre o valor `Average_Score` e por que às vezes ele é diferente da pontuação média calculada. Como não podemos saber por que alguns dos valores coincidem, mas outros têm uma diferença, é mais seguro, neste caso, usar as pontuações das avaliações que temos para calcular a média nós mesmos. Dito isso, as diferenças geralmente são muito pequenas, aqui estão os hotéis com a maior divergência da média do conjunto de dados e a média calculada: | Average_Score_Difference | Average_Score | Calc_Average_Score | Hotel_Name | | :----------------------: | :-----------: | :----------------: | ------------------------------------------: | | -0.8 | 7.7 | 8.5 | Best Western Hotel Astoria | | -0.7 | 8.8 | 9.5 | Hotel Stendhal Place Vend me Paris MGallery | | -0.7 | 7.5 | 8.2 | Mercure Paris Porte d Orleans | | -0.7 | 7.9 | 8.6 | Renaissance Paris Vendome Hotel | | -0.5 | 7.0 | 7.5 | Hotel Royal Elys es | | ... | ... | ... | ... | | 0.7 | 7.5 | 6.8 | Mercure Paris Op ra Faubourg Montmartre | | 0.8 | 7.1 | 6.3 | Holiday Inn Paris Montparnasse Pasteur | | 0.9 | 6.8 | 5.9 | Villa Eugenie | | 0.9 | 8.6 | 7.7 | MARQUIS Faubourg St Honor Relais Ch teaux | | 1.3 | 7.2 | 5.9 | Kube Hotel Ice Bar | Com apenas 1 hotel tendo uma diferença de pontuação maior que 1, isso significa que provavelmente podemos ignorar a diferença e usar a pontuação média calculada. 6. Calcule e imprima quantas linhas têm valores da coluna `Negative_Review` de "No Negative" 7. Calcule e imprima quantas linhas têm valores da coluna `Positive_Review` de "No Positive" 8. Calcule e imprima quantas linhas têm valores da coluna `Positive_Review` de "No Positive" **e** `Negative_Review` de "No Negative" ```python
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
   ``` ## Outra maneira Outra maneira de contar itens sem Lambdas, e usar a soma para contar as linhas: ```python
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
   ``` Você pode ter notado que há 127 linhas que têm tanto valores "No Negative" quanto "No Positive" para as colunas `Negative_Review` e `Positive_Review`, respectivamente. Isso significa que o revisor deu ao hotel uma pontuação numérica, mas se recusou a escrever uma avaliação positiva ou negativa. Felizmente, essa é uma pequena quantidade de linhas (127 de 515738, ou 0,02%), então provavelmente não distorcerá nosso modelo ou resultados em nenhuma direção particular, mas você pode não ter esperado que um conjunto de dados de avaliações tivesse linhas sem avaliações, então vale a pena explorar os dados para descobrir linhas como essa. Agora que você explorou o conjunto de dados, na próxima lição você filtrará os dados e adicionará alguma análise de sentimentos. --- ## 🚀Desafio Esta lição demonstra, como vimos em lições anteriores, quão criticamente importante é entender seus dados e suas peculiaridades antes de realizar operações sobre eles. Dados baseados em texto, em particular, requerem uma análise cuidadosa. Explore vários conjuntos de dados ricos em texto e veja se consegue descobrir áreas que poderiam introduzir viés ou sentimentos distorcidos em um modelo. ## [Quiz pós-aula](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/38/) ## Revisão e Autoestudo Faça [este Caminho de Aprendizagem sobre NLP](https://docs.microsoft.com/learn/paths/explore-natural-language-processing/?WT.mc_id=academic-77952-leestott) para descobrir ferramentas a serem testadas ao construir modelos de fala e texto. ## Tarefa [NLTK](assignment.md) Por favor, escreva a saída da esquerda para a direita.

**Aviso Legal**:  
Este documento foi traduzido utilizando serviços de tradução automática baseados em IA. Embora nos esforcemos pela precisão, esteja ciente de que traduções automatizadas podem conter erros ou imprecisões. O documento original em seu idioma nativo deve ser considerado a fonte autoritária. Para informações críticas, recomenda-se a tradução profissional por humanos. Não nos responsabilizamos por quaisquer mal-entendidos ou interpretações errôneas decorrentes do uso desta tradução.