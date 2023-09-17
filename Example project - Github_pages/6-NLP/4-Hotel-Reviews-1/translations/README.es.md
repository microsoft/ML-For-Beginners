# Análisis de sentimiento con reseñas de hoteles - procesando los datos

En esta sección usarás las técnicas de las lecciones anteriores para hacer un ańalisis exploratorio de datos de un gran conjunto de datos. Una vez que tengas un buen entendimiento de la utilidad de las distintas, aprenderás:

- cómo eliminar las columnas innecesarias
- cómo calcular algunos datos nuevos basándote en las columnas existentes
- cómo guardar el conjunto de datos resultante para usarlo en el desafío final

## [Examen previo a la lección](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/37?loc=es)

### Introducción

hasta ahora has aprendido acerca de cómo los datos de texto es muy diferente a los tipos de datos numéricos. Si el texto fue escrito o hablado por un humano, si puede ser analizado para encontrar patrones y frecuencias, sentimiento y significado. Este lección te lleva a un conjunto de datos reales con un desafío real: **[515K datos de reseñas de hoteles en Europa](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)** e incluye una [licencia de dominio público: CC0](https://creativecommons.org/publicdomain/zero/1.0/). Se obtuvo mediante scraping de Booking.com de fuentes públicas. El creador del conjunto de datos fue Jiashen Liu.

### Preparación

Necesitarás:

* La habilidad de ejecutar notebooks .ipynb usando Python 3
* pandas
* NLTK, [la cual deberías instalar localmente](https://www.nltk.org/install.html)
* El conjunto de datos el cual está disponible en Kaggle [515K Hotel Reviews Data in Europe](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe). El cual de alrededor de 230MB desempequetado. Descargalo en el directorio raíz `/data` asociado con estas lecciones de NLP.

## Análisis exploratorio de datos

Este desafío asume que estás construyendo un bot de recomendación de hoteles usando análisis de sentimiento y puntajes de reseñas de huéspedes. El conjunto de datos que usarás incluye reseñas de 1493 hoteles distintos en 6 ciudades.

Usando Python, un conjunto de datos de reseñas de hoteles y análisis de sentimiento de NLTK podrías encontrar:

* ¿Cuáles son las frases y palabras usadas con mayor frecuencia en las reseñas?
* ¿Las *etiquetas* oficiales que describen un hotel se correlacionan con los puntajes de las reseñas (por ejemplo son más negativas para un hotel en particular de *Familia con niños pequeños* que *viajero solitario*, quizá indicando que es mejor para *viajeros solitarios*)?
* ¿Los puntajes de sentimiento de NLTK 'concuerdan' (agree) con los puntajes numéricos de quién reseña el hotel?

#### Conjuntos de datos

Exploremos el conjunto de datos que descargaste y guardaste de forma local. Abre el archivo en un editor como VS Code o Excel.

Los encabezados en el conjunto de datos se ven así:

*Hotel_Address, Additional_Number_of_Scoring, Review_Date, Average_Score, Hotel_Name, Reviewer_Nationality, Negative_Review, Review_Total_Negative_Word_Counts, Total_Number_of_Reviews, Positive_Review, Review_Total_Positive_Word_Counts, Total_Number_of_Reviews_Reviewer_Has_Given, Reviewer_Score, Tags, days_since_review, lat, lng*

Aquí están agrupados de cierta forma que sean fáciles de examinar:
##### Columnas Hotel

* `Hotel_Name`, `Hotel_Address`, `lat` (latitud), `lng` (longitud)
  * Usando *lat* y *lng* podrías graficar un map con Python mostrando las ubicaciones de hoteles (quiźa usando un código de colores para las reseñas positivas y negativas)
  * Hotel_Address obviamente no nos es útil, y probablemente la reemplazaremos con un país para una búsqueda y ordenamiento más fácil.

**Columnas Hotel Meta-review**

* `Average_Score`
  * De acuerdo al creador del conjunto de datos, esta columna es el *Puntaje promedio del hotel, basado en el último comentario del último año*. Esto parece una forma inusual de calcular el puntaje, pero son los datos extraídos por lo que podemos tomarlos como valor por ahora.
  ✅ Basado en las otras columnas de estos datos, ¿puedes pensar en otra forma de calcular el puntaje promedio?

* `Total_Number_of_Reviews`
  * El número total de reseñas que ha recibido este hotel. No está claro (sin escribir algo de código) si esto se refiere a las reseñas en el conjunto de datos.
* `Additional_Number_of_Scoring`
  * Esto significa que se dió un puntaje de reseña pero no se escribió reseña positiva o negativa por el crítico.

**Columnas Review**

- `Reviewer_Score`
  - Este es un valor numérico con máximo 1 posición decimal entre los valores máximos y mínimos de 2.5 y 10
  - No se explica por qué 2.5 es el puntaje más bajo posible
- `Negative_Review`
  - Si un crítico no escribió nada, este campo tendrá "**No Negative**"
  - Nota que un crítico puede escribir una reseña positiva en la columna de reseña negativa (por ejemplo, "there is nothing bad about this hotel")
- `Review_Total_Negative_Word_Counts`
  - El conteo de palabras negativas más altas indica un puntaje más bajo (sin revisar el sentimentalismo)
- `Positive_Review`
  - Si un crítico no escribió nada, este campo tendrá "**No Positive**"
  - Nota que un cŕitico puede escribir una reseña negativa en la columna de reseña positiva (por ejemplo, "there is nothing good about this hotel at all")
- `Review_Total_Positive_Word_Counts`
  - El conteo de palabras positivas más altas indica un puntaje más alto (sin revisar el sentimentalismo)
- `Review_Date` y `days_since_review`
  - Una medida de frescura o ranciedad puede aplicarse a una reseña (las reseñas más biejas pueden no ser tan precisas como las nuevas ya que la administración del hotel ha cambiado, o se hicieron remodelaciones, o se agregó una alberca, etc.)
- `Tags`
  - Estas son breves descriptores que un crítico puede seleccionar para describir el tipo de huéspesd que fueron (ejemplo, `solo` o `family`), el tipo de cuarto que se les asignó, la duración de la estancia y cómo se envió la reseña.
  - Desafortunadamente, el usar estas etiquetas es problemático, revisa la sección de abajo la cual discute su utilidad

**Columnas Reviewer**

- `Total_Number_of_Reviews_Reviewer_Has_Given`
  - Esta podría ser un factor en un modelo de recomendación, por ejemplo, si pudieras determinar que los críticos más prolíficos con cientos de reseñas tendieran a ser más negativos que positivos. Sin embargo, el crítico de cualquier reseña en particular no se identifica con un código único, y por lo tanto no puede ser vinculado a un conjunto de reseñas. Hay 30 críticos con 100 reseñas o más, pero es díficil ver cómo esto puede ayudar al modelo de recomendación.
- `Reviewer_Nationality`
  - Algunas personas podrían pensar que ciertas nacionalidades tienden dar una reseña positiva o negatica debido a una inclinación nacional. Sea cuidadoso al construir dichas vistas anecdóticas en tus modelos. Estos son estereotipos nacionales (y algunas veces raciales), y cada crítico fue un individuo que escribió una reseña basándose en su experiencia. Podría haber sido filtrado a través de varios lentes tal como sus estadías anteriores en hoteles, la distancia viajada, y su temperamento personal. Pensar que su nacionalidad fue el motivo del puntaje de una reseña es difícil de justificar.

##### Ejemplos

| Average  Score | Total Number   Reviews | Reviewer   Score | Negative <br />Review                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Positive   Review                 | Tags                                                                                      |
| -------------- | ---------------------- | ---------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- | ----------------------------------------------------------------------------------------- |
| 7.8            | 1945                   | 2.5              | This is  currently not a hotel but a construction site I was terrorized from early  morning and all day with unacceptable building noise while resting after a  long trip and working in the room People were working all day i e with  jackhammers in the adjacent rooms I asked for a room change but no silent  room was available To make things worse I was overcharged I checked out in  the evening since I had to leave very early flight and received an appropriate  bill A day later the hotel made another charge without my consent in excess  of booked price It's a terrible place Don't punish yourself by booking  here | Nothing  Terrible place Stay away | Business trip                                Couple Standard Double  Room Stayed 2 nights |

Como puedes ver, este huésped no tuvo una estancia agradable en el hotel. El hotel tiene un puntaje promedio bueno de 7.8 y 1945 reseñas, pero este crítico le dió 2.5 y escribió 115 palabras acerca de qué tan negativa fue su estancia. Si no escribió nada en la columna Positive_Review, podrías suponer que no hubo nada positivo, pero por desgracia escribió 7 palabras de advertencia. Si contaramos sólo palabras en lugar del significado, o el sentimiento de las palabras,  podríamos tener una vista sesgada de la intención del crítico. Extrañamente, su puntaje de 2.5 es confuso, ya que si esa estancia en el hotel fue tan mala, ¿por qué darle una puntuación? Investigando el conjunto de datos más de cerca, verás que el puntaje más bajo posible es de 2.5, no de 0. El puntaje más alto posible es de 10.

##### Etiquetas

Como se mencionó anteriormente, a primera vista, la idea de usar etiquetas (`Tags`) para categorizar los datos, hace sentido. Desafortunadamente estas etiquetas no están estandarizadas, lo cual significa que en cierto hotel, las opciones podrían ser *Single room*, *Twin room*, y *Double room*, pero en otro hotel, son *Deluxe Single Room*, *Classic Queen Room*, y *Executive King Room*. Estas pueden ser las mismas cosas, pero hay tantas variaciones que la elección se convierte en:

1. Intenta cambiar todos los términos a un estándar único, lo cual es muy difícil, por qué no está claro cuál sería la ruta de conversión en cada caso (por ejemplo *Classic single room* se asigna a *Single room* pero *Superior Queen Room with Courtyard Garden or City View* es más difícil de asignar).

1. Podemos tomar un enfoque de NLP y medir la frecuencia de ciertos términos como *Solo*, *Business Traveller*, o *Family with young kids* ya que aplican para cada hotel, y tenerlo en cuenta en la recomendación.

Las etiquetas con frecuencia son (pero no siempre) un campo único que contiene una lista de 5 a 6 valores separados por coma que se alinean al *Tipo de viaje*, *Tipo de huésped*, *Tipo de habitación*, *Número de noches*, y *Tipo de dispositivo de revisión con el que se envió*. Sin embargo, ya que algunos críticos no llenan cada campo (pueden dejar uno en blanco), los valores no siempre aparecen en el mismo orden.

Como ejemplo toma el *Tipo de grupo*. Existen 1025 posibilidades únicas en este campo en la columna `Tags`, y desafortunadamente sólo algunos de ellos se refieren al grupo (algunos son de el tipo de habitación, etc.). Si filtras sólo aquellos que mencionan familia, los resultados contienen varios tipos de resultados *Family room*. Si incluyes el término *with*, esto es, cuenta los valores *Family with*, los resultados mejoran, con más de 80,000 de 515,000 resultados que contienen la frase "Family with young children" o "Family with older children".

Esto significa que la columna `Tags` no nos es completamente unútil, pero nos tomará algo de trabajo hacerla útil.

##### Puntaje promedio de hotel

Hay cierto número de rarezas o discrepancias con el conjunto de datos que no puedo comprender, pero se ilustran aquí para que estés al tanto de ello cuando construyes tus modelos. ¡Si las averiguas, por favor háznoslo saber en la sección de discusión!

El conjunto de datos tiene las siguientes columnas relacionadas al puntaje promedio y el número de reseñas:

1. Hotel_Name
2. Additional_Number_of_Scoring
3. Average_Score
4. Total_Number_of_Reviews
5. Reviewer_Score  

El hotel con la mayor cantidad de reseñas de este conjunto de datos es *Britannia International Hotel Canary Wharf* con 4789 reseñas de 515,000. Pero si miramos el valor del `Total_Number_of_Reviews` para este hotel, es de 9086. Puedes suponer que hay mucho más puntaje sin las reseñas, así que quizá deberíamos agregar el valor de la columna `Additional_Number_of_Scoring`. Ese valor es 2682, y sumándolo a 4789 obtenemos 7,471 lo cual está 1615 por debajo de `Total_Number_of_Reviews`.

Si tomas la columna `Average_Score`, puedes suponer que es el promedio de las reseñas del conjunto de datos, pero la descripción de Kaggle es "*Average Score of the hotel, calculado basándose en el último comentario del último año*". Eso no parece ser tan útil, pero podemos calcular nuestro propio promedio basándonos en los puntajes de reseñas del conjunto de datos. Usando el mismo hotel como ejemplo, el puntaje promedio del hotel es 7,1 pero el puntaje calculado (el puntaje del crítico promedio *en* el conjunto de datos) es 6.8. Esto está cerca, pero no es el mismo valor, y sólo podemos adivinar que los puntajes dados en las reseñas de `Additional_Number_of_Scoring` incrementaron el promedio a 7.1. Desafortunadamente sin forma de probar o demostrar esa afirmación, es difícil usar o confiar en `Average_Score`, `Additional_Number_of_Scoring` y `Total_Number_of_Reviews` cuando se basan en, o se refieren a los datos que no tenemos.

Para complicar más las cosas, el hotel con el segundo número más alto de reseñas tiene un puntaje promedio calculado de 8.12 y el conjunto de datos `Average_Score` es de 8.1. ¿este puntaje es correcto o es una coincidencia o el primer hotel tiene una discrepancia?

En la posibilidad que estos hoteles puedan sea un caso atípico, y que tal vez la mayoría de los valores se sumen (pero algunos no por alguna razón), escribiremos un pequeño programa para explorar los valores en le conjunto de datos y determinar el uso correcto (o no uso) de los valores.

> 🚨 Una nota de advertencia
>
> Al trabajar con este conjunto de datos escribirás código que calcula algo a partir del texto sin tener que leer o analizar el texto por ti mismo. Esta es la esencia del procesamiento del lenguaje natural (NLP), el interpretar el significado o sentimiento sin tener que depender de un humano que lo haga. Sin embargo, es posible que leas algunas de las reseñas negativas. Te insisto no lo hagas, ya que no tienes porqué hacerlo. Algunas son tontas, o reseñas negativas irrelevantes del hotel, como "The weather wasn't great", algo fuera del control del hotel, o de hecho nada. Pero también hay un lado obscuro de algunas reseñas. En ocasiones, las reseñas negativas son racistas, sexistas o edadistas. Lo cual es desafortunado pero esperado e un conjunto de datos obtenido de un sitio web público. Algunos críticos dejan reseñas que encontrarás desagradables, incómodas o molestas. Es mejor dejar que el código mida el sentimiento en lugarde leerlo tú mismo y enfadarte. Dicho esto, es una minoría la que escribe esas cosas, pero existen de todas formas.

## Ejercicio - Exploración de datos
### Carga los datos

Ya es suficiente de examinar los datos de forma visual, ¡ahora escribirás algo de código para obtener algunas respuestas! Esta sección usa la biblioteca pandas. Tu primer tarea es asegurarte que puedes cargar y leer los datos del CSV. La biblioteca pandas tiene un cargador CSV rápido, y el resultado se coloca en un dataframe, como en lecciones anteriores. El CSV que estamos cargando tiene más de medio millón filas, pero sólo 17 columnas. Pandas te proporciona muchas formas poderosas de interactuar con un dataframe, incluyendo la capacidad de realizar operaciones en cada fila.

A partir de aquí en esta lección, habrá fragmentos de código y algunas explicaciones del mismo, además de algunas discusiones acerca de lo que significan los resultados. Usa el _notebook.ipynb_ incluido para tu código.

Empecemos cargando el archivo de datos que usarás:

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

Ahora que se han cargado los datos, podemos realizar operaciones en ellos. Coloca este código al principio de tu programa para la siguiente parte.

## Explora los datos

En este caso, los datos ya se encuentranb *limpios*, lo cual significa que están listos para que trabajemos sobre ellos, y no contienen caracteres en otros idiomas que podrían entorpecer a los algoritmos que esperan únicamente caracteres en Inglés.

✅ Tendrás que trabajar con datos que requieren un procesamiento inicial para formatearlos antes de aplicar técnicas de NLP, pero no este vez. Si así fuera, ¿cómo manejarías los caracteres no pertenecientes al Inglés?

Dedica un momento a asegurarte que una vez se carguen los datos, puedes explorarlos con código. Es muy fácil querer enfocarte en las columnas `Negative_Review` y `Positive_Review`. Las cuales contienen texto natural para procesar por tus algoritmos de NLP. ¡Pero espera! Antes que comiences el procesamiento de lenguaje natural y sentimiento, deberías seguir el código siguiente para cerciorarte si los valores dados en el conjunto de datos coinciden con los valores que calculaste con pandas.

## Operaciones de dataframe

La primer tarea en esta lección es revisar si las siguientes afirmaciones son correctas al escribir algo de código que examine el dataframe (sin modificarlo).

> Como muchas tareas de programación, hay varias formas de completarla, pero un buen consejo es hacerlo de la forma más simple y fácil que puedas, especialmente si seŕa más fácil entenderla cuando volvamos a este código en el futuro. Con dataframes, hay una API eshaustiva que tendrá frecuentemente una forma de hacer lo que quieres de forma eficiente.

Trata las siguientes preguntas como tareas de programación e intenta responderlas sin mirar la solución.

1. Imprime la *forma* del dataframe que acabas de cargar (la forma es el número de filas y columnas)
2. Calcula el conteo de frecuencia para las nacionalidades de los críticos:
   1. ¿Cuántos valores distintos existen para la columna `Reviewer_Nationality` y cuáles son?
   2. ¿Cuál es la nacionalidad del crítico que es la más común en el conjunto de datos (imprime el país y el número de reseñas)?
   3. ¿Cuáles son las 10 nacionalidades encontradas más frecuentemente, y el conteo de sus frecuencias?
3. ¿Cuál fue el hotel más frecuentemente reseñado por cada una del top 10 de nacionalidades de críticos?
4. ¿Cuántas reseñas hay por hotel (conteo de frecuencia de hotel) en el conjunto de datos?
5. Mientras que hay una columna `Average_Score` por cada hotel en el conjunto de datos, también puedes calcular un puntaje promedio (obteniendo el promedio de todos los puntajes de los críticos en el conjunto de datos para cada hotel). Agrega una nueva columna a tu dataframe con el encabezado `Calc_Average_Score` que contenga el promedio calculado.
6. ¿Algunos hoteles tienen el mismo `Average_Score` y `Calc_Average_Score` (redondeado a una posición decimal)?
   1. Intenta escribir una función en Python que tome una Serie (fila) como argumento y compare los valores, imprimiendo el mensaje cuando los valores no son iguales. Luego, usa el método `.apply()` para procesar cada fila con la función.
7. Calcula e imprime cuántas filas tienen en la columna `Negative_Review` valores de "No Negative" .
8. Calcula e imprime cuántas filas tienen en la columna `Positive_Review` valores de "No Positive" .
9. Calcula e imprime cuántas filas tienen en la columna `Positive_Review` valores de "No Positive" **y** en la columna `Negative_Review` valores de "No Negative".

### Respuestas al código

1. Imprime la *forma* del dataframe que acabas de cargar (la forma es el número de filas y columnas).

   ```python
   print("The shape of the data (rows, cols) is " + str(df.shape))
   > The shape of the data (rows, cols) is (515738, 17)
   ```

2. Calcula el conteo de frecuencia para las nacionalidades de los críticos:

   1. ¿Cuańtos valores distintos hay en la columna `Reviewer_Nationality` y cuáles son?
   2. ¿Cuál es la nacionalidad más común para los críticos en el conjunto de datos (imprime el país y el número de reseñas)?

   ```python
   # value_counts() creates a Series object that has index and values in this case, the country and the frequency they occur in reviewer nationality
   nationality_freq = df["Reviewer_Nationality"].value_counts()
   print("There are " + str(nationality_freq.size) + " different nationalities")
   # print first and last rows of the Series. Change to nationality_freq.to_string() to print all of the data
   print(nationality_freq) 
   ```

   ```output
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

   3. ¿Cuáles son los siguientes 10 nacionalidades encontradas más frecuentemente, y su conteo de frecuencia?

      ```python
      print("The highest frequency reviewer nationality is " + str(nationality_freq.index[0]).strip() + " with " + str(nationality_freq[0]) + " reviews.")
      # Notice there is a leading space on the values, strip() removes that for printing
      # What is the top 10 most common nationalities and their frequencies?
      print("The next 10 highest frequency reviewer nationalities are:")
      print(nationality_freq[1:11].to_string())
      ```

      ```output
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

3. ¿Cuál fue el hotel mayormente reseñado para cada uno del top 10 de las nacionalidades de críticos?

  ```python
   # What was the most frequently reviewed hotel for the top 10 nationalities
   # Normally with pandas you will avoid an explicit loop, but wanted to show creating a new dataframe using criteria (don't do this with large amounts of data because it could be very slow)
   for nat in nationality_freq[:10].index:
      # First, extract all the rows that match the criteria into a new dataframe
      nat_df = df[df["Reviewer_Nationality"] == nat]   
      # Now get the hotel freq
      freq = nat_df["Hotel_Name"].value_counts()
      print("The most reviewed hotel for " + str(nat).strip() + " was " + str(freq.index[0]) + " with " + str(freq[0]) + " reviews.") 
  ```

  ```output
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

4. ¿Cuántas reseñas hay por hotel (conteo de frecuencia del hotel) en el conjunto de datos?

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

   Has notado que los resultados de los *contados en el conjunto de datos* no coinciden con el valor en `Total_Number_of_Reviews`. No está claro si este valor en el conjunto de datos representó el número total de reseñas que tuvo el hotel, pero no fueron extraídas, o algún otro cálculo. La columna `Total_Number_of_Reviews`no se usa en el modelo porque no es del todo clara.

5. Mientras que hay una columna `Average_Score` para cada hotel en el conjunto de datos, también puedes calcular un puntaje promedio (obteniendo el promedio de todos los puntajes de los críticos en el conjunto de datos para cada hotel). Agrega una nueva columna a tu dataframe con el encabezado `Calc_Average_Score` que contenga dicho promedio calculado. Imprime las columnas `Hotel_Name`, `Average_Score`, y `Calc_Average_Score`.

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

   También te preguntarás aceca de el valor de `Average_Score` y por qué algunas veces es diferente del puntaje promedio calculado. Como no podemos saber por qué algunos de los valores coinciden, pero otros difieren, en esta situación lo más seguro es usar los puntajes de reseñas que tenemos para calcular el promedio por nosotros mismos. Dicho esto, las diferencias suelen ser mínimas, aquí tienes los hoteles con la mayor desviación del promedio del conjunto de datos y el promedio calculado:

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

   Con sólo 1 hotel que tiene un diferencia de puntaje mayor a 1, esto significa que probablemente podemos ignorar la diferencia y usar el puntaje promedio calculado.

6. Calcula e imprime cuántas filas tienen en la columna `Negative_Review` valores de "No Negative"

7. Calcula e imprime cuántas filas tienen en la columna `Positive_Review` valores de "No Positive"

8. Calcula e imprime cuántas filas tienen en la columna `Positive_Review` valores de "No Positive" **y** en la columna `Negative_Review` valores de "No Negative"

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
   ```

   ```output
   Number of No Negative reviews: 127890
   Number of No Positive reviews: 35946
   Number of both No Negative and No Positive reviews: 127
   Lambdas took 9.64 seconds
   ```

## Otra forma de hacerlo

Otra forma de contar los elementos sin usar Lambdas, y usar la suma para contar las filas:

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
   ```

   ```output
   Number of No Negative reviews: 127890
   Number of No Positive reviews: 35946
   Number of both No Negative and No Positive reviews: 127
   Sum took 0.19 seconds
   ```

   Quizá hayas notado que hay 127 filas que tienen valores tanto "No Negative" como "No Positive" para las columnas `Negative_Review` y `Positive_Review` respectivamente. Lo cual significa que los críticos le dieron al hotel un puntaje numérico, pero se negaron a escribir tanto una reseña positiva como negativa. Afortunadamente este es un número pequeño de filas (127 de 515738, o 0.02%), así que probablemente no sesgará nuestro modelo o los resultados en alguna dirección en particular, pero podrías no haber esperado que un conjunto de datos de reseñas tenga filas sin reseñas, por lo que vale la pena explorar los datos para descubrir filas como esta.

Ahora que has explorado el conjunto de datos, en la próxima lección filtrarás los datos y agregarás algo de análisis de sentimiento.

---
## 🚀Desafío

Esta lección demuestra, como vimos en lecciones anteriores, qué tan críticamente importante es entender tus datos y sus imperfecciones antes de realizar operaciones sobre ellos. Los datos basados en texto, requieren particularmente un minucioso escrutinio. Profundiza en grandes conjuntos de datos basados en texto y ve si puedes descubrir áreas que podrían presentar sesgos o sentimientos sesgados en un modelo.

## [Examen posterior a la lección](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/38?loc=es)

## Revisión y autoestudio

Toma [esta ruta de aprendizaje de NLP](https://docs.microsoft.com/learn/paths/explore-natural-language-processing/?WT.mc_id=academic-77952-leestott) para descubrir herramientas a probar al construir modelos de voz y de gran cantidad de datos.

## Asignación

[NLTK](assignment.es.md)
