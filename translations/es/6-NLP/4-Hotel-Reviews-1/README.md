<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "3c4738bb0836dd838c552ab9cab7e09d",
  "translation_date": "2025-09-04T00:41:00+00:00",
  "source_file": "6-NLP/4-Hotel-Reviews-1/README.md",
  "language_code": "es"
}
-->
# Análisis de sentimientos con reseñas de hoteles - procesando los datos

En esta sección, utilizarás las técnicas de las lecciones anteriores para realizar un análisis exploratorio de datos en un conjunto de datos grande. Una vez que tengas una buena comprensión de la utilidad de las diferentes columnas, aprenderás:

- cómo eliminar las columnas innecesarias
- cómo calcular nuevos datos basados en las columnas existentes
- cómo guardar el conjunto de datos resultante para usarlo en el desafío final

## [Cuestionario previo a la lección](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/37/)

### Introducción

Hasta ahora has aprendido que los datos de texto son bastante diferentes de los datos numéricos. Si el texto fue escrito o hablado por un humano, puede analizarse para encontrar patrones, frecuencias, sentimientos y significados. Esta lección te lleva a un conjunto de datos real con un desafío real: **[515K Hotel Reviews Data in Europe](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe)**, que incluye una [licencia CC0: Dominio Público](https://creativecommons.org/publicdomain/zero/1.0/). Fue recopilado de Booking.com a partir de fuentes públicas. El creador del conjunto de datos es Jiashen Liu.

### Preparación

Necesitarás:

* La capacidad de ejecutar notebooks .ipynb usando Python 3
* pandas
* NLTK, [que deberías instalar localmente](https://www.nltk.org/install.html)
* El conjunto de datos disponible en Kaggle [515K Hotel Reviews Data in Europe](https://www.kaggle.com/jiashenliu/515k-hotel-reviews-data-in-europe). Tiene un tamaño aproximado de 230 MB descomprimido. Descárgalo en la carpeta raíz `/data` asociada con estas lecciones de NLP.

## Análisis exploratorio de datos

Este desafío asume que estás construyendo un bot de recomendación de hoteles utilizando análisis de sentimientos y puntuaciones de reseñas de huéspedes. El conjunto de datos que usarás incluye reseñas de 1493 hoteles diferentes en 6 ciudades.

Usando Python, un conjunto de datos de reseñas de hoteles y el análisis de sentimientos de NLTK, podrías descubrir:

* ¿Cuáles son las palabras y frases más utilizadas en las reseñas?
* ¿Las *etiquetas* oficiales que describen un hotel se correlacionan con las puntuaciones de las reseñas (por ejemplo, hay más reseñas negativas para un hotel en particular por *Familia con niños pequeños* que por *Viajero solo*, lo que podría indicar que es mejor para *Viajeros solos*)?
* ¿Las puntuaciones de sentimiento de NLTK "coinciden" con la puntuación numérica del revisor del hotel?

#### Conjunto de datos

Exploremos el conjunto de datos que has descargado y guardado localmente. Abre el archivo en un editor como VS Code o incluso Excel.

Los encabezados en el conjunto de datos son los siguientes:

*Hotel_Address, Additional_Number_of_Scoring, Review_Date, Average_Score, Hotel_Name, Reviewer_Nationality, Negative_Review, Review_Total_Negative_Word_Counts, Total_Number_of_Reviews, Positive_Review, Review_Total_Positive_Word_Counts, Total_Number_of_Reviews_Reviewer_Has_Given, Reviewer_Score, Tags, days_since_review, lat, lng*

Aquí están agrupados de una manera que podría ser más fácil de examinar: 
##### Columnas del hotel

* `Hotel_Name`, `Hotel_Address`, `lat` (latitud), `lng` (longitud)
  * Usando *lat* y *lng* podrías trazar un mapa con Python mostrando las ubicaciones de los hoteles (quizás codificado por colores para reseñas negativas y positivas).
  * Hotel_Address no parece ser útil para nosotros, y probablemente lo reemplazaremos con un país para facilitar la clasificación y búsqueda.

**Columnas de meta-reseñas del hotel**

* `Average_Score`
  * Según el creador del conjunto de datos, esta columna es la *Puntuación promedio del hotel, calculada en base al último comentario del último año*. Esto parece una forma inusual de calcular la puntuación, pero es el dato recopilado, así que por ahora lo tomaremos como válido.
  
  ✅ Basándote en las otras columnas de este conjunto de datos, ¿puedes pensar en otra forma de calcular la puntuación promedio?

* `Total_Number_of_Reviews`
  * El número total de reseñas que ha recibido este hotel - no está claro (sin escribir algo de código) si esto se refiere a las reseñas en el conjunto de datos.
* `Additional_Number_of_Scoring`
  * Esto significa que se dio una puntuación pero el revisor no escribió una reseña positiva o negativa.

**Columnas de reseñas**

- `Reviewer_Score`
  - Este es un valor numérico con un máximo de 1 decimal entre los valores mínimos y máximos 2.5 y 10.
  - No se explica por qué 2.5 es la puntuación más baja posible.
- `Negative_Review`
  - Si un revisor no escribió nada, este campo tendrá "**No Negative**".
  - Ten en cuenta que un revisor puede escribir una reseña positiva en la columna de reseñas negativas (por ejemplo, "no hay nada malo en este hotel").
- `Review_Total_Negative_Word_Counts`
  - Un mayor conteo de palabras negativas indica una puntuación más baja (sin verificar la sentimentalidad).
- `Positive_Review`
  - Si un revisor no escribió nada, este campo tendrá "**No Positive**".
  - Ten en cuenta que un revisor puede escribir una reseña negativa en la columna de reseñas positivas (por ejemplo, "no hay nada bueno en este hotel en absoluto").
- `Review_Total_Positive_Word_Counts`
  - Un mayor conteo de palabras positivas indica una puntuación más alta (sin verificar la sentimentalidad).
- `Review_Date` y `days_since_review`
  - Se podría aplicar una medida de frescura o antigüedad a una reseña (las reseñas más antiguas podrían no ser tan precisas como las más recientes debido a cambios en la administración del hotel, renovaciones, adición de una piscina, etc.).
- `Tags`
  - Estas son descripciones breves que un revisor puede seleccionar para describir el tipo de huésped que era (por ejemplo, solo o en familia), el tipo de habitación que tenía, la duración de la estancia y cómo se envió la reseña.
  - Desafortunadamente, usar estas etiquetas es problemático, revisa la sección a continuación que discute su utilidad.

**Columnas del revisor**

- `Total_Number_of_Reviews_Reviewer_Has_Given`
  - Esto podría ser un factor en un modelo de recomendación, por ejemplo, si pudieras determinar que los revisores más prolíficos con cientos de reseñas eran más propensos a ser negativos que positivos. Sin embargo, el revisor de cualquier reseña en particular no está identificado con un código único, y por lo tanto no puede vincularse a un conjunto de reseñas. Hay 30 revisores con 100 o más reseñas, pero es difícil ver cómo esto puede ayudar al modelo de recomendación.
- `Reviewer_Nationality`
  - Algunas personas podrían pensar que ciertas nacionalidades son más propensas a dar una reseña positiva o negativa debido a una inclinación nacional. Ten cuidado al construir este tipo de puntos de vista anecdóticos en tus modelos. Estos son estereotipos nacionales (y a veces raciales), y cada revisor fue un individuo que escribió una reseña basada en su experiencia. Esta pudo haber sido filtrada a través de muchas perspectivas, como sus estancias previas en hoteles, la distancia recorrida y su temperamento personal. Pensar que su nacionalidad fue la razón de una puntuación de reseña es difícil de justificar.

##### Ejemplos

| Average  Score | Total Number   Reviews | Reviewer   Score | Negative <br />Review                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Positive   Review                 | Tags                                                                                      |
| -------------- | ---------------------- | ---------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- | ----------------------------------------------------------------------------------------- |
| 7.8            | 1945                   | 2.5              | Este no es actualmente un hotel sino un sitio de construcción. Fui aterrorizado desde temprano en la mañana y todo el día con ruidos de construcción inaceptables mientras descansaba después de un largo viaje y trabajaba en la habitación. La gente trabajaba todo el día, es decir, con martillos neumáticos en las habitaciones adyacentes. Pedí un cambio de habitación pero no había ninguna habitación silenciosa disponible. Para empeorar las cosas, me cobraron de más. Me fui en la noche ya que tenía un vuelo muy temprano y recibí una factura adecuada. Un día después, el hotel hizo otro cargo sin mi consentimiento por encima del precio reservado. Es un lugar terrible. No te castigues reservando aquí. | Nada. Lugar terrible. Mantente alejado. | Viaje de negocios. Pareja. Habitación doble estándar. Estancia de 2 noches. |

Como puedes ver, este huésped no tuvo una estancia feliz en este hotel. El hotel tiene una buena puntuación promedio de 7.8 y 1945 reseñas, pero este revisor le dio un 2.5 y escribió 115 palabras sobre lo negativa que fue su estancia. Si no escribió nada en absoluto en la columna Positive_Review, podrías deducir que no hubo nada positivo, pero aún así escribió 7 palabras de advertencia. Si solo contáramos palabras en lugar del significado o sentimiento de las palabras, podríamos tener una visión sesgada de la intención del revisor. Curiosamente, su puntuación de 2.5 es confusa, porque si esa estancia en el hotel fue tan mala, ¿por qué darle algún punto? Al investigar el conjunto de datos de cerca, verás que la puntuación más baja posible es 2.5, no 0. La puntuación más alta posible es 10.

##### Tags

Como se mencionó anteriormente, a primera vista, la idea de usar `Tags` para categorizar los datos tiene sentido. Desafortunadamente, estas etiquetas no están estandarizadas, lo que significa que en un hotel dado, las opciones podrían ser *Single room*, *Twin room* y *Double room*, pero en el siguiente hotel, son *Deluxe Single Room*, *Classic Queen Room* y *Executive King Room*. Estas podrían ser las mismas cosas, pero hay tantas variaciones que la elección se convierte en:

1. Intentar cambiar todos los términos a un estándar único, lo cual es muy difícil, porque no está claro cuál sería la ruta de conversión en cada caso (por ejemplo, *Classic single room* se mapea a *Single room* pero *Superior Queen Room with Courtyard Garden or City View* es mucho más difícil de mapear).

1. Podemos tomar un enfoque de NLP y medir la frecuencia de ciertos términos como *Solo*, *Viajero de negocios* o *Familia con niños pequeños* según se aplican a cada hotel, y factorizar eso en la recomendación.

Las etiquetas suelen (pero no siempre) ser un solo campo que contiene una lista de 5 a 6 valores separados por comas que se alinean con *Tipo de viaje*, *Tipo de huéspedes*, *Tipo de habitación*, *Número de noches* y *Tipo de dispositivo en el que se envió la reseña*. Sin embargo, debido a que algunos revisores no completan cada campo (pueden dejar uno en blanco), los valores no siempre están en el mismo orden.

Como ejemplo, toma *Tipo de grupo*. Hay 1025 posibilidades únicas en este campo en la columna `Tags`, y desafortunadamente solo algunas de ellas se refieren a un grupo (algunas son el tipo de habitación, etc.). Si filtras solo las que mencionan familia, los resultados contienen muchos valores de tipo *Family room*. Si incluyes el término *with*, es decir, cuentas los valores *Family with*, los resultados son mejores, con más de 80,000 de los 515,000 resultados que contienen la frase "Family with young children" o "Family with older children".

Esto significa que la columna de etiquetas no es completamente inútil para nosotros, pero requerirá algo de trabajo para hacerla útil.

##### Puntuación promedio del hotel

Hay una serie de rarezas o discrepancias con el conjunto de datos que no puedo resolver, pero se ilustran aquí para que estés al tanto de ellas al construir tus modelos. Si las resuelves, por favor háznoslo saber en la sección de discusión.

El conjunto de datos tiene las siguientes columnas relacionadas con la puntuación promedio y el número de reseñas:

1. Hotel_Name
2. Additional_Number_of_Scoring
3. Average_Score
4. Total_Number_of_Reviews
5. Reviewer_Score  

El único hotel con más reseñas en este conjunto de datos es *Britannia International Hotel Canary Wharf* con 4789 reseñas de 515,000. Pero si miramos el valor de `Total_Number_of_Reviews` para este hotel, es 9086. Podrías deducir que hay muchas más puntuaciones sin reseñas, así que tal vez deberíamos sumar el valor de la columna `Additional_Number_of_Scoring`. Ese valor es 2682, y al sumarlo a 4789 obtenemos 7471, lo cual sigue siendo 1615 menos que el valor de `Total_Number_of_Reviews`.

Si tomas la columna `Average_Score`, podrías deducir que es el promedio de las reseñas en el conjunto de datos, pero la descripción de Kaggle es "*Puntuación promedio del hotel, calculada en base al último comentario del último año*". Esto no parece tan útil, pero podemos calcular nuestro propio promedio basado en las puntuaciones de las reseñas en el conjunto de datos. Usando el mismo hotel como ejemplo, la puntuación promedio del hotel se da como 7.1 pero la puntuación calculada (promedio de las puntuaciones de los revisores *en* el conjunto de datos) es 6.8. Esto es cercano, pero no el mismo valor, y solo podemos suponer que las puntuaciones dadas en las reseñas de `Additional_Number_of_Scoring` aumentaron el promedio a 7.1. Desafortunadamente, sin forma de probar o demostrar esa afirmación, es difícil usar o confiar en `Average_Score`, `Additional_Number_of_Scoring` y `Total_Number_of_Reviews` cuando se basan en, o se refieren a, datos que no tenemos.

Para complicar aún más las cosas, el hotel con el segundo mayor número de reseñas tiene una puntuación promedio calculada de 8.12 y la `Average_Score` del conjunto de datos es 8.1. ¿Es esta puntuación correcta una coincidencia o es el primer hotel una discrepancia?
Sobre la posibilidad de que este hotel sea un caso atípico, y que tal vez la mayoría de los valores coincidan (pero algunos no por alguna razón), escribiremos un programa breve para explorar los valores en el conjunto de datos y determinar el uso correcto (o no uso) de los valores.

> 🚨 Una nota de precaución
>
> Al trabajar con este conjunto de datos, escribirás código que calcula algo a partir del texto sin necesidad de leer o analizar el texto tú mismo. Esta es la esencia del NLP: interpretar significado o sentimiento sin que un humano tenga que hacerlo. Sin embargo, es posible que leas algunas de las reseñas negativas. Te insto a que no lo hagas, porque no es necesario. Algunas son absurdas o irrelevantes, como "El clima no fue bueno", algo fuera del control del hotel o de cualquier persona. Pero también hay un lado oscuro en algunas reseñas. A veces las reseñas negativas son racistas, sexistas o discriminatorias por edad. Esto es desafortunado pero esperable en un conjunto de datos extraído de un sitio web público. Algunos usuarios dejan reseñas que podrían resultarte desagradables, incómodas o perturbadoras. Es mejor dejar que el código mida el sentimiento en lugar de leerlas tú mismo y molestarte. Dicho esto, es una minoría la que escribe este tipo de cosas, pero existen de todos modos.

## Ejercicio - Exploración de datos
### Cargar los datos

Ya es suficiente examinar los datos visualmente, ahora escribirás algo de código y obtendrás respuestas. Esta sección utiliza la biblioteca pandas. Tu primera tarea es asegurarte de que puedes cargar y leer los datos en formato CSV. La biblioteca pandas tiene un cargador rápido de CSV, y el resultado se coloca en un dataframe, como en lecciones anteriores. El CSV que estamos cargando tiene más de medio millón de filas, pero solo 17 columnas. Pandas te ofrece muchas formas poderosas de interactuar con un dataframe, incluida la capacidad de realizar operaciones en cada fila.

A partir de aquí, en esta lección habrá fragmentos de código y algunas explicaciones del código, así como discusiones sobre lo que significan los resultados. Usa el archivo _notebook.ipynb_ incluido para tu código.

Comencemos cargando el archivo de datos que usarás:

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

Ahora que los datos están cargados, podemos realizar algunas operaciones sobre ellos. Mantén este código en la parte superior de tu programa para la siguiente parte.

## Explorar los datos

En este caso, los datos ya están *limpios*, lo que significa que están listos para trabajar y no tienen caracteres en otros idiomas que puedan causar problemas a los algoritmos que esperan solo caracteres en inglés.

✅ Es posible que tengas que trabajar con datos que requieran un procesamiento inicial para formatearlos antes de aplicar técnicas de NLP, pero no en esta ocasión. Si tuvieras que hacerlo, ¿cómo manejarías caracteres que no están en inglés?

Tómate un momento para asegurarte de que, una vez cargados los datos, puedes explorarlos con código. Es muy fácil querer centrarse en las columnas `Negative_Review` y `Positive_Review`. Están llenas de texto natural para que tus algoritmos de NLP lo procesen. Pero espera, antes de saltar al NLP y al análisis de sentimientos, deberías seguir el código a continuación para verificar si los valores dados en el conjunto de datos coinciden con los valores que calculas con pandas.

## Operaciones con el dataframe

La primera tarea en esta lección es verificar si las siguientes afirmaciones son correctas escribiendo código que examine el dataframe (sin modificarlo).

> Como en muchas tareas de programación, hay varias formas de completarlas, pero un buen consejo es hacerlo de la manera más simple y fácil posible, especialmente si será más fácil de entender cuando vuelvas a este código en el futuro. Con dataframes, hay una API completa que a menudo tendrá una forma eficiente de hacer lo que necesitas.

Trata las siguientes preguntas como tareas de codificación e intenta responderlas sin mirar la solución.

1. Imprime la *forma* del dataframe que acabas de cargar (la forma es el número de filas y columnas).
2. Calcula la frecuencia de las nacionalidades de los revisores:
   1. ¿Cuántos valores distintos hay en la columna `Reviewer_Nationality` y cuáles son?
   2. ¿Qué nacionalidad de revisor es la más común en el conjunto de datos (imprime el país y el número de reseñas)?
   3. ¿Cuáles son las siguientes 10 nacionalidades más frecuentes y su conteo de frecuencia?
3. ¿Cuál fue el hotel más reseñado por cada una de las 10 nacionalidades de revisores más frecuentes?
4. ¿Cuántas reseñas hay por hotel (conteo de frecuencia de hotel) en el conjunto de datos?
5. Aunque hay una columna `Average_Score` para cada hotel en el conjunto de datos, también puedes calcular un puntaje promedio (obteniendo el promedio de todos los puntajes de los revisores en el conjunto de datos para cada hotel). Agrega una nueva columna a tu dataframe con el encabezado `Calc_Average_Score` que contenga ese promedio calculado.
6. ¿Hay hoteles que tengan el mismo `Average_Score` (redondeado a 1 decimal) y `Calc_Average_Score`?
   1. Intenta escribir una función en Python que tome una Serie (fila) como argumento y compare los valores, imprimiendo un mensaje cuando los valores no sean iguales. Luego usa el método `.apply()` para procesar cada fila con la función.
7. Calcula e imprime cuántas filas tienen valores de columna `Negative_Review` iguales a "No Negative".
8. Calcula e imprime cuántas filas tienen valores de columna `Positive_Review` iguales a "No Positive".
9. Calcula e imprime cuántas filas tienen valores de columna `Positive_Review` iguales a "No Positive" **y** valores de columna `Negative_Review` iguales a "No Negative".

### Respuestas en código

1. Imprime la *forma* del dataframe que acabas de cargar (la forma es el número de filas y columnas).

   ```python
   print("The shape of the data (rows, cols) is " + str(df.shape))
   > The shape of the data (rows, cols) is (515738, 17)
   ```

2. Calcula la frecuencia de las nacionalidades de los revisores:

   1. ¿Cuántos valores distintos hay en la columna `Reviewer_Nationality` y cuáles son?
   2. ¿Qué nacionalidad de revisor es la más común en el conjunto de datos (imprime el país y el número de reseñas)?

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

   3. ¿Cuáles son las siguientes 10 nacionalidades más frecuentes y su conteo de frecuencia?

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

3. ¿Cuál fue el hotel más reseñado por cada una de las 10 nacionalidades de revisores más frecuentes?

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

4. ¿Cuántas reseñas hay por hotel (conteo de frecuencia de hotel) en el conjunto de datos?

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
   
   Puede que notes que los resultados *contados en el conjunto de datos* no coinciden con el valor en `Total_Number_of_Reviews`. No está claro si este valor en el conjunto de datos representa el número total de reseñas que tuvo el hotel, pero no todas fueron extraídas, o algún otro cálculo. `Total_Number_of_Reviews` no se utiliza en el modelo debido a esta falta de claridad.

5. Aunque hay una columna `Average_Score` para cada hotel en el conjunto de datos, también puedes calcular un puntaje promedio (obteniendo el promedio de todos los puntajes de los revisores en el conjunto de datos para cada hotel). Agrega una nueva columna a tu dataframe con el encabezado `Calc_Average_Score` que contenga ese promedio calculado. Imprime las columnas `Hotel_Name`, `Average_Score` y `Calc_Average_Score`.

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

   También podrías preguntarte sobre el valor de `Average_Score` y por qué a veces es diferente del puntaje promedio calculado. Como no podemos saber por qué algunos valores coinciden, pero otros tienen una diferencia, es más seguro en este caso usar los puntajes de las reseñas que tenemos para calcular el promedio nosotros mismos. Dicho esto, las diferencias suelen ser muy pequeñas. Aquí están los hoteles con la mayor desviación entre el promedio del conjunto de datos y el promedio calculado:

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

   Con solo 1 hotel que tiene una diferencia de puntaje mayor a 1, significa que probablemente podamos ignorar la diferencia y usar el puntaje promedio calculado.

6. Calcula e imprime cuántas filas tienen valores de columna `Negative_Review` iguales a "No Negative".

7. Calcula e imprime cuántas filas tienen valores de columna `Positive_Review` iguales a "No Positive".

8. Calcula e imprime cuántas filas tienen valores de columna `Positive_Review` iguales a "No Positive" **y** valores de columna `Negative_Review` iguales a "No Negative".

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

## Otra forma

Otra forma de contar elementos sin Lambdas, y usar sum para contar las filas:

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

   Puede que hayas notado que hay 127 filas que tienen valores "No Negative" y "No Positive" en las columnas `Negative_Review` y `Positive_Review`, respectivamente. Esto significa que el revisor dio al hotel un puntaje numérico, pero se negó a escribir una reseña positiva o negativa. Afortunadamente, esta es una pequeña cantidad de filas (127 de 515738, o 0.02%), por lo que probablemente no sesgará nuestro modelo o resultados en ninguna dirección en particular. Sin embargo, podrías no haber esperado que un conjunto de datos de reseñas tuviera filas sin reseñas, por lo que vale la pena explorar los datos para descubrir filas como esta.

Ahora que has explorado el conjunto de datos, en la próxima lección filtrarás los datos y agregarás análisis de sentimientos.

---
## 🚀Desafío

Esta lección demuestra, como vimos en lecciones anteriores, lo críticamente importante que es entender tus datos y sus peculiaridades antes de realizar operaciones sobre ellos. Los datos basados en texto, en particular, requieren un escrutinio cuidadoso. Explora varios conjuntos de datos ricos en texto y ve si puedes descubrir áreas que podrían introducir sesgos o sentimientos sesgados en un modelo.

## [Cuestionario posterior a la lección](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/38/)

## Revisión y autoestudio

Toma [este Learning Path sobre NLP](https://docs.microsoft.com/learn/paths/explore-natural-language-processing/?WT.mc_id=academic-77952-leestott) para descubrir herramientas que puedes probar al construir modelos basados en texto y voz.

## Tarea 

[NLTK](assignment.md)

---

**Descargo de responsabilidad**:  
Este documento ha sido traducido utilizando el servicio de traducción automática [Co-op Translator](https://github.com/Azure/co-op-translator). Si bien nos esforzamos por lograr precisión, tenga en cuenta que las traducciones automáticas pueden contener errores o imprecisiones. El documento original en su idioma nativo debe considerarse como la fuente autorizada. Para información crítica, se recomienda una traducción profesional realizada por humanos. No nos hacemos responsables de malentendidos o interpretaciones erróneas que puedan surgir del uso de esta traducción.