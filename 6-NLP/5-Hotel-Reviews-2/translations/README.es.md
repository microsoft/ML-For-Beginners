# Análisis de sentimiento con reseñas de hoteles

Ahora que has explorado a detalle el conjunto de datos, es momento de filtrar las columnas y luego usar técnicas de procesamiento del lenguaje natural sobre el conjunto de datos para obtener nuevos conocimientos acerca de los hoteles.

## [Examen previo a la lección](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/39?loc=es)

### Filtrado y operaciones de análisis de sentimiento

Como quizá ya has notado, el conjunto de datos presenta unos cuantos problemas. Algunas columnas contienen información inútil, otras parecen incorrectas. Si estas son correctas, no está claro cómo fueron calculadas, y las respuestas no pueden verificarse de forma independiente al realizar nuestros propios cálculos.

## Ejercicio: un poco más de procesamiento de datos

Limpia un poco más los datos. Agrega las columnas que usaremos más tarde, cambia los valores en otras columnas y elimina completamente ciertas columnas.

1. Procesamiento inicial de columnas

   1. Elimina `lat` y `lng`

   2. Reemplaza los valores de `Hotel_Address` con los siguientes valores (si la dirección contiene lo mismo que la ciudad y el país, cámbialo sólo a la ciudad y el país).

      Estas son las únicas ciudades y países en el conjunto de datos:

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

      Ahora puedes consultar datos a nivel de país:

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

2. Procesa las columnas Hotel Meta-review

  1. Elimina `Additional_Number_of_Scoring`

  1. Reemplaza `Total_Number_of_Reviews` con el número total de reseñas para ese hotel que realmente están en el conjunto de datos

  1. Reemplaza `Average_Score` con nuestros propio puntaje calculado

  ```python
  # Drop `Additional_Number_of_Scoring`
  df.drop(["Additional_Number_of_Scoring"], axis = 1, inplace=True)
  # Replace `Total_Number_of_Reviews` and `Average_Score` with our own calculated values
  df.Total_Number_of_Reviews = df.groupby('Hotel_Name').transform('count')
  df.Average_Score = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
  ```

3. Procesa las columnas review

   1. Elimina `Review_Total_Negative_Word_Counts`, `Review_Total_Positive_Word_Counts`, `Review_Date` y `days_since_review`

   2. Conserva `Reviewer_Score`, `Negative_Review`, y `Positive_Review` tal cual están,

   3. Conserva `Tags` por ahora

     - Realizaremos operaciones de filtrado adicionales sobre las etiquetas en la siguiente lección y luego se eliminarán etiquetas

4. Procesa las columnas reviewer

  1. Elimina `Total_Number_of_Reviews_Reviewer_Has_Given`

  2. Conserva `Reviewer_Nationality`

### Columnas de etiqueta (Tag)

La columna `Tag` es problemática ya que está almacenada como lista (en forma de texto) en la columna. Desafortunadamente el orden y número de subsecciones es esta columna no son siempre los mismos. Es difícil para un humano el identificar las frases correctas en las que estar interesado, ya que hay 515,000 filas, y 1427 hoteles, y cada uno tiene opciones ligeramente diferentes que un crítico podría elegir. Aquí es donde destaca el procesamiento del lenguaje natural. Puedes escanear texto y encontrar las frases más comunes, y contarlas.

Desafortunadamente, no estamos interesados en palabras simples, sino en frases multi-palabra (por ejemplo, *Business trip*). Ejecutar un algoritmo de distribución de frecuencia multi-palabra sobre tantos datos (6762646 palabras) podría tomar un largo período de tiempo, pero sin mirar los datos, estos parecería que es un gasto necesario. Aquí es donde el análisis exploratorio de datos es útil, porque ya has visto una muestra de estas etiquetas tales como `[' Business trip  ', ' Solo traveler ', ' Single Room ', ' Stayed 5 nights ', ' Submitted from  a mobile device ']` , puedes empezar a preguntar si es posible reducir ampliamente el procesamiento que tienes que hacer. Afortunadamente, así es - pero primero necesitas seguir algunos pasos para determinar las etiquetas de interés.

### Filtrar las etiquetas

Recuerda que el objetivo de los conjuntos de datos es agregar sentimiento y columnas que nos ayudarán a elegir el mejor hotel (para ti o quizá un cliente te pidió realizar un bot de recomendación de hoteles). Necesitas preguntarte si la etiquetas son útiles o no en el conjunto de datos final. Aquí hay una interpretación (si necesitabas el conjunto de datos por otras razones, distintas etiquetas podrían permanecer dentro/fuera de la selección):

1. El tipo de viaje es relevante, y así debería seguir
2. El tipo de grupo de huéspedes es importante, y así debería seguir
3. El tipo de cuarto, suite o estudio en que se quedó el huésped es irrelevante (todos los hoteles tienen básicamente los mismos cuartos)
4. El dispositivo desde el cual se envió la reseña es irrelevante
5. El número de noches que se hospedó el crítico *podría* ser relevante si atribuiste a las estancias largas con mayor gusto por el hotel, pero es exagerado, y probablemente irrelevante

En resumen, **conserva 2 tipos de etiquetas y elimina el resto**.

Primero, no quieres contar las etiquetas hasta que estén en un mejor formato, lo cual significa eliminar los corchetes y comillas. Puedes hacerlo de varias formas, pero quieres la más rápida ya que podría tomar un largo tiempo el procesar demasiados datos. Afortunadamente, pandas tiene una forma fácil de hacer cada uno de estos pasos.

```Python
# Remove opening and closing brackets
df.Tags = df.Tags.str.strip("[']")
# remove all quotes too
df.Tags = df.Tags.str.replace(" ', '", ",", regex = False)
```

Cada etiqueta se convierte en algo así: `Business trip, Solo traveler, Single Room, Stayed 5 nights, Submitted from a mobile device`.

Ahora encontramos un problema. Algunas reseñas o filas tienen 5 columnas, algunas 3, otras 6. Esto es resultado de cómo se creó el conjunto de datos, y es difícil de corregir. Quieres obtener un conteo de frecuencia de cada frase, pero están en distinto orden en cada reseña, ya que el conteo podría estar apagado y un hotel podría no obtener una etiqueta asignada como lo merece.

En su lugar, usarás el orden distinto a tu favor, ¡porque cada etiqueta es multi-palabra pero también está separada por una coma! La forma más simple de hacerlo es crear 6 columnas temporales con cada etiqueta insertada en la columna correspondiente a su orden en la etiqueta. Luego puedes unir las 6 columnas en una sola y ejecutar el método `value_counts()` en la columna resultante. Imprímelo, verás que había 2428 etiquetas únicas. Aquí una pequeña muestra:

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

Algunas de las etiquetas comunes como `Submitted from a mobile device` no nos son útiles, por lo que podría ser una buena idea quitarlas antes de contar la ocurrencia de las frases, pero es una operación tan rápida que puedes dejarlos e ignorarlos.

### Eliminando la longitud de las etiquetas de estadía

Eliminar estas etiquetas es el paso 1, reduce levemente el número total de etiquetas a ser consideradas. Nota que no las eliminas del conjunto de datos, sólo eliges eliminarlas de la consideración de valores a contar/mantener en el conjunto de datos de reseñas.

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

Existe una gran variedad de cuartos, suites, studios, departamentos, y así sucesivamente. Estos significan casi lo mismo y no te son relevantes, así que elimínalos de consideración.

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

Finalmente, y esto es placentero (porque no requirió casi nada de procesamiento), te quedarás con las siguientes etiquetas *útiles*:

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

Podrías argumentar que `Travellers with friends` es casi lo mismo que `Group`, y sería justo combinar las dos de arriba. El código para identificar las etiquetas correctas es [el notebook de Etiquetas](../solution/1-notebook.ipynb).

El último paso es crear nuevas columnas para cada una de estas etiquetas. Luego, para cada fila de reseña, si la columna `Tag` coincide con una de las nuevas columnas, agrega un 1, si no es así, agrega un 0. El resultado final será un conteo de cuántos críticos eligieron este hotel (en agregado), digamos, para negocios vs ocio, o para traer mascota, y esta es información útil al recomendar un hotel.

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

### Guarda tu archivo

Finalmente, guarda el conjunto de datos tal cual se encuentra ahora con un nuevo nombre.

```python
df.drop(["Review_Total_Negative_Word_Counts", "Review_Total_Positive_Word_Counts", "days_since_review", "Total_Number_of_Reviews_Reviewer_Has_Given"], axis = 1, inplace=True)

# Saving new data file with calculated columns
print("Saving results to Hotel_Reviews_Filtered.csv")
df.to_csv(r'../data/Hotel_Reviews_Filtered.csv', index = False)
```

## Operaciones de análisis de sentimiento

En esta sección final, aplicarás análisis de sentimiento a las columnas de reseña y guardarás los resultados en un conjunto de datos.

## Ejercicio: Carga y guarda los datos filtrados

Nota que ahora estás cargando el conjunto de datos filtrado que fue guardado en la sección anterior, **no** el conjunto de datos original.

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

### Eliminando stop words

Si fueses a ejecutar el análisis de sentimiento en las columnas de reseñas positivas y negativas, te llevaría mucho tiempo. Al probarlo en un laptop de pruebas poderosa con una CPU rápida, tomó de 12 - 14 minutos dependiendo en qué biblioteca de sentimientos fue usada. Eso es demasiado tiempo (relativamente), por lo que vale la pena investigar si se puede acelerar.

Eliminar stop words, o palabras comunes del Inglés que no cambian el sentimiento de una oración, es el primer paso. Al eliminarlas, el análisis de sentimiento debería ejecutarse más rápido, pero no sería menos preciso (ya que las stop words no afectan el sentimiento, pero sí ralentizan el análisis).

La reseña negativa más larga fue de 395 palabras, pero después de eliminar las stop words, es de 195 palabras.

Eliminar las stops words también es una operación rápida, remover las stop words de 2 columnas de reseñas de 515,000 filas tomó 3.3 segundos en el dispositivo de prueba, Podría tomar ligeramente más o menos tiempo dependiendo de la velocidad del CPU de tu dispositivo, la RAM, si tienes un disco de estado sólido o no, entre otros factores. La relatividad de corto período de operación significa que si se mejora el tiempo de análisis de sentimiento, entonces vale la pena hacerlo.

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

### Realizando análisis de sentimiento

Ahora deberías calcular el análisis de sentimiento tanto para las columnas de reseñas positivas como negativas y almacenar el resultado en 2 nuevas columnas. La prueba del sentimiento será compararlo con el puntaje del crítico para la misma reseña. Por ejemplo, si el sentimiento piensa que la reseña negativa tuvo un sentimiento de 1 (sentimiento extremadamente positivo) y un sentimiento de reseña positiva de 1, pero el crítico le dió al hotel el puntaje más bajo posible, luego que el texto de la reseña no coincide con el puntaje o el analizador de sentimiento no pudo reconocer el sentimiento de forma correcta. Deberías esperar que algunos puntajes de sentimiento sean completamente erróneos, y a menudo será explicable, por ejemplo, la reseña podría ser extremadamente sarcástica "Of course I LOVED sleeping in a room with no heating" y el analizador de sentimiento piensa que es un sentimiento positivo, aunque un humano leyéndolo sabría que fue sarcasmo.

NLTK provee distintos analizadores de sentimiento con los cuales aprender, y puedes sustituirlos y así ver si el sentimiento es más o menos preciso. El análisis de sentimiento VADER se usó aquí.

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

Después en tu programa cuando estés listo para calcular el sentimiento, puedes aplicarlo a ca reseña como sigue:

```python
# Add a negative sentiment and positive sentiment column
print("Calculating sentiment columns for both positive and negative reviews")
start = time.time()
df["Negative_Sentiment"] = df.Negative_Review.apply(calc_sentiment)
df["Positive_Sentiment"] = df.Positive_Review.apply(calc_sentiment)
end = time.time()
print("Calculating sentiment took " + str(round(end - start, 2)) + " seconds")
```

Esto toma aproximadamente 120 segundos en mi computadora, pero varía en cada equipo. Si quieres imprimir los resultados y ver si el sentimiento coincide con la reseña:

```python
df = df.sort_values(by=["Negative_Sentiment"], ascending=True)
print(df[["Negative_Review", "Negative_Sentiment"]])
df = df.sort_values(by=["Positive_Sentiment"], ascending=True)
print(df[["Positive_Review", "Positive_Sentiment"]])
```

¡Lo último por hacer con el archivo antes de usarlo en el desafío, es guardarlo! También deberías considerar reordenar todas tus nuevas columnas para que sean fáciles de usar (para un humano, es un cambio cosmético).

```python
# Reorder the columns (This is cosmetic, but to make it easier to explore the data later)
df = df.reindex(["Hotel_Name", "Hotel_Address", "Total_Number_of_Reviews", "Average_Score", "Reviewer_Score", "Negative_Sentiment", "Positive_Sentiment", "Reviewer_Nationality", "Leisure_trip", "Couple", "Solo_traveler", "Business_trip", "Group", "Family_with_young_children", "Family_with_older_children", "With_a_pet", "Negative_Review", "Positive_Review"], axis=1)

print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r"../data/Hotel_Reviews_NLP.csv", index = False)
```

Deberías ejecutar el código completo del [notebook de análisis](../solution/3-notebook.ipynb) (después de que hayas ejecutado [tu notebook de filtrado](../solution/1-notebook.ipynb) para generar el archivo Hotel_Reviews_Filtered.csv).

Para revisar, los pasos son:

1. Se exploró el archivo original del conjunto de datos **Hotel_Reviews.csv** en la lección anterior con [el notebook explorador](../../4-Hotel-Reviews-1/solution/notebook.ipynb)
2. Se filtró Hotel_Reviews.csv con [el notebook de filtrado](../solution/1-notebook.ipynb) resultando el archivo **Hotel_Reviews_Filtered.csv**
3. Se procesó Hotel_Reviews_Filtered.csv con [el notebook de análisis de sentimiento](../solution/3-notebook.ipynb) obteniendo como resultado **Hotel_Reviews_NLP.csv**
4. Usa Hotel_Reviews_NLP.csv en el desafío NLP de abajo

### Conclusión

Cuando iniciaste, tenías un conjunto de datos con columnas y datos pero no todos ello podían ser verificados o usados. Exploraste los datos, filtraste lo que no necesitas, convertiste etiquetas en algo útil, calculaste tus propios promedios, agregaste algunas columnas de sentimiento y espero hayas aprendido cosas interesantes acerca de procesar texto natural.

## [Examen posterior a la lección](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/40?loc=es)

## Desafío

Ahora que tienes tu conjunto de datos analizado por sentimiento, observa si puedes usar las estrategias que aprendiste en este plan de estudios (agrupamiento, ¿quizá?) para determinar patrones alrededor del sentimiento.

## Revisión y autoestudio

Toma [este módulo de aprendizaje](https://docs.microsoft.com/en-us/learn/modules/classify-user-feedback-with-the-text-analytics-api/?WT.mc_id=academic-77952-leestott) para aprender más y usar distintas herramientas para explorar el sentimiento en el texto.

## Asignación

[Prueba distintos conjuntos de datos](assignment.es.md)
