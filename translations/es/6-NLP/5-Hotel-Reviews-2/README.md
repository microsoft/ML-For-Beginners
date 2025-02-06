# Análisis de sentimiento con reseñas de hoteles

Ahora que has explorado el conjunto de datos en detalle, es hora de filtrar las columnas y luego usar técnicas de PLN en el conjunto de datos para obtener nuevas perspectivas sobre los hoteles.
## [Cuestionario previo a la lección](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/39/)

### Operaciones de filtrado y análisis de sentimiento

Como probablemente has notado, el conjunto de datos tiene algunos problemas. Algunas columnas están llenas de información inútil, otras parecen incorrectas. Si son correctas, no está claro cómo fueron calculadas, y las respuestas no pueden ser verificadas independientemente por tus propios cálculos.

## Ejercicio: un poco más de procesamiento de datos

Limpia los datos un poco más. Agrega columnas que serán útiles más adelante, cambia los valores en otras columnas y elimina ciertas columnas por completo.

1. Procesamiento inicial de columnas

   1. Elimina `lat` y `lng`

   2. Reemplaza los valores de `Hotel_Address` con los siguientes valores (si la dirección contiene el nombre de la ciudad y el país, cámbialo solo por la ciudad y el país).

      Estas son las únicas ciudades y países en el conjunto de datos:

      Ámsterdam, Países Bajos

      Barcelona, España

      Londres, Reino Unido

      Milán, Italia

      París, Francia

      Viena, Austria 

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

2. Procesa las columnas de Meta-reseñas del hotel

  1. Elimina `Additional_Number_of_Scoring`

  1. Replace `Total_Number_of_Reviews` with the total number of reviews for that hotel that are actually in the dataset 

  1. Replace `Average_Score` con nuestro propio puntaje calculado

  ```python
  # Drop `Additional_Number_of_Scoring`
  df.drop(["Additional_Number_of_Scoring"], axis = 1, inplace=True)
  # Replace `Total_Number_of_Reviews` and `Average_Score` with our own calculated values
  df.Total_Number_of_Reviews = df.groupby('Hotel_Name').transform('count')
  df.Average_Score = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
  ```

3. Procesa las columnas de reseñas

   1. Elimina `Review_Total_Negative_Word_Counts`, `Review_Total_Positive_Word_Counts`, `Review_Date` and `days_since_review`

   2. Keep `Reviewer_Score`, `Negative_Review`, and `Positive_Review` as they are,
     
   3. Keep `Tags` for now

     - We'll be doing some additional filtering operations on the tags in the next section and then tags will be dropped

4. Process reviewer columns

  1. Drop `Total_Number_of_Reviews_Reviewer_Has_Given`
  
  2. Keep `Reviewer_Nationality`

### Tag columns

The `Tag` column is problematic as it is a list (in text form) stored in the column. Unfortunately the order and number of sub sections in this column are not always the same. It's hard for a human to identify the correct phrases to be interested in, because there are 515,000 rows, and 1427 hotels, and each has slightly different options a reviewer could choose. This is where NLP shines. You can scan the text and find the most common phrases, and count them.

Unfortunately, we are not interested in single words, but multi-word phrases (e.g. *Business trip*). Running a multi-word frequency distribution algorithm on that much data (6762646 words) could take an extraordinary amount of time, but without looking at the data, it would seem that is a necessary expense. This is where exploratory data analysis comes in useful, because you've seen a sample of the tags such as `[' Business trip  ', ' Solo traveler ', ' Single Room ', ' Stayed 5 nights ', ' Submitted from  a mobile device ']`, puedes comenzar a preguntarte si es posible reducir significativamente el procesamiento que tienes que hacer. Afortunadamente, es posible, pero primero necesitas seguir algunos pasos para determinar las etiquetas de interés.

### Filtrando etiquetas

Recuerda que el objetivo del conjunto de datos es agregar sentimiento y columnas que te ayudarán a elegir el mejor hotel (para ti o tal vez para un cliente que te pide que hagas un bot de recomendación de hoteles). Necesitas preguntarte si las etiquetas son útiles o no en el conjunto de datos final. Aquí hay una interpretación (si necesitaras el conjunto de datos por otras razones, diferentes etiquetas podrían quedarse o salir de la selección):

1. El tipo de viaje es relevante, y eso debería quedarse
2. El tipo de grupo de huéspedes es importante, y eso debería quedarse
3. El tipo de habitación, suite o estudio en el que se alojó el huésped es irrelevante (todos los hoteles tienen básicamente las mismas habitaciones)
4. El dispositivo desde el cual se envió la reseña es irrelevante
5. El número de noches que el revisor se quedó *podría* ser relevante si atribuyes estancias más largas a que les gustó más el hotel, pero es un poco exagerado y probablemente irrelevante

En resumen, **mantén 2 tipos de etiquetas y elimina las demás**.

Primero, no quieres contar las etiquetas hasta que estén en un mejor formato, lo que significa eliminar los corchetes y las comillas. Puedes hacer esto de varias maneras, pero quieres la más rápida ya que podría llevar mucho tiempo procesar muchos datos. Afortunadamente, pandas tiene una manera fácil de hacer cada uno de estos pasos.

```Python
# Remove opening and closing brackets
df.Tags = df.Tags.str.strip("[']")
# remove all quotes too
df.Tags = df.Tags.str.replace(" ', '", ",", regex = False)
```

Cada etiqueta se convierte en algo como: `Business trip, Solo traveler, Single Room, Stayed 5 nights, Submitted from a mobile device`. 

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

The final step is to create new columns for each of these tags. Then, for every review row, if the `Tag` columna coincide con una de las nuevas columnas, agrega un 1, si no, agrega un 0. El resultado final será un recuento de cuántos revisores eligieron este hotel (en conjunto) para, por ejemplo, negocios vs ocio, o para llevar una mascota, y esta es información útil al recomendar un hotel.

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

Finalmente, guarda el conjunto de datos tal como está ahora con un nuevo nombre.

```python
df.drop(["Review_Total_Negative_Word_Counts", "Review_Total_Positive_Word_Counts", "days_since_review", "Total_Number_of_Reviews_Reviewer_Has_Given"], axis = 1, inplace=True)

# Saving new data file with calculated columns
print("Saving results to Hotel_Reviews_Filtered.csv")
df.to_csv(r'../data/Hotel_Reviews_Filtered.csv', index = False)
```

## Operaciones de análisis de sentimiento

En esta sección final, aplicarás análisis de sentimiento a las columnas de reseñas y guardarás los resultados en un conjunto de datos.

## Ejercicio: carga y guarda los datos filtrados

Ten en cuenta que ahora estás cargando el conjunto de datos filtrado que se guardó en la sección anterior, **no** el conjunto de datos original.

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

### Eliminando palabras vacías

Si fueras a ejecutar el Análisis de Sentimiento en las columnas de reseñas negativas y positivas, podría llevar mucho tiempo. Probado en un portátil de prueba potente con CPU rápida, tomó 12 - 14 minutos dependiendo de la biblioteca de sentimiento utilizada. Eso es un tiempo (relativamente) largo, por lo que vale la pena investigar si se puede acelerar.

Eliminar palabras vacías, o palabras comunes en inglés que no cambian el sentimiento de una oración, es el primer paso. Al eliminarlas, el análisis de sentimiento debería ejecutarse más rápido, pero no ser menos preciso (ya que las palabras vacías no afectan el sentimiento, pero sí ralentizan el análisis).

La reseña negativa más larga tenía 395 palabras, pero después de eliminar las palabras vacías, tiene 195 palabras.

Eliminar las palabras vacías también es una operación rápida, eliminar las palabras vacías de 2 columnas de reseñas sobre 515,000 filas tomó 3.3 segundos en el dispositivo de prueba. Podría tomar un poco más o menos tiempo para ti dependiendo de la velocidad de tu CPU, RAM, si tienes un SSD o no, y algunos otros factores. La relativa brevedad de la operación significa que si mejora el tiempo de análisis de sentimiento, entonces vale la pena hacerlo.

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

Ahora deberías calcular el análisis de sentimiento para ambas columnas de reseñas negativas y positivas, y almacenar el resultado en 2 nuevas columnas. La prueba del sentimiento será compararlo con la puntuación del revisor para la misma reseña. Por ejemplo, si el sentimiento piensa que la reseña negativa tuvo un sentimiento de 1 (sentimiento extremadamente positivo) y el sentimiento de la reseña positiva de 1, pero el revisor le dio al hotel la puntuación más baja posible, entonces o el texto de la reseña no coincide con la puntuación, o el analizador de sentimientos no pudo reconocer el sentimiento correctamente. Deberías esperar que algunos puntajes de sentimiento estén completamente equivocados, y a menudo eso será explicable, por ejemplo, la reseña podría ser extremadamente sarcástica "Por supuesto que ME ENCANTÓ dormir en una habitación sin calefacción" y el analizador de sentimientos piensa que eso es un sentimiento positivo, aunque un humano que lo lea sabría que era sarcasmo.

NLTK proporciona diferentes analizadores de sentimiento para aprender, y puedes sustituirlos y ver si el sentimiento es más o menos preciso. El análisis de sentimiento VADER se utiliza aquí.

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

Más adelante en tu programa, cuando estés listo para calcular el sentimiento, puedes aplicarlo a cada reseña de la siguiente manera:

```python
# Add a negative sentiment and positive sentiment column
print("Calculating sentiment columns for both positive and negative reviews")
start = time.time()
df["Negative_Sentiment"] = df.Negative_Review.apply(calc_sentiment)
df["Positive_Sentiment"] = df.Positive_Review.apply(calc_sentiment)
end = time.time()
print("Calculating sentiment took " + str(round(end - start, 2)) + " seconds")
```

Esto toma aproximadamente 120 segundos en mi computadora, pero variará en cada computadora. Si deseas imprimir los resultados y ver si el sentimiento coincide con la reseña:

```python
df = df.sort_values(by=["Negative_Sentiment"], ascending=True)
print(df[["Negative_Review", "Negative_Sentiment"]])
df = df.sort_values(by=["Positive_Sentiment"], ascending=True)
print(df[["Positive_Review", "Positive_Sentiment"]])
```

Lo último que debes hacer con el archivo antes de usarlo en el desafío, es guardarlo. También deberías considerar reorganizar todas tus nuevas columnas para que sean fáciles de trabajar (para un humano, es un cambio cosmético).

```python
# Reorder the columns (This is cosmetic, but to make it easier to explore the data later)
df = df.reindex(["Hotel_Name", "Hotel_Address", "Total_Number_of_Reviews", "Average_Score", "Reviewer_Score", "Negative_Sentiment", "Positive_Sentiment", "Reviewer_Nationality", "Leisure_trip", "Couple", "Solo_traveler", "Business_trip", "Group", "Family_with_young_children", "Family_with_older_children", "With_a_pet", "Negative_Review", "Positive_Review"], axis=1)

print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r"../data/Hotel_Reviews_NLP.csv", index = False)
```

Debes ejecutar todo el código para [el cuaderno de análisis](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) (después de haber ejecutado [tu cuaderno de filtrado](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) para generar el archivo Hotel_Reviews_Filtered.csv).

Para revisar, los pasos son:

1. El archivo del conjunto de datos original **Hotel_Reviews.csv** se explora en la lección anterior con [el cuaderno explorador](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/4-Hotel-Reviews-1/solution/notebook.ipynb)
2. Hotel_Reviews.csv se filtra mediante [el cuaderno de filtrado](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) resultando en **Hotel_Reviews_Filtered.csv**
3. Hotel_Reviews_Filtered.csv se procesa mediante [el cuaderno de análisis de sentimiento](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) resultando en **Hotel_Reviews_NLP.csv**
4. Usa Hotel_Reviews_NLP.csv en el Desafío de PLN a continuación

### Conclusión

Cuando comenzaste, tenías un conjunto de datos con columnas y datos, pero no todo podía ser verificado o utilizado. Has explorado los datos, filtrado lo que no necesitas, convertido etiquetas en algo útil, calculado tus propios promedios, agregado algunas columnas de sentimiento y, con suerte, aprendido algunas cosas interesantes sobre el procesamiento de texto natural.

## [Cuestionario posterior a la lección](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/40/)

## Desafío

Ahora que tienes tu conjunto de datos analizado para sentimiento, ve si puedes usar estrategias que has aprendido en este currículo (¿quizás clustering?) para determinar patrones en torno al sentimiento.

## Revisión y autoestudio

Toma [este módulo de Learn](https://docs.microsoft.com/en-us/learn/modules/classify-user-feedback-with-the-text-analytics-api/?WT.mc_id=academic-77952-leestott) para aprender más y usar diferentes herramientas para explorar el sentimiento en el texto.
## Tarea 

[Prueba con un conjunto de datos diferente](assignment.md)

**Descargo de responsabilidad**:
Este documento ha sido traducido utilizando servicios de traducción automática basados en inteligencia artificial. Si bien nos esforzamos por lograr precisión, tenga en cuenta que las traducciones automatizadas pueden contener errores o inexactitudes. El documento original en su idioma nativo debe considerarse la fuente autorizada. Para información crítica, se recomienda la traducción profesional humana. No nos hacemos responsables de cualquier malentendido o interpretación errónea que surja del uso de esta traducción.