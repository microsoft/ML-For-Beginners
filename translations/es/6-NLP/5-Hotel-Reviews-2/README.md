<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "a2aa4e9b91b9640db2c15363c4299d8b",
  "translation_date": "2025-09-04T00:57:04+00:00",
  "source_file": "6-NLP/5-Hotel-Reviews-2/README.md",
  "language_code": "es"
}
-->
# Análisis de sentimientos con reseñas de hoteles

Ahora que has explorado el conjunto de datos en detalle, es momento de filtrar las columnas y luego usar técnicas de PLN (Procesamiento de Lenguaje Natural) en el conjunto de datos para obtener nuevas perspectivas sobre los hoteles.

## [Cuestionario previo a la lección](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/39/)

### Operaciones de filtrado y análisis de sentimientos

Como probablemente hayas notado, el conjunto de datos tiene algunos problemas. Algunas columnas están llenas de información inútil, otras parecen incorrectas. Si son correctas, no está claro cómo se calcularon, y las respuestas no pueden ser verificadas de forma independiente con tus propios cálculos.

## Ejercicio: un poco más de procesamiento de datos

Limpia los datos un poco más. Agrega columnas que serán útiles más adelante, cambia los valores en otras columnas y elimina ciertas columnas por completo.

1. Procesamiento inicial de columnas

   1. Elimina `lat` y `lng`.

   2. Sustituye los valores de `Hotel_Address` con los siguientes valores (si la dirección contiene el nombre de la ciudad y el país, cámbialo solo por la ciudad y el país).

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

2. Procesar columnas de meta-reseñas de hoteles

   1. Elimina `Additional_Number_of_Scoring`.

   2. Sustituye `Total_Number_of_Reviews` con el número total de reseñas para ese hotel que realmente están en el conjunto de datos.

   3. Sustituye `Average_Score` con nuestra propia puntuación calculada.

      ```python
  # Drop `Additional_Number_of_Scoring`
  df.drop(["Additional_Number_of_Scoring"], axis = 1, inplace=True)
  # Replace `Total_Number_of_Reviews` and `Average_Score` with our own calculated values
  df.Total_Number_of_Reviews = df.groupby('Hotel_Name').transform('count')
  df.Average_Score = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
  ```

3. Procesar columnas de reseñas

   1. Elimina `Review_Total_Negative_Word_Counts`, `Review_Total_Positive_Word_Counts`, `Review_Date` y `days_since_review`.

   2. Conserva `Reviewer_Score`, `Negative_Review` y `Positive_Review` tal como están.

   3. Conserva `Tags` por ahora.

      - Realizaremos algunas operaciones de filtrado adicionales en las etiquetas en la siguiente sección y luego eliminaremos las etiquetas.

4. Procesar columnas de los revisores

   1. Elimina `Total_Number_of_Reviews_Reviewer_Has_Given`.

   2. Conserva `Reviewer_Nationality`.

### Columnas de etiquetas

La columna `Tag` es problemática ya que es una lista (en forma de texto) almacenada en la columna. Desafortunadamente, el orden y el número de subsecciones en esta columna no siempre son los mismos. Es difícil para una persona identificar las frases correctas de interés, porque hay 515,000 filas y 1,427 hoteles, y cada uno tiene opciones ligeramente diferentes que un revisor podría elegir. Aquí es donde el PLN brilla. Puedes escanear el texto y encontrar las frases más comunes, y contarlas.

Desafortunadamente, no estamos interesados en palabras individuales, sino en frases de varias palabras (por ejemplo, *Viaje de negocios*). Ejecutar un algoritmo de distribución de frecuencia de frases en tantos datos (6,762,646 palabras) podría tomar un tiempo extraordinario, pero sin mirar los datos, parecería que es un gasto necesario. Aquí es donde el análisis exploratorio de datos resulta útil, porque al haber visto una muestra de las etiquetas como `[' Viaje de negocios  ', ' Viajero solo ', ' Habitación individual ', ' Estancia de 5 noches ', ' Enviado desde un dispositivo móvil ']`, puedes comenzar a preguntarte si es posible reducir enormemente el procesamiento que tienes que hacer. Afortunadamente, lo es, pero primero necesitas seguir algunos pasos para determinar las etiquetas de interés.

### Filtrando etiquetas

Recuerda que el objetivo del conjunto de datos es agregar sentimiento y columnas que te ayuden a elegir el mejor hotel (para ti o tal vez para un cliente que te encargue crear un bot de recomendaciones de hoteles). Debes preguntarte si las etiquetas son útiles o no en el conjunto de datos final. Aquí hay una interpretación (si necesitaras el conjunto de datos para otros fines, diferentes etiquetas podrían incluirse/excluirse):

1. El tipo de viaje es relevante y debería mantenerse.
2. El tipo de grupo de huéspedes es importante y debería mantenerse.
3. El tipo de habitación, suite o estudio en el que se hospedó el huésped es irrelevante (todos los hoteles tienen básicamente las mismas habitaciones).
4. El dispositivo desde el que se envió la reseña es irrelevante.
5. El número de noches que el revisor se quedó *podría* ser relevante si atribuyes estancias más largas con que les gustó más el hotel, pero es poco probable y probablemente irrelevante.

En resumen, **mantén 2 tipos de etiquetas y elimina las demás**.

Primero, no quieres contar las etiquetas hasta que estén en un mejor formato, lo que significa eliminar los corchetes y las comillas. Puedes hacer esto de varias maneras, pero quieres la más rápida ya que podría tomar mucho tiempo procesar tantos datos. Afortunadamente, pandas tiene una forma sencilla de realizar cada uno de estos pasos.

```Python
# Remove opening and closing brackets
df.Tags = df.Tags.str.strip("[']")
# remove all quotes too
df.Tags = df.Tags.str.replace(" ', '", ",", regex = False)
```

Cada etiqueta se convierte en algo como: `Viaje de negocios, Viajero solo, Habitación individual, Estancia de 5 noches, Enviado desde un dispositivo móvil`.

A continuación, encontramos un problema. Algunas reseñas, o filas, tienen 5 columnas, otras 3, otras 6. Esto es resultado de cómo se creó el conjunto de datos y es difícil de corregir. Quieres obtener un conteo de frecuencia de cada frase, pero están en diferente orden en cada reseña, por lo que el conteo podría estar desfasado, y un hotel podría no recibir una etiqueta que merecía.

En su lugar, usarás el orden diferente a tu favor, porque cada etiqueta es de varias palabras pero también está separada por una coma. La forma más sencilla de hacer esto es crear 6 columnas temporales con cada etiqueta insertada en la columna correspondiente a su orden en la etiqueta. Luego puedes fusionar las 6 columnas en una gran columna y ejecutar el método `value_counts()` en la columna resultante. Al imprimir eso, verás que había 2,428 etiquetas únicas. Aquí hay una pequeña muestra:

| Etiqueta                        | Conteo  |
| ------------------------------- | ------- |
| Viaje de ocio                   | 417778  |
| Enviado desde un dispositivo móvil | 307640 |
| Pareja                          | 252294  |
| Estancia de 1 noche             | 193645  |
| Estancia de 2 noches            | 133937  |
| Viajero solo                    | 108545  |
| Estancia de 3 noches            | 95821   |
| Viaje de negocios               | 82939   |
| Grupo                           | 65392   |
| Familia con niños pequeños      | 61015   |
| Estancia de 4 noches            | 47817   |
| Habitación doble                | 35207   |
| Habitación doble estándar       | 32248   |
| Habitación doble superior       | 31393   |
| Familia con niños mayores       | 26349   |
| Habitación doble deluxe         | 24823   |
| Habitación doble o twin         | 22393   |
| Estancia de 5 noches            | 20845   |
| Habitación doble o twin estándar | 17483  |
| Habitación doble clásica        | 16989   |
| Habitación doble o twin superior | 13570  |
| 2 habitaciones                  | 12393   |

Algunas de las etiquetas comunes como `Enviado desde un dispositivo móvil` no nos son útiles, por lo que podría ser inteligente eliminarlas antes de contar la ocurrencia de frases, pero es una operación tan rápida que puedes dejarlas y simplemente ignorarlas.

### Eliminando las etiquetas de duración de la estancia

Eliminar estas etiquetas es el paso 1, lo que reduce ligeramente el número total de etiquetas a considerar. Nota que no las eliminas del conjunto de datos, solo decides no considerarlas como valores a contar/mantener en el conjunto de datos de reseñas.

| Duración de la estancia | Conteo  |
| ------------------------ | ------- |
| Estancia de 1 noche      | 193645  |
| Estancia de 2 noches     | 133937  |
| Estancia de 3 noches     | 95821   |
| Estancia de 4 noches     | 47817   |
| Estancia de 5 noches     | 20845   |
| Estancia de 6 noches     | 9776    |
| Estancia de 7 noches     | 7399    |
| Estancia de 8 noches     | 2502    |
| Estancia de 9 noches     | 1293    |
| ...                      | ...     |

Hay una gran variedad de habitaciones, suites, estudios, apartamentos, etc. Todos significan más o menos lo mismo y no son relevantes para ti, así que elimínalos de la consideración.

| Tipo de habitación         | Conteo |
| -------------------------- | ------ |
| Habitación doble           | 35207  |
| Habitación doble estándar  | 32248  |
| Habitación doble superior  | 31393  |
| Habitación doble deluxe    | 24823  |
| Habitación doble o twin    | 22393  |
| Habitación doble o twin estándar | 17483 |
| Habitación doble clásica   | 16989  |
| Habitación doble o twin superior | 13570 |

Finalmente, y esto es satisfactorio (porque no requirió mucho procesamiento), te quedarás con las siguientes etiquetas *útiles*:

| Etiqueta                                      | Conteo  |
| -------------------------------------------- | ------- |
| Viaje de ocio                                | 417778  |
| Pareja                                       | 252294  |
| Viajero solo                                 | 108545  |
| Viaje de negocios                            | 82939   |
| Grupo (combinado con Viajeros con amigos)    | 67535   |
| Familia con niños pequeños                   | 61015   |
| Familia con niños mayores                    | 26349   |
| Con una mascota                              | 1405    |

Podrías argumentar que `Viajeros con amigos` es lo mismo que `Grupo`, más o menos, y sería razonable combinarlos como se muestra arriba. El código para identificar las etiquetas correctas está en [el cuaderno de etiquetas](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb).

El paso final es crear nuevas columnas para cada una de estas etiquetas. Luego, para cada fila de reseña, si la columna `Tag` coincide con una de las nuevas columnas, agrega un 1; si no, agrega un 0. El resultado final será un conteo de cuántos revisores eligieron este hotel (en conjunto) para, por ejemplo, negocios vs ocio, o para llevar una mascota, y esta es información útil al recomendar un hotel.

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

## Operaciones de análisis de sentimientos

En esta sección final, aplicarás análisis de sentimientos a las columnas de reseñas y guardarás los resultados en un conjunto de datos.

## Ejercicio: carga y guarda los datos filtrados

Nota que ahora estás cargando el conjunto de datos filtrado que se guardó en la sección anterior, **no** el conjunto de datos original.

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

Si ejecutaras el análisis de sentimientos en las columnas de reseñas negativas y positivas, podría tomar mucho tiempo. Probado en un portátil de prueba potente con un CPU rápido, tomó entre 12 y 14 minutos dependiendo de la biblioteca de análisis de sentimientos utilizada. Ese es un tiempo (relativamente) largo, por lo que vale la pena investigar si se puede acelerar.

Eliminar palabras vacías, o palabras comunes en inglés que no cambian el sentimiento de una oración, es el primer paso. Al eliminarlas, el análisis de sentimientos debería ejecutarse más rápido, pero no ser menos preciso (ya que las palabras vacías no afectan el sentimiento, pero sí ralentizan el análisis).

La reseña negativa más larga tenía 395 palabras, pero después de eliminar las palabras vacías, tiene 195 palabras.

Eliminar las palabras vacías también es una operación rápida; eliminarlas de 2 columnas de reseñas en más de 515,000 filas tomó 3.3 segundos en el dispositivo de prueba. Podría tomar un poco más o menos tiempo dependiendo de la velocidad de tu CPU, RAM, si tienes un SSD o no, y otros factores. La relativa brevedad de la operación significa que, si mejora el tiempo de análisis de sentimientos, vale la pena hacerlo.

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

### Realizando análisis de sentimientos
Ahora deberías calcular el análisis de sentimiento para las columnas de reseñas negativas y positivas, y almacenar el resultado en 2 nuevas columnas. La prueba del sentimiento será compararlo con la puntuación del revisor para la misma reseña. Por ejemplo, si el análisis de sentimiento determina que la reseña negativa tiene un sentimiento de 1 (sentimiento extremadamente positivo) y la reseña positiva también tiene un sentimiento de 1, pero el revisor dio al hotel la puntuación más baja posible, entonces o el texto de la reseña no coincide con la puntuación, o el analizador de sentimientos no pudo reconocer correctamente el sentimiento. Deberías esperar que algunos puntajes de sentimiento sean completamente incorrectos, y a menudo eso será explicable, por ejemplo, la reseña podría ser extremadamente sarcástica: "Por supuesto que AMÉ dormir en una habitación sin calefacción", y el analizador de sentimientos podría interpretar eso como un sentimiento positivo, aunque un humano que lo lea sabría que es sarcasmo.

NLTK proporciona diferentes analizadores de sentimientos para aprender, y puedes sustituirlos para ver si el sentimiento es más o menos preciso. Aquí se utiliza el análisis de sentimiento VADER.

> Hutto, C.J. & Gilbert, E.E. (2014). VADER: Un modelo parsimonioso basado en reglas para el análisis de sentimientos en texto de redes sociales. Octava Conferencia Internacional sobre Blogs y Medios Sociales (ICWSM-14). Ann Arbor, MI, junio de 2014.

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

Esto toma aproximadamente 120 segundos en mi computadora, pero puede variar en cada equipo. Si quieres imprimir los resultados y verificar si el sentimiento coincide con la reseña:

```python
df = df.sort_values(by=["Negative_Sentiment"], ascending=True)
print(df[["Negative_Review", "Negative_Sentiment"]])
df = df.sort_values(by=["Positive_Sentiment"], ascending=True)
print(df[["Positive_Review", "Positive_Sentiment"]])
```

Lo último que debes hacer con el archivo antes de usarlo en el desafío es ¡guardarlo! También deberías considerar reordenar todas tus nuevas columnas para que sean fáciles de trabajar (para un humano, es un cambio cosmético).

```python
# Reorder the columns (This is cosmetic, but to make it easier to explore the data later)
df = df.reindex(["Hotel_Name", "Hotel_Address", "Total_Number_of_Reviews", "Average_Score", "Reviewer_Score", "Negative_Sentiment", "Positive_Sentiment", "Reviewer_Nationality", "Leisure_trip", "Couple", "Solo_traveler", "Business_trip", "Group", "Family_with_young_children", "Family_with_older_children", "With_a_pet", "Negative_Review", "Positive_Review"], axis=1)

print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r"../data/Hotel_Reviews_NLP.csv", index = False)
```

Deberías ejecutar todo el código del [notebook de análisis](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) (después de haber ejecutado [tu notebook de filtrado](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) para generar el archivo Hotel_Reviews_Filtered.csv).

Para repasar, los pasos son:

1. El archivo del conjunto de datos original **Hotel_Reviews.csv** se explora en la lección anterior con [el notebook de exploración](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/4-Hotel-Reviews-1/solution/notebook.ipynb)
2. Hotel_Reviews.csv se filtra con [el notebook de filtrado](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) resultando en **Hotel_Reviews_Filtered.csv**
3. Hotel_Reviews_Filtered.csv se procesa con [el notebook de análisis de sentimiento](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) resultando en **Hotel_Reviews_NLP.csv**
4. Usa Hotel_Reviews_NLP.csv en el Desafío de NLP a continuación

### Conclusión

Cuando comenzaste, tenías un conjunto de datos con columnas y datos, pero no todo podía ser verificado o utilizado. Has explorado los datos, filtrado lo que no necesitas, convertido etiquetas en algo útil, calculado tus propios promedios, añadido algunas columnas de sentimiento y, con suerte, aprendido cosas interesantes sobre el procesamiento de texto natural.

## [Cuestionario posterior a la lección](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/40/)

## Desafío

Ahora que tienes tu conjunto de datos analizado para el sentimiento, intenta usar estrategias que has aprendido en este curso (¿quizás clustering?) para determinar patrones alrededor del sentimiento.

## Revisión y autoestudio

Toma [este módulo de Learn](https://docs.microsoft.com/en-us/learn/modules/classify-user-feedback-with-the-text-analytics-api/?WT.mc_id=academic-77952-leestott) para aprender más y usar diferentes herramientas para explorar el sentimiento en texto.

## Tarea

[Prueba con un conjunto de datos diferente](assignment.md)

---

**Descargo de responsabilidad**:  
Este documento ha sido traducido utilizando el servicio de traducción automática [Co-op Translator](https://github.com/Azure/co-op-translator). Si bien nos esforzamos por lograr precisión, tenga en cuenta que las traducciones automáticas pueden contener errores o imprecisiones. El documento original en su idioma nativo debe considerarse como la fuente autorizada. Para información crítica, se recomienda una traducción profesional realizada por humanos. No nos hacemos responsables de malentendidos o interpretaciones erróneas que puedan surgir del uso de esta traducción.