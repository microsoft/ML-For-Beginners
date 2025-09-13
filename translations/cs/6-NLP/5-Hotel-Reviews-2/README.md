<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "2c742993fe95d5bcbb2846eda3d442a1",
  "translation_date": "2025-09-05T01:42:53+00:00",
  "source_file": "6-NLP/5-Hotel-Reviews-2/README.md",
  "language_code": "cs"
}
-->
# Analýza sentimentu pomocí recenzí hotelů

Nyní, když jste dataset podrobně prozkoumali, je čas filtrovat sloupce a použít techniky NLP na dataset, abyste získali nové poznatky o hotelech.

## [Kvíz před přednáškou](https://ff-quizzes.netlify.app/en/ml/)

### Operace filtrování a analýzy sentimentu

Jak jste si pravděpodobně všimli, dataset má několik problémů. Některé sloupce jsou plné nepotřebných informací, jiné se zdají být nesprávné. Pokud jsou správné, není jasné, jak byly vypočítány, a odpovědi nelze nezávisle ověřit vašimi vlastními výpočty.

## Cvičení: trochu více zpracování dat

Vyčistěte data o něco více. Přidejte sloupce, které budou užitečné později, změňte hodnoty v jiných sloupcích a některé sloupce úplně odstraňte.

1. Počáteční zpracování sloupců

   1. Odstraňte `lat` a `lng`

   2. Nahraďte hodnoty `Hotel_Address` následujícími hodnotami (pokud adresa obsahuje název města a země, změňte ji pouze na město a zemi).

      Toto jsou jediná města a země v datasetu:

      Amsterdam, Nizozemsko

      Barcelona, Španělsko

      Londýn, Spojené království

      Milán, Itálie

      Paříž, Francie

      Vídeň, Rakousko 

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

      Nyní můžete dotazovat data na úrovni země:

      ```python
      display(df.groupby("Hotel_Address").agg({"Hotel_Name": "nunique"}))
      ```

      | Hotel_Address          | Hotel_Name |
      | :--------------------- | :--------: |
      | Amsterdam, Nizozemsko  |    105     |
      | Barcelona, Španělsko   |    211     |
      | Londýn, Spojené království | 400     |
      | Milán, Itálie          |    162     |
      | Paříž, Francie         |    458     |
      | Vídeň, Rakousko        |    158     |

2. Zpracování sloupců meta-recenze hotelu

  1. Odstraňte `Additional_Number_of_Scoring`

  1. Nahraďte `Total_Number_of_Reviews` celkovým počtem recenzí pro daný hotel, které jsou skutečně v datasetu 

  1. Nahraďte `Average_Score` vlastním vypočítaným skóre

  ```python
  # Drop `Additional_Number_of_Scoring`
  df.drop(["Additional_Number_of_Scoring"], axis = 1, inplace=True)
  # Replace `Total_Number_of_Reviews` and `Average_Score` with our own calculated values
  df.Total_Number_of_Reviews = df.groupby('Hotel_Name').transform('count')
  df.Average_Score = round(df.groupby('Hotel_Name').Reviewer_Score.transform('mean'), 1)
  ```

3. Zpracování sloupců recenzí

   1. Odstraňte `Review_Total_Negative_Word_Counts`, `Review_Total_Positive_Word_Counts`, `Review_Date` a `days_since_review`

   2. Zachovejte `Reviewer_Score`, `Negative_Review` a `Positive_Review` tak, jak jsou,
     
   3. Prozatím zachovejte `Tags`

     - V další části provedeme další filtrovací operace na značkách a poté budou značky odstraněny

4. Zpracování sloupců recenzentů

  1. Odstraňte `Total_Number_of_Reviews_Reviewer_Has_Given`
  
  2. Zachovejte `Reviewer_Nationality`

### Sloupce značek

Sloupec `Tag` je problematický, protože je to seznam (ve formě textu) uložený ve sloupci. Bohužel pořadí a počet podsekcí v tomto sloupci nejsou vždy stejné. Je těžké pro člověka identifikovat správné fráze, které by ho mohly zajímat, protože dataset obsahuje 515 000 řádků a 1427 hotelů, a každý má mírně odlišné možnosti, které si recenzent mohl vybrat. Zde přichází na řadu NLP. Můžete prohledat text a najít nejčastější fráze a spočítat je.

Bohužel nás nezajímají jednotlivá slova, ale víceslovné fráze (např. *Služební cesta*). Spuštění algoritmu pro frekvenční distribuci víceslovných frází na tak velkém množství dat (6762646 slov) by mohlo trvat mimořádně dlouho, ale bez pohledu na data by se zdálo, že je to nutný výdaj. Zde je užitečná průzkumná analýza dat, protože jste viděli vzorek značek, jako například `[' Služební cesta  ', ' Cestovatel sólo ', ' Jednolůžkový pokoj ', ' Pobyt na 5 nocí ', ' Odesláno z mobilního zařízení ']`, můžete začít zjišťovat, zda je možné výrazně snížit zpracování, které musíte provést. Naštěstí to možné je - ale nejprve musíte provést několik kroků, abyste určili značky, které vás zajímají.

### Filtrování značek

Pamatujte, že cílem datasetu je přidat sentiment a sloupce, které vám pomohou vybrat nejlepší hotel (pro vás nebo možná pro klienta, který vás pověřil vytvořením bota pro doporučení hotelů). Musíte se sami sebe zeptat, zda jsou značky užitečné nebo ne v konečném datasetu. Zde je jedna interpretace (pokud byste dataset potřebovali z jiných důvodů, mohly by být různé značky zahrnuty/vynechány):

1. Typ cesty je relevantní a měl by zůstat
2. Typ skupiny hostů je důležitý a měl by zůstat
3. Typ pokoje, apartmá nebo studia, ve kterém host pobýval, je irelevantní (všechny hotely mají v podstatě stejné pokoje)
4. Zařízení, ze kterého byla recenze odeslána, je irelevantní
5. Počet nocí, které recenzent zůstal, *může* být relevantní, pokud byste delší pobyty přisuzovali tomu, že se jim hotel líbil více, ale je to sporné a pravděpodobně irelevantní

Shrnuto, **ponechte 2 typy značek a ostatní odstraňte**.

Nejprve nechcete počítat značky, dokud nejsou v lepším formátu, což znamená odstranění hranatých závorek a uvozovek. Můžete to udělat několika způsoby, ale chcete nejrychlejší, protože zpracování velkého množství dat by mohlo trvat dlouho. Naštěstí pandas nabízí snadný způsob, jak provést každý z těchto kroků.

```Python
# Remove opening and closing brackets
df.Tags = df.Tags.str.strip("[']")
# remove all quotes too
df.Tags = df.Tags.str.replace(" ', '", ",", regex = False)
```

Každá značka se stane něčím jako: `Služební cesta, Cestovatel sólo, Jednolůžkový pokoj, Pobyt na 5 nocí, Odesláno z mobilního zařízení`. 

Dále narazíme na problém. Některé recenze nebo řádky mají 5 sloupců, některé 3, některé 6. To je výsledek způsobu, jakým byl dataset vytvořen, a je těžké to opravit. Chcete získat frekvenční počet každé fráze, ale jsou v různém pořadí v každé recenzi, takže počet může být nesprávný a hotel nemusí dostat značku, kterou si zasloužil.

Místo toho využijete různé pořadí ve svůj prospěch, protože každá značka je víceslovná, ale také oddělená čárkou! Nejjednodušší způsob, jak to udělat, je vytvořit 6 dočasných sloupců, do kterých vložíte každou značku podle jejího pořadí ve značce. Poté můžete sloučit těchto 6 sloupců do jednoho velkého sloupce a spustit metodu `value_counts()` na výsledném sloupci. Po vytištění zjistíte, že existovalo 2428 unikátních značek. Zde je malý vzorek:

| Značka                          | Počet  |
| ------------------------------- | ------ |
| Rekreační cesta                 | 417778 |
| Odesláno z mobilního zařízení   | 307640 |
| Pár                             | 252294 |
| Pobyt na 1 noc                  | 193645 |
| Pobyt na 2 noci                 | 133937 |
| Cestovatel sólo                 | 108545 |
| Pobyt na 3 noci                 | 95821  |
| Služební cesta                  | 82939  |
| Skupina                         | 65392  |
| Rodina s malými dětmi           | 61015  |
| Pobyt na 4 noci                 | 47817  |
| Dvoulůžkový pokoj               | 35207  |
| Standardní dvoulůžkový pokoj    | 32248  |
| Superior dvoulůžkový pokoj      | 31393  |
| Rodina se staršími dětmi        | 26349  |
| Deluxe dvoulůžkový pokoj        | 24823  |
| Dvoulůžkový nebo dvoulůžkový pokoj | 22393  |
| Pobyt na 5 nocí                 | 20845  |
| Standardní dvoulůžkový nebo dvoulůžkový pokoj | 17483  |
| Klasický dvoulůžkový pokoj      | 16989  |
| Superior dvoulůžkový nebo dvoulůžkový pokoj | 13570 |
| 2 pokoje                        | 12393  |

Některé běžné značky jako `Odesláno z mobilního zařízení` jsou pro nás zbytečné, takže by mohlo být chytré je odstranit před počítáním výskytu frází, ale je to tak rychlá operace, že je můžete ponechat a ignorovat.

### Odstranění značek délky pobytu

Odstranění těchto značek je krok 1, což mírně snižuje celkový počet značek, které je třeba zvážit. Všimněte si, že je neodstraňujete z datasetu, pouze se rozhodnete je odstranit z úvahy jako hodnoty, které chcete počítat/zachovat v datasetu recenzí.

| Délka pobytu   | Počet  |
| -------------- | ------ |
| Pobyt na 1 noc | 193645 |
| Pobyt na 2 noci | 133937 |
| Pobyt na 3 noci | 95821  |
| Pobyt na 4 noci | 47817  |
| Pobyt na 5 nocí | 20845  |
| Pobyt na 6 nocí | 9776   |
| Pobyt na 7 nocí | 7399   |
| Pobyt na 8 nocí | 2502   |
| Pobyt na 9 nocí | 1293   |
| ...            | ...    |

Existuje obrovská škála pokojů, apartmá, studií, bytů a podobně. Všechny znamenají zhruba totéž a nejsou pro vás relevantní, takže je odstraňte z úvahy.

| Typ pokoje                  | Počet |
| --------------------------- | ----- |
| Dvoulůžkový pokoj           | 35207 |
| Standardní dvoulůžkový pokoj | 32248 |
| Superior dvoulůžkový pokoj  | 31393 |
| Deluxe dvoulůžkový pokoj    | 24823 |
| Dvoulůžkový nebo dvoulůžkový pokoj | 22393 |
| Standardní dvoulůžkový nebo dvoulůžkový pokoj | 17483 |
| Klasický dvoulůžkový pokoj  | 16989 |
| Superior dvoulůžkový nebo dvoulůžkový pokoj | 13570 |

Nakonec, a to je potěšující (protože to nevyžadovalo mnoho zpracování), vám zůstanou následující *užitečné* značky:

| Značka                                        | Počet  |
| -------------------------------------------- | ------ |
| Rekreační cesta                              | 417778 |
| Pár                                          | 252294 |
| Cestovatel sólo                              | 108545 |
| Služební cesta                               | 82939  |
| Skupina (sloučeno s Cestovatelé s přáteli)   | 67535  |
| Rodina s malými dětmi                        | 61015  |
| Rodina se staršími dětmi                     | 26349  |
| S domácím mazlíčkem                          | 1405   |

Můžete argumentovat, že `Cestovatelé s přáteli` je více méně totéž jako `Skupina`, a bylo by spravedlivé tyto dvě sloučit, jak je uvedeno výše. Kód pro identifikaci správných značek je [notebook Značky](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb).

Posledním krokem je vytvoření nových sloupců pro každou z těchto značek. Poté, pro každý řádek recenze, pokud sloupec `Tag` odpovídá jednomu z nových sloupců, přidejte 1, pokud ne, přidejte 0. Konečným výsledkem bude počet recenzentů, kteří si vybrali tento hotel (v souhrnu) například pro služební cestu vs rekreační cestu, nebo aby si s sebou vzali domácího mazlíčka, což je užitečná informace při doporučování hotelu.

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

### Uložení souboru

Nakonec uložte dataset tak, jak je nyní, pod novým názvem.

```python
df.drop(["Review_Total_Negative_Word_Counts", "Review_Total_Positive_Word_Counts", "days_since_review", "Total_Number_of_Reviews_Reviewer_Has_Given"], axis = 1, inplace=True)

# Saving new data file with calculated columns
print("Saving results to Hotel_Reviews_Filtered.csv")
df.to_csv(r'../data/Hotel_Reviews_Filtered.csv', index = False)
```

## Operace analýzy sentimentu

V této poslední části provedete analýzu sentimentu na sloupcích recenzí a uložíte výsledky do datasetu.

## Cvičení: načtení a uložení filtrovaných dat

Všimněte si, že nyní načítáte filtrovaný dataset, který byl uložen v předchozí části, **ne** původní dataset.

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

### Odstranění stop slov

Pokud byste provedli analýzu sentimentu na sloupcích negativních a pozitivních recenzí, mohlo by to trvat dlouho. Testováno na výkonném testovacím notebooku s rychlým CPU, trvalo to 12–14 minut v závislosti na tom, která knihovna sentimentu byla použita. To je (relativně) dlouhá doba, takže stojí za to prozkoumat, zda to lze urychlit. 

Odstranění stop slov, tedy běžných anglických slov, která nemění sentiment věty, je prvním krokem. Jejich odstraněním by měla analýza sentimentu probíhat rychleji, ale nebude méně přesná (protože stop slova sentiment neovlivňují, ale zpomalují analýzu). 

Nejdelší negativní recenze měla 395 slov, ale po odstranění stop slov má 195 slov.

Odstranění stop slov je také rychlá operace, odstranění stop slov ze 2 sloupců recenzí přes 515 000 řádků trvalo na testovacím zařízení 3,3 sekundy. Může to trvat o něco více nebo méně času v závislosti na rychlosti vašeho zařízení, RAM, zda máte SSD nebo ne, a na některých dalších faktorech. Relativní krátkost operace znamená, že pokud zlepší čas analýzy sentimentu, pak stojí za to ji provést.

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

### Provádění analýzy sentimentu

Nyní byste měli vypočítat analýzu sentimentu pro sloupce negativních a pozitivních recenzí a uložit výsledek do 2 nových sloupců. Test sentimentu bude porovnat jej se skóre recenzenta pro stejnou recenzi. Například pokud sentiment ukazuje, že negativní recenze měla sentiment 1 (extrémně pozitivní sentiment) a sentiment pozitivní recenze byl 1, ale recenzent dal hotelu nejnižší možné skóre, pak buď text recenze neodpovídá skóre, nebo sentimentový analyzátor nedokázal správně rozpoznat sentiment. Měli byste očekávat, že některá sentimentová skóre budou zcela nesprávná, a často to bude vysvětlitelné, např. recenze může být extrémně sarkastická "Samozřejmě jsem MILOVAL spát v pokoji bez topení" a sentimentový analyzátor si myslí, že je to pozitivní sentiment, i když člověk, který to čte, by věděl, že jde o sarkasmus.
NLTK nabízí různé analyzátory sentimentu, se kterými se můžete učit, a můžete je zaměnit, abyste zjistili, zda je sentiment přesnější nebo méně přesný. Zde je použita analýza sentimentu VADER.

> Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, červen 2014.

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

Později ve vašem programu, když budete připraveni vypočítat sentiment, můžete jej aplikovat na každou recenzi následujícím způsobem:

```python
# Add a negative sentiment and positive sentiment column
print("Calculating sentiment columns for both positive and negative reviews")
start = time.time()
df["Negative_Sentiment"] = df.Negative_Review.apply(calc_sentiment)
df["Positive_Sentiment"] = df.Positive_Review.apply(calc_sentiment)
end = time.time()
print("Calculating sentiment took " + str(round(end - start, 2)) + " seconds")
```

Toto trvá přibližně 120 sekund na mém počítači, ale na každém počítači se to může lišit. Pokud chcete výsledky vytisknout a zjistit, zda sentiment odpovídá recenzi:

```python
df = df.sort_values(by=["Negative_Sentiment"], ascending=True)
print(df[["Negative_Review", "Negative_Sentiment"]])
df = df.sort_values(by=["Positive_Sentiment"], ascending=True)
print(df[["Positive_Review", "Positive_Sentiment"]])
```

Poslední věc, kterou je třeba udělat se souborem před jeho použitím ve výzvě, je uložit ho! Měli byste také zvážit přeorganizování všech nových sloupců, aby byly snadno použitelné (pro člověka je to kosmetická změna).

```python
# Reorder the columns (This is cosmetic, but to make it easier to explore the data later)
df = df.reindex(["Hotel_Name", "Hotel_Address", "Total_Number_of_Reviews", "Average_Score", "Reviewer_Score", "Negative_Sentiment", "Positive_Sentiment", "Reviewer_Nationality", "Leisure_trip", "Couple", "Solo_traveler", "Business_trip", "Group", "Family_with_young_children", "Family_with_older_children", "With_a_pet", "Negative_Review", "Positive_Review"], axis=1)

print("Saving results to Hotel_Reviews_NLP.csv")
df.to_csv(r"../data/Hotel_Reviews_NLP.csv", index = False)
```

Měli byste spustit celý kód pro [analytický notebook](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb) (poté, co jste spustili [notebook pro filtrování](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb) k vytvoření souboru Hotel_Reviews_Filtered.csv).

Shrnutí kroků:

1. Původní soubor dat **Hotel_Reviews.csv** byl prozkoumán v předchozí lekci pomocí [exploračního notebooku](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/4-Hotel-Reviews-1/solution/notebook.ipynb)
2. Hotel_Reviews.csv byl filtrován pomocí [notebooku pro filtrování](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/1-notebook.ipynb), což vedlo k vytvoření **Hotel_Reviews_Filtered.csv**
3. Hotel_Reviews_Filtered.csv byl zpracován pomocí [notebooku pro analýzu sentimentu](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/5-Hotel-Reviews-2/solution/3-notebook.ipynb), což vedlo k vytvoření **Hotel_Reviews_NLP.csv**
4. Použijte Hotel_Reviews_NLP.csv ve výzvě NLP níže

### Závěr

Na začátku jste měli dataset se sloupci a daty, ale ne všechna byla ověřitelná nebo použitelná. Prozkoumali jste data, odfiltrovali nepotřebné, převedli značky na něco užitečného, vypočítali vlastní průměry, přidali sloupce sentimentu a doufejme, že jste se naučili něco zajímavého o zpracování přirozeného textu.

## [Kvíz po přednášce](https://ff-quizzes.netlify.app/en/ml/)

## Výzva

Nyní, když máte dataset analyzovaný na sentiment, zkuste použít strategie, které jste se naučili v tomto kurzu (například clustering), abyste určili vzorce kolem sentimentu.

## Přehled & Samostudium

Vezměte si [tento modul Learn](https://docs.microsoft.com/en-us/learn/modules/classify-user-feedback-with-the-text-analytics-api/?WT.mc_id=academic-77952-leestott), abyste se dozvěděli více a použili různé nástroje k prozkoumání sentimentu v textu.

## Úkol

[Vyzkoušejte jiný dataset](assignment.md)

---

**Prohlášení**:  
Tento dokument byl přeložen pomocí služby pro automatický překlad [Co-op Translator](https://github.com/Azure/co-op-translator). Ačkoli se snažíme o přesnost, mějte prosím na paměti, že automatické překlady mohou obsahovat chyby nebo nepřesnosti. Původní dokument v jeho původním jazyce by měl být považován za závazný zdroj. Pro důležité informace doporučujeme profesionální lidský překlad. Neodpovídáme za žádné nedorozumění nebo nesprávné interpretace vyplývající z použití tohoto překladu.