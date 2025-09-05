<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "be03c8182982b87ced155e4e9d1438e8",
  "translation_date": "2025-09-05T08:32:21+00:00",
  "source_file": "6-NLP/3-Translation-Sentiment/README.md",
  "language_code": "pl"
}
-->
# Tłumaczenie i analiza sentymentu za pomocą ML

W poprzednich lekcjach nauczyłeś się, jak zbudować podstawowego bota używając `TextBlob`, biblioteki, która wykorzystuje uczenie maszynowe w tle do wykonywania podstawowych zadań NLP, takich jak ekstrakcja fraz rzeczownikowych. Kolejnym ważnym wyzwaniem w lingwistyce komputerowej jest dokładne _tłumaczenie_ zdania z jednego języka mówionego lub pisanego na inny.

## [Quiz przed wykładem](https://ff-quizzes.netlify.app/en/ml/)

Tłumaczenie to bardzo trudny problem, który jest dodatkowo skomplikowany przez fakt, że istnieją tysiące języków, a każdy z nich może mieć bardzo różne zasady gramatyczne. Jednym z podejść jest przekształcenie formalnych zasad gramatycznych jednego języka, na przykład angielskiego, w strukturę niezależną od języka, a następnie przetłumaczenie jej poprzez konwersję na inny język. To podejście oznacza, że wykonasz następujące kroki:

1. **Identyfikacja**. Zidentyfikuj lub oznacz słowa w języku wejściowym jako rzeczowniki, czasowniki itd.
2. **Tworzenie tłumaczenia**. Wygeneruj bezpośrednie tłumaczenie każdego słowa w formacie docelowego języka.

### Przykładowe zdanie, angielski na irlandzki

W języku 'angielskim' zdanie _I feel happy_ składa się z trzech słów w kolejności:

- **podmiot** (I)
- **czasownik** (feel)
- **przymiotnik** (happy)

Jednak w języku 'irlandzkim' to samo zdanie ma zupełnie inną strukturę gramatyczną - emocje takie jak "*happy*" czy "*sad*" są wyrażane jako coś *spoczywającego* na tobie.

Angielskie wyrażenie `I feel happy` w języku irlandzkim brzmiałoby `Tá athas orm`. Dosłowne tłumaczenie to `Happy is upon me`.

Osoba mówiąca po irlandzku, tłumacząc na angielski, powiedziałaby `I feel happy`, a nie `Happy is upon me`, ponieważ rozumie znaczenie zdania, nawet jeśli słowa i struktura zdania są różne.

Formalna kolejność zdania w języku irlandzkim to:

- **czasownik** (Tá, czyli is)
- **przymiotnik** (athas, czyli happy)
- **podmiot** (orm, czyli upon me)

## Tłumaczenie

Prosty program tłumaczący mógłby tłumaczyć tylko słowa, ignorując strukturę zdania.

✅ Jeśli nauczyłeś się drugiego (lub trzeciego lub więcej) języka jako dorosły, mogłeś zacząć od myślenia w swoim ojczystym języku, tłumacząc pojęcia słowo po słowie w swojej głowie na drugi język, a następnie wypowiadając swoje tłumaczenie. To jest podobne do tego, co robią proste programy tłumaczące. Ważne jest, aby przejść ten etap, aby osiągnąć płynność!

Proste tłumaczenie prowadzi do złych (a czasem zabawnych) błędów tłumaczeniowych: `I feel happy` tłumaczy się dosłownie na `Mise bhraitheann athas` w języku irlandzkim. To oznacza (dosłownie) `me feel happy` i nie jest poprawnym zdaniem w języku irlandzkim. Mimo że angielski i irlandzki to języki używane na dwóch blisko sąsiadujących wyspach, są to bardzo różne języki z różnymi strukturami gramatycznymi.

> Możesz obejrzeć kilka filmów o tradycjach językowych Irlandii, takich jak [ten](https://www.youtube.com/watch?v=mRIaLSdRMMs)

### Podejścia z użyciem uczenia maszynowego

Do tej pory nauczyłeś się o podejściu opartym na formalnych zasadach w przetwarzaniu języka naturalnego. Innym podejściem jest ignorowanie znaczenia słów i _zamiast tego użycie uczenia maszynowego do wykrywania wzorców_. Może to działać w tłumaczeniu, jeśli masz dużo tekstu (*korpus*) lub tekstów (*korpora*) w języku źródłowym i docelowym.

Na przykład, rozważ przypadek *Dumy i uprzedzenia*, znanej angielskiej powieści napisanej przez Jane Austen w 1813 roku. Jeśli porównasz książkę w języku angielskim z ludzkim tłumaczeniem książki na *francuski*, możesz wykryć frazy w jednym języku, które są _idiomatycznie_ przetłumaczone na drugi. Zaraz to zrobisz.

Na przykład, gdy angielskie wyrażenie `I have no money` jest tłumaczone dosłownie na francuski, może stać się `Je n'ai pas de monnaie`. "Monnaie" to trudny francuski 'fałszywy przyjaciel', ponieważ 'money' i 'monnaie' nie są synonimami. Lepsze tłumaczenie, które mógłby zrobić człowiek, to `Je n'ai pas d'argent`, ponieważ lepiej oddaje znaczenie, że nie masz pieniędzy (a nie 'drobnych', co oznacza 'monnaie').

![monnaie](../../../../6-NLP/3-Translation-Sentiment/images/monnaie.png)

> Obraz autorstwa [Jen Looper](https://twitter.com/jenlooper)

Jeśli model ML ma wystarczająco dużo ludzkich tłumaczeń, aby zbudować na nich model, może poprawić dokładność tłumaczeń, identyfikując wspólne wzorce w tekstach, które zostały wcześniej przetłumaczone przez ekspertów mówiących w obu językach.

### Ćwiczenie - tłumaczenie

Możesz użyć `TextBlob`, aby tłumaczyć zdania. Spróbuj słynnej pierwszej linijki **Dumy i uprzedzenia**:

```python
from textblob import TextBlob

blob = TextBlob(
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife!"
)
print(blob.translate(to="fr"))

```

`TextBlob` całkiem dobrze radzi sobie z tłumaczeniem: "C'est une vérité universellement reconnue, qu'un homme célibataire en possession d'une bonne fortune doit avoir besoin d'une femme!". 

Można argumentować, że tłumaczenie TextBlob jest znacznie bardziej precyzyjne niż francuskie tłumaczenie książki z 1932 roku autorstwa V. Leconte i Ch. Pressoir:

"C'est une vérité universelle qu'un célibataire pourvu d'une belle fortune doit avoir envie de se marier, et, si peu que l'on sache de son sentiment à cet egard, lorsqu'il arrive dans une nouvelle résidence, cette idée est si bien fixée dans l'esprit de ses voisins qu'ils le considèrent sur-le-champ comme la propriété légitime de l'une ou l'autre de leurs filles."

W tym przypadku tłumaczenie oparte na ML radzi sobie lepiej niż tłumacz ludzki, który niepotrzebnie dodaje słowa do oryginalnego tekstu autora dla 'jasności'.

> Co tu się dzieje? Dlaczego TextBlob jest tak dobry w tłumaczeniu? Otóż, w tle używa Google Translate, zaawansowanej sztucznej inteligencji zdolnej do analizy milionów fraz, aby przewidzieć najlepsze ciągi znaków dla danego zadania. Nie ma tu nic manualnego, a do użycia `blob.translate` potrzebne jest połączenie z internetem.

✅ Spróbuj kilku innych zdań. Które tłumaczenie jest lepsze, ML czy ludzkie? W jakich przypadkach?

## Analiza sentymentu

Innym obszarem, w którym uczenie maszynowe może działać bardzo dobrze, jest analiza sentymentu. Podejście nieoparte na ML polega na identyfikacji słów i fraz, które są 'pozytywne' i 'negatywne'. Następnie, dla nowego tekstu, oblicza się całkowitą wartość słów pozytywnych, negatywnych i neutralnych, aby określić ogólny sentyment. 

To podejście łatwo oszukać, jak mogłeś zauważyć w zadaniu z Marvinem - zdanie `Great, that was a wonderful waste of time, I'm glad we are lost on this dark road` jest sarkastyczne, negatywne, ale prosty algorytm wykrywa 'great', 'wonderful', 'glad' jako pozytywne i 'waste', 'lost' oraz 'dark' jako negatywne. Ogólny sentyment jest zakłócony przez te sprzeczne słowa.

✅ Zatrzymaj się na chwilę i pomyśl, jak jako ludzie przekazujemy sarkazm. Intonacja głosu odgrywa dużą rolę. Spróbuj powiedzieć frazę "Well, that film was awesome" na różne sposoby, aby odkryć, jak twój głos przekazuje znaczenie.

### Podejścia ML

Podejście ML polegałoby na ręcznym zgromadzeniu negatywnych i pozytywnych tekstów - tweetów, recenzji filmów lub czegokolwiek, gdzie człowiek podał ocenę *i* pisemną opinię. Następnie można zastosować techniki NLP do opinii i ocen, aby wyłoniły się wzorce (np. pozytywne recenzje filmów częściej zawierają frazę 'Oscar worthy' niż negatywne recenzje filmów, a pozytywne recenzje restauracji częściej mówią 'gourmet' niż 'disgusting').

> ⚖️ **Przykład**: Jeśli pracujesz w biurze polityka i debatuje się nad nowym prawem, wyborcy mogą pisać do biura e-maile popierające lub przeciwko danemu prawu. Załóżmy, że twoim zadaniem jest przeczytanie e-maili i posortowanie ich na 2 stosy, *za* i *przeciw*. Jeśli byłoby dużo e-maili, możesz być przytłoczony próbą przeczytania ich wszystkich. Czy nie byłoby miło, gdyby bot mógł przeczytać je za ciebie, zrozumieć je i powiedzieć, do którego stosu należy każdy e-mail? 
> 
> Jednym ze sposobów osiągnięcia tego jest użycie uczenia maszynowego. Możesz wytrenować model na części e-maili *przeciw* i części e-maili *za*. Model skojarzyłby frazy i słowa z jedną lub drugą stroną, *ale nie rozumiałby żadnej treści*, tylko to, że pewne słowa i wzorce częściej pojawiają się w e-mailach *przeciw* lub *za*. Możesz przetestować go na e-mailach, których nie użyłeś do trenowania modelu, i sprawdzić, czy doszedł do tych samych wniosków co ty. Następnie, gdy będziesz zadowolony z dokładności modelu, możesz przetwarzać przyszłe e-maile bez konieczności czytania każdego z nich.

✅ Czy ten proces przypomina procesy, które stosowałeś w poprzednich lekcjach?

## Ćwiczenie - sentymentalne zdania

Sentyment jest mierzony za pomocą *polaryzacji* od -1 do 1, gdzie -1 oznacza najbardziej negatywny sentyment, a 1 najbardziej pozytywny. Sentyment jest również mierzony za pomocą skali od 0 do 1 dla obiektywności (0) i subiektywności (1).

Przyjrzyj się ponownie *Dumie i uprzedzeniu* Jane Austen. Tekst jest dostępny tutaj: [Project Gutenberg](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm). Poniższy przykład pokazuje krótki program, który analizuje sentyment pierwszego i ostatniego zdania z książki oraz wyświetla jego polaryzację sentymentu i wynik obiektywności/subiektywności.

Powinieneś użyć biblioteki `TextBlob` (opisanej powyżej), aby określić `sentiment` (nie musisz pisać własnego kalkulatora sentymentu) w następującym zadaniu.

```python
from textblob import TextBlob

quote1 = """It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife."""

quote2 = """Darcy, as well as Elizabeth, really loved them; and they were both ever sensible of the warmest gratitude towards the persons who, by bringing her into Derbyshire, had been the means of uniting them."""

sentiment1 = TextBlob(quote1).sentiment
sentiment2 = TextBlob(quote2).sentiment

print(quote1 + " has a sentiment of " + str(sentiment1))
print(quote2 + " has a sentiment of " + str(sentiment2))
```

Widzisz następujący wynik:

```output
It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want # of a wife. has a sentiment of Sentiment(polarity=0.20952380952380953, subjectivity=0.27142857142857146)

Darcy, as well as Elizabeth, really loved them; and they were
     both ever sensible of the warmest gratitude towards the persons
      who, by bringing her into Derbyshire, had been the means of
      uniting them. has a sentiment of Sentiment(polarity=0.7, subjectivity=0.8)
```

## Wyzwanie - sprawdź polaryzację sentymentu

Twoim zadaniem jest określenie, za pomocą polaryzacji sentymentu, czy *Duma i uprzedzenie* ma więcej absolutnie pozytywnych zdań niż absolutnie negatywnych. W tym zadaniu możesz założyć, że wynik polaryzacji 1 lub -1 jest absolutnie pozytywny lub negatywny.

**Kroki:**

1. Pobierz [kopię Dumy i uprzedzenia](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm) z Project Gutenberg jako plik .txt. Usuń metadane na początku i końcu pliku, pozostawiając tylko oryginalny tekst.
2. Otwórz plik w Pythonie i wyodrębnij zawartość jako ciąg znaków.
3. Utwórz TextBlob używając ciągu znaków z książki.
4. Analizuj każde zdanie w książce w pętli:
   1. Jeśli polaryzacja wynosi 1 lub -1, zapisz zdanie w tablicy lub liście pozytywnych lub negatywnych wiadomości.
5. Na końcu wydrukuj wszystkie pozytywne zdania i negatywne zdania (osobno) oraz ich liczbę.

Oto przykładowe [rozwiązanie](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/3-Translation-Sentiment/solution/notebook.ipynb).

✅ Sprawdź swoją wiedzę

1. Sentyment opiera się na słowach użytych w zdaniu, ale czy kod *rozumie* te słowa?
2. Czy uważasz, że polaryzacja sentymentu jest dokładna, innymi słowy, czy *zgadzasz się* z wynikami?
   1. W szczególności, czy zgadzasz się lub nie zgadzasz z absolutnie **pozytywną** polaryzacją następujących zdań?
      * “What an excellent father you have, girls!” said she, when the door was shut.
      * “Your examination of Mr. Darcy is over, I presume,” said Miss Bingley; “and pray what is the result?” “I am perfectly convinced by it that Mr. Darcy has no defect.
      * How wonderfully these sort of things occur!
      * I have the greatest dislike in the world to that sort of thing.
      * Charlotte is an excellent manager, I dare say.
      * “This is delightful indeed!
      * I am so happy!
      * Your idea of the ponies is delightful.
   2. Następne 3 zdania zostały ocenione jako absolutnie pozytywne, ale po dokładnym przeczytaniu nie są pozytywnymi zdaniami. Dlaczego analiza sentymentu uznała je za pozytywne zdania?
      * Happy shall I be, when his stay at Netherfield is over!” “I wish I could say anything to comfort you,” replied Elizabeth; “but it is wholly out of my power.
      * If I could but see you as happy!
      * Our distress, my dear Lizzy, is very great.
   3. Czy zgadzasz się lub nie zgadzasz z absolutnie **negatywną** polaryzacją następujących zdań?
      - Everybody is disgusted with his pride.
      - “I should like to know how he behaves among strangers.” “You shall hear then—but prepare yourself for something very dreadful.
      - The pause was to Elizabeth’s feelings dreadful.
      - It would be dreadful!

✅ Każdy miłośnik Jane Austen zrozumie, że często używa swoich książek do krytykowania bardziej absurdalnych aspektów angielskiego społeczeństwa regencyjnego. Elizabeth Bennett, główna bohaterka *Dumy i uprzedzenia*, jest bystrą obserwatorką społeczną (jak autorka), a jej język jest często mocno zniuansowany. Nawet pan Darcy (miłość w tej historii) zauważa zabawne i żartobliwe użycie języka przez Elizabeth: "I have had the pleasure of your acquaintance long enough to know that you find great enjoyment in occasionally professing opinions which in fact are not your own."

---

## 🚀Wyzwanie

Czy możesz uczynić Marvina jeszcze lepszym, wyodrębniając inne cechy z danych wejściowych użytkownika?

## [Quiz po wykładzie](https://ff-quizzes.netlify.app/en/ml/)

## Przegląd i samodzielna nauka
Istnieje wiele sposobów na wydobycie sentymentu z tekstu. Pomyśl o zastosowaniach biznesowych, które mogą korzystać z tej techniki. Zastanów się, jak może to pójść nie tak. Przeczytaj więcej o zaawansowanych systemach gotowych do użytku w przedsiębiorstwach, które analizują sentyment, takich jak [Azure Text Analysis](https://docs.microsoft.com/azure/cognitive-services/Text-Analytics/how-tos/text-analytics-how-to-sentiment-analysis?tabs=version-3-1?WT.mc_id=academic-77952-leestott). Przetestuj niektóre z powyższych zdań z "Dumy i uprzedzenia" i sprawdź, czy potrafi wykryć niuanse.

## Zadanie

[Licencja poetycka](assignment.md)

---

**Zastrzeżenie**:  
Ten dokument został przetłumaczony za pomocą usługi tłumaczeniowej AI [Co-op Translator](https://github.com/Azure/co-op-translator). Chociaż dokładamy wszelkich starań, aby tłumaczenie było precyzyjne, prosimy pamiętać, że automatyczne tłumaczenia mogą zawierać błędy lub nieścisłości. Oryginalny dokument w jego rodzimym języku powinien być uznawany za wiarygodne źródło. W przypadku informacji krytycznych zaleca się skorzystanie z profesjonalnego tłumaczenia wykonanego przez człowieka. Nie ponosimy odpowiedzialności za jakiekolwiek nieporozumienia lub błędne interpretacje wynikające z korzystania z tego tłumaczenia.