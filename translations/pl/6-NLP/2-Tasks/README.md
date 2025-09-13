<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "5f3cb462e3122e1afe7ab0050ccf2bd3",
  "translation_date": "2025-09-05T08:29:20+00:00",
  "source_file": "6-NLP/2-Tasks/README.md",
  "language_code": "pl"
}
-->
# Typowe zadania i techniki przetwarzania języka naturalnego

W większości zadań związanych z *przetwarzaniem języka naturalnego* tekst, który ma być przetworzony, musi zostać podzielony, przeanalizowany, a wyniki zapisane lub porównane z regułami i zestawami danych. Te zadania pozwalają programiście wyciągnąć _znaczenie_, _intencję_ lub tylko _częstotliwość_ występowania terminów i słów w tekście.

## [Quiz przed wykładem](https://ff-quizzes.netlify.app/en/ml/)

Poznajmy typowe techniki stosowane w przetwarzaniu tekstu. W połączeniu z uczeniem maszynowym techniki te pomagają efektywnie analizować duże ilości tekstu. Zanim jednak zastosujesz ML do tych zadań, zrozummy problemy, z którymi mierzy się specjalista NLP.

## Typowe zadania w NLP

Istnieje wiele sposobów analizy tekstu, nad którym pracujesz. Są zadania, które możesz wykonać, a dzięki nim możesz zrozumieć tekst i wyciągnąć wnioski. Zazwyczaj wykonujesz te zadania w określonej kolejności.

### Tokenizacja

Prawdopodobnie pierwszym krokiem, który większość algorytmów NLP musi wykonać, jest podzielenie tekstu na tokeny, czyli słowa. Choć brzmi to prosto, uwzględnienie znaków interpunkcyjnych oraz różnych językowych ograniczników zdań i słów może być trudne. Możesz potrzebować różnych metod, aby określić granice.

![tokenizacja](../../../../6-NLP/2-Tasks/images/tokenization.png)
> Tokenizacja zdania z **Dumy i uprzedzenia**. Infografika autorstwa [Jen Looper](https://twitter.com/jenlooper)

### Osadzenia (Embeddings)

[Osadzenia słów](https://wikipedia.org/wiki/Word_embedding) to sposób na numeryczne przekształcenie danych tekstowych. Osadzenia są tworzone w taki sposób, aby słowa o podobnym znaczeniu lub używane razem grupowały się w klastrach.

![osadzenia słów](../../../../6-NLP/2-Tasks/images/embedding.png)
> "Mam najwyższy szacunek dla twoich nerwów, są moimi starymi przyjaciółmi." - Osadzenia słów dla zdania z **Dumy i uprzedzenia**. Infografika autorstwa [Jen Looper](https://twitter.com/jenlooper)

✅ Wypróbuj [to ciekawe narzędzie](https://projector.tensorflow.org/), aby eksperymentować z osadzeniami słów. Kliknięcie na jedno słowo pokazuje klastry podobnych słów: 'toy' grupuje się z 'disney', 'lego', 'playstation' i 'console'.

### Parsowanie i Tagowanie Części Mowy

Każde słowo, które zostało podzielone na tokeny, może być oznaczone jako część mowy - rzeczownik, czasownik lub przymiotnik. Zdanie `the quick red fox jumped over the lazy brown dog` może być oznaczone jako fox = rzeczownik, jumped = czasownik.

![parsowanie](../../../../6-NLP/2-Tasks/images/parse.png)

> Parsowanie zdania z **Dumy i uprzedzenia**. Infografika autorstwa [Jen Looper](https://twitter.com/jenlooper)

Parsowanie polega na rozpoznawaniu, które słowa są ze sobą powiązane w zdaniu - na przykład `the quick red fox jumped` to sekwencja przymiotnik-rzeczownik-czasownik, która jest oddzielna od sekwencji `lazy brown dog`.

### Częstotliwość słów i fraz

Przydatnym procesem podczas analizy dużej ilości tekstu jest stworzenie słownika wszystkich interesujących słów lub fraz oraz ich częstotliwości występowania. Fraza `the quick red fox jumped over the lazy brown dog` ma częstotliwość 2 dla słowa "the".

Spójrzmy na przykład tekstu, w którym liczymy częstotliwość słów. Wiersz Rudyard Kiplinga "The Winners" zawiera następujący fragment:

```output
What the moral? Who rides may read.
When the night is thick and the tracks are blind
A friend at a pinch is a friend, indeed,
But a fool to wait for the laggard behind.
Down to Gehenna or up to the Throne,
He travels the fastest who travels alone.
```

Ponieważ częstotliwość fraz może być wrażliwa na wielkość liter lub nie, fraza `a friend` występuje 2 razy, `the` występuje 6 razy, a `travels` występuje 2 razy.

### N-gramy

Tekst można podzielić na sekwencje słów o określonej długości: jedno słowo (unigram), dwa słowa (bigramy), trzy słowa (trigramy) lub dowolną liczbę słów (n-gramy).

Na przykład `the quick red fox jumped over the lazy brown dog` z wartością n-gramu równą 2 daje następujące n-gramy:

1. the quick 
2. quick red 
3. red fox
4. fox jumped 
5. jumped over 
6. over the 
7. the lazy 
8. lazy brown 
9. brown dog

Można to łatwiej zobrazować jako przesuwające się okno nad zdaniem. Oto przykład dla n-gramów złożonych z 3 słów, gdzie n-gram jest pogrubiony w każdym zdaniu:

1.   <u>**the quick red**</u> fox jumped over the lazy brown dog
2.   the **<u>quick red fox</u>** jumped over the lazy brown dog
3.   the quick **<u>red fox jumped</u>** over the lazy brown dog
4.   the quick red **<u>fox jumped over</u>** the lazy brown dog
5.   the quick red fox **<u>jumped over the</u>** lazy brown dog
6.   the quick red fox jumped **<u>over the lazy</u>** brown dog
7.   the quick red fox jumped over <u>**the lazy brown**</u> dog
8.   the quick red fox jumped over the **<u>lazy brown dog</u>**

![przesuwające się okno n-gramów](../../../../6-NLP/2-Tasks/images/n-grams.gif)

> Wartość n-gramu: 3. Infografika autorstwa [Jen Looper](https://twitter.com/jenlooper)

### Ekstrakcja fraz rzeczownikowych

W większości zdań znajduje się rzeczownik, który jest podmiotem lub dopełnieniem zdania. W języku angielskim często można go zidentyfikować jako poprzedzony słowami 'a', 'an' lub 'the'. Identyfikacja podmiotu lub dopełnienia zdania poprzez 'ekstrakcję frazy rzeczownikowej' jest częstym zadaniem w NLP, gdy próbuje się zrozumieć znaczenie zdania.

✅ W zdaniu "I cannot fix on the hour, or the spot, or the look or the words, which laid the foundation. It is too long ago. I was in the middle before I knew that I had begun." czy potrafisz zidentyfikować frazy rzeczownikowe?

W zdaniu `the quick red fox jumped over the lazy brown dog` znajdują się 2 frazy rzeczownikowe: **quick red fox** i **lazy brown dog**.

### Analiza sentymentu

Zdanie lub tekst można przeanalizować pod kątem sentymentu, czyli tego, jak *pozytywny* lub *negatywny* jest. Sentyment jest mierzony w *polaryzacji* oraz *obiektywności/subiektywności*. Polaryzacja jest mierzona od -1.0 do 1.0 (negatywna do pozytywnej) oraz od 0.0 do 1.0 (najbardziej obiektywna do najbardziej subiektywnej).

✅ Później dowiesz się, że istnieją różne sposoby określania sentymentu za pomocą uczenia maszynowego, ale jednym ze sposobów jest posiadanie listy słów i fraz sklasyfikowanych jako pozytywne lub negatywne przez eksperta i zastosowanie tego modelu do tekstu w celu obliczenia wyniku polaryzacji. Czy widzisz, jak to mogłoby działać w niektórych okolicznościach, a w innych mniej?

### Odmiana

Odmiana pozwala na uzyskanie liczby pojedynczej lub mnogiej danego słowa.

### Lematyzacja

*Lemmatyzacja* to proces sprowadzania słowa do jego podstawowej formy, czyli *lematu*. Na przykład *flew*, *flies*, *flying* mają lemat czasownika *fly*.

Dostępne są również przydatne bazy danych dla badaczy NLP, w tym:

### WordNet

[WordNet](https://wordnet.princeton.edu/) to baza danych słów, synonimów, antonimów i wielu innych szczegółów dla każdego słowa w różnych językach. Jest niezwykle przydatna przy tworzeniu tłumaczeń, korektorów pisowni czy narzędzi językowych każdego rodzaju.

## Biblioteki NLP

Na szczęście nie musisz samodzielnie budować wszystkich tych technik, ponieważ istnieją doskonałe biblioteki Python, które sprawiają, że NLP jest bardziej dostępne dla programistów, którzy nie specjalizują się w przetwarzaniu języka naturalnego ani uczeniu maszynowym. W kolejnych lekcjach znajdziesz więcej przykładów, ale tutaj poznasz kilka przydatnych przykładów, które pomogą Ci w następnym zadaniu.

### Ćwiczenie - korzystanie z biblioteki `TextBlob`

Użyjmy biblioteki TextBlob, ponieważ zawiera ona pomocne API do radzenia sobie z tego typu zadaniami. TextBlob "opiera się na potężnych fundamentach [NLTK](https://nltk.org) i [pattern](https://github.com/clips/pattern), i dobrze współpracuje z obiema."

> Uwaga: Przydatny [Przewodnik Szybkiego Startu](https://textblob.readthedocs.io/en/dev/quickstart.html#quickstart) jest dostępny dla TextBlob i polecany dla doświadczonych programistów Python.

Podczas próby identyfikacji *frazy rzeczownikowej* TextBlob oferuje kilka opcji ekstraktorów do znajdowania fraz rzeczownikowych.

1. Spójrz na `ConllExtractor`.

    ```python
    from textblob import TextBlob
    from textblob.np_extractors import ConllExtractor
    # import and create a Conll extractor to use later 
    extractor = ConllExtractor()
    
    # later when you need a noun phrase extractor:
    user_input = input("> ")
    user_input_blob = TextBlob(user_input, np_extractor=extractor)  # note non-default extractor specified
    np = user_input_blob.noun_phrases                                    
    ```

    > Co tu się dzieje? [ConllExtractor](https://textblob.readthedocs.io/en/dev/api_reference.html?highlight=Conll#textblob.en.np_extractors.ConllExtractor) to "Ekstraktor fraz rzeczownikowych, który wykorzystuje parsowanie chunków wytrenowane na korpusie treningowym ConLL-2000." ConLL-2000 odnosi się do konferencji Computational Natural Language Learning z 2000 roku. Każdego roku konferencja organizowała warsztaty, aby rozwiązać trudny problem NLP, a w 2000 roku był to chunking fraz rzeczownikowych. Model został wytrenowany na danych z Wall Street Journal, z "sekcjami 15-18 jako danymi treningowymi (211727 tokenów) i sekcją 20 jako danymi testowymi (47377 tokenów)". Możesz zapoznać się z procedurami użytymi [tutaj](https://www.clips.uantwerpen.be/conll2000/chunking/) oraz z [wynikami](https://ifarm.nl/erikt/research/np-chunking.html).

### Wyzwanie - ulepszanie swojego bota za pomocą NLP

W poprzedniej lekcji stworzyłeś bardzo prostego bota Q&A. Teraz sprawisz, że Marvin będzie bardziej empatyczny, analizując Twoje dane wejściowe pod kątem sentymentu i drukując odpowiedź pasującą do tego sentymentu. Musisz również zidentyfikować `noun_phrase` i zapytać o nią.

Twoje kroki przy budowaniu lepszego bota konwersacyjnego:

1. Wydrukuj instrukcje informujące użytkownika, jak wchodzić w interakcję z botem
2. Rozpocznij pętlę 
   1. Przyjmij dane wejściowe od użytkownika
   2. Jeśli użytkownik poprosi o zakończenie, zakończ
   3. Przetwórz dane wejściowe użytkownika i określ odpowiednią odpowiedź na podstawie sentymentu
   4. Jeśli w sentymencie zostanie wykryta fraza rzeczownikowa, zmień jej liczbę mnogą i zapytaj o nią
   5. Wydrukuj odpowiedź
3. Wróć do kroku 2

Oto fragment kodu do określania sentymentu za pomocą TextBlob. Zauważ, że są tylko cztery *gradienty* odpowiedzi na sentyment (możesz dodać więcej, jeśli chcesz):

```python
if user_input_blob.polarity <= -0.5:
  response = "Oh dear, that sounds bad. "
elif user_input_blob.polarity <= 0:
  response = "Hmm, that's not great. "
elif user_input_blob.polarity <= 0.5:
  response = "Well, that sounds positive. "
elif user_input_blob.polarity <= 1:
  response = "Wow, that sounds great. "
```

Oto przykładowy wynik, który może Cię poprowadzić (dane wejściowe użytkownika są na liniach zaczynających się od >):

```output
Hello, I am Marvin, the friendly robot.
You can end this conversation at any time by typing 'bye'
After typing each answer, press 'enter'
How are you today?
> I am ok
Well, that sounds positive. Can you tell me more?
> I went for a walk and saw a lovely cat
Well, that sounds positive. Can you tell me more about lovely cats?
> cats are the best. But I also have a cool dog
Wow, that sounds great. Can you tell me more about cool dogs?
> I have an old hounddog but he is sick
Hmm, that's not great. Can you tell me more about old hounddogs?
> bye
It was nice talking to you, goodbye!
```

Jedno z możliwych rozwiązań zadania znajduje się [tutaj](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/2-Tasks/solution/bot.py)

✅ Sprawdź swoją wiedzę

1. Czy uważasz, że empatyczne odpowiedzi mogłyby 'oszukać' kogoś, że bot faktycznie go rozumie?
2. Czy identyfikacja frazy rzeczownikowej sprawia, że bot jest bardziej 'wiarygodny'?
3. Dlaczego ekstrakcja 'frazy rzeczownikowej' ze zdania jest przydatna?

---

Zaimplementuj bota z poprzedniego sprawdzania wiedzy i przetestuj go na znajomym. Czy potrafi ich oszukać? Czy możesz sprawić, że Twój bot będzie bardziej 'wiarygodny'?

## 🚀Wyzwanie

Wykonaj zadanie z poprzedniego sprawdzania wiedzy i spróbuj je zaimplementować. Przetestuj bota na znajomym. Czy potrafi ich oszukać? Czy możesz sprawić, że Twój bot będzie bardziej 'wiarygodny'?

## [Quiz po wykładzie](https://ff-quizzes.netlify.app/en/ml/)

## Przegląd i samodzielna nauka

W kolejnych lekcjach dowiesz się więcej o analizie sentymentu. Zbadaj tę interesującą technikę w artykułach takich jak te na [KDNuggets](https://www.kdnuggets.com/tag/nlp)

## Zadanie 

[Spraw, by bot odpowiadał](assignment.md)

---

**Zastrzeżenie**:  
Ten dokument został przetłumaczony za pomocą usługi tłumaczeniowej AI [Co-op Translator](https://github.com/Azure/co-op-translator). Chociaż dokładamy wszelkich starań, aby tłumaczenie było precyzyjne, prosimy pamiętać, że automatyczne tłumaczenia mogą zawierać błędy lub nieścisłości. Oryginalny dokument w jego rodzimym języku powinien być uznawany za wiarygodne źródło. W przypadku informacji krytycznych zaleca się skorzystanie z profesjonalnego tłumaczenia wykonanego przez człowieka. Nie ponosimy odpowiedzialności za jakiekolwiek nieporozumienia lub błędne interpretacje wynikające z korzystania z tego tłumaczenia.