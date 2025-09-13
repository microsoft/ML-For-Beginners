<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1c2ec40cf55c98a028a359c27ef7e45a",
  "translation_date": "2025-09-05T08:31:36+00:00",
  "source_file": "6-NLP/1-Introduction-to-NLP/README.md",
  "language_code": "pl"
}
-->
# Wprowadzenie do przetwarzania języka naturalnego

Ta lekcja obejmuje krótką historię i ważne pojęcia związane z *przetwarzaniem języka naturalnego*, poddziedziną *lingwistyki komputerowej*.

## [Quiz przed wykładem](https://ff-quizzes.netlify.app/en/ml/)

## Wprowadzenie

NLP, jak jest powszechnie nazywane, to jedna z najbardziej znanych dziedzin, w których uczenie maszynowe zostało zastosowane i używane w oprogramowaniu produkcyjnym.

✅ Czy możesz pomyśleć o oprogramowaniu, którego używasz codziennie, a które prawdopodobnie zawiera elementy NLP? A co z programami do edycji tekstu lub aplikacjami mobilnymi, których używasz regularnie?

Dowiesz się o:

- **Idei języków**. Jak rozwijały się języki i jakie były główne obszary badań.
- **Definicjach i pojęciach**. Poznasz definicje i pojęcia dotyczące tego, jak komputery przetwarzają tekst, w tym analizę składniową, gramatykę oraz identyfikację rzeczowników i czasowników. W tej lekcji znajdziesz kilka zadań programistycznych oraz wprowadzenie do ważnych koncepcji, które później nauczysz się kodować w kolejnych lekcjach.

## Lingwistyka komputerowa

Lingwistyka komputerowa to dziedzina badań i rozwoju, która od wielu dekad zajmuje się tym, jak komputery mogą pracować z językami, a nawet je rozumieć, tłumaczyć i komunikować się za ich pomocą. Przetwarzanie języka naturalnego (NLP) to pokrewna dziedzina, skupiająca się na tym, jak komputery mogą przetwarzać języki "naturalne", czyli ludzkie.

### Przykład - dyktowanie na telefonie

Jeśli kiedykolwiek dyktowałeś coś swojemu telefonowi zamiast pisać lub zadawałeś pytanie wirtualnemu asystentowi, Twoja mowa została przekształcona w formę tekstową, a następnie przetworzona lub *zanalizowana* w języku, którym się posługiwałeś. Wykryte słowa kluczowe zostały następnie przetworzone na format, który telefon lub asystent mógł zrozumieć i na jego podstawie podjąć działanie.

![comprehension](../../../../6-NLP/1-Introduction-to-NLP/images/comprehension.png)
> Prawdziwe zrozumienie językowe jest trudne! Obraz autorstwa [Jen Looper](https://twitter.com/jenlooper)

### Jak to jest możliwe?

Jest to możliwe dzięki temu, że ktoś napisał program komputerowy, który to umożliwia. Kilka dekad temu niektórzy pisarze science fiction przewidywali, że ludzie będą głównie rozmawiać ze swoimi komputerami, a komputery zawsze będą dokładnie rozumieć, co mają na myśli. Niestety, okazało się, że jest to trudniejszy problem, niż wielu sobie wyobrażało, i choć dziś jest to problem znacznie lepiej zrozumiany, nadal istnieją znaczące wyzwania w osiągnięciu "doskonałego" przetwarzania języka naturalnego, szczególnie w kontekście rozumienia znaczenia zdania. Szczególnie trudne jest rozumienie humoru czy wykrywanie emocji, takich jak sarkazm, w zdaniu.

W tym momencie możesz przypomnieć sobie lekcje w szkole, na których nauczyciel omawiał części gramatyczne zdania. W niektórych krajach uczniowie uczą się gramatyki i lingwistyki jako osobnego przedmiotu, ale w wielu krajach te tematy są częścią nauki języka: albo ojczystego w szkole podstawowej (nauka czytania i pisania), albo drugiego języka w szkole średniej. Nie martw się, jeśli nie jesteś ekspertem w rozróżnianiu rzeczowników od czasowników czy przysłówków od przymiotników!

Jeśli masz trudności z rozróżnieniem *czasu teraźniejszego prostego* od *czasu teraźniejszego ciągłego*, nie jesteś sam. To trudne dla wielu osób, nawet rodzimych użytkowników języka. Dobrą wiadomością jest to, że komputery są naprawdę dobre w stosowaniu formalnych reguł, a Ty nauczysz się pisać kod, który potrafi *analizować składnię* zdania równie dobrze jak człowiek. Większym wyzwaniem, które później zbadamy, jest rozumienie *znaczenia* i *emocji* zawartych w zdaniu.

## Wymagania wstępne

Głównym wymaganiem wstępnym dla tej lekcji jest umiejętność czytania i rozumienia języka, w którym jest napisana. Nie ma tu problemów matematycznych ani równań do rozwiązania. Choć pierwotny autor napisał tę lekcję po angielsku, jest ona również tłumaczona na inne języki, więc możesz czytać tłumaczenie. W przykładach używane są różne języki (aby porównać różne reguły gramatyczne różnych języków). Te fragmenty *nie* są tłumaczone, ale tekst objaśniający jest, więc znaczenie powinno być jasne.

Do zadań programistycznych będziesz używać Pythona, a przykłady są napisane w wersji Python 3.8.

W tej sekcji będziesz potrzebować i używać:

- **Znajomości Pythona 3**. Zrozumienie języka programowania Python 3, w tej lekcji używane są wejścia, pętle, odczyt plików, tablice.
- **Visual Studio Code + rozszerzenie**. Użyjemy Visual Studio Code i jego rozszerzenia dla Pythona. Możesz również użyć dowolnego IDE dla Pythona.
- **TextBlob**. [TextBlob](https://github.com/sloria/TextBlob) to uproszczona biblioteka do przetwarzania tekstu w Pythonie. Postępuj zgodnie z instrukcjami na stronie TextBlob, aby zainstalować ją na swoim systemie (zainstaluj również korpusy, jak pokazano poniżej):

   ```bash
   pip install -U textblob
   python -m textblob.download_corpora
   ```

> 💡 Wskazówka: Możesz uruchamiać Pythona bezpośrednio w środowiskach VS Code. Sprawdź [dokumentację](https://code.visualstudio.com/docs/languages/python?WT.mc_id=academic-77952-leestott), aby uzyskać więcej informacji.

## Rozmowa z maszynami

Historia prób nauczenia komputerów rozumienia ludzkiego języka sięga dziesięcioleci, a jednym z pierwszych naukowców, którzy rozważali przetwarzanie języka naturalnego, był *Alan Turing*.

### Test Turinga

Kiedy Turing badał *sztuczną inteligencję* w latach 50., zastanawiał się, czy można przeprowadzić test konwersacyjny, w którym człowiek i komputer (poprzez korespondencję pisemną) rozmawiają, a człowiek nie jest pewien, czy rozmawia z innym człowiekiem czy z komputerem.

Jeśli po pewnym czasie rozmowy człowiek nie byłby w stanie określić, czy odpowiedzi pochodzą od komputera czy nie, czy można powiedzieć, że komputer *myśli*?

### Inspiracja - 'gra naśladowcza'

Pomysł ten pochodził z gry towarzyskiej zwanej *Gra naśladowcza*, w której osoba przesłuchująca znajduje się sama w pokoju i ma za zadanie określić, które z dwóch osób (w innym pokoju) są odpowiednio mężczyzną i kobietą. Osoba przesłuchująca może wysyłać notatki i musi wymyślać pytania, na które pisemne odpowiedzi ujawnią płeć tajemniczej osoby. Oczywiście osoby w drugim pokoju próbują zmylić przesłuchującego, odpowiadając w sposób wprowadzający w błąd lub dezorientujący, jednocześnie sprawiając wrażenie, że odpowiadają szczerze.

### Rozwój Elizy

W latach 60. naukowiec z MIT, *Joseph Weizenbaum*, stworzył [*Elizę*](https://wikipedia.org/wiki/ELIZA), komputerowego "terapeutę", który zadawał człowiekowi pytania i sprawiał wrażenie, że rozumie jego odpowiedzi. Jednakże, choć Eliza potrafiła analizować zdanie i identyfikować pewne konstrukcje gramatyczne oraz słowa kluczowe, aby udzielić sensownej odpowiedzi, nie można było powiedzieć, że *rozumie* zdanie. Jeśli Eliza otrzymała zdanie w formacie "**Jestem** <u>smutny</u>", mogła przekształcić i zastąpić słowa w zdaniu, aby utworzyć odpowiedź "Jak długo **jesteś** <u>smutny</u>".

Dawało to wrażenie, że Eliza rozumie wypowiedź i zadaje pytanie uzupełniające, podczas gdy w rzeczywistości zmieniała czas i dodawała kilka słów. Jeśli Eliza nie mogła zidentyfikować słowa kluczowego, dla którego miała odpowiedź, zamiast tego udzielała losowej odpowiedzi, która mogła pasować do wielu różnych wypowiedzi. Elizę można było łatwo oszukać, na przykład jeśli użytkownik napisał "**Jesteś** <u>rowerem</u>", mogła odpowiedzieć "Jak długo **byłem** <u>rowerem</u>?", zamiast bardziej sensownej odpowiedzi.

[![Rozmowa z Elizą](https://img.youtube.com/vi/RMK9AphfLco/0.jpg)](https://youtu.be/RMK9AphfLco "Rozmowa z Elizą")

> 🎥 Kliknij obrazek powyżej, aby obejrzeć film o oryginalnym programie ELIZA

> Uwaga: Możesz przeczytać oryginalny opis [Elizy](https://cacm.acm.org/magazines/1966/1/13317-elizaa-computer-program-for-the-study-of-natural-language-communication-between-man-and-machine/abstract) opublikowany w 1966 roku, jeśli masz konto ACM. Alternatywnie, przeczytaj o Elizie na [wikipedii](https://wikipedia.org/wiki/ELIZA).

## Ćwiczenie - kodowanie podstawowego bota konwersacyjnego

Bot konwersacyjny, taki jak Eliza, to program, który pobiera dane wejściowe od użytkownika i sprawia wrażenie, że rozumie i inteligentnie odpowiada. W przeciwieństwie do Elizy, nasz bot nie będzie miał wielu reguł, które sprawiają wrażenie inteligentnej rozmowy. Zamiast tego nasz bot będzie miał tylko jedną umiejętność: utrzymywanie rozmowy za pomocą losowych odpowiedzi, które mogą pasować do niemal każdej trywialnej rozmowy.

### Plan

Twoje kroki przy budowaniu bota konwersacyjnego:

1. Wyświetl instrukcje informujące użytkownika, jak wchodzić w interakcję z botem
2. Rozpocznij pętlę
   1. Pobierz dane wejściowe od użytkownika
   2. Jeśli użytkownik poprosi o zakończenie, zakończ
   3. Przetwórz dane wejściowe użytkownika i określ odpowiedź (w tym przypadku odpowiedź jest losowym wyborem z listy możliwych ogólnych odpowiedzi)
   4. Wyświetl odpowiedź
3. Wróć do kroku 2

### Budowanie bota

Stwórzmy teraz bota. Zaczniemy od zdefiniowania kilku fraz.

1. Stwórz tego bota samodzielnie w Pythonie, używając następujących losowych odpowiedzi:

    ```python
    random_responses = ["That is quite interesting, please tell me more.",
                        "I see. Do go on.",
                        "Why do you say that?",
                        "Funny weather we've been having, isn't it?",
                        "Let's change the subject.",
                        "Did you catch the game last night?"]
    ```

    Oto przykładowy wynik, który może Cię poprowadzić (dane wejściowe użytkownika są na liniach zaczynających się od `>`):

    ```output
    Hello, I am Marvin, the simple robot.
    You can end this conversation at any time by typing 'bye'
    After typing each answer, press 'enter'
    How are you today?
    > I am good thanks
    That is quite interesting, please tell me more.
    > today I went for a walk     
    Did you catch the game last night?
    > I did, but my team lost
    Funny weather we've been having, isn't it?
    > yes but I hope next week is better
    Let's change the subject.
    > ok, lets talk about music
    Why do you say that?
    > because I like music!
    Why do you say that?
    > bye
    It was nice talking to you, goodbye!
    ```

    Jedno z możliwych rozwiązań zadania znajduje się [tutaj](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/1-Introduction-to-NLP/solution/bot.py)

    ✅ Zatrzymaj się i zastanów

    1. Czy uważasz, że losowe odpowiedzi mogłyby "oszukać" kogoś, aby pomyślał, że bot faktycznie go rozumie?
    2. Jakie funkcje musiałby mieć bot, aby być bardziej skutecznym?
    3. Jeśli bot naprawdę mógłby "rozumieć" znaczenie zdania, czy musiałby "pamiętać" znaczenie poprzednich zdań w rozmowie?

---

## 🚀Wyzwanie

Wybierz jeden z elementów "zatrzymaj się i zastanów" powyżej i spróbuj go zaimplementować w kodzie lub napisz rozwiązanie na papierze, używając pseudokodu.

W następnej lekcji dowiesz się o kilku innych podejściach do analizy języka naturalnego i uczenia maszynowego.

## [Quiz po wykładzie](https://ff-quizzes.netlify.app/en/ml/)

## Przegląd i samodzielna nauka

Zapoznaj się z poniższymi odniesieniami jako możliwościami dalszej lektury.

### Odniesienia

1. Schubert, Lenhart, "Lingwistyka komputerowa", *The Stanford Encyclopedia of Philosophy* (Spring 2020 Edition), Edward N. Zalta (ed.), URL = <https://plato.stanford.edu/archives/spr2020/entries/computational-linguistics/>.
2. Princeton University "O WordNet." [WordNet](https://wordnet.princeton.edu/). Princeton University. 2010. 

## Zadanie 

[Znajdź bota](assignment.md)

---

**Zastrzeżenie**:  
Ten dokument został przetłumaczony za pomocą usługi tłumaczeniowej AI [Co-op Translator](https://github.com/Azure/co-op-translator). Chociaż dokładamy wszelkich starań, aby tłumaczenie było precyzyjne, prosimy pamiętać, że automatyczne tłumaczenia mogą zawierać błędy lub nieścisłości. Oryginalny dokument w jego rodzimym języku powinien być uznawany za wiarygodne źródło. W przypadku informacji krytycznych zaleca się skorzystanie z profesjonalnego tłumaczenia wykonanego przez człowieka. Nie ponosimy odpowiedzialności za jakiekolwiek nieporozumienia lub błędne interpretacje wynikające z korzystania z tego tłumaczenia.