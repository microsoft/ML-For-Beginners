<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "69389392fa6346e0dfa30f664b7b6fec",
  "translation_date": "2025-09-05T08:22:20+00:00",
  "source_file": "1-Introduction/1-intro-to-ML/README.md",
  "language_code": "pl"
}
-->
# Wprowadzenie do uczenia maszynowego

## [Quiz przed wykładem](https://ff-quizzes.netlify.app/en/ml/)

---

[![ML dla początkujących - Wprowadzenie do uczenia maszynowego dla początkujących](https://img.youtube.com/vi/6mSx_KJxcHI/0.jpg)](https://youtu.be/6mSx_KJxcHI "ML dla początkujących - Wprowadzenie do uczenia maszynowego dla początkujących")

> 🎥 Kliknij obrazek powyżej, aby obejrzeć krótki film związany z tą lekcją.

Witamy na kursie klasycznego uczenia maszynowego dla początkujących! Niezależnie od tego, czy dopiero zaczynasz swoją przygodę z tym tematem, czy jesteś doświadczonym praktykiem ML, który chce odświeżyć wiedzę w danej dziedzinie, cieszymy się, że do nas dołączasz! Chcemy stworzyć przyjazne miejsce startowe dla Twojej nauki ML i chętnie ocenimy, odpowiemy na Twoje [opinie](https://github.com/microsoft/ML-For-Beginners/discussions) oraz uwzględnimy je w kursie.

[![Wprowadzenie do ML](https://img.youtube.com/vi/h0e2HAPTGF4/0.jpg)](https://youtu.be/h0e2HAPTGF4 "Wprowadzenie do ML")

> 🎥 Kliknij obrazek powyżej, aby obejrzeć film: John Guttag z MIT wprowadza do uczenia maszynowego

---
## Rozpoczęcie pracy z uczeniem maszynowym

Przed rozpoczęciem pracy z tym kursem musisz przygotować swój komputer do lokalnego uruchamiania notebooków.

- **Skonfiguruj swój komputer za pomocą tych filmów**. Skorzystaj z poniższych linków, aby dowiedzieć się [jak zainstalować Python](https://youtu.be/CXZYvNRIAKM) na swoim systemie oraz [jak skonfigurować edytor tekstu](https://youtu.be/EU8eayHWoZg) do programowania.
- **Naucz się Pythona**. Zaleca się również podstawową znajomość [Pythona](https://docs.microsoft.com/learn/paths/python-language/?WT.mc_id=academic-77952-leestott), języka programowania przydatnego dla naukowców zajmujących się danymi, którego używamy w tym kursie.
- **Poznaj Node.js i JavaScript**. Kilka razy w tym kursie używamy JavaScriptu do tworzenia aplikacji webowych, więc będziesz potrzebować [node](https://nodejs.org) i [npm](https://www.npmjs.com/) oraz [Visual Studio Code](https://code.visualstudio.com/) do programowania w Pythonie i JavaScript.
- **Załóż konto na GitHub**. Skoro znalazłeś nas tutaj na [GitHub](https://github.com), być może już masz konto, ale jeśli nie, załóż je, a następnie zrób fork tego kursu, aby korzystać z niego na własny użytek. (Możesz też dać nam gwiazdkę 😊)
- **Poznaj Scikit-learn**. Zapoznaj się z [Scikit-learn](https://scikit-learn.org/stable/user_guide.html), zestawem bibliotek ML, które wykorzystujemy w tych lekcjach.

---
## Czym jest uczenie maszynowe?

Termin 'uczenie maszynowe' jest jednym z najpopularniejszych i najczęściej używanych terminów współczesności. Istnieje spore prawdopodobieństwo, że słyszałeś ten termin przynajmniej raz, jeśli masz jakąkolwiek styczność z technologią, niezależnie od dziedziny, w której pracujesz. Mechanizmy uczenia maszynowego są jednak tajemnicą dla większości ludzi. Dla początkującego w tej dziedzinie temat może czasami wydawać się przytłaczający. Dlatego ważne jest, aby zrozumieć, czym właściwie jest uczenie maszynowe i poznawać je krok po kroku, poprzez praktyczne przykłady.

---
## Krzywa popularności

![ml hype curve](../../../../1-Introduction/1-intro-to-ML/images/hype.png)

> Google Trends pokazuje ostatnią 'krzywą popularności' terminu 'uczenie maszynowe'

---
## Tajemniczy wszechświat

Żyjemy w wszechświecie pełnym fascynujących tajemnic. Wielcy naukowcy, tacy jak Stephen Hawking, Albert Einstein i wielu innych, poświęcili swoje życie na poszukiwanie znaczących informacji, które odkrywają tajemnice otaczającego nas świata. To jest ludzka kondycja uczenia się: dziecko uczy się nowych rzeczy i odkrywa strukturę swojego świata rok po roku, dorastając do dorosłości.

---
## Mózg dziecka

Mózg dziecka i jego zmysły postrzegają fakty otoczenia i stopniowo uczą się ukrytych wzorców życia, które pomagają dziecku tworzyć logiczne zasady identyfikacji poznanych wzorców. Proces uczenia się ludzkiego mózgu sprawia, że ludzie są najbardziej zaawansowanymi istotami żyjącymi na świecie. Ciągłe uczenie się poprzez odkrywanie ukrytych wzorców, a następnie innowacje na ich podstawie, pozwala nam stawać się coraz lepszymi przez całe życie. Ta zdolność uczenia się i ewolucji jest związana z koncepcją zwaną [plastycznością mózgu](https://www.simplypsychology.org/brain-plasticity.html). Powierzchownie możemy dostrzec pewne motywacyjne podobieństwa między procesem uczenia się ludzkiego mózgu a koncepcjami uczenia maszynowego.

---
## Ludzki mózg

[Ludzki mózg](https://www.livescience.com/29365-human-brain.html) postrzega rzeczy ze świata rzeczywistego, przetwarza postrzegane informacje, podejmuje racjonalne decyzje i wykonuje określone działania w zależności od okoliczności. To właśnie nazywamy inteligentnym zachowaniem. Kiedy programujemy imitację procesu inteligentnego zachowania na maszynie, nazywa się to sztuczną inteligencją (AI).

---
## Kilka terminów

Chociaż terminy mogą być mylone, uczenie maszynowe (ML) jest ważnym podzbiorem sztucznej inteligencji. **ML zajmuje się wykorzystaniem specjalistycznych algorytmów do odkrywania znaczących informacji i znajdowania ukrytych wzorców w postrzeganych danych, aby wspierać proces racjonalnego podejmowania decyzji**.

---
## AI, ML, Deep Learning

![AI, ML, deep learning, data science](../../../../1-Introduction/1-intro-to-ML/images/ai-ml-ds.png)

> Diagram pokazujący relacje między AI, ML, deep learning i data science. Infografika autorstwa [Jen Looper](https://twitter.com/jenlooper) inspirowana [tym grafikiem](https://softwareengineering.stackexchange.com/questions/366996/distinction-between-ai-ml-neural-networks-deep-learning-and-data-mining)

---
## Koncepcje do omówienia

W tym kursie omówimy tylko podstawowe koncepcje uczenia maszynowego, które początkujący musi znać. Skupimy się na tym, co nazywamy 'klasycznym uczeniem maszynowym', głównie korzystając z Scikit-learn, doskonałej biblioteki, którą wielu studentów wykorzystuje do nauki podstaw. Aby zrozumieć szersze koncepcje sztucznej inteligencji lub deep learning, niezbędna jest solidna wiedza podstawowa z zakresu uczenia maszynowego, którą chcemy tutaj zaoferować.

---
## W tym kursie nauczysz się:

- podstawowych koncepcji uczenia maszynowego
- historii ML
- ML i sprawiedliwości
- technik regresji w ML
- technik klasyfikacji w ML
- technik klastrowania w ML
- technik przetwarzania języka naturalnego w ML
- technik prognozowania szeregów czasowych w ML
- uczenia przez wzmacnianie
- zastosowań uczenia maszynowego w rzeczywistości

---
## Czego nie omówimy

- deep learning
- sieci neuronowych
- AI

Aby zapewnić lepsze doświadczenie edukacyjne, unikniemy złożoności sieci neuronowych, 'deep learning' - wielowarstwowego budowania modeli za pomocą sieci neuronowych - oraz AI, które omówimy w innym kursie. Oferujemy również nadchodzący kurs data science, który skupi się na tym aspekcie tej większej dziedziny.

---
## Dlaczego warto studiować uczenie maszynowe?

Uczenie maszynowe, z perspektywy systemowej, definiuje się jako tworzenie zautomatyzowanych systemów, które mogą uczyć się ukrytych wzorców z danych, aby wspierać podejmowanie inteligentnych decyzji.

Ta motywacja jest luźno inspirowana tym, jak ludzki mózg uczy się pewnych rzeczy na podstawie danych, które postrzega ze świata zewnętrznego.

✅ Zastanów się przez chwilę, dlaczego firma chciałaby zastosować strategie uczenia maszynowego zamiast tworzenia silnika opartego na twardo zakodowanych regułach.

---
## Zastosowania uczenia maszynowego

Zastosowania uczenia maszynowego są teraz niemal wszędzie i są tak wszechobecne jak dane przepływające w naszych społeczeństwach, generowane przez nasze smartfony, urządzenia połączone i inne systemy. Biorąc pod uwagę ogromny potencjał najnowocześniejszych algorytmów uczenia maszynowego, naukowcy badają ich zdolność do rozwiązywania wielowymiarowych i wielodyscyplinarnych problemów życia codziennego z wielkimi pozytywnymi rezultatami.

---
## Przykłady zastosowania ML

**Uczenie maszynowe można wykorzystać na wiele sposobów**:

- Do przewidywania prawdopodobieństwa wystąpienia choroby na podstawie historii medycznej pacjenta lub raportów.
- Do wykorzystania danych pogodowych w celu przewidywania zjawisk atmosferycznych.
- Do analizy sentymentu tekstu.
- Do wykrywania fałszywych wiadomości, aby zatrzymać rozprzestrzenianie się propagandy.

Finanse, ekonomia, nauki o Ziemi, eksploracja kosmosu, inżynieria biomedyczna, nauki kognitywne, a nawet dziedziny humanistyczne zaadaptowały uczenie maszynowe do rozwiązywania trudnych, wymagających przetwarzania danych problemów w swoich dziedzinach.

---
## Podsumowanie

Uczenie maszynowe automatyzuje proces odkrywania wzorców poprzez znajdowanie znaczących informacji z danych rzeczywistych lub generowanych. Udowodniło swoją wartość w biznesie, zdrowiu i aplikacjach finansowych, między innymi.

W niedalekiej przyszłości zrozumienie podstaw uczenia maszynowego stanie się koniecznością dla ludzi z każdej dziedziny ze względu na jego szerokie zastosowanie.

---
# 🚀 Wyzwanie

Naszkicuj, na papierze lub za pomocą aplikacji online, takiej jak [Excalidraw](https://excalidraw.com/), swoje rozumienie różnic między AI, ML, deep learning i data science. Dodaj kilka pomysłów na problemy, które każda z tych technik jest dobra w rozwiązywaniu.

# [Quiz po wykładzie](https://ff-quizzes.netlify.app/en/ml/)

---
# Przegląd i samodzielna nauka

Aby dowiedzieć się więcej o tym, jak pracować z algorytmami ML w chmurze, skorzystaj z tego [kursu](https://docs.microsoft.com/learn/paths/create-no-code-predictive-models-azure-machine-learning/?WT.mc_id=academic-77952-leestott).

Weź udział w [kursie](https://docs.microsoft.com/learn/modules/introduction-to-machine-learning/?WT.mc_id=academic-77952-leestott) dotyczącym podstaw ML.

---
# Zadanie

[Przygotuj się do pracy](assignment.md)

---

**Zastrzeżenie**:  
Ten dokument został przetłumaczony za pomocą usługi tłumaczeniowej AI [Co-op Translator](https://github.com/Azure/co-op-translator). Chociaż dokładamy wszelkich starań, aby tłumaczenie było precyzyjne, prosimy pamiętać, że automatyczne tłumaczenia mogą zawierać błędy lub nieścisłości. Oryginalny dokument w jego rodzimym języku powinien być uznawany za wiarygodne źródło. W przypadku informacji krytycznych zaleca się skorzystanie z profesjonalnego tłumaczenia wykonanego przez człowieka. Nie ponosimy odpowiedzialności za jakiekolwiek nieporozumienia lub błędne interpretacje wynikające z korzystania z tego tłumaczenia.