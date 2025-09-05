<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "83320d6b6994909e35d830cebf214039",
  "translation_date": "2025-09-05T08:18:26+00:00",
  "source_file": "9-Real-World/1-Applications/README.md",
  "language_code": "pl"
}
-->
# Postscript: Uczenie maszynowe w rzeczywistym świecie

![Podsumowanie uczenia maszynowego w rzeczywistym świecie w formie sketchnote](../../../../sketchnotes/ml-realworld.png)  
> Sketchnote autorstwa [Tomomi Imura](https://www.twitter.com/girlie_mac)

W tym kursie nauczyłeś się wielu sposobów przygotowywania danych do treningu i tworzenia modeli uczenia maszynowego. Zbudowałeś serię klasycznych modeli regresji, klasteryzacji, klasyfikacji, przetwarzania języka naturalnego oraz szeregów czasowych. Gratulacje! Teraz możesz się zastanawiać, do czego to wszystko służy... jakie są rzeczywiste zastosowania tych modeli?

Chociaż w przemyśle wiele uwagi przyciąga AI, które zazwyczaj wykorzystuje głębokie uczenie, klasyczne modele uczenia maszynowego wciąż znajdują cenne zastosowania. Być może nawet korzystasz z niektórych z tych zastosowań na co dzień! W tej lekcji dowiesz się, jak osiem różnych branż i dziedzin wykorzystuje te modele, aby ich aplikacje były bardziej wydajne, niezawodne, inteligentne i wartościowe dla użytkowników.

## [Quiz przed wykładem](https://ff-quizzes.netlify.app/en/ml/)

## 💰 Finanse

Sektor finansowy oferuje wiele możliwości zastosowania uczenia maszynowego. Wiele problemów w tej dziedzinie można modelować i rozwiązywać za pomocą ML.

### Wykrywanie oszustw związanych z kartami kredytowymi

Wcześniej w kursie poznaliśmy [klasteryzację metodą k-średnich](../../5-Clustering/2-K-Means/README.md), ale jak można ją wykorzystać do rozwiązywania problemów związanych z oszustwami kartowymi?

Klasteryzacja k-średnich jest przydatna w technice wykrywania oszustw kartowych zwanej **wykrywaniem odstających wartości**. Odstające wartości, czyli odchylenia w obserwacjach dotyczących zbioru danych, mogą wskazywać, czy karta kredytowa jest używana w normalny sposób, czy dzieje się coś nietypowego. Jak pokazano w poniższym artykule, dane dotyczące kart kredytowych można sortować za pomocą algorytmu k-średnich i przypisywać każdą transakcję do klastra na podstawie tego, jak bardzo odstaje od normy. Następnie można ocenić najbardziej ryzykowne klastry pod kątem transakcji oszukańczych i legalnych.  
[Źródło](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.680.1195&rep=rep1&type=pdf)

### Zarządzanie majątkiem

W zarządzaniu majątkiem osoba lub firma zajmuje się inwestycjami w imieniu swoich klientów. Ich zadaniem jest długoterminowe utrzymanie i pomnażanie majątku, dlatego kluczowe jest wybieranie inwestycji, które przynoszą dobre wyniki.

Jednym ze sposobów oceny wyników inwestycji jest regresja statystyczna. [Regresja liniowa](../../2-Regression/1-Tools/README.md) to cenne narzędzie do zrozumienia, jak fundusz radzi sobie w porównaniu z jakimś benchmarkiem. Możemy również określić, czy wyniki regresji są statystycznie istotne, czyli jak bardzo mogą wpłynąć na inwestycje klienta. Można nawet rozszerzyć analizę, stosując regresję wielokrotną, uwzględniając dodatkowe czynniki ryzyka. Przykład zastosowania tej metody dla konkretnego funduszu znajdziesz w poniższym artykule.  
[Źródło](http://www.brightwoodventures.com/evaluating-fund-performance-using-regression/)

## 🎓 Edukacja

Sektor edukacyjny to również bardzo interesująca dziedzina, w której można zastosować ML. Istnieją ciekawe problemy do rozwiązania, takie jak wykrywanie oszustw podczas testów lub esejów czy zarządzanie uprzedzeniami, zarówno zamierzonymi, jak i niezamierzonymi, w procesie oceniania.

### Przewidywanie zachowań uczniów

[Coursera](https://coursera.com), dostawca otwartych kursów online, prowadzi świetny blog technologiczny, na którym omawia wiele decyzji inżynieryjnych. W tym studium przypadku przedstawili linię regresji, aby zbadać, czy istnieje korelacja między niską oceną NPS (Net Promoter Score) a utrzymaniem lub rezygnacją z kursu.  
[Źródło](https://medium.com/coursera-engineering/controlled-regression-quantifying-the-impact-of-course-quality-on-learner-retention-31f956bd592a)

### Łagodzenie uprzedzeń

[Grammarly](https://grammarly.com), asystent pisania sprawdzający błędy ortograficzne i gramatyczne, wykorzystuje zaawansowane [systemy przetwarzania języka naturalnego](../../6-NLP/README.md) w swoich produktach. Na swoim blogu technologicznym opublikowali interesujące studium przypadku dotyczące tego, jak radzili sobie z uprzedzeniami związanymi z płcią w uczeniu maszynowym, o czym uczyłeś się w naszej [lekcji wprowadzającej na temat sprawiedliwości](../../1-Introduction/3-fairness/README.md).  
[Źródło](https://www.grammarly.com/blog/engineering/mitigating-gender-bias-in-autocorrect/)

## 👜 Handel detaliczny

Sektor handlu detalicznego może zdecydowanie skorzystać z zastosowania ML, począwszy od tworzenia lepszej ścieżki klienta, aż po optymalne zarządzanie zapasami.

### Personalizacja ścieżki klienta

W Wayfair, firmie sprzedającej artykuły domowe, takie jak meble, kluczowe jest pomaganie klientom w znajdowaniu produktów odpowiadających ich gustom i potrzebom. W tym artykule inżynierowie firmy opisują, jak wykorzystują ML i NLP do "prezentowania odpowiednich wyników dla klientów". Ich silnik Query Intent Engine wykorzystuje ekstrakcję jednostek, trenowanie klasyfikatorów, ekstrakcję opinii i oznaczanie sentymentu w recenzjach klientów. To klasyczny przykład zastosowania NLP w handlu online.  
[Źródło](https://www.aboutwayfair.com/tech-innovation/how-we-use-machine-learning-and-natural-language-processing-to-empower-search)

### Zarządzanie zapasami

Innowacyjne, elastyczne firmy, takie jak [StitchFix](https://stitchfix.com), usługa wysyłkowa odzieży, w dużym stopniu polegają na ML w zakresie rekomendacji i zarządzania zapasami. Ich zespoły stylizacji współpracują z zespołami merchandisingu: "jeden z naszych naukowców danych eksperymentował z algorytmem genetycznym i zastosował go do odzieży, aby przewidzieć, jakie ubranie, które jeszcze nie istnieje, odniesie sukces. Przedstawiliśmy to zespołowi merchandisingu, który teraz może korzystać z tego narzędzia."  
[Źródło](https://www.zdnet.com/article/how-stitch-fix-uses-machine-learning-to-master-the-science-of-styling/)

## 🏥 Opieka zdrowotna

Sektor opieki zdrowotnej może wykorzystać ML do optymalizacji zadań badawczych, a także problemów logistycznych, takich jak ponowne przyjęcia pacjentów czy zapobieganie rozprzestrzenianiu się chorób.

### Zarządzanie badaniami klinicznymi

Toksyczność w badaniach klinicznych to poważny problem dla producentów leków. Jak dużo toksyczności jest dopuszczalne? W tym badaniu analiza różnych metod badań klinicznych doprowadziła do opracowania nowego podejścia do przewidywania wyników badań klinicznych. W szczególności wykorzystano algorytm random forest do stworzenia [klasyfikatora](../../4-Classification/README.md), który potrafi odróżnić grupy leków.  
[Źródło](https://www.sciencedirect.com/science/article/pii/S2451945616302914)

### Zarządzanie ponownymi przyjęciami do szpitala

Opieka szpitalna jest kosztowna, zwłaszcza gdy pacjenci muszą być ponownie przyjmowani. W tym artykule omówiono firmę, która wykorzystuje ML do przewidywania potencjalnych ponownych przyjęć za pomocą algorytmów [klasteryzacji](../../5-Clustering/README.md). Te klastry pomagają analitykom "odkrywać grupy ponownych przyjęć, które mogą mieć wspólną przyczynę".  
[Źródło](https://healthmanagement.org/c/healthmanagement/issuearticle/hospital-readmissions-and-machine-learning)

### Zarządzanie chorobami

Niedawna pandemia uwydatniła, w jaki sposób uczenie maszynowe może pomóc w powstrzymywaniu rozprzestrzeniania się chorób. W tym artykule rozpoznasz zastosowanie ARIMA, krzywych logistycznych, regresji liniowej i SARIMA. "Praca ta jest próbą obliczenia tempa rozprzestrzeniania się wirusa, a tym samym przewidywania liczby zgonów, wyzdrowień i potwierdzonych przypadków, aby pomóc nam lepiej się przygotować i przetrwać."  
[Źródło](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7979218/)

## 🌲 Ekologia i zielone technologie

Przyroda i ekologia składają się z wielu wrażliwych systemów, w których interakcje między zwierzętami a naturą są kluczowe. Ważne jest, aby dokładnie mierzyć te systemy i odpowiednio reagować, jeśli coś się dzieje, na przykład pożar lasu lub spadek populacji zwierząt.

### Zarządzanie lasami

W poprzednich lekcjach nauczyłeś się o [uczeniu przez wzmacnianie](../../8-Reinforcement/README.md). Może ono być bardzo przydatne przy przewidywaniu wzorców w przyrodzie. W szczególności można je wykorzystać do śledzenia problemów ekologicznych, takich jak pożary lasów i rozprzestrzenianie się gatunków inwazyjnych. W Kanadzie grupa badaczy wykorzystała uczenie przez wzmacnianie do budowy modeli dynamiki pożarów lasów na podstawie zdjęć satelitarnych. Korzystając z innowacyjnego "procesu rozprzestrzeniania przestrzennego (SSP)", wyobrazili sobie pożar lasu jako "agenta w dowolnej komórce krajobrazu". "Zestaw działań, które ogień może podjąć z dowolnej lokalizacji w dowolnym momencie, obejmuje rozprzestrzenianie się na północ, południe, wschód, zachód lub brak rozprzestrzeniania się."  
[Źródło](https://www.frontiersin.org/articles/10.3389/fict.2018.00006/full)

### Monitorowanie ruchu zwierząt

Chociaż głębokie uczenie zrewolucjonizowało wizualne śledzenie ruchów zwierząt (możesz stworzyć własny [tracker niedźwiedzi polarnych](https://docs.microsoft.com/learn/modules/build-ml-model-with-azure-stream-analytics/?WT.mc_id=academic-77952-leestott) tutaj), klasyczne techniki ML wciąż mają swoje miejsce w tym zadaniu.

Czujniki do śledzenia ruchów zwierząt gospodarskich i IoT wykorzystują tego rodzaju przetwarzanie wizualne, ale bardziej podstawowe techniki ML są przydatne do wstępnego przetwarzania danych. Na przykład w tym artykule monitorowano i analizowano postawy owiec za pomocą różnych algorytmów klasyfikacyjnych. Możesz rozpoznać krzywą ROC na stronie 335.  
[Źródło](https://druckhaus-hofmann.de/gallery/31-wj-feb-2020.pdf)

### ⚡️ Zarządzanie energią

W naszych lekcjach dotyczących [prognozowania szeregów czasowych](../../7-TimeSeries/README.md) wprowadziliśmy koncepcję inteligentnych parkometrów, które generują dochody dla miasta na podstawie zrozumienia podaży i popytu. Ten artykuł szczegółowo omawia, w jaki sposób klasteryzacja, regresja i prognozowanie szeregów czasowych połączono, aby pomóc przewidywać przyszłe zużycie energii w Irlandii na podstawie inteligentnego pomiaru.  
[Źródło](https://www-cdn.knime.com/sites/default/files/inline-images/knime_bigdata_energy_timeseries_whitepaper.pdf)

## 💼 Ubezpieczenia

Sektor ubezpieczeń to kolejna dziedzina, która wykorzystuje ML do budowy i optymalizacji modeli finansowych i aktuarialnych.

### Zarządzanie zmiennością

MetLife, dostawca ubezpieczeń na życie, otwarcie dzieli się sposobami analizy i łagodzenia zmienności w swoich modelach finansowych. W tym artykule znajdziesz wizualizacje klasyfikacji binarnej i porządkowej. Odkryjesz także wizualizacje prognozowania.  
[Źródło](https://investments.metlife.com/content/dam/metlifecom/us/investments/insights/research-topics/macro-strategy/pdf/MetLifeInvestmentManagement_MachineLearnedRanking_070920.pdf)

## 🎨 Sztuka, kultura i literatura

W sztuce, na przykład w dziennikarstwie, istnieje wiele interesujących problemów. Wykrywanie fałszywych wiadomości to ogromne wyzwanie, ponieważ udowodniono, że wpływa na opinię publiczną, a nawet może obalać demokracje. Muzea również mogą korzystać z ML, od znajdowania powiązań między artefaktami po planowanie zasobów.

### Wykrywanie fałszywych wiadomości

Wykrywanie fałszywych wiadomości stało się grą w kotka i myszkę we współczesnych mediach. W tym artykule badacze sugerują, że system łączący kilka technik ML, które poznaliśmy, może być testowany, a najlepszy model wdrożony: "System ten opiera się na przetwarzaniu języka naturalnego w celu wydobycia cech z danych, a następnie te cechy są wykorzystywane do trenowania klasyfikatorów uczenia maszynowego, takich jak Naive Bayes, Support Vector Machine (SVM), Random Forest (RF), Stochastic Gradient Descent (SGD) i Logistic Regression (LR)."  
[Źródło](https://www.irjet.net/archives/V7/i6/IRJET-V7I6688.pdf)

Ten artykuł pokazuje, jak łączenie różnych dziedzin ML może przynieść interesujące rezultaty, które mogą pomóc w powstrzymaniu rozprzestrzeniania się fałszywych wiadomości i zapobieganiu realnym szkodom; w tym przypadku impulsem była dezinformacja na temat leczenia COVID, która wywołała przemoc tłumu.

### ML w muzeach

Muzea stoją na progu rewolucji AI, w której katalogowanie i cyfryzacja zbiorów oraz znajdowanie powiązań między artefaktami staje się łatwiejsze dzięki postępowi technologicznemu. Projekty takie jak [In Codice Ratio](https://www.sciencedirect.com/science/article/abs/pii/S0306457321001035#:~:text=1.,studies%20over%20large%20historical%20sources.) pomagają odkrywać tajemnice niedostępnych zbiorów, takich jak Archiwa Watykańskie. Jednak biznesowy aspekt muzeów również korzysta z modeli ML.

Na przykład Instytut Sztuki w Chicago stworzył modele przewidujące, czym interesują się odwiedzający i kiedy będą uczestniczyć w wystawach. Celem jest tworzenie spersonalizowanych i zoptymalizowanych doświadczeń dla każdego użytkownika podczas wizyty w muzeum. "W roku fiskalnym 2017 model przewidział frekwencję i przychody z biletów z dokładnością do 1 procenta, mówi Andrew Simnick, starszy wiceprezes w Instytucie Sztuki."  
[Źródło](https://www.chicagobusiness.com/article/20180518/ISSUE01/180519840/art-institute-of-chicago-uses-data-to-make-exhibit-choices)

## 🏷 Marketing

### Segmentacja klientów

Najskuteczniejsze strategie marketingowe kierują się do klientów w różny sposób, w zależności od różnych grup. W tym artykule omówiono zastosowanie algorytmów klasteryzacji w celu wsparcia zróżnicowanego marketingu. Zróżnicowany marketing pomaga firmom poprawić rozpoznawalność marki, dotrzeć do większej liczby klientów i zwiększyć zyski.  
[Źródło](https://ai.inqline.com/machine-learning-for-marketing-customer-segmentation/)

## 🚀 Wyzwanie

Zidentyfikuj inną dziedzinę, która korzysta z niektórych technik, których nauczyłeś się w tym kursie, i odkryj, jak wykorzystuje ML.
## [Quiz po wykładzie](https://ff-quizzes.netlify.app/en/ml/)

## Przegląd i samodzielna nauka

Zespół ds. nauki o danych w Wayfair ma kilka interesujących filmów o tym, jak wykorzystują ML w swojej firmie. Warto [zajrzeć](https://www.youtube.com/channel/UCe2PjkQXqOuwkW1gw6Ameuw/videos)!

## Zadanie

[Polowanie na ML](assignment.md)

---

**Zastrzeżenie**:  
Ten dokument został przetłumaczony za pomocą usługi tłumaczenia AI [Co-op Translator](https://github.com/Azure/co-op-translator). Chociaż dokładamy wszelkich starań, aby tłumaczenie było precyzyjne, prosimy pamiętać, że automatyczne tłumaczenia mogą zawierać błędy lub nieścisłości. Oryginalny dokument w jego rodzimym języku powinien być uznawany za wiarygodne źródło. W przypadku informacji o kluczowym znaczeniu zaleca się skorzystanie z profesjonalnego tłumaczenia przez człowieka. Nie ponosimy odpowiedzialności za jakiekolwiek nieporozumienia lub błędne interpretacje wynikające z użycia tego tłumaczenia.