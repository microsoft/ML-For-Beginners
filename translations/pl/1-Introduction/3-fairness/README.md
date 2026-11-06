# Budowanie rozwiązań Machine Learning z odpowiedzialną SI
 
![Podsumowanie odpowiedzialnej SI w Machine Learning w formie szkicownika](../../../../translated_images/pl/ml-fairness.ef296ebec6afc98a.webp)
> Szkicownik autorstwa [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Quiz przed wykładem](https://ff-quizzes.netlify.app/en/ml/)
 
## Wprowadzenie

W tym programie nauczania zaczniesz odkrywać, jak uczenie maszynowe może wpływać i wpływa na nasze codzienne życie. Już dziś systemy i modele biorą udział w codziennych zadaniach decyzyjnych, takich jak diagnozy medyczne, udzielanie pożyczek czy wykrywanie oszustw. Dlatego ważne jest, aby te modele działały poprawnie i dostarczały wiarygodnych wyników. Podobnie jak każda aplikacja programowa, systemy SI mogą nie spełniać oczekiwań lub mieć niepożądane skutki. Dlatego kluczowe jest zrozumienie i wyjaśnienie zachowania modelu SI. 

Wyobraź sobie, co może się zdarzyć, gdy dane, których używasz do budowy tych modeli, nie obejmują pewnych grup demograficznych, takich jak rasa, płeć, poglądy polityczne, religia lub gdy pewne demografie są nadmiernie reprezentowane. Co się stanie, gdy wynik modelu będzie interpretowany tak, aby faworyzować jakąś grupę demograficzną? Jakie są tego konsekwencje dla aplikacji? Co więcej, co się stanie, gdy model będzie mieć niekorzystny skutek i zaszkodzi ludziom? Kto ponosi odpowiedzialność za zachowanie systemu SI? To są pytania, które będziemy badać w tym programie nauczania. 

W tej lekcji: 

- Podniesiesz świadomość ważności sprawiedliwości w uczeniu maszynowym oraz szkód związanych z niesprawiedliwością.
- Zapoznasz się z praktyką badania odchyleń i nietypowych scenariuszy dla zapewnienia niezawodności i bezpieczeństwa.
- Zrozumiesz potrzebę wzmacniania wszystkich poprzez projektowanie inkluzywnych systemów.
- Poznasz, jak ważne jest chronienie prywatności i bezpieczeństwa danych oraz ludzi.
- Zobaczysz znaczenie podejścia typu "szklana skrzynka" do wyjaśniania zachowania modeli SI.
- Będziesz świadomy, jak odpowiedzialność jest niezbędna do budowy zaufania w systemach SI.

## Wymagania wstępne

Jako warunek wstępny proszę ukończ ścieżkę nauki "Zasady odpowiedzialnej SI" i obejrzyj poniższe wideo na ten temat:

Dowiedz się więcej o odpowiedzialnej SI, podążając za tą [Ścieżką Nauki](https://docs.microsoft.com/learn/modules/responsible-ai-principles/?WT.mc_id=academic-77952-leestott)

[![Podejście Microsoft do odpowiedzialnej SI](https://img.youtube.com/vi/dnC8-uUZXSc/0.jpg)](https://youtu.be/dnC8-uUZXSc "Podejście Microsoft do odpowiedzialnej SI")

> 🎥 Kliknij powyższy obraz, aby obejrzeć wideo: Podejście Microsoft do odpowiedzialnej SI

## Sprawiedliwość

Systemy SI powinny traktować wszystkich sprawiedliwie i unikać różnego traktowania podobnych grup ludzi. Na przykład gdy systemy SI udzielają wskazówek dotyczących leczenia medycznego, wniosków o pożyczkę czy zatrudnienia, powinny one dawać te same rekomendacje wszystkim z podobnymi objawami, sytuacją finansową lub kwalifikacjami zawodowymi. Każdy z nas ma wrodzone uprzedzenia, które wpływają na nasze decyzje i działania. Uprzedzenia te mogą być widoczne w danych używanych do trenowania systemów SI. Manipulacje takie mogą czasami pojawić się nieświadomie. Często trudno jest świadomie zauważyć, kiedy wprowadzamy uprzedzenie do danych. 

**„Niesprawiedliwość”** obejmuje negatywne skutki lub „szkody” dla określonej grupy ludzi, na przykład definiowanej według rasy, płci, wieku lub statusu niepełnosprawności. Główne szkody związane ze sprawiedliwością można sklasyfikować jako: 

- **Alokacja**, jeśli na przykład faworyzowana jest płeć lub pochodzenie etniczne ponad inną.
- **Jakość usługi**. Jeśli dane są trenowane na jednym specyficznym scenariuszu, a rzeczywistość jest dużo bardziej złożona, skutkuje to słabą jakością usługi. Na przykład, dozownik mydła, który nie potrafił wykrywać osób o ciemnej karnacji. [Odnośnik](https://gizmodo.com/why-cant-this-soap-dispenser-identify-dark-skin-1797931773)
- **Znieważanie**. Niesprawiedliwe krytykowanie lub etykietowanie czegoś lub kogoś. Na przykład technologia etykietowania zdjęć niesłusznie oznaczyła zdjęcia ciemnoskórych osób jako goryle.
- **Nadmierna lub niedostateczna reprezentacja**. Chodzi o to, że dana grupa jest niewidoczna w pewnym zawodzie, a każda usługa lub funkcja, która utrwala ten stan, przyczynia się do szkody.
- **Stereotypowanie**. Kojarzenie danej grupy z przypisanymi z góry cechami. Na przykład system tłumaczenia języka między angielskim a tureckim może mieć nieścisłości związane z stereotypowymi skojarzeniami słów z płcią.

![tłumaczenie na turecki](../../../../translated_images/pl/gender-bias-translate-en-tr.f185fd8822c2d437.webp)
> tłumaczenie na turecki

![tłumaczenie z powrotem na angielski](../../../../translated_images/pl/gender-bias-translate-tr-en.4eee7e3cecb8c70e.webp)
> tłumaczenie z powrotem na angielski

Projektując i testując systemy SI, musimy zapewnić, że SI jest sprawiedliwa i nie jest zaprogramowana do podejmowania decyzji stronniczych lub dyskryminujących, czego również zakazuje się ludziom. Gwarantowanie sprawiedliwości w SI i uczeniu maszynowym pozostaje złożonym wyzwaniem społeczno-technicznym. 

### Niezawodność i bezpieczeństwo

Aby budować zaufanie, systemy SI muszą być niezawodne, bezpieczne i spójne w normalnych i nieoczekiwanych warunkach. Ważne jest, aby wiedzieć, jak systemy SI zachowują się w różnych sytuacjach, szczególnie gdy występują odchylenia. Budując rozwiązania SI, należy w dużym stopniu skupić się na tym, jak radzić sobie z szerokim wachlarzem okoliczności, które systemy mogą napotkać. Na przykład, samochód autonomiczny musi stawiać bezpieczeństwo ludzi na pierwszym miejscu. W rezultacie SI napędzająca samochód musi uwzględniać wszystkie możliwe scenariusze, takie jak noc, burze czy zamiecie śnieżne, dzieci przebiegające przez ulicę, zwierzęta domowe, roboty drogowe itp. To, jak dobrze system SI radzi sobie w szerokim zakresie warunków w sposób niezawodny i bezpieczny, odzwierciedla poziom przewidywań, jakie specjalista ds. danych lub deweloper SI uwzględnił podczas projektowania lub testowania systemu.  

> [🎥 Kliknij tutaj, aby obejrzeć wideo: Niezawodność i bezpieczeństwo w SI](https://www.microsoft.com/videoplayer/embed/RE4vvIl)

### Inkluzywność

Systemy SI powinny być projektowane tak, by angażować i wzmacniać wszystkich. Projektując i implementując systemy SI, specjaliści ds. danych i deweloperzy SI identyfikują i usuwają potencjalne bariery, które mogłyby nieumyślnie wykluczać ludzi. Na przykład na świecie jest miliard osób z niepełnosprawnościami. Dzięki postępowi SI mają oni łatwiejszy dostęp do szerokiego zakresu informacji i możliwości w codziennym życiu. Usuwanie barier stwarza okazje do innowacji i opracowywania produktów SI z lepszymi doświadczeniami, które przynoszą korzyści wszystkim. 

> [🎥 Kliknij tutaj, aby obejrzeć wideo: Inkluzywność w SI](https://www.microsoft.com/videoplayer/embed/RE4vl9v)

### Bezpieczeństwo i prywatność 

Systemy SI powinny być bezpieczne i szanować prywatność ludzi. Ludzie mają mniejsze zaufanie do systemów, które narażają ich prywatność, informacje lub życie na ryzyko. Podczas trenowania modeli uczenia maszynowego opieramy się na danych, aby uzyskać najlepsze wyniki. W tym procesie należy uwzględnić pochodzenie danych i ich integralność. Na przykład, czy dane zostały przesłane przez użytkowników czy są publicznie dostępne? Następnie, pracując z danymi, kluczowe jest opracowanie systemów SI, które chronią poufne informacje i są odporne na ataki. W miarę jak SI staje się bardziej powszechna, ochrona prywatności i bezpieczeństwo ważnych informacji osobistych i biznesowych staje się coraz bardziej krytyczna i złożona. Problemy prywatności i bezpieczeństwa danych wymagają szczególnej uwagi w SI, ponieważ dostęp do danych jest niezbędny, aby systemy SI mogły podejmować dokładne i świadome przewidywania oraz decyzje dotyczące ludzi. 

> [🎥 Kliknij tutaj, aby obejrzeć wideo: Bezpieczeństwo w SI](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Jako branża dokonaliśmy znaczących postępów w zakresie prywatności i bezpieczeństwa, znacząco wspieranych przez regulacje takie jak GDPR (Ogólne Rozporządzenie o Ochronie Danych). 
- Jednak w systemach SI musimy uznać napięcie między potrzebą większej ilości danych osobowych, aby uczynić systemy bardziej spersonalizowanymi i skutecznymi – a prywatnością. 
- Podobnie jak przy narodzinach komputerów połączonych z internetem, obserwujemy także znaczny wzrost liczby problemów związanych z bezpieczeństwem, dotyczących SI. 
- Jednocześnie widzimy, że SI jest wykorzystywana do poprawy bezpieczeństwa. Na przykład większość nowoczesnych skanerów antywirusowych jest dziś napędzana heurystyką SI. 
- Musimy zapewnić, że nasze procesy Data Science harmonijnie łączą się z najnowszymi praktykami w zakresie prywatności i bezpieczeństwa. 


### Przejrzystość
Systemy SI powinny być zrozumiałe. Kluczowym elementem przejrzystości jest wyjaśnianie zachowania systemów SI i ich komponentów. Poprawa zrozumienia systemów SI wymaga, aby interesariusze pojmowali, jak i dlaczego systemy działają, by mogli zidentyfikować potencjalne problemy z wydajnością, obawy dotyczące bezpieczeństwa i prywatności, uprzedzenia, praktyki wykluczające lub niezamierzone skutki. Wierzymy także, że osoby korzystające z systemów SI powinny być uczciwe i otwarte co do momentu, powodów i sposobów ich wdrażania. A także co do ograniczeń używanych systemów. Na przykład, jeśli bank używa systemu SI do wspierania decyzji o udzielaniu kredytów konsumenckich, ważne jest, aby analizować wyniki i rozumieć, które dane wpływają na rekomendacje systemu. Rządy zaczynają regulować SI w różnych branżach, więc specjaliści ds. danych i organizacje muszą wyjaśniać, czy system SI spełnia wymogi regulacyjne, zwłaszcza gdy pojawia się niepożądany skutek. 

> [🎥 Kliknij tutaj, aby obejrzeć wideo: Przejrzystość w SI](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Ponieważ systemy SI są tak złożone, trudno jest zrozumieć, jak działają i interpretować wyniki. 
- Ten brak zrozumienia wpływa na sposób zarządzania, wdrażania i dokumentowania tych systemów. 
- Co ważniejsze, brak zrozumienia wpływa na decyzje podejmowane na podstawie wyników dostarczanych przez te systemy. 

### Odpowiedzialność 
 
Osoby projektujące i wdrażające systemy SI muszą być odpowiedzialne za funkcjonowanie swoich systemów. Potrzeba odpowiedzialności jest szczególnie ważna w przypadku technologii wrażliwych, takich jak rozpoznawanie twarzy. Ostatnio rośnie zapotrzebowanie na technologię rozpoznawania twarzy, zwłaszcza wśród organów ścigania, które widzą potencjał tej technologii w takich zastosowaniach jak odnajdywanie zaginionych dzieci. Jednakże te technologie mogą być używane przez rządy do naruszania podstawowych wolności obywatelskich, na przykład umożliwiając ciągłe monitorowanie wybranych osób. Dlatego specjaliści ds. danych i organizacje muszą być odpowiedzialni za to, jaki wpływ ma ich system SI na jednostki lub społeczeństwo.

[![Wiodący badacz SI ostrzega przed masową inwigilacją przez rozpoznawanie twarzy](../../../../translated_images/pl/accountability.41d8c0f4b85b6231.webp)](https://www.youtube.com/watch?v=Wldt8P5V6D0 "Podejście Microsoft do odpowiedzialnej SI")

> 🎥 Kliknij powyższy obraz, aby obejrzeć wideo: Ostrzeżenia przed masową inwigilacją przez rozpoznawanie twarzy 

Ostatecznie jedno z największych pytań dla naszego pokolenia, jako pierwszego, które wprowadza SI do społeczeństwa, brzmi: jak zapewnić, że komputery pozostaną odpowiedzialne przed ludźmi i jak zagwarantować, że osoby projektujące komputery pozostaną odpowiedzialne przed wszystkimi innymi.

## Ocena wpływu 

Przed trenowaniem modelu uczenia maszynowego ważne jest przeprowadzenie oceny wpływu, aby zrozumieć cel systemu SI; jakie jest jego zamierzone użycie; gdzie będzie wdrażany; oraz kto będzie z nim wchodził w interakcje. Pomaga to recenzentom lub testerom ocenić, jakie czynniki brać pod uwagę przy identyfikacji potencjalnych ryzyk i spodziewanych konsekwencji.

Poniżej przedstawiono obszary skupienia podczas przeprowadzania oceny wpływu:

* **Negatywny wpływ na jednostki**. Świadomość wszelkich ograniczeń, wymagań, nieautoryzowanego użycia lub znanych ograniczeń wpływających na działanie systemu jest kluczowa, aby zapewnić, że system nie będzie używany w sposób mogący szkodzić ludziom.
* **Wymagania dotyczące danych**. Zrozumienie, jak i gdzie system będzie wykorzystywał dane, pozwala recenzentom przeanalizować wymagania dotyczące danych, które trzeba mieć na uwadze (np. regulacje GDPR lub HIPAA). Dodatkowo należy ocenić, czy źródło i ilość danych są wystarczające do trenowania.
* **Podsumowanie wpływu**. Zebranie listy potencjalnych szkód, które mogą wyniknąć z używania systemu. W trakcie cyklu życia ML należy sprawdzać, czy zidentyfikowane problemy są łagodzone lub rozwiązywane.
* **Osiągalne cele** dla każdej z sześciu podstawowych zasad. Ocena, czy cele wynikające z każdej zasady są spełnione oraz czy istnieją jakiekolwiek luki.


## Debugowanie z odpowiedzialną SI  

Podobnie jak debugowanie aplikacji programowej, debugowanie systemu SI jest koniecznym procesem identyfikowania i rozwiązywania problemów w systemie. Wiele czynników może powodować, że model nie działa zgodnie z oczekiwaniami lub w sposób odpowiedzialny. Większość tradycyjnych wskaźników wydajności modeli to ilościowe agregaty wydajności, które nie wystarczają do analizy, jak model narusza zasady odpowiedzialnej SI. Ponadto model uczenia maszynowego jest czarną skrzynką, co utrudnia zrozumienie, co powoduje jego wyniki lub dostarczenie wyjaśnienia błędu. W dalszej części tego kursu nauczymy się, jak korzystać z panelu odpowiedzialnej SI, aby pomóc w debugowaniu systemów SI. Panel ten dostarcza kompleksowe narzędzie dla specjalistów ds. danych i deweloperów SI do wykonywania:

* **Analizy błędów**. Aby zidentyfikować rozkład błędów modelu, które mogą wpływać na sprawiedliwość lub niezawodność systemu.
* **Przeglądu modelu**. Aby odkryć, gdzie występują nierówności w wydajności modelu w różnych grupach danych.
* **Analizy danych**. Aby zrozumieć rozkład danych i zidentyfikować potencjalne uprzedzenia w danych, które mogą prowadzić do problemów sprawiedliwości, inkluzywności i niezawodności.
* **Interpretowalności modelu**. Aby zrozumieć, co wpływa na przewidywania modelu. Pomaga to w wyjaśnianiu zachowania modelu, co jest ważne dla przejrzystości i odpowiedzialności.


## 🚀 Wyzwanie 
 
Aby zapobiec wprowadzaniu szkód w pierwszej kolejności, powinniśmy: 

- mieć różnorodność pochodzenia i perspektyw wśród osób pracujących nad systemami 
- inwestować w zbiór danych odzwierciedniający różnorodność naszego społeczeństwa 
- rozwijać lepsze metody w całym cyklu życia uczenia maszynowego do wykrywania i naprawiania nieodpowiedzialnej SI, gdy się pojawia 

Pomyśl o rzeczywistych scenariuszach, gdzie brak zaufania do modelu jest widoczny w budowie i używaniu modelu. Co jeszcze powinniśmy rozważyć? 

## [Quiz po wykładzie](https://ff-quizzes.netlify.app/en/ml/)

## Podsumowanie i samodzielna nauka 
 
W tej lekcji nauczyłeś się podstaw pojęć sprawiedliwości i niesprawiedliwości w uczeniu maszynowym.  
 
Obejrzyj ten warsztat, aby zgłębić tematy: 

- W poszukiwaniu odpowiedzialnej SI: Wdrażanie zasad w praktyce, autorstwa Besmiry Nushi, Mehrnoosh Sameki i Amita Sharmy

[![Responsible AI Toolbox: Otwarte ramy do budowania odpowiedzialnej SI](https://img.youtube.com/vi/tGgJCrA-MZU/0.jpg)](https://www.youtube.com/watch?v=tGgJCrA-MZU "RAI Toolbox: Otwarte ramy do budowania odpowiedzialnej SI")

> 🎥 Kliknij powyższy obraz, aby obejrzeć wideo: RAI Toolbox: Otwarte ramy do budowania odpowiedzialnej SI autorstwa Besmiry Nushi, Mehrnoosh Sameki i Amit Sharma

Przeczytaj również:

- Centrum zasobów RAI Microsoftu: [Responsible AI Resources – Microsoft AI](https://www.microsoft.com/ai/responsible-ai-resources?activetab=pivot1%3aprimaryr4)

- Grupa badawcza FATE Microsoftu: [FATE: Fairness, Accountability, Transparency, and Ethics in AI - Microsoft Research](https://www.microsoft.com/research/theme/fate/)

RAI Toolbox:

- [Repozytorium GitHub odpowiedzialnego AI](https://github.com/microsoft/responsible-ai-toolbox)

Przeczytaj o narzędziach Azure Machine Learning zapewniających sprawiedliwość:

- [Azure Machine Learning](https://docs.microsoft.com/azure/machine-learning/concept-fairness-ml?WT.mc_id=academic-77952-leestott)

## Zadanie

[Poznaj RAI Toolbox](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Zastrzeżenie**:
Niniejszy dokument został przetłumaczony za pomocą usługi tłumaczenia AI [Co-op Translator](https://github.com/Azure/co-op-translator). Choć dążymy do dokładności, prosimy pamiętać, że automatyczne tłumaczenia mogą zawierać błędy lub niedokładności. Oryginalny dokument w jego języku źródłowym należy uznawać za autorytatywne źródło. W przypadku informacji krytycznych zalecane jest skorzystanie z profesjonalnego tłumaczenia wykonanego przez człowieka. Nie ponosimy odpowiedzialności za jakiekolwiek nieporozumienia lub błędne interpretacje wynikające z użycia tego tłumaczenia.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->