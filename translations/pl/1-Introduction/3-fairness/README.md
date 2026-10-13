# Tworzenie rozwiązań uczenia maszynowego z odpowiedzialną AI
 
![Podsumowanie odpowiedzialnej AI w uczeniu maszynowym w formie szkicu](../../../../translated_images/pl/ml-fairness.ef296ebec6afc98a.webp)
> Szkic autorstwa [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Quiz przed wykładem](https://ff-quizzes.netlify.app/en/ml/)
 
## Wprowadzenie

W tym programie nauczania zaczniesz odkrywać, jak uczenie maszynowe może wpływać i wpływa na nasze codzienne życie. Już teraz systemy i modele biorą udział w codziennych zadaniach decyzyjnych, takich jak diagnozy medyczne, zatwierdzanie pożyczek czy wykrywanie oszustw. Dlatego ważne jest, aby te modele działały dobrze i dostarczały wyniki godne zaufania. Podobnie jak każda aplikacja oprogramowania, systemy AI mogą nie spełnić oczekiwań lub mieć niepożądany rezultat. Dlatego tak istotne jest, aby rozumieć i móc wyjaśnić zachowanie modelu AI.

Wyobraź sobie, co może się stać, gdy dane używane do budowy tych modeli nie zawierają pewnych grup demograficznych, takich jak rasa, płeć, poglądy polityczne, religia, lub gdy takie grupy są nierównomiernie reprezentowane. Co się stanie, gdy wyniki modelu będą interpretowane tak, aby faworyzować niektóre grupy demograficzne? Jakie będą tego konsekwencje dla aplikacji? Dodatkowo, co się stanie, gdy model da negatywny wynik i zaszkodzi ludziom? Kto jest odpowiedzialny za zachowanie systemów AI? To są pytania, które omówimy w tym programie nauczania.

W tej lekcji:

- Zwiększysz świadomość na temat znaczenia sprawiedliwości w uczeniu maszynowym oraz szkodliwych skutków związanych z niesprawiedliwością.
- Zapoznasz się z praktyką badania wartości odstających i nietypowych scenariuszy, aby zapewnić niezawodność i bezpieczeństwo.
- Zdobędziesz zrozumienie potrzeby upodmiotowienia wszystkich poprzez projektowanie systemów inkluzywnych.
- Zbadasz, jak ważne jest ochronienie prywatności i bezpieczeństwa danych oraz osób.
- Zrozumiesz znaczenie podejścia typu „szklana skrzynka” do wyjaśniania zachowania modeli AI.
- Będziesz świadomy, jak istotna jest odpowiedzialność dla budowania zaufania do systemów AI.

## Wymagania wstępne

Jako wymaganie wstępne, proszę ukończ ścieżkę nauki „Zasady odpowiedzialnej AI” oraz obejrzyj poniższy film na ten temat:

Dowiedz się więcej o odpowiedzialnej AI, śledząc tę [Ścieżkę nauki](https://docs.microsoft.com/learn/modules/responsible-ai-principles/?WT.mc_id=academic-77952-leestott)

[![Podejście Microsoft do odpowiedzialnej AI](https://img.youtube.com/vi/dnC8-uUZXSc/0.jpg)](https://youtu.be/dnC8-uUZXSc "Podejście Microsoft do odpowiedzialnej AI")

> 🎥 Kliknij obraz powyżej, aby obejrzeć film: Podejście Microsoft do odpowiedzialnej AI

## Sprawiedliwość

Systemy AI powinny traktować wszystkich sprawiedliwie i unikać różnicowania podobnych grup osób w różny sposób. Na przykład, gdy systemy AI udzielają wskazówek dotyczących leczenia medycznego, aplikacji o pożyczkę czy zatrudnienia, powinny udzielać tych samych rekomendacji wszystkim z podobnymi objawami, sytuacją finansową lub kwalifikacjami zawodowymi. Każdy z nas jako człowiek nosi w sobie dziedziczone uprzedzenia, które wpływają na nasze decyzje i działania. Uprzedzenia te mogą być widoczne w danych, których używamy do trenowania systemów AI. Takie manipulacje mogą czasem zachodzić nieświadomie. Często trudno jest świadomie rozpoznać, kiedy wprowadzasz stronniczość do danych.

**„Niesprawiedliwość”** obejmuje negatywne skutki, czyli „szkody”, które dotyczą pewnej grupy ludzi, definiowanej np. przez rasę, płeć, wiek czy status niepełnosprawności. Główne szkody związane ze sprawiedliwością można sklasyfikować jako:

- **Alokacja**, jeśli np. jedna płeć lub etniczność jest faworyzowana nad inną.
- **Jakość usług**. Jeśli dane są szkolone dla jednego konkretnego scenariusza, ale rzeczywistość jest znacznie bardziej złożona, prowadzi to do słabej jakości usług. Na przykład dozownik mydła, który nie jest w stanie wykryć osób o ciemnej skórze. [Odnośnik](https://gizmodo.com/why-cant-this-soap-dispenser-identify-dark-skin-1797931773)
- **Oczernianie**. Niesprawiedliwe krytykowanie i etykietowanie czegoś lub kogoś. Na przykład technologia rozpoznawania obrazów niesłusznie oznaczyła zdjęcia osób o ciemnej skórze jako goryle.
- **Nadmierna lub niedostateczna reprezentacja**. Idea, że dana grupa nie występuje w pewnym zawodzie, a każda usługa lub funkcja, która to utrzymuje, przyczynia się do szkód.
- **Stereotypowanie**. Kojarzenie danej grupy z przypisanymi atrybutami. Na przykład system tłumaczeń między angielskim a tureckim może mieć niedokładności ze względu na słowa o stereotypowych powiązaniach z płcią.

![tłumaczenie na turecki](../../../../translated_images/pl/gender-bias-translate-en-tr.f185fd8822c2d437.webp)
> tłumaczenie na turecki

![tłumaczenie z powrotem na angielski](../../../../translated_images/pl/gender-bias-translate-tr-en.4eee7e3cecb8c70e.webp)
> tłumaczenie z powrotem na angielski

Projektując i testując systemy AI, musimy zapewnić, że AI jest sprawiedliwa i nie jest zaprogramowana do podejmowania stronniczych lub dyskryminujących decyzji, których również zakazuje się ludziom. Zapewnienie sprawiedliwości w AI i uczeniu maszynowym pozostaje skomplikowanym wyzwaniem socjotechnicznym.

### Niezawodność i bezpieczeństwo

Aby budować zaufanie, systemy AI muszą być niezawodne, bezpieczne i spójne w normalnych i nieoczekiwanych warunkach. Ważne jest, by znać zachowanie systemów AI w różnych sytuacjach, zwłaszcza gdy są wartościami odstającymi. Budując rozwiązania AI, należy poświęcić dużo uwagi na to, jak poradzić sobie z różnorodnymi okolicznościami, które system AI może napotkać. Na przykład samochód autonomiczny musi stawiać bezpieczeństwo ludzi na pierwszym miejscu. W efekcie AI napędzająca ten samochód musi rozważyć wszystkie możliwe scenariusze, takie jak noc, burze, śnieżyce, dzieci przebiegające przez ulicę, zwierzęta domowe, roboty drogowe itd. Jak dobrze system AI radzi sobie z szerokim zakresem warunków, odzwierciedla poziom przewidywań, które naukowiec danych lub deweloper AI uwzględnił podczas projektowania lub testowania systemu.

> [🎥 Kliknij tutaj, aby obejrzeć film: ](https://www.microsoft.com/videoplayer/embed/RE4vvIl)

### Inkluzywność

Systemy AI powinny być projektowane tak, aby angażować i upodmiotawiać wszystkich. Projektując i wdrażając systemy AI, naukowcy danych i deweloperzy AI identyfikują i usuwają potencjalne bariery w systemie, które mogą nieświadomie wykluczać ludzi. Na przykład na świecie jest 1 miliard osób z niepełnosprawnościami. Dzięki rozwojowi AI mogą one łatwiej mieć dostęp do szerokiej gamy informacji i możliwości w codziennym życiu. Usuwanie barier tworzy możliwości innowacji i opracowania produktów AI z lepszym doświadczeniem, które przynosi korzyści wszystkim.

> [🎥 Kliknij tutaj, aby obejrzeć film: inkluzywność w AI](https://www.microsoft.com/videoplayer/embed/RE4vl9v)

### Bezpieczeństwo i prywatność

Systemy AI powinny być bezpieczne i szanować prywatność ludzi. Ludzie mniej ufają systemom, które narażają ich prywatność, informacje lub życie na ryzyko. Trenując modele uczenia maszynowego, opieramy się na danych, aby uzyskać najlepsze wyniki. W związku z tym należy rozważyć pochodzenie danych i ich integralność. Na przykład, czy dane zostały przesłane przez użytkownika, czy są publicznie dostępne? Następnie, podczas pracy z danymi, kluczowe jest opracowanie systemów AI, które chronią informacje poufne i są odporne na ataki. W miarę jak AI staje się bardziej powszechna, ochrona prywatności i zabezpieczenie ważnych informacji osobistych i biznesowych staje się coraz ważniejsze i bardziej skomplikowane. Problemy prywatności i bezpieczeństwa danych wymagają szczególnej uwagi w AI, ponieważ dostęp do danych jest niezbędny, aby systemy AI mogły dokonywać dokładnych i świadomych przewidywań oraz decyzji dotyczących ludzi.

> [🎥 Kliknij tutaj, aby obejrzeć film: bezpieczeństwo w AI](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Jako branża dokonaliśmy znaczących postępów w zakresie prywatności i bezpieczeństwa, w dużej mierze dzięki regulacjom takim jak RODO (Ogólne rozporządzenie o ochronie danych).
- Jednak w systemach AI musimy uznać napięcie między potrzebą większej ilości danych osobowych, aby uczynić systemy bardziej osobistymi i skutecznymi – a prywatnością.
- Podobnie jak przy narodzinach połączonych komputerów z internetem, obserwujemy również gwałtowny wzrost problemów związanych z bezpieczeństwem w AI.
- Jednocześnie widzimy, że AI jest wykorzystywana do poprawy bezpieczeństwa. Na przykład większość nowoczesnych skanerów antywirusowych opiera się obecnie na heurystykach AI.
- Musimy zapewnić, że nasze procesy Data Science harmonijnie współgrają z najnowszymi praktykami w zakresie prywatności i bezpieczeństwa.


### Przejrzystość
Systemy AI powinny być zrozumiałe. Kluczową częścią przejrzystości jest wyjaśnianie zachowania systemów AI i ich komponentów. Poprawa zrozumienia systemów AI wymaga, aby interesariusze rozumieli, jak i dlaczego one funkcjonują, tak aby mogli zidentyfikować potencjalne problemy z wydajnością, bezpieczeństwem i prywatnością, uprzedzeniami, praktykami wykluczającymi lub niezamierzonymi skutkami. Wierzymy również, że osoby korzystające z systemów AI powinny być uczciwe i otwarte, kiedy, dlaczego i jak decydują się je wdrożyć, a także znać ograniczenia używanych systemów. Na przykład, jeśli bank wykorzystuje system AI do wspierania decyzji kredytowych, ważne jest, aby badać wyniki i rozumieć, które dane wpływają na rekomendacje systemu. Rządy zaczynają regulować AI w różnych branżach, dlatego naukowcy danych i organizacje muszą wyjaśnić, czy system AI spełnia wymagania regulacyjne, szczególnie gdy wystąpi niepożądany wynik.

> [🎥 Kliknij tutaj, aby obejrzeć film: przejrzystość w AI](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Ponieważ systemy AI są tak złożone, trudno jest zrozumieć, jak działają i interpretować wyniki.
- Brak zrozumienia wpływa na sposób, w jaki systemy te są zarządzane, wdrażane i dokumentowane.
- Co ważniejsze, brak zrozumienia wpływa na decyzje podejmowane na podstawie wyników produkowanych przez te systemy.

### Odpowiedzialność
 
Osoby projektujące i wdrażające systemy AI muszą być odpowiedzialne za sposób działania swoich systemów. Potrzeba odpowiedzialności jest szczególnie ważna w przypadku technologii wrażliwych, takich jak rozpoznawanie twarzy. Ostatnio rośnie zapotrzebowanie na technologię rozpoznawania twarzy, zwłaszcza ze strony organów ścigania, które widzą potencjał tej technologii w zastosowaniach, takich jak odnajdywanie zaginionych dzieci. Jednakże technologie te mogą być potencjalnie używane przez rządy do zagrożenia podstawowym wolnościom obywateli, na przykład umożliwiając ciągłe monitorowanie określonych osób. Dlatego też naukowcy danych i organizacje muszą ponosić odpowiedzialność za wpływ ich systemów AI na jednostki lub społeczeństwo.

[![Wiodący badacz AI ostrzega przed masową inwigilacją przez rozpoznawanie twarzy](../../../../translated_images/pl/accountability.41d8c0f4b85b6231.webp)](https://www.youtube.com/watch?v=Wldt8P5V6D0 "Podejście Microsoft do odpowiedzialnej AI")

> 🎥 Kliknij obraz powyżej, aby obejrzeć film: Ostrzeżenia przed masową inwigilacją przez rozpoznawanie twarzy

Ostatecznie, jedno z największych pytań dla naszego pokolenia, jako pierwszego, które wprowadza AI do społeczeństwa, brzmi: jak zapewnić, aby komputery pozostały odpowiedzialne wobec ludzi oraz jak sprawić, by osoby projektujące komputery pozostały odpowiedzialne wobec wszystkich innych.

## Ocena wpływu

Przed trenowaniem modelu uczenia maszynowego ważne jest przeprowadzenie oceny wpływu, aby zrozumieć cel systemu AI; do jakiego użytku jest przeznaczony; gdzie będzie wdrażany oraz kto będzie wchodził w interakcję z systemem. To jest pomocne dla recenzenta(ów) lub testerów oceniających system, którzy muszą wiedzieć, jakie czynniki wziąć pod uwagę przy identyfikowaniu potencjalnych ryzyk i przewidywanych konsekwencji.

Poniżej przedstawiono obszary uwagi przy przeprowadzaniu oceny wpływu:

* **Negatywny wpływ na jednostki**. Świadomość wszelkich ograniczeń lub wymagań, nieobsługiwanych zastosowań lub znanych ograniczeń utrudniających działanie systemu jest niezbędna, aby zapobiec wykorzystywaniu systemu w sposób mogący szkodzić jednostkom.
* **Wymagania dotyczące danych**. Zrozumienie, jak i gdzie system będzie używał danych, pozwala recenzentom zbadać wszelkie wymagania dotyczące danych, na które należy uważać (np. przepisy RODO lub HIPAA). Dodatkowo należy sprawdzić, czy źródło lub ilość danych jest wystarczająca do treningu.
* **Podsumowanie wpływu**. Sporządź listę potencjalnych szkód, które mogą wyniknąć z używania systemu. W trakcie cyklu życia ML sprawdź, czy kwestie te zostały zminimalizowane lub rozwiązane.
* **Celowe cele** dla każdego z sześciu podstawowych zasad. Oceń, czy cele z każdej zasady zostały spełnione i czy istnieją jakiekolwiek luki.


## Debugowanie z odpowiedzialną AI  

Podobnie jak debugowanie aplikacji oprogramowania, debugowanie systemu AI to niezbędny proces identyfikowania i rozwiązywania problemów w systemie. Istnieje wiele czynników, które mogą wpływać na to, że model nie działa zgodnie z oczekiwaniami lub odpowiedzialnie. Większość tradycyjnych metryk wydajności modelu to ilościowe agregaty wydajności modelu, które nie wystarczają do analizy, jak model narusza zasady odpowiedzialnej AI. Ponadto model uczenia maszynowego jest czarną skrzynką, co utrudnia zrozumienie, co napędza jego wynik lub wyjaśnienie, gdy popełnia błąd. W dalszej części tego kursu nauczymy się, jak korzystać z pulpitu odpowiedzialnej AI, aby pomóc w debugowaniu systemów AI. Pulpit zapewnia całościowe narzędzie dla naukowców danych i deweloperów AI do wykonywania:

* **Analizy błędów**. Aby zidentyfikować rozkład błędów modelu, które mogą wpłynąć na sprawiedliwość lub niezawodność systemu.
* **Przeglądu modelu**. Aby odkryć miejsca, gdzie występują różnice w wydajności modelu w różnych kohortach danych.
* **Analizy danych**. Aby zrozumieć rozkład danych i zidentyfikować potencjalne uprzedzenia w danych, które mogą prowadzić do problemów ze sprawiedliwością, inkluzywnością i niezawodnością.
* **Interpretowalności modelu**. Aby zrozumieć, co wpływa lub wpływa na prognozy modelu. Pomaga to wyjaśnić zachowanie modelu, co jest ważne dla przejrzystości i odpowiedzialności.


## 🚀 Wyzwanie
 
Aby zapobiec pojawieniu się szkód, powinniśmy:

- zatrudniać osoby o różnorodnym pochodzeniu i perspektywach wśród zespołów pracujących nad systemami
- inwestować w zestawy danych odzwierciedlające różnorodność naszego społeczeństwa
- rozwijać lepsze metody na wszystkich etapach cyklu życia uczenia maszynowego, aby wykrywać i naprawiać problemy z odpowiedzialną AI, gdy się pojawiają

Pomyśl o scenariuszach z życia, gdzie brak wiarygodności modelu jest widoczny podczas tworzenia i używania modelu. Co jeszcze powinniśmy rozważyć?

## [Quiz po wykładzie](https://ff-quizzes.netlify.app/en/ml/)

## Podsumowanie i samodzielna nauka
 

W tej lekcji nauczyłeś się podstaw pojęć sprawiedliwości i niesprawiedliwości w uczeniu maszynowym.  
 
Obejrzyj ten warsztat, aby zagłębić się w te tematy: 

- W dążeniu do odpowiedzialnej sztucznej inteligencji: Wprowadzanie zasad w praktykę przez Besmirę Nushi, Mehrnoosh Sameki i Amit Sharmę

[![Responsible AI Toolbox: An open-source framework for building responsible AI](https://img.youtube.com/vi/tGgJCrA-MZU/0.jpg)](https://www.youtube.com/watch?v=tGgJCrA-MZU "RAI Toolbox: An open-source framework for building responsible AI")

> 🎥 Kliknij obraz powyżej, aby obejrzeć wideo: RAI Toolbox: Open-source’owy framework do tworzenia odpowiedzialnej SI przez Besmirę Nushi, Mehrnoosh Sameki i Amit Sharmę

Przeczytaj również: 

- Centrum zasobów Microsoft ds. RAI: [Responsible AI Resources – Microsoft AI](https://www.microsoft.com/ai/responsible-ai-resources?activetab=pivot1%3aprimaryr4) 

- Grupa badawcza Microsoft FATE: [FATE: Fairness, Accountability, Transparency, and Ethics in AI - Microsoft Research](https://www.microsoft.com/research/theme/fate/) 

RAI Toolbox: 

- [Repozytorium Responsible AI Toolbox na GitHub](https://github.com/microsoft/responsible-ai-toolbox)

Przeczytaj o narzędziach Azure Machine Learning zapewniających sprawiedliwość:

- [Azure Machine Learning](https://docs.microsoft.com/azure/machine-learning/concept-fairness-ml?WT.mc_id=academic-77952-leestott) 

## Zadanie

[Poznaj RAI Toolbox](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Zastrzeżenie**:
Niniejszy dokument został przetłumaczony za pomocą usługi tłumaczenia AI [Co-op Translator](https://github.com/Azure/co-op-translator). Choć dążymy do dokładności, prosimy pamiętać, że automatyczne tłumaczenia mogą zawierać błędy lub niedokładności. Oryginalny dokument w jego języku źródłowym należy uznawać za autorytatywne źródło. W przypadku informacji krytycznych zalecane jest skorzystanie z profesjonalnego tłumaczenia wykonanego przez człowieka. Nie ponosimy odpowiedzialności za jakiekolwiek nieporozumienia lub błędne interpretacje wynikające z użycia tego tłumaczenia.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->