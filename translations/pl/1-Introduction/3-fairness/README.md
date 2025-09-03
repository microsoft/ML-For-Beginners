<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "8f819813b2ca08ec7b9f60a2c9336045",
  "translation_date": "2025-09-03T17:37:05+00:00",
  "source_file": "1-Introduction/3-fairness/README.md",
  "language_code": "pl"
}
-->
# Budowanie rozwiązań Machine Learning z odpowiedzialną AI

![Podsumowanie odpowiedzialnej AI w Machine Learning na szkicowniku](../../../../translated_images/ml-fairness.ef296ebec6afc98a44566d7b6c1ed18dc2bf1115c13ec679bb626028e852fa1d.pl.png)
> Szkicownik autorstwa [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Quiz przed wykładem](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/5/)

## Wprowadzenie

W tym kursie zaczniesz odkrywać, jak uczenie maszynowe wpływa na nasze codzienne życie. Już teraz systemy i modele są zaangażowane w codzienne zadania decyzyjne, takie jak diagnozy medyczne, zatwierdzanie kredytów czy wykrywanie oszustw. Dlatego ważne jest, aby te modele działały dobrze i dostarczały wyniki, którym można zaufać. Podobnie jak każda aplikacja, systemy AI mogą nie spełniać oczekiwań lub prowadzić do niepożądanych rezultatów. Dlatego kluczowe jest zrozumienie i wyjaśnienie zachowania modelu AI.

Wyobraź sobie, co może się stać, gdy dane używane do budowy tych modeli nie uwzględniają pewnych grup demograficznych, takich jak rasa, płeć, poglądy polityczne, religia, lub gdy są one nadmiernie reprezentowane. Co, jeśli wyniki modelu są interpretowane w sposób faworyzujący pewne grupy? Jakie są konsekwencje dla aplikacji? Co się dzieje, gdy model prowadzi do szkodliwych rezultatów? Kto jest odpowiedzialny za zachowanie systemu AI? To są pytania, które będziemy eksplorować w tym kursie.

W tej lekcji dowiesz się:

- Dlaczego sprawiedliwość w uczeniu maszynowym i związane z nią szkody są tak ważne.
- Jak badać odstające przypadki i nietypowe scenariusze, aby zapewnić niezawodność i bezpieczeństwo.
- Jak projektować inkluzywne systemy, które wspierają wszystkich.
- Dlaczego ochrona prywatności i bezpieczeństwa danych oraz ludzi jest kluczowa.
- Jak ważne jest wyjaśnianie zachowania modeli AI w sposób przejrzysty.
- Dlaczego odpowiedzialność jest kluczowa dla budowania zaufania do systemów AI.

## Wymagania wstępne

Przed rozpoczęciem, zapoznaj się z "Zasadami odpowiedzialnej AI" w ścieżce edukacyjnej i obejrzyj poniższy film na ten temat:

Dowiedz się więcej o odpowiedzialnej AI, korzystając z tej [ścieżki edukacyjnej](https://docs.microsoft.com/learn/modules/responsible-ai-principles/?WT.mc_id=academic-77952-leestott)

[![Podejście Microsoftu do odpowiedzialnej AI](https://img.youtube.com/vi/dnC8-uUZXSc/0.jpg)](https://youtu.be/dnC8-uUZXSc "Podejście Microsoftu do odpowiedzialnej AI")

> 🎥 Kliknij obrazek powyżej, aby obejrzeć film: Podejście Microsoftu do odpowiedzialnej AI

## Sprawiedliwość

Systemy AI powinny traktować wszystkich sprawiedliwie i unikać różnic w traktowaniu podobnych grup ludzi. Na przykład, gdy systemy AI udzielają wskazówek dotyczących leczenia, aplikacji kredytowych czy zatrudnienia, powinny wydawać takie same rekomendacje wszystkim osobom o podobnych objawach, sytuacji finansowej czy kwalifikacjach zawodowych. Każdy z nas jako człowiek nosi w sobie odziedziczone uprzedzenia, które wpływają na nasze decyzje i działania. Te uprzedzenia mogą być widoczne w danych, które wykorzystujemy do trenowania systemów AI. Czasami takie manipulacje zdarzają się nieumyślnie. Trudno jest świadomie zauważyć, kiedy wprowadzamy uprzedzenia do danych.

**„Niesprawiedliwość”** obejmuje negatywne skutki, czyli „szkody”, dla grup ludzi, takich jak te zdefiniowane na podstawie rasy, płci, wieku czy niepełnosprawności. Główne szkody związane ze sprawiedliwością można sklasyfikować jako:

- **Alokacja**, gdy na przykład płeć lub etniczność jest faworyzowana kosztem innych.
- **Jakość usług**. Jeśli dane są trenowane dla jednego konkretnego scenariusza, ale rzeczywistość jest znacznie bardziej złożona, prowadzi to do słabej jakości usług. Na przykład dozownik mydła, który nie potrafił rozpoznać osób o ciemnej skórze. [Źródło](https://gizmodo.com/why-cant-this-soap-dispenser-identify-dark-skin-1797931773)
- **Oczernianie**. Niesprawiedliwe krytykowanie i etykietowanie czegoś lub kogoś. Na przykład technologia etykietowania obrazów błędnie oznaczyła zdjęcia osób o ciemnej skórze jako goryle.
- **Nadmierna lub niedostateczna reprezentacja**. Chodzi o to, że pewna grupa nie jest widoczna w określonym zawodzie, a każda usługa lub funkcja, która to utrwala, przyczynia się do szkody.
- **Stereotypizacja**. Przypisywanie określonej grupie z góry ustalonych cech. Na przykład system tłumaczenia językowego między angielskim a tureckim może mieć nieścisłości wynikające ze stereotypowych skojarzeń z płcią.

![tłumaczenie na turecki](../../../../translated_images/gender-bias-translate-en-tr.f185fd8822c2d4372912f2b690f6aaddd306ffbb49d795ad8d12a4bf141e7af0.pl.png)
> tłumaczenie na turecki

![tłumaczenie z powrotem na angielski](../../../../translated_images/gender-bias-translate-tr-en.4eee7e3cecb8c70e13a8abbc379209bc8032714169e585bdeac75af09b1752aa.pl.png)
> tłumaczenie z powrotem na angielski

Podczas projektowania i testowania systemów AI musimy upewnić się, że AI jest sprawiedliwe i nie jest zaprogramowane do podejmowania uprzedzonych lub dyskryminujących decyzji, których ludzie również nie powinni podejmować. Zapewnienie sprawiedliwości w AI i uczeniu maszynowym pozostaje złożonym wyzwaniem społeczno-technologicznym.

### Niezawodność i bezpieczeństwo

Aby budować zaufanie, systemy AI muszą być niezawodne, bezpieczne i spójne w normalnych i nieoczekiwanych warunkach. Ważne jest, aby wiedzieć, jak systemy AI będą się zachowywać w różnych sytuacjach, zwłaszcza w przypadku odstających przypadków. Podczas budowania rozwiązań AI należy poświęcić znaczną uwagę temu, jak radzić sobie z szeroką gamą okoliczności, z którymi mogą się spotkać. Na przykład samochód autonomiczny musi stawiać bezpieczeństwo ludzi na pierwszym miejscu. W związku z tym AI napędzające samochód musi uwzględniać wszystkie możliwe scenariusze, takie jak noc, burze, zamiecie śnieżne, dzieci przebiegające przez ulicę, zwierzęta, roboty drogowe itd. To, jak dobrze system AI radzi sobie z szerokim zakresem warunków w sposób niezawodny i bezpieczny, odzwierciedla poziom przewidywania, który naukowiec danych lub programista AI uwzględnił podczas projektowania lub testowania systemu.

> [🎥 Kliknij tutaj, aby obejrzeć film: ](https://www.microsoft.com/videoplayer/embed/RE4vvIl)

### Inkluzywność

Systemy AI powinny być projektowane tak, aby angażować i wspierać wszystkich. Podczas projektowania i wdrażania systemów AI naukowcy danych i programiści AI identyfikują i eliminują potencjalne bariery w systemie, które mogłyby nieumyślnie wykluczać ludzi. Na przykład na świecie jest 1 miliard osób z niepełnosprawnościami. Dzięki postępowi AI mogą oni łatwiej uzyskiwać dostęp do szerokiego zakresu informacji i możliwości w codziennym życiu. Eliminowanie barier tworzy możliwości innowacji i rozwijania produktów AI z lepszymi doświadczeniami, które przynoszą korzyści wszystkim.

> [🎥 Kliknij tutaj, aby obejrzeć film: inkluzywność w AI](https://www.microsoft.com/videoplayer/embed/RE4vl9v)

### Bezpieczeństwo i prywatność

Systemy AI powinny być bezpieczne i szanować prywatność ludzi. Ludzie mają mniejsze zaufanie do systemów, które narażają ich prywatność, informacje lub życie na ryzyko. Podczas trenowania modeli uczenia maszynowego polegamy na danych, aby uzyskać najlepsze wyniki. W związku z tym należy uwzględnić pochodzenie danych i ich integralność. Na przykład, czy dane zostały dostarczone przez użytkownika, czy są publicznie dostępne? Następnie, pracując z danymi, kluczowe jest opracowanie systemów AI, które mogą chronić poufne informacje i opierać się atakom. W miarę jak AI staje się coraz bardziej powszechne, ochrona prywatności i zabezpieczanie ważnych informacji osobistych i biznesowych staje się coraz bardziej krytyczna i złożona. Problemy związane z prywatnością i bezpieczeństwem danych wymagają szczególnej uwagi w przypadku AI, ponieważ dostęp do danych jest niezbędny, aby systemy AI mogły podejmować dokładne i świadome decyzje dotyczące ludzi.

> [🎥 Kliknij tutaj, aby obejrzeć film: bezpieczeństwo w AI](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Branża poczyniła znaczące postępy w zakresie prywatności i bezpieczeństwa, w dużej mierze dzięki regulacjom takim jak GDPR (Ogólne Rozporządzenie o Ochronie Danych).
- Jednak w przypadku systemów AI musimy uznać napięcie między potrzebą większej ilości danych osobowych, aby systemy były bardziej efektywne, a prywatnością.
- Podobnie jak w przypadku narodzin połączonych komputerów z internetem, obserwujemy również ogromny wzrost liczby problemów związanych z bezpieczeństwem w kontekście AI.
- Jednocześnie widzimy, że AI jest wykorzystywane do poprawy bezpieczeństwa. Na przykład większość nowoczesnych skanerów antywirusowych jest dziś napędzana przez heurystyki AI.
- Musimy upewnić się, że nasze procesy Data Science harmonijnie współgrają z najnowszymi praktykami dotyczącymi prywatności i bezpieczeństwa.

### Przejrzystość

Systemy AI powinny być zrozumiałe. Kluczowym elementem przejrzystości jest wyjaśnianie zachowania systemów AI i ich komponentów. Poprawa zrozumienia systemów AI wymaga, aby interesariusze rozumieli, jak i dlaczego działają, aby mogli zidentyfikować potencjalne problemy z wydajnością, obawy dotyczące bezpieczeństwa i prywatności, uprzedzenia, praktyki wykluczające lub niezamierzone rezultaty. Wierzymy również, że ci, którzy korzystają z systemów AI, powinni być uczciwi i otwarci w kwestii tego, kiedy, dlaczego i jak decydują się je wdrażać, a także ograniczeń systemów, z których korzystają. Na przykład, jeśli bank korzysta z systemu AI, aby wspierać decyzje dotyczące kredytów konsumenckich, ważne jest, aby zbadać wyniki i zrozumieć, które dane wpływają na rekomendacje systemu. Rządy zaczynają regulować AI w różnych branżach, więc naukowcy danych i organizacje muszą wyjaśniać, czy system AI spełnia wymagania regulacyjne, zwłaszcza gdy pojawia się niepożądany rezultat.

> [🎥 Kliknij tutaj, aby obejrzeć film: przejrzystość w AI](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Ponieważ systemy AI są tak złożone, trudno jest zrozumieć, jak działają i interpretować wyniki.
- Ten brak zrozumienia wpływa na sposób zarządzania, operacjonalizacji i dokumentowania tych systemów.
- Co ważniejsze, brak zrozumienia wpływa na decyzje podejmowane na podstawie wyników, które te systemy produkują.

### Odpowiedzialność

Osoby projektujące i wdrażające systemy AI muszą być odpowiedzialne za sposób, w jaki ich systemy działają. Potrzeba odpowiedzialności jest szczególnie istotna w przypadku technologii wrażliwych, takich jak rozpoznawanie twarzy. W ostatnim czasie rośnie zapotrzebowanie na technologię rozpoznawania twarzy, zwłaszcza ze strony organizacji zajmujących się egzekwowaniem prawa, które dostrzegają potencjał tej technologii w takich zastosowaniach jak odnajdywanie zaginionych dzieci. Jednak te technologie mogą być potencjalnie wykorzystywane przez rządy do naruszania podstawowych wolności obywateli, na przykład poprzez umożliwienie ciągłego monitorowania konkretnych osób. Dlatego naukowcy danych i organizacje muszą być odpowiedzialni za to, jak ich system AI wpływa na jednostki lub społeczeństwo.

[![Czołowy badacz AI ostrzega przed masową inwigilacją za pomocą rozpoznawania twarzy](../../../../translated_images/accountability.41d8c0f4b85b6231301d97f17a450a805b7a07aaeb56b34015d71c757cad142e.pl.png)](https://www.youtube.com/watch?v=Wldt8P5V6D0 "Podejście Microsoftu do odpowiedzialnej AI")

> 🎥 Kliknij obrazek powyżej, aby obejrzeć film: Ostrzeżenia przed masową inwigilacją za pomocą rozpoznawania twarzy

Ostatecznie jednym z największych pytań dla naszego pokolenia, jako pierwszego, które wprowadza AI do społeczeństwa, jest to, jak zapewnić, że komputery pozostaną odpowiedzialne wobec ludzi i jak zapewnić, że osoby projektujące komputery pozostaną odpowiedzialne wobec innych.

## Ocena wpływu

Przed trenowaniem modelu uczenia maszynowego ważne jest przeprowadzenie oceny wpływu, aby zrozumieć cel systemu AI, jego zamierzone zastosowanie, miejsce wdrożenia oraz osoby, które będą wchodzić w interakcję z systemem. Są to pomocne informacje dla recenzentów lub testerów oceniających system, aby wiedzieć, jakie czynniki należy wziąć pod uwagę przy identyfikowaniu potencjalnych ryzyk i oczekiwanych konsekwencji.

Oto obszary, na które należy zwrócić uwagę podczas przeprowadzania oceny wpływu:

* **Negatywny wpływ na jednostki**. Ważne jest, aby być świadomym wszelkich ograniczeń, wymagań, nieobsługiwanych zastosowań lub znanych ograniczeń, które mogą utrudniać działanie systemu, aby upewnić się, że system nie jest używany w sposób, który mógłby zaszkodzić jednostkom.
* **Wymagania dotyczące danych**. Zrozumienie, jak i gdzie system będzie korzystał z danych, pozwala recenzentom zbadać wszelkie wymagania dotyczące danych, które należy uwzględnić (np. regulacje GDPR lub HIPPA). Ponadto należy sprawdzić, czy źródło lub ilość danych są wystarczające do trenowania.
* **Podsumowanie wpływu**. Zbierz listę potencjalnych szkód, które mogą wyniknąć z używania systemu. Przez cały cykl życia ML sprawdzaj, czy zidentyfikowane problemy zostały złagodzone lub rozwiązane.
* **Cele związane z sześcioma podstawowymi zasadami**. Oceń, czy cele wynikające z każdej z zasad zostały osiągnięte i czy istnieją jakieś luki.

## Debugowanie z odpowiedzialną AI

Podobnie jak debugowanie aplikacji, debugowanie systemu AI jest niezbędnym procesem identyfikowania i rozwiązywania problemów w systemie. Istnieje wiele czynników, które mogą wpływać na to, że model nie działa zgodnie z oczekiwaniami lub odpowiedzialnie. Większość tradycyjnych metryk wydajności modelu to ilościowe agregaty wydajności modelu, które nie są wystarczające do analizy, w jaki sposób model narusza zasady odpowiedzialnej AI. Ponadto model uczenia maszynowego jest czarną skrzynką, co utrudnia zrozumienie, co napędza jego wyniki lub wyjaśnienie, dlaczego popełnia błędy. W dalszej części kursu nauczymy się korzystać z pulpitu odpowiedzialnej AI, który pomaga debugować systemy AI. Pulpit zapewnia kompleksowe narzędzie dla naukowców danych i programistów AI do wykonywania:

* **Analizy błędów**. Aby zidentyfikować rozkład błędów modelu, który może wpływać na sprawiedliwość lub niezawodność systemu.
* **Przeglądu modelu**. Aby odkryć, gdzie występują różnice w wydajności modelu w różnych grupach danych.
* **Analizy danych**. Aby zrozumieć rozkład danych i zidentyfikować potencjalne uprzedzenia w danych, które mogą prowadzić do problemów ze sprawiedliwością, inkluzywnością i niezawodnością.
* **Interpretacji modelu**. Aby zrozumieć, co wpływa na przewidywania modelu. To pomaga wyjaśnić zachowanie modelu, co jest ważne dla przejrzystości i odpowiedzial
W tej lekcji nauczyłeś się podstawowych pojęć dotyczących sprawiedliwości i niesprawiedliwości w uczeniu maszynowym.  

Obejrzyj ten warsztat, aby zgłębić temat: 

- W pogoni za odpowiedzialną sztuczną inteligencją: Wprowadzenie zasad w praktykę, prowadzone przez Besmirę Nushi, Mehrnoosh Sameki i Amita Sharmę

[![Responsible AI Toolbox: Otwartoźródłowe narzędzie do budowy odpowiedzialnej AI](https://img.youtube.com/vi/tGgJCrA-MZU/0.jpg)](https://www.youtube.com/watch?v=tGgJCrA-MZU "RAI Toolbox: Otwartoźródłowe narzędzie do budowy odpowiedzialnej AI")


> 🎥 Kliknij obrazek powyżej, aby obejrzeć wideo: RAI Toolbox: Otwartoźródłowe narzędzie do budowy odpowiedzialnej AI, prowadzone przez Besmirę Nushi, Mehrnoosh Sameki i Amita Sharmę

Przeczytaj również: 

- Centrum zasobów RAI Microsoftu: [Responsible AI Resources – Microsoft AI](https://www.microsoft.com/ai/responsible-ai-resources?activetab=pivot1%3aprimaryr4) 

- Grupa badawcza FATE Microsoftu: [FATE: Fairness, Accountability, Transparency, and Ethics in AI - Microsoft Research](https://www.microsoft.com/research/theme/fate/) 

RAI Toolbox: 

- [Repozytorium GitHub Responsible AI Toolbox](https://github.com/microsoft/responsible-ai-toolbox)

Przeczytaj o narzędziach Azure Machine Learning zapewniających sprawiedliwość:

- [Azure Machine Learning](https://docs.microsoft.com/azure/machine-learning/concept-fairness-ml?WT.mc_id=academic-77952-leestott) 

## Zadanie

[Poznaj RAI Toolbox](assignment.md)

---

**Zastrzeżenie**:  
Ten dokument został przetłumaczony za pomocą usługi tłumaczenia AI [Co-op Translator](https://github.com/Azure/co-op-translator). Chociaż dokładamy wszelkich starań, aby tłumaczenie było precyzyjne, prosimy pamiętać, że automatyczne tłumaczenia mogą zawierać błędy lub nieścisłości. Oryginalny dokument w jego języku źródłowym powinien być uznawany za autorytatywne źródło. W przypadku informacji o kluczowym znaczeniu zaleca się skorzystanie z profesjonalnego tłumaczenia przez człowieka. Nie ponosimy odpowiedzialności za jakiekolwiek nieporozumienia lub błędne interpretacje wynikające z użycia tego tłumaczenia.