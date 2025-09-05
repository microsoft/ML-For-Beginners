<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9a6b702d1437c0467e3c5c28d763dac2",
  "translation_date": "2025-09-05T08:20:26+00:00",
  "source_file": "1-Introduction/3-fairness/README.md",
  "language_code": "pl"
}
-->
# Budowanie rozwiązań Machine Learning z odpowiedzialną sztuczną inteligencją

![Podsumowanie odpowiedzialnej AI w Machine Learning na szkicowej notatce](../../../../sketchnotes/ml-fairness.png)
> Szkicowa notatka autorstwa [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Quiz przed wykładem](https://ff-quizzes.netlify.app/en/ml/)

## Wprowadzenie

W tym programie nauczania zaczniesz odkrywać, jak uczenie maszynowe wpływa na nasze codzienne życie. Już teraz systemy i modele są zaangażowane w codzienne zadania decyzyjne, takie jak diagnozy medyczne, zatwierdzanie kredytów czy wykrywanie oszustw. Dlatego ważne jest, aby te modele działały dobrze i dostarczały wyniki, którym można zaufać. Podobnie jak każda aplikacja programowa, systemy AI mogą nie spełniać oczekiwań lub prowadzić do niepożądanych rezultatów. Dlatego kluczowe jest zrozumienie i wyjaśnienie zachowania modelu AI.

Wyobraź sobie, co może się stać, gdy dane używane do budowy tych modeli nie uwzględniają pewnych grup demograficznych, takich jak rasa, płeć, poglądy polityczne, religia, lub gdy są one nadmiernie reprezentowane. Co, jeśli wyniki modelu są interpretowane w sposób faworyzujący pewne grupy demograficzne? Jakie są konsekwencje dla aplikacji? Co się dzieje, gdy model prowadzi do szkodliwych rezultatów? Kto jest odpowiedzialny za zachowanie systemów AI? To są pytania, które będziemy eksplorować w tym programie nauczania.

W tej lekcji dowiesz się:

- Dlaczego sprawiedliwość w uczeniu maszynowym i związane z nią szkody są tak ważne.
- Jak badać odstające przypadki i nietypowe scenariusze, aby zapewnić niezawodność i bezpieczeństwo.
- Dlaczego projektowanie inkluzywnych systemów jest kluczowe dla wzmocnienia pozycji wszystkich ludzi.
- Jak istotne jest chronienie prywatności i bezpieczeństwa danych oraz osób.
- Dlaczego podejście „szklanej skrzynki” jest ważne dla wyjaśnienia zachowania modeli AI.
- Jak odpowiedzialność buduje zaufanie do systemów AI.

## Wymagania wstępne

Jako wymaganie wstępne, zapoznaj się z „Zasadami odpowiedzialnej AI” w ścieżce nauki i obejrzyj poniższy film na ten temat:

Dowiedz się więcej o odpowiedzialnej AI, korzystając z tej [ścieżki nauki](https://docs.microsoft.com/learn/modules/responsible-ai-principles/?WT.mc_id=academic-77952-leestott)

[![Podejście Microsoftu do odpowiedzialnej AI](https://img.youtube.com/vi/dnC8-uUZXSc/0.jpg)](https://youtu.be/dnC8-uUZXSc "Podejście Microsoftu do odpowiedzialnej AI")

> 🎥 Kliknij obrazek powyżej, aby obejrzeć film: Podejście Microsoftu do odpowiedzialnej AI

## Sprawiedliwość

Systemy AI powinny traktować wszystkich sprawiedliwie i unikać różnic w traktowaniu podobnych grup ludzi. Na przykład, gdy systemy AI udzielają wskazówek dotyczących leczenia medycznego, aplikacji kredytowych czy zatrudnienia, powinny wydawać takie same rekomendacje wszystkim osobom o podobnych objawach, sytuacji finansowej czy kwalifikacjach zawodowych. Każdy z nas jako człowiek nosi w sobie odziedziczone uprzedzenia, które wpływają na nasze decyzje i działania. Te uprzedzenia mogą być widoczne w danych, które wykorzystujemy do trenowania systemów AI. Czasami takie manipulacje zdarzają się nieumyślnie. Często trudno jest świadomie zauważyć, kiedy wprowadzamy uprzedzenia do danych.

**„Niesprawiedliwość”** obejmuje negatywne skutki, czyli „szkody”, dla grupy ludzi, takich jak te zdefiniowane na podstawie rasy, płci, wieku czy statusu niepełnosprawności. Główne szkody związane ze sprawiedliwością można sklasyfikować jako:

- **Alokacja**, gdy na przykład płeć lub etniczność jest faworyzowana kosztem innych.
- **Jakość usług**. Jeśli dane są trenowane dla jednego konkretnego scenariusza, ale rzeczywistość jest znacznie bardziej złożona, prowadzi to do słabo działającej usługi. Na przykład dozownik mydła, który nie potrafił rozpoznać osób o ciemnej skórze. [Źródło](https://gizmodo.com/why-cant-this-soap-dispenser-identify-dark-skin-1797931773)
- **Oczernianie**. Niesprawiedliwe krytykowanie i etykietowanie czegoś lub kogoś. Na przykład technologia etykietowania obrazów niesławnie błędnie oznaczyła zdjęcia osób o ciemnej skórze jako goryle.
- **Nadmierna lub niedostateczna reprezentacja**. Chodzi o to, że pewna grupa nie jest widoczna w określonym zawodzie, a każda usługa lub funkcja, która to utrwala, przyczynia się do szkody.
- **Stereotypizacja**. Przypisywanie określonej grupie z góry ustalonych cech. Na przykład system tłumaczenia językowego między angielskim a tureckim może mieć nieścisłości wynikające ze stereotypowych skojarzeń z płcią.

![tłumaczenie na turecki](../../../../1-Introduction/3-fairness/images/gender-bias-translate-en-tr.png)
> tłumaczenie na turecki

![tłumaczenie z powrotem na angielski](../../../../1-Introduction/3-fairness/images/gender-bias-translate-tr-en.png)
> tłumaczenie z powrotem na angielski

Podczas projektowania i testowania systemów AI musimy upewnić się, że AI jest sprawiedliwe i nie jest zaprogramowane do podejmowania stronniczych lub dyskryminujących decyzji, których ludzie również nie powinni podejmować. Zapewnienie sprawiedliwości w AI i uczeniu maszynowym pozostaje złożonym wyzwaniem socjotechnicznym.

### Niezawodność i bezpieczeństwo

Aby budować zaufanie, systemy AI muszą być niezawodne, bezpieczne i spójne w normalnych i nieoczekiwanych warunkach. Ważne jest, aby wiedzieć, jak systemy AI będą się zachowywać w różnych sytuacjach, zwłaszcza w przypadku odstających przypadków. Podczas budowania rozwiązań AI należy poświęcić znaczną uwagę temu, jak radzić sobie z szeroką gamą okoliczności, które mogą napotkać rozwiązania AI. Na przykład samochód autonomiczny musi stawiać bezpieczeństwo ludzi na pierwszym miejscu. W związku z tym AI napędzające samochód musi uwzględniać wszystkie możliwe scenariusze, które samochód może napotkać, takie jak noc, burze, zamiecie śnieżne, dzieci biegnące przez ulicę, zwierzęta domowe, roboty drogowe itd. To, jak dobrze system AI radzi sobie z szerokim zakresem warunków w sposób niezawodny i bezpieczny, odzwierciedla poziom przewidywania, który naukowiec danych lub programista AI uwzględnił podczas projektowania lub testowania systemu.

> [🎥 Kliknij tutaj, aby obejrzeć film: ](https://www.microsoft.com/videoplayer/embed/RE4vvIl)

### Inkluzywność

Systemy AI powinny być projektowane tak, aby angażować i wzmacniać pozycję wszystkich ludzi. Podczas projektowania i wdrażania systemów AI naukowcy danych i programiści AI identyfikują i rozwiązują potencjalne bariery w systemie, które mogłyby nieumyślnie wykluczyć ludzi. Na przykład na świecie jest 1 miliard osób z niepełnosprawnościami. Dzięki postępowi w dziedzinie AI mogą oni łatwiej uzyskiwać dostęp do szerokiego zakresu informacji i możliwości w codziennym życiu. Rozwiązywanie barier tworzy możliwości innowacji i rozwijania produktów AI z lepszymi doświadczeniami, które przynoszą korzyści wszystkim.

> [🎥 Kliknij tutaj, aby obejrzeć film: inkluzywność w AI](https://www.microsoft.com/videoplayer/embed/RE4vl9v)

### Bezpieczeństwo i prywatność

Systemy AI powinny być bezpieczne i szanować prywatność ludzi. Ludzie mają mniejsze zaufanie do systemów, które narażają ich prywatność, informacje lub życie na ryzyko. Podczas trenowania modeli uczenia maszynowego polegamy na danych, aby uzyskać najlepsze wyniki. W związku z tym należy wziąć pod uwagę pochodzenie danych i ich integralność. Na przykład, czy dane zostały dostarczone przez użytkownika, czy były publicznie dostępne? Następnie, pracując z danymi, kluczowe jest opracowanie systemów AI, które mogą chronić poufne informacje i opierać się atakom. W miarę jak AI staje się coraz bardziej powszechne, ochrona prywatności i zabezpieczanie ważnych informacji osobistych i biznesowych staje się coraz bardziej istotna i złożona. Problemy związane z prywatnością i bezpieczeństwem danych wymagają szczególnej uwagi w przypadku AI, ponieważ dostęp do danych jest niezbędny, aby systemy AI mogły dokonywać dokładnych i świadomych prognoz oraz podejmować decyzje dotyczące ludzi.

> [🎥 Kliknij tutaj, aby obejrzeć film: bezpieczeństwo w AI](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Branża poczyniła znaczące postępy w zakresie prywatności i bezpieczeństwa, w dużej mierze dzięki regulacjom takim jak GDPR (Ogólne Rozporządzenie o Ochronie Danych).
- Jednak w przypadku systemów AI musimy uznać napięcie między potrzebą większej ilości danych osobowych w celu uczynienia systemów bardziej osobistymi i skutecznymi – a prywatnością.
- Podobnie jak w przypadku narodzin połączonych komputerów z internetem, obserwujemy również ogromny wzrost liczby problemów związanych z bezpieczeństwem w kontekście AI.
- Jednocześnie widzimy, że AI jest wykorzystywane do poprawy bezpieczeństwa. Na przykład większość nowoczesnych skanerów antywirusowych jest dziś napędzana przez heurystyki AI.
- Musimy upewnić się, że nasze procesy Data Science harmonijnie współgrają z najnowszymi praktykami dotyczącymi prywatności i bezpieczeństwa.

### Przejrzystość

Systemy AI powinny być zrozumiałe. Kluczowym elementem przejrzystości jest wyjaśnienie zachowania systemów AI i ich komponentów. Poprawa zrozumienia systemów AI wymaga, aby interesariusze rozumieli, jak i dlaczego działają, aby mogli zidentyfikować potencjalne problemy z wydajnością, obawy dotyczące bezpieczeństwa i prywatności, uprzedzenia, praktyki wykluczające lub niezamierzone rezultaty. Wierzymy również, że ci, którzy korzystają z systemów AI, powinni być uczciwi i otwarci w kwestii tego, kiedy, dlaczego i jak decydują się je wdrażać. A także w kwestii ograniczeń systemów, z których korzystają. Na przykład, jeśli bank korzysta z systemu AI wspierającego decyzje kredytowe dla konsumentów, ważne jest, aby przeanalizować wyniki i zrozumieć, które dane wpływają na rekomendacje systemu. Rządy zaczynają regulować AI w różnych branżach, więc naukowcy danych i organizacje muszą wyjaśnić, czy system AI spełnia wymagania regulacyjne, zwłaszcza gdy pojawia się niepożądany rezultat.

> [🎥 Kliknij tutaj, aby obejrzeć film: przejrzystość w AI](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Ponieważ systemy AI są tak złożone, trudno jest zrozumieć, jak działają i interpretować wyniki.
- Ten brak zrozumienia wpływa na sposób zarządzania, operacjonalizacji i dokumentowania tych systemów.
- Co ważniejsze, brak zrozumienia wpływa na decyzje podejmowane na podstawie wyników, które te systemy produkują.

### Odpowiedzialność

Osoby projektujące i wdrażające systemy AI muszą być odpowiedzialne za sposób, w jaki ich systemy działają. Potrzeba odpowiedzialności jest szczególnie istotna w przypadku technologii wrażliwych, takich jak rozpoznawanie twarzy. W ostatnim czasie rośnie zapotrzebowanie na technologię rozpoznawania twarzy, zwłaszcza ze strony organizacji zajmujących się egzekwowaniem prawa, które dostrzegają potencjał tej technologii w takich zastosowaniach jak odnajdywanie zaginionych dzieci. Jednak te technologie mogą być potencjalnie wykorzystywane przez rządy do naruszania podstawowych wolności obywateli, na przykład poprzez umożliwienie ciągłego monitorowania konkretnych osób. Dlatego naukowcy danych i organizacje muszą być odpowiedzialni za to, jak ich systemy AI wpływają na jednostki lub społeczeństwo.

[![Czołowy badacz AI ostrzega przed masową inwigilacją za pomocą rozpoznawania twarzy](../../../../1-Introduction/3-fairness/images/accountability.png)](https://www.youtube.com/watch?v=Wldt8P5V6D0 "Podejście Microsoftu do odpowiedzialnej AI")

> 🎥 Kliknij obrazek powyżej, aby obejrzeć film: Ostrzeżenia przed masową inwigilacją za pomocą rozpoznawania twarzy

Ostatecznie jednym z największych pytań dla naszego pokolenia, jako pierwszego pokolenia wprowadzającego AI do społeczeństwa, jest to, jak zapewnić, że komputery pozostaną odpowiedzialne wobec ludzi i jak zapewnić, że osoby projektujące komputery pozostaną odpowiedzialne wobec innych.

## Ocena wpływu

Przed trenowaniem modelu uczenia maszynowego ważne jest przeprowadzenie oceny wpływu, aby zrozumieć cel systemu AI; jego zamierzone zastosowanie; miejsce wdrożenia; oraz osoby, które będą wchodzić w interakcję z systemem. Są to pomocne informacje dla recenzentów lub testerów oceniających system, aby wiedzieć, jakie czynniki należy wziąć pod uwagę przy identyfikowaniu potencjalnych ryzyk i oczekiwanych konsekwencji.

Poniżej znajdują się obszary, na których należy się skupić podczas przeprowadzania oceny wpływu:

* **Negatywny wpływ na jednostki**. Ważne jest, aby być świadomym wszelkich ograniczeń lub wymagań, nieobsługiwanych zastosowań lub znanych ograniczeń utrudniających działanie systemu, aby upewnić się, że system nie jest używany w sposób, który mógłby zaszkodzić jednostkom.
* **Wymagania dotyczące danych**. Zrozumienie, jak i gdzie system będzie korzystał z danych, pozwala recenzentom zbadać wszelkie wymagania dotyczące danych, które należy uwzględnić (np. regulacje GDPR lub HIPPA). Ponadto należy sprawdzić, czy źródło lub ilość danych są wystarczające do trenowania.
* **Podsumowanie wpływu**. Zbierz listę potencjalnych szkód, które mogą wyniknąć z używania systemu. Przez cały cykl życia ML sprawdzaj, czy zidentyfikowane problemy zostały złagodzone lub rozwiązane.
* **Cele dla każdej z sześciu podstawowych zasad**. Oceń, czy cele wynikające z każdej zasady zostały osiągnięte i czy istnieją jakieś luki.

## Debugowanie z odpowiedzialną AI

Podobnie jak debugowanie aplikacji programowej, debugowanie systemu AI jest niezbędnym procesem identyfikowania i rozwiązywania problemów w systemie. Istnieje wiele czynników, które mogą wpływać na to, że model nie działa zgodnie z oczekiwaniami lub odpowiedzialnie. Większość tradycyjnych metryk wydajności modelu to ilościowe agregaty wydajności modelu, które nie są wystarczające do analizy, w jaki sposób model narusza zasady odpowiedzialnej AI. Ponadto model uczenia maszynowego jest „czarną skrzynką”, co utrudnia zrozumienie, co napędza jego wyniki lub wyjaśnienie, dlaczego popełnia błędy. W dalszej części tego kursu nauczymy się, jak korzystać z pulpitu odpowiedzialnej AI, aby pomóc w debugowaniu systemów AI. Pulpit zapewnia kompleksowe narzędzie dla naukowców danych i programistów AI do wykonywania:

* **Analizy błędów**. Aby zidentyfikować rozkład błędów modelu, który może wpływać na sprawiedliwość lub niezawodność systemu.
* **Przeglądu modelu**. Aby odkryć, gdzie występują różnice w wydajności modelu w różnych grupach danych.
* **Analizy danych**. Aby zrozumieć rozkład danych i zidentyfik
Obejrzyj ten warsztat, aby zgłębić tematy:

- W pogoni za odpowiedzialną AI: Wprowadzenie zasad w praktykę przez Besmirę Nushi, Mehrnoosh Sameki i Amita Sharmę

[![Responsible AI Toolbox: Otwartoźródłowe narzędzie do budowy odpowiedzialnej AI](https://img.youtube.com/vi/tGgJCrA-MZU/0.jpg)](https://www.youtube.com/watch?v=tGgJCrA-MZU "RAI Toolbox: Otwartoźródłowe narzędzie do budowy odpowiedzialnej AI")

> 🎥 Kliknij obrazek powyżej, aby obejrzeć wideo: RAI Toolbox: Otwartoźródłowe narzędzie do budowy odpowiedzialnej AI przez Besmirę Nushi, Mehrnoosh Sameki i Amita Sharmę

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
Ten dokument został przetłumaczony za pomocą usługi tłumaczeniowej AI [Co-op Translator](https://github.com/Azure/co-op-translator). Chociaż dokładamy wszelkich starań, aby zapewnić dokładność, prosimy pamiętać, że automatyczne tłumaczenia mogą zawierać błędy lub nieścisłości. Oryginalny dokument w jego rodzimym języku powinien być uznawany za wiarygodne źródło. W przypadku informacji krytycznych zaleca się skorzystanie z profesjonalnego tłumaczenia wykonanego przez człowieka. Nie ponosimy odpowiedzialności za jakiekolwiek nieporozumienia lub błędne interpretacje wynikające z korzystania z tego tłumaczenia.