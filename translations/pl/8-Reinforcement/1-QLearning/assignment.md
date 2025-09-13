<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "68394b2102d3503882e5e914bd0ff5c1",
  "translation_date": "2025-09-03T18:36:35+00:00",
  "source_file": "8-Reinforcement/1-QLearning/assignment.md",
  "language_code": "pl"
}
-->
# Bardziej Realistyczny Świat

W naszej sytuacji Piotr mógł poruszać się niemal bez zmęczenia czy głodu. W bardziej realistycznym świecie musi od czasu do czasu usiąść i odpocząć, a także coś zjeść. Uczyńmy nasz świat bardziej realistycznym, wprowadzając następujące zasady:

1. Przemieszczając się z jednego miejsca do drugiego, Piotr traci **energię** i zyskuje **zmęczenie**.
2. Piotr może odzyskać energię, jedząc jabłka.
3. Piotr może pozbyć się zmęczenia, odpoczywając pod drzewem lub na trawie (czyli wchodząc na pole planszy z drzewem lub trawą - zielone pole).
4. Piotr musi znaleźć i zabić wilka.
5. Aby zabić wilka, Piotr musi mieć odpowiedni poziom energii i zmęczenia, w przeciwnym razie przegrywa walkę.

## Instrukcje

Użyj oryginalnego notebooka [notebook.ipynb](notebook.ipynb) jako punktu wyjścia dla swojego rozwiązania.

Zmodyfikuj funkcję nagrody zgodnie z zasadami gry, uruchom algorytm uczenia ze wzmocnieniem, aby nauczyć się najlepszej strategii wygrywania gry, i porównaj wyniki losowego spaceru z wynikami swojego algorytmu pod względem liczby wygranych i przegranych gier.

> **Note**: W Twoim nowym świecie stan jest bardziej złożony i oprócz pozycji człowieka obejmuje również poziomy zmęczenia i energii. Możesz zdecydować się na reprezentowanie stanu jako krotki (Plansza, energia, zmęczenie), zdefiniować klasę dla stanu (możesz również chcieć wyprowadzić ją z `Board`), lub nawet zmodyfikować oryginalną klasę `Board` w pliku [rlboard.py](../../../../8-Reinforcement/1-QLearning/rlboard.py).

W swoim rozwiązaniu zachowaj kod odpowiedzialny za strategię losowego spaceru i porównaj wyniki swojego algorytmu z losowym spacerem na końcu.

> **Note**: Możesz potrzebować dostosować hiperparametry, aby wszystko działało, szczególnie liczbę epok. Ponieważ sukces w grze (walka z wilkiem) jest rzadkim wydarzeniem, możesz spodziewać się znacznie dłuższego czasu treningu.

## Kryteria Oceny

| Kryterium | Wzorowe                                                                                                                                                                                                 | Wystarczające                                                                                                                                                                           | Wymaga Poprawy                                                                                                                             |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
|           | Notebook zawiera definicję nowych zasad świata, algorytm Q-Learning oraz tekstowe wyjaśnienia. Q-Learning znacząco poprawia wyniki w porównaniu do losowego spaceru.                                    | Notebook jest przedstawiony, Q-Learning jest zaimplementowany i poprawia wyniki w porównaniu do losowego spaceru, ale nie znacząco; lub notebook jest słabo udokumentowany, a kod nie jest dobrze zorganizowany. | Podjęto próbę redefinicji zasad świata, ale algorytm Q-Learning nie działa, lub funkcja nagrody nie jest w pełni zdefiniowana.             |

---

**Zastrzeżenie**:  
Ten dokument został przetłumaczony za pomocą usługi tłumaczenia AI [Co-op Translator](https://github.com/Azure/co-op-translator). Chociaż dokładamy wszelkich starań, aby tłumaczenie było precyzyjne, prosimy pamiętać, że automatyczne tłumaczenia mogą zawierać błędy lub nieścisłości. Oryginalny dokument w jego rodzimym języku powinien być uznawany za wiarygodne źródło. W przypadku informacji o kluczowym znaczeniu zaleca się skorzystanie z profesjonalnego tłumaczenia przez człowieka. Nie ponosimy odpowiedzialności za jakiekolwiek nieporozumienia lub błędne interpretacje wynikające z użycia tego tłumaczenia.