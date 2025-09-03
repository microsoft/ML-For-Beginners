<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1f2b7441745eb52e25745423b247016b",
  "translation_date": "2025-09-03T18:43:46+00:00",
  "source_file": "8-Reinforcement/2-Gym/assignment.md",
  "language_code": "pl"
}
-->
# Trenowanie Mountain Car

[OpenAI Gym](http://gym.openai.com) został zaprojektowany w taki sposób, że wszystkie środowiska oferują ten sam interfejs API - tj. te same metody `reset`, `step` i `render`, oraz te same abstrakcje **przestrzeni akcji** i **przestrzeni obserwacji**. Dzięki temu powinno być możliwe dostosowanie tych samych algorytmów uczenia ze wzmocnieniem do różnych środowisk przy minimalnych zmianach w kodzie.

## Środowisko Mountain Car

[Środowisko Mountain Car](https://gym.openai.com/envs/MountainCar-v0/) zawiera samochód uwięziony w dolinie:

Celem jest wydostanie się z doliny i zdobycie flagi, wykonując w każdym kroku jedną z następujących akcji:

| Wartość | Znaczenie |
|---|---|
| 0 | Przyspieszenie w lewo |
| 1 | Brak przyspieszenia |
| 2 | Przyspieszenie w prawo |

Głównym wyzwaniem w tym problemie jest jednak to, że silnik samochodu nie jest wystarczająco mocny, aby pokonać górę za jednym razem. Dlatego jedynym sposobem na sukces jest jazda tam i z powrotem, aby nabrać rozpędu.

Przestrzeń obserwacji składa się tylko z dwóch wartości:

| Nr | Obserwacja  | Min | Max |
|-----|--------------|-----|-----|
|  0  | Pozycja samochodu | -1.2| 0.6 |
|  1  | Prędkość samochodu | -0.07 | 0.07 |

System nagród dla Mountain Car jest dość wymagający:

 * Nagroda 0 jest przyznawana, jeśli agent dotrze do flagi (pozycja = 0.5) na szczycie góry.
 * Nagroda -1 jest przyznawana, jeśli pozycja agenta jest mniejsza niż 0.5.

Epizod kończy się, jeśli pozycja samochodu przekracza 0.5 lub długość epizodu wynosi więcej niż 200 kroków.

## Instrukcje

Dostosuj nasz algorytm uczenia ze wzmocnieniem, aby rozwiązać problem Mountain Car. Rozpocznij od istniejącego kodu w [notebook.ipynb](notebook.ipynb), zastąp środowisko, zmień funkcje dyskretyzacji stanu i spróbuj sprawić, aby istniejący algorytm trenował przy minimalnych modyfikacjach kodu. Optymalizuj wynik, dostosowując hiperparametry.

> **Note**: Dostosowanie hiperparametrów prawdopodobnie będzie konieczne, aby algorytm się zbiegał.

## Kryteria oceny

| Kryterium | Wzorowe | Wystarczające | Wymaga poprawy |
| -------- | --------- | -------- | ----------------- |
|          | Algorytm Q-Learning został pomyślnie dostosowany z przykładu CartPole, przy minimalnych modyfikacjach kodu, i jest w stanie rozwiązać problem zdobycia flagi w mniej niż 200 krokach. | Nowy algorytm Q-Learning został zaadaptowany z Internetu, ale jest dobrze udokumentowany; lub istniejący algorytm został zaadaptowany, ale nie osiąga pożądanych wyników. | Student nie był w stanie pomyślnie zaadaptować żadnego algorytmu, ale poczynił znaczące kroki w kierunku rozwiązania (zaimplementował dyskretyzację stanu, strukturę danych Q-Table, itp.) |

---

**Zastrzeżenie**:  
Ten dokument został przetłumaczony za pomocą usługi tłumaczeniowej AI [Co-op Translator](https://github.com/Azure/co-op-translator). Chociaż dokładamy wszelkich starań, aby tłumaczenie było precyzyjne, prosimy pamiętać, że automatyczne tłumaczenia mogą zawierać błędy lub nieścisłości. Oryginalny dokument w jego rodzimym języku powinien być uznawany za wiarygodne źródło. W przypadku informacji krytycznych zaleca się skorzystanie z profesjonalnego tłumaczenia wykonanego przez człowieka. Nie ponosimy odpowiedzialności za jakiekolwiek nieporozumienia lub błędne interpretacje wynikające z korzystania z tego tłumaczenia.