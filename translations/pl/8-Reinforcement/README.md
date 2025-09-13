<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "20ca019012b1725de956681d036d8b18",
  "translation_date": "2025-09-03T18:26:17+00:00",
  "source_file": "8-Reinforcement/README.md",
  "language_code": "pl"
}
-->
# Wprowadzenie do uczenia ze wzmocnieniem

Uczenie ze wzmocnieniem (RL) jest postrzegane jako jeden z podstawowych paradygmatów uczenia maszynowego, obok uczenia nadzorowanego i nienadzorowanego. RL dotyczy podejmowania decyzji: dostarczania właściwych decyzji lub przynajmniej uczenia się na ich podstawie.

Wyobraź sobie, że masz symulowane środowisko, takie jak rynek akcji. Co się stanie, jeśli wprowadzisz określone regulacje? Czy będzie to miało pozytywny czy negatywny efekt? Jeśli wydarzy się coś negatywnego, musisz przyjąć tę _negatywną informację zwrotną_, nauczyć się z niej i zmienić kierunek działania. Jeśli wynik jest pozytywny, musisz budować na tej _pozytywnej informacji zwrotnej_.

![Piotruś i wilk](../../../translated_images/peter.779730f9ba3a8a8d9290600dcf55f2e491c0640c785af7ac0d64f583c49b8864.pl.png)

> Piotruś i jego przyjaciele muszą uciec przed głodnym wilkiem! Obraz autorstwa [Jen Looper](https://twitter.com/jenlooper)

## Temat regionalny: Piotruś i wilk (Rosja)

[Piotruś i wilk](https://en.wikipedia.org/wiki/Peter_and_the_Wolf) to muzyczna bajka napisana przez rosyjskiego kompozytora [Siergieja Prokofiewa](https://en.wikipedia.org/wiki/Sergei_Prokofiev). Opowiada historię młodego pioniera Piotrusia, który odważnie wychodzi z domu na polanę w lesie, aby ścigać wilka. W tej sekcji będziemy trenować algorytmy uczenia maszynowego, które pomogą Piotrusiowi:

- **Eksplorować** otaczający teren i stworzyć optymalną mapę nawigacyjną.
- **Nauczyć się** korzystać z deskorolki i utrzymywać równowagę, aby poruszać się szybciej.

[![Piotruś i wilk](https://img.youtube.com/vi/Fmi5zHg4QSM/0.jpg)](https://www.youtube.com/watch?v=Fmi5zHg4QSM)

> 🎥 Kliknij obrazek powyżej, aby posłuchać "Piotruś i wilk" autorstwa Prokofiewa

## Uczenie ze wzmocnieniem

W poprzednich sekcjach widziałeś dwa przykłady problemów uczenia maszynowego:

- **Nadzorowane**, gdzie mamy zbiory danych sugerujące przykładowe rozwiązania problemu, który chcemy rozwiązać. [Klasyfikacja](../4-Classification/README.md) i [regresja](../2-Regression/README.md) to zadania uczenia nadzorowanego.
- **Nienadzorowane**, w którym nie mamy oznaczonych danych treningowych. Głównym przykładem uczenia nienadzorowanego jest [Grupowanie](../5-Clustering/README.md).

W tej sekcji wprowadzimy nowy typ problemu uczenia, który nie wymaga oznaczonych danych treningowych. Istnieje kilka rodzajów takich problemów:

- **[Uczenie półnadzorowane](https://wikipedia.org/wiki/Semi-supervised_learning)**, gdzie mamy dużo nieoznaczonych danych, które można wykorzystać do wstępnego trenowania modelu.
- **[Uczenie ze wzmocnieniem](https://wikipedia.org/wiki/Reinforcement_learning)**, w którym agent uczy się, jak się zachowywać, wykonując eksperymenty w symulowanym środowisku.

### Przykład - gra komputerowa

Załóżmy, że chcesz nauczyć komputer grać w grę, na przykład w szachy lub [Super Mario](https://wikipedia.org/wiki/Super_Mario). Aby komputer mógł grać w grę, musimy nauczyć go przewidywać, jaki ruch wykonać w każdym stanie gry. Choć może się to wydawać problemem klasyfikacji, tak nie jest - ponieważ nie mamy zbioru danych ze stanami i odpowiadającymi im akcjami. Chociaż możemy mieć dane, takie jak istniejące partie szachowe lub nagrania graczy grających w Super Mario, prawdopodobnie te dane nie będą wystarczająco obejmować dużej liczby możliwych stanów.

Zamiast szukać istniejących danych o grze, **Uczenie ze wzmocnieniem** (RL) opiera się na idei *sprawienia, by komputer grał* wiele razy i obserwował wynik. Aby zastosować uczenie ze wzmocnieniem, potrzebujemy dwóch rzeczy:

- **Środowiska** i **symulatora**, które pozwolą nam grać w grę wiele razy. Ten symulator definiuje wszystkie zasady gry, a także możliwe stany i akcje.

- **Funkcji nagrody**, która powie nam, jak dobrze radziliśmy sobie podczas każdego ruchu lub gry.

Główna różnica między innymi typami uczenia maszynowego a RL polega na tym, że w RL zazwyczaj nie wiemy, czy wygrywamy, czy przegrywamy, dopóki nie zakończymy gry. Dlatego nie możemy powiedzieć, czy dany ruch sam w sobie jest dobry czy nie - otrzymujemy nagrodę dopiero na końcu gry. Naszym celem jest zaprojektowanie algorytmów, które pozwolą nam trenować model w warunkach niepewności. Poznamy jeden algorytm RL zwany **Q-learning**.

## Lekcje

1. [Wprowadzenie do uczenia ze wzmocnieniem i Q-Learning](1-QLearning/README.md)
2. [Korzystanie z symulowanego środowiska gym](2-Gym/README.md)

## Podziękowania

"Wprowadzenie do uczenia ze wzmocnieniem" zostało napisane z ♥️ przez [Dmitry Soshnikov](http://soshnikov.com)

---

**Zastrzeżenie**:  
Ten dokument został przetłumaczony za pomocą usługi tłumaczeniowej AI [Co-op Translator](https://github.com/Azure/co-op-translator). Chociaż dokładamy wszelkich starań, aby tłumaczenie było precyzyjne, prosimy pamiętać, że automatyczne tłumaczenia mogą zawierać błędy lub nieścisłości. Oryginalny dokument w jego rodzimym języku powinien być uznawany za źródło autorytatywne. W przypadku informacji o kluczowym znaczeniu zaleca się skorzystanie z profesjonalnego tłumaczenia przez człowieka. Nie ponosimy odpowiedzialności za jakiekolwiek nieporozumienia lub błędne interpretacje wynikające z użycia tego tłumaczenia.