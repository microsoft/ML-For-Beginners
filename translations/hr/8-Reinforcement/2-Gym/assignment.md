<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1f2b7441745eb52e25745423b247016b",
  "translation_date": "2025-09-05T13:50:41+00:00",
  "source_file": "8-Reinforcement/2-Gym/assignment.md",
  "language_code": "hr"
}
-->
# Treniranje Mountain Car

[OpenAI Gym](http://gym.openai.com) je dizajniran na način da svi okoliši pružaju isti API - tj. iste metode `reset`, `step` i `render`, te iste apstrakcije **prostora akcija** i **prostora opažanja**. Stoga bi trebalo biti moguće prilagoditi iste algoritme za učenje pojačanjem različitim okruženjima uz minimalne promjene koda.

## Okruženje Mountain Car

[Okruženje Mountain Car](https://gym.openai.com/envs/MountainCar-v0/) sadrži automobil zaglavljen u dolini:

Cilj je izaći iz doline i dohvatiti zastavu, izvodeći pri svakom koraku jednu od sljedećih akcija:

| Vrijednost | Značenje |
|---|---|
| 0 | Ubrzaj ulijevo |
| 1 | Ne ubrzavaj |
| 2 | Ubrzaj udesno |

Glavni trik ovog problema je, međutim, da motor automobila nije dovoljno snažan da prijeđe planinu u jednom pokušaju. Stoga je jedini način za uspjeh vožnja naprijed-nazad kako bi se stvorio zamah.

Prostor opažanja sastoji se od samo dvije vrijednosti:

| Br. | Opažanje  | Min | Max |
|-----|--------------|-----|-----|
|  0  | Pozicija automobila | -1.2| 0.6 |
|  1  | Brzina automobila | -0.07 | 0.07 |

Sustav nagrađivanja za Mountain Car je prilično izazovan:

 * Nagrada od 0 dodjeljuje se ako agent dosegne zastavu (pozicija = 0.5) na vrhu planine.
 * Nagrada od -1 dodjeljuje se ako je pozicija agenta manja od 0.5.

Epizoda završava ako je pozicija automobila veća od 0.5 ili ako duljina epizode premaši 200 koraka.

## Upute

Prilagodite naš algoritam za učenje pojačanjem kako biste riješili problem Mountain Car. Počnite s postojećim kodom iz [notebook.ipynb](../../../../8-Reinforcement/2-Gym/notebook.ipynb), zamijenite okruženje, promijenite funkcije za diskretizaciju stanja i pokušajte natjerati postojeći algoritam da trenira uz minimalne izmjene koda. Optimizirajte rezultat podešavanjem hiperparametara.

> **Napomena**: Podešavanje hiperparametara vjerojatno će biti potrebno kako bi algoritam konvergirao.

## Rubrika

| Kriterij | Izvrsno | Zadovoljavajuće | Potrebno poboljšanje |
| -------- | --------- | -------- | ----------------- |
|          | Algoritam Q-Learning uspješno je prilagođen iz primjera CartPole uz minimalne izmjene koda i sposoban je riješiti problem dohvaćanja zastave u manje od 200 koraka. | Novi algoritam Q-Learning preuzet je s interneta, ali je dobro dokumentiran; ili je postojeći algoritam prilagođen, ali ne postiže željene rezultate. | Student nije uspio uspješno prilagoditi nijedan algoritam, ali je napravio značajne korake prema rješenju (implementirao diskretizaciju stanja, strukturu podataka Q-Tablice itd.) |

---

**Odricanje od odgovornosti**:  
Ovaj dokument je preveden korištenjem AI usluge za prevođenje [Co-op Translator](https://github.com/Azure/co-op-translator). Iako nastojimo osigurati točnost, imajte na umu da automatski prijevodi mogu sadržavati pogreške ili netočnosti. Izvorni dokument na izvornom jeziku treba smatrati mjerodavnim izvorom. Za ključne informacije preporučuje se profesionalni prijevod od strane stručnjaka. Ne preuzimamo odgovornost za bilo kakve nesporazume ili pogrešne interpretacije proizašle iz korištenja ovog prijevoda.