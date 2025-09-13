<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "20ca019012b1725de956681d036d8b18",
  "translation_date": "2025-09-05T13:31:55+00:00",
  "source_file": "8-Reinforcement/README.md",
  "language_code": "hr"
}
-->
# Uvod u učenje pojačanjem

Učenje pojačanjem, RL, smatra se jednim od osnovnih paradigmi strojnog učenja, uz nadzirano i nenadzirano učenje. RL se bavi donošenjem odluka: donošenjem ispravnih odluka ili barem učenjem iz njih.

Zamislite da imate simulirano okruženje poput burze. Što se događa ako uvedete određenu regulaciju? Ima li to pozitivan ili negativan učinak? Ako se dogodi nešto negativno, trebate uzeti tu _negativnu povratnu informaciju_, učiti iz nje i promijeniti smjer. Ako je ishod pozitivan, trebate graditi na toj _pozitivnoj povratnoj informaciji_.

![peter i vuk](../../../8-Reinforcement/images/peter.png)

> Peter i njegovi prijatelji moraju pobjeći od gladnog vuka! Slika: [Jen Looper](https://twitter.com/jenlooper)

## Regionalna tema: Peter i vuk (Rusija)

[Peter i vuk](https://en.wikipedia.org/wiki/Peter_and_the_Wolf) je glazbena bajka koju je napisao ruski skladatelj [Sergej Prokofjev](https://en.wikipedia.org/wiki/Sergei_Prokofiev). To je priča o mladom pioniru Peteru, koji hrabro izlazi iz svoje kuće na šumsku čistinu kako bi ulovio vuka. U ovom dijelu ćemo trenirati algoritme strojnog učenja koji će pomoći Peteru:

- **Istražiti** okolno područje i izgraditi optimalnu kartu navigacije
- **Naučiti** kako koristiti skateboard i održavati ravnotežu na njemu kako bi se brže kretao.

[![Peter i vuk](https://img.youtube.com/vi/Fmi5zHg4QSM/0.jpg)](https://www.youtube.com/watch?v=Fmi5zHg4QSM)

> 🎥 Kliknite na sliku iznad kako biste poslušali "Peter i vuk" od Prokofjeva

## Učenje pojačanjem

U prethodnim dijelovima vidjeli ste dva primjera problema strojnog učenja:

- **Nadzirano učenje**, gdje imamo skupove podataka koji sugeriraju primjere rješenja problema koji želimo riješiti. [Klasifikacija](../4-Classification/README.md) i [regresija](../2-Regression/README.md) su zadaci nadziranog učenja.
- **Nadzirano učenje**, u kojem nemamo označene podatke za treniranje. Glavni primjer nenadziranog učenja je [Grupiranje](../5-Clustering/README.md).

U ovom dijelu ćemo vas upoznati s novom vrstom problema učenja koji ne zahtijeva označene podatke za treniranje. Postoji nekoliko vrsta takvih problema:

- **[Polunadzirano učenje](https://wikipedia.org/wiki/Semi-supervised_learning)**, gdje imamo puno neoznačenih podataka koji se mogu koristiti za predtreniranje modela.
- **[Učenje pojačanjem](https://wikipedia.org/wiki/Reinforcement_learning)**, u kojem agent uči kako se ponašati izvođenjem eksperimenata u nekom simuliranom okruženju.

### Primjer - računalna igra

Pretpostavimo da želite naučiti računalo kako igrati igru, poput šaha ili [Super Maria](https://wikipedia.org/wiki/Super_Mario). Da bi računalo moglo igrati igru, potrebno je predvidjeti koji potez napraviti u svakom stanju igre. Iako se to može činiti kao problem klasifikacije, nije - jer nemamo skup podataka sa stanjima i odgovarajućim akcijama. Iako možda imamo neke podatke poput postojećih šahovskih partija ili snimki igrača koji igraju Super Maria, vjerojatno ti podaci neće dovoljno pokriti veliki broj mogućih stanja.

Umjesto traženja postojećih podataka o igri, **učenje pojačanjem** (RL) temelji se na ideji da *računalo igra* mnogo puta i promatra rezultat. Dakle, za primjenu učenja pojačanjem potrebne su nam dvije stvari:

- **Okruženje** i **simulator** koji nam omogućuju da igru igramo mnogo puta. Ovaj simulator bi definirao sva pravila igre, kao i moguća stanja i akcije.

- **Funkcija nagrade**, koja bi nam govorila koliko smo dobro igrali tijekom svakog poteza ili igre.

Glavna razlika između drugih vrsta strojnog učenja i RL-a je ta što u RL-u obično ne znamo hoćemo li pobijediti ili izgubiti dok ne završimo igru. Dakle, ne možemo reći je li određeni potez sam po sebi dobar ili ne - nagradu dobivamo tek na kraju igre. Naš cilj je osmisliti algoritme koji će nam omogućiti treniranje modela u uvjetima nesigurnosti. Naučit ćemo o jednom RL algoritmu zvanom **Q-učenje**.

## Lekcije

1. [Uvod u učenje pojačanjem i Q-učenje](1-QLearning/README.md)
2. [Korištenje simulacijskog okruženja Gym](2-Gym/README.md)

## Zasluge

"Uvod u učenje pojačanjem" napisano je s ♥️ od strane [Dmitry Soshnikov](http://soshnikov.com)

---

**Odricanje od odgovornosti**:  
Ovaj dokument je preveden pomoću AI usluge za prevođenje [Co-op Translator](https://github.com/Azure/co-op-translator). Iako nastojimo osigurati točnost, imajte na umu da automatski prijevodi mogu sadržavati pogreške ili netočnosti. Izvorni dokument na izvornom jeziku treba smatrati autoritativnim izvorom. Za ključne informacije preporučuje se profesionalni prijevod od strane čovjeka. Ne preuzimamo odgovornost za bilo kakva nesporazuma ili pogrešna tumačenja koja proizlaze iz korištenja ovog prijevoda.