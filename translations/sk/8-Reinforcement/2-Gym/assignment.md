<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1f2b7441745eb52e25745423b247016b",
  "translation_date": "2025-09-05T16:48:19+00:00",
  "source_file": "8-Reinforcement/2-Gym/assignment.md",
  "language_code": "sk"
}
-->
# Trénovanie Mountain Car

[OpenAI Gym](http://gym.openai.com) je navrhnutý tak, že všetky prostredia poskytujú rovnaké API - teda rovnaké metódy `reset`, `step` a `render`, a rovnaké abstrakcie **akčného priestoru** a **pozorovacieho priestoru**. Preto by malo byť možné prispôsobiť rovnaké algoritmy posilneného učenia rôznym prostrediam s minimálnymi zmenami kódu.

## Prostredie Mountain Car

[Prostredie Mountain Car](https://gym.openai.com/envs/MountainCar-v0/) obsahuje auto uviaznuté v údolí:

Cieľom je dostať sa z údolia a zachytiť vlajku, pričom na každom kroku vykonáte jednu z nasledujúcich akcií:

| Hodnota | Význam |
|---|---|
| 0 | Zrýchliť doľava |
| 1 | Nezrýchľovať |
| 2 | Zrýchliť doprava |

Hlavný trik tohto problému však spočíva v tom, že motor auta nie je dostatočne silný na to, aby vyšiel na horu na jeden pokus. Jediný spôsob, ako uspieť, je jazdiť tam a späť, aby sa získala hybnosť.

Pozorovací priestor pozostáva len z dvoch hodnôt:

| Číslo | Pozorovanie  | Min | Max |
|-----|--------------|-----|-----|
|  0  | Poloha auta  | -1.2| 0.6 |
|  1  | Rýchlosť auta | -0.07 | 0.07 |

Systém odmien pre Mountain Car je pomerne zložitý:

 * Odmena 0 sa udeľuje, ak agent dosiahne vlajku (poloha = 0.5) na vrchole hory.
 * Odmena -1 sa udeľuje, ak je poloha agenta menšia ako 0.5.

Epizóda sa ukončí, ak je poloha auta väčšia ako 0.5, alebo ak dĺžka epizódy presiahne 200.
## Pokyny

Prispôsobte náš algoritmus posilneného učenia na riešenie problému Mountain Car. Začnite s existujúcim kódom [notebook.ipynb](../../../../8-Reinforcement/2-Gym/notebook.ipynb), nahraďte nové prostredie, zmeňte funkcie na diskretizáciu stavu a pokúste sa upraviť existujúci algoritmus tak, aby sa trénoval s minimálnymi úpravami kódu. Optimalizujte výsledok úpravou hyperparametrov.

> **Poznámka**: Úprava hyperparametrov bude pravdepodobne potrebná na dosiahnutie konvergencie algoritmu. 
## Hodnotenie

| Kritérium | Vynikajúce | Dostatočné | Potrebuje zlepšenie |
| --------- | ---------- | ---------- | ------------------- |
|          | Algoritmus Q-Learning je úspešne prispôsobený z príkladu CartPole s minimálnymi úpravami kódu, ktorý dokáže vyriešiť problém zachytenia vlajky do 200 krokov. | Bol prijatý nový algoritmus Q-Learning z internetu, ale je dobre zdokumentovaný; alebo bol prijatý existujúci algoritmus, ale nedosahuje požadované výsledky. | Študent nebol schopný úspešne prijať žiadny algoritmus, ale urobil podstatné kroky k riešeniu (implementoval diskretizáciu stavu, dátovú štruktúru Q-Table, atď.) |

---

**Upozornenie**:  
Tento dokument bol preložený pomocou služby AI prekladu [Co-op Translator](https://github.com/Azure/co-op-translator). Hoci sa snažíme o presnosť, prosím, berte na vedomie, že automatizované preklady môžu obsahovať chyby alebo nepresnosti. Pôvodný dokument v jeho pôvodnom jazyku by mal byť považovaný za autoritatívny zdroj. Pre kritické informácie sa odporúča profesionálny ľudský preklad. Nenesieme zodpovednosť za akékoľvek nedorozumenia alebo nesprávne interpretácie vyplývajúce z použitia tohto prekladu.