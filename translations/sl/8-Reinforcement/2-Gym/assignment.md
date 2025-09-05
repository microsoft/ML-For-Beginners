<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1f2b7441745eb52e25745423b247016b",
  "translation_date": "2025-09-05T13:50:53+00:00",
  "source_file": "8-Reinforcement/2-Gym/assignment.md",
  "language_code": "sl"
}
-->
# Treniraj Mountain Car

[OpenAI Gym](http://gym.openai.com) je zasnovan tako, da vsa okolja zagotavljajo enak API - tj. iste metode `reset`, `step` in `render`, ter enake abstrakcije **akcijskega prostora** in **prostora opazovanj**. Zato bi moralo biti mogoče prilagoditi iste algoritme za krepitev učenja različnim okoljem z minimalnimi spremembami kode.

## Okolje Mountain Car

[Okolje Mountain Car](https://gym.openai.com/envs/MountainCar-v0/) vsebuje avto, ki je obtičal v dolini:

Cilj je priti iz doline in ujeti zastavo, pri čemer na vsakem koraku izvedemo eno od naslednjih akcij:

| Vrednost | Pomen |
|---|---|
| 0 | Pospeši v levo |
| 1 | Ne pospešuj |
| 2 | Pospeši v desno |

Glavna težava tega problema pa je, da motor avtomobila ni dovolj močan, da bi premagal goro v enem poskusu. Zato je edini način za uspeh vožnja naprej in nazaj, da se pridobi zagon.

Prostor opazovanj vsebuje le dve vrednosti:

| Št. | Opazovanje  | Min | Max |
|-----|--------------|-----|-----|
|  0  | Položaj avtomobila | -1.2| 0.6 |
|  1  | Hitrost avtomobila | -0.07 | 0.07 |

Sistem nagrajevanja za Mountain Car je precej zahteven:

 * Nagrada 0 je podeljena, če agent doseže zastavo (položaj = 0.5) na vrhu gore.
 * Nagrada -1 je podeljena, če je položaj agenta manjši od 0.5.

Epizoda se zaključi, če je položaj avtomobila večji od 0.5 ali če dolžina epizode presega 200.
## Navodila

Prilagodite naš algoritem za krepitev učenja, da rešite problem Mountain Car. Začnite z obstoječo kodo [notebook.ipynb](../../../../8-Reinforcement/2-Gym/notebook.ipynb), zamenjajte okolje, spremenite funkcije za diskretizacijo stanja in poskusite obstoječi algoritem usposobiti z minimalnimi spremembami kode. Optimizirajte rezultat z nastavitvijo hiperparametrov.

> **Opomba**: Nastavitev hiperparametrov bo verjetno potrebna, da se algoritem konvergira.
## Merila

| Merila | Odlično | Zadostno | Potrebno izboljšanje |
| -------- | --------- | -------- | ----------------- |
|          | Algoritem Q-Learning je uspešno prilagojen iz primera CartPole z minimalnimi spremembami kode, ki je sposoben rešiti problem ujetja zastave v manj kot 200 korakih. | Nov algoritem Q-Learning je bil prevzet z interneta, vendar je dobro dokumentiran; ali obstoječi algoritem prilagojen, vendar ne dosega želenih rezultatov. | Študent ni uspel uspešno prilagoditi nobenega algoritma, vendar je naredil pomembne korake proti rešitvi (implementiral diskretizacijo stanja, podatkovno strukturo Q-Table itd.) |

---

**Omejitev odgovornosti**:  
Ta dokument je bil preveden z uporabo storitve za strojno prevajanje [Co-op Translator](https://github.com/Azure/co-op-translator). Čeprav si prizadevamo za natančnost, vas opozarjamo, da lahko avtomatizirani prevodi vsebujejo napake ali netočnosti. Izvirni dokument v njegovem izvirnem jeziku je treba obravnavati kot avtoritativni vir. Za ključne informacije priporočamo strokovno človeško prevajanje. Ne prevzemamo odgovornosti za morebitna nesporazumevanja ali napačne razlage, ki izhajajo iz uporabe tega prevoda.