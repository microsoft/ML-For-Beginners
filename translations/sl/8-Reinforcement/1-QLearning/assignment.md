<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "68394b2102d3503882e5e914bd0ff5c1",
  "translation_date": "2025-09-05T13:42:42+00:00",
  "source_file": "8-Reinforcement/1-QLearning/assignment.md",
  "language_code": "sl"
}
-->
# Bolj realističen svet

V naši situaciji se je Peter lahko premikal skoraj brez utrujenosti ali lakote. V bolj realističnem svetu se mora občasno ustaviti, da si odpočije, in se tudi nahraniti. Naredimo naš svet bolj realističen z uvedbo naslednjih pravil:

1. Pri premikanju iz enega kraja v drugega Peter izgublja **energijo** in pridobiva **utrujenost**.
2. Peter lahko pridobi več energije z uživanjem jabolk.
3. Peter se lahko znebi utrujenosti z počitkom pod drevesom ali na travi (tj. ko stopi na polje z drevesom ali travo - zeleno polje).
4. Peter mora najti in ubiti volka.
5. Da bi ubil volka, mora Peter imeti določene ravni energije in utrujenosti, sicer izgubi bitko.

## Navodila

Uporabite originalni zvezek [notebook.ipynb](../../../../8-Reinforcement/1-QLearning/notebook.ipynb) kot izhodišče za vašo rešitev.

Spremenite funkcijo nagrajevanja glede na pravila igre, zaženite algoritem za okrepitev učenja, da se naučite najboljše strategije za zmago v igri, in primerjajte rezultate naključnega sprehoda z vašim algoritmom glede na število zmag in porazov.

> **Note**: V vašem novem svetu je stanje bolj kompleksno in poleg človeške pozicije vključuje tudi ravni utrujenosti in energije. Stanje lahko predstavite kot nabor (Polje, energija, utrujenost) ali definirate razred za stanje (lahko ga tudi izpeljete iz `Board`), ali pa celo spremenite originalni razred `Board` znotraj [rlboard.py](../../../../8-Reinforcement/1-QLearning/rlboard.py).

V vaši rešitvi prosimo, da ohranite kodo, ki je odgovorna za strategijo naključnega sprehoda, in na koncu primerjajte rezultate vašega algoritma z naključnim sprehodom.

> **Note**: Morda boste morali prilagoditi hiperparametre, da bo delovalo, še posebej število epoh. Ker je uspeh igre (boj z volkom) redek dogodek, lahko pričakujete precej daljši čas učenja.

## Merila

| Merilo   | Odlično                                                                                                                                                                                               | Zadostno                                                                                                                                                                                | Potrebno izboljšanje                                                                                                                       |
| -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
|          | Predstavljen je zvezek z definicijo novih pravil sveta, algoritmom Q-Learning in nekaj besedilnimi razlagami. Q-Learning bistveno izboljša rezultate v primerjavi z naključnim sprehodom.              | Predstavljen je zvezek, Q-Learning je implementiran in izboljša rezultate v primerjavi z naključnim sprehodom, vendar ne bistveno; ali zvezek je slabo dokumentiran in koda ni dobro strukturirana | Narejen je poskus redefinicije pravil sveta, vendar algoritem Q-Learning ne deluje ali funkcija nagrajevanja ni v celoti definirana                                             |

---

**Omejitev odgovornosti**:  
Ta dokument je bil preveden z uporabo storitve za strojno prevajanje [Co-op Translator](https://github.com/Azure/co-op-translator). Čeprav si prizadevamo za natančnost, vas prosimo, da upoštevate, da lahko avtomatizirani prevodi vsebujejo napake ali netočnosti. Izvirni dokument v njegovem izvirnem jeziku je treba obravnavati kot avtoritativni vir. Za ključne informacije priporočamo strokovno človeško prevajanje. Ne prevzemamo odgovornosti za morebitna nesporazumevanja ali napačne razlage, ki izhajajo iz uporabe tega prevoda.