<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "68394b2102d3503882e5e914bd0ff5c1",
  "translation_date": "2025-09-05T16:43:33+00:00",
  "source_file": "8-Reinforcement/1-QLearning/assignment.md",
  "language_code": "sk"
}
-->
# Realistickejší svet

V našej situácii sa Peter mohol pohybovať takmer bez toho, aby sa unavil alebo vyhladol. V realistickejšom svete si musí občas sadnúť a oddýchnuť si, a tiež sa najesť. Urobme náš svet realistickejším zavedením nasledujúcich pravidiel:

1. Pri presune z jedného miesta na druhé Peter stráca **energiu** a získava **únavu**.
2. Peter môže získať viac energie jedením jabĺk.
3. Peter sa môže zbaviť únavy odpočinkom pod stromom alebo na tráve (t.j. vstúpením na políčko s umiestneným stromom alebo trávou - zelené pole).
4. Peter musí nájsť a zabiť vlka.
5. Aby Peter dokázal zabiť vlka, musí mať určité úrovne energie a únavy, inak prehrá boj.

## Pokyny

Použite pôvodný [notebook.ipynb](../../../../8-Reinforcement/1-QLearning/notebook.ipynb) ako východiskový bod pre vaše riešenie.

Upravte funkciu odmeny podľa pravidiel hry, spustite algoritmus posilneného učenia na naučenie najlepšej stratégie na výhru v hre a porovnajte výsledky náhodného pohybu s vaším algoritmom z hľadiska počtu vyhraných a prehratých hier.

> **Note**: Vo vašom novom svete je stav zložitejší a okrem pozície človeka zahŕňa aj úrovne únavy a energie. Môžete sa rozhodnúť reprezentovať stav ako n-ticu (Board, energy, fatigue), alebo definovať triedu pre stav (môžete ju tiež odvodiť z `Board`), alebo dokonca upraviť pôvodnú triedu `Board` v súbore [rlboard.py](../../../../8-Reinforcement/1-QLearning/rlboard.py).

Vo vašom riešení prosím ponechajte kód zodpovedný za stratégiu náhodného pohybu a na konci porovnajte výsledky vášho algoritmu s náhodným pohybom.

> **Note**: Možno budete musieť upraviť hyperparametre, aby to fungovalo, najmä počet epoch. Keďže úspech v hre (boj s vlkom) je zriedkavá udalosť, môžete očakávať oveľa dlhší čas trénovania.

## Hodnotenie

| Kritérium | Vynikajúce                                                                                                                                                                                             | Dostatočné                                                                                                                                                                              | Potrebuje zlepšenie                                                                                                                        |
| --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
|           | Notebook obsahuje definíciu nových pravidiel sveta, algoritmus Q-Learning a niektoré textové vysvetlenia. Q-Learning dokáže výrazne zlepšiť výsledky v porovnaní s náhodným pohybom.                   | Notebook je prezentovaný, Q-Learning je implementovaný a zlepšuje výsledky v porovnaní s náhodným pohybom, ale nie výrazne; alebo notebook je slabo zdokumentovaný a kód nie je dobre štruktúrovaný | Urobil sa pokus o redefinovanie pravidiel sveta, ale algoritmus Q-Learning nefunguje, alebo funkcia odmeny nie je úplne definovaná.         |

---

**Upozornenie**:  
Tento dokument bol preložený pomocou služby AI prekladu [Co-op Translator](https://github.com/Azure/co-op-translator). Aj keď sa snažíme o presnosť, prosím, berte na vedomie, že automatizované preklady môžu obsahovať chyby alebo nepresnosti. Pôvodný dokument v jeho pôvodnom jazyku by mal byť považovaný za autoritatívny zdroj. Pre kritické informácie sa odporúča profesionálny ľudský preklad. Nie sme zodpovední za akékoľvek nedorozumenia alebo nesprávne interpretácie vyplývajúce z použitia tohto prekladu.