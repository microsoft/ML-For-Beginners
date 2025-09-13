<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "68394b2102d3503882e5e914bd0ff5c1",
  "translation_date": "2025-09-05T01:12:16+00:00",
  "source_file": "8-Reinforcement/1-QLearning/assignment.md",
  "language_code": "cs"
}
-->
# Realističtější svět

V naší situaci se Peter mohl pohybovat téměř bez únavy nebo hladu. V realističtějším světě by si musel čas od času sednout a odpočinout si, stejně jako se najíst. Udělejme náš svět realističtější implementací následujících pravidel:

1. Při pohybu z jednoho místa na druhé Peter ztrácí **energii** a získává **únavu**.
2. Peter může získat více energie tím, že jí jablka.
3. Peter se může zbavit únavy odpočinkem pod stromem nebo na trávě (tj. vstupem na pole s umístěním stromu nebo trávy - zelené pole).
4. Peter musí najít a zabít vlka.
5. Aby mohl Peter zabít vlka, musí mít určité úrovně energie a únavy, jinak prohraje bitvu.

## Instrukce

Použijte původní [notebook.ipynb](../../../../8-Reinforcement/1-QLearning/notebook.ipynb) jako výchozí bod pro vaše řešení.

Upravte výše uvedenou funkci odměny podle pravidel hry, spusťte algoritmus posilovaného učení, aby se naučil nejlepší strategii pro vítězství ve hře, a porovnejte výsledky náhodné procházky s vaším algoritmem z hlediska počtu vyhraných a prohraných her.

> **Note**: Ve vašem novém světě je stav složitější a kromě pozice člověka zahrnuje také úrovně únavy a energie. Můžete se rozhodnout reprezentovat stav jako n-tici (Board, energy, fatigue), nebo definovat třídu pro stav (můžete ji také odvodit z `Board`), nebo dokonce upravit původní třídu `Board` uvnitř [rlboard.py](../../../../8-Reinforcement/1-QLearning/rlboard.py).

Ve vašem řešení prosím ponechte kód odpovědný za strategii náhodné procházky a na konci porovnejte výsledky vašeho algoritmu s náhodnou procházkou.

> **Note**: Možná budete muset upravit hyperparametry, aby vše fungovalo, zejména počet epoch. Protože úspěch ve hře (boj s vlkem) je vzácná událost, můžete očekávat mnohem delší dobu trénování.

## Hodnocení

| Kritéria | Vynikající                                                                                                                                                                                             | Přiměřené                                                                                                                                                                               | Potřebuje zlepšení                                                                                                                         |
| -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
|          | Notebook je prezentován s definicí nových pravidel světa, algoritmem Q-Learning a některými textovými vysvětleními. Q-Learning dokáže výrazně zlepšit výsledky ve srovnání s náhodnou procházkou.       | Notebook je prezentován, Q-Learning je implementován a zlepšuje výsledky ve srovnání s náhodnou procházkou, ale ne výrazně; nebo je notebook špatně dokumentován a kód není dobře strukturován | Byly učiněny pokusy o redefinici pravidel světa, ale algoritmus Q-Learning nefunguje, nebo funkce odměny není plně definována.              |

---

**Prohlášení**:  
Tento dokument byl přeložen pomocí služby pro automatický překlad [Co-op Translator](https://github.com/Azure/co-op-translator). Přestože se snažíme o přesnost, mějte na paměti, že automatické překlady mohou obsahovat chyby nebo nepřesnosti. Původní dokument v jeho původním jazyce by měl být považován za autoritativní zdroj. Pro kritické informace doporučujeme profesionální lidský překlad. Neodpovídáme za žádná nedorozumění nebo nesprávné interpretace vyplývající z použití tohoto překladu.