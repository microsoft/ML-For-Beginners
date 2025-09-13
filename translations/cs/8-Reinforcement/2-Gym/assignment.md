<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1f2b7441745eb52e25745423b247016b",
  "translation_date": "2025-09-05T01:18:24+00:00",
  "source_file": "8-Reinforcement/2-Gym/assignment.md",
  "language_code": "cs"
}
-->
# Trénink Mountain Car

[OpenAI Gym](http://gym.openai.com) byl navržen tak, aby všechna prostředí poskytovala stejnou API - tj. stejné metody `reset`, `step` a `render`, a stejné abstrakce **akčního prostoru** a **pozorovacího prostoru**. Díky tomu by mělo být možné přizpůsobit stejné algoritmy pro posilované učení různým prostředím s minimálními změnami kódu.

## Prostředí Mountain Car

[Prostředí Mountain Car](https://gym.openai.com/envs/MountainCar-v0/) obsahuje auto uvízlé v údolí:

Cílem je dostat se z údolí a získat vlajku, přičemž v každém kroku lze provést jednu z následujících akcí:

| Hodnota | Význam |
|---|---|
| 0 | Zrychlit doleva |
| 1 | Nezrychlovat |
| 2 | Zrychlit doprava |

Hlavní trik tohoto problému však spočívá v tom, že motor auta není dostatečně silný na to, aby vyjel na horu na jeden pokus. Jediný způsob, jak uspět, je jezdit tam a zpět, aby se nashromáždila hybnost.

Pozorovací prostor obsahuje pouze dvě hodnoty:

| Číslo | Pozorování  | Min | Max |
|-----|--------------|-----|-----|
|  0  | Pozice auta  | -1.2| 0.6 |
|  1  | Rychlost auta | -0.07 | 0.07 |

Systém odměn pro Mountain Car je poměrně složitý:

 * Odměna 0 je udělena, pokud agent dosáhne vlajky (pozice = 0.5) na vrcholu hory.
 * Odměna -1 je udělena, pokud je pozice agenta menší než 0.5.

Epizoda končí, pokud je pozice auta větší než 0.5, nebo pokud délka epizody přesáhne 200 kroků.

## Pokyny

Přizpůsobte náš algoritmus pro posilované učení k vyřešení problému Mountain Car. Začněte s existujícím kódem [notebook.ipynb](../../../../8-Reinforcement/2-Gym/notebook.ipynb), nahraďte nové prostředí, změňte funkce pro diskretizaci stavu a pokuste se upravit existující algoritmus tak, aby se trénoval s minimálními úpravami kódu. Optimalizujte výsledek úpravou hyperparametrů.

> **Note**: Úprava hyperparametrů bude pravděpodobně nutná, aby algoritmus konvergoval.

## Hodnocení

| Kritéria | Vynikající | Přiměřené | Potřebuje zlepšení |
| -------- | --------- | --------- | ------------------ |
|          | Algoritmus Q-Learning byl úspěšně přizpůsoben z příkladu CartPole s minimálními úpravami kódu a dokáže vyřešit problém získání vlajky do 200 kroků. | Nový algoritmus Q-Learning byl převzat z internetu, ale je dobře zdokumentován; nebo byl přizpůsoben existující algoritmus, ale nedosahuje požadovaných výsledků. | Student nebyl schopen úspěšně přizpůsobit žádný algoritmus, ale učinil podstatné kroky k řešení (implementoval diskretizaci stavu, datovou strukturu Q-Tabulky atd.). |

---

**Prohlášení**:  
Tento dokument byl přeložen pomocí služby AI pro překlady [Co-op Translator](https://github.com/Azure/co-op-translator). Ačkoli se snažíme o přesnost, mějte prosím na paměti, že automatizované překlady mohou obsahovat chyby nebo nepřesnosti. Původní dokument v jeho původním jazyce by měl být považován za autoritativní zdroj. Pro důležité informace se doporučuje profesionální lidský překlad. Neodpovídáme za žádná nedorozumění nebo nesprávné interpretace vyplývající z použití tohoto překladu.