<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "68394b2102d3503882e5e914bd0ff5c1",
  "translation_date": "2025-09-05T16:43:57+00:00",
  "source_file": "8-Reinforcement/1-QLearning/assignment.md",
  "language_code": "ro"
}
-->
# O Lume Mai Realistă

În situația noastră, Peter a putut să se deplaseze aproape fără să obosească sau să îi fie foame. Într-o lume mai realistă, el trebuie să se așeze și să se odihnească din când în când, și de asemenea să se hrănească. Hai să facem lumea noastră mai realistă, implementând următoarele reguli:

1. Prin deplasarea dintr-un loc în altul, Peter pierde **energie** și acumulează **oboseală**.
2. Peter poate câștiga mai multă energie mâncând mere.
3. Peter poate scăpa de oboseală odihnindu-se sub copac sau pe iarbă (adică mergând într-o locație de pe tablă care are un copac sau iarbă - câmp verde).
4. Peter trebuie să găsească și să omoare lupul.
5. Pentru a omorî lupul, Peter trebuie să aibă anumite niveluri de energie și oboseală, altfel pierde lupta.

## Instrucțiuni

Folosește notebook-ul original [notebook.ipynb](../../../../8-Reinforcement/1-QLearning/notebook.ipynb) ca punct de plecare pentru soluția ta.

Modifică funcția de recompensă de mai sus conform regulilor jocului, rulează algoritmul de învățare prin întărire pentru a învăța cea mai bună strategie de câștig al jocului și compară rezultatele plimbării aleatorii cu algoritmul tău în termeni de număr de jocuri câștigate și pierdute.

> **Note**: În noua ta lume, starea este mai complexă și, pe lângă poziția umană, include și nivelurile de oboseală și energie. Poți alege să reprezinți starea ca un tuplu (Board, energie, oboseală), să definești o clasă pentru stare (poți de asemenea să o derivezi din `Board`), sau chiar să modifici clasa originală `Board` din [rlboard.py](../../../../8-Reinforcement/1-QLearning/rlboard.py).

În soluția ta, te rog să păstrezi codul responsabil pentru strategia de plimbare aleatorie și să compari rezultatele algoritmului tău cu plimbarea aleatorie la final.

> **Note**: Este posibil să fie nevoie să ajustezi hiperparametrii pentru ca algoritmul să funcționeze, în special numărul de epoci. Deoarece succesul jocului (lupta cu lupul) este un eveniment rar, te poți aștepta la un timp de antrenament mult mai lung.

## Criterii de Evaluare

| Criterii | Exemplară                                                                                                                                                                                             | Adecvată                                                                                                                                                                                | Necesită Îmbunătățiri                                                                                                                      |
| -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
|          | Un notebook este prezentat cu definirea noilor reguli ale lumii, algoritmul Q-Learning și câteva explicații textuale. Q-Learning reușește să îmbunătățească semnificativ rezultatele comparativ cu plimbarea aleatorie. | Notebook-ul este prezentat, Q-Learning este implementat și îmbunătățește rezultatele comparativ cu plimbarea aleatorie, dar nu semnificativ; sau notebook-ul este slab documentat și codul nu este bine structurat. | Se fac unele încercări de a redefini regulile lumii, dar algoritmul Q-Learning nu funcționează sau funcția de recompensă nu este complet definită. |

---

**Declinare de responsabilitate**:  
Acest document a fost tradus folosind serviciul de traducere AI [Co-op Translator](https://github.com/Azure/co-op-translator). Deși ne străduim să asigurăm acuratețea, vă rugăm să rețineți că traducerile automate pot conține erori sau inexactități. Documentul original în limba sa natală ar trebui considerat sursa autoritară. Pentru informații critice, se recomandă traducerea profesională realizată de un specialist uman. Nu ne asumăm responsabilitatea pentru eventualele neînțelegeri sau interpretări greșite care pot apărea din utilizarea acestei traduceri.