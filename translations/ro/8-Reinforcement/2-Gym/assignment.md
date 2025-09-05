<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1f2b7441745eb52e25745423b247016b",
  "translation_date": "2025-09-05T16:48:33+00:00",
  "source_file": "8-Reinforcement/2-Gym/assignment.md",
  "language_code": "ro"
}
-->
# Antrenează Mountain Car

[OpenAI Gym](http://gym.openai.com) a fost conceput astfel încât toate mediile să ofere aceeași API - adică aceleași metode `reset`, `step` și `render`, și aceleași abstracții ale **spațiului de acțiune** și **spațiului de observație**. Astfel, ar trebui să fie posibil să adaptăm aceleași algoritmi de învățare prin întărire la diferite medii cu modificări minime de cod.

## Un Mediu Mountain Car

[Mediul Mountain Car](https://gym.openai.com/envs/MountainCar-v0/) conține o mașină blocată într-o vale:

Scopul este să ieși din vale și să capturezi steagul, efectuând la fiecare pas una dintre următoarele acțiuni:

| Valoare | Semnificație |
|---|---|
| 0 | Accelerează spre stânga |
| 1 | Nu accelerează |
| 2 | Accelerează spre dreapta |

Principala dificultate a acestei probleme este, totuși, că motorul mașinii nu este suficient de puternic pentru a urca muntele dintr-o singură încercare. Prin urmare, singura modalitate de a reuși este să conduci înainte și înapoi pentru a acumula impuls.

Spațiul de observație constă doar din două valori:

| Nr. | Observație  | Min | Max |
|-----|--------------|-----|-----|
|  0  | Poziția mașinii | -1.2| 0.6 |
|  1  | Viteza mașinii | -0.07 | 0.07 |

Sistemul de recompense pentru Mountain Car este destul de complicat:

 * O recompensă de 0 este acordată dacă agentul ajunge la steag (poziția = 0.5) în vârful muntelui.
 * O recompensă de -1 este acordată dacă poziția agentului este mai mică de 0.5.

Episodul se termină dacă poziția mașinii este mai mare de 0.5 sau dacă lungimea episodului depășește 200.
## Instrucțiuni

Adaptează algoritmul nostru de învățare prin întărire pentru a rezolva problema Mountain Car. Începe cu codul existent din [notebook.ipynb](../../../../8-Reinforcement/2-Gym/notebook.ipynb), înlocuiește mediul, modifică funcțiile de discretizare a stării și încearcă să faci algoritmul existent să se antreneze cu modificări minime de cod. Optimizează rezultatul ajustând hiperparametrii.

> **Notă**: Ajustarea hiperparametrilor este probabil necesară pentru ca algoritmul să convergă. 
## Criterii de evaluare

| Criteriu | Exemplu | Adecvat | Necesită îmbunătățiri |
| -------- | --------- | -------- | ----------------- |
|          | Algoritmul Q-Learning este adaptat cu succes din exemplul CartPole, cu modificări minime de cod, și este capabil să rezolve problema capturării steagului în mai puțin de 200 de pași. | Un nou algoritm Q-Learning a fost adoptat de pe Internet, dar este bine documentat; sau algoritmul existent a fost adoptat, dar nu atinge rezultatele dorite | Studentul nu a reușit să adopte cu succes niciun algoritm, dar a făcut pași substanțiali spre soluție (a implementat discretizarea stării, structura de date Q-Table etc.) |

---

**Declinare de responsabilitate**:  
Acest document a fost tradus folosind serviciul de traducere AI [Co-op Translator](https://github.com/Azure/co-op-translator). Deși ne străduim să asigurăm acuratețea, vă rugăm să fiți conștienți că traducerile automate pot conține erori sau inexactități. Documentul original în limba sa natală ar trebui considerat sursa autoritară. Pentru informații critice, se recomandă traducerea profesională realizată de un specialist uman. Nu ne asumăm responsabilitatea pentru eventualele neînțelegeri sau interpretări greșite care pot apărea din utilizarea acestei traduceri.