<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "20ca019012b1725de956681d036d8b18",
  "translation_date": "2025-09-05T16:36:06+00:00",
  "source_file": "8-Reinforcement/README.md",
  "language_code": "ro"
}
-->
# Introducere în învățarea prin întărire

Învățarea prin întărire, RL, este considerată unul dintre paradigmele de bază ale învățării automate, alături de învățarea supravegheată și cea nesupravegheată. RL se concentrează pe luarea deciziilor: luarea deciziilor corecte sau cel puțin învățarea din ele.

Imaginează-ți că ai un mediu simulat, cum ar fi piața bursieră. Ce se întâmplă dacă impui o anumită reglementare? Are un efect pozitiv sau negativ? Dacă se întâmplă ceva negativ, trebuie să iei acest _întărire negativă_, să înveți din ea și să schimbi direcția. Dacă rezultatul este pozitiv, trebuie să construiești pe baza acestui _întărire pozitivă_.

![peter și lupul](../../../8-Reinforcement/images/peter.png)

> Peter și prietenii săi trebuie să scape de lupul flămând! Imagine de [Jen Looper](https://twitter.com/jenlooper)

## Subiect regional: Peter și Lupul (Rusia)

[Peter și Lupul](https://en.wikipedia.org/wiki/Peter_and_the_Wolf) este o poveste muzicală scrisă de compozitorul rus [Sergei Prokofiev](https://en.wikipedia.org/wiki/Sergei_Prokofiev). Este povestea unui tânăr pionier, Peter, care se aventurează curajos din casa sa spre poiana din pădure pentru a urmări lupul. În această secțiune, vom antrena algoritmi de învățare automată care îl vor ajuta pe Peter:

- **Să exploreze** zona înconjurătoare și să construiască o hartă optimă de navigare
- **Să învețe** cum să folosească un skateboard și să își mențină echilibrul pe el, pentru a se deplasa mai rapid.

[![Peter și Lupul](https://img.youtube.com/vi/Fmi5zHg4QSM/0.jpg)](https://www.youtube.com/watch?v=Fmi5zHg4QSM)

> 🎥 Click pe imaginea de mai sus pentru a asculta Peter și Lupul de Prokofiev

## Învățarea prin întărire

În secțiunile anterioare, ai văzut două exemple de probleme de învățare automată:

- **Supravegheată**, unde avem seturi de date care sugerează soluții de exemplu pentru problema pe care dorim să o rezolvăm. [Clasificarea](../4-Classification/README.md) și [regresia](../2-Regression/README.md) sunt sarcini de învățare supravegheată.
- **Nesupravegheată**, în care nu avem date de antrenament etichetate. Principalul exemplu de învățare nesupravegheată este [Clustering](../5-Clustering/README.md).

În această secțiune, îți vom prezenta un nou tip de problemă de învățare care nu necesită date de antrenament etichetate. Există mai multe tipuri de astfel de probleme:

- **[Învățare semi-supravegheată](https://wikipedia.org/wiki/Semi-supervised_learning)**, unde avem o cantitate mare de date neetichetate care pot fi folosite pentru pre-antrenarea modelului.
- **[Învățare prin întărire](https://wikipedia.org/wiki/Reinforcement_learning)**, în care un agent învață cum să se comporte prin efectuarea de experimente într-un mediu simulat.

### Exemplu - joc pe calculator

Să presupunem că vrei să înveți un calculator să joace un joc, cum ar fi șahul sau [Super Mario](https://wikipedia.org/wiki/Super_Mario). Pentru ca calculatorul să joace un joc, trebuie să prezică ce mișcare să facă în fiecare dintre stările jocului. Deși acest lucru poate părea o problemă de clasificare, nu este - deoarece nu avem un set de date cu stări și acțiuni corespunzătoare. Deși putem avea unele date, cum ar fi meciuri de șah existente sau înregistrări ale jucătorilor care joacă Super Mario, este probabil ca aceste date să nu acopere suficient de bine un număr mare de stări posibile.

În loc să căutăm date existente despre joc, **Învățarea prin întărire** (RL) se bazează pe ideea de *a face calculatorul să joace* de multe ori și să observe rezultatul. Astfel, pentru a aplica Învățarea prin întărire, avem nevoie de două lucruri:

- **Un mediu** și **un simulator** care ne permit să jucăm un joc de multe ori. Acest simulator ar defini toate regulile jocului, precum și stările și acțiunile posibile.

- **O funcție de recompensă**, care ne-ar spune cât de bine ne-am descurcat în timpul fiecărei mișcări sau joc.

Principala diferență între alte tipuri de învățare automată și RL este că în RL, de obicei, nu știm dacă câștigăm sau pierdem până nu terminăm jocul. Astfel, nu putem spune dacă o anumită mișcare este bună sau nu - primim o recompensă doar la sfârșitul jocului. Iar scopul nostru este să proiectăm algoritmi care să ne permită să antrenăm un model în condiții de incertitudine. Vom învăța despre un algoritm RL numit **Q-learning**.

## Lecții

1. [Introducere în învățarea prin întărire și Q-Learning](1-QLearning/README.md)
2. [Utilizarea unui mediu de simulare gym](2-Gym/README.md)

## Credite

"Introducere în Învățarea prin Întărire" a fost scrisă cu ♥️ de [Dmitry Soshnikov](http://soshnikov.com)

---

**Declinare de responsabilitate**:  
Acest document a fost tradus folosind serviciul de traducere AI [Co-op Translator](https://github.com/Azure/co-op-translator). Deși ne străduim să asigurăm acuratețea, vă rugăm să rețineți că traducerile automate pot conține erori sau inexactități. Documentul original în limba sa natală ar trebui considerat sursa autoritară. Pentru informații critice, se recomandă traducerea profesională realizată de un specialist uman. Nu ne asumăm responsabilitatea pentru eventualele neînțelegeri sau interpretări greșite care pot apărea din utilizarea acestei traduceri.