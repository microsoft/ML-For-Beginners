<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "7c077988328ebfe33b24d07945f16eca",
  "translation_date": "2025-09-05T15:26:15+00:00",
  "source_file": "2-Regression/2-Data/README.md",
  "language_code": "ro"
}
-->
# Construiește un model de regresie folosind Scikit-learn: pregătirea și vizualizarea datelor

![Infografic despre vizualizarea datelor](../../../../2-Regression/2-Data/images/data-visualization.png)

Infografic realizat de [Dasani Madipalli](https://twitter.com/dasani_decoded)

## [Chestionar înainte de lecție](https://ff-quizzes.netlify.app/en/ml/)

> ### [Această lecție este disponibilă și în R!](../../../../2-Regression/2-Data/solution/R/lesson_2.html)

## Introducere

Acum că ai configurat instrumentele necesare pentru a începe să construiești modele de învățare automată cu Scikit-learn, ești pregătit să începi să pui întrebări datelor tale. Pe măsură ce lucrezi cu date și aplici soluții de ML, este foarte important să înțelegi cum să pui întrebarea potrivită pentru a valorifica pe deplin potențialul setului tău de date.

În această lecție vei învăța:

- Cum să pregătești datele pentru construirea modelului.
- Cum să folosești Matplotlib pentru vizualizarea datelor.

## Punerea întrebării potrivite datelor tale

Întrebarea la care ai nevoie de răspuns va determina ce tip de algoritmi ML vei utiliza. Iar calitatea răspunsului pe care îl primești va depinde în mare măsură de natura datelor tale.

Aruncă o privire la [datele](https://github.com/microsoft/ML-For-Beginners/blob/main/2-Regression/data/US-pumpkins.csv) furnizate pentru această lecție. Poți deschide acest fișier .csv în VS Code. O privire rapidă arată imediat că există spații goale și un amestec de date text și numerice. Există, de asemenea, o coloană ciudată numită 'Package', unde datele sunt un amestec între 'sacks', 'bins' și alte valori. Datele, de fapt, sunt destul de dezordonate.

[![ML pentru începători - Cum să analizezi și să cureți un set de date](https://img.youtube.com/vi/5qGjczWTrDQ/0.jpg)](https://youtu.be/5qGjczWTrDQ "ML pentru începători - Cum să analizezi și să cureți un set de date")

> 🎥 Fă clic pe imaginea de mai sus pentru un scurt videoclip despre pregătirea datelor pentru această lecție.

De fapt, nu este foarte comun să primești un set de date complet pregătit pentru a crea un model ML direct. În această lecție, vei învăța cum să pregătești un set de date brut folosind biblioteci standard Python. De asemenea, vei învăța diverse tehnici pentru vizualizarea datelor.

## Studiu de caz: 'piața dovlecilor'

În acest folder vei găsi un fișier .csv în folderul rădăcină `data` numit [US-pumpkins.csv](https://github.com/microsoft/ML-For-Beginners/blob/main/2-Regression/data/US-pumpkins.csv), care include 1757 linii de date despre piața dovlecilor, grupate pe orașe. Acestea sunt date brute extrase din [Specialty Crops Terminal Markets Standard Reports](https://www.marketnews.usda.gov/mnp/fv-report-config-step1?type=termPrice) distribuite de Departamentul Agriculturii al Statelor Unite.

### Pregătirea datelor

Aceste date sunt în domeniul public. Ele pot fi descărcate în mai multe fișiere separate, pe orașe, de pe site-ul USDA. Pentru a evita prea multe fișiere separate, am concatenat toate datele pe orașe într-un singur fișier, astfel încât am _pregătit_ deja puțin datele. Acum, să aruncăm o privire mai atentă asupra datelor.

### Datele despre dovleci - concluzii inițiale

Ce observi despre aceste date? Ai văzut deja că există un amestec de text, numere, spații goale și valori ciudate pe care trebuie să le înțelegi.

Ce întrebare poți pune acestor date, folosind o tehnică de regresie? Ce zici de "Prezice prețul unui dovleac de vânzare într-o anumită lună"? Privind din nou datele, există câteva modificări pe care trebuie să le faci pentru a crea structura de date necesară pentru această sarcină.

## Exercițiu - analizează datele despre dovleci

Să folosim [Pandas](https://pandas.pydata.org/), (numele vine de la `Python Data Analysis`) un instrument foarte util pentru modelarea datelor, pentru a analiza și pregăti aceste date despre dovleci.

### Mai întâi, verifică datele lipsă

Mai întâi va trebui să iei măsuri pentru a verifica datele lipsă:

1. Convertește datele în format de lună (acestea sunt date din SUA, deci formatul este `MM/DD/YYYY`).
2. Extrage luna într-o coloană nouă.

Deschide fișierul _notebook.ipynb_ în Visual Studio Code și importă foaia de calcul într-un nou dataframe Pandas.

1. Folosește funcția `head()` pentru a vizualiza primele cinci rânduri.

    ```python
    import pandas as pd
    pumpkins = pd.read_csv('../data/US-pumpkins.csv')
    pumpkins.head()
    ```

    ✅ Ce funcție ai folosi pentru a vizualiza ultimele cinci rânduri?

1. Verifică dacă există date lipsă în dataframe-ul curent:

    ```python
    pumpkins.isnull().sum()
    ```

    Există date lipsă, dar poate că nu vor conta pentru sarcina de față.

1. Pentru a face dataframe-ul mai ușor de utilizat, selectează doar coloanele de care ai nevoie, folosind funcția `loc`, care extrage din dataframe-ul original un grup de rânduri (transmise ca prim parametru) și coloane (transmise ca al doilea parametru). Expresia `:` în cazul de mai jos înseamnă "toate rândurile".

    ```python
    columns_to_select = ['Package', 'Low Price', 'High Price', 'Date']
    pumpkins = pumpkins.loc[:, columns_to_select]
    ```

### În al doilea rând, determină prețul mediu al dovlecilor

Gândește-te cum să determini prețul mediu al unui dovleac într-o anumită lună. Ce coloane ai alege pentru această sarcină? Indiciu: vei avea nevoie de 3 coloane.

Soluție: ia media coloanelor `Low Price` și `High Price` pentru a popula noua coloană Price și convertește coloana Date astfel încât să afișeze doar luna. Din fericire, conform verificării de mai sus, nu există date lipsă pentru date sau prețuri.

1. Pentru a calcula media, adaugă următorul cod:

    ```python
    price = (pumpkins['Low Price'] + pumpkins['High Price']) / 2

    month = pd.DatetimeIndex(pumpkins['Date']).month

    ```

   ✅ Simte-te liber să afișezi orice date dorești să verifici folosind `print(month)`.

2. Acum, copiază datele convertite într-un nou dataframe Pandas:

    ```python
    new_pumpkins = pd.DataFrame({'Month': month, 'Package': pumpkins['Package'], 'Low Price': pumpkins['Low Price'],'High Price': pumpkins['High Price'], 'Price': price})
    ```

    Afișarea dataframe-ului va arăta un set de date curat și ordonat pe care poți construi noul model de regresie.

### Dar stai! E ceva ciudat aici

Dacă te uiți la coloana `Package`, dovlecii sunt vânduți în multe configurații diferite. Unii sunt vânduți în măsuri de '1 1/9 bushel', alții în măsuri de '1/2 bushel', unii per dovleac, alții per kilogram, și unii în cutii mari cu lățimi variabile.

> Dovlecii par foarte greu de cântărit în mod consistent

Analizând datele originale, este interesant că orice are `Unit of Sale` egal cu 'EACH' sau 'PER BIN' are, de asemenea, tipul `Package` per inch, per bin sau 'each'. Dovlecii par foarte greu de cântărit în mod consistent, așa că să-i filtrăm selectând doar dovlecii cu șirul 'bushel' în coloana `Package`.

1. Adaugă un filtru în partea de sus a fișierului, sub importul inițial .csv:

    ```python
    pumpkins = pumpkins[pumpkins['Package'].str.contains('bushel', case=True, regex=True)]
    ```

    Dacă afișezi datele acum, poți vedea că obții doar cele aproximativ 415 rânduri de date care conțin dovleci la bushel.

### Dar stai! Mai e ceva de făcut

Ai observat că cantitatea de bushel variază pe rând? Trebuie să normalizezi prețurile astfel încât să afișezi prețurile per bushel, așa că fă niște calcule pentru a le standardiza.

1. Adaugă aceste linii după blocul care creează dataframe-ul new_pumpkins:

    ```python
    new_pumpkins.loc[new_pumpkins['Package'].str.contains('1 1/9'), 'Price'] = price/(1 + 1/9)

    new_pumpkins.loc[new_pumpkins['Package'].str.contains('1/2'), 'Price'] = price/(1/2)
    ```

✅ Conform [The Spruce Eats](https://www.thespruceeats.com/how-much-is-a-bushel-1389308), greutatea unui bushel depinde de tipul de produs, deoarece este o măsurare a volumului. "Un bushel de roșii, de exemplu, ar trebui să cântărească 56 de livre... Frunzele și verdețurile ocupă mai mult spațiu cu mai puțină greutate, astfel încât un bushel de spanac cântărește doar 20 de livre." Este destul de complicat! Să nu ne deranjăm cu conversia bushel-livre și să stabilim prețul per bushel. Tot acest studiu despre bushel-uri de dovleci, însă, arată cât de important este să înțelegi natura datelor tale!

Acum, poți analiza prețurile per unitate pe baza măsurării bushel-ului. Dacă afișezi datele încă o dată, poți vedea cum sunt standardizate.

✅ Ai observat că dovlecii vânduți la jumătate de bushel sunt foarte scumpi? Poți să-ți dai seama de ce? Indiciu: dovlecii mici sunt mult mai scumpi decât cei mari, probabil pentru că sunt mult mai mulți per bushel, având în vedere spațiul neutilizat ocupat de un dovleac mare pentru plăcintă.

## Strategii de vizualizare

O parte din rolul unui specialist în date este să demonstreze calitatea și natura datelor cu care lucrează. Pentru a face acest lucru, ei creează adesea vizualizări interesante, cum ar fi diagrame, grafice și tabele, care arată diferite aspecte ale datelor. În acest fel, pot arăta vizual relații și lacune care altfel ar fi greu de descoperit.

[![ML pentru începători - Cum să vizualizezi datele cu Matplotlib](https://img.youtube.com/vi/SbUkxH6IJo0/0.jpg)](https://youtu.be/SbUkxH6IJo0 "ML pentru începători - Cum să vizualizezi datele cu Matplotlib")

> 🎥 Fă clic pe imaginea de mai sus pentru un scurt videoclip despre vizualizarea datelor pentru această lecție.

Vizualizările pot ajuta, de asemenea, la determinarea tehnicii de învățare automată cea mai potrivită pentru date. Un scatterplot care pare să urmeze o linie, de exemplu, indică faptul că datele sunt un candidat bun pentru un exercițiu de regresie liniară.

O bibliotecă de vizualizare a datelor care funcționează bine în notebook-urile Jupyter este [Matplotlib](https://matplotlib.org/) (pe care ai văzut-o și în lecția anterioară).

> Obține mai multă experiență cu vizualizarea datelor în [aceste tutoriale](https://docs.microsoft.com/learn/modules/explore-analyze-data-with-python?WT.mc_id=academic-77952-leestott).

## Exercițiu - experimentează cu Matplotlib

Încearcă să creezi câteva diagrame de bază pentru a afișa noul dataframe pe care tocmai l-ai creat. Ce ar arăta o diagramă liniară de bază?

1. Importă Matplotlib în partea de sus a fișierului, sub importul Pandas:

    ```python
    import matplotlib.pyplot as plt
    ```

1. Rulează din nou întregul notebook pentru a-l actualiza.
1. În partea de jos a notebook-ului, adaugă o celulă pentru a afișa datele sub formă de boxplot:

    ```python
    price = new_pumpkins.Price
    month = new_pumpkins.Month
    plt.scatter(price, month)
    plt.show()
    ```

    ![Un scatterplot care arată relația dintre preț și lună](../../../../2-Regression/2-Data/images/scatterplot.png)

    Este aceasta o diagramă utilă? Te surprinde ceva la ea?

    Nu este foarte utilă, deoarece tot ce face este să afișeze datele tale ca o răspândire de puncte într-o lună dată.

### Fă-o utilă

Pentru a obține diagrame care să afișeze date utile, de obicei trebuie să grupezi datele cumva. Să încercăm să creăm o diagramă în care axa y să afișeze lunile, iar datele să demonstreze distribuția datelor.

1. Adaugă o celulă pentru a crea o diagramă de bare grupată:

    ```python
    new_pumpkins.groupby(['Month'])['Price'].mean().plot(kind='bar')
    plt.ylabel("Pumpkin Price")
    ```

    ![O diagramă de bare care arată relația dintre preț și lună](../../../../2-Regression/2-Data/images/barchart.png)

    Aceasta este o vizualizare a datelor mai utilă! Pare să indice că cel mai mare preț pentru dovleci apare în septembrie și octombrie. Se potrivește aceasta cu așteptările tale? De ce sau de ce nu?

---

## 🚀Provocare

Explorează diferitele tipuri de vizualizări pe care le oferă Matplotlib. Care tipuri sunt cele mai potrivite pentru problemele de regresie?

## [Chestionar după lecție](https://ff-quizzes.netlify.app/en/ml/)

## Recapitulare și studiu individual

Aruncă o privire la numeroasele moduri de a vizualiza datele. Fă o listă cu diversele biblioteci disponibile și notează care sunt cele mai bune pentru anumite tipuri de sarcini, de exemplu vizualizări 2D vs. vizualizări 3D. Ce descoperi?

## Temă

[Explorarea vizualizării](assignment.md)

---

**Declinare de responsabilitate**:  
Acest document a fost tradus folosind serviciul de traducere AI [Co-op Translator](https://github.com/Azure/co-op-translator). Deși ne străduim să asigurăm acuratețea, vă rugăm să fiți conștienți că traducerile automate pot conține erori sau inexactități. Documentul original în limba sa natală ar trebui considerat sursa autoritară. Pentru informații critice, se recomandă traducerea profesională realizată de un specialist uman. Nu ne asumăm responsabilitatea pentru eventualele neînțelegeri sau interpretări greșite care pot apărea din utilizarea acestei traduceri.