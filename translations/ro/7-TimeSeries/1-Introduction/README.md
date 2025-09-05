<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "662b509c39eee205687726636d0a8455",
  "translation_date": "2025-09-05T15:34:32+00:00",
  "source_file": "7-TimeSeries/1-Introduction/README.md",
  "language_code": "ro"
}
-->
# Introducere în prognoza seriilor temporale

![Rezumat al seriilor temporale într-un sketchnote](../../../../sketchnotes/ml-timeseries.png)

> Sketchnote de [Tomomi Imura](https://www.twitter.com/girlie_mac)

În această lecție și în cea următoare, vei învăța câte ceva despre prognoza seriilor temporale, o parte interesantă și valoroasă din repertoriul unui om de știință în ML, care este puțin mai puțin cunoscută decât alte subiecte. Prognoza seriilor temporale este un fel de „glob de cristal”: pe baza performanței trecute a unei variabile, cum ar fi prețul, poți prezice valoarea sa potențială viitoare.

[![Introducere în prognoza seriilor temporale](https://img.youtube.com/vi/cBojo1hsHiI/0.jpg)](https://youtu.be/cBojo1hsHiI "Introducere în prognoza seriilor temporale")

> 🎥 Fă clic pe imaginea de mai sus pentru un videoclip despre prognoza seriilor temporale

## [Chestionar înainte de lecție](https://ff-quizzes.netlify.app/en/ml/)

Este un domeniu util și interesant, cu valoare reală pentru afaceri, având aplicații directe în probleme de prețuri, inventar și lanțuri de aprovizionare. Deși tehnicile de învățare profundă au început să fie utilizate pentru a obține mai multe informații și a prezice mai bine performanța viitoare, prognoza seriilor temporale rămâne un domeniu puternic influențat de tehnicile clasice de ML.

> Curriculumul util despre serii temporale de la Penn State poate fi găsit [aici](https://online.stat.psu.edu/stat510/lesson/1)

## Introducere

Să presupunem că administrezi o serie de parcometre inteligente care oferă date despre cât de des sunt utilizate și pentru cât timp, de-a lungul timpului.

> Ce-ar fi dacă ai putea prezice, pe baza performanței trecute a parcometrului, valoarea sa viitoare conform legilor cererii și ofertei?

Prezicerea exactă a momentului în care să acționezi pentru a-ți atinge obiectivul este o provocare care ar putea fi abordată prin prognoza seriilor temporale. Nu ar face oamenii fericiți să fie taxați mai mult în perioadele aglomerate când caută un loc de parcare, dar ar fi o modalitate sigură de a genera venituri pentru curățarea străzilor!

Să explorăm câteva tipuri de algoritmi pentru serii temporale și să începem un notebook pentru a curăța și pregăti niște date. Datele pe care le vei analiza sunt preluate din competiția de prognoză GEFCom2014. Acestea constau în 3 ani de valori orare ale consumului de energie electrică și temperaturii, între 2012 și 2014. Având în vedere modelele istorice ale consumului de energie electrică și temperaturii, poți prezice valorile viitoare ale consumului de energie electrică.

În acest exemplu, vei învăța cum să prognozezi un pas de timp înainte, folosind doar datele istorice ale consumului. Totuși, înainte de a începe, este util să înțelegi ce se întâmplă în culise.

## Câteva definiții

Când întâlnești termenul „serii temporale”, trebuie să înțelegi utilizarea sa în mai multe contexte diferite.

🎓 **Serii temporale**

În matematică, „o serie temporală este o serie de puncte de date indexate (sau listate sau reprezentate grafic) în ordine temporală. Cel mai frecvent, o serie temporală este o secvență luată la puncte succesive, spațiate egal în timp.” Un exemplu de serie temporală este valoarea de închidere zilnică a [Dow Jones Industrial Average](https://wikipedia.org/wiki/Time_series). Utilizarea graficelor de serii temporale și modelarea statistică sunt frecvent întâlnite în procesarea semnalelor, prognoza meteo, predicția cutremurelor și alte domenii în care evenimentele au loc și punctele de date pot fi reprezentate grafic în timp.

🎓 **Analiza seriilor temporale**

Analiza seriilor temporale este analiza datelor menționate mai sus. Datele de serii temporale pot lua forme distincte, inclusiv „serii temporale întrerupte”, care detectează modele în evoluția unei serii temporale înainte și după un eveniment întreruptor. Tipul de analiză necesar pentru seria temporală depinde de natura datelor. Datele de serii temporale în sine pot lua forma unor serii de numere sau caractere.

Analiza care urmează să fie efectuată utilizează o varietate de metode, inclusiv domeniul frecvenței și domeniul timpului, metode liniare și neliniare și altele. [Află mai multe](https://www.itl.nist.gov/div898/handbook/pmc/section4/pmc4.htm) despre numeroasele moduri de a analiza acest tip de date.

🎓 **Prognoza seriilor temporale**

Prognoza seriilor temporale este utilizarea unui model pentru a prezice valori viitoare pe baza modelelor afișate de datele colectate anterior, pe măsură ce au avut loc în trecut. Deși este posibil să folosești modele de regresie pentru a explora datele de serii temporale, cu indici de timp ca variabile x pe un grafic, astfel de date sunt cel mai bine analizate folosind tipuri speciale de modele.

Datele de serii temporale sunt o listă de observații ordonate, spre deosebire de datele care pot fi analizate prin regresie liniară. Cel mai comun model este ARIMA, un acronim care înseamnă „Autoregressive Integrated Moving Average”.

[Modelele ARIMA](https://online.stat.psu.edu/stat510/lesson/1/1.1) „relatează valoarea prezentă a unei serii cu valorile trecute și erorile de predicție trecute.” Acestea sunt cele mai potrivite pentru analizarea datelor din domeniul timpului, unde datele sunt ordonate în timp.

> Există mai multe tipuri de modele ARIMA, despre care poți învăța [aici](https://people.duke.edu/~rnau/411arim.htm) și pe care le vei aborda în lecția următoare.

În lecția următoare, vei construi un model ARIMA folosind [Serii Temporale Univariate](https://itl.nist.gov/div898/handbook/pmc/section4/pmc44.htm), care se concentrează pe o singură variabilă ce își schimbă valoarea în timp. Un exemplu de acest tip de date este [acest set de date](https://itl.nist.gov/div898/handbook/pmc/section4/pmc4411.htm) care înregistrează concentrația lunară de CO2 la Observatorul Mauna Loa:

|  CO2   | YearMonth | Year  | Month |
| :----: | :-------: | :---: | :---: |
| 330.62 |  1975.04  | 1975  |   1   |
| 331.40 |  1975.13  | 1975  |   2   |
| 331.87 |  1975.21  | 1975  |   3   |
| 333.18 |  1975.29  | 1975  |   4   |
| 333.92 |  1975.38  | 1975  |   5   |
| 333.43 |  1975.46  | 1975  |   6   |
| 331.85 |  1975.54  | 1975  |   7   |
| 330.01 |  1975.63  | 1975  |   8   |
| 328.51 |  1975.71  | 1975  |   9   |
| 328.41 |  1975.79  | 1975  |  10   |
| 329.25 |  1975.88  | 1975  |  11   |
| 330.97 |  1975.96  | 1975  |  12   |

✅ Identifică variabila care se schimbă în timp în acest set de date

## Caracteristici ale datelor de serii temporale de luat în considerare

Când analizezi datele de serii temporale, s-ar putea să observi că acestea au [anumite caracteristici](https://online.stat.psu.edu/stat510/lesson/1/1.1) pe care trebuie să le iei în considerare și să le atenuezi pentru a înțelege mai bine modelele lor. Dacă consideri datele de serii temporale ca potențial oferind un „semnal” pe care vrei să-l analizezi, aceste caracteristici pot fi considerate „zgomot”. Adesea va trebui să reduci acest „zgomot” prin compensarea unor caracteristici folosind tehnici statistice.

Iată câteva concepte pe care ar trebui să le cunoști pentru a putea lucra cu serii temporale:

🎓 **Tendințe**

Tendințele sunt definite ca creșteri și scăderi măsurabile în timp. [Citește mai multe](https://machinelearningmastery.com/time-series-trends-in-python). În contextul seriilor temporale, este vorba despre cum să utilizezi și, dacă este necesar, să elimini tendințele din seria ta temporală.

🎓 **[Sezonalitate](https://machinelearningmastery.com/time-series-seasonality-with-python/)**

Sezonalitatea este definită ca fluctuații periodice, cum ar fi creșterile de vânzări în perioada sărbătorilor, de exemplu. [Aruncă o privire](https://itl.nist.gov/div898/handbook/pmc/section4/pmc443.htm) la modul în care diferite tipuri de grafice afișează sezonalitatea în date.

🎓 **Valori extreme**

Valorile extreme sunt departe de variația standard a datelor.

🎓 **Cicluri pe termen lung**

Independent de sezonalitate, datele pot afișa un ciclu pe termen lung, cum ar fi o recesiune economică care durează mai mult de un an.

🎓 **Variație constantă**

De-a lungul timpului, unele date afișează fluctuații constante, cum ar fi consumul de energie pe zi și noapte.

🎓 **Schimbări bruște**

Datele pot afișa o schimbare bruscă care ar putea necesita o analiză suplimentară. Închiderea bruscă a afacerilor din cauza COVID, de exemplu, a cauzat schimbări în date.

✅ Iată un [exemplu de grafic de serii temporale](https://www.kaggle.com/kashnitsky/topic-9-part-1-time-series-analysis-in-python) care arată cheltuielile zilnice cu moneda din joc pe parcursul câtorva ani. Poți identifica oricare dintre caracteristicile enumerate mai sus în aceste date?

![Cheltuieli cu moneda din joc](../../../../7-TimeSeries/1-Introduction/images/currency.png)

## Exercițiu - începerea cu datele despre consumul de energie

Să începem să creăm un model de serii temporale pentru a prezice consumul viitor de energie pe baza consumului trecut.

> Datele din acest exemplu sunt preluate din competiția de prognoză GEFCom2014. Acestea constau în 3 ani de valori orare ale consumului de energie electrică și temperaturii, între 2012 și 2014.
>
> Tao Hong, Pierre Pinson, Shu Fan, Hamidreza Zareipour, Alberto Troccoli și Rob J. Hyndman, „Prognoza probabilistică a energiei: Competiția Globală de Prognoză a Energiei 2014 și dincolo de aceasta”, International Journal of Forecasting, vol.32, nr.3, pp 896-913, iulie-septembrie, 2016.

1. În folderul `working` al acestei lecții, deschide fișierul _notebook.ipynb_. Începe prin adăugarea bibliotecilor care te vor ajuta să încarci și să vizualizezi datele:

    ```python
    import os
    import matplotlib.pyplot as plt
    from common.utils import load_data
    %matplotlib inline
    ```

    Observă că folosești fișierele din folderul inclus `common`, care configurează mediul și se ocupă de descărcarea datelor.

2. Apoi, examinează datele ca un dataframe apelând `load_data()` și `head()`:

    ```python
    data_dir = './data'
    energy = load_data(data_dir)[['load']]
    energy.head()
    ```

    Poți vedea că există două coloane care reprezintă data și consumul:

    |                     |  load  |
    | :-----------------: | :----: |
    | 2012-01-01 00:00:00 | 2698.0 |
    | 2012-01-01 01:00:00 | 2558.0 |
    | 2012-01-01 02:00:00 | 2444.0 |
    | 2012-01-01 03:00:00 | 2402.0 |
    | 2012-01-01 04:00:00 | 2403.0 |

3. Acum, reprezintă grafic datele apelând `plot()`:

    ```python
    energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![grafic energie](../../../../7-TimeSeries/1-Introduction/images/energy-plot.png)

4. Acum, reprezintă grafic prima săptămână din iulie 2014, oferind-o ca input pentru `energy` în modelul `[de la data]: [până la data]`:

    ```python
    energy['2014-07-01':'2014-07-07'].plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![iulie](../../../../7-TimeSeries/1-Introduction/images/july-2014.png)

    Un grafic frumos! Aruncă o privire la aceste grafice și vezi dacă poți determina oricare dintre caracteristicile enumerate mai sus. Ce putem deduce prin vizualizarea datelor?

În lecția următoare, vei crea un model ARIMA pentru a realiza câteva prognoze.

---

## 🚀Provocare

Fă o listă cu toate industriile și domeniile de cercetare pe care le poți gândi că ar beneficia de prognoza seriilor temporale. Poți gândi o aplicație a acestor tehnici în arte? În econometrie? Ecologie? Retail? Industrie? Finanțe? Unde altundeva?

## [Chestionar după lecție](https://ff-quizzes.netlify.app/en/ml/)

## Recapitulare și studiu individual

Deși nu le vom acoperi aici, rețelele neuronale sunt uneori utilizate pentru a îmbunătăți metodele clasice de prognoză a seriilor temporale. Citește mai multe despre ele [în acest articol](https://medium.com/microsoftazure/neural-networks-for-forecasting-financial-and-economic-time-series-6aca370ff412)

## Temă

[Vizualizează mai multe serii temporale](assignment.md)

---

**Declinare de responsabilitate**:  
Acest document a fost tradus folosind serviciul de traducere AI [Co-op Translator](https://github.com/Azure/co-op-translator). Deși ne străduim să asigurăm acuratețea, vă rugăm să fiți conștienți că traducerile automate pot conține erori sau inexactități. Documentul original în limba sa natală ar trebui considerat sursa autoritară. Pentru informații critice, se recomandă traducerea profesională realizată de un specialist uman. Nu ne asumăm responsabilitatea pentru eventualele neînțelegeri sau interpretări greșite care pot apărea din utilizarea acestei traduceri.