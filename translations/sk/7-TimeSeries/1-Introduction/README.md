<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "662b509c39eee205687726636d0a8455",
  "translation_date": "2025-09-05T15:33:49+00:00",
  "source_file": "7-TimeSeries/1-Introduction/README.md",
  "language_code": "sk"
}
-->
# Úvod do predikcie časových radov

![Zhrnutie časových radov v sketchnote](../../../../sketchnotes/ml-timeseries.png)

> Sketchnote od [Tomomi Imura](https://www.twitter.com/girlie_mac)

V tejto lekcii a nasledujúcej sa naučíte niečo o predikcii časových radov, zaujímavej a hodnotnej časti repertoáru ML vedca, ktorá je o niečo menej známa ako iné témy. Predikcia časových radov je akýsi „krištáľový glóbus“: na základe minulého výkonu premenných, ako je cena, môžete predpovedať jej budúcu potenciálnu hodnotu.

[![Úvod do predikcie časových radov](https://img.youtube.com/vi/cBojo1hsHiI/0.jpg)](https://youtu.be/cBojo1hsHiI "Úvod do predikcie časových radov")

> 🎥 Kliknite na obrázok vyššie pre video o predikcii časových radov

## [Kvíz pred prednáškou](https://ff-quizzes.netlify.app/en/ml/)

Je to užitočná a zaujímavá oblasť s reálnou hodnotou pre podnikanie, vzhľadom na jej priamu aplikáciu na problémy s cenami, inventárom a otázkami dodávateľského reťazca. Hoci sa začali používať techniky hlbokého učenia na získanie lepších poznatkov pre presnejšiu predikciu budúceho výkonu, predikcia časových radov zostáva oblasťou, ktorú výrazne ovplyvňujú klasické techniky ML.

> Užitočný učebný plán časových radov od Penn State nájdete [tu](https://online.stat.psu.edu/stat510/lesson/1)

## Úvod

Predstavte si, že spravujete pole inteligentných parkovacích automatov, ktoré poskytujú údaje o tom, ako často sa používajú a ako dlho v priebehu času.

> Čo keby ste mohli predpovedať, na základe minulého výkonu automatu, jeho budúcu hodnotu podľa zákonov ponuky a dopytu?

Presné predpovedanie, kedy konať, aby ste dosiahli svoj cieľ, je výzva, ktorú by mohla riešiť predikcia časových radov. Ľudí by síce nepotešilo, keby boli účtované vyššie poplatky v rušných časoch, keď hľadajú parkovacie miesto, ale bol by to istý spôsob, ako generovať príjem na čistenie ulíc!

Poďme preskúmať niektoré typy algoritmov časových radov a začnime notebook na čistenie a prípravu údajov. Údaje, ktoré budete analyzovať, pochádzajú zo súťaže GEFCom2014 o predikciu. Obsahujú 3 roky hodinových hodnôt elektrickej záťaže a teploty medzi rokmi 2012 a 2014. Na základe historických vzorcov elektrickej záťaže a teploty môžete predpovedať budúce hodnoty elektrickej záťaže.

V tomto príklade sa naučíte predpovedať jeden časový krok dopredu, pričom použijete iba historické údaje o záťaži. Pred začiatkom je však užitočné pochopiť, čo sa deje v zákulisí.

## Niektoré definície

Pri stretnutí s pojmom „časové rady“ je potrebné pochopiť jeho použitie v niekoľkých rôznych kontextoch.

🎓 **Časové rady**

V matematike sú „časové rady sériou dátových bodov indexovaných (alebo uvedených alebo graficky znázornených) v časovom poradí. Najčastejšie sú časové rady sekvenciou zaznamenanou v po sebe nasledujúcich rovnako vzdialených časových bodoch.“ Príkladom časových radov je denná uzatváracia hodnota [Dow Jones Industrial Average](https://wikipedia.org/wiki/Time_series). Použitie grafov časových radov a štatistického modelovania sa často vyskytuje pri spracovaní signálov, predpovedi počasia, predpovedi zemetrasení a v iných oblastiach, kde sa udalosti vyskytujú a dátové body môžu byť znázornené v čase.

🎓 **Analýza časových radov**

Analýza časových radov je analýza vyššie uvedených údajov časových radov. Údaje časových radov môžu mať rôzne formy, vrátane „prerušených časových radov“, ktoré detekujú vzorce vo vývoji časových radov pred a po prerušenom udalosti. Typ analýzy potrebnej pre časové rady závisí od povahy údajov. Údaje časových radov samotné môžu mať formu sérií čísel alebo znakov.

Analýza, ktorá sa má vykonať, používa rôzne metódy, vrátane frekvenčnej domény a časovej domény, lineárne a nelineárne a ďalšie. [Viac sa dozviete](https://www.itl.nist.gov/div898/handbook/pmc/section4/pmc4.htm) o mnohých spôsoboch analýzy tohto typu údajov.

🎓 **Predikcia časových radov**

Predikcia časových radov je použitie modelu na predpovedanie budúcich hodnôt na základe vzorcov zobrazených predtým zhromaždenými údajmi, ako sa vyskytovali v minulosti. Hoci je možné použiť regresné modely na skúmanie údajov časových radov, s časovými indexmi ako x premennými na grafe, takéto údaje je najlepšie analyzovať pomocou špeciálnych typov modelov.

Údaje časových radov sú zoznamom usporiadaných pozorovaní, na rozdiel od údajov, ktoré je možné analyzovať lineárnou regresiou. Najbežnejším modelom je ARIMA, skratka pre „Autoregressive Integrated Moving Average“.

[ARIMA modely](https://online.stat.psu.edu/stat510/lesson/1/1.1) „spájajú súčasnú hodnotu série s minulými hodnotami a minulými chybami predpovede.“ Sú najvhodnejšie na analýzu údajov časovej domény, kde sú údaje usporiadané v čase.

> Existuje niekoľko typov ARIMA modelov, o ktorých sa môžete dozvedieť [tu](https://people.duke.edu/~rnau/411arim.htm) a ktorých sa dotknete v nasledujúcej lekcii.

V nasledujúcej lekcii vytvoríte ARIMA model pomocou [Jednorozmerných časových radov](https://itl.nist.gov/div898/handbook/pmc/section4/pmc44.htm), ktoré sa zameriavajú na jednu premennú, ktorá mení svoju hodnotu v čase. Príkladom tohto typu údajov je [tento dataset](https://itl.nist.gov/div898/handbook/pmc/section4/pmc4411.htm), ktorý zaznamenáva mesačnú koncentráciu CO2 na observatóriu Mauna Loa:

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

✅ Identifikujte premennú, ktorá sa v tomto datasete mení v čase.

## Charakteristiky údajov časových radov, ktoré treba zvážiť

Pri pohľade na údaje časových radov si môžete všimnúť, že majú [určité charakteristiky](https://online.stat.psu.edu/stat510/lesson/1/1.1), ktoré je potrebné zohľadniť a zmierniť, aby ste lepšie pochopili ich vzorce. Ak považujete údaje časových radov za potenciálne poskytujúce „signál“, ktorý chcete analyzovať, tieto charakteristiky možno považovať za „šum“. Často budete musieť tento „šum“ znížiť kompenzovaním niektorých z týchto charakteristík pomocou štatistických techník.

Tu sú niektoré koncepty, ktoré by ste mali poznať, aby ste mohli pracovať s časovými radmi:

🎓 **Trendy**

Trendy sú definované ako merateľné nárasty a poklesy v priebehu času. [Prečítajte si viac](https://machinelearningmastery.com/time-series-trends-in-python). V kontexte časových radov ide o to, ako používať a, ak je to potrebné, odstrániť trendy z vašich časových radov.

🎓 **[Sezónnosť](https://machinelearningmastery.com/time-series-seasonality-with-python/)**

Sezónnosť je definovaná ako periodické výkyvy, napríklad sviatočné nákupy, ktoré môžu ovplyvniť predaj. [Pozrite sa](https://itl.nist.gov/div898/handbook/pmc/section4/pmc443.htm), ako rôzne typy grafov zobrazujú sezónnosť v údajoch.

🎓 **Odľahlé hodnoty**

Odľahlé hodnoty sú ďaleko od štandardnej variability údajov.

🎓 **Dlhodobý cyklus**

Nezávisle od sezónnosti môžu údaje vykazovať dlhodobý cyklus, ako je hospodársky pokles, ktorý trvá dlhšie ako rok.

🎓 **Konštantná variancia**

V priebehu času niektoré údaje vykazujú konštantné výkyvy, ako je spotreba energie počas dňa a noci.

🎓 **Náhle zmeny**

Údaje môžu vykazovať náhlu zmenu, ktorá si môže vyžadovať ďalšiu analýzu. Náhle zatvorenie podnikov kvôli COVID-u, napríklad, spôsobilo zmeny v údajoch.

✅ Tu je [ukážkový graf časových radov](https://www.kaggle.com/kashnitsky/topic-9-part-1-time-series-analysis-in-python), ktorý zobrazuje denné výdavky na hernú menu počas niekoľkých rokov. Dokážete identifikovať niektoré z vyššie uvedených charakteristík v týchto údajoch?

![Výdavky na hernú menu](../../../../7-TimeSeries/1-Introduction/images/currency.png)

## Cvičenie - začíname s údajmi o spotrebe energie

Začnime vytvárať model časových radov na predpovedanie budúcej spotreby energie na základe minulých údajov.

> Údaje v tomto príklade pochádzajú zo súťaže GEFCom2014 o predikciu. Obsahujú 3 roky hodinových hodnôt elektrickej záťaže a teploty medzi rokmi 2012 a 2014.
>
> Tao Hong, Pierre Pinson, Shu Fan, Hamidreza Zareipour, Alberto Troccoli a Rob J. Hyndman, "Probabilistic energy forecasting: Global Energy Forecasting Competition 2014 and beyond", International Journal of Forecasting, vol.32, no.3, pp 896-913, July-September, 2016.

1. V priečinku `working` tejto lekcie otvorte súbor _notebook.ipynb_. Začnite pridaním knižníc, ktoré vám pomôžu načítať a vizualizovať údaje.

    ```python
    import os
    import matplotlib.pyplot as plt
    from common.utils import load_data
    %matplotlib inline
    ```

    Upozornenie: používate súbory zo zahrnutého priečinka `common`, ktoré nastavujú vaše prostredie a spracovávajú sťahovanie údajov.

2. Ďalej preskúmajte údaje ako dataframe pomocou `load_data()` a `head()`:

    ```python
    data_dir = './data'
    energy = load_data(data_dir)[['load']]
    energy.head()
    ```

    Môžete vidieť, že existujú dva stĺpce reprezentujúce dátum a záťaž:

    |                     |  load  |
    | :-----------------: | :----: |
    | 2012-01-01 00:00:00 | 2698.0 |
    | 2012-01-01 01:00:00 | 2558.0 |
    | 2012-01-01 02:00:00 | 2444.0 |
    | 2012-01-01 03:00:00 | 2402.0 |
    | 2012-01-01 04:00:00 | 2403.0 |

3. Teraz vykreslite údaje pomocou `plot()`:

    ```python
    energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![graf energie](../../../../7-TimeSeries/1-Introduction/images/energy-plot.png)

4. Teraz vykreslite prvý týždeň júla 2014, poskytnutím vstupu do `energy` vo formáte `[od dátumu]: [do dátumu]`:

    ```python
    energy['2014-07-01':'2014-07-07'].plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![júl](../../../../7-TimeSeries/1-Introduction/images/july-2014.png)

    Krásny graf! Pozrite sa na tieto grafy a zistite, či dokážete určiť niektoré z vyššie uvedených charakteristík. Čo môžeme usúdiť vizualizáciou údajov?

V nasledujúcej lekcii vytvoríte ARIMA model na vytvorenie niektorých predpovedí.

---

## 🚀Výzva

Vytvorte zoznam všetkých odvetví a oblastí výskumu, ktoré by mohli profitovať z predikcie časových radov. Dokážete si predstaviť aplikáciu týchto techník v umení? V ekonometrii? Ekológii? Maloobchode? Priemysle? Financiách? Kde ešte?

## [Kvíz po prednáške](https://ff-quizzes.netlify.app/en/ml/)

## Prehľad a samostatné štúdium

Hoci ich tu nebudeme pokrývať, neurónové siete sa niekedy používajú na zlepšenie klasických metód predikcie časových radov. Prečítajte si o nich viac [v tomto článku](https://medium.com/microsoftazure/neural-networks-for-forecasting-financial-and-economic-time-series-6aca370ff412)

## Zadanie

[Vizualizujte ďalšie časové rady](assignment.md)

---

**Upozornenie**:  
Tento dokument bol preložený pomocou služby AI prekladu [Co-op Translator](https://github.com/Azure/co-op-translator). Aj keď sa snažíme o presnosť, prosím, berte na vedomie, že automatizované preklady môžu obsahovať chyby alebo nepresnosti. Pôvodný dokument v jeho pôvodnom jazyku by mal byť považovaný za autoritatívny zdroj. Pre kritické informácie sa odporúča profesionálny ľudský preklad. Nie sme zodpovední za akékoľvek nedorozumenia alebo nesprávne interpretácie vyplývajúce z použitia tohto prekladu.