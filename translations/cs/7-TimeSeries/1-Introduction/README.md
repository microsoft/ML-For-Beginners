<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "662b509c39eee205687726636d0a8455",
  "translation_date": "2025-09-04T23:50:19+00:00",
  "source_file": "7-TimeSeries/1-Introduction/README.md",
  "language_code": "cs"
}
-->
# Úvod do předpovědi časových řad

![Shrnutí časových řad ve sketchnote](../../../../sketchnotes/ml-timeseries.png)

> Sketchnote od [Tomomi Imura](https://www.twitter.com/girlie_mac)

V této lekci a následující se naučíte něco o předpovědi časových řad, zajímavé a cenné části repertoáru vědce v oblasti strojového učení, která je méně známá než jiné tématy. Předpověď časových řad je něco jako „křišťálová koule“: na základě minulého výkonu proměnné, jako je cena, můžete předpovědět její budoucí potenciální hodnotu.

[![Úvod do předpovědi časových řad](https://img.youtube.com/vi/cBojo1hsHiI/0.jpg)](https://youtu.be/cBojo1hsHiI "Úvod do předpovědi časových řad")

> 🎥 Klikněte na obrázek výše pro video o předpovědi časových řad

## [Kvíz před lekcí](https://ff-quizzes.netlify.app/en/ml/)

Je to užitečné a zajímavé pole s reálnou hodnotou pro podnikání, vzhledem k jeho přímé aplikaci na problémy s cenami, inventářem a otázkami dodavatelského řetězce. Zatímco techniky hlubokého učení se začaly používat k získání lepších poznatků pro přesnější předpovědi budoucího výkonu, předpověď časových řad zůstává polem, které je stále velmi ovlivněno klasickými technikami strojového učení.

> Užitečný učební plán časových řad od Penn State najdete [zde](https://online.stat.psu.edu/stat510/lesson/1)

## Úvod

Představte si, že spravujete řadu chytrých parkovacích automatů, které poskytují data o tom, jak často a jak dlouho jsou používány v průběhu času.

> Co kdybyste mohli na základě minulého výkonu automatu předpovědět jeho budoucí hodnotu podle zákonů nabídky a poptávky?

Přesné předpovězení, kdy jednat, abyste dosáhli svého cíle, je výzva, kterou by mohla řešit předpověď časových řad. Nebylo by to příjemné pro lidi, kdyby byli účtováni více v rušných časech, když hledají parkovací místo, ale byl by to jistý způsob, jak generovat příjmy na čištění ulic!

Pojďme prozkoumat některé typy algoritmů časových řad a začít notebook pro čištění a přípravu dat. Data, která budete analyzovat, pocházejí ze soutěže GEFCom2014 o předpovědi. Obsahují 3 roky hodinových hodnot elektrické zátěže a teploty mezi lety 2012 a 2014. Na základě historických vzorců elektrické zátěže a teploty můžete předpovědět budoucí hodnoty elektrické zátěže.

V tomto příkladu se naučíte, jak předpovědět jeden časový krok dopředu, pouze pomocí historických dat o zátěži. Než však začnete, je užitečné pochopit, co se děje v pozadí.

## Některé definice

Když narazíte na termín „časové řady“, je důležité pochopit jeho použití v několika různých kontextech.

🎓 **Časové řady**

V matematice jsou „časové řady řadou datových bodů indexovaných (nebo seřazených nebo graficky znázorněných) v časovém pořadí. Nejčastěji jsou časové řady sekvencí zaznamenanou v pravidelných časových intervalech.“ Příkladem časových řad je denní závěrečná hodnota [Dow Jones Industrial Average](https://wikipedia.org/wiki/Time_series). Použití grafů časových řad a statistického modelování se často objevuje v zpracování signálů, předpovědi počasí, předpovědi zemětřesení a dalších oblastech, kde se události vyskytují a datové body lze znázornit v čase.

🎓 **Analýza časových řad**

Analýza časových řad je analýza výše zmíněných dat časových řad. Data časových řad mohou mít různé formy, včetně „přerušených časových řad“, které detekují vzorce ve vývoji časových řad před a po přerušující události. Typ analýzy potřebné pro časové řady závisí na povaze dat. Data časových řad samotná mohou mít formu číselných nebo znakových sérií.

Analýza se provádí pomocí různých metod, včetně frekvenčního a časového doménového zpracování, lineárních a nelineárních metod a dalších. [Zjistěte více](https://www.itl.nist.gov/div898/handbook/pmc/section4/pmc4.htm) o mnoha způsobech analýzy tohoto typu dat.

🎓 **Předpověď časových řad**

Předpověď časových řad je použití modelu k předpovědi budoucích hodnot na základě vzorců zobrazených dříve shromážděnými daty, jak se vyskytovala v minulosti. Zatímco je možné použít regresní modely k prozkoumání dat časových řad, s časovými indexy jako x proměnnými na grafu, taková data se nejlépe analyzují pomocí speciálních typů modelů.

Data časových řad jsou seznamem uspořádaných pozorování, na rozdíl od dat, která lze analyzovat pomocí lineární regrese. Nejčastějším modelem je ARIMA, což je zkratka pro „Autoregressive Integrated Moving Average“.

[ARIMA modely](https://online.stat.psu.edu/stat510/lesson/1/1.1) „spojují současnou hodnotu série s minulými hodnotami a minulými chybami předpovědi.“ Jsou nejvhodnější pro analýzu dat v časové doméně, kde jsou data uspořádána v čase.

> Existuje několik typů ARIMA modelů, o kterých se můžete dozvědět [zde](https://people.duke.edu/~rnau/411arim.htm) a které se dotknete v příští lekci.

V příští lekci vytvoříte ARIMA model pomocí [Jednorozměrných časových řad](https://itl.nist.gov/div898/handbook/pmc/section4/pmc44.htm), které se zaměřují na jednu proměnnou, která mění svou hodnotu v čase. Příkladem tohoto typu dat je [tento dataset](https://itl.nist.gov/div898/handbook/pmc/section4/pmc4411.htm), který zaznamenává měsíční koncentraci CO2 na observatoři Mauna Loa:

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

✅ Identifikujte proměnnou, která se v tomto datasetu mění v čase.

## Charakteristiky dat časových řad, které je třeba zvážit

Když se podíváte na data časových řad, můžete si všimnout, že mají [určité charakteristiky](https://online.stat.psu.edu/stat510/lesson/1/1.1), které je třeba vzít v úvahu a zmírnit, abyste lépe pochopili jejich vzorce. Pokud považujete data časových řad za potenciální „signál“, který chcete analyzovat, tyto charakteristiky lze považovat za „šum“. Často budete muset tento „šum“ snížit kompenzací některých z těchto charakteristik pomocí statistických technik.

Zde jsou některé koncepty, které byste měli znát, abyste mohli pracovat s časovými řadami:

🎓 **Trendy**

Trendy jsou definovány jako měřitelné nárůsty a poklesy v čase. [Zjistěte více](https://machinelearningmastery.com/time-series-trends-in-python). V kontextu časových řad jde o to, jak používat a případně odstraňovat trendy z vašich časových řad.

🎓 **[Sezónnost](https://machinelearningmastery.com/time-series-seasonality-with-python/)**

Sezónnost je definována jako periodické výkyvy, například nákupní horečky během svátků, které mohou ovlivnit prodeje. [Podívejte se](https://itl.nist.gov/div898/handbook/pmc/section4/pmc443.htm), jak různé typy grafů zobrazují sezónnost v datech.

🎓 **Odlehlé hodnoty**

Odlehlé hodnoty jsou daleko od standardní datové variance.

🎓 **Dlouhodobý cyklus**

Nezávisle na sezónnosti mohou data vykazovat dlouhodobý cyklus, například ekonomický pokles, který trvá déle než rok.

🎓 **Konstantní variance**

V průběhu času některá data vykazují konstantní výkyvy, například spotřeba energie během dne a noci.

🎓 **Náhlé změny**

Data mohou vykazovat náhlou změnu, která může vyžadovat další analýzu. Například náhlé uzavření podniků kvůli COVIDu způsobilo změny v datech.

✅ Zde je [ukázkový graf časových řad](https://www.kaggle.com/kashnitsky/topic-9-part-1-time-series-analysis-in-python) zobrazující denní útraty za herní měnu během několika let. Dokážete v těchto datech identifikovat některé z výše uvedených charakteristik?

![Útraty za herní měnu](../../../../7-TimeSeries/1-Introduction/images/currency.png)

## Cvičení - začínáme s daty o spotřebě energie

Začněme vytvářet model časových řad pro předpověď budoucí spotřeby energie na základě minulých hodnot.

> Data v tomto příkladu pocházejí ze soutěže GEFCom2014 o předpovědi. Obsahují 3 roky hodinových hodnot elektrické zátěže a teploty mezi lety 2012 a 2014.
>
> Tao Hong, Pierre Pinson, Shu Fan, Hamidreza Zareipour, Alberto Troccoli a Rob J. Hyndman, "Probabilistic energy forecasting: Global Energy Forecasting Competition 2014 and beyond", International Journal of Forecasting, vol.32, no.3, pp 896-913, červenec-září, 2016.

1. Ve složce `working` této lekce otevřete soubor _notebook.ipynb_. Začněte přidáním knihoven, které vám pomohou načíst a vizualizovat data:

    ```python
    import os
    import matplotlib.pyplot as plt
    from common.utils import load_data
    %matplotlib inline
    ```

    Poznámka: Používáte soubory ze zahrnuté složky `common`, které nastavují vaše prostředí a zajišťují stahování dat.

2. Dále prozkoumejte data jako dataframe pomocí `load_data()` a `head()`:

    ```python
    data_dir = './data'
    energy = load_data(data_dir)[['load']]
    energy.head()
    ```

    Můžete vidět, že existují dva sloupce reprezentující datum a zátěž:

    |                     |  load  |
    | :-----------------: | :----: |
    | 2012-01-01 00:00:00 | 2698.0 |
    | 2012-01-01 01:00:00 | 2558.0 |
    | 2012-01-01 02:00:00 | 2444.0 |
    | 2012-01-01 03:00:00 | 2402.0 |
    | 2012-01-01 04:00:00 | 2403.0 |

3. Nyní vykreslete data pomocí `plot()`:

    ```python
    energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![graf energie](../../../../7-TimeSeries/1-Introduction/images/energy-plot.png)

4. Nyní vykreslete první týden července 2014 zadáním jako vstup do `energy` ve formátu `[od data]: [do data]`:

    ```python
    energy['2014-07-01':'2014-07-07'].plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
    plt.xlabel('timestamp', fontsize=12)
    plt.ylabel('load', fontsize=12)
    plt.show()
    ```

    ![červenec](../../../../7-TimeSeries/1-Introduction/images/july-2014.png)

    Krásný graf! Podívejte se na tyto grafy a zkuste určit některé z výše uvedených charakteristik. Co můžeme usoudit z vizualizace dat?

V příští lekci vytvoříte ARIMA model pro vytvoření předpovědí.

---

## 🚀Výzva

Vytvořte seznam všech odvětví a oblastí zkoumání, které by mohly těžit z předpovědi časových řad. Dokážete si představit aplikaci těchto technik v umění? V ekonometrice? Ekologii? Maloobchodu? Průmyslu? Financích? Kde ještě?

## [Kvíz po lekci](https://ff-quizzes.netlify.app/en/ml/)

## Přehled & Samostudium

Ačkoli je zde nebudeme pokrývat, neuronové sítě se někdy používají k vylepšení klasických metod předpovědi časových řad. Přečtěte si o nich více [v tomto článku](https://medium.com/microsoftazure/neural-networks-for-forecasting-financial-and-economic-time-series-6aca370ff412)

## Zadání

[Vizualizujte další časové řady](assignment.md)

---

**Prohlášení**:  
Tento dokument byl přeložen pomocí služby pro automatický překlad [Co-op Translator](https://github.com/Azure/co-op-translator). Ačkoli se snažíme o přesnost, mějte prosím na paměti, že automatické překlady mohou obsahovat chyby nebo nepřesnosti. Původní dokument v jeho původním jazyce by měl být považován za autoritativní zdroj. Pro důležité informace se doporučuje profesionální lidský překlad. Neodpovídáme za žádná nedorozumění nebo nesprávné interpretace vyplývající z použití tohoto překladu.