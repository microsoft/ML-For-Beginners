<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "7c077988328ebfe33b24d07945f16eca",
  "translation_date": "2025-09-04T23:41:01+00:00",
  "source_file": "2-Regression/2-Data/README.md",
  "language_code": "cs"
}
-->
# Vytvoření regresního modelu pomocí Scikit-learn: příprava a vizualizace dat

![Infografika vizualizace dat](../../../../2-Regression/2-Data/images/data-visualization.png)

Infografika od [Dasani Madipalli](https://twitter.com/dasani_decoded)

## [Kvíz před lekcí](https://ff-quizzes.netlify.app/en/ml/)

> ### [Tato lekce je dostupná v R!](../../../../2-Regression/2-Data/solution/R/lesson_2.html)

## Úvod

Nyní, když máte k dispozici nástroje potřebné k zahájení práce na vytváření modelů strojového učení pomocí Scikit-learn, jste připraveni začít klást otázky svým datům. Při práci s daty a aplikaci řešení ML je velmi důležité vědět, jak položit správnou otázku, abyste mohli plně využít potenciál svého datasetu.

V této lekci se naučíte:

- Jak připravit data pro vytváření modelů.
- Jak používat Matplotlib pro vizualizaci dat.

## Kladení správných otázek svým datům

Otázka, na kterou potřebujete odpověď, určí, jaký typ algoritmů ML budete používat. Kvalita odpovědi, kterou získáte, bude silně záviset na povaze vašich dat.

Podívejte se na [data](https://github.com/microsoft/ML-For-Beginners/blob/main/2-Regression/data/US-pumpkins.csv) poskytnutá pro tuto lekci. Tento soubor .csv můžete otevřít ve VS Code. Rychlý pohled okamžitě ukáže, že jsou zde prázdné hodnoty a směs textových a číselných dat. Je zde také zvláštní sloupec nazvaný 'Package', kde jsou data směsí hodnot jako 'sacks', 'bins' a dalších. Data jsou vlastně trochu chaotická.

[![ML pro začátečníky - Jak analyzovat a čistit dataset](https://img.youtube.com/vi/5qGjczWTrDQ/0.jpg)](https://youtu.be/5qGjczWTrDQ "ML pro začátečníky - Jak analyzovat a čistit dataset")

> 🎥 Klikněte na obrázek výše pro krátké video o přípravě dat pro tuto lekci.

Ve skutečnosti není příliš běžné dostat dataset, který je zcela připraven k použití pro vytvoření modelu ML bez jakýchkoli úprav. V této lekci se naučíte, jak připravit surový dataset pomocí standardních knihoven Pythonu. Naučíte se také různé techniky vizualizace dat.

## Případová studie: 'trh s dýněmi'

V této složce najdete soubor .csv v kořenové složce `data` nazvaný [US-pumpkins.csv](https://github.com/microsoft/ML-For-Beginners/blob/main/2-Regression/data/US-pumpkins.csv), který obsahuje 1757 řádků dat o trhu s dýněmi, rozdělených do skupin podle měst. Jedná se o surová data získaná z [Specialty Crops Terminal Markets Standard Reports](https://www.marketnews.usda.gov/mnp/fv-report-config-step1?type=termPrice), která distribuuje Ministerstvo zemědělství Spojených států.

### Příprava dat

Tato data jsou veřejně dostupná. Mohou být stažena v mnoha samostatných souborech, podle města, z webu USDA. Abychom se vyhnuli příliš mnoha samostatným souborům, spojili jsme všechna data měst do jedné tabulky, takže jsme data již _částečně připravili_. Nyní se podívejme na data podrobněji.

### Data o dýních - první závěry

Co si všimnete na těchto datech? Už jste viděli, že je zde směs textů, čísel, prázdných hodnot a zvláštních hodnot, které je třeba pochopit.

Jakou otázku můžete položit těmto datům pomocí regresní techniky? Co třeba "Předpovědět cenu dýně na prodej během daného měsíce". Při pohledu na data je třeba provést určité změny, aby se vytvořila datová struktura potřebná pro tento úkol.

## Cvičení - analýza dat o dýních

Použijme [Pandas](https://pandas.pydata.org/) (název znamená `Python Data Analysis`), nástroj velmi užitečný pro tvarování dat, k analýze a přípravě těchto dat o dýních.

### Nejprve zkontrolujte chybějící data

Nejprve budete muset podniknout kroky k ověření chybějících dat:

1. Převést data na formát měsíce (jedná se o americká data, takže formát je `MM/DD/YYYY`).
2. Extrahovat měsíc do nového sloupce.

Otevřete soubor _notebook.ipynb_ ve Visual Studio Code a importujte tabulku do nového dataframe Pandas.

1. Použijte funkci `head()`, abyste zobrazili prvních pět řádků.

    ```python
    import pandas as pd
    pumpkins = pd.read_csv('../data/US-pumpkins.csv')
    pumpkins.head()
    ```

    ✅ Jakou funkci byste použili k zobrazení posledních pěti řádků?

1. Zkontrolujte, zda v aktuálním dataframe chybí data:

    ```python
    pumpkins.isnull().sum()
    ```

    Chybí data, ale možná to nebude mít vliv na daný úkol.

1. Aby byl váš dataframe snazší na práci, vyberte pouze sloupce, které potřebujete, pomocí funkce `loc`, která extrahuje z původního dataframe skupinu řádků (předaná jako první parametr) a sloupců (předaná jako druhý parametr). Výraz `:` v níže uvedeném případě znamená "všechny řádky".

    ```python
    columns_to_select = ['Package', 'Low Price', 'High Price', 'Date']
    pumpkins = pumpkins.loc[:, columns_to_select]
    ```

### Dále určete průměrnou cenu dýně

Přemýšlejte o tom, jak určit průměrnou cenu dýně v daném měsíci. Jaké sloupce byste si vybrali pro tento úkol? Nápověda: budete potřebovat 3 sloupce.

Řešení: vezměte průměr sloupců `Low Price` a `High Price`, abyste naplnili nový sloupec Price, a převeďte sloupec Date tak, aby zobrazoval pouze měsíc. Naštěstí podle výše uvedené kontroly nechybí žádná data pro datumy nebo ceny.

1. Pro výpočet průměru přidejte následující kód:

    ```python
    price = (pumpkins['Low Price'] + pumpkins['High Price']) / 2

    month = pd.DatetimeIndex(pumpkins['Date']).month

    ```

   ✅ Neváhejte si vytisknout jakákoli data, která chcete zkontrolovat, pomocí `print(month)`.

2. Nyní zkopírujte převedená data do nového dataframe Pandas:

    ```python
    new_pumpkins = pd.DataFrame({'Month': month, 'Package': pumpkins['Package'], 'Low Price': pumpkins['Low Price'],'High Price': pumpkins['High Price'], 'Price': price})
    ```

    Pokud si vytisknete svůj dataframe, uvidíte čistý, upravený dataset, na kterém můžete vytvořit nový regresní model.

### Ale počkejte! Něco je tu zvláštní

Pokud se podíváte na sloupec `Package`, dýně se prodávají v mnoha různých konfiguracích. Některé se prodávají v mírách '1 1/9 bushel', některé v '1/2 bushel', některé na kus, některé na libru a některé ve velkých krabicích s různými šířkami.

> Dýně se zdají být velmi těžké vážit konzistentně

Při zkoumání původních dat je zajímavé, že vše, co má `Unit of Sale` rovné 'EACH' nebo 'PER BIN', má také typ `Package` na palec, na bin nebo 'each'. Dýně se zdají být velmi těžké vážit konzistentně, takže je filtrujme výběrem pouze dýní s řetězcem 'bushel' ve sloupci `Package`.

1. Přidejte filtr na začátek souboru, pod počáteční import .csv:

    ```python
    pumpkins = pumpkins[pumpkins['Package'].str.contains('bushel', case=True, regex=True)]
    ```

    Pokud nyní vytisknete data, uvidíte, že získáváte pouze asi 415 řádků dat obsahujících dýně podle bushelu.

### Ale počkejte! Je tu ještě jedna věc, kterou je třeba udělat

Všimli jste si, že množství bushelu se liší podle řádku? Musíte normalizovat ceny tak, aby ukazovaly ceny za bushel, takže proveďte nějaké výpočty pro standardizaci.

1. Přidejte tyto řádky po bloku vytvářejícím dataframe new_pumpkins:

    ```python
    new_pumpkins.loc[new_pumpkins['Package'].str.contains('1 1/9'), 'Price'] = price/(1 + 1/9)

    new_pumpkins.loc[new_pumpkins['Package'].str.contains('1/2'), 'Price'] = price/(1/2)
    ```

✅ Podle [The Spruce Eats](https://www.thespruceeats.com/how-much-is-a-bushel-1389308) váha bushelu závisí na typu produktu, protože se jedná o objemové měření. "Bushel rajčat, například, by měl vážit 56 liber... Listy a zelenina zabírají více prostoru s menší váhou, takže bushel špenátu váží pouze 20 liber." Je to všechno docela komplikované! Nebudeme se zabývat konverzí bushelu na libry, místo toho budeme určovat cenu podle bushelu. Všechny tyto studie bushelů dýní však ukazují, jak velmi důležité je pochopit povahu vašich dat!

Nyní můžete analyzovat ceny za jednotku na základě jejich měření bushelu. Pokud si data vytisknete ještě jednou, uvidíte, jak jsou standardizována.

✅ Všimli jste si, že dýně prodávané na půl bushelu jsou velmi drahé? Dokážete zjistit proč? Nápověda: malé dýně jsou mnohem dražší než velké, pravděpodobně proto, že jich je mnohem více na bushel, vzhledem k nevyužitému prostoru, který zabírá jedna velká dutá dýně na koláč.

## Strategie vizualizace

Součástí role datového vědce je demonstrovat kvalitu a povahu dat, se kterými pracuje. K tomu často vytvářejí zajímavé vizualizace, jako jsou grafy, diagramy a tabulky, které ukazují různé aspekty dat. Tímto způsobem mohou vizuálně ukázat vztahy a mezery, které by jinak bylo těžké odhalit.

[![ML pro začátečníky - Jak vizualizovat data pomocí Matplotlib](https://img.youtube.com/vi/SbUkxH6IJo0/0.jpg)](https://youtu.be/SbUkxH6IJo0 "ML pro začátečníky - Jak vizualizovat data pomocí Matplotlib")

> 🎥 Klikněte na obrázek výše pro krátké video o vizualizaci dat pro tuto lekci.

Vizualizace mohou také pomoci určit techniku strojového učení, která je pro data nejvhodnější. Například scatterplot, který se zdá sledovat linii, naznačuje, že data jsou dobrým kandidátem pro cvičení lineární regrese.

Jedna knihovna pro vizualizaci dat, která dobře funguje v Jupyter notebooku, je [Matplotlib](https://matplotlib.org/) (kterou jste viděli i v předchozí lekci).

> Získejte více zkušeností s vizualizací dat v [těchto tutoriálech](https://docs.microsoft.com/learn/modules/explore-analyze-data-with-python?WT.mc_id=academic-77952-leestott).

## Cvičení - experimentujte s Matplotlib

Zkuste vytvořit základní grafy pro zobrazení nového dataframe, který jste právě vytvořili. Co by ukázal základní čárový graf?

1. Importujte Matplotlib na začátek souboru, pod import Pandas:

    ```python
    import matplotlib.pyplot as plt
    ```

1. Znovu spusťte celý notebook, aby se aktualizoval.
1. Na konec notebooku přidejte buňku pro vykreslení dat jako box:

    ```python
    price = new_pumpkins.Price
    month = new_pumpkins.Month
    plt.scatter(price, month)
    plt.show()
    ```

    ![Scatterplot ukazující vztah mezi cenou a měsícem](../../../../2-Regression/2-Data/images/scatterplot.png)

    Je tento graf užitečný? Překvapilo vás na něm něco?

    Není příliš užitečný, protože pouze zobrazuje vaše data jako rozptyl bodů v daném měsíci.

### Udělejte to užitečné

Aby grafy zobrazovaly užitečná data, obvykle je třeba data nějak seskupit. Zkusme vytvořit graf, kde osa y ukazuje měsíce a data demonstrují rozložení dat.

1. Přidejte buňku pro vytvoření seskupeného sloupcového grafu:

    ```python
    new_pumpkins.groupby(['Month'])['Price'].mean().plot(kind='bar')
    plt.ylabel("Pumpkin Price")
    ```

    ![Sloupcový graf ukazující vztah mezi cenou a měsícem](../../../../2-Regression/2-Data/images/barchart.png)

    Toto je užitečnější vizualizace dat! Zdá se, že naznačuje, že nejvyšší cena za dýně se vyskytuje v září a říjnu. Odpovídá to vašemu očekávání? Proč ano nebo ne?

---

## 🚀Výzva

Prozkoumejte různé typy vizualizací, které Matplotlib nabízí. Které typy jsou nejvhodnější pro regresní problémy?

## [Kvíz po lekci](https://ff-quizzes.netlify.app/en/ml/)

## Přehled & Samostudium

Podívejte se na různé způsoby vizualizace dat. Udělejte si seznam různých dostupných knihoven a poznamenejte si, které jsou nejlepší pro dané typy úkolů, například 2D vizualizace vs. 3D vizualizace. Co objevíte?

## Úkol

[Prozkoumání vizualizace](assignment.md)

---

**Prohlášení**:  
Tento dokument byl přeložen pomocí služby pro automatický překlad [Co-op Translator](https://github.com/Azure/co-op-translator). Ačkoli se snažíme o přesnost, mějte na paměti, že automatické překlady mohou obsahovat chyby nebo nepřesnosti. Původní dokument v jeho původním jazyce by měl být považován za autoritativní zdroj. Pro důležité informace se doporučuje profesionální lidský překlad. Neodpovídáme za žádná nedorozumění nebo nesprávné interpretace vyplývající z použití tohoto překladu.