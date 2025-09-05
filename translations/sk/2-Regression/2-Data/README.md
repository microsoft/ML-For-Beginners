<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "7c077988328ebfe33b24d07945f16eca",
  "translation_date": "2025-09-05T15:25:38+00:00",
  "source_file": "2-Regression/2-Data/README.md",
  "language_code": "sk"
}
-->
# Vytvorenie regresného modelu pomocou Scikit-learn: príprava a vizualizácia dát

![Infografika vizualizácie dát](../../../../2-Regression/2-Data/images/data-visualization.png)

Infografiku vytvoril [Dasani Madipalli](https://twitter.com/dasani_decoded)

## [Kvíz pred prednáškou](https://ff-quizzes.netlify.app/en/ml/)

> ### [Táto lekcia je dostupná aj v R!](../../../../2-Regression/2-Data/solution/R/lesson_2.html)

## Úvod

Teraz, keď máte pripravené nástroje na budovanie modelov strojového učenia pomocou Scikit-learn, ste pripravení začať klásť otázky o svojich dátach. Pri práci s dátami a aplikovaní riešení strojového učenia je veľmi dôležité vedieť, ako položiť správnu otázku, aby ste mohli plne využiť potenciál svojho datasetu.

V tejto lekcii sa naučíte:

- Ako pripraviť dáta na budovanie modelov.
- Ako používať Matplotlib na vizualizáciu dát.

## Kladenie správnych otázok o vašich dátach

Otázka, na ktorú potrebujete odpoveď, určí, aký typ algoritmov strojového učenia budete používať. Kvalita odpovede, ktorú dostanete, bude výrazne závisieť od povahy vašich dát.

Pozrite sa na [dáta](https://github.com/microsoft/ML-For-Beginners/blob/main/2-Regression/data/US-pumpkins.csv) poskytnuté pre túto lekciu. Tento .csv súbor môžete otvoriť vo VS Code. Rýchly pohľad okamžite ukáže, že sú tam prázdne miesta a mix textových a číselných dát. Je tam tiež zvláštny stĺpec nazvaný 'Package', kde sú dáta zmiešané medzi 'sacks', 'bins' a inými hodnotami. Dáta sú vlastne dosť chaotické.

[![ML pre začiatočníkov - Ako analyzovať a čistiť dataset](https://img.youtube.com/vi/5qGjczWTrDQ/0.jpg)](https://youtu.be/5qGjczWTrDQ "ML pre začiatočníkov - Ako analyzovať a čistiť dataset")

> 🎥 Kliknite na obrázok vyššie pre krátke video o príprave dát pre túto lekciu.

Nie je veľmi bežné dostať dataset, ktorý je úplne pripravený na použitie na vytvorenie modelu strojového učenia. V tejto lekcii sa naučíte, ako pripraviť surový dataset pomocou štandardných knižníc Pythonu. Naučíte sa tiež rôzne techniky vizualizácie dát.

## Prípadová štúdia: 'trh s tekvicami'

V tomto priečinku nájdete .csv súbor v koreňovom priečinku `data` nazvaný [US-pumpkins.csv](https://github.com/microsoft/ML-For-Beginners/blob/main/2-Regression/data/US-pumpkins.csv), ktorý obsahuje 1757 riadkov dát o trhu s tekvicami, rozdelených do skupín podľa miest. Ide o surové dáta extrahované z [Specialty Crops Terminal Markets Standard Reports](https://www.marketnews.usda.gov/mnp/fv-report-config-step1?type=termPrice), ktoré distribuuje Ministerstvo poľnohospodárstva USA.

### Príprava dát

Tieto dáta sú vo verejnej doméne. Môžu byť stiahnuté v mnohých samostatných súboroch, podľa miest, z webovej stránky USDA. Aby sme sa vyhli príliš mnohým samostatným súborom, spojili sme všetky dáta miest do jednej tabuľky, takže sme už dáta trochu _pripravili_. Teraz sa pozrime bližšie na dáta.

### Dáta o tekviciach - prvé závery

Čo si všimnete na týchto dátach? Už ste videli, že je tam mix textov, čísel, prázdnych miest a zvláštnych hodnôt, ktoré musíte pochopiť.

Akú otázku môžete položiť o týchto dátach pomocou regresnej techniky? Čo tak "Predpovedať cenu tekvice na predaj počas daného mesiaca". Pri pohľade na dáta je potrebné urobiť niekoľko zmien, aby ste vytvorili štruktúru dát potrebnú na túto úlohu.

## Cvičenie - analýza dát o tekviciach

Použime [Pandas](https://pandas.pydata.org/) (názov znamená `Python Data Analysis`), nástroj veľmi užitočný na tvarovanie dát, na analýzu a prípravu týchto dát o tekviciach.

### Najskôr skontrolujte chýbajúce dátumy

Najprv musíte podniknúť kroky na kontrolu chýbajúcich dátumov:

1. Konvertujte dátumy na formát mesiaca (ide o americké dátumy, takže formát je `MM/DD/YYYY`).
2. Extrahujte mesiac do nového stĺpca.

Otvorte súbor _notebook.ipynb_ vo Visual Studio Code a importujte tabuľku do nového Pandas dataframe.

1. Použite funkciu `head()`, aby ste si pozreli prvých päť riadkov.

    ```python
    import pandas as pd
    pumpkins = pd.read_csv('../data/US-pumpkins.csv')
    pumpkins.head()
    ```

    ✅ Akú funkciu by ste použili na zobrazenie posledných piatich riadkov?

1. Skontrolujte, či sú v aktuálnom dataframe chýbajúce dáta:

    ```python
    pumpkins.isnull().sum()
    ```

    Sú tam chýbajúce dáta, ale možno to nebude mať vplyv na danú úlohu.

1. Aby bol váš dataframe ľahšie spracovateľný, vyberte iba stĺpce, ktoré potrebujete, pomocou funkcie `loc`, ktorá extrahuje z pôvodného dataframe skupinu riadkov (zadaných ako prvý parameter) a stĺpcov (zadaných ako druhý parameter). Výraz `:` v prípade nižšie znamená "všetky riadky".

    ```python
    columns_to_select = ['Package', 'Low Price', 'High Price', 'Date']
    pumpkins = pumpkins.loc[:, columns_to_select]
    ```

### Ďalej určte priemernú cenu tekvice

Premýšľajte o tom, ako určiť priemernú cenu tekvice v danom mesiaci. Ktoré stĺpce by ste si vybrali na túto úlohu? Tip: budete potrebovať 3 stĺpce.

Riešenie: vezmite priemer stĺpcov `Low Price` a `High Price`, aby ste naplnili nový stĺpec Price, a konvertujte stĺpec Date tak, aby zobrazoval iba mesiac. Našťastie, podľa vyššie uvedenej kontroly, nie sú chýbajúce dáta pre dátumy alebo ceny.

1. Na výpočet priemeru pridajte nasledujúci kód:

    ```python
    price = (pumpkins['Low Price'] + pumpkins['High Price']) / 2

    month = pd.DatetimeIndex(pumpkins['Date']).month

    ```

   ✅ Ak chcete skontrolovať akékoľvek dáta, môžete použiť `print(month)`.

2. Teraz skopírujte svoje konvertované dáta do nového Pandas dataframe:

    ```python
    new_pumpkins = pd.DataFrame({'Month': month, 'Package': pumpkins['Package'], 'Low Price': pumpkins['Low Price'],'High Price': pumpkins['High Price'], 'Price': price})
    ```

    Ak si vytlačíte svoj dataframe, uvidíte čistý, upravený dataset, na ktorom môžete postaviť svoj nový regresný model.

### Ale počkajte! Niečo tu nesedí

Ak sa pozriete na stĺpec `Package`, tekvice sa predávajú v mnohých rôznych konfiguráciách. Niektoré sa predávajú v jednotkách '1 1/9 bushel', niektoré v '1/2 bushel', niektoré na kus, niektoré na váhu, a niektoré vo veľkých boxoch s rôznymi šírkami.

> Tekvice sa zdajú byť veľmi ťažké vážiť konzistentne

Pri skúmaní pôvodných dát je zaujímavé, že všetko, čo má `Unit of Sale` rovné 'EACH' alebo 'PER BIN', má tiež typ `Package` na palec, na box alebo 'each'. Tekvice sa zdajú byť veľmi ťažké vážiť konzistentne, takže ich filtrujme výberom iba tekvíc s reťazcom 'bushel' v ich stĺpci `Package`.

1. Pridajte filter na začiatok súboru, pod počiatočný import .csv:

    ```python
    pumpkins = pumpkins[pumpkins['Package'].str.contains('bushel', case=True, regex=True)]
    ```

    Ak si teraz vytlačíte dáta, uvidíte, že dostávate iba približne 415 riadkov dát obsahujúcich tekvice podľa bushel.

### Ale počkajte! Je tu ešte jedna vec, ktorú treba urobiť

Všimli ste si, že množstvo bushel sa líši podľa riadku? Musíte normalizovať ceny tak, aby ste zobrazili ceny za bushel, takže urobte nejaké výpočty na ich štandardizáciu.

1. Pridajte tieto riadky po bloku vytvárajúcom nový dataframe new_pumpkins:

    ```python
    new_pumpkins.loc[new_pumpkins['Package'].str.contains('1 1/9'), 'Price'] = price/(1 + 1/9)

    new_pumpkins.loc[new_pumpkins['Package'].str.contains('1/2'), 'Price'] = price/(1/2)
    ```

✅ Podľa [The Spruce Eats](https://www.thespruceeats.com/how-much-is-a-bushel-1389308) hmotnosť bushel závisí od typu produktu, pretože ide o objemové meranie. "Bushel paradajok, napríklad, by mal vážiť 56 libier... Listy a zelenina zaberajú viac miesta s menšou hmotnosťou, takže bushel špenátu váži iba 20 libier." Je to všetko dosť komplikované! Nebudeme sa zaoberať konverziou bushel na libry, namiesto toho budeme určovať cenu za bushel. Celá táto štúdia bushel tekvíc však ukazuje, aké dôležité je pochopiť povahu vašich dát!

Teraz môžete analyzovať ceny za jednotku na základe ich merania bushel. Ak si ešte raz vytlačíte dáta, uvidíte, ako sú štandardizované.

✅ Všimli ste si, že tekvice predávané na polovičný bushel sú veľmi drahé? Dokážete zistiť prečo? Tip: malé tekvice sú oveľa drahšie ako veľké, pravdepodobne preto, že ich je oveľa viac na bushel, vzhľadom na nevyužitý priestor, ktorý zaberá jedna veľká dutá tekvica na koláč.

## Stratégie vizualizácie

Súčasťou úlohy dátového vedca je demonštrovať kvalitu a povahu dát, s ktorými pracuje. Na tento účel často vytvárajú zaujímavé vizualizácie, ako sú grafy, diagramy a tabuľky, ktoré ukazujú rôzne aspekty dát. Týmto spôsobom môžu vizuálne ukázať vzťahy a medzery, ktoré by inak bolo ťažké odhaliť.

[![ML pre začiatočníkov - Ako vizualizovať dáta pomocou Matplotlib](https://img.youtube.com/vi/SbUkxH6IJo0/0.jpg)](https://youtu.be/SbUkxH6IJo0 "ML pre začiatočníkov - Ako vizualizovať dáta pomocou Matplotlib")

> 🎥 Kliknite na obrázok vyššie pre krátke video o vizualizácii dát pre túto lekciu.

Vizualizácie môžu tiež pomôcť určiť techniku strojového učenia, ktorá je najvhodnejšia pre dané dáta. Napríklad scatterplot, ktorý sa zdá nasledovať líniu, naznačuje, že dáta sú dobrým kandidátom na cvičenie lineárnej regresie.

Jedna knižnica na vizualizáciu dát, ktorá dobre funguje v Jupyter notebookoch, je [Matplotlib](https://matplotlib.org/) (ktorú ste videli aj v predchádzajúcej lekcii).

> Získajte viac skúseností s vizualizáciou dát v [týchto tutoriáloch](https://docs.microsoft.com/learn/modules/explore-analyze-data-with-python?WT.mc_id=academic-77952-leestott).

## Cvičenie - experimentujte s Matplotlib

Skúste vytvoriť niekoľko základných grafov na zobrazenie nového dataframe, ktorý ste práve vytvorili. Čo by ukázal základný čiarový graf?

1. Importujte Matplotlib na začiatok súboru, pod import Pandas:

    ```python
    import matplotlib.pyplot as plt
    ```

1. Znovu spustite celý notebook, aby ste ho aktualizovali.
1. Na konci notebooku pridajte bunku na vykreslenie dát ako box:

    ```python
    price = new_pumpkins.Price
    month = new_pumpkins.Month
    plt.scatter(price, month)
    plt.show()
    ```

    ![Scatterplot zobrazujúci vzťah medzi cenou a mesiacom](../../../../2-Regression/2-Data/images/scatterplot.png)

    Je tento graf užitočný? Prekvapilo vás na ňom niečo?

    Nie je obzvlášť užitočný, pretože iba zobrazuje vaše dáta ako rozptyl bodov v danom mesiaci.

### Urobte ho užitočným

Aby grafy zobrazovali užitočné dáta, zvyčajne je potrebné dáta nejako zoskupiť. Skúsme vytvoriť graf, kde os y zobrazuje mesiace a dáta ukazujú rozdelenie dát.

1. Pridajte bunku na vytvorenie zoskupeného stĺpcového grafu:

    ```python
    new_pumpkins.groupby(['Month'])['Price'].mean().plot(kind='bar')
    plt.ylabel("Pumpkin Price")
    ```

    ![Stĺpcový graf zobrazujúci vzťah medzi cenou a mesiacom](../../../../2-Regression/2-Data/images/barchart.png)

    Toto je užitočnejšia vizualizácia dát! Zdá sa, že najvyššia cena za tekvice sa vyskytuje v septembri a októbri. Zodpovedá to vašim očakávaniam? Prečo áno alebo nie?

---

## 🚀Výzva

Preskúmajte rôzne typy vizualizácií, ktoré Matplotlib ponúka. Ktoré typy sú najvhodnejšie pre regresné problémy?

## [Kvíz po prednáške](https://ff-quizzes.netlify.app/en/ml/)

## Prehľad a samostatné štúdium

Pozrite sa na mnohé spôsoby vizualizácie dát. Vytvorte zoznam rôznych dostupných knižníc a poznačte si, ktoré sú najlepšie pre dané typy úloh, napríklad 2D vizualizácie vs. 3D vizualizácie. Čo ste zistili?

## Zadanie

[Preskúmanie vizualizácie](assignment.md)

---

**Upozornenie**:  
Tento dokument bol preložený pomocou služby AI prekladu [Co-op Translator](https://github.com/Azure/co-op-translator). Hoci sa snažíme o presnosť, prosím, berte na vedomie, že automatizované preklady môžu obsahovať chyby alebo nepresnosti. Pôvodný dokument v jeho rodnom jazyku by mal byť považovaný za autoritatívny zdroj. Pre kritické informácie sa odporúča profesionálny ľudský preklad. Nenesieme zodpovednosť za akékoľvek nedorozumenia alebo nesprávne interpretácie vyplývajúce z použitia tohto prekladu.