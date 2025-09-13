<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "7c077988328ebfe33b24d07945f16eca",
  "translation_date": "2025-09-05T11:45:44+00:00",
  "source_file": "2-Regression/2-Data/README.md",
  "language_code": "hr"
}
-->
# Izgradnja regresijskog modela pomoću Scikit-learn: priprema i vizualizacija podataka

![Infografika o vizualizaciji podataka](../../../../2-Regression/2-Data/images/data-visualization.png)

Infografiku izradio [Dasani Madipalli](https://twitter.com/dasani_decoded)

## [Kviz prije predavanja](https://ff-quizzes.netlify.app/en/ml/)

> ### [Ova lekcija dostupna je u R-u!](../../../../2-Regression/2-Data/solution/R/lesson_2.html)

## Uvod

Sada kada imate alate potrebne za izgradnju modela strojnog učenja pomoću Scikit-learn, spremni ste početi postavljati pitanja o svojim podacima. Dok radite s podacima i primjenjujete rješenja strojnog učenja, vrlo je važno znati kako postaviti pravo pitanje kako biste pravilno iskoristili potencijale svog skupa podataka.

U ovoj lekciji naučit ćete:

- Kako pripremiti podatke za izgradnju modela.
- Kako koristiti Matplotlib za vizualizaciju podataka.

## Postavljanje pravog pitanja o podacima

Pitanje na koje trebate odgovoriti odredit će vrstu algoritama strojnog učenja koje ćete koristiti. Kvaliteta odgovora koji dobijete uvelike će ovisiti o prirodi vaših podataka.

Pogledajte [podatke](https://github.com/microsoft/ML-For-Beginners/blob/main/2-Regression/data/US-pumpkins.csv) koji su dostupni za ovu lekciju. Možete otvoriti ovu .csv datoteku u VS Codeu. Brzi pregled odmah pokazuje da postoje praznine i mješavina tekstualnih i numeričkih podataka. Tu je i neobičan stupac nazvan 'Package' gdje su podaci mješavina između 'sacks', 'bins' i drugih vrijednosti. Podaci su, zapravo, pomalo neuredni.

[![ML za početnike - Kako analizirati i očistiti skup podataka](https://img.youtube.com/vi/5qGjczWTrDQ/0.jpg)](https://youtu.be/5qGjczWTrDQ "ML za početnike - Kako analizirati i očistiti skup podataka")

> 🎥 Kliknite na sliku iznad za kratki video o pripremi podataka za ovu lekciju.

Zapravo, nije uobičajeno dobiti skup podataka koji je potpuno spreman za korištenje u stvaranju modela strojnog učenja. U ovoj lekciji naučit ćete kako pripremiti sirove podatke koristeći standardne Python biblioteke. Također ćete naučiti različite tehnike za vizualizaciju podataka.

## Studija slučaja: 'tržište bundeva'

U ovom direktoriju pronaći ćete .csv datoteku u korijenskoj mapi `data` pod nazivom [US-pumpkins.csv](https://github.com/microsoft/ML-For-Beginners/blob/main/2-Regression/data/US-pumpkins.csv) koja uključuje 1757 redaka podataka o tržištu bundeva, grupiranih po gradovima. Ovo su sirovi podaci izvučeni iz [Specialty Crops Terminal Markets Standard Reports](https://www.marketnews.usda.gov/mnp/fv-report-config-step1?type=termPrice) koje distribuira Ministarstvo poljoprivrede Sjedinjenih Američkih Država.

### Priprema podataka

Ovi podaci su javno dostupni. Mogu se preuzeti u mnogo zasebnih datoteka, po gradovima, s web stranice USDA. Kako bismo izbjegli previše zasebnih datoteka, spojili smo sve podatke po gradovima u jednu tablicu, tako da smo već _pripremili_ podatke malo. Sada, pogledajmo podatke detaljnije.

### Podaci o bundevama - prvi zaključci

Što primjećujete o ovim podacima? Već ste vidjeli da postoji mješavina tekstualnih podataka, brojeva, praznina i neobičnih vrijednosti koje trebate razumjeti.

Koje pitanje možete postaviti o ovim podacima koristeći tehniku regresije? Što kažete na "Predvidjeti cijenu bundeve za prodaju tijekom određenog mjeseca". Ponovno pogledajući podatke, postoje neke promjene koje trebate napraviti kako biste stvorili strukturu podataka potrebnu za zadatak.

## Vježba - analizirajte podatke o bundevama

Koristimo [Pandas](https://pandas.pydata.org/), (ime dolazi od `Python Data Analysis`) alat vrlo koristan za oblikovanje podataka, kako bismo analizirali i pripremili ove podatke o bundevama.

### Prvo, provjerite nedostaju li datumi

Prvo ćete morati poduzeti korake kako biste provjerili nedostaju li datumi:

1. Pretvorite datume u format mjeseca (ovo su američki datumi, pa je format `MM/DD/YYYY`).
2. Izvucite mjesec u novi stupac.

Otvorite datoteku _notebook.ipynb_ u Visual Studio Codeu i uvezite tablicu u novi Pandas dataframe.

1. Koristite funkciju `head()` za pregled prvih pet redaka.

    ```python
    import pandas as pd
    pumpkins = pd.read_csv('../data/US-pumpkins.csv')
    pumpkins.head()
    ```

    ✅ Koju biste funkciju koristili za pregled zadnjih pet redaka?

1. Provjerite ima li nedostajućih podataka u trenutnom dataframeu:

    ```python
    pumpkins.isnull().sum()
    ```

    Postoje nedostajući podaci, ali možda neće biti važni za zadatak.

1. Kako biste olakšali rad s dataframeom, odaberite samo stupce koji su vam potrebni koristeći funkciju `loc`, koja iz izvornog dataframea izvlači grupu redaka (proslijeđeno kao prvi parametar) i stupaca (proslijeđeno kao drugi parametar). Izraz `:` u slučaju ispod znači "svi redovi".

    ```python
    columns_to_select = ['Package', 'Low Price', 'High Price', 'Date']
    pumpkins = pumpkins.loc[:, columns_to_select]
    ```

### Drugo, odredite prosječnu cijenu bundeve

Razmislite kako odrediti prosječnu cijenu bundeve u određenom mjesecu. Koje biste stupce odabrali za ovaj zadatak? Savjet: trebat će vam 3 stupca.

Rješenje: uzmite prosjek stupaca `Low Price` i `High Price` kako biste popunili novi stupac Price, i pretvorite stupac Date da prikazuje samo mjesec. Srećom, prema provjeri iznad, nema nedostajućih podataka za datume ili cijene.

1. Za izračunavanje prosjeka dodajte sljedeći kod:

    ```python
    price = (pumpkins['Low Price'] + pumpkins['High Price']) / 2

    month = pd.DatetimeIndex(pumpkins['Date']).month

    ```

   ✅ Slobodno ispišite bilo koje podatke koje želite provjeriti koristeći `print(month)`.

2. Sada kopirajte svoje konvertirane podatke u novi Pandas dataframe:

    ```python
    new_pumpkins = pd.DataFrame({'Month': month, 'Package': pumpkins['Package'], 'Low Price': pumpkins['Low Price'],'High Price': pumpkins['High Price'], 'Price': price})
    ```

    Ispisivanje vašeg dataframea pokazat će vam čist, uredan skup podataka na kojem možete izgraditi svoj novi regresijski model.

### Ali čekajte! Nešto ovdje izgleda čudno

Ako pogledate stupac `Package`, bundeve se prodaju u mnogim različitim konfiguracijama. Neke se prodaju u mjerama '1 1/9 bushel', neke u '1/2 bushel', neke po bundevi, neke po funti, a neke u velikim kutijama različitih širina.

> Čini se da je bundeve vrlo teško dosljedno vagati

Kopajući po izvornim podacima, zanimljivo je da sve što ima `Unit of Sale` jednako 'EACH' ili 'PER BIN' također ima tip `Package` po inču, po binu ili 'each'. Čini se da je bundeve vrlo teško dosljedno vagati, pa ih filtrirajmo odabirom samo bundeva s nizom 'bushel' u stupcu `Package`.

1. Dodajte filter na vrh datoteke, ispod početnog uvoza .csv datoteke:

    ```python
    pumpkins = pumpkins[pumpkins['Package'].str.contains('bushel', case=True, regex=True)]
    ```

    Ako sada ispišete podatke, možete vidjeti da dobivate samo oko 415 redaka podataka koji sadrže bundeve po bushelu.

### Ali čekajte! Još nešto treba napraviti

Jeste li primijetili da se količina bushela razlikuje po retku? Trebate normalizirati cijene tako da prikazujete cijene po bushelu, pa napravite neke izračune kako biste ih standardizirali.

1. Dodajte ove linije nakon bloka koji stvara dataframe new_pumpkins:

    ```python
    new_pumpkins.loc[new_pumpkins['Package'].str.contains('1 1/9'), 'Price'] = price/(1 + 1/9)

    new_pumpkins.loc[new_pumpkins['Package'].str.contains('1/2'), 'Price'] = price/(1/2)
    ```

✅ Prema [The Spruce Eats](https://www.thespruceeats.com/how-much-is-a-bushel-1389308), težina bushela ovisi o vrsti proizvoda, jer je to mjera volumena. "Bushel rajčica, na primjer, trebao bi težiti 56 funti... Lišće i zelje zauzimaju više prostora s manje težine, pa bushel špinata teži samo 20 funti." Sve je to prilično komplicirano! Nećemo se zamarati konverzijom bushel-u-funtu, već ćemo cijene prikazivati po bushelu. Sva ova studija bushela bundeva, međutim, pokazuje koliko je važno razumjeti prirodu svojih podataka!

Sada možete analizirati cijene po jedinici na temelju njihove mjere bushela. Ako još jednom ispišete podatke, možete vidjeti kako su standardizirani.

✅ Jeste li primijetili da su bundeve prodane po pola bushela vrlo skupe? Možete li shvatiti zašto? Savjet: male bundeve su puno skuplje od velikih, vjerojatno zato što ih ima puno više po bushelu, s obzirom na neiskorišten prostor koji zauzima jedna velika šuplja bundeva za pitu.

## Strategije vizualizacije

Dio uloge znanstvenika za podatke je demonstrirati kvalitetu i prirodu podataka s kojima rade. Da bi to učinili, često stvaraju zanimljive vizualizacije, poput grafikona, dijagrama i tablica, koje prikazuju različite aspekte podataka. Na taj način mogu vizualno pokazati odnose i praznine koje je inače teško otkriti.

[![ML za početnike - Kako vizualizirati podatke pomoću Matplotliba](https://img.youtube.com/vi/SbUkxH6IJo0/0.jpg)](https://youtu.be/SbUkxH6IJo0 "ML za početnike - Kako vizualizirati podatke pomoću Matplotliba")

> 🎥 Kliknite na sliku iznad za kratki video o vizualizaciji podataka za ovu lekciju.

Vizualizacije također mogu pomoći u određivanju tehnike strojnog učenja koja je najprikladnija za podatke. Na primjer, scatterplot koji izgleda kao da slijedi liniju ukazuje na to da su podaci dobar kandidat za vježbu linearne regresije.

Jedna biblioteka za vizualizaciju podataka koja dobro funkcionira u Jupyter notebookovima je [Matplotlib](https://matplotlib.org/) (koju ste također vidjeli u prethodnoj lekciji).

> Steknite više iskustva s vizualizacijom podataka u [ovim tutorijalima](https://docs.microsoft.com/learn/modules/explore-analyze-data-with-python?WT.mc_id=academic-77952-leestott).

## Vježba - eksperimentirajte s Matplotlibom

Pokušajte stvoriti osnovne grafikone za prikaz novog dataframea koji ste upravo stvorili. Što bi pokazao osnovni linijski grafikon?

1. Uvezite Matplotlib na vrh datoteke, ispod uvoza Pandasa:

    ```python
    import matplotlib.pyplot as plt
    ```

1. Ponovno pokrenite cijeli notebook za osvježavanje.
1. Na dnu notebooka dodajte ćeliju za prikaz podataka kao kutiju:

    ```python
    price = new_pumpkins.Price
    month = new_pumpkins.Month
    plt.scatter(price, month)
    plt.show()
    ```

    ![Scatterplot koji prikazuje odnos cijene i mjeseca](../../../../2-Regression/2-Data/images/scatterplot.png)

    Je li ovo koristan grafikon? Iznenađuje li vas nešto u vezi s njim?

    Nije osobito koristan jer samo prikazuje vaše podatke kao raspršene točke u određenom mjesecu.

### Učinite ga korisnim

Da bi grafikoni prikazivali korisne podatke, obično trebate grupirati podatke na neki način. Pokušajmo stvoriti grafikon gdje y os prikazuje mjesece, a podaci pokazuju distribuciju podataka.

1. Dodajte ćeliju za stvaranje grupiranog stupčastog grafikona:

    ```python
    new_pumpkins.groupby(['Month'])['Price'].mean().plot(kind='bar')
    plt.ylabel("Pumpkin Price")
    ```

    ![Stupčasti grafikon koji prikazuje odnos cijene i mjeseca](../../../../2-Regression/2-Data/images/barchart.png)

    Ovo je korisnija vizualizacija podataka! Čini se da pokazuje da je najviša cijena bundeva u rujnu i listopadu. Odgovara li to vašim očekivanjima? Zašto ili zašto ne?

---

## 🚀Izazov

Istražite različite vrste vizualizacija koje Matplotlib nudi. Koje vrste su najprikladnije za regresijske probleme?

## [Kviz nakon predavanja](https://ff-quizzes.netlify.app/en/ml/)

## Pregled i samostalno učenje

Pogledajte mnoge načine vizualizacije podataka. Napravite popis raznih dostupnih biblioteka i zabilježite koje su najbolje za određene vrste zadataka, na primjer 2D vizualizacije naspram 3D vizualizacija. Što otkrivate?

## Zadatak

[Istrazivanje vizualizacije](assignment.md)

---

**Odricanje od odgovornosti**:  
Ovaj dokument je preveden korištenjem AI usluge za prevođenje [Co-op Translator](https://github.com/Azure/co-op-translator). Iako nastojimo osigurati točnost, imajte na umu da automatski prijevodi mogu sadržavati pogreške ili netočnosti. Izvorni dokument na izvornom jeziku treba smatrati mjerodavnim izvorom. Za ključne informacije preporučuje se profesionalni prijevod od strane stručnjaka. Ne preuzimamo odgovornost za bilo kakve nesporazume ili pogrešne interpretacije proizašle iz korištenja ovog prijevoda.