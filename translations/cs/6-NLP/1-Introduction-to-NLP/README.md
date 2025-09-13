<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1c2ec40cf55c98a028a359c27ef7e45a",
  "translation_date": "2025-09-05T01:34:17+00:00",
  "source_file": "6-NLP/1-Introduction-to-NLP/README.md",
  "language_code": "cs"
}
-->
# Úvod do zpracování přirozeného jazyka

Tato lekce se zabývá stručnou historií a důležitými koncepty *zpracování přirozeného jazyka*, což je podoblast *počítačové lingvistiky*.

## [Kvíz před lekcí](https://ff-quizzes.netlify.app/en/ml/)

## Úvod

NLP, jak se běžně nazývá, je jednou z nejznámějších oblastí, kde bylo strojové učení aplikováno a používáno v produkčním softwaru.

✅ Dokážete si představit software, který používáte každý den a který pravděpodobně obsahuje nějaké NLP? Co třeba vaše textové procesory nebo mobilní aplikace, které pravidelně používáte?

Dozvíte se o:

- **Myšlence jazyků**. Jak se jazyky vyvíjely a jaké byly hlavní oblasti studia.
- **Definici a konceptech**. Naučíte se také definice a koncepty o tom, jak počítače zpracovávají text, včetně analýzy, gramatiky a identifikace podstatných jmen a sloves. V této lekci jsou některé programovací úkoly a několik důležitých konceptů, které se později naučíte programovat v dalších lekcích.

## Počítačová lingvistika

Počítačová lingvistika je oblast výzkumu a vývoje, která se po mnoho desetiletí zabývá tím, jak mohou počítače pracovat s jazyky, rozumět jim, překládat je a komunikovat v nich. Zpracování přirozeného jazyka (NLP) je příbuzná oblast zaměřená na to, jak mohou počítače zpracovávat „přirozené“, tedy lidské jazyky.

### Příklad - diktování na telefonu

Pokud jste někdy diktovali svému telefonu místo psaní nebo se ptali virtuálního asistenta na otázku, váš hlas byl převeden do textové podoby a poté zpracován nebo *analyzován* z jazyka, kterým jste mluvili. Detekovaná klíčová slova byla poté zpracována do formátu, kterému telefon nebo asistent rozuměl a mohl na něj reagovat.

![porozumění](../../../../6-NLP/1-Introduction-to-NLP/images/comprehension.png)
> Skutečné jazykové porozumění je těžké! Obrázek od [Jen Looper](https://twitter.com/jenlooper)

### Jak je tato technologie možná?

To je možné díky tomu, že někdo napsal počítačový program, který to umožňuje. Před několika desetiletími někteří autoři sci-fi předpovídali, že lidé budou většinou mluvit se svými počítači a počítače vždy přesně pochopí, co tím myslí. Bohužel se ukázalo, že je to těžší problém, než si mnozí představovali, a přestože je dnes mnohem lépe pochopený, stále existují významné výzvy při dosažení „dokonalého“ zpracování přirozeného jazyka, pokud jde o pochopení významu věty. To je obzvláště obtížné, pokud jde o pochopení humoru nebo detekci emocí, jako je sarkasmus, ve větě.

V tuto chvíli si možná vzpomínáte na školní hodiny, kde učitel probíral části gramatiky ve větě. V některých zemích se studenti učí gramatiku a lingvistiku jako samostatný předmět, ale v mnoha zemích jsou tyto témata zahrnuty jako součást výuky jazyka: buď vašeho prvního jazyka na základní škole (učení čtení a psaní) a možná druhého jazyka na střední škole. Nemějte obavy, pokud nejste odborníkem na rozlišování podstatných jmen od sloves nebo příslovcí od přídavných jmen!

Pokud máte potíže s rozdílem mezi *jednoduchým přítomným časem* a *přítomným průběhovým časem*, nejste sami. To je náročné pro mnoho lidí, dokonce i rodilé mluvčí jazyka. Dobrou zprávou je, že počítače jsou opravdu dobré v aplikaci formálních pravidel, a naučíte se psát kód, který dokáže *analyzovat* větu stejně dobře jako člověk. Větší výzvou, kterou později prozkoumáte, je pochopení *významu* a *sentimentu* věty.

## Předpoklady

Pro tuto lekci je hlavním předpokladem schopnost číst a rozumět jazyku této lekce. Nejsou zde žádné matematické problémy ani rovnice k řešení. Zatímco původní autor napsal tuto lekci v angličtině, je také přeložena do jiných jazyků, takže byste mohli číst překlad. Existují příklady, kde je použito několik různých jazyků (pro porovnání různých gramatických pravidel různých jazyků). Tyto *nejsou* přeloženy, ale vysvětlující text ano, takže význam by měl být jasný.

Pro programovací úkoly budete používat Python a příklady jsou v Pythonu 3.8.

V této části budete potřebovat a používat:

- **Porozumění Pythonu 3**. Porozumění programovacímu jazyku Python 3, tato lekce používá vstupy, smyčky, čtení souborů, pole.
- **Visual Studio Code + rozšíření**. Budeme používat Visual Studio Code a jeho rozšíření pro Python. Můžete také použít Python IDE dle svého výběru.
- **TextBlob**. [TextBlob](https://github.com/sloria/TextBlob) je zjednodušená knihovna pro zpracování textu v Pythonu. Postupujte podle pokynů na stránkách TextBlob pro instalaci na váš systém (nainstalujte také korpusy, jak je uvedeno níže):

   ```bash
   pip install -U textblob
   python -m textblob.download_corpora
   ```

> 💡 Tip: Python můžete spouštět přímo v prostředí VS Code. Podívejte se na [dokumentaci](https://code.visualstudio.com/docs/languages/python?WT.mc_id=academic-77952-leestott) pro více informací.

## Komunikace s počítači

Historie snahy přimět počítače, aby rozuměly lidskému jazyku, sahá desítky let zpět, a jedním z prvních vědců, kteří se zabývali zpracováním přirozeného jazyka, byl *Alan Turing*.

### Turingův test

Když Turing v 50. letech 20. století zkoumal *umělou inteligenci*, uvažoval, zda by mohl být člověku a počítači (prostřednictvím psané komunikace) zadán konverzační test, při kterém by člověk v konverzaci nebyl schopen určit, zda komunikuje s jiným člověkem nebo počítačem.

Pokud by po určité délce konverzace člověk nemohl určit, zda odpovědi pocházejí od počítače nebo ne, mohl by být počítač považován za *myslící*?

### Inspirace - hra „imitační hra“

Myšlenka na to přišla z párty hry nazvané *Imitační hra*, kde je vyšetřovatel sám v místnosti a má za úkol určit, kdo ze dvou lidí (v jiné místnosti) je muž a žena. Vyšetřovatel může posílat poznámky a musí se snažit vymyslet otázky, na které písemné odpovědi odhalí pohlaví tajemné osoby. Samozřejmě, hráči v jiné místnosti se snaží zmást vyšetřovatele tím, že odpovídají na otázky takovým způsobem, aby ho uvedli v omyl nebo zmátli, zatímco zároveň dávají dojem, že odpovídají upřímně.

### Vývoj Elizy

V 60. letech 20. století vyvinul vědec z MIT *Joseph Weizenbaum* [*Elizu*](https://wikipedia.org/wiki/ELIZA), počítačového „terapeuta“, který by kladl člověku otázky a dával dojem, že rozumí jeho odpovědím. Nicméně, zatímco Eliza dokázala analyzovat větu a identifikovat určité gramatické konstrukce a klíčová slova, aby dala rozumnou odpověď, nemohlo se říci, že větu *rozumí*. Pokud byla Elize předložena věta ve formátu "**Jsem** <u>smutný</u>", mohla by přeskupit a nahradit slova ve větě, aby vytvořila odpověď "Jak dlouho jste **smutný**?".

To dávalo dojem, že Eliza rozumí tvrzení a klade následnou otázku, zatímco ve skutečnosti měnila čas a přidávala některá slova. Pokud Eliza nemohla identifikovat klíčové slovo, na které měla odpověď, místo toho by dala náhodnou odpověď, která by mohla být použitelná pro mnoho různých tvrzení. Elizu bylo snadné oklamat, například pokud uživatel napsal "**Ty jsi** <u>kolo</u>", mohla by odpovědět "Jak dlouho jsem **kolo**?", místo rozumnější odpovědi.

[![Rozhovor s Elizou](https://img.youtube.com/vi/RMK9AphfLco/0.jpg)](https://youtu.be/RMK9AphfLco "Rozhovor s Elizou")

> 🎥 Klikněte na obrázek výše pro video o původním programu ELIZA

> Poznámka: Původní popis [Elizy](https://cacm.acm.org/magazines/1966/1/13317-elizaa-computer-program-for-the-study-of-natural-language-communication-between-man-and-machine/abstract) publikovaný v roce 1966 si můžete přečíst, pokud máte účet ACM. Alternativně si přečtěte o Elize na [wikipedii](https://wikipedia.org/wiki/ELIZA).

## Cvičení - programování základního konverzačního bota

Konverzační bot, jako Eliza, je program, který vyžaduje vstup od uživatele a zdá se, že rozumí a inteligentně reaguje. Na rozdíl od Elizy náš bot nebude mít několik pravidel, která by mu dávala dojem inteligentní konverzace. Místo toho bude mít pouze jednu schopnost, a to pokračovat v konverzaci s náhodnými odpověďmi, které by mohly fungovat téměř v jakékoli triviální konverzaci.

### Plán

Vaše kroky při vytváření konverzačního bota:

1. Vytiskněte pokyny, které uživateli poradí, jak komunikovat s botem.
2. Spusťte smyčku.
   1. Přijměte vstup od uživatele.
   2. Pokud uživatel požádá o ukončení, ukončete.
   3. Zpracujte vstup uživatele a určete odpověď (v tomto případě je odpověď náhodný výběr ze seznamu možných obecných odpovědí).
   4. Vytiskněte odpověď.
3. Vraťte se zpět ke kroku 2.

### Vytvoření bota

Pojďme nyní vytvořit bota. Začneme definováním některých frází.

1. Vytvořte si tohoto bota sami v Pythonu s následujícími náhodnými odpověďmi:

    ```python
    random_responses = ["That is quite interesting, please tell me more.",
                        "I see. Do go on.",
                        "Why do you say that?",
                        "Funny weather we've been having, isn't it?",
                        "Let's change the subject.",
                        "Did you catch the game last night?"]
    ```

    Zde je ukázkový výstup, který vás může vést (vstup uživatele je na řádcích začínajících `>`):

    ```output
    Hello, I am Marvin, the simple robot.
    You can end this conversation at any time by typing 'bye'
    After typing each answer, press 'enter'
    How are you today?
    > I am good thanks
    That is quite interesting, please tell me more.
    > today I went for a walk     
    Did you catch the game last night?
    > I did, but my team lost
    Funny weather we've been having, isn't it?
    > yes but I hope next week is better
    Let's change the subject.
    > ok, lets talk about music
    Why do you say that?
    > because I like music!
    Why do you say that?
    > bye
    It was nice talking to you, goodbye!
    ```

    Jedno možné řešení úkolu je [zde](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/1-Introduction-to-NLP/solution/bot.py)

    ✅ Zastavte se a zamyslete se

    1. Myslíte si, že náhodné odpovědi by „oklamaly“ někoho, aby si myslel, že bot skutečně rozumí?
    2. Jaké funkce by bot potřeboval, aby byl efektivnější?
    3. Pokud by bot skutečně „rozuměl“ významu věty, potřeboval by si „pamatovat“ význam předchozích vět v konverzaci?

---

## 🚀Výzva

Vyberte si jeden z prvků „zastavte se a zamyslete se“ výše a buď se ho pokuste implementovat v kódu, nebo napište řešení na papír pomocí pseudokódu.

V další lekci se dozvíte o řadě dalších přístupů k analýze přirozeného jazyka a strojovému učení.

## [Kvíz po lekci](https://ff-quizzes.netlify.app/en/ml/)

## Přehled & Samostudium

Podívejte se na níže uvedené odkazy jako příležitosti k dalšímu čtení.

### Odkazy

1. Schubert, Lenhart, "Computational Linguistics", *The Stanford Encyclopedia of Philosophy* (Spring 2020 Edition), Edward N. Zalta (ed.), URL = <https://plato.stanford.edu/archives/spr2020/entries/computational-linguistics/>.
2. Princeton University "About WordNet." [WordNet](https://wordnet.princeton.edu/). Princeton University. 2010. 

## Úkol 

[Vyhledejte bota](assignment.md)

---

**Prohlášení**:  
Tento dokument byl přeložen pomocí služby pro automatický překlad [Co-op Translator](https://github.com/Azure/co-op-translator). Ačkoli se snažíme o přesnost, mějte prosím na paměti, že automatické překlady mohou obsahovat chyby nebo nepřesnosti. Původní dokument v jeho původním jazyce by měl být považován za závazný zdroj. Pro důležité informace doporučujeme profesionální lidský překlad. Neodpovídáme za žádná nedorozumění nebo nesprávné interpretace vyplývající z použití tohoto překladu.