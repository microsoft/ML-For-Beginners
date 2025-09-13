<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1c2ec40cf55c98a028a359c27ef7e45a",
  "translation_date": "2025-09-05T17:00:35+00:00",
  "source_file": "6-NLP/1-Introduction-to-NLP/README.md",
  "language_code": "sk"
}
-->
# Úvod do spracovania prirodzeného jazyka

Táto lekcia pokrýva stručnú históriu a dôležité koncepty *spracovania prirodzeného jazyka*, podpolia *počítačovej lingvistiky*.

## [Kvíz pred prednáškou](https://ff-quizzes.netlify.app/en/ml/)

## Úvod

NLP, ako sa často nazýva, je jednou z najznámejších oblastí, kde sa strojové učenie aplikovalo a používa v produkčnom softvéri.

✅ Dokážete si predstaviť softvér, ktorý používate každý deň a pravdepodobne obsahuje nejaké NLP? Čo tak vaše programy na spracovanie textu alebo mobilné aplikácie, ktoré pravidelne používate?

Naučíte sa o:

- **Myšlienke jazykov**. Ako sa jazyky vyvíjali a aké boli hlavné oblasti štúdia.
- **Definíciách a konceptoch**. Naučíte sa definície a koncepty o tom, ako počítače spracovávajú text, vrátane parsovania, gramatiky a identifikácie podstatných mien a slovies. V tejto lekcii sú niektoré úlohy na programovanie a predstavené sú viaceré dôležité koncepty, ktoré sa naučíte programovať v ďalších lekciách.

## Počítačová lingvistika

Počítačová lingvistika je oblasť výskumu a vývoja, ktorá sa už desaťročia zaoberá tým, ako môžu počítače pracovať s jazykmi, rozumieť im, prekladať ich a komunikovať v nich. Spracovanie prirodzeného jazyka (NLP) je príbuzná oblasť zameraná na to, ako môžu počítače spracovávať 'prirodzené', teda ľudské jazyky.

### Príklad - diktovanie do telefónu

Ak ste niekedy diktovali do telefónu namiesto písania alebo sa pýtali virtuálneho asistenta otázku, vaša reč bola prevedená do textovej formy a následne spracovaná alebo *parsovaná* z jazyka, ktorým ste hovorili. Zistené kľúčové slová boli potom spracované do formátu, ktorému telefón alebo asistent rozumel a na základe toho konal.

![porozumenie](../../../../6-NLP/1-Introduction-to-NLP/images/comprehension.png)
> Skutočné lingvistické porozumenie je náročné! Obrázok od [Jen Looper](https://twitter.com/jenlooper)

### Ako je táto technológia možná?

Je to možné, pretože niekto napísal počítačový program, ktorý to dokáže. Pred niekoľkými desaťročiami niektorí autori sci-fi predpovedali, že ľudia budú väčšinou hovoriť so svojimi počítačmi a počítače budú vždy presne rozumieť tomu, čo majú na mysli. Bohužiaľ sa ukázalo, že je to oveľa ťažší problém, než si mnohí predstavovali, a hoci je dnes oveľa lepšie pochopený, stále existujú významné výzvy pri dosahovaní 'dokonalého' spracovania prirodzeného jazyka, pokiaľ ide o porozumenie významu vety. Toto je obzvlášť náročné pri porozumení humoru alebo detekcii emócií, ako je sarkazmus vo vete.

V tomto momente si možno spomeniete na školské hodiny, kde učiteľ preberal časti gramatiky vo vete. V niektorých krajinách sa študenti učia gramatiku a lingvistiku ako samostatný predmet, ale v mnohých sú tieto témy zahrnuté ako súčasť učenia sa jazyka: buď vášho prvého jazyka na základnej škole (učenie sa čítať a písať) a možno druhého jazyka na strednej škole. Nemajte obavy, ak nie ste odborníkom na rozlišovanie podstatných mien od slovies alebo prísloviek od prídavných mien!

Ak máte problém s rozdielom medzi *jednoduchým prítomným časom* a *prítomným priebehovým časom*, nie ste sami. Toto je náročná vec pre mnohých ľudí, dokonca aj pre rodených hovorcov jazyka. Dobrou správou je, že počítače sú veľmi dobré v aplikovaní formálnych pravidiel, a naučíte sa písať kód, ktorý dokáže *parsovať* vetu rovnako dobre ako človek. Väčšou výzvou, ktorú budete skúmať neskôr, je porozumenie *významu* a *sentimentu* vety.

## Predpoklady

Pre túto lekciu je hlavným predpokladom schopnosť čítať a rozumieť jazyku tejto lekcie. Neexistujú žiadne matematické problémy ani rovnice na riešenie. Hoci pôvodný autor napísal túto lekciu v angličtine, je tiež preložená do iných jazykov, takže by ste mohli čítať preklad. Existujú príklady, kde sa používa niekoľko rôznych jazykov (na porovnanie rôznych gramatických pravidiel rôznych jazykov). Tieto *nie sú* preložené, ale vysvetľujúci text áno, takže význam by mal byť jasný.

Pre úlohy na programovanie budete používať Python a príklady sú v Python 3.8.

V tejto sekcii budete potrebovať a používať:

- **Porozumenie Pythonu 3**. Porozumenie programovaciemu jazyku Python 3, táto lekcia používa vstupy, cykly, čítanie súborov, polia.
- **Visual Studio Code + rozšírenie**. Budeme používať Visual Studio Code a jeho rozšírenie pre Python. Môžete tiež použiť IDE pre Python podľa vášho výberu.
- **TextBlob**. [TextBlob](https://github.com/sloria/TextBlob) je zjednodušená knižnica na spracovanie textu pre Python. Postupujte podľa pokynov na stránke TextBlob na jeho inštaláciu do vášho systému (nainštalujte aj korpusy, ako je uvedené nižšie):

   ```bash
   pip install -U textblob
   python -m textblob.download_corpora
   ```

> 💡 Tip: Python môžete spustiť priamo v prostredí VS Code. Pozrite si [dokumentáciu](https://code.visualstudio.com/docs/languages/python?WT.mc_id=academic-77952-leestott) pre viac informácií.

## Rozprávanie s počítačmi

História snahy prinútiť počítače rozumieť ľudskému jazyku siaha desaťročia dozadu a jedným z prvých vedcov, ktorí sa zaoberali spracovaním prirodzeného jazyka, bol *Alan Turing*.

### Turingov test

Keď Turing skúmal *umelú inteligenciu* v 50. rokoch, uvažoval, či by sa mohol uskutočniť konverzačný test medzi človekom a počítačom (prostredníctvom písomnej komunikácie), kde by človek v konverzácii nevedel, či komunikuje s iným človekom alebo počítačom.

Ak by po určitej dĺžke konverzácie človek nedokázal určiť, či odpovede pochádzajú od počítača alebo nie, mohlo by sa povedať, že počítač *myslí*?

### Inšpirácia - 'hra na imitáciu'

Myšlienka na to vznikla z párty hry nazývanej *Hra na imitáciu*, kde je vyšetrovateľ sám v miestnosti a má za úlohu určiť, kto z dvoch ľudí (v inej miestnosti) je muž a žena. Vyšetrovateľ môže posielať poznámky a musí sa snažiť vymyslieť otázky, kde písomné odpovede odhalia pohlavie tajomnej osoby. Samozrejme, hráči v druhej miestnosti sa snažia zmiasť vyšetrovateľa tým, že odpovedajú na otázky takým spôsobom, aby ho zaviedli alebo zmiatli, pričom zároveň dávajú dojem, že odpovedajú úprimne.

### Vývoj Elizy

V 60. rokoch vyvinul vedec z MIT *Joseph Weizenbaum* [*Elizu*](https://wikipedia.org/wiki/ELIZA), počítačového 'terapeuta', ktorý kládol človeku otázky a dával dojem, že rozumie jeho odpovediam. Avšak, hoci Eliza dokázala parsovať vetu a identifikovať určité gramatické konštrukty a kľúčové slová, aby poskytla rozumnú odpoveď, nedalo sa povedať, že vetu *rozumie*. Ak by bola Eliza konfrontovaná s vetou vo formáte "**Som** <u>smutný</u>", mohla by preusporiadať a nahradiť slová vo vete, aby vytvorila odpoveď "Ako dlho ste **vy** <u>smutný</u>?".

To dávalo dojem, že Eliza rozumie výroku a kladie následnú otázku, zatiaľ čo v skutočnosti iba menila čas a pridávala niektoré slová. Ak by Eliza nedokázala identifikovať kľúčové slovo, na ktoré mala odpoveď, namiesto toho by poskytla náhodnú odpoveď, ktorá by mohla byť použiteľná pre mnoho rôznych výrokov. Elizu bolo možné ľahko oklamať, napríklad ak by používateľ napísal "**Ty si** <u>bicykel</u>", mohla by odpovedať "Ako dlho som **ja** <u>bicykel</u>?", namiesto rozumnejšej odpovede.

[![Rozhovor s Elizou](https://img.youtube.com/vi/RMK9AphfLco/0.jpg)](https://youtu.be/RMK9AphfLco "Rozhovor s Elizou")

> 🎥 Kliknite na obrázok vyššie pre video o pôvodnom programe ELIZA

> Poznámka: Pôvodný popis [Elizy](https://cacm.acm.org/magazines/1966/1/13317-elizaa-computer-program-for-the-study-of-natural-language-communication-between-man-and-machine/abstract) publikovaný v roku 1966 si môžete prečítať, ak máte účet ACM. Alternatívne si prečítajte o Elize na [wikipédii](https://wikipedia.org/wiki/ELIZA).

## Cvičenie - programovanie základného konverzačného bota

Konverzačný bot, ako Eliza, je program, ktorý vyžaduje vstup od používateľa a zdá sa, že rozumie a inteligentne odpovedá. Na rozdiel od Elizy náš bot nebude mať niekoľko pravidiel, ktoré by mu dávali dojem inteligentnej konverzácie. Namiesto toho bude mať bot iba jednu schopnosť, a to pokračovať v konverzácii s náhodnými odpoveďami, ktoré by mohli fungovať takmer v každej triviálnej konverzácii.

### Plán

Vaše kroky pri vytváraní konverzačného bota:

1. Vytlačte pokyny, ktoré používateľovi poradia, ako komunikovať s botom
2. Spustite cyklus
   1. Prijmite vstup od používateľa
   2. Ak používateľ požiadal o ukončenie, ukončite
   3. Spracujte vstup od používateľa a určte odpoveď (v tomto prípade je odpoveď náhodný výber zo zoznamu možných všeobecných odpovedí)
   4. Vytlačte odpoveď
3. Vráťte sa späť na krok 2

### Vytvorenie bota

Teraz vytvoríme bota. Začneme definovaním niektorých fráz.

1. Vytvorte si tohto bota sami v Pythone s nasledujúcimi náhodnými odpoveďami:

    ```python
    random_responses = ["That is quite interesting, please tell me more.",
                        "I see. Do go on.",
                        "Why do you say that?",
                        "Funny weather we've been having, isn't it?",
                        "Let's change the subject.",
                        "Did you catch the game last night?"]
    ```

    Tu je ukážka výstupu, ktorá vás môže usmerniť (vstup používateľa je na riadkoch začínajúcich `>`):

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

    Jedno možné riešenie úlohy nájdete [tu](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/1-Introduction-to-NLP/solution/bot.py)

    ✅ Zastavte sa a zamyslite

    1. Myslíte si, že náhodné odpovede by 'oklamali' niekoho, aby si myslel, že bot skutočne rozumie?
    2. Aké funkcie by bot potreboval, aby bol efektívnejší?
    3. Ak by bot skutočne 'rozumel' významu vety, potreboval by si 'pamätať' význam predchádzajúcich viet v konverzácii tiež?

---

## 🚀Výzva

Vyberte si jeden z prvkov "zastavte sa a zamyslite" vyššie a buď sa ho pokúste implementovať v kóde, alebo napíšte riešenie na papier pomocou pseudokódu.

V ďalšej lekcii sa naučíte o viacerých prístupoch k parsovaniu prirodzeného jazyka a strojovému učeniu.

## [Kvíz po prednáške](https://ff-quizzes.netlify.app/en/ml/)

## Prehľad a samostatné štúdium

Pozrite si nižšie uvedené odkazy ako ďalšie možnosti čítania.

### Odkazy

1. Schubert, Lenhart, "Computational Linguistics", *The Stanford Encyclopedia of Philosophy* (Spring 2020 Edition), Edward N. Zalta (ed.), URL = <https://plato.stanford.edu/archives/spr2020/entries/computational-linguistics/>.
2. Princeton University "About WordNet." [WordNet](https://wordnet.princeton.edu/). Princeton University. 2010. 

## Zadanie 

[Vyhľadajte bota](assignment.md)

---

**Upozornenie**:  
Tento dokument bol preložený pomocou služby AI prekladu [Co-op Translator](https://github.com/Azure/co-op-translator). Aj keď sa snažíme o presnosť, prosím, berte na vedomie, že automatizované preklady môžu obsahovať chyby alebo nepresnosti. Pôvodný dokument v jeho rodnom jazyku by mal byť považovaný za autoritatívny zdroj. Pre kritické informácie sa odporúča profesionálny ľudský preklad. Nenesieme zodpovednosť za akékoľvek nedorozumenia alebo nesprávne interpretácie vyplývajúce z použitia tohto prekladu.