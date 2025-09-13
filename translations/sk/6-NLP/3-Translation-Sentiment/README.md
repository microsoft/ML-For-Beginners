<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "be03c8182982b87ced155e4e9d1438e8",
  "translation_date": "2025-09-05T17:03:50+00:00",
  "source_file": "6-NLP/3-Translation-Sentiment/README.md",
  "language_code": "sk"
}
-->
# Preklad a analýza sentimentu pomocou ML

V predchádzajúcich lekciách ste sa naučili, ako vytvoriť základného bota pomocou knižnice `TextBlob`, ktorá využíva strojové učenie na vykonávanie základných úloh spracovania prirodzeného jazyka, ako je extrakcia podstatných fráz. Ďalšou dôležitou výzvou v oblasti počítačovej lingvistiky je presný _preklad_ vety z jedného hovoreného alebo písaného jazyka do druhého.

## [Kvíz pred prednáškou](https://ff-quizzes.netlify.app/en/ml/)

Preklad je veľmi náročný problém, ktorý je ešte zložitejší vzhľadom na to, že existujú tisíce jazykov, z ktorých každý môže mať veľmi odlišné gramatické pravidlá. Jedným z prístupov je konvertovať formálne gramatické pravidlá jedného jazyka, napríklad angličtiny, do štruktúry nezávislej od jazyka a potom ich preložiť späť do iného jazyka. Tento prístup zahŕňa nasledujúce kroky:

1. **Identifikácia**. Identifikujte alebo označte slová v vstupnom jazyku ako podstatné mená, slovesá atď.
2. **Vytvorenie prekladu**. Vytvorte priamy preklad každého slova vo formáte cieľového jazyka.

### Príklad vety, angličtina do írčiny

V angličtine je veta _I feel happy_ tri slová v poradí:

- **podmet** (I)
- **sloveso** (feel)
- **prídavné meno** (happy)

Avšak v írskom jazyku má tá istá veta úplne odlišnú gramatickú štruktúru – emócie ako "*šťastný*" alebo "*smutný*" sa vyjadrujú ako niečo, čo je *na vás*.

Anglická fráza `I feel happy` by sa v írčine preložila ako `Tá athas orm`. Doslovný preklad by bol `Šťastie je na mne`.

Írsky hovoriaci, ktorý prekladá do angličtiny, by povedal `I feel happy`, nie `Happy is upon me`, pretože chápe význam vety, aj keď sú slová a štruktúra vety odlišné.

Formálne poradie vety v írčine je:

- **sloveso** (Tá alebo je)
- **prídavné meno** (athas alebo šťastný)
- **podmet** (orm alebo na mne)

## Preklad

Naivný prekladový program by mohol prekladať iba slová, ignorujúc štruktúru vety.

✅ Ak ste sa ako dospelý naučili druhý (alebo tretí či viac) jazyk, možno ste začali tým, že ste premýšľali vo svojom rodnom jazyku, prekladali koncept slovo po slove vo svojej hlave do druhého jazyka a potom vyslovili svoj preklad. Toto je podobné tomu, čo robia naivné prekladové počítačové programy. Je dôležité prekonať túto fázu, aby ste dosiahli plynulosť!

Naivný preklad vedie k zlým (a niekedy vtipným) nesprávnym prekladom: `I feel happy` sa doslovne preloží ako `Mise bhraitheann athas` v írčine. To znamená (doslovne) `ja cítim šťastie` a nie je to platná írska veta. Aj keď angličtina a írčina sú jazyky hovorené na dvoch blízko susediacich ostrovoch, sú to veľmi odlišné jazyky s rôznymi gramatickými štruktúrami.

> Môžete si pozrieť niektoré videá o írskych jazykových tradíciách, ako napríklad [toto](https://www.youtube.com/watch?v=mRIaLSdRMMs)

### Prístupy strojového učenia

Doteraz ste sa naučili o prístupe formálnych pravidiel k spracovaniu prirodzeného jazyka. Ďalším prístupom je ignorovať význam slov a _namiesto toho použiť strojové učenie na detekciu vzorcov_. Toto môže fungovať pri preklade, ak máte veľa textov (*korpus*) alebo textov (*korpora*) v pôvodnom a cieľovom jazyku.

Napríklad, vezmite si prípad *Pýcha a predsudok*, známeho anglického románu napísaného Jane Austenovou v roku 1813. Ak si pozriete knihu v angličtine a ľudský preklad knihy do *francúzštiny*, mohli by ste detekovať frázy v jednom, ktoré sú _idiomaticky_ preložené do druhého. To si vyskúšate za chvíľu.

Napríklad, keď sa anglická fráza `I have no money` doslovne preloží do francúzštiny, môže sa stať `Je n'ai pas de monnaie`. "Monnaie" je zradný francúzsky 'falošný príbuzný', pretože 'money' a 'monnaie' nie sú synonymá. Lepší preklad, ktorý by mohol urobiť človek, by bol `Je n'ai pas d'argent`, pretože lepšie vyjadruje význam, že nemáte peniaze (skôr než 'drobné', čo je význam 'monnaie').

![monnaie](../../../../6-NLP/3-Translation-Sentiment/images/monnaie.png)

> Obrázok od [Jen Looper](https://twitter.com/jenlooper)

Ak má model strojového učenia dostatok ľudských prekladov na vytvorenie modelu, môže zlepšiť presnosť prekladov identifikovaním bežných vzorcov v textoch, ktoré už predtým preložili odborní ľudskí hovoriaci oboch jazykov.

### Cvičenie - preklad

Môžete použiť `TextBlob` na preklad viet. Vyskúšajte slávnu prvú vetu z **Pýcha a predsudok**:

```python
from textblob import TextBlob

blob = TextBlob(
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife!"
)
print(blob.translate(to="fr"))

```

`TextBlob` odvádza celkom dobrú prácu pri preklade: "C'est une vérité universellement reconnue, qu'un homme célibataire en possession d'une bonne fortune doit avoir besoin d'une femme!". 

Dá sa povedať, že preklad od TextBlob je oveľa presnejší ako francúzsky preklad knihy z roku 1932 od V. Leconte a Ch. Pressoir:

"C'est une vérité universelle qu'un célibataire pourvu d'une belle fortune doit avoir envie de se marier, et, si peu que l'on sache de son sentiment à cet egard, lorsqu'il arrive dans une nouvelle résidence, cette idée est si bien fixée dans l'esprit de ses voisins qu'ils le considèrent sur-le-champ comme la propriété légitime de l'une ou l'autre de leurs filles."

V tomto prípade preklad informovaný strojovým učením odvádza lepšiu prácu ako ľudský prekladateľ, ktorý zbytočne vkladá slová do úst pôvodného autora pre "jasnosť".

> Čo sa tu deje? A prečo je TextBlob taký dobrý v preklade? Nuž, v zákulisí používa Google Translate, sofistikovanú AI schopnú analyzovať milióny fráz na predpovedanie najlepších reťazcov pre danú úlohu. Nič manuálne sa tu nedeje a na používanie `blob.translate` potrebujete internetové pripojenie.

✅ Vyskúšajte niekoľko ďalších viet. Ktorý preklad je lepší, strojové učenie alebo ľudský preklad? V ktorých prípadoch?

## Analýza sentimentu

Ďalšou oblasťou, kde strojové učenie môže veľmi dobre fungovať, je analýza sentimentu. Ne-ML prístup k sentimentu je identifikovať slová a frázy, ktoré sú 'pozitívne' a 'negatívne'. Potom, vzhľadom na nový text, vypočítať celkovú hodnotu pozitívnych, negatívnych a neutrálnych slov na identifikáciu celkového sentimentu. 

Tento prístup sa dá ľahko oklamať, ako ste mohli vidieť v úlohe Marvin - veta `Great, that was a wonderful waste of time, I'm glad we are lost on this dark road` je sarkastická, negatívna veta, ale jednoduchý algoritmus detekuje 'great', 'wonderful', 'glad' ako pozitívne a 'waste', 'lost' a 'dark' ako negatívne. Celkový sentiment je ovplyvnený týmito protichodnými slovami.

✅ Zastavte sa na chvíľu a zamyslite sa nad tým, ako ako ľudskí hovoriaci vyjadrujeme sarkazmus. Intonácia hlasu hrá veľkú úlohu. Skúste povedať frázu "Well, that film was awesome" rôznymi spôsobmi, aby ste zistili, ako váš hlas vyjadruje význam.

### Prístupy ML

Prístup ML by bol manuálne zhromaždiť negatívne a pozitívne texty - tweety, alebo recenzie filmov, alebo čokoľvek, kde človek dal skóre *a* písomný názor. Potom sa môžu aplikovať techniky NLP na názory a skóre, aby sa objavili vzorce (napr. pozitívne recenzie filmov majú tendenciu obsahovať frázu 'Oscar worthy' viac ako negatívne recenzie filmov, alebo pozitívne recenzie reštaurácií hovoria 'gourmet' oveľa viac ako 'disgusting').

> ⚖️ **Príklad**: Ak by ste pracovali v kancelárii politika a diskutoval by sa nejaký nový zákon, voliči by mohli písať do kancelárie e-maily podporujúce alebo e-maily proti konkrétnemu novému zákonu. Povedzme, že by ste mali za úlohu čítať e-maily a triediť ich do 2 hromád, *za* a *proti*. Ak by bolo veľa e-mailov, mohli by ste byť zahltení pokusom prečítať ich všetky. Nebolo by pekné, keby bot mohol prečítať všetky za vás, pochopiť ich a povedať vám, do ktorej hromady patrí každý e-mail? 
> 
> Jedným zo spôsobov, ako to dosiahnuť, je použiť strojové učenie. Model by ste trénovali s časťou e-mailov *proti* a časťou e-mailov *za*. Model by mal tendenciu spájať frázy a slová so stranou proti a stranou za, *ale nerozumel by žiadnemu obsahu*, iba že určité slová a vzorce sa pravdepodobnejšie objavia v e-mailoch *proti* alebo *za*. Mohli by ste ho otestovať s niektorými e-mailmi, ktoré ste nepoužili na trénovanie modelu, a zistiť, či dospel k rovnakému záveru ako vy. Potom, keď by ste boli spokojní s presnosťou modelu, mohli by ste spracovať budúce e-maily bez toho, aby ste museli čítať každý jeden.

✅ Znie tento proces ako procesy, ktoré ste použili v predchádzajúcich lekciách?

## Cvičenie - sentimentálne vety

Sentiment sa meria pomocou *polarizácie* od -1 do 1, pričom -1 je najnegatívnejší sentiment a 1 je najpozitívnejší. Sentiment sa tiež meria pomocou skóre od 0 do 1 pre objektivitu (0) a subjektivitu (1).

Pozrite sa znova na *Pýcha a predsudok* od Jane Austenovej. Text je dostupný tu na [Project Gutenberg](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm). Ukážka nižšie zobrazuje krátky program, ktorý analyzuje sentiment prvej a poslednej vety z knihy a zobrazí jej polarizáciu sentimentu a skóre subjektivity/objektivity.

Mali by ste použiť knižnicu `TextBlob` (opísanú vyššie) na určenie `sentimentu` (nemusíte písať vlastný kalkulátor sentimentu) v nasledujúcej úlohe.

```python
from textblob import TextBlob

quote1 = """It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife."""

quote2 = """Darcy, as well as Elizabeth, really loved them; and they were both ever sensible of the warmest gratitude towards the persons who, by bringing her into Derbyshire, had been the means of uniting them."""

sentiment1 = TextBlob(quote1).sentiment
sentiment2 = TextBlob(quote2).sentiment

print(quote1 + " has a sentiment of " + str(sentiment1))
print(quote2 + " has a sentiment of " + str(sentiment2))
```

Vidíte nasledujúci výstup:

```output
It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want # of a wife. has a sentiment of Sentiment(polarity=0.20952380952380953, subjectivity=0.27142857142857146)

Darcy, as well as Elizabeth, really loved them; and they were
     both ever sensible of the warmest gratitude towards the persons
      who, by bringing her into Derbyshire, had been the means of
      uniting them. has a sentiment of Sentiment(polarity=0.7, subjectivity=0.8)
```

## Výzva - skontrolujte polarizáciu sentimentu

Vašou úlohou je určiť, pomocou polarizácie sentimentu, či má *Pýcha a predsudok* viac absolútne pozitívnych viet ako absolútne negatívnych. Pre túto úlohu môžete predpokladať, že polarizačné skóre 1 alebo -1 je absolútne pozitívne alebo negatívne.

**Kroky:**

1. Stiahnite si [kópiu Pýcha a predsudok](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm) z Project Gutenberg ako .txt súbor. Odstráňte metadáta na začiatku a konci súboru, ponechajte iba pôvodný text
2. Otvorte súbor v Pythone a extrahujte obsah ako reťazec
3. Vytvorte TextBlob pomocou reťazca knihy
4. Analyzujte každú vetu v knihe v cykle
   1. Ak je polarizácia 1 alebo -1, uložte vetu do poľa alebo zoznamu pozitívnych alebo negatívnych správ
5. Na konci vytlačte všetky pozitívne vety a negatívne vety (samostatne) a ich počet.

Tu je ukážka [riešenia](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/3-Translation-Sentiment/solution/notebook.ipynb).

✅ Kontrola vedomostí

1. Sentiment je založený na slovách použitých vo vete, ale rozumie kód *slovám*?
2. Myslíte si, že polarizácia sentimentu je presná, alebo inými slovami, *súhlasíte* s hodnotami?
   1. Konkrétne, súhlasíte alebo nesúhlasíte s absolútnou **pozitívnou** polarizáciou nasledujúcich viet?
      * “What an excellent father you have, girls!” said she, when the door was shut.
      * “Your examination of Mr. Darcy is over, I presume,” said Miss Bingley; “and pray what is the result?” “I am perfectly convinced by it that Mr. Darcy has no defect.
      * How wonderfully these sort of things occur!
      * I have the greatest dislike in the world to that sort of thing.
      * Charlotte is an excellent manager, I dare say.
      * “This is delightful indeed!
      * I am so happy!
      * Your idea of the ponies is delightful.
   2. Nasledujúce 3 vety boli ohodnotené absolútne pozitívnym sentimentom, ale pri bližšom čítaní to nie sú pozitívne vety. Prečo si analýza sentimentu myslela, že sú pozitívne?
      * Happy shall I be, when his stay at Netherfield is over!” “I wish I could say anything to comfort you,” replied Elizabeth; “but it is wholly out of my power.
      * If I could but see you as happy!
      * Our distress, my dear Lizzy, is very great.
   3. Súhlasíte alebo nesúhlasíte s absolútnou **negatívnou** polarizáciou nasledujúcich viet?
      - Everybody is disgusted with his pride.
      - “I should like to know how he behaves among strangers.” “You shall hear then—but prepare yourself for something very dreadful.
      - The pause was to Elizabeth’s feelings dreadful.
      - It would be dreadful!

✅ Každý nadšenec Jane Austenovej pochopí, že často používa svoje knihy na kritiku absurdnejších aspektov anglickej regentskej spoločnosti. Elizabeth Bennettová, hlavná postava v *Pýcha a predsudok*, je bystrá pozorovateľka spoločnosti (ako autorka) a jej jazyk je často veľmi nuansovaný. Dokonca aj pán Darcy (milostný záujem v príbehu) poznamenáva Elizabethin hravý a škádlivý spôsob používania jazyka: "I have had the pleasure of your acquaintance long enough to know that you find great enjoyment in occasionally professing opinions which in fact are not your own."

---

## 🚀Výzva

Dokážete urobiť Marvina ešte lepším extrahovaním ďalších vlastností z používateľského vstupu?

## [Kvíz po prednáške](https://ff-quizzes.netlify.app/en/ml/)

## Prehľad a samostatné štúdium
Existuje mnoho spôsobov, ako extrahovať sentiment z textu. Zamyslite sa nad obchodnými aplikáciami, ktoré by mohli využiť túto techniku. Premýšľajte o tom, ako sa to môže pokaziť. Prečítajte si viac o sofistikovaných systémoch pripravených pre podniky, ktoré analyzujú sentiment, ako napríklad [Azure Text Analysis](https://docs.microsoft.com/azure/cognitive-services/Text-Analytics/how-tos/text-analytics-how-to-sentiment-analysis?tabs=version-3-1?WT.mc_id=academic-77952-leestott). Otestujte niektoré z viet z knihy Pýcha a predsudok uvedených vyššie a zistite, či dokáže rozpoznať jemné nuansy.

## Zadanie

[Poetická licencia](assignment.md)

---

**Upozornenie**:  
Tento dokument bol preložený pomocou služby AI prekladu [Co-op Translator](https://github.com/Azure/co-op-translator). Hoci sa snažíme o presnosť, prosím, berte na vedomie, že automatizované preklady môžu obsahovať chyby alebo nepresnosti. Pôvodný dokument v jeho pôvodnom jazyku by mal byť považovaný za autoritatívny zdroj. Pre kritické informácie sa odporúča profesionálny ľudský preklad. Nie sme zodpovední za žiadne nedorozumenia alebo nesprávne interpretácie vyplývajúce z použitia tohto prekladu.