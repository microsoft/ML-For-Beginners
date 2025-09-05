<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "be03c8182982b87ced155e4e9d1438e8",
  "translation_date": "2025-09-05T01:37:49+00:00",
  "source_file": "6-NLP/3-Translation-Sentiment/README.md",
  "language_code": "cs"
}
-->
# Překlad a analýza sentimentu pomocí ML

V předchozích lekcích jste se naučili, jak vytvořit základního bota pomocí knihovny `TextBlob`, která využívá strojové učení v pozadí k provádění základních úkolů NLP, jako je extrakce podstatných jmen. Další důležitou výzvou v oblasti počítačové lingvistiky je přesný _překlad_ věty z jednoho mluveného nebo psaného jazyka do jiného.

## [Kvíz před lekcí](https://ff-quizzes.netlify.app/en/ml/)

Překlad je velmi obtížný problém, který je komplikován tím, že existují tisíce jazyků, z nichž každý může mít velmi odlišná gramatická pravidla. Jedním z přístupů je převést formální gramatická pravidla jednoho jazyka, například angličtiny, do struktury nezávislé na jazyku a poté je přeložit zpět do jiného jazyka. Tento přístup zahrnuje následující kroky:

1. **Identifikace**. Identifikujte nebo označte slova ve vstupním jazyce jako podstatná jména, slovesa atd.
2. **Vytvoření překladu**. Vytvořte přímý překlad každého slova ve formátu cílového jazyka.

### Příklad věty, angličtina do irštiny

V angličtině je věta _I feel happy_ složena ze tří slov v pořadí:

- **podmět** (I)
- **sloveso** (feel)
- **přídavné jméno** (happy)

Nicméně v irštině má stejná věta velmi odlišnou gramatickou strukturu – emoce jako "*happy*" nebo "*sad*" jsou vyjádřeny jako něco, co je *na vás*.

Anglická fráze `I feel happy` by se v irštině přeložila jako `Tá athas orm`. Doslovný překlad by byl `Štěstí je na mně`.

Irský mluvčí překládající do angličtiny by řekl `I feel happy`, nikoli `Happy is upon me`, protože rozumí významu věty, i když se slova a struktura věty liší.

Formální pořadí věty v irštině je:

- **sloveso** (Tá nebo is)
- **přídavné jméno** (athas, nebo happy)
- **podmět** (orm, nebo upon me)

## Překlad

Naivní překladový program by mohl překládat pouze slova, ignorujíc strukturu věty.

✅ Pokud jste se jako dospělí učili druhý (nebo třetí či více) jazyk, možná jste začali tím, že jste přemýšleli ve svém mateřském jazyce, překládali koncept slovo po slovu ve své hlavě do druhého jazyka a poté vyslovili svůj překlad. To je podobné tomu, co dělají naivní překladové počítačové programy. Je důležité překonat tuto fázi, abyste dosáhli plynulosti!

Naivní překlad vede ke špatným (a někdy veselým) překladům: `I feel happy` se doslovně přeloží jako `Mise bhraitheann athas` v irštině. To znamená (doslovně) `já cítím štěstí` a není to platná irská věta. I když angličtina a irština jsou jazyky mluvené na dvou blízce sousedících ostrovech, jsou to velmi odlišné jazyky s různými gramatickými strukturami.

> Můžete se podívat na některá videa o irských lingvistických tradicích, například [toto](https://www.youtube.com/watch?v=mRIaLSdRMMs)

### Přístupy strojového učení

Doposud jste se naučili o přístupu založeném na formálních pravidlech k zpracování přirozeného jazyka. Dalším přístupem je ignorovat význam slov a _místo toho použít strojové učení k detekci vzorců_. To může fungovat při překladu, pokud máte hodně textu (*korpus*) nebo textů (*korpora*) v původním i cílovém jazyce.

Například vezměte případ *Pýchy a předsudku*, známého anglického románu napsaného Jane Austenovou v roce 1813. Pokud si prohlédnete knihu v angličtině a lidský překlad knihy do *francouzštiny*, mohli byste detekovat fráze v jednom jazyce, které jsou _idiomaticky_ přeloženy do druhého. To si vyzkoušíte za chvíli.

Například když je anglická fráze `I have no money` doslovně přeložena do francouzštiny, může se stát `Je n'ai pas de monnaie`. "Monnaie" je zrádný francouzský 'falešný přítel', protože 'money' a 'monnaie' nejsou synonymní. Lepší překlad, který by mohl vytvořit člověk, by byl `Je n'ai pas d'argent`, protože lépe vyjadřuje význam, že nemáte žádné peníze (spíše než 'drobné', což je význam 'monnaie').

![monnaie](../../../../6-NLP/3-Translation-Sentiment/images/monnaie.png)

> Obrázek od [Jen Looper](https://twitter.com/jenlooper)

Pokud má model strojového učení dostatek lidských překladů, na kterých může stavět model, může zlepšit přesnost překladů identifikací běžných vzorců v textech, které byly dříve přeloženy odbornými lidskými mluvčími obou jazyků.

### Cvičení - překlad

Můžete použít `TextBlob` k překladu vět. Vyzkoušejte slavní první větu z **Pýchy a předsudku**:

```python
from textblob import TextBlob

blob = TextBlob(
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife!"
)
print(blob.translate(to="fr"))

```

`TextBlob` odvede docela dobrou práci při překladu: "C'est une vérité universellement reconnue, qu'un homme célibataire en possession d'une bonne fortune doit avoir besoin d'une femme!". 

Lze tvrdit, že překlad TextBlob je mnohem přesnější než francouzský překlad knihy z roku 1932 od V. Leconte a Ch. Pressoir:

"C'est une vérité universelle qu'un célibataire pourvu d'une belle fortune doit avoir envie de se marier, et, si peu que l'on sache de son sentiment à cet egard, lorsqu'il arrive dans une nouvelle résidence, cette idée est si bien fixée dans l'esprit de ses voisins qu'ils le considèrent sur-le-champ comme la propriété légitime de l'une ou l'autre de leurs filles."

V tomto případě překlad informovaný strojovým učením odvádí lepší práci než lidský překladatel, který zbytečně vkládá slova do úst původnímu autorovi pro 'jasnost'.

> Co se zde děje? A proč je TextBlob tak dobrý v překladu? No, v pozadí používá Google Translate, sofistikovanou AI schopnou analyzovat miliony frází a předpovídat nejlepší řetězce pro daný úkol. Nic manuálního se zde neděje a k použití `blob.translate` potřebujete připojení k internetu.

✅ Vyzkoušejte další věty. Který překlad je lepší, strojové učení nebo lidský překlad? V jakých případech?

## Analýza sentimentu

Další oblastí, kde strojové učení může velmi dobře fungovat, je analýza sentimentu. Přístup bez strojového učení k sentimentu spočívá v identifikaci slov a frází, které jsou 'pozitivní' a 'negativní'. Poté, při zpracování nového textu, vypočítá celkovou hodnotu pozitivních, negativních a neutrálních slov, aby identifikoval celkový sentiment. 

Tento přístup lze snadno oklamat, jak jste mohli vidět v úkolu Marvin - věta `Great, that was a wonderful waste of time, I'm glad we are lost on this dark road` je sarkastická, negativní věta, ale jednoduchý algoritmus detekuje 'great', 'wonderful', 'glad' jako pozitivní a 'waste', 'lost' a 'dark' jako negativní. Celkový sentiment je ovlivněn těmito protichůdnými slovy.

✅ Zastavte se na chvíli a zamyslete se nad tím, jak jako lidští mluvčí vyjadřujeme sarkasmus. Intonace hraje velkou roli. Zkuste říct frázi "Well, that film was awesome" různými způsoby, abyste zjistili, jak váš hlas vyjadřuje význam.

### Přístupy strojového učení

Přístup strojového učení by spočíval v ručním shromáždění negativních a pozitivních textů - tweetů, recenzí filmů nebo čehokoli, kde člověk dal hodnocení *a* napsal názor. Poté lze na názory a hodnocení aplikovat techniky NLP, aby se objevily vzorce (např. pozitivní recenze filmů mají tendenci obsahovat frázi 'Oscar worthy' více než negativní recenze filmů, nebo pozitivní recenze restaurací říkají 'gourmet' mnohem více než 'disgusting').

> ⚖️ **Příklad**: Pokud byste pracovali v kanceláři politika a projednával se nový zákon, voliči by mohli psát do kanceláře e-maily podporující nebo e-maily proti konkrétnímu novému zákonu. Řekněme, že byste byli pověřeni čtením e-mailů a jejich tříděním do 2 hromádek, *pro* a *proti*. Pokud by bylo mnoho e-mailů, mohli byste být zahlceni pokusem přečíst je všechny. Nebylo by hezké, kdyby bot mohl všechny přečíst za vás, porozumět jim a říct vám, do které hromádky každý e-mail patří? 
> 
> Jedním ze způsobů, jak toho dosáhnout, je použití strojového učení. Model byste trénovali na části e-mailů *proti* a části e-mailů *pro*. Model by měl tendenci spojovat fráze a slova s proti stranou a pro stranou, *ale nerozuměl by žádnému obsahu*, pouze by určité slova a vzorce byly pravděpodobnější v e-mailech *proti* nebo *pro*. Mohli byste jej otestovat na některých e-mailech, které jste nepoužili k trénování modelu, a zjistit, zda došel ke stejnému závěru jako vy. Poté, co byste byli spokojeni s přesností modelu, mohli byste zpracovávat budoucí e-maily, aniž byste museli číst každý z nich.

✅ Zní tento proces jako procesy, které jste použili v předchozích lekcích?

## Cvičení - sentimentální věty

Sentiment se měří pomocí *polarizace* od -1 do 1, což znamená, že -1 je nejvíce negativní sentiment a 1 je nejvíce pozitivní. Sentiment se také měří pomocí skóre od 0 do 1 pro objektivitu (0) a subjektivitu (1).

Podívejte se znovu na *Pýchu a předsudek* od Jane Austenové. Text je dostupný zde na [Project Gutenberg](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm). Níže je ukázka krátkého programu, který analyzuje sentiment první a poslední věty z knihy a zobrazí její polarizaci sentimentu a skóre subjektivity/objektivity.

Měli byste použít knihovnu `TextBlob` (popisovanou výše) k určení `sentimentu` (nemusíte psát vlastní kalkulátor sentimentu) v následujícím úkolu.

```python
from textblob import TextBlob

quote1 = """It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife."""

quote2 = """Darcy, as well as Elizabeth, really loved them; and they were both ever sensible of the warmest gratitude towards the persons who, by bringing her into Derbyshire, had been the means of uniting them."""

sentiment1 = TextBlob(quote1).sentiment
sentiment2 = TextBlob(quote2).sentiment

print(quote1 + " has a sentiment of " + str(sentiment1))
print(quote2 + " has a sentiment of " + str(sentiment2))
```

Vidíte následující výstup:

```output
It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want # of a wife. has a sentiment of Sentiment(polarity=0.20952380952380953, subjectivity=0.27142857142857146)

Darcy, as well as Elizabeth, really loved them; and they were
     both ever sensible of the warmest gratitude towards the persons
      who, by bringing her into Derbyshire, had been the means of
      uniting them. has a sentiment of Sentiment(polarity=0.7, subjectivity=0.8)
```

## Výzva - zkontrolujte polarizaci sentimentu

Vaším úkolem je určit, pomocí polarizace sentimentu, zda má *Pýcha a předsudek* více absolutně pozitivních vět než absolutně negativních. Pro tento úkol můžete předpokládat, že polarizační skóre 1 nebo -1 je absolutně pozitivní nebo negativní.

**Kroky:**

1. Stáhněte si [kopii Pýchy a předsudku](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm) z Project Gutenberg jako .txt soubor. Odstraňte metadata na začátku a konci souboru, ponechte pouze původní text
2. Otevřete soubor v Pythonu a extrahujte obsah jako řetězec
3. Vytvořte TextBlob pomocí řetězce knihy
4. Analyzujte každou větu v knize v cyklu
   1. Pokud je polarizace 1 nebo -1, uložte větu do pole nebo seznamu pozitivních nebo negativních zpráv
5. Na konci vytiskněte všechny pozitivní věty a negativní věty (samostatně) a jejich počet.

Zde je ukázkové [řešení](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/3-Translation-Sentiment/solution/notebook.ipynb).

✅ Kontrola znalostí

1. Sentiment je založen na slovech použitých ve větě, ale rozumí kód *slovům*?
2. Myslíte si, že polarizace sentimentu je přesná, nebo jinými slovy, *souhlasíte* se skóre?
   1. Zejména souhlasíte nebo nesouhlasíte s absolutní **pozitivní** polarizací následujících vět?
      * “What an excellent father you have, girls!” said she, when the door was shut.
      * “Your examination of Mr. Darcy is over, I presume,” said Miss Bingley; “and pray what is the result?” “I am perfectly convinced by it that Mr. Darcy has no defect.
      * How wonderfully these sort of things occur!
      * I have the greatest dislike in the world to that sort of thing.
      * Charlotte is an excellent manager, I dare say.
      * “This is delightful indeed!
      * I am so happy!
      * Your idea of the ponies is delightful.
   2. Následující 3 věty byly ohodnoceny absolutně pozitivním sentimentem, ale při bližším čtení nejsou pozitivními větami. Proč si analýza sentimentu myslela, že jsou pozitivními větami?
      * Happy shall I be, when his stay at Netherfield is over!” “I wish I could say anything to comfort you,” replied Elizabeth; “but it is wholly out of my power.
      * If I could but see you as happy!
      * Our distress, my dear Lizzy, is very great.
   3. Souhlasíte nebo nesouhlasíte s absolutní **negativní** polarizací následujících vět?
      - Everybody is disgusted with his pride.
      - “I should like to know how he behaves among strangers.” “You shall hear then—but prepare yourself for something very dreadful.
      - The pause was to Elizabeth’s feelings dreadful.
      - It would be dreadful!

✅ Každý nadšenec Jane Austenové pochopí, že často používá své knihy k tomu, aby kritizovala absurdnější aspekty anglické regentské společnosti. Elizabeth Bennettová, hlavní postava v *Pýše a předsudku*, je bystrou společenskou pozorovatelkou (jako autorka) a její jazyk je často silně nuancovaný. Dokonce i pan Darcy (milostný zájem v příběhu) poznamenává Elizabethin hravý a škádlivý způsob používání jazyka: "Měl jsem tu čest vás poznat dost dlouho na to, abych věděl, že si velmi užíváte příležitostné vyjadřování názorů, které ve skutečnosti nejsou vaše vlastní."

---

## 🚀Výzva

Dokážete udělat Marvina ještě lepším tím, že z uživatelského vstupu extrahujete další vlastnosti?

## [Kvíz po lekci](https://ff-quizzes.netlify.app/en/ml/)

## Přehled & Samostudium
Existuje mnoho způsobů, jak získat sentiment z textu. Zamyslete se nad obchodními aplikacemi, které by mohly využívat tuto techniku. Přemýšlejte o tom, jak by se to mohlo pokazit. Přečtěte si více o sofistikovaných systémech připravených pro podniky, které analyzují sentiment, jako například [Azure Text Analysis](https://docs.microsoft.com/azure/cognitive-services/Text-Analytics/how-tos/text-analytics-how-to-sentiment-analysis?tabs=version-3-1?WT.mc_id=academic-77952-leestott). Otestujte některé z vět z knihy Pýcha a předsudek výše a zjistěte, zda dokáže rozpoznat nuance.

## Zadání

[Poetická licence](assignment.md)

---

**Prohlášení**:  
Tento dokument byl přeložen pomocí služby pro automatický překlad [Co-op Translator](https://github.com/Azure/co-op-translator). Ačkoli se snažíme o přesnost, mějte prosím na paměti, že automatické překlady mohou obsahovat chyby nebo nepřesnosti. Původní dokument v jeho původním jazyce by měl být považován za autoritativní zdroj. Pro důležité informace doporučujeme profesionální lidský překlad. Neodpovídáme za žádná nedorozumění nebo nesprávné interpretace vyplývající z použití tohoto překladu.