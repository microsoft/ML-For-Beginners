<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "5f3cb462e3122e1afe7ab0050ccf2bd3",
  "translation_date": "2025-09-05T01:21:18+00:00",
  "source_file": "6-NLP/2-Tasks/README.md",
  "language_code": "cs"
}
-->
# Běžné úlohy a techniky zpracování přirozeného jazyka

Pro většinu úloh *zpracování přirozeného jazyka* je nutné text rozdělit, analyzovat a výsledky uložit nebo porovnat s pravidly a datovými sadami. Tyto úlohy umožňují programátorovi odvodit _význam_, _záměr_ nebo pouze _četnost_ termínů a slov v textu.

## [Kvíz před přednáškou](https://ff-quizzes.netlify.app/en/ml/)

Pojďme objevit běžné techniky používané při zpracování textu. V kombinaci se strojovým učením vám tyto techniky pomohou efektivně analyzovat velké množství textu. Než však tyto úlohy aplikujete na ML, je důležité pochopit problémy, se kterými se odborník na NLP setkává.

## Běžné úlohy v NLP

Existuje několik způsobů, jak analyzovat text, na kterém pracujete. Existují úlohy, které můžete provádět, a díky nim získáte porozumění textu a můžete vyvodit závěry. Tyto úlohy obvykle provádíte v určitém pořadí.

### Tokenizace

Pravděpodobně první věc, kterou většina algoritmů NLP musí udělat, je rozdělit text na tokeny, tedy slova. I když to zní jednoduše, zohlednění interpunkce a různých jazykových oddělovačů slov a vět může být složité. Možná budete muset použít různé metody k určení hranic.

![tokenizace](../../../../6-NLP/2-Tasks/images/tokenization.png)
> Tokenizace věty z **Pýchy a předsudku**. Infografika od [Jen Looper](https://twitter.com/jenlooper)

### Vektorizace (Embeddings)

[Word embeddings](https://wikipedia.org/wiki/Word_embedding) jsou způsob, jak převést textová data na číselnou podobu. Vektorizace se provádí tak, aby slova s podobným významem nebo slova používaná společně byla seskupena.

![word embeddings](../../../../6-NLP/2-Tasks/images/embedding.png)
> "Mám nejvyšší respekt k vašim nervům, jsou to moji staří přátelé." - Vektorizace věty z **Pýchy a předsudku**. Infografika od [Jen Looper](https://twitter.com/jenlooper)

✅ Vyzkoušejte [tento zajímavý nástroj](https://projector.tensorflow.org/) pro experimentování s vektorizací slov. Kliknutím na jedno slovo zobrazíte shluky podobných slov: 'toy' se shlukuje s 'disney', 'lego', 'playstation' a 'console'.

### Syntaktická analýza a označování částí řeči

Každé slovo, které bylo tokenizováno, může být označeno jako část řeči - podstatné jméno, sloveso nebo přídavné jméno. Věta `rychlá červená liška přeskočila líného hnědého psa` může být označena jako liška = podstatné jméno, přeskočila = sloveso.

![syntaktická analýza](../../../../6-NLP/2-Tasks/images/parse.png)

> Syntaktická analýza věty z **Pýchy a předsudku**. Infografika od [Jen Looper](https://twitter.com/jenlooper)

Syntaktická analýza znamená rozpoznání, která slova jsou ve větě vzájemně propojena - například `rychlá červená liška přeskočila` je sekvence přídavné jméno-podstatné jméno-sloveso, která je oddělená od sekvence `líný hnědý pes`.

### Četnost slov a frází

Užitečným postupem při analýze velkého množství textu je vytvoření slovníku každého slova nebo fráze, která nás zajímá, a jak často se v textu objevuje. Fráze `rychlá červená liška přeskočila líného hnědého psa` má četnost slova "the" 2.

Podívejme se na příklad textu, kde počítáme četnost slov. Báseň The Winners od Rudyarda Kiplinga obsahuje následující verš:

```output
What the moral? Who rides may read.
When the night is thick and the tracks are blind
A friend at a pinch is a friend, indeed,
But a fool to wait for the laggard behind.
Down to Gehenna or up to the Throne,
He travels the fastest who travels alone.
```

Protože četnost frází může být citlivá na velikost písmen nebo naopak, fráze `a friend` má četnost 2, `the` má četnost 6 a `travels` má četnost 2.

### N-gramy

Text může být rozdělen na sekvence slov určité délky, jedno slovo (unigram), dvě slova (bigramy), tři slova (trigramy) nebo libovolný počet slov (n-gramy).

Například `rychlá červená liška přeskočila líného hnědého psa` s n-gram skóre 2 vytvoří následující n-gramy:

1. rychlá červená  
2. červená liška  
3. liška přeskočila  
4. přeskočila líného  
5. líného hnědého  
6. hnědého psa  

Může být snazší si to představit jako posuvné okno nad větou. Zde je to pro n-gramy o 3 slovech, n-gram je tučně v každé větě:

1.   <u>**rychlá červená liška**</u> přeskočila líného hnědého psa  
2.   rychlá **<u>červená liška přeskočila</u>** líného hnědého psa  
3.   rychlá červená **<u>liška přeskočila líného</u>** hnědého psa  
4.   rychlá červená liška **<u>přeskočila líného hnědého</u>** psa  
5.   rychlá červená liška přeskočila **<u>líného hnědého psa</u>**  

![posuvné okno n-gramů](../../../../6-NLP/2-Tasks/images/n-grams.gif)

> N-gram hodnota 3: Infografika od [Jen Looper](https://twitter.com/jenlooper)

### Extrakce podstatných jmenných frází

Většina vět obsahuje podstatné jméno, které je předmětem nebo objektem věty. V angličtině je často identifikovatelné díky předcházejícímu 'a', 'an' nebo 'the'. Identifikace předmětu nebo objektu věty pomocí 'extrakce podstatné jmenné fráze' je běžnou úlohou v NLP při pokusu o pochopení významu věty.

✅ Ve větě "Nemohu si vzpomenout na hodinu, místo, pohled nebo slova, která položila základ. Je to příliš dávno. Byl jsem uprostřed, než jsem si uvědomil, že jsem začal." dokážete identifikovat podstatné jmenné fráze?

Ve větě `rychlá červená liška přeskočila líného hnědého psa` jsou 2 podstatné jmenné fráze: **rychlá červená liška** a **líný hnědý pes**.

### Analýza sentimentu

Věta nebo text může být analyzován z hlediska sentimentu, tedy jak *pozitivní* nebo *negativní* je. Sentiment se měří pomocí *polarizace* a *objektivity/subjektivity*. Polarizace se měří od -1.0 do 1.0 (negativní až pozitivní) a 0.0 do 1.0 (nejvíce objektivní až nejvíce subjektivní).

✅ Později se naučíte, že existují různé způsoby, jak určit sentiment pomocí strojového učení, ale jedním ze způsobů je mít seznam slov a frází, které jsou lidským expertem kategorizovány jako pozitivní nebo negativní, a aplikovat tento model na text k výpočtu skóre polarizace. Vidíte, jak by to mohlo fungovat v některých situacích a méně dobře v jiných?

### Inflekce

Inflekce umožňuje vzít slovo a získat jeho jednotné nebo množné číslo.

### Lemmatizace

*Lemma* je kořenové nebo základní slovo pro sadu slov, například *letěl*, *letí*, *létání* mají lemma slovesa *letět*.

Existují také užitečné databáze dostupné pro výzkumníka NLP, zejména:

### WordNet

[WordNet](https://wordnet.princeton.edu/) je databáze slov, synonym, antonym a mnoha dalších detailů pro každé slovo v mnoha různých jazycích. Je neuvěřitelně užitečná při pokusu o vytváření překladů, kontrolu pravopisu nebo jazykových nástrojů jakéhokoli typu.

## Knihovny NLP

Naštěstí nemusíte všechny tyto techniky vytvářet sami, protože existují vynikající knihovny v Pythonu, které je zpřístupňují vývojářům, kteří nejsou specializovaní na zpracování přirozeného jazyka nebo strojové učení. V dalších lekcích najdete více příkladů, ale zde se naučíte několik užitečných příkladů, které vám pomohou s dalším úkolem.

### Cvičení - použití knihovny `TextBlob`

Použijme knihovnu TextBlob, protože obsahuje užitečná API pro řešení těchto typů úloh. TextBlob "stojí na obrovských ramenou [NLTK](https://nltk.org) a [pattern](https://github.com/clips/pattern) a dobře spolupracuje s oběma." Má značné množství ML zabudované ve svém API.

> Poznámka: Doporučuje se [rychlý průvodce](https://textblob.readthedocs.io/en/dev/quickstart.html#quickstart) pro TextBlob, který je určen pro zkušené vývojáře v Pythonu.

Při pokusu o identifikaci *podstatných jmenných frází* nabízí TextBlob několik možností extraktorů pro nalezení podstatných jmenných frází.

1. Podívejte se na `ConllExtractor`.

    ```python
    from textblob import TextBlob
    from textblob.np_extractors import ConllExtractor
    # import and create a Conll extractor to use later 
    extractor = ConllExtractor()
    
    # later when you need a noun phrase extractor:
    user_input = input("> ")
    user_input_blob = TextBlob(user_input, np_extractor=extractor)  # note non-default extractor specified
    np = user_input_blob.noun_phrases                                    
    ```

    > Co se zde děje? [ConllExtractor](https://textblob.readthedocs.io/en/dev/api_reference.html?highlight=Conll#textblob.en.np_extractors.ConllExtractor) je "extraktor podstatných jmenných frází, který používá chunk parsing trénovaný na korpusu ConLL-2000." ConLL-2000 odkazuje na konferenci o zpracování přirozeného jazyka z roku 2000. Každý rok konference pořádala workshop na řešení obtížného problému NLP, a v roce 2000 to bylo chunkování podstatných jmenných frází. Model byl trénován na Wall Street Journal, s "sekcemi 15-18 jako trénovací data (211727 tokenů) a sekcí 20 jako testovací data (47377 tokenů)". Postupy použité můžete najít [zde](https://www.clips.uantwerpen.be/conll2000/chunking/) a [výsledky](https://ifarm.nl/erikt/research/np-chunking.html).

### Výzva - zlepšení vašeho bota pomocí NLP

V předchozí lekci jste vytvořili velmi jednoduchého Q&A bota. Nyní uděláte Marvina trochu empatičtějšího tím, že analyzujete váš vstup na sentiment a vytisknete odpověď odpovídající sentimentu. Budete také muset identifikovat `noun_phrase` a zeptat se na ni.

Vaše kroky při vytváření lepšího konverzačního bota:

1. Vytiskněte instrukce, jak komunikovat s botem
2. Spusťte smyčku 
   1. Přijměte uživatelský vstup
   2. Pokud uživatel požádal o ukončení, ukončete
   3. Zpracujte uživatelský vstup a určete odpovídající odpověď na sentiment
   4. Pokud je v sentimentu detekována podstatná jmenná fráze, vytvořte její množné číslo a požádejte o další vstup na toto téma
   5. Vytiskněte odpověď
3. Vraťte se zpět ke kroku 2

Zde je ukázka kódu pro určení sentimentu pomocí TextBlob. Všimněte si, že existují pouze čtyři *stupně* odpovědi na sentiment (můžete jich mít více, pokud chcete):

```python
if user_input_blob.polarity <= -0.5:
  response = "Oh dear, that sounds bad. "
elif user_input_blob.polarity <= 0:
  response = "Hmm, that's not great. "
elif user_input_blob.polarity <= 0.5:
  response = "Well, that sounds positive. "
elif user_input_blob.polarity <= 1:
  response = "Wow, that sounds great. "
```

Zde je ukázkový výstup, který vás může vést (uživatelský vstup je na řádcích začínajících >):

```output
Hello, I am Marvin, the friendly robot.
You can end this conversation at any time by typing 'bye'
After typing each answer, press 'enter'
How are you today?
> I am ok
Well, that sounds positive. Can you tell me more?
> I went for a walk and saw a lovely cat
Well, that sounds positive. Can you tell me more about lovely cats?
> cats are the best. But I also have a cool dog
Wow, that sounds great. Can you tell me more about cool dogs?
> I have an old hounddog but he is sick
Hmm, that's not great. Can you tell me more about old hounddogs?
> bye
It was nice talking to you, goodbye!
```

Jedno možné řešení úkolu je [zde](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/2-Tasks/solution/bot.py)

✅ Kontrola znalostí

1. Myslíte si, že empatické odpovědi by mohly 'oklamat' někoho, aby si myslel, že bot skutečně rozumí?  
2. Dělá identifikace podstatné jmenné fráze bota více 'uvěřitelným'?  
3. Proč by extrakce 'podstatné jmenné fráze' z věty byla užitečná věc?  

---

Implementujte bota z předchozí kontroly znalostí a otestujte ho na příteli. Dokáže je oklamat? Dokážete udělat svého bota více 'uvěřitelným'?

## 🚀Výzva

Vezměte úkol z předchozí kontroly znalostí a zkuste ho implementovat. Otestujte bota na příteli. Dokáže je oklamat? Dokážete udělat svého bota více 'uvěřitelným'?

## [Kvíz po přednášce](https://ff-quizzes.netlify.app/en/ml/)

## Přehled & Samostudium

V následujících lekcích se dozvíte více o analýze sentimentu. Prozkoumejte tuto zajímavou techniku v článcích, jako jsou tyto na [KDNuggets](https://www.kdnuggets.com/tag/nlp)

## Úkol 

[Nechte bota odpovídat](assignment.md)

---

**Prohlášení**:  
Tento dokument byl přeložen pomocí služby pro automatický překlad [Co-op Translator](https://github.com/Azure/co-op-translator). Ačkoli se snažíme o přesnost, mějte prosím na paměti, že automatické překlady mohou obsahovat chyby nebo nepřesnosti. Původní dokument v jeho původním jazyce by měl být považován za závazný zdroj. Pro důležité informace doporučujeme profesionální lidský překlad. Neodpovídáme za žádná nedorozumění nebo nesprávné interpretace vyplývající z použití tohoto překladu.