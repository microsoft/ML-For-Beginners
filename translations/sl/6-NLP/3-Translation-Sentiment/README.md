<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "be03c8182982b87ced155e4e9d1438e8",
  "translation_date": "2025-09-05T14:15:00+00:00",
  "source_file": "6-NLP/3-Translation-Sentiment/README.md",
  "language_code": "sl"
}
-->
# Prevajanje in analiza sentimenta z ML

V prejšnjih lekcijah ste se naučili, kako zgraditi osnovnega bota z uporabo knjižnice `TextBlob`, ki vključuje strojno učenje za izvajanje osnovnih nalog obdelave naravnega jezika, kot je ekstrakcija samostalniških fraz. Drug pomemben izziv v računalniški lingvistiki je natančno _prevajanje_ stavka iz enega govorjenega ali pisanega jezika v drugega.

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

Prevajanje je zelo težaven problem, ki ga otežuje dejstvo, da obstaja na tisoče jezikov, od katerih ima vsak lahko zelo različna slovnična pravila. Eden od pristopov je pretvorba formalnih slovničnih pravil enega jezika, kot je angleščina, v strukturo, ki ni odvisna od jezika, nato pa prevod nazaj v drug jezik. Ta pristop vključuje naslednje korake:

1. **Identifikacija**. Identificirajte ali označite besede v vhodnem jeziku kot samostalnike, glagole itd.
2. **Ustvarjanje prevoda**. Ustvarite neposreden prevod vsake besede v formatu ciljnega jezika.

### Primer stavka, angleščina v irščino

V 'angleščini' je stavek _I feel happy_ sestavljen iz treh besed v naslednjem vrstnem redu:

- **osebek** (I)
- **glagol** (feel)
- **pridevnik** (happy)

Vendar pa ima v 'irščini' isti stavek zelo drugačno slovnično strukturo - čustva, kot sta "*happy*" ali "*sad*", se izražajo kot nekaj, kar je *na tebi*.

Angleški stavek `I feel happy` v irščini postane `Tá athas orm`. Dobesedni prevod bi bil `Happy is upon me`.

Govorec irščine, ki prevaja v angleščino, bi rekel `I feel happy`, ne pa `Happy is upon me`, ker razume pomen stavka, tudi če so besede in struktura stavka različne.

Formalni vrstni red stavka v irščini je:

- **glagol** (Tá ali is)
- **pridevnik** (athas, ali happy)
- **osebek** (orm, ali upon me)

## Prevajanje

Naiven program za prevajanje bi morda prevedel samo besede, ne da bi upošteval strukturo stavka.

✅ Če ste se kot odrasli naučili drugega (ali tretjega ali več) jezika, ste morda začeli razmišljati v svojem maternem jeziku, koncept prevajali besedo za besedo v glavi v drugi jezik in nato izgovorili prevod. To je podobno temu, kar počnejo naivni računalniški programi za prevajanje. Pomembno je preseči to fazo, da dosežete tekoče znanje jezika!

Naivno prevajanje vodi do slabih (in včasih smešnih) napačnih prevodov: `I feel happy` se dobesedno prevede v `Mise bhraitheann athas` v irščini. To pomeni (dobesedno) `me feel happy` in ni veljaven irski stavek. Čeprav sta angleščina in irščina jezika, ki se govorita na dveh sosednjih otokih, sta zelo različna jezika z različnimi slovničnimi strukturami.

> Lahko si ogledate nekaj videoposnetkov o irskih jezikovnih tradicijah, kot je [ta](https://www.youtube.com/watch?v=mRIaLSdRMMs)

### Pristopi strojnega učenja

Do sedaj ste se naučili o formalnem pristopu pravil k obdelavi naravnega jezika. Drug pristop je ignoriranje pomena besed in _namesto tega uporaba strojnega učenja za zaznavanje vzorcev_. To lahko deluje pri prevajanju, če imate veliko besedila (*korpus*) ali besedil (*korpusi*) v izvornih in ciljnih jezikih.

Na primer, razmislite o primeru *Prevzetnost in pristranost*, znanega angleškega romana, ki ga je leta 1813 napisala Jane Austen. Če preučite knjigo v angleščini in njen človeški prevod v *francoščino*, lahko zaznate fraze v enem jeziku, ki so _idiomatično_ prevedene v drugega. To boste storili čez trenutek.

Na primer, ko se angleška fraza `I have no money` dobesedno prevede v francoščino, postane `Je n'ai pas de monnaie`. "Monnaie" je zavajajoča francoska 'lažna sorodnica', saj 'money' in 'monnaie' nista sinonima. Boljši prevod, ki bi ga naredil človek, bi bil `Je n'ai pas d'argent`, ker bolje izraža pomen, da nimate denarja (namesto 'drobnega denarja', kar pomeni 'monnaie').

![monnaie](../../../../6-NLP/3-Translation-Sentiment/images/monnaie.png)

> Slika avtorice [Jen Looper](https://twitter.com/jenlooper)

Če ima model strojnega učenja dovolj človeških prevodov za izdelavo modela, lahko izboljša natančnost prevodov z identifikacijo pogostih vzorcev v besedilih, ki so jih prej prevedli strokovni govorci obeh jezikov.

### Naloga - prevajanje

Uporabite lahko `TextBlob` za prevajanje stavkov. Poskusite slavni prvi stavek iz **Prevzetnosti in pristranosti**:

```python
from textblob import TextBlob

blob = TextBlob(
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife!"
)
print(blob.translate(to="fr"))

```

`TextBlob` naredi precej dober prevod: "C'est une vérité universellement reconnue, qu'un homme célibataire en possession d'une bonne fortune doit avoir besoin d'une femme!". 

Lahko bi trdili, da je prevod TextBlob-a dejansko veliko natančnejši od francoskega prevoda knjige iz leta 1932, ki sta ga naredila V. Leconte in Ch. Pressoir:

"C'est une vérité universelle qu'un célibataire pourvu d'une belle fortune doit avoir envie de se marier, et, si peu que l'on sache de son sentiment à cet egard, lorsqu'il arrive dans une nouvelle résidence, cette idée est si bien fixée dans l'esprit de ses voisins qu'ils le considèrent sur-le-champ comme la propriété légitime de l'une ou l'autre de leurs filles."

V tem primeru prevod, ki ga vodi strojno učenje, opravi boljše delo kot človeški prevajalec, ki nepotrebno dodaja besede v ustih originalnega avtorja za 'jasnost'.

> Kaj se tukaj dogaja? In zakaj je TextBlob tako dober pri prevajanju? No, v ozadju uporablja Google Translate, sofisticirano umetno inteligenco, ki lahko analizira milijone fraz za napovedovanje najboljših nizov za določeno nalogo. Tukaj ni nič ročnega, za uporabo `blob.translate` pa potrebujete internetno povezavo.

✅ Poskusite še nekaj stavkov. Kaj je boljše, strojno učenje ali človeški prevod? V katerih primerih?

## Analiza sentimenta

Drugo področje, kjer strojno učenje deluje zelo dobro, je analiza sentimenta. Pristop brez strojnega učenja k sentimentu je identificiranje besed in fraz, ki so 'pozitivne' in 'negativne'. Nato, glede na novo besedilo, izračunajte skupno vrednost pozitivnih, negativnih in nevtralnih besed za identifikacijo splošnega sentimenta. 

Ta pristop je enostavno zavajati, kot ste morda videli v nalogi Marvin - stavek `Great, that was a wonderful waste of time, I'm glad we are lost on this dark road` je sarkastičen, negativno naravnan stavek, vendar preprost algoritem zazna 'great', 'wonderful', 'glad' kot pozitivne in 'waste', 'lost' ter 'dark' kot negativne. Splošni sentiment je zmeden zaradi teh nasprotujočih si besed.

✅ Ustavite se za trenutek in razmislite, kako kot ljudje izražamo sarkazem. Ton glasu igra veliko vlogo. Poskusite izgovoriti frazo "Well, that film was awesome" na različne načine, da odkrijete, kako vaš glas izraža pomen.

### Pristopi strojnega učenja

Pristop strojnega učenja bi bil ročno zbrati negativna in pozitivna besedila - tvite, ocene filmov ali karkoli, kjer je človek podal oceno *in* pisno mnenje. Nato se lahko na mnenja in ocene uporabijo tehnike NLP, tako da se pojavijo vzorci (npr. pozitivne ocene filmov pogosto vsebujejo frazo 'Oscar worthy' bolj kot negativne ocene filmov, ali pozitivne ocene restavracij pogosteje uporabljajo 'gourmet' kot 'disgusting').

> ⚖️ **Primer**: Če bi delali v pisarni politika in bi se razpravljalo o novem zakonu, bi volivci morda pisali pisarni z e-poštnimi sporočili v podporo ali proti določenemu novemu zakonu. Recimo, da bi vam bilo dodeljeno branje teh e-poštnih sporočil in razvrščanje v 2 kupčka, *za* in *proti*. Če bi bilo veliko e-poštnih sporočil, bi vas lahko preplavilo branje vseh. Ali ne bi bilo lepo, če bi bot lahko prebral vsa sporočila namesto vas, jih razumel in vam povedal, v kateri kupček spada vsako sporočilo? 
> 
> Eden od načinov za dosego tega je uporaba strojnega učenja. Model bi trenirali z delom *proti* e-poštnih sporočil in delom *za* e-poštnih sporočil. Model bi težil k povezovanju fraz in besed z nasprotno stranjo in stranjo za, *vendar ne bi razumel nobene vsebine*, le da se določene besede in vzorci pogosteje pojavljajo v *proti* ali *za* e-poštnem sporočilu. Testirali bi ga z nekaterimi e-poštnimi sporočili, ki jih niste uporabili za treniranje modela, in preverili, ali je prišel do enakega zaključka kot vi. Ko bi bili zadovoljni z natančnostjo modela, bi lahko obdelali prihodnja e-poštna sporočila, ne da bi morali prebrati vsakega posebej.

✅ Ali se vam ta proces zdi podoben procesom, ki ste jih uporabljali v prejšnjih lekcijah?

## Naloga - sentimentalni stavki

Sentiment se meri z *polarizacijo* od -1 do 1, kar pomeni, da je -1 najbolj negativni sentiment, 1 pa najbolj pozitivni. Sentiment se meri tudi z oceno od 0 do 1 za objektivnost (0) in subjektivnost (1).

Ponovno si oglejte *Prevzetnost in pristranost* Jane Austen. Besedilo je na voljo tukaj na [Project Gutenberg](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm). Spodnji vzorec prikazuje kratek program, ki analizira sentiment prvega in zadnjega stavka iz knjige ter prikaže njegovo polarizacijo sentimenta in oceno subjektivnosti/objektivnosti.

Uporabiti morate knjižnico `TextBlob` (opisano zgoraj) za določanje `sentimenta` (ni vam treba napisati lastnega kalkulatorja sentimenta) v naslednji nalogi.

```python
from textblob import TextBlob

quote1 = """It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife."""

quote2 = """Darcy, as well as Elizabeth, really loved them; and they were both ever sensible of the warmest gratitude towards the persons who, by bringing her into Derbyshire, had been the means of uniting them."""

sentiment1 = TextBlob(quote1).sentiment
sentiment2 = TextBlob(quote2).sentiment

print(quote1 + " has a sentiment of " + str(sentiment1))
print(quote2 + " has a sentiment of " + str(sentiment2))
```

Vidite naslednji izhod:

```output
It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want # of a wife. has a sentiment of Sentiment(polarity=0.20952380952380953, subjectivity=0.27142857142857146)

Darcy, as well as Elizabeth, really loved them; and they were
     both ever sensible of the warmest gratitude towards the persons
      who, by bringing her into Derbyshire, had been the means of
      uniting them. has a sentiment of Sentiment(polarity=0.7, subjectivity=0.8)
```

## Izziv - preverite polarizacijo sentimenta

Vaša naloga je določiti, ali ima *Prevzetnost in pristranost* več absolutno pozitivnih stavkov kot absolutno negativnih. Za to nalogo lahko predpostavite, da je polarizacijska ocena 1 ali -1 absolutno pozitivna ali negativna.

**Koraki:**

1. Prenesite [kopijo Prevzetnosti in pristranosti](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm) s Project Gutenberg kot .txt datoteko. Odstranite metapodatke na začetku in koncu datoteke, tako da ostane samo originalno besedilo.
2. Odprite datoteko v Pythonu in izvlecite vsebino kot niz.
3. Ustvarite TextBlob z nizom knjige.
4. Analizirajte vsak stavek v knjigi v zanki.
   1. Če je polarizacija 1 ali -1, shranite stavek v seznam pozitivnih ali negativnih sporočil.
5. Na koncu natisnite vse pozitivne stavke in negativne stavke (ločeno) ter število vsakega.

Tukaj je vzorec [rešitve](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/3-Translation-Sentiment/solution/notebook.ipynb).

✅ Preverjanje znanja

1. Sentiment temelji na besedah, uporabljenih v stavku, vendar ali koda *razume* besede?
2. Ali menite, da je polarizacija sentimenta natančna, oziroma ali se *strinjate* z ocenami?
   1. Zlasti, ali se strinjate ali ne strinjate z absolutno **pozitivno** polarizacijo naslednjih stavkov?
      * “What an excellent father you have, girls!” said she, when the door was shut.
      * “Your examination of Mr. Darcy is over, I presume,” said Miss Bingley; “and pray what is the result?” “I am perfectly convinced by it that Mr. Darcy has no defect.
      * How wonderfully these sort of things occur!
      * I have the greatest dislike in the world to that sort of thing.
      * Charlotte is an excellent manager, I dare say.
      * “This is delightful indeed!
      * I am so happy!
      * Your idea of the ponies is delightful.
   2. Naslednji 3 stavki so bili ocenjeni z absolutno pozitivnim sentimentom, vendar ob natančnem branju niso pozitivni stavki. Zakaj je analiza sentimenta menila, da so pozitivni stavki?
      * Happy shall I be, when his stay at Netherfield is over!” “I wish I could say anything to comfort you,” replied Elizabeth; “but it is wholly out of my power.
      * If I could but see you as happy!
      * Our distress, my dear Lizzy, is very great.
   3. Ali se strinjate ali ne strinjate z absolutno **negativno** polarizacijo naslednjih stavkov?
      - Everybody is disgusted with his pride.
      - “I should like to know how he behaves among strangers.” “You shall hear then—but prepare yourself for something very dreadful.
      - The pause was to Elizabeth’s feelings dreadful.
      - It would be dreadful!

✅ Vsak ljubitelj Jane Austen bo razumel, da pogosto uporablja svoje knjige za kritiko bolj smešnih vidikov angleške regentske družbe. Elizabeth Bennett, glavna junakinja v *Prevzetnosti in pristranosti*, je ostra opazovalka družbe (kot avtorica) in njen jezik je pogosto močno niansiran. Tudi gospod Darcy (ljubezenski interes v zgodbi) opazi Elizabethino igrivo in dražilno uporabo jezika: "I have had the pleasure of your acquaintance long enough to know that you find great enjoyment in occasionally professing opinions which in fact are not your own."

---

## 🚀Izziv

Ali lahko Marvina še izboljšate z ekstrakcijo drugih značilnosti iz uporabniškega vnosa?

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Pregled in samostojno učenje
Obstaja veliko načinov za pridobivanje sentimenta iz besedila. Pomislite na poslovne aplikacije, ki bi lahko uporabile to tehniko. Razmislite o tem, kako lahko gre kaj narobe. Preberite več o naprednih sistemih, pripravljenih za podjetja, ki analizirajo sentiment, kot je [Azure Text Analysis](https://docs.microsoft.com/azure/cognitive-services/Text-Analytics/how-tos/text-analytics-how-to-sentiment-analysis?tabs=version-3-1?WT.mc_id=academic-77952-leestott). Preizkusite nekaj stavkov iz "Prevzetnost in pristranost" zgoraj in preverite, ali lahko zazna nianse.

## Naloga

[Pesniška svoboda](assignment.md)

---

**Omejitev odgovornosti**:  
Ta dokument je bil preveden z uporabo storitve za strojno prevajanje [Co-op Translator](https://github.com/Azure/co-op-translator). Čeprav si prizadevamo za natančnost, vas prosimo, da upoštevate, da lahko avtomatizirani prevodi vsebujejo napake ali netočnosti. Izvirni dokument v njegovem izvirnem jeziku je treba obravnavati kot avtoritativni vir. Za ključne informacije priporočamo strokovno človeško prevajanje. Ne prevzemamo odgovornosti za morebitna nesporazumevanja ali napačne razlage, ki izhajajo iz uporabe tega prevoda.