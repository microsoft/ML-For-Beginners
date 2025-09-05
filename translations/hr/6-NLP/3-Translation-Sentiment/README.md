<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "be03c8182982b87ced155e4e9d1438e8",
  "translation_date": "2025-09-05T14:14:15+00:00",
  "source_file": "6-NLP/3-Translation-Sentiment/README.md",
  "language_code": "hr"
}
-->
# Prijevod i analiza sentimenta s ML-om

U prethodnim lekcijama naučili ste kako izraditi osnovnog bota koristeći `TextBlob`, biblioteku koja koristi strojno učenje iza kulisa za obavljanje osnovnih NLP zadataka poput izdvajanja imenskih fraza. Još jedan važan izazov u računalnoj lingvistici je točno _prevođenje_ rečenice s jednog govornog ili pisanog jezika na drugi.

## [Pre-lecture kviz](https://ff-quizzes.netlify.app/en/ml/)

Prevođenje je vrlo težak problem, dodatno otežan činjenicom da postoje tisuće jezika, od kojih svaki može imati vrlo različita gramatička pravila. Jedan pristup je pretvoriti formalna gramatička pravila jednog jezika, poput engleskog, u strukturu neovisnu o jeziku, a zatim je prevesti pretvaranjem natrag u drugi jezik. Ovaj pristup uključuje sljedeće korake:

1. **Identifikacija**. Identificirajte ili označite riječi u ulaznom jeziku kao imenice, glagole itd.
2. **Izrada prijevoda**. Proizvedite izravan prijevod svake riječi u formatu ciljnog jezika.

### Primjer rečenice, engleski na irski

Na 'engleskom', rečenica _I feel happy_ sastoji se od tri riječi u sljedećem redoslijedu:

- **subjekt** (I)
- **glagol** (feel)
- **pridjev** (happy)

Međutim, na 'irskom' jeziku, ista rečenica ima vrlo drugačiju gramatičku strukturu - emocije poput "*happy*" ili "*sad*" izražavaju se kao da su *na tebi*.

Engleska fraza `I feel happy` na irskom bi bila `Tá athas orm`. Doslovni prijevod bio bi `Sreća je na meni`.

Govornik irskog jezika koji prevodi na engleski rekao bi `I feel happy`, a ne `Happy is upon me`, jer razumije značenje rečenice, čak i ako su riječi i struktura rečenice različite.

Formalni redoslijed za rečenicu na irskom je:

- **glagol** (Tá ili jest)
- **pridjev** (athas, ili sretan)
- **subjekt** (orm, ili na meni)

## Prevođenje

Naivan program za prevođenje mogao bi prevoditi samo riječi, ignorirajući strukturu rečenice.

✅ Ako ste naučili drugi (ili treći ili više) jezik kao odrasla osoba, možda ste započeli razmišljanjem na svom materinjem jeziku, prevodeći koncept riječ po riječ u svojoj glavi na drugi jezik, a zatim izgovarajući svoj prijevod. Ovo je slično onome što rade naivni računalni programi za prevođenje. Važno je prijeći ovu fazu kako biste postigli tečnost!

Naivno prevođenje dovodi do loših (i ponekad smiješnih) pogrešnih prijevoda: `I feel happy` doslovno se prevodi kao `Mise bhraitheann athas` na irskom. To znači (doslovno) `ja osjećam sreću` i nije valjana irska rečenica. Iako su engleski i irski jezici koji se govore na dva susjedna otoka, oni su vrlo različiti jezici s različitim gramatičkim strukturama.

> Možete pogledati neke videozapise o irskim jezičnim tradicijama, poput [ovog](https://www.youtube.com/watch?v=mRIaLSdRMMs)

### Pristupi strojnog učenja

Do sada ste naučili o pristupu formalnih pravila u obradi prirodnog jezika. Drugi pristup je ignorirati značenje riječi i _umjesto toga koristiti strojno učenje za otkrivanje obrazaca_. Ovo može funkcionirati u prevođenju ako imate puno teksta (*korpus*) ili tekstova (*korpusi*) na izvornom i ciljanom jeziku.

Na primjer, razmotrite slučaj *Ponos i predrasude*, poznatog engleskog romana koji je napisala Jane Austen 1813. godine. Ako konzultirate knjigu na engleskom i ljudski prijevod knjige na *francuski*, mogli biste otkriti fraze u jednoj koje su _idiomatski_ prevedene u drugu. To ćete učiniti za trenutak.

Na primjer, kada se engleska fraza `I have no money` doslovno prevede na francuski, mogla bi postati `Je n'ai pas de monnaie`. "Monnaie" je nezgodan francuski 'lažni prijatelj', jer 'money' i 'monnaie' nisu sinonimi. Bolji prijevod koji bi ljudski prevoditelj mogao napraviti bio bi `Je n'ai pas d'argent`, jer bolje prenosi značenje da nemate novca (umjesto 'sitniša', što je značenje 'monnaie').

![monnaie](../../../../6-NLP/3-Translation-Sentiment/images/monnaie.png)

> Slika od [Jen Looper](https://twitter.com/jenlooper)

Ako ML model ima dovoljno ljudskih prijevoda na temelju kojih može izgraditi model, može poboljšati točnost prijevoda identificiranjem uobičajenih obrazaca u tekstovima koje su prethodno preveli stručni govornici oba jezika.

### Vježba - prevođenje

Možete koristiti `TextBlob` za prevođenje rečenica. Isprobajte poznatu prvu rečenicu **Ponosa i predrasuda**:

```python
from textblob import TextBlob

blob = TextBlob(
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife!"
)
print(blob.translate(to="fr"))

```

`TextBlob` prilično dobro obavlja prijevod: "C'est une vérité universellement reconnue, qu'un homme célibataire en possession d'une bonne fortune doit avoir besoin d'une femme!". 

Može se tvrditi da je prijevod TextBlob-a daleko točniji, zapravo, od francuskog prijevoda knjige iz 1932. godine od V. Leconte i Ch. Pressoir:

"C'est une vérité universelle qu'un célibataire pourvu d'une belle fortune doit avoir envie de se marier, et, si peu que l'on sache de son sentiment à cet egard, lorsqu'il arrive dans une nouvelle résidence, cette idée est si bien fixée dans l'esprit de ses voisins qu'ils le considèrent sur-le-champ comme la propriété légitime de l'une ou l'autre de leurs filles."

U ovom slučaju, prijevod informiran ML-om obavlja bolji posao od ljudskog prevoditelja koji nepotrebno stavlja riječi u usta originalnog autora radi 'jasnoće'.

> Što se ovdje događa? I zašto je TextBlob tako dobar u prevođenju? Pa, iza kulisa koristi Google Translate, sofisticirani AI sposoban analizirati milijune fraza kako bi predvidio najbolje nizove za zadatak. Ovdje se ništa ne radi ručno i potrebna vam je internetska veza za korištenje `blob.translate`.

✅ Isprobajte još nekoliko rečenica. Što je bolje, ML ili ljudski prijevod? U kojim slučajevima?

## Analiza sentimenta

Još jedno područje gdje strojno učenje može vrlo dobro funkcionirati je analiza sentimenta. Pristup bez ML-a sentimentu je identificiranje riječi i fraza koje su 'pozitivne' i 'negativne'. Zatim, s obzirom na novi tekst, izračunajte ukupnu vrijednost pozitivnih, negativnih i neutralnih riječi kako biste identificirali ukupni sentiment. 

Ovaj pristup se lako može prevariti, kao što ste možda vidjeli u zadatku Marvin - rečenica `Great, that was a wonderful waste of time, I'm glad we are lost on this dark road` je sarkastična, negativna rečenica, ali jednostavni algoritam detektira 'great', 'wonderful', 'glad' kao pozitivne i 'waste', 'lost' i 'dark' kao negativne. Ukupni sentiment je pod utjecajem ovih suprotstavljenih riječi.

✅ Zastanite na trenutak i razmislite o tome kako kao govornici prenosimo sarkazam. Intonacija igra veliku ulogu. Pokušajte izgovoriti frazu "Well, that film was awesome" na različite načine kako biste otkrili kako vaš glas prenosi značenje.

### Pristupi ML-a

Pristup ML-a bio bi ručno prikupljanje negativnih i pozitivnih tekstova - tweetova, recenzija filmova ili bilo čega gdje je čovjek dao ocjenu *i* napisano mišljenje. Zatim se NLP tehnike mogu primijeniti na mišljenja i ocjene, tako da se pojavljuju obrasci (npr., pozitivne recenzije filmova imaju frazu 'Oscar worthy' više nego negativne recenzije filmova, ili pozitivne recenzije restorana kažu 'gourmet' mnogo više nego 'disgusting').

> ⚖️ **Primjer**: Ako radite u uredu političara i raspravlja se o nekom novom zakonu, birači bi mogli pisati uredu s e-mailovima koji podržavaju ili se protive određenom novom zakonu. Recimo da vam je zadatak čitati e-mailove i razvrstavati ih u 2 hrpe, *za* i *protiv*. Ako bi bilo puno e-mailova, mogli biste biti preopterećeni pokušajem da ih sve pročitate. Ne bi li bilo lijepo da bot može pročitati sve za vas, razumjeti ih i reći vam u koju hrpu svaki e-mail pripada? 
> 
> Jedan način da se to postigne je korištenje strojnog učenja. Model biste trenirali s dijelom *protiv* e-mailova i dijelom *za* e-mailova. Model bi imao tendenciju povezivanja fraza i riječi s protiv stranom i za stranom, *ali ne bi razumio nijedan sadržaj*, samo da se određene riječi i obrasci češće pojavljuju u *protiv* ili *za* e-mailovima. Mogli biste ga testirati s nekim e-mailovima koje niste koristili za treniranje modela i vidjeti dolazi li do istog zaključka kao i vi. Zatim, kada budete zadovoljni točnošću modela, mogli biste obrađivati buduće e-mailove bez potrebe da čitate svaki.

✅ Zvuči li vam ovaj proces kao procesi koje ste koristili u prethodnim lekcijama?

## Vježba - sentimentalne rečenice

Sentiment se mjeri s *polaritetom* od -1 do 1, što znači da je -1 najnegativniji sentiment, a 1 najpozitivniji. Sentiment se također mjeri s ocjenom od 0 - 1 za objektivnost (0) i subjektivnost (1).

Pogledajte ponovno *Ponos i predrasude* Jane Austen. Tekst je dostupan ovdje na [Project Gutenberg](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm). Uzorak u nastavku prikazuje kratki program koji analizira sentiment prve i posljednje rečenice iz knjige i prikazuje njezin polaritet sentimenta i ocjenu subjektivnosti/objektivnosti.

Trebali biste koristiti biblioteku `TextBlob` (opisanu gore) za određivanje `sentimenta` (ne morate pisati vlastiti kalkulator sentimenta) u sljedećem zadatku.

```python
from textblob import TextBlob

quote1 = """It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife."""

quote2 = """Darcy, as well as Elizabeth, really loved them; and they were both ever sensible of the warmest gratitude towards the persons who, by bringing her into Derbyshire, had been the means of uniting them."""

sentiment1 = TextBlob(quote1).sentiment
sentiment2 = TextBlob(quote2).sentiment

print(quote1 + " has a sentiment of " + str(sentiment1))
print(quote2 + " has a sentiment of " + str(sentiment2))
```

Vidite sljedeći izlaz:

```output
It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want # of a wife. has a sentiment of Sentiment(polarity=0.20952380952380953, subjectivity=0.27142857142857146)

Darcy, as well as Elizabeth, really loved them; and they were
     both ever sensible of the warmest gratitude towards the persons
      who, by bringing her into Derbyshire, had been the means of
      uniting them. has a sentiment of Sentiment(polarity=0.7, subjectivity=0.8)
```

## Izazov - provjerite polaritet sentimenta

Vaš zadatak je odrediti, koristeći polaritet sentimenta, ima li *Ponos i predrasude* više apsolutno pozitivnih rečenica nego apsolutno negativnih. Za ovaj zadatak možete pretpostaviti da je polaritetna ocjena od 1 ili -1 apsolutno pozitivna ili negativna.

**Koraci:**

1. Preuzmite [kopiju Ponosa i predrasuda](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm) s Project Gutenberg kao .txt datoteku. Uklonite metapodatke na početku i kraju datoteke, ostavljajući samo originalni tekst
2. Otvorite datoteku u Pythonu i izdvojite sadržaj kao string
3. Kreirajte TextBlob koristeći string knjige
4. Analizirajte svaku rečenicu u knjizi u petlji
   1. Ako je polaritet 1 ili -1, pohranite rečenicu u niz ili popis pozitivnih ili negativnih poruka
5. Na kraju, ispišite sve pozitivne rečenice i negativne rečenice (odvojeno) i broj svake.

Evo uzorka [rješenja](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/3-Translation-Sentiment/solution/notebook.ipynb).

✅ Provjera znanja

1. Sentiment se temelji na riječima koje se koriste u rečenici, ali razumije li kod *riječi*?
2. Mislite li da je polaritet sentimenta točan, odnosno slažete li se s ocjenama?
   1. Konkretno, slažete li se ili ne slažete s apsolutno **pozitivnim** polaritetom sljedećih rečenica?
      * “What an excellent father you have, girls!” said she, when the door was shut.
      * “Your examination of Mr. Darcy is over, I presume,” said Miss Bingley; “and pray what is the result?” “I am perfectly convinced by it that Mr. Darcy has no defect.
      * How wonderfully these sort of things occur!
      * I have the greatest dislike in the world to that sort of thing.
      * Charlotte is an excellent manager, I dare say.
      * “This is delightful indeed!
      * I am so happy!
      * Your idea of the ponies is delightful.
   2. Sljedeće 3 rečenice ocijenjene su apsolutno pozitivnim sentimentom, ali pri pažljivom čitanju, one nisu pozitivne rečenice. Zašto je analiza sentimenta mislila da su pozitivne rečenice?
      * Happy shall I be, when his stay at Netherfield is over!” “I wish I could say anything to comfort you,” replied Elizabeth; “but it is wholly out of my power.
      * If I could but see you as happy!
      * Our distress, my dear Lizzy, is very great.
   3. Slažete li se ili ne slažete s apsolutno **negativnim** polaritetom sljedećih rečenica?
      - Everybody is disgusted with his pride.
      - “I should like to know how he behaves among strangers.” “You shall hear then—but prepare yourself for something very dreadful.
      - The pause was to Elizabeth’s feelings dreadful.
      - It would be dreadful!

✅ Svaki zaljubljenik u Jane Austen razumjet će da ona često koristi svoje knjige za kritiku smiješnijih aspekata engleskog društva iz razdoblja regencije. Elizabeth Bennett, glavni lik u *Ponosa i predrasuda*, je oštroumna društvena promatračica (poput autorice) i njezin jezik često je vrlo nijansiran. Čak i gospodin Darcy (ljubavni interes u priči) primjećuje Elizabethinu razigranu i zadirkujuću upotrebu jezika: "I have had the pleasure of your acquaintance long enough to know that you find great enjoyment in occasionally professing opinions which in fact are not your own."

---

## 🚀Izazov

Možete li učiniti Marvina još boljim tako da izvučete druge značajke iz korisničkog unosa?

## [Post-lecture kviz](https://ff-quizzes.netlify.app/en/ml/)

## Pregled i samostalno učenje
Postoji mnogo načina za izdvajanje sentimenta iz teksta. Razmislite o poslovnim primjenama koje bi mogle koristiti ovu tehniku. Razmislite o tome kako može poći po zlu. Pročitajte više o sofisticiranim sustavima spremnim za poduzeća koji analiziraju sentiment, poput [Azure Text Analysis](https://docs.microsoft.com/azure/cognitive-services/Text-Analytics/how-tos/text-analytics-how-to-sentiment-analysis?tabs=version-3-1?WT.mc_id=academic-77952-leestott). Testirajte neke od rečenica iz "Ponosa i predrasuda" iznad i provjerite može li otkriti nijanse.

## Zadatak

[Poetska sloboda](assignment.md)

---

**Odricanje od odgovornosti**:  
Ovaj dokument je preveden korištenjem AI usluge za prevođenje [Co-op Translator](https://github.com/Azure/co-op-translator). Iako nastojimo osigurati točnost, imajte na umu da automatski prijevodi mogu sadržavati pogreške ili netočnosti. Izvorni dokument na izvornom jeziku treba smatrati mjerodavnim izvorom. Za ključne informacije preporučuje se profesionalni prijevod od strane stručnjaka. Ne preuzimamo odgovornost za bilo kakve nesporazume ili pogrešne interpretacije proizašle iz korištenja ovog prijevoda.