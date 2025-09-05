<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "be03c8182982b87ced155e4e9d1438e8",
  "translation_date": "2025-09-05T08:07:13+00:00",
  "source_file": "6-NLP/3-Translation-Sentiment/README.md",
  "language_code": "lt"
}
-->
# Vertimas ir nuotaikos analizė su ML

Ankstesnėse pamokose išmokote sukurti pagrindinį botą naudojant `TextBlob`, biblioteką, kuri užkulisiuose naudoja ML, kad atliktų pagrindines NLP užduotis, tokias kaip daiktavardžių frazių ištraukimas. Kitas svarbus iššūkis kompiuterinėje lingvistikoje yra tiksli _sakinių vertimo_ iš vienos kalbos į kitą problema.

## [Prieš paskaitą viktorina](https://ff-quizzes.netlify.app/en/ml/)

Vertimas yra labai sudėtinga problema, kurią dar labiau apsunkina tai, kad pasaulyje yra tūkstančiai kalbų, ir kiekviena jų turi labai skirtingas gramatikos taisykles. Vienas iš būdų yra konvertuoti vienos kalbos, pavyzdžiui, anglų, gramatikos taisykles į struktūrą, nepriklausomą nuo kalbos, ir tada išversti ją, konvertuojant į kitą kalbą. Šis metodas apima šiuos žingsnius:

1. **Identifikacija**. Identifikuoti arba pažymėti žodžius įvesties kalboje kaip daiktavardžius, veiksmažodžius ir pan.
2. **Sukurti vertimą**. Sukurti tiesioginį kiekvieno žodžio vertimą tikslinės kalbos formatu.

### Pavyzdinis sakinys, iš anglų į airių kalbą

Anglų kalboje sakinys _I feel happy_ yra trijų žodžių eilės tvarka:

- **subjektas** (I)
- **veiksmažodis** (feel)
- **būdvardis** (happy)

Tačiau airių kalboje tas pats sakinys turi visiškai kitokią gramatinę struktūrą – emocijos, tokios kaip "*happy*" ar "*sad*", išreiškiamos kaip *esančios ant tavęs*.

Angliškas sakinys `I feel happy` airių kalboje būtų `Tá athas orm`. *Tiesioginis* vertimas būtų `Happy is upon me`.

Airių kalbos vartotojas, versdamas į anglų kalbą, pasakytų `I feel happy`, o ne `Happy is upon me`, nes jis supranta sakinio prasmę, net jei žodžiai ir sakinio struktūra skiriasi.

Formalus sakinio tvarkos išdėstymas airių kalboje yra:

- **veiksmažodis** (Tá arba is)
- **būdvardis** (athas, arba happy)
- **subjektas** (orm, arba upon me)

## Vertimas

Naivus vertimo programa gali versti tik žodžius, ignoruodama sakinio struktūrą.

✅ Jei išmokote antrą (ar trečią ar daugiau) kalbą kaip suaugęs, galbūt pradėjote galvoti savo gimtąja kalba, mintyse versti žodį po žodžio į antrą kalbą, o tada išsakyti savo vertimą. Tai panašu į tai, ką daro naivios vertimo kompiuterinės programos. Svarbu pereiti šį etapą, kad pasiektumėte sklandumą!

Naivus vertimas sukelia blogus (ir kartais juokingus) klaidingus vertimus: `I feel happy` tiesiogiai verčiamas į `Mise bhraitheann athas` airių kalboje. Tai reiškia (tiesiogiai) `me feel happy` ir nėra tinkamas airių kalbos sakinys. Nors anglų ir airių kalbos yra kalbos, vartojamos dviejose artimai kaimyninėse salose, jos yra labai skirtingos kalbos su skirtingomis gramatikos struktūromis.

> Galite pažiūrėti keletą vaizdo įrašų apie airių kalbos tradicijas, pavyzdžiui, [šį](https://www.youtube.com/watch?v=mRIaLSdRMMs)

### Mašininio mokymosi metodai

Iki šiol sužinojote apie formalų taisyklių metodą natūralios kalbos apdorojimui. Kitas metodas yra ignoruoti žodžių prasmę ir _vietoj to naudoti mašininį mokymąsi, kad aptiktumėte dėsningumus_. Tai gali veikti vertime, jei turite daug tekstų (*korpusą*) arba tekstų (*korpusus*) tiek originalo, tiek tikslinės kalbos.

Pavyzdžiui, apsvarstykite *Puikybė ir prietarai* atvejį – gerai žinomą anglų romaną, kurį 1813 m. parašė Jane Austen. Jei peržiūrėtumėte knygą anglų kalba ir žmogaus vertimą į *prancūzų* kalbą, galėtumėte aptikti frazes, kurios vienoje kalboje yra _idiomatiškai_ išverstos į kitą. Tai padarysite netrukus.

Pavyzdžiui, kai angliška frazė `I have no money` tiesiogiai išverčiama į prancūzų kalbą, ji gali tapti `Je n'ai pas de monnaie`. "Monnaie" yra sudėtingas prancūzų 'klaidingas draugas', nes 'money' ir 'monnaie' nėra sinonimai. Geresnis vertimas, kurį galėtų atlikti žmogus, būtų `Je n'ai pas d'argent`, nes jis geriau perteikia prasmę, kad neturite pinigų (o ne 'smulkių', kas yra 'monnaie' reikšmė).

![monnaie](../../../../6-NLP/3-Translation-Sentiment/images/monnaie.png)

> Vaizdas sukurtas [Jen Looper](https://twitter.com/jenlooper)

Jei ML modelis turi pakankamai žmogaus vertimų, kad galėtų sukurti modelį, jis gali pagerinti vertimų tikslumą, identifikuodamas bendrus dėsningumus tekstuose, kurie anksčiau buvo išversti ekspertų, kalbančių abiem kalbomis.

### Užduotis - vertimas

Galite naudoti `TextBlob`, kad išverstumėte sakinius. Išbandykite garsųjį pirmąjį **Puikybės ir prietarų** sakinį:

```python
from textblob import TextBlob

blob = TextBlob(
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife!"
)
print(blob.translate(to="fr"))

```

`TextBlob` gana gerai atlieka vertimą: "C'est une vérité universellement reconnue, qu'un homme célibataire en possession d'une bonne fortune doit avoir besoin d'une femme!". 

Galima teigti, kad TextBlob vertimas yra daug tikslesnis, nei 1932 m. prancūzų vertimas, kurį atliko V. Leconte ir Ch. Pressoir:

"C'est une vérité universelle qu'un célibataire pourvu d'une belle fortune doit avoir envie de se marier, et, si peu que l'on sache de son sentiment à cet egard, lorsqu'il arrive dans une nouvelle résidence, cette idée est si bien fixée dans l'esprit de ses voisins qu'ils le considèrent sur-le-champ comme la propriété légitime de l'une ou l'autre de leurs filles."

Šiuo atveju ML pagrįstas vertimas atlieka geresnį darbą nei žmogaus vertėjas, kuris nereikalingai prideda žodžių originalaus autoriaus tekstui, siekdamas 'aiškumo'.

> Kas čia vyksta? Kodėl TextBlob taip gerai atlieka vertimą? Na, užkulisiuose jis naudoja Google Translate, sudėtingą AI, galintį analizuoti milijonus frazių, kad numatytų geriausias eilutes užduočiai atlikti. Čia nėra nieko rankinio, ir jums reikia interneto ryšio, kad galėtumėte naudoti `blob.translate`.

✅ Išbandykite keletą kitų sakinių. Kuris geresnis – ML ar žmogaus vertimas? Kokiais atvejais?

## Nuotaikos analizė

Kita sritis, kurioje mašininis mokymasis gali veikti labai gerai, yra nuotaikos analizė. Ne ML metodas nuotaikai nustatyti yra identifikuoti žodžius ir frazes, kurios yra 'teigiamos' ir 'neigiamos'. Tada, gavus naują tekstą, apskaičiuoti bendrą teigiamų, neigiamų ir neutralių žodžių vertę, kad nustatytumėte bendrą nuotaiką.

Šis metodas lengvai apgaunamas, kaip galėjote pastebėti Marvin užduotyje – sakinys `Great, that was a wonderful waste of time, I'm glad we are lost on this dark road` yra sarkastiškas, neigiamas nuotaikos sakinys, tačiau paprastas algoritmas aptinka 'great', 'wonderful', 'glad' kaip teigiamus ir 'waste', 'lost' ir 'dark' kaip neigiamus. Bendrą nuotaiką paveikia šie prieštaringi žodžiai.

✅ Sustokite akimirkai ir pagalvokite, kaip mes, žmonės, perteikiame sarkazmą. Tonas vaidina didelį vaidmenį. Pabandykite pasakyti frazę "Well, that film was awesome" skirtingais būdais, kad suprastumėte, kaip jūsų balsas perteikia prasmę.

### ML metodai

ML metodas būtų rankiniu būdu surinkti neigiamus ir teigiamus tekstus – tviterio žinutes, filmų apžvalgas ar bet ką, kur žmogus pateikė įvertinimą *ir* rašytą nuomonę. Tada NLP technikos gali būti taikomos nuomonėms ir įvertinimams, kad atsirastų dėsningumai (pvz., teigiamos filmų apžvalgos dažniau turi frazę 'Oscar worthy' nei neigiamos filmų apžvalgos, arba teigiamos restoranų apžvalgos dažniau naudoja 'gourmet' nei 'disgusting').

> ⚖️ **Pavyzdys**: Jei dirbtumėte politiko biure ir būtų svarstomas naujas įstatymas, rinkėjai galėtų rašyti biurui el. laiškus, palaikančius arba prieštaraujančius tam tikram naujam įstatymui. Tarkime, jums pavesta perskaityti el. laiškus ir suskirstyti juos į 2 krūvas, *už* ir *prieš*. Jei būtų daug el. laiškų, galėtumėte jaustis priblokšti bandydami perskaityti juos visus. Ar nebūtų puiku, jei bot'as galėtų perskaityti juos visus už jus, suprasti juos ir pasakyti, į kurią krūvą kiekvienas el. laiškas priklauso? 
> 
> Vienas būdas tai pasiekti yra naudoti mašininį mokymąsi. Jūs treniruotumėte modelį su dalimi *prieš* el. laiškų ir dalimi *už* el. laiškų. Modelis linkęs susieti frazes ir žodžius su prieš arba už pusėmis, *bet jis nesuprastų jokio turinio*, tik tai, kad tam tikri žodžiai ir dėsningumai dažniau pasirodo *prieš* arba *už* el. laiške. Galėtumėte jį išbandyti su kai kuriais el. laiškais, kurių nenaudojote modelio treniravimui, ir pažiūrėti, ar jis priėjo prie tokios pačios išvados kaip jūs. Tada, kai būtumėte patenkinti modelio tikslumu, galėtumėte apdoroti būsimus el. laiškus, nereikėdami skaityti kiekvieno.

✅ Ar šis procesas panašus į procesus, kuriuos naudojote ankstesnėse pamokose?

## Užduotis - nuotaikos sakiniai

Nuotaika matuojama *poliarumu* nuo -1 iki 1, kur -1 yra pati neigiama nuotaika, o 1 yra pati teigiama. Nuotaika taip pat matuojama 0 - 1 skalėje objektyvumo (0) ir subjektyvumo (1).

Dar kartą pažvelkite į Jane Austen *Puikybę ir prietarus*. Tekstas pasiekiamas čia: [Project Gutenberg](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm). Žemiau pateiktas pavyzdys rodo trumpą programą, kuri analizuoja pirmo ir paskutinio sakinio nuotaiką iš knygos ir parodo jos poliarumo bei subjektyvumo/objektyvumo įvertinimą.

Turėtumėte naudoti `TextBlob` biblioteką (aprašytą aukščiau), kad nustatytumėte `nuotaiką` (jums nereikia rašyti savo nuotaikos skaičiuoklės) šioje užduotyje.

```python
from textblob import TextBlob

quote1 = """It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife."""

quote2 = """Darcy, as well as Elizabeth, really loved them; and they were both ever sensible of the warmest gratitude towards the persons who, by bringing her into Derbyshire, had been the means of uniting them."""

sentiment1 = TextBlob(quote1).sentiment
sentiment2 = TextBlob(quote2).sentiment

print(quote1 + " has a sentiment of " + str(sentiment1))
print(quote2 + " has a sentiment of " + str(sentiment2))
```

Matote šį rezultatą:

```output
It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want # of a wife. has a sentiment of Sentiment(polarity=0.20952380952380953, subjectivity=0.27142857142857146)

Darcy, as well as Elizabeth, really loved them; and they were
     both ever sensible of the warmest gratitude towards the persons
      who, by bringing her into Derbyshire, had been the means of
      uniting them. has a sentiment of Sentiment(polarity=0.7, subjectivity=0.8)
```

## Iššūkis - patikrinkite nuotaikos poliarumą

Jūsų užduotis yra nustatyti, naudojant nuotaikos poliarumą, ar *Puikybė ir prietarai* turi daugiau absoliučiai teigiamų sakinių nei absoliučiai neigiamų. Šiai užduočiai galite manyti, kad poliarumo įvertinimas 1 arba -1 yra absoliučiai teigiamas arba neigiamas atitinkamai.

**Žingsniai:**

1. Atsisiųskite [Puikybės ir prietarų kopiją](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm) iš Project Gutenberg kaip .txt failą. Pašalinkite metaduomenis failo pradžioje ir pabaigoje, palikdami tik originalų tekstą
2. Atidarykite failą Python'e ir ištraukite turinį kaip eilutę
3. Sukurkite TextBlob naudodami knygos eilutę
4. Analizuokite kiekvieną sakinio knygoje cikle
   1. Jei poliarumas yra 1 arba -1, išsaugokite sakinį teigiamų arba neigiamų pranešimų masyve ar sąraše
5. Pabaigoje atspausdinkite visus teigiamus sakinius ir neigiamus sakinius (atskirai) bei jų skaičių.

Štai pavyzdinis [sprendimas](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/3-Translation-Sentiment/solution/notebook.ipynb).

✅ Žinių patikrinimas

1. Nuotaika pagrįsta sakinyje naudojamais žodžiais, bet ar kodas *supranta* žodžius?
2. Ar manote, kad nuotaikos poliarumas yra tikslus, kitaip tariant, ar jūs *sutinkate* su įvertinimais?
   1. Visų pirma, ar sutinkate ar nesutinkate su absoliučiai **teigiamu** šių sakinių poliarumu?
      * “What an excellent father you have, girls!” said she, when the door was shut.
      * “Your examination of Mr. Darcy is over, I presume,” said Miss Bingley; “and pray what is the result?” “I am perfectly convinced by it that Mr. Darcy has no defect.
      * How wonderfully these sort of things occur!
      * I have the greatest dislike in the world to that sort of thing.
      * Charlotte is an excellent manager, I dare say.
      * “This is delightful indeed!
      * I am so happy!
      * Your idea of the ponies is delightful.
   2. Kiti 3 sakiniai buvo įvertinti absoliučiai teigiamai, tačiau atidžiai perskaičius, jie nėra teigiami sakiniai. Kodėl nuotaikos analizė manė, kad jie buvo teigiami sakiniai?
      * Happy shall I be, when his stay at Netherfield is over!” “I wish I could say anything to comfort you,” replied Elizabeth; “but it is wholly out of my power.
      * If I could but see you as happy!
      * Our distress, my dear Lizzy, is very great.
   3. Ar sutinkate ar nesutinkate su absoliučiai **neigiamu** šių sakinių poliarumu?
      - Everybody is disgusted with his pride.
      - “I should like to know how he behaves among strangers.” “You shall hear then—but prepare yourself for something very dreadful.
      - The pause was to Elizabeth’s feelings dreadful.
      - It would be dreadful!

✅ Kiekvienas Jane Austen gerbėjas supras, kad ji dažnai naudoja savo knygas, kad kritikuotų absurdiškus Anglijos regencijos laikotarpio visuomenės aspektus. Elizabeth Bennett, pagrindinė veikėja *Puikybėje ir prietaruose*, yra įžvalgi socialinė stebėtoja (kaip ir autorė), ir jos kalba dažnai yra labai subtili. Net Mr. Darcy (istorijos meilės objektas) pastebi Elizabeth žaismingą ir erzinančią kalbos vartojimą: "Aš turėjau malonumą pažinti jus pakankamai ilgai, kad žinočiau, jog jums labai patinka kartais išreikšti nuomones, kurios iš tikrųjų nėra jūsų."

---

## 🚀Iššūkis

Ar galite padaryti Marvin dar geresnį
Yra daug būdų išgauti emocijas iš teksto. Pagalvokite apie verslo taikymus, kurie galėtų pasinaudoti šia technika. Pagalvokite, kaip tai gali suklysti. Skaitykite daugiau apie sudėtingas, įmonėms pritaikytas sistemas, kurios analizuoja emocijas, tokias kaip [Azure Text Analysis](https://docs.microsoft.com/azure/cognitive-services/Text-Analytics/how-tos/text-analytics-how-to-sentiment-analysis?tabs=version-3-1?WT.mc_id=academic-77952-leestott). Išbandykite kai kurias aukščiau pateiktas „Puikybė ir prietarai“ sakinių ir pažiūrėkite, ar sistema gali aptikti niuansus.

## Užduotis

[Poetinė licencija](assignment.md)

---

**Atsakomybės apribojimas**:  
Šis dokumentas buvo išverstas naudojant AI vertimo paslaugą [Co-op Translator](https://github.com/Azure/co-op-translator). Nors siekiame tikslumo, prašome atkreipti dėmesį, kad automatiniai vertimai gali turėti klaidų ar netikslumų. Originalus dokumentas jo gimtąja kalba turėtų būti laikomas autoritetingu šaltiniu. Kritinei informacijai rekomenduojama naudoti profesionalų žmogaus vertimą. Mes neprisiimame atsakomybės už nesusipratimus ar klaidingus interpretavimus, atsiradusius dėl šio vertimo naudojimo.