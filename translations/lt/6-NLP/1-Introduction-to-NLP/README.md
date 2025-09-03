<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "89e923cf3e8bdff9662536e8bf9516e6",
  "translation_date": "2025-09-03T18:59:04+00:00",
  "source_file": "6-NLP/1-Introduction-to-NLP/README.md",
  "language_code": "lt"
}
-->
# Įvadas į natūralios kalbos apdorojimą

Šioje pamokoje aptariama trumpa *natūralios kalbos apdorojimo* (NLP), kuris yra *kompiuterinės lingvistikos* poskyris, istorija ir svarbiausios sąvokos.

## [Prieš paskaitą: testas](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/31/)

## Įvadas

NLP, kaip jis dažnai vadinamas, yra viena iš geriausiai žinomų sričių, kuriose mašininis mokymasis buvo pritaikytas ir naudojamas gamybos programinėje įrangoje.

✅ Ar galite pagalvoti apie programinę įrangą, kurią naudojate kasdien ir kurioje tikriausiai yra įdiegta NLP? O kaip dėl jūsų teksto redagavimo programų ar mobiliųjų programėlių, kurias naudojate reguliariai?

Jūs sužinosite apie:

- **Kalbų idėją**. Kaip kalbos vystėsi ir kokios buvo pagrindinės jų tyrimų sritys.
- **Apibrėžimus ir sąvokas**. Taip pat sužinosite, kaip kompiuteriai apdoroja tekstą, įskaitant sakinių analizę, gramatiką ir daiktavardžių bei veiksmažodžių atpažinimą. Šioje pamokoje yra keletas programavimo užduočių, taip pat pristatomos kelios svarbios sąvokos, kurias išmoksite programuoti vėlesnėse pamokose.

## Kompiuterinė lingvistika

Kompiuterinė lingvistika yra tyrimų ir plėtros sritis, kuri dešimtmečius nagrinėja, kaip kompiuteriai gali dirbti su kalbomis, jas suprasti, versti ir netgi komunikuoti. Natūralios kalbos apdorojimas (NLP) yra susijusi sritis, orientuota į tai, kaip kompiuteriai gali apdoroti „natūralias“, arba žmonių, kalbas.

### Pavyzdys – diktavimas telefonu

Jei kada nors diktavote tekstą savo telefonui vietoj rašymo arba klausėte virtualaus asistento klausimo, jūsų kalba buvo paversta tekstu ir tada apdorota arba *analizuota* pagal kalbą, kuria kalbėjote. Aptikti raktiniai žodžiai buvo apdoroti formatu, kurį telefonas ar asistentas galėjo suprasti ir pagal kurį veikti.

![supratimas](../../../../translated_images/comprehension.619708fc5959b0f6a24ebffba2ad7b0625391a476141df65b43b59de24e45c6f.lt.png)
> Tikras lingvistinis supratimas yra sudėtingas! Vaizdas: [Jen Looper](https://twitter.com/jenlooper)

### Kaip ši technologija tapo įmanoma?

Tai įmanoma, nes kažkas parašė kompiuterinę programą, kuri tai atlieka. Prieš kelis dešimtmečius kai kurie mokslinės fantastikos rašytojai prognozavo, kad žmonės dažniausiai kalbės su savo kompiuteriais, o šie visada tiksliai supras, ką jie turi omenyje. Deja, paaiškėjo, kad tai yra sudėtingesnė problema, nei daugelis įsivaizdavo. Nors šiandien problema yra daug geriau suprantama, vis dar yra didelių iššūkių siekiant „tobulo“ natūralios kalbos apdorojimo, ypač kai reikia suprasti sakinio prasmę. Tai ypač sudėtinga, kai reikia suprasti humorą ar aptikti emocijas, tokias kaip sarkazmas.

Galbūt prisimenate mokyklos pamokas, kuriose mokytojas aiškino sakinio gramatikos dalis. Kai kuriose šalyse mokiniai mokomi gramatikos ir lingvistikos kaip atskiro dalyko, tačiau daugelyje šių temų mokoma kaip kalbos mokymosi dalies: arba pradinėje mokykloje mokantis skaityti ir rašyti gimtąja kalba, arba vidurinėje mokykloje mokantis antrosios kalbos. Nesijaudinkite, jei nesate ekspertas, gebantis atskirti daiktavardžius nuo veiksmažodžių ar prieveiksmius nuo būdvardžių!

Jei jums sunku atskirti *paprastąjį esamąjį laiką* nuo *esamojo progresyviojo*, jūs nesate vieni. Tai sudėtinga daugeliui žmonių, netgi gimtakalbiams. Geros naujienos yra tai, kad kompiuteriai labai gerai taiko formaliąsias taisykles, ir jūs išmoksite rašyti kodą, kuris gali *analizuoti* sakinį taip pat gerai, kaip žmogus. Didžiausias iššūkis, kurį nagrinėsite vėliau, yra sakinio *prasmės* ir *nuotaikos* supratimas.

## Reikalavimai

Šiai pamokai pagrindinis reikalavimas yra gebėjimas skaityti ir suprasti šios pamokos kalbą. Čia nėra matematikos uždavinių ar lygčių, kurias reikėtų spręsti. Nors originalus autorius rašė šią pamoką anglų kalba, ji taip pat išversta į kitas kalbas, todėl jūs galite skaityti vertimą. Yra pavyzdžių, kuriuose naudojamos kelios skirtingos kalbos (siekiant palyginti skirtingų kalbų gramatikos taisykles). Šie pavyzdžiai *nėra* išversti, tačiau aiškinamasis tekstas yra, todėl prasmė turėtų būti aiški.

Programavimo užduotims atlikti naudosite Python, o pavyzdžiai pateikiami naudojant Python 3.8.

Šioje dalyje jums reikės ir naudosite:

- **Python 3 supratimą**. Programavimo kalbos Python 3 supratimą, ši pamoka naudoja įvestį, ciklus, failų skaitymą, masyvus.
- **Visual Studio Code + plėtinį**. Naudosime Visual Studio Code ir jo Python plėtinį. Taip pat galite naudoti bet kurį kitą Python IDE.
- **TextBlob**. [TextBlob](https://github.com/sloria/TextBlob) yra supaprastinta teksto apdorojimo biblioteka Python kalbai. Vadovaukitės TextBlob svetainėje pateiktomis instrukcijomis, kad ją įdiegtumėte savo sistemoje (taip pat įdiekite korpusus, kaip parodyta žemiau):

   ```bash
   pip install -U textblob
   python -m textblob.download_corpora
   ```

> 💡 Patarimas: Python galite paleisti tiesiogiai VS Code aplinkose. Daugiau informacijos rasite [dokumentacijoje](https://code.visualstudio.com/docs/languages/python?WT.mc_id=academic-77952-leestott).

## Kalbėjimas su mašinomis

Bandymų priversti kompiuterius suprasti žmonių kalbą istorija siekia dešimtmečius, o vienas iš pirmųjų mokslininkų, nagrinėjusių natūralios kalbos apdorojimą, buvo *Alanas Tiuringas*.

### „Tiuringo testas“

Kai Tiuringas 1950-aisiais tyrinėjo *dirbtinį intelektą*, jis svarstė, ar būtų galima atlikti pokalbio testą tarp žmogaus ir kompiuterio (rašytinės komunikacijos būdu), kur žmogus nebūtų tikras, ar bendrauja su kitu žmogumi, ar su kompiuteriu.

Jei po tam tikro pokalbio laiko žmogus negalėtų nustatyti, ar atsakymai buvo iš kompiuterio, ar ne, ar tada būtų galima sakyti, kad kompiuteris *mąsto*?

### Įkvėpimas – „imitacijos žaidimas“

Ši idėja kilo iš vakarėlių žaidimo, vadinamo *Imitacijos žaidimu*, kur tardytojas yra vienas kambaryje ir turi nustatyti, kurie iš dviejų žmonių (esančių kitame kambaryje) yra vyras ir moteris. Tardytojas gali siųsti užrašus ir turi sugalvoti klausimus, kurių atsakymai atskleistų paslaptingo asmens lytį. Žinoma, kiti žaidėjai stengiasi suklaidinti tardytoją, atsakydami taip, kad sukeltų painiavą, tačiau kartu atrodytų, kad atsako sąžiningai.

### Elizos kūrimas

1960-aisiais MIT mokslininkas *Joseph Weizenbaum* sukūrė [*Elizą*](https://wikipedia.org/wiki/ELIZA), kompiuterinį „terapeutą“, kuris užduodavo žmogui klausimus ir sudarydavo įspūdį, kad supranta jų atsakymus. Tačiau nors Eliza galėjo analizuoti sakinį ir atpažinti tam tikras gramatines struktūras bei raktinius žodžius, kad pateiktų tinkamą atsakymą, negalima sakyti, kad ji *suprato* sakinį. Jei Elizai buvo pateikiamas sakinys formatu „**Aš esu** <u>liūdnas</u>“, ji galėjo pertvarkyti ir pakeisti žodžius sakinyje, kad suformuotų atsakymą „Kaip ilgai **jūs buvote** <u>liūdnas</u>?“.

Tai sudarydavo įspūdį, kad Eliza suprato teiginį ir uždavė tęstinį klausimą, nors iš tikrųjų ji tik pakeitė laiką ir pridėjo keletą žodžių. Jei Eliza negalėjo atpažinti raktinio žodžio, kuriam turėjo atsakymą, ji pateikdavo atsitiktinį atsakymą, kuris galėjo tikti daugeliui skirtingų teiginių. Elizą buvo lengva apgauti, pavyzdžiui, jei vartotojas parašydavo „**Tu esi** <u>dviratis</u>“, ji galėjo atsakyti „Kaip ilgai **aš buvau** <u>dviratis</u>?“, užuot pateikusi labiau pagrįstą atsakymą.

[![Pokalbis su Eliza](https://img.youtube.com/vi/RMK9AphfLco/0.jpg)](https://youtu.be/RMK9AphfLco "Pokalbis su Eliza")

> 🎥 Spustelėkite aukščiau esantį vaizdą, kad pamatytumėte vaizdo įrašą apie originalią ELIZA programą

> Pastaba: Galite perskaityti originalų [Elizos aprašymą](https://cacm.acm.org/magazines/1966/1/13317-elizaa-computer-program-for-the-study-of-natural-language-communication-between-man-and-machine/abstract), paskelbtą 1966 m., jei turite ACM paskyrą. Arba skaitykite apie Elizą [Vikipedijoje](https://wikipedia.org/wiki/ELIZA).

## Užduotis – sukurti paprastą pokalbių botą

Pokalbių botas, kaip Eliza, yra programa, kuri renka vartotojo įvestį ir atrodo, kad supranta bei protingai atsako. Skirtingai nei Eliza, mūsų botas neturės kelių taisyklių, kurios sudarytų įspūdį apie intelektualų pokalbį. Vietoj to, mūsų botas turės tik vieną gebėjimą – tęsti pokalbį su atsitiktiniais atsakymais, kurie galėtų tikti beveik bet kokiam nereikšmingam pokalbiui.

### Planas

Jūsų žingsniai kuriant pokalbių botą:

1. Atspausdinkite instrukcijas, kaip vartotojas turėtų bendrauti su botu
2. Pradėkite ciklą
   1. Priimkite vartotojo įvestį
   2. Jei vartotojas paprašė išeiti, tada išeikite
   3. Apdorokite vartotojo įvestį ir nustatykite atsakymą (šiuo atveju atsakymas yra atsitiktinis pasirinkimas iš galimų bendrų atsakymų sąrašo)
   4. Atspausdinkite atsakymą
3. Grįžkite į 2 žingsnį

### Boto kūrimas

Sukurkime botą. Pradėsime apibrėždami keletą frazių.

1. Sukurkite šį botą Python kalba su šiais atsitiktiniais atsakymais:

    ```python
    random_responses = ["That is quite interesting, please tell me more.",
                        "I see. Do go on.",
                        "Why do you say that?",
                        "Funny weather we've been having, isn't it?",
                        "Let's change the subject.",
                        "Did you catch the game last night?"]
    ```

    Štai pavyzdinė išvestis, kuri padės jums (vartotojo įvestis pateikiama eilutėse, prasidedančiose `>`):

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

    Vienas galimas šios užduoties sprendimas pateiktas [čia](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/1-Introduction-to-NLP/solution/bot.py)

    ✅ Sustokite ir apsvarstykite

    1. Ar manote, kad atsitiktiniai atsakymai galėtų „apgauti“ ką nors, kad jie manytų, jog botas iš tikrųjų juos supranta?
    2. Kokios savybės būtų reikalingos, kad botas būtų efektyvesnis?
    3. Jei botas iš tikrųjų galėtų „suprasti“ sakinio prasmę, ar jam reikėtų „atsiminti“ ankstesnių sakinių prasmę pokalbyje?

---

## 🚀Iššūkis

Pasirinkite vieną iš aukščiau pateiktų „sustokite ir apsvarstykite“ elementų ir pabandykite jį įgyvendinti kode arba parašykite sprendimą popieriuje naudodami pseudokodą.

Kitoje pamokoje sužinosite apie įvairius kitus natūralios kalbos analizės ir mašininio mokymosi metodus.

## [Po paskaitos: testas](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/32/)

## Apžvalga ir savarankiškas mokymasis

Peržiūrėkite toliau pateiktas nuorodas kaip papildomas skaitymo galimybes.

### Nuorodos

1. Schubert, Lenhart, "Computational Linguistics", *The Stanford Encyclopedia of Philosophy* (Spring 2020 Edition), Edward N. Zalta (ed.), URL = <https://plato.stanford.edu/archives/spr2020/entries/computational-linguistics/>.
2. Princeton University "About WordNet." [WordNet](https://wordnet.princeton.edu/). Princeton University. 2010. 

## Užduotis 

[Suraskite botą](assignment.md)

---

**Atsakomybės apribojimas**:  
Šis dokumentas buvo išverstas naudojant AI vertimo paslaugą [Co-op Translator](https://github.com/Azure/co-op-translator). Nors siekiame tikslumo, prašome atkreipti dėmesį, kad automatiniai vertimai gali turėti klaidų ar netikslumų. Originalus dokumentas jo gimtąja kalba turėtų būti laikomas autoritetingu šaltiniu. Kritinei informacijai rekomenduojama profesionali žmogaus vertimo paslauga. Mes neprisiimame atsakomybės už nesusipratimus ar klaidingus interpretavimus, atsiradusius naudojant šį vertimą.