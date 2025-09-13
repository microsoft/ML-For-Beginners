<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1c2ec40cf55c98a028a359c27ef7e45a",
  "translation_date": "2025-09-05T08:06:34+00:00",
  "source_file": "6-NLP/1-Introduction-to-NLP/README.md",
  "language_code": "lt"
}
-->
# Įvadas į natūralios kalbos apdorojimą

Ši pamoka apima trumpą istoriją ir svarbias *natūralios kalbos apdorojimo* (NLP), kuris yra *kompiuterinės lingvistikos* posritis, sąvokas.

## [Prieš paskaitos testas](https://ff-quizzes.netlify.app/en/ml/)

## Įvadas

NLP, kaip dažnai vadinama, yra viena geriausiai žinomų sričių, kuriose mašininis mokymasis buvo pritaikytas ir naudojamas gamybos programinėje įrangoje.

✅ Ar galite pagalvoti apie programinę įrangą, kurią naudojate kasdien ir kurioje greičiausiai yra integruotas NLP? O kaip dėl jūsų teksto redagavimo programų ar mobiliųjų programėlių, kurias naudojate reguliariai?

Jūs sužinosite apie:

- **Kalbų idėją**. Kaip kalbos vystėsi ir kokios buvo pagrindinės tyrimų sritys.
- **Apibrėžimus ir sąvokas**. Taip pat sužinosite apibrėžimus ir sąvokas apie tai, kaip kompiuteriai apdoroja tekstą, įskaitant sintaksės analizę, gramatiką ir daiktavardžių bei veiksmažodžių identifikavimą. Šioje pamokoje yra keletas kodavimo užduočių, ir pristatomos kelios svarbios sąvokos, kurias vėliau išmoksite programuoti kitose pamokose.

## Kompiuterinė lingvistika

Kompiuterinė lingvistika yra tyrimų ir plėtros sritis, kuri per daugelį dešimtmečių nagrinėjo, kaip kompiuteriai gali dirbti su kalbomis, jas suprasti, versti ir net komunikuoti. Natūralios kalbos apdorojimas (NLP) yra susijusi sritis, orientuota į tai, kaip kompiuteriai gali apdoroti „natūralias“, arba žmonių, kalbas.

### Pavyzdys - diktavimas telefonu

Jei kada nors diktavote savo telefonui vietoj rašymo arba uždavėte klausimą virtualiam asistentui, jūsų kalba buvo konvertuota į tekstinę formą ir tada apdorota arba *analizuota* iš kalbos, kuria kalbėjote. Aptikti raktiniai žodžiai buvo apdoroti į formatą, kurį telefonas ar asistentas galėjo suprasti ir pagal kurį veikti.

![supratimas](../../../../6-NLP/1-Introduction-to-NLP/images/comprehension.png)
> Tikras lingvistinis supratimas yra sudėtingas! Vaizdas sukurtas [Jen Looper](https://twitter.com/jenlooper)

### Kaip ši technologija tapo įmanoma?

Tai tapo įmanoma, nes kažkas parašė kompiuterinę programą, kad tai atliktų. Prieš kelis dešimtmečius kai kurie mokslinės fantastikos rašytojai prognozavo, kad žmonės daugiausia kalbės su savo kompiuteriais, o kompiuteriai visada tiksliai supras, ką jie turi omenyje. Deja, paaiškėjo, kad tai yra sudėtingesnė problema, nei daugelis įsivaizdavo, ir nors ši problema šiandien yra daug geriau suprantama, vis dar yra didelių iššūkių pasiekti „tobulą“ natūralios kalbos apdorojimą, kai kalbama apie sakinio prasmės supratimą. Tai ypač sudėtinga, kai reikia suprasti humorą ar aptikti emocijas, tokias kaip sarkazmas, sakinyje.

Šiuo metu galbūt prisimenate mokyklos pamokas, kuriose mokytojas aptarė sakinio gramatikos dalis. Kai kuriose šalyse mokiniai mokomi gramatikos ir lingvistikos kaip atskiro dalyko, tačiau daugelyje šių temų mokoma kaip kalbos mokymosi dalis: arba pirmosios kalbos pradinėje mokykloje (mokymasis skaityti ir rašyti), o galbūt antrosios kalbos vidurinėje mokykloje. Nesijaudinkite, jei nesate ekspertas, skiriantis daiktavardžius nuo veiksmažodžių ar prieveiksmius nuo būdvardžių!

Jei jums sunku atskirti *paprastąjį esamąjį laiką* nuo *esamojo progresyvaus*, jūs nesate vieni. Tai sudėtingas dalykas daugeliui žmonių, net gimtakalbiams. Geros naujienos yra tai, kad kompiuteriai yra labai geri taikant formaliąsias taisykles, ir jūs išmoksite rašyti kodą, kuris gali *analizuoti* sakinį taip pat gerai, kaip žmogus. Didžiausias iššūkis, kurį vėliau nagrinėsite, yra suprasti sakinio *prasmę* ir *nuotaiką*.

## Reikalavimai

Šiai pamokai pagrindinis reikalavimas yra gebėjimas skaityti ir suprasti šios pamokos kalbą. Nėra matematikos užduočių ar lygčių, kurias reikėtų spręsti. Nors originalus autorius parašė šią pamoką anglų kalba, ji taip pat išversta į kitas kalbas, todėl jūs galite skaityti vertimą. Yra pavyzdžių, kuriuose naudojamos kelios skirtingos kalbos (norint palyginti skirtingų kalbų gramatikos taisykles). Šios kalbos *nėra* išverstos, tačiau aiškinamasis tekstas yra, todėl prasmė turėtų būti aiški.

Kodavimo užduotims atlikti naudosite Python, o pavyzdžiai pateikti naudojant Python 3.8.

Šioje dalyje jums reikės ir naudosite:

- **Python 3 supratimas**. Programavimo kalbos supratimas Python 3, ši pamoka naudoja įvestį, ciklus, failų skaitymą, masyvus.
- **Visual Studio Code + plėtinys**. Naudosime Visual Studio Code ir jo Python plėtinį. Taip pat galite naudoti bet kurį Python IDE pagal savo pasirinkimą.
- **TextBlob**. [TextBlob](https://github.com/sloria/TextBlob) yra supaprastinta teksto apdorojimo biblioteka Python. Sekite instrukcijas TextBlob svetainėje, kad ją įdiegtumėte savo sistemoje (taip pat įdiekite korpusus, kaip parodyta žemiau):

   ```bash
   pip install -U textblob
   python -m textblob.download_corpora
   ```

> 💡 Patarimas: Python galite paleisti tiesiai VS Code aplinkose. Peržiūrėkite [dokumentaciją](https://code.visualstudio.com/docs/languages/python?WT.mc_id=academic-77952-leestott) norėdami gauti daugiau informacijos.

## Kalbėjimas su mašinomis

Istorija apie bandymus priversti kompiuterius suprasti žmonių kalbą siekia dešimtmečius, o vienas iš ankstyviausių mokslininkų, svarstęs natūralios kalbos apdorojimą, buvo *Alanas Turingas*.

### „Turingo testas“

Kai Turingas 1950-aisiais tyrinėjo *dirbtinį intelektą*, jis svarstė, ar pokalbio testas galėtų būti pateiktas žmogui ir kompiuteriui (per rašytinę korespondenciją), kur žmogus pokalbyje nebūtų tikras, ar bendrauja su kitu žmogumi, ar su kompiuteriu.

Jei po tam tikro pokalbio laiko žmogus negalėtų nustatyti, ar atsakymai yra iš kompiuterio, ar ne, ar tada kompiuteris galėtų būti laikomas *mąstančiu*?

### Įkvėpimas - „imitacijos žaidimas“

Ši idėja kilo iš vakarėlio žaidimo, vadinamo *Imitacijos žaidimu*, kur tardytojas yra vienas kambaryje ir turi nustatyti, kurie iš dviejų žmonių (kitame kambaryje) yra vyras ir moteris. Tardytojas gali siųsti užrašus ir turi stengtis sugalvoti klausimus, kurių rašytiniai atsakymai atskleistų paslaptingo žmogaus lytį. Žinoma, žaidėjai kitame kambaryje stengiasi suklaidinti tardytoją, atsakydami į klausimus taip, kad suklaidintų ar supainiotų tardytoją, tuo pačiu suteikdami įspūdį, kad atsako sąžiningai.

### Eliza kūrimas

1960-aisiais MIT mokslininkas *Joseph Weizenbaum* sukūrė [*Eliza*](https://wikipedia.org/wiki/ELIZA), kompiuterinį „terapeutą“, kuris užduodavo žmogui klausimus ir sudarydavo įspūdį, kad supranta jų atsakymus. Tačiau, nors Eliza galėjo analizuoti sakinį ir identifikuoti tam tikras gramatines struktūras bei raktinius žodžius, kad pateiktų tinkamą atsakymą, negalima sakyti, kad ji *suprato* sakinį. Jei Eliza buvo pateiktas sakinys, atitinkantis formatą „**Aš esu** <u>liūdnas</u>“, ji galėjo pertvarkyti ir pakeisti žodžius sakinyje, kad suformuotų atsakymą „Kaip ilgai **jūs buvote** <u>liūdnas</u>?“.

Tai sudarė įspūdį, kad Eliza suprato teiginį ir uždavė tęstinį klausimą, nors iš tikrųjų ji keitė laiką ir pridėjo keletą žodžių. Jei Eliza negalėjo identifikuoti raktinio žodžio, kuriam turėjo atsakymą, ji vietoj to pateikdavo atsitiktinį atsakymą, kuris galėjo būti taikomas daugeliui skirtingų teiginių. Eliza galėjo būti lengvai apgauta, pavyzdžiui, jei vartotojas parašė „**Tu esi** <u>dviratis</u>“, ji galėjo atsakyti „Kaip ilgai **aš buvau** <u>dviratis</u>?“, užuot pateikusi labiau pagrįstą atsakymą.

[![Pokalbis su Eliza](https://img.youtube.com/vi/RMK9AphfLco/0.jpg)](https://youtu.be/RMK9AphfLco "Pokalbis su Eliza")

> 🎥 Spustelėkite aukščiau esančią nuotrauką, kad pamatytumėte vaizdo įrašą apie originalią ELIZA programą

> Pastaba: Galite perskaityti originalų [Eliza](https://cacm.acm.org/magazines/1966/1/13317-elizaa-computer-program-for-the-study-of-natural-language-communication-between-man-and-machine/abstract) aprašymą, paskelbtą 1966 m., jei turite ACM paskyrą. Arba skaitykite apie Eliza [wikipedia](https://wikipedia.org/wiki/ELIZA)

## Užduotis - sukurti paprastą pokalbių botą

Pokalbių botas, kaip Eliza, yra programa, kuri gauna vartotojo įvestį ir atrodo, kad supranta bei atsako protingai. Skirtingai nei Eliza, mūsų botas neturės kelių taisyklių, kurios suteiktų jam išvaizdą, kad jis turi protingą pokalbį. Vietoj to, mūsų botas turės tik vieną gebėjimą - tęsti pokalbį su atsitiktiniais atsakymais, kurie galėtų veikti beveik bet kokiame trivialiame pokalbyje.

### Planas

Jūsų žingsniai kuriant pokalbių botą:

1. Atspausdinkite instrukcijas, patariančias vartotojui, kaip bendrauti su botu
2. Pradėkite ciklą
   1. Priimkite vartotojo įvestį
   2. Jei vartotojas paprašė išeiti, tada išeikite
   3. Apdorokite vartotojo įvestį ir nustatykite atsakymą (šiuo atveju atsakymas yra atsitiktinis pasirinkimas iš galimų bendrų atsakymų sąrašo)
   4. Atspausdinkite atsakymą
3. Grįžkite į 2 žingsnį

### Boto kūrimas

Sukurkime botą. Pradėsime apibrėždami keletą frazių.

1. Sukurkite šį botą patys Python kalba su šiais atsitiktiniais atsakymais:

    ```python
    random_responses = ["That is quite interesting, please tell me more.",
                        "I see. Do go on.",
                        "Why do you say that?",
                        "Funny weather we've been having, isn't it?",
                        "Let's change the subject.",
                        "Did you catch the game last night?"]
    ```

    Štai keletas pavyzdinių rezultatų, kurie padės jums (vartotojo įvestis yra eilutėse, prasidedančiose `>`):

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

    Vienas galimas užduoties sprendimas yra [čia](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/1-Introduction-to-NLP/solution/bot.py)

    ✅ Sustokite ir apsvarstykite

    1. Ar manote, kad atsitiktiniai atsakymai galėtų „apgauti“ ką nors, kad jie manytų, jog botas iš tikrųjų juos supranta?
    2. Kokios funkcijos botui būtų reikalingos, kad jis būtų efektyvesnis?
    3. Jei botas iš tikrųjų galėtų „suprasti“ sakinio prasmę, ar jam reikėtų „prisiminti“ ankstesnių sakinių prasmę pokalbyje?

---

## 🚀Iššūkis

Pasirinkite vieną iš aukščiau pateiktų „sustokite ir apsvarstykite“ elementų ir pabandykite jį įgyvendinti kode arba parašykite sprendimą popieriuje naudodami pseudokodą.

Kitoje pamokoje sužinosite apie daugybę kitų natūralios kalbos analizės ir mašininio mokymosi metodų.

## [Po paskaitos testas](https://ff-quizzes.netlify.app/en/ml/)

## Apžvalga ir savarankiškas mokymasis

Peržiūrėkite toliau pateiktas nuorodas kaip papildomas skaitymo galimybes.

### Nuorodos

1. Schubert, Lenhart, "Kompiuterinė lingvistika", *The Stanford Encyclopedia of Philosophy* (2020 m. pavasario leidimas), Edward N. Zalta (red.), URL = <https://plato.stanford.edu/archives/spr2020/entries/computational-linguistics/>.
2. Prinstono universitetas "Apie WordNet." [WordNet](https://wordnet.princeton.edu/). Prinstono universitetas. 2010. 

## Užduotis 

[Suraskite botą](assignment.md)

---

**Atsakomybės apribojimas**:  
Šis dokumentas buvo išverstas naudojant AI vertimo paslaugą [Co-op Translator](https://github.com/Azure/co-op-translator). Nors siekiame tikslumo, prašome atkreipti dėmesį, kad automatiniai vertimai gali turėti klaidų ar netikslumų. Originalus dokumentas jo gimtąja kalba turėtų būti laikomas autoritetingu šaltiniu. Kritinei informacijai rekomenduojama naudoti profesionalų žmogaus vertimą. Mes neprisiimame atsakomybės už nesusipratimus ar klaidingus interpretavimus, atsiradusius dėl šio vertimo naudojimo.