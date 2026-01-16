<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1c2ec40cf55c98a028a359c27ef7e45a",
  "translation_date": "2025-10-11T11:31:07+00:00",
  "source_file": "6-NLP/1-Introduction-to-NLP/README.md",
  "language_code": "et"
}
-->
# Sissejuhatus loomuliku keele töötlemisse

See õppetund hõlmab lühikest ajalugu ja olulisi mõisteid *loomuliku keele töötlemisest*, mis on *arvutilingvistika* alavaldkond.

## [Eelloengu viktoriin](https://ff-quizzes.netlify.app/en/ml/)

## Sissejuhatus

NLP (loomuliku keele töötlemine), nagu seda tavaliselt nimetatakse, on üks tuntumaid valdkondi, kus masinõpet on rakendatud ja kasutatud tootmistarkvaras.

✅ Kas oskad nimetada tarkvara, mida sa iga päev kasutad ja milles tõenäoliselt on integreeritud NLP? Mis saab sinu tekstitöötlusprogrammidest või mobiilirakendustest, mida sa regulaarselt kasutad?

Sa õpid:

- **Keelte idee**. Kuidas keeled arenesid ja millised on olnud peamised uurimisvaldkonnad.
- **Mõisted ja definitsioonid**. Sa õpid ka definitsioone ja mõisteid selle kohta, kuidas arvutid teksti töötlevad, sealhulgas lauseparsimist, grammatikat ning nimisõnade ja tegusõnade tuvastamist. Selles õppetunnis on mõned kodeerimisülesanded ja tutvustatakse mitmeid olulisi mõisteid, mida sa õpid hiljem kodeerima järgmistes õppetundides.

## Arvutilingvistika

Arvutilingvistika on aastakümnete pikkune uurimis- ja arendusvaldkond, mis uurib, kuidas arvutid saavad töötada keeltega, neid mõista, tõlkida ja nendega suhelda. Loomuliku keele töötlemine (NLP) on seotud valdkond, mis keskendub sellele, kuidas arvutid saavad töödelda 'loomulikke', ehk inimkeeli.

### Näide - telefoni dikteerimine

Kui oled kunagi dikteerinud oma telefonile teksti asemel või küsinud virtuaalselt assistendilt küsimuse, siis sinu kõne on muudetud tekstivormiks ja seejärel töödeldud või *parsitud* keeles, mida sa rääkisid. Tuvastatud märksõnad töödeldi seejärel formaadiks, mida telefon või assistent suudaks mõista ja millele reageerida.

![mõistmine](../../../../translated_images/et/comprehension.619708fc5959b0f6.png)
> Tõeline keeleline mõistmine on keeruline! Pilt autorilt [Jen Looper](https://twitter.com/jenlooper)

### Kuidas on see tehnoloogia võimalik?

See on võimalik, kuna keegi kirjutas selleks arvutiprogrammi. Mõned aastakümned tagasi ennustasid ulmekirjanikud, et inimesed räägivad peamiselt oma arvutitega ja arvutid mõistavad alati täpselt, mida nad mõtlevad. Kahjuks osutus see probleem raskemaks, kui paljud ette kujutasid, ja kuigi see on tänapäeval palju paremini mõistetav probleem, on 'täiusliku' loomuliku keele töötlemise saavutamisel märkimisväärseid väljakutseid, eriti lause tähenduse mõistmisel. See on eriti keeruline probleem, kui tegemist on huumori mõistmise või emotsioonide, nagu sarkasmi, tuvastamisega lauses.

Praegu võid meenutada koolitunde, kus õpetaja käsitles lause grammatilisi osi. Mõnes riigis õpetatakse grammatikat ja lingvistikat eraldi õppeainena, kuid paljudes riikides on need teemad osa keele õppimisest: kas esimest keelt algkoolis (lugemise ja kirjutamise õppimine) ja võib-olla teist keelt põhikoolis või keskkoolis. Ära muretse, kui sa ei ole ekspert nimisõnade ja tegusõnade või määrsõnade ja omadussõnade eristamisel!

Kui sul on raskusi *lihtoleviku* ja *kestev oleviku* erinevuse mõistmisega, siis sa ei ole üksi. See on keeruline asi paljudele inimestele, isegi keele emakeelena rääkijatele. Hea uudis on see, et arvutid on väga head formaalsete reeglite rakendamisel, ja sa õpid kirjutama koodi, mis suudab lauset *parsida* sama hästi kui inimene. Suurem väljakutse, mida sa hiljem uurid, on lause *tähenduse* ja *meeleolu* mõistmine.

## Eeldused

Selle õppetunni peamine eeldus on oskus lugeda ja mõista selle õppetunni keelt. Siin ei ole matemaatilisi probleeme ega võrrandeid lahendamiseks. Kuigi algne autor kirjutas selle õppetunni inglise keeles, on see tõlgitud ka teistesse keeltesse, nii et sa võid lugeda tõlget. On näiteid, kus kasutatakse mitmeid erinevaid keeli (et võrrelda erinevate keelte grammatikareegleid). Need ei ole *tõlgitud*, kuid selgitav tekst on, nii et tähendus peaks olema selge.

Kodeerimisülesannete jaoks kasutad Pythonit ja näited on kirjutatud Python 3.8 keeles.

Selles osas vajad ja kasutad:

- **Python 3 mõistmine**. Programmeerimiskeele mõistmine Python 3-s, see õppetund kasutab sisendit, tsükleid, failide lugemist, massiive.
- **Visual Studio Code + laiendus**. Kasutame Visual Studio Code'i ja selle Python laiendust. Võid kasutada ka endale sobivat Python IDE-d.
- **TextBlob**. [TextBlob](https://github.com/sloria/TextBlob) on lihtsustatud tekstitöötluse teek Pythonile. Järgi TextBlobi veebisaidil toodud juhiseid, et see oma süsteemi installida (installi ka korpused, nagu allpool näidatud):

   ```bash
   pip install -U textblob
   python -m textblob.download_corpora
   ```

> 💡 Näpunäide: Pythonit saab otse käivitada VS Code keskkondades. Vaata [dokumentatsiooni](https://code.visualstudio.com/docs/languages/python?WT.mc_id=academic-77952-leestott) lisainfo saamiseks.

## Masinatega rääkimine

Inimkeele mõistmise õpetamise ajalugu arvutitele ulatub aastakümnete taha ja üks esimesi teadlasi, kes loomuliku keele töötlemist uuris, oli *Alan Turing*.

### 'Turingi test'

Kui Turing uuris *tehisintellekti* 1950ndatel, mõtles ta, kas vestlustesti võiks anda inimesele ja arvutile (kirjaliku suhtluse kaudu), kus vestluses osalev inimene ei ole kindel, kas ta suhtleb teise inimese või arvutiga.

Kui pärast teatud pikkusega vestlust ei suuda inimene kindlaks teha, kas vastused pärinevad arvutist või mitte, siis kas arvutit võiks pidada *mõtlevaks*?

### Inspiratsioon - 'imiteerimismäng'

Selle idee sai ta peomängust nimega *Imiteerimismäng*, kus küsitleja on üksi ruumis ja peab kindlaks tegema, kes kahest inimesest (teises ruumis) on mees ja kes naine. Küsitleja saab saata märkmeid ja peab mõtlema küsimusi, mille kirjalikud vastused paljastavad salapärase inimese soo. Muidugi püüavad teises ruumis olevad mängijad küsitlejat eksitada, vastates küsimustele viisil, mis segab või eksitab küsitlejat, samal ajal andes vastuste näol ausa mulje.

### Eliza loomine

1960ndatel töötas MIT teadlane *Joseph Weizenbaum* välja [*Eliza*](https://wikipedia.org/wiki/ELIZA), arvutiterapeudi, kes küsis inimeselt küsimusi ja jättis mulje, et mõistab nende vastuseid. Kuid kuigi Eliza suutis lauset parsida ja tuvastada teatud grammatilisi konstruktsioone ja märksõnu, et anda mõistlik vastus, ei saanud öelda, et ta *mõistab* lauset. Kui Elizale esitati lause formaadis "**Ma olen** <u>kurb</u>", võis ta lause sõnu ümber korraldada ja asendada, et moodustada vastus "Kui kaua olete **olnud** <u>kurb</u>".

See jättis mulje, et Eliza mõistis väidet ja esitas järelküsimuse, kuigi tegelikult muutis ta aega ja lisas mõned sõnad. Kui Eliza ei suutnud tuvastada märksõna, millele tal oli vastus, andis ta selle asemel juhusliku vastuse, mis võis sobida paljudele erinevatele väidetele. Elizat oli lihtne petta, näiteks kui kasutaja kirjutas "**Sa oled** <u>jalgratas</u>", võis ta vastata "Kui kaua olen **olnud** <u>jalgratas</u>?", selle asemel et anda mõistlikum vastus.

[![Vestlus Elizaga](https://img.youtube.com/vi/RMK9AphfLco/0.jpg)](https://youtu.be/RMK9AphfLco "Vestlus Elizaga")

> 🎥 Klõpsa ülaloleval pildil, et vaadata videot originaalsest ELIZA programmist

> Märkus: Sa saad lugeda Eliza algset kirjeldust, mis avaldati [1966. aastal](https://cacm.acm.org/magazines/1966/1/13317-elizaa-computer-program-for-the-study-of-natural-language-communication-between-man-and-machine/abstract), kui sul on ACM konto. Alternatiivselt loe Elizast [wikipedia](https://wikipedia.org/wiki/ELIZA) lehel.

## Harjutus - lihtsa vestlusbot'i kodeerimine

Vestlusbot, nagu Eliza, on programm, mis kutsub kasutajat sisestama ja jätab mulje, et mõistab ja vastab intelligentselt. Erinevalt Elizast ei ole meie bot'il mitmeid reegleid, mis annaksid talle intelligentse vestluse mulje. Selle asemel on meie bot'il ainult üks võime: vestlust jätkata juhuslike vastustega, mis võivad sobida peaaegu igas triviaalvestluses.

### Plaan

Sinu sammud vestlusbot'i loomisel:

1. Prindi juhised, mis annavad kasutajale teada, kuidas bot'iga suhelda
2. Alusta tsüklit
   1. Võta kasutaja sisend
   2. Kui kasutaja on palunud väljuda, siis lõpeta
   3. Töötle kasutaja sisend ja määra vastus (antud juhul on vastus juhuslik valik võimalike üldiste vastuste loendist)
   4. Prindi vastus
3. Mine tagasi 2. sammu juurde

### Bot'i loomine

Loome bot'i järgmisena. Alustame fraaside määratlemisest.

1. Loo see bot ise Pythonis, kasutades järgmisi juhuslikke vastuseid:

    ```python
    random_responses = ["That is quite interesting, please tell me more.",
                        "I see. Do go on.",
                        "Why do you say that?",
                        "Funny weather we've been having, isn't it?",
                        "Let's change the subject.",
                        "Did you catch the game last night?"]
    ```

    Siin on näidisväljund, mis juhendab sind (kasutaja sisend on ridadel, mis algavad `>`):

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

    Üks võimalik lahendus ülesandele on [siin](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/1-Introduction-to-NLP/solution/bot.py)

    ✅ Peatu ja mõtle

    1. Kas arvad, et juhuslikud vastused suudaksid kedagi 'petta', et bot tegelikult mõistab neid?
    2. Milliseid funktsioone vajaks bot, et olla tõhusam?
    3. Kui bot suudaks tõesti 'mõista' lause tähendust, kas ta peaks ka 'mäletama' eelmiste lausete tähendust vestluses?

---

## 🚀Väljakutse

Vali üks ülaltoodud "peatu ja mõtle" elementidest ja proovi kas seda koodis rakendada või kirjuta lahendus paberil, kasutades pseudokoodi.

Järgmises õppetunnis õpid mitmeid teisi lähenemisviise loomuliku keele parsimisele ja masinõppele.

## [Järelloengu viktoriin](https://ff-quizzes.netlify.app/en/ml/)

## Ülevaade ja iseseisev õppimine

Vaata allpool toodud viiteid, et saada lisalugemiseks võimalusi.

### Viited

1. Schubert, Lenhart, "Computational Linguistics", *The Stanford Encyclopedia of Philosophy* (Kevad 2020 väljaanne), Edward N. Zalta (toim.), URL = <https://plato.stanford.edu/archives/spr2020/entries/computational-linguistics/>.
2. Princeton University "About WordNet." [WordNet](https://wordnet.princeton.edu/). Princeton University. 2010. 

## Ülesanne 

[Otsi bot'i](assignment.md)

---

**Lahtiütlus**:  
See dokument on tõlgitud AI tõlketeenuse [Co-op Translator](https://github.com/Azure/co-op-translator) abil. Kuigi püüame tagada täpsust, palume arvestada, et automaatsed tõlked võivad sisaldada vigu või ebatäpsusi. Algne dokument selle algses keeles tuleks pidada autoriteetseks allikaks. Olulise teabe puhul soovitame kasutada professionaalset inimtõlget. Me ei vastuta selle tõlke kasutamisest tulenevate arusaamatuste või valesti tõlgenduste eest.