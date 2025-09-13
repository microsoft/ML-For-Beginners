<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1c2ec40cf55c98a028a359c27ef7e45a",
  "translation_date": "2025-09-05T14:10:08+00:00",
  "source_file": "6-NLP/1-Introduction-to-NLP/README.md",
  "language_code": "sl"
}
-->
# Uvod v obdelavo naravnega jezika

Ta lekcija zajema kratko zgodovino in pomembne koncepte *obdelave naravnega jezika* (ONJ), podpodročja *računalniške lingvistike*.

## [Predhodni kviz](https://ff-quizzes.netlify.app/en/ml/)

## Uvod

ONJ, kot jo pogosto imenujemo, je eno najbolj znanih področij, kjer se strojno učenje uporablja in vključuje v produkcijsko programsko opremo.

✅ Ali lahko pomislite na programsko opremo, ki jo uporabljate vsak dan in ki verjetno vključuje ONJ? Kaj pa vaši programi za urejanje besedil ali mobilne aplikacije, ki jih redno uporabljate?

Naučili se boste:

- **Ideje o jezikih**. Kako so se jeziki razvijali in katera so bila glavna področja raziskovanja.
- **Definicije in koncepti**. Spoznali boste definicije in koncepte, kako računalniki obdelujejo besedilo, vključno z razčlenjevanjem, slovnico ter prepoznavanjem samostalnikov in glagolov. V tej lekciji so vključene nekatere naloge s kodiranjem, prav tako pa so predstavljeni pomembni koncepti, ki se jih boste naučili kodirati v naslednjih lekcijah.

## Računalniška lingvistika

Računalniška lingvistika je področje raziskav in razvoja, ki že desetletja preučuje, kako lahko računalniki delujejo z jeziki, jih razumejo, prevajajo in komunicirajo z njimi. Obdelava naravnega jezika (ONJ) je sorodno področje, osredotočeno na to, kako računalniki obdelujejo 'naravne' ali človeške jezike.

### Primer - narekovanje na telefonu

Če ste kdaj narekovali svojemu telefonu namesto tipkanja ali vprašali virtualnega asistenta, je vaš govor pretvorjen v besedilo in nato obdelan oziroma *razčlenjen* iz jezika, ki ste ga govorili. Zaznane ključne besede so nato obdelane v obliki, ki jo telefon ali asistent razume in na katero lahko ukrepa.

![razumevanje](../../../../6-NLP/1-Introduction-to-NLP/images/comprehension.png)  
> Resnično razumevanje jezika je težko! Slika: [Jen Looper](https://twitter.com/jenlooper)

### Kako je ta tehnologija mogoča?

To je mogoče, ker je nekdo napisal računalniški program, ki to omogoča. Pred nekaj desetletji so nekateri pisci znanstvene fantastike napovedovali, da bodo ljudje večinoma govorili s svojimi računalniki, ki bodo vedno natančno razumeli, kaj mislijo. Žal se je izkazalo, da je ta problem težji, kot so mnogi mislili. Čeprav je danes veliko bolje razumljen, še vedno obstajajo pomembni izzivi pri doseganju 'popolne' obdelave naravnega jezika, zlasti ko gre za razumevanje pomena stavka. To je še posebej težavno pri razumevanju humorja ali zaznavanju čustev, kot je sarkazem, v stavku.

Morda se zdaj spomnite šolskih ur, kjer je učitelj razlagal dele stavčne slovnice. V nekaterih državah se učenci učijo slovnice in lingvistike kot samostojnega predmeta, v mnogih pa so te teme vključene v učenje jezika: bodisi vašega prvega jezika v osnovni šoli (učenje branja in pisanja) bodisi morda drugega jezika v srednji šoli. Ne skrbite, če niste strokovnjak za razlikovanje med samostalniki in glagoli ali prislovi in pridevniki!

Če se vam zdi težko razlikovati med *enostavnim sedanjikom* in *sedanjim trpnikom*, niste edini. To je izziv za mnoge ljudi, tudi za naravne govorce jezika. Dobra novica je, da so računalniki zelo dobri pri uporabi formalnih pravil, in naučili se boste pisati kodo, ki lahko *razčleni* stavek tako dobro kot človek. Večji izziv, ki ga boste raziskali kasneje, je razumevanje *pomena* in *čustev* stavka.

## Predznanje

Za to lekcijo je glavni pogoj sposobnost branja in razumevanja jezika te lekcije. Ni matematičnih nalog ali enačb za reševanje. Čeprav je avtor to lekcijo napisal v angleščini, je prevedena tudi v druge jezike, zato morda berete prevod. Obstajajo primeri, kjer se uporablja več različnih jezikov (za primerjavo različnih slovničnih pravil jezikov). Ti *niso* prevedeni, vendar je razlagalno besedilo prevedeno, zato bi moral biti pomen jasen.

Za naloge s kodiranjem boste uporabljali Python, primeri pa so napisani v Pythonu 3.8.

V tem razdelku boste potrebovali in uporabljali:

- **Razumevanje Pythona 3**. Razumevanje programskega jezika Python 3, ta lekcija uporablja vnos, zanke, branje datotek, tabele.
- **Visual Studio Code + razširitev**. Uporabili bomo Visual Studio Code in njegovo razširitev za Python. Lahko pa uporabite tudi IDE za Python po svoji izbiri.
- **TextBlob**. [TextBlob](https://github.com/sloria/TextBlob) je poenostavljena knjižnica za obdelavo besedila v Pythonu. Sledite navodilom na strani TextBlob za namestitev na vaš sistem (namestite tudi korpuse, kot je prikazano spodaj):

   ```bash
   pip install -U textblob
   python -m textblob.download_corpora
   ```

> 💡 Nasvet: Python lahko zaženete neposredno v okolju VS Code. Preverite [dokumentacijo](https://code.visualstudio.com/docs/languages/python?WT.mc_id=academic-77952-leestott) za več informacij.

## Pogovor z računalniki

Zgodovina poskusov, da bi računalniki razumeli človeški jezik, sega desetletja nazaj. Eden prvih znanstvenikov, ki je razmišljal o obdelavi naravnega jezika, je bil *Alan Turing*.

### 'Turingov test'

Ko je Turing v 50. letih raziskoval *umetno inteligenco*, je razmišljal, ali bi lahko izvedli pogovorni test, kjer bi človek in računalnik (prek tipkanja) komunicirala, pri čemer človek ne bi bil prepričan, ali se pogovarja z drugim človekom ali računalnikom.

Če po določenem času pogovora človek ne bi mogel ugotoviti, ali odgovori prihajajo od računalnika ali ne, bi lahko rekli, da računalnik *razmišlja*?

### Navdih - 'igra posnemanja'

Ideja za to je prišla iz družabne igre *Igra posnemanja*, kjer je zasliševalec sam v sobi in mora ugotoviti, kdo od dveh oseb (v drugi sobi) je moški in kdo ženska. Zasliševalec lahko pošilja zapiske in mora poskušati oblikovati vprašanja, kjer pisni odgovori razkrivajo spol skrivnostne osebe. Seveda se igralci v drugi sobi trudijo zavajati zasliševalca z odgovori, ki ga zmedejo, hkrati pa dajejo vtis, da odgovarjajo iskreno.

### Razvoj Elize

V 60. letih je znanstvenik z MIT, *Joseph Weizenbaum*, razvil [*Elizo*](https://wikipedia.org/wiki/ELIZA), računalniškega 'terapevta', ki je postavljal vprašanja in dajal vtis, da razume odgovore. Vendar pa, čeprav je Eliza lahko razčlenila stavek in prepoznala določene slovnične strukture in ključne besede, da bi dala razumen odgovor, ni mogla *razumeti* stavka. Če je bila Elizi predstavljena poved v obliki "**Jaz sem** <u>žalosten</u>", bi morda preuredila in zamenjala besede v stavku, da bi oblikovala odgovor "Kako dolgo ste **vi** <u>žalostni</u>?".

To je dalo vtis, da Eliza razume izjavo in postavlja nadaljnje vprašanje, medtem ko je v resnici le spreminjala čas in dodajala nekaj besed. Če Eliza ni mogla prepoznati ključne besede, za katero je imela odgovor, bi namesto tega podala naključen odgovor, ki bi lahko ustrezal številnim različnim izjavam. Elizo je bilo mogoče zlahka pretentati, na primer, če je uporabnik napisal "**Ti si** <u>kolo</u>", bi morda odgovorila "Kako dolgo sem **jaz** <u>kolo</u>?", namesto bolj smiselnega odgovora.

[![Pogovor z Elizo](https://img.youtube.com/vi/RMK9AphfLco/0.jpg)](https://youtu.be/RMK9AphfLco "Pogovor z Elizo")

> 🎥 Kliknite zgornjo sliko za video o izvirnem programu ELIZA

> Opomba: Izvirni opis [Elize](https://cacm.acm.org/magazines/1966/1/13317-elizaa-computer-program-for-the-study-of-natural-language-communication-between-man-and-machine/abstract), objavljen leta 1966, lahko preberete, če imate račun ACM. Alternativno lahko o Elizi preberete na [Wikipediji](https://wikipedia.org/wiki/ELIZA).

## Vaja - kodiranje osnovnega pogovornega bota

Pogovorni bot, kot je Eliza, je program, ki pridobiva uporabniški vnos in daje vtis, da razume in inteligentno odgovarja. Za razliko od Elize naš bot ne bo imel več pravil, ki bi dajala vtis inteligentnega pogovora. Namesto tega bo imel bot le eno sposobnost: nadaljevati pogovor z naključnimi odgovori, ki bi lahko delovali v skoraj vsakem trivialnem pogovoru.

### Načrt

Koraki pri gradnji pogovornega bota:

1. Natisnite navodila, kako naj uporabnik komunicira z botom.
2. Zaženite zanko:
   1. Sprejmite uporabniški vnos.
   2. Če uporabnik zahteva izhod, izstopite.
   3. Obdelajte uporabniški vnos in določite odgovor (v tem primeru je odgovor naključna izbira iz seznama možnih splošnih odgovorov).
   4. Natisnite odgovor.
3. Vrni se na korak 2.

### Gradnja bota

Zdaj bomo ustvarili bota. Začeli bomo z definiranjem nekaterih fraz.

1. Ustvarite tega bota sami v Pythonu z naslednjimi naključnimi odgovori:

    ```python
    random_responses = ["That is quite interesting, please tell me more.",
                        "I see. Do go on.",
                        "Why do you say that?",
                        "Funny weather we've been having, isn't it?",
                        "Let's change the subject.",
                        "Did you catch the game last night?"]
    ```

    Tukaj je nekaj primerov izhoda za orientacijo (uporabniški vnos je na vrsticah, ki se začnejo z `>`):

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

    Ena možna rešitev naloge je [tukaj](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/1-Introduction-to-NLP/solution/bot.py).

    ✅ Ustavi se in razmisli

    1. Ali menite, da bi naključni odgovori 'pretentali' nekoga, da bi mislil, da bot dejansko razume?
    2. Katere funkcije bi bot potreboval, da bi bil bolj učinkovit?
    3. Če bi bot resnično 'razumel' pomen stavka, ali bi moral 'zapomniti' pomen prejšnjih stavkov v pogovoru?

---

## 🚀 Izziv

Izberite enega od zgornjih elementov "ustavi se in razmisli" in ga poskusite implementirati v kodi ali napišite rešitev na papirju z uporabo psevdo kode.

V naslednji lekciji boste spoznali številne druge pristope k razčlenjevanju naravnega jezika in strojnega učenja.

## [Kviz po predavanju](https://ff-quizzes.netlify.app/en/ml/)

## Pregled in samostojno učenje

Oglejte si spodnje reference za dodatne priložnosti za branje.

### Reference

1. Schubert, Lenhart, "Computational Linguistics", *The Stanford Encyclopedia of Philosophy* (Spring 2020 Edition), Edward N. Zalta (ed.), URL = <https://plato.stanford.edu/archives/spr2020/entries/computational-linguistics/>.
2. Princeton University "About WordNet." [WordNet](https://wordnet.princeton.edu/). Princeton University. 2010. 

## Naloga 

[Poiščite bota](assignment.md)

---

**Omejitev odgovornosti**:  
Ta dokument je bil preveden z uporabo storitve za strojno prevajanje [Co-op Translator](https://github.com/Azure/co-op-translator). Čeprav si prizadevamo za natančnost, vas prosimo, da upoštevate, da lahko avtomatizirani prevodi vsebujejo napake ali netočnosti. Izvirni dokument v njegovem izvirnem jeziku je treba obravnavati kot avtoritativni vir. Za ključne informacije priporočamo strokovno človeško prevajanje. Ne prevzemamo odgovornosti za morebitna nesporazumevanja ali napačne razlage, ki izhajajo iz uporabe tega prevoda.