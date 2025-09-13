<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1c2ec40cf55c98a028a359c27ef7e45a",
  "translation_date": "2025-09-05T14:09:37+00:00",
  "source_file": "6-NLP/1-Introduction-to-NLP/README.md",
  "language_code": "hr"
}
-->
# Uvod u obradu prirodnog jezika

Ova lekcija pokriva kratku povijest i važne koncepte *obrade prirodnog jezika*, područja unutar *računalne lingvistike*.

## [Kviz prije predavanja](https://ff-quizzes.netlify.app/en/ml/)

## Uvod

NLP, kako se često naziva, jedno je od najpoznatijih područja gdje se strojno učenje primjenjuje i koristi u proizvodnom softveru.

✅ Možete li se sjetiti softvera koji svakodnevno koristite, a koji vjerojatno ima ugrađen NLP? Što je s vašim programima za obradu teksta ili mobilnim aplikacijama koje redovito koristite?

Naučit ćete o:

- **Ideji jezika**. Kako su se jezici razvijali i koja su glavna područja proučavanja.
- **Definicijama i konceptima**. Također ćete naučiti definicije i koncepte o tome kako računala obrađuju tekst, uključujući parsiranje, gramatiku i prepoznavanje imenica i glagola. U ovoj lekciji postoje neki zadaci kodiranja, a uvode se nekoliko važnih koncepata koje ćete kasnije naučiti kodirati u sljedećim lekcijama.

## Računalna lingvistika

Računalna lingvistika je područje istraživanja i razvoja koje se proučava desetljećima, a bavi se time kako računala mogu raditi s jezicima, pa čak i razumjeti, prevoditi i komunicirati s njima. Obrada prirodnog jezika (NLP) je povezano područje koje se fokusira na to kako računala mogu obrađivati 'prirodne', odnosno ljudske jezike.

### Primjer - diktiranje na telefonu

Ako ste ikada diktirali svom telefonu umjesto da tipkate ili pitali virtualnog asistenta pitanje, vaš govor je pretvoren u tekstualni oblik, a zatim obrađen ili *parsiran* iz jezika kojim ste govorili. Detektirane ključne riječi zatim su obrađene u format koji telefon ili asistent može razumjeti i na temelju kojeg može djelovati.

![razumijevanje](../../../../6-NLP/1-Introduction-to-NLP/images/comprehension.png)
> Pravo lingvističko razumijevanje je teško! Slika: [Jen Looper](https://twitter.com/jenlooper)

### Kako je ova tehnologija omogućena?

Ovo je moguće jer je netko napisao računalni program za to. Prije nekoliko desetljeća, neki pisci znanstvene fantastike predviđali su da će ljudi uglavnom razgovarati sa svojim računalima, a računala će uvijek točno razumjeti što misle. Nažalost, pokazalo se da je to teži problem nego što su mnogi zamišljali, i iako je danas puno bolje razumljiv problem, postoje značajni izazovi u postizanju 'savršene' obrade prirodnog jezika kada je riječ o razumijevanju značenja rečenice. Ovo je posebno težak problem kada je riječ o razumijevanju humora ili otkrivanju emocija poput sarkazma u rečenici.

U ovom trenutku možda se prisjećate školskih sati gdje je učitelj objašnjavao dijelove gramatike u rečenici. U nekim zemljama učenici uče gramatiku i lingvistiku kao poseban predmet, dok su u mnogima ti sadržaji uključeni u učenje jezika: bilo prvog jezika u osnovnoj školi (učenje čitanja i pisanja) ili možda drugog jezika u srednjoj školi. Ne brinite ako niste stručnjak za razlikovanje imenica od glagola ili priloga od pridjeva!

Ako se mučite s razlikom između *jednostavnog sadašnjeg vremena* i *sadašnjeg trajnog vremena*, niste sami. Ovo je izazov za mnoge ljude, čak i za izvorne govornike jezika. Dobra vijest je da su računala jako dobra u primjeni formalnih pravila, i naučit ćete pisati kod koji može *parsirati* rečenicu jednako dobro kao i čovjek. Veći izazov koji ćete kasnije istražiti je razumijevanje *značenja* i *sentimenta* rečenice.

## Preduvjeti

Za ovu lekciju, glavni preduvjet je sposobnost čitanja i razumijevanja jezika ove lekcije. Nema matematičkih problema ili jednadžbi za rješavanje. Iako je originalni autor napisao ovu lekciju na engleskom, ona je također prevedena na druge jezike, pa biste mogli čitati prijevod. Postoje primjeri gdje se koristi nekoliko različitih jezika (za usporedbu različitih gramatičkih pravila različitih jezika). Ti primjeri *nisu* prevedeni, ali objašnjavajući tekst jest, pa bi značenje trebalo biti jasno.

Za zadatke kodiranja koristit ćete Python, a primjeri koriste Python 3.8.

U ovom dijelu trebat ćete i koristiti:

- **Razumijevanje Pythona 3**. Razumijevanje programskog jezika Python 3, ova lekcija koristi unos, petlje, čitanje datoteka, nizove.
- **Visual Studio Code + ekstenzija**. Koristit ćemo Visual Studio Code i njegovu Python ekstenziju. Također možete koristiti Python IDE po svom izboru.
- **TextBlob**. [TextBlob](https://github.com/sloria/TextBlob) je pojednostavljena biblioteka za obradu teksta u Pythonu. Slijedite upute na web stranici TextBlob-a kako biste ga instalirali na svoj sustav (instalirajte i korpuse, kao što je prikazano dolje):

   ```bash
   pip install -U textblob
   python -m textblob.download_corpora
   ```

> 💡 Savjet: Python možete pokrenuti izravno u okruženjima VS Code-a. Pogledajte [dokumentaciju](https://code.visualstudio.com/docs/languages/python?WT.mc_id=academic-77952-leestott) za više informacija.

## Razgovor s računalima

Povijest pokušaja da se računala natjeraju da razumiju ljudski jezik seže desetljećima unatrag, a jedan od najranijih znanstvenika koji je razmatrao obradu prirodnog jezika bio je *Alan Turing*.

### 'Turingov test'

Kada je Turing istraživao *umjetnu inteligenciju* 1950-ih, razmatrao je može li se provesti test razgovora između čovjeka i računala (putem pisanog dopisivanja) gdje čovjek u razgovoru nije siguran razgovara li s drugim čovjekom ili računalom.

Ako, nakon određenog trajanja razgovora, čovjek ne može odrediti jesu li odgovori od računala ili ne, može li se tada reći da računalo *razmišlja*?

### Inspiracija - 'igra imitacije'

Ideja za ovo došla je iz društvene igre zvane *Igra imitacije* gdje je ispitivač sam u sobi i ima zadatak odrediti tko su od dvije osobe (u drugoj sobi) muškarac i žena. Ispitivač može slati bilješke i mora pokušati smisliti pitanja gdje pisani odgovori otkrivaju spol misteriozne osobe. Naravno, igrači u drugoj sobi pokušavaju zavarati ispitivača odgovarajući na pitanja na način koji zbunjuje ili dovodi u zabludu ispitivača, dok istovremeno daju dojam da odgovaraju iskreno.

### Razvoj Elize

1960-ih, znanstvenik s MIT-a po imenu *Joseph Weizenbaum* razvio je [*Elizu*](https://wikipedia.org/wiki/ELIZA), računalnog 'terapeuta' koji bi postavljao ljudima pitanja i davao dojam da razumije njihove odgovore. Međutim, iako je Eliza mogla parsirati rečenicu i identificirati određene gramatičke konstrukte i ključne riječi kako bi dala razuman odgovor, nije se moglo reći da *razumije* rečenicu. Ako bi Eliza bila suočena s rečenicom u formatu "**Ja sam** <u>tužan</u>", mogla bi preurediti i zamijeniti riječi u rečenici kako bi formirala odgovor "Koliko dugo ste **vi bili** <u>tužni</u>".

To je davalo dojam da Eliza razumije izjavu i postavlja dodatno pitanje, dok je u stvarnosti mijenjala vrijeme i dodavala neke riječi. Ako Eliza nije mogla identificirati ključnu riječ za koju ima odgovor, umjesto toga bi dala nasumičan odgovor koji bi mogao biti primjenjiv na mnoge različite izjave. Elizu se lako moglo zavarati, na primjer, ako bi korisnik napisao "**Vi ste** <u>bicikl</u>", mogla bi odgovoriti s "Koliko dugo sam **ja bio** <u>bicikl</u>?", umjesto razumnijeg odgovora.

[![Razgovor s Elizom](https://img.youtube.com/vi/RMK9AphfLco/0.jpg)](https://youtu.be/RMK9AphfLco "Razgovor s Elizom")

> 🎥 Kliknite na sliku iznad za video o originalnom programu ELIZA

> Napomena: Možete pročitati originalni opis [Elize](https://cacm.acm.org/magazines/1966/1/13317-elizaa-computer-program-for-the-study-of-natural-language-communication-between-man-and-machine/abstract) objavljen 1966. ako imate ACM račun. Alternativno, pročitajte o Elizi na [wikipediji](https://wikipedia.org/wiki/ELIZA)

## Vježba - kodiranje osnovnog razgovornog bota

Razgovorni bot, poput Elize, je program koji potiče unos korisnika i daje dojam da razumije i odgovara inteligentno. Za razliku od Elize, naš bot neće imati nekoliko pravila koja mu daju dojam inteligentnog razgovora. Umjesto toga, naš bot će imati samo jednu sposobnost: održavati razgovor s nasumičnim odgovorima koji bi mogli funkcionirati u gotovo svakom trivijalnom razgovoru.

### Plan

Koraci za izradu razgovornog bota:

1. Ispišite upute koje savjetuju korisniku kako komunicirati s botom
2. Pokrenite petlju
   1. Prihvatite unos korisnika
   2. Ako korisnik zatraži izlaz, izađite
   3. Obradite unos korisnika i odredite odgovor (u ovom slučaju, odgovor je nasumičan odabir iz popisa mogućih generičkih odgovora)
   4. Ispišite odgovor
3. Vratite se na korak 2

### Izrada bota

Sljedeće ćemo izraditi bota. Počet ćemo definiranjem nekih fraza.

1. Napravite ovog bota sami u Pythonu s sljedećim nasumičnim odgovorima:

    ```python
    random_responses = ["That is quite interesting, please tell me more.",
                        "I see. Do go on.",
                        "Why do you say that?",
                        "Funny weather we've been having, isn't it?",
                        "Let's change the subject.",
                        "Did you catch the game last night?"]
    ```

    Evo uzorka izlaza koji će vas voditi (unos korisnika je na linijama koje počinju s `>`):

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

    Jedno moguće rješenje zadatka nalazi se [ovdje](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/1-Introduction-to-NLP/solution/bot.py)

    ✅ Zaustavite se i razmislite

    1. Mislite li da bi nasumični odgovori 'zavarali' nekoga da pomisli da bot zapravo razumije?
    2. Koje bi značajke bot trebao imati da bi bio učinkovitiji?
    3. Ako bi bot zaista mogao 'razumjeti' značenje rečenice, bi li trebao 'zapamtiti' značenje prethodnih rečenica u razgovoru?

---

## 🚀Izazov

Odaberite jedan od elemenata "zaustavite se i razmislite" iznad i pokušajte ga implementirati u kodu ili napišite rješenje na papiru koristeći pseudokod.

U sljedećoj lekciji naučit ćete o brojnim drugim pristupima parsiranju prirodnog jezika i strojnog učenja.

## [Kviz nakon predavanja](https://ff-quizzes.netlify.app/en/ml/)

## Pregled i samostalno učenje

Pogledajte dolje navedene reference kao prilike za daljnje čitanje.

### Reference

1. Schubert, Lenhart, "Računalna lingvistika", *Stanford Encyclopedia of Philosophy* (Proljetno izdanje 2020.), Edward N. Zalta (ur.), URL = <https://plato.stanford.edu/archives/spr2020/entries/computational-linguistics/>.
2. Princeton University "O WordNetu." [WordNet](https://wordnet.princeton.edu/). Princeton University. 2010. 

## Zadatak 

[Potražite bota](assignment.md)

---

**Odricanje od odgovornosti**:  
Ovaj dokument je preveden pomoću AI usluge za prevođenje [Co-op Translator](https://github.com/Azure/co-op-translator). Iako nastojimo osigurati točnost, imajte na umu da automatski prijevodi mogu sadržavati pogreške ili netočnosti. Izvorni dokument na izvornom jeziku treba smatrati autoritativnim izvorom. Za ključne informacije preporučuje se profesionalni prijevod od strane čovjeka. Ne preuzimamo odgovornost za bilo kakva nesporazuma ili pogrešna tumačenja koja proizlaze iz korištenja ovog prijevoda.