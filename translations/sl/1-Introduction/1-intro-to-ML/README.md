<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "69389392fa6346e0dfa30f664b7b6fec",
  "translation_date": "2025-09-05T12:47:53+00:00",
  "source_file": "1-Introduction/1-intro-to-ML/README.md",
  "language_code": "sl"
}
-->
# Uvod v strojno učenje

## [Predhodni kviz](https://ff-quizzes.netlify.app/en/ml/)

---

[![ML za začetnike - Uvod v strojno učenje za začetnike](https://img.youtube.com/vi/6mSx_KJxcHI/0.jpg)](https://youtu.be/6mSx_KJxcHI "ML za začetnike - Uvod v strojno učenje za začetnike")

> 🎥 Kliknite na zgornjo sliko za kratek video, ki obravnava to lekcijo.

Dobrodošli v tem tečaju klasičnega strojnega učenja za začetnike! Ne glede na to, ali ste popolnoma novi na tem področju ali izkušen strokovnjak za strojno učenje, ki želi osvežiti svoje znanje, veseli smo, da ste se nam pridružili! Želimo ustvariti prijazno izhodišče za vaše študije strojnega učenja in z veseljem ocenimo, odgovorimo ter vključimo vaše [povratne informacije](https://github.com/microsoft/ML-For-Beginners/discussions).

[![Uvod v strojno učenje](https://img.youtube.com/vi/h0e2HAPTGF4/0.jpg)](https://youtu.be/h0e2HAPTGF4 "Uvod v strojno učenje")

> 🎥 Kliknite na zgornjo sliko za video: MIT-ov John Guttag predstavlja strojno učenje

---
## Začetek s strojnim učenjem

Preden začnete s tem učnim načrtom, morate pripraviti svoj računalnik za lokalno izvajanje beležk.

- **Pripravite svoj računalnik s temi videi**. Uporabite naslednje povezave, da se naučite [kako namestiti Python](https://youtu.be/CXZYvNRIAKM) na vaš sistem in [nastaviti urejevalnik besedila](https://youtu.be/EU8eayHWoZg) za razvoj.
- **Naučite se Python**. Priporočljivo je, da imate osnovno razumevanje [Python](https://docs.microsoft.com/learn/paths/python-language/?WT.mc_id=academic-77952-leestott), programskega jezika, ki je koristen za podatkovne znanstvenike in ga uporabljamo v tem tečaju.
- **Naučite se Node.js in JavaScript**. JavaScript uporabljamo nekajkrat v tem tečaju pri gradnji spletnih aplikacij, zato boste potrebovali [node](https://nodejs.org) in [npm](https://www.npmjs.com/), pa tudi [Visual Studio Code](https://code.visualstudio.com/) za razvoj v Pythonu in JavaScriptu.
- **Ustvarite GitHub račun**. Ker ste nas našli tukaj na [GitHub](https://github.com), morda že imate račun, če pa ne, ga ustvarite in nato razvejite ta učni načrt za lastno uporabo. (Lahko nam tudi podarite zvezdico 😊)
- **Raziščite Scikit-learn**. Seznanite se z [Scikit-learn](https://scikit-learn.org/stable/user_guide.html), nizom knjižnic za strojno učenje, ki jih uporabljamo v teh lekcijah.

---
## Kaj je strojno učenje?

Izraz 'strojno učenje' je eden najbolj priljubljenih in pogosto uporabljenih izrazov danes. Obstaja velika verjetnost, da ste ta izraz vsaj enkrat slišali, če imate kakršnokoli povezavo s tehnologijo, ne glede na področje, v katerem delate. Mehanika strojnega učenja pa je za večino ljudi skrivnost. Za začetnika v strojnem učenju se lahko tema včasih zdi preobsežna. Zato je pomembno razumeti, kaj strojno učenje dejansko je, in se o njem učiti korak za korakom, skozi praktične primere.

---
## Krivulja navdušenja

![krivulja navdušenja strojnega učenja](../../../../1-Introduction/1-intro-to-ML/images/hype.png)

> Google Trends prikazuje nedavno 'krivuljo navdušenja' izraza 'strojno učenje'

---
## Skrivnostno vesolje

Živimo v vesolju, polnem fascinantnih skrivnosti. Veliki znanstveniki, kot so Stephen Hawking, Albert Einstein in mnogi drugi, so svoja življenja posvetili iskanju smiselnih informacij, ki razkrivajo skrivnosti sveta okoli nas. To je človeška narava učenja: človeški otrok se uči novih stvari in odkriva strukturo svojega sveta leto za letom, ko odrašča.

---
## Otroški možgani

Otroški možgani in čuti zaznavajo dejstva iz okolice ter postopoma spoznavajo skrite vzorce življenja, ki otroku pomagajo oblikovati logična pravila za prepoznavanje naučenih vzorcev. Proces učenja človeških možganov naredi ljudi najbolj sofisticirana živa bitja na tem svetu. Nenehno učenje z odkrivanjem skritih vzorcev in nato inoviranje na podlagi teh vzorcev nam omogoča, da se skozi življenje nenehno izboljšujemo. Ta sposobnost učenja in evolucije je povezana s konceptom, imenovanim [plastičnost možganov](https://www.simplypsychology.org/brain-plasticity.html). Površinsko lahko potegnemo nekaj motivacijskih podobnosti med procesom učenja človeških možganov in koncepti strojnega učenja.

---
## Človeški možgani

[Človeški možgani](https://www.livescience.com/29365-human-brain.html) zaznavajo stvari iz resničnega sveta, obdelujejo zaznane informacije, sprejemajo racionalne odločitve in izvajajo določena dejanja glede na okoliščine. To imenujemo inteligentno vedenje. Ko programiramo posnemanje inteligentnega vedenjskega procesa v stroj, to imenujemo umetna inteligenca (AI).

---
## Nekaj terminologije

Čeprav se izrazi lahko zamenjujejo, je strojno učenje (ML) pomemben podsklop umetne inteligence. **ML se ukvarja z uporabo specializiranih algoritmov za odkrivanje smiselnih informacij in iskanje skritih vzorcev iz zaznanih podatkov, da podpre racionalni proces odločanja**.

---
## AI, ML, globoko učenje

![AI, ML, globoko učenje, podatkovna znanost](../../../../1-Introduction/1-intro-to-ML/images/ai-ml-ds.png)

> Diagram, ki prikazuje odnose med AI, ML, globokim učenjem in podatkovno znanostjo. Infografika avtorice [Jen Looper](https://twitter.com/jenlooper), navdihnjena z [to grafiko](https://softwareengineering.stackexchange.com/questions/366996/distinction-between-ai-ml-neural-networks-deep-learning-and-data-mining)

---
## Koncepti, ki jih bomo obravnavali

V tem učnem načrtu bomo obravnavali le osnovne koncepte strojnega učenja, ki jih mora poznati začetnik. Obravnavamo tisto, kar imenujemo 'klasično strojno učenje', predvsem z uporabo Scikit-learn, odlične knjižnice, ki jo mnogi študenti uporabljajo za učenje osnov. Za razumevanje širših konceptov umetne inteligence ali globokega učenja je močno temeljno znanje strojnega učenja nepogrešljivo, zato ga želimo ponuditi tukaj.

---
## V tem tečaju se boste naučili:

- osnovnih konceptov strojnega učenja
- zgodovine ML
- ML in pravičnosti
- regresijskih tehnik ML
- klasifikacijskih tehnik ML
- tehnik grupiranja ML
- tehnik obdelave naravnega jezika ML
- tehnik napovedovanja časovnih vrst ML
- okrepljenega učenja
- resničnih aplikacij za ML

---
## Kaj ne bomo obravnavali

- globoko učenje
- nevronske mreže
- AI

Za boljšo izkušnjo učenja se bomo izognili kompleksnostim nevronskih mrež, 'globokega učenja' - gradnje modelov z več plastmi z uporabo nevronskih mrež - in AI, o čemer bomo razpravljali v drugem učnem načrtu. Prav tako bomo ponudili prihajajoči učni načrt podatkovne znanosti, ki se bo osredotočil na ta vidik širšega področja.

---
## Zakaj študirati strojno učenje?

Strojno učenje je z vidika sistemov opredeljeno kot ustvarjanje avtomatiziranih sistemov, ki lahko iz podatkov odkrijejo skrite vzorce za pomoč pri sprejemanju inteligentnih odločitev.

Ta motivacija je ohlapno navdihnjena z načinom, kako človeški možgani učijo določene stvari na podlagi podatkov, ki jih zaznavajo iz zunanjega sveta.

✅ Premislite za trenutek, zakaj bi podjetje želelo uporabiti strategije strojnega učenja namesto ustvarjanja sistema s trdo kodiranimi pravili.

---
## Aplikacije strojnega učenja

Aplikacije strojnega učenja so zdaj skoraj povsod in so tako razširjene kot podatki, ki krožijo po naših družbah, ustvarjeni s pametnimi telefoni, povezanimi napravami in drugimi sistemi. Glede na izjemen potencial najsodobnejših algoritmov strojnega učenja raziskovalci preučujejo njihovo sposobnost reševanja večdimenzionalnih in večdisciplinarnih resničnih problemov z odličnimi pozitivnimi rezultati.

---
## Primeri uporabe ML

**Strojno učenje lahko uporabite na številne načine**:

- Za napovedovanje verjetnosti bolezni na podlagi pacientove zdravstvene zgodovine ali poročil.
- Za uporabo vremenskih podatkov za napovedovanje vremenskih dogodkov.
- Za razumevanje sentimenta besedila.
- Za odkrivanje lažnih novic in preprečevanje širjenja propagande.

Finance, ekonomija, znanost o Zemlji, raziskovanje vesolja, biomedicinski inženiring, kognitivna znanost in celo področja humanistike so prilagodili strojno učenje za reševanje težkih problemov obdelave podatkov v svojih domenah.

---
## Zaključek

Strojno učenje avtomatizira proces odkrivanja vzorcev z iskanjem smiselnih vpogledov iz resničnih ali generiranih podatkov. Izkazalo se je, da je izjemno dragoceno na področjih poslovanja, zdravja in financ, med drugim.

V bližnji prihodnosti bo razumevanje osnov strojnega učenja postalo nujno za ljudi iz katerega koli področja zaradi njegove široke uporabe.

---
# 🚀 Izziv

Narišite, na papirju ali z uporabo spletne aplikacije, kot je [Excalidraw](https://excalidraw.com/), vaše razumevanje razlik med AI, ML, globokim učenjem in podatkovno znanostjo. Dodajte nekaj idej o problemih, ki jih lahko vsaka od teh tehnik dobro rešuje.

# [Kviz po predavanju](https://ff-quizzes.netlify.app/en/ml/)

---
# Pregled in samostojno učenje

Če želite izvedeti več o tem, kako lahko delate z algoritmi ML v oblaku, sledite tej [učni poti](https://docs.microsoft.com/learn/paths/create-no-code-predictive-models-azure-machine-learning/?WT.mc_id=academic-77952-leestott).

Sprejmite [učni načrt](https://docs.microsoft.com/learn/modules/introduction-to-machine-learning/?WT.mc_id=academic-77952-leestott) o osnovah ML.

---
# Naloga

[Začnite z delom](assignment.md)

---

**Omejitev odgovornosti**:  
Ta dokument je bil preveden z uporabo storitve za prevajanje z umetno inteligenco [Co-op Translator](https://github.com/Azure/co-op-translator). Čeprav si prizadevamo za natančnost, vas prosimo, da upoštevate, da lahko avtomatizirani prevodi vsebujejo napake ali netočnosti. Izvirni dokument v njegovem maternem jeziku je treba obravnavati kot avtoritativni vir. Za ključne informacije priporočamo profesionalni človeški prevod. Ne prevzemamo odgovornosti za morebitna nesporazumevanja ali napačne razlage, ki bi nastale zaradi uporabe tega prevoda.