<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "69389392fa6346e0dfa30f664b7b6fec",
  "translation_date": "2025-09-05T16:07:12+00:00",
  "source_file": "1-Introduction/1-intro-to-ML/README.md",
  "language_code": "sk"
}
-->
# Úvod do strojového učenia

## [Kvíz pred prednáškou](https://ff-quizzes.netlify.app/en/ml/)

---

[![ML pre začiatočníkov - Úvod do strojového učenia pre začiatočníkov](https://img.youtube.com/vi/6mSx_KJxcHI/0.jpg)](https://youtu.be/6mSx_KJxcHI "ML pre začiatočníkov - Úvod do strojového učenia pre začiatočníkov")

> 🎥 Kliknite na obrázok vyššie pre krátke video k tejto lekcii.

Vitajte v tomto kurze klasického strojového učenia pre začiatočníkov! Či už ste v tejto téme úplne noví, alebo skúsený odborník na strojové učenie, ktorý si chce zopakovať určité oblasti, sme radi, že ste sa k nám pridali! Chceme vytvoriť priateľské miesto na začiatok vášho štúdia strojového učenia a radi by sme vyhodnotili, reagovali na vaše [spätné väzby](https://github.com/microsoft/ML-For-Beginners/discussions) a začlenili ich.

[![Úvod do ML](https://img.youtube.com/vi/h0e2HAPTGF4/0.jpg)](https://youtu.be/h0e2HAPTGF4 "Úvod do ML")

> 🎥 Kliknite na obrázok vyššie pre video: John Guttag z MIT predstavuje strojové učenie

---
## Začíname so strojovým učením

Predtým, než začnete s týmto učebným plánom, je potrebné pripraviť váš počítač na spúšťanie notebookov lokálne.

- **Nakonfigurujte svoj počítač pomocou týchto videí**. Použite nasledujúce odkazy na [inštaláciu Pythonu](https://youtu.be/CXZYvNRIAKM) vo vašom systéme a [nastavenie textového editora](https://youtu.be/EU8eayHWoZg) pre vývoj.
- **Naučte sa Python**. Odporúča sa mať základné znalosti [Pythonu](https://docs.microsoft.com/learn/paths/python-language/?WT.mc_id=academic-77952-leestott), programovacieho jazyka užitočného pre dátových vedcov, ktorý používame v tomto kurze.
- **Naučte sa Node.js a JavaScript**. JavaScript používame niekoľkokrát v tomto kurze pri tvorbe webových aplikácií, takže budete potrebovať [node](https://nodejs.org) a [npm](https://www.npmjs.com/) nainštalované, ako aj [Visual Studio Code](https://code.visualstudio.com/) dostupné pre vývoj v Pythone a JavaScripte.
- **Vytvorte si GitHub účet**. Keďže ste nás našli na [GitHube](https://github.com), možno už máte účet, ale ak nie, vytvorte si ho a potom si tento učebný plán forknite na vlastné použitie. (Môžete nám dať aj hviezdičku 😊)
- **Preskúmajte Scikit-learn**. Zoznámte sa s [Scikit-learn](https://scikit-learn.org/stable/user_guide.html), súborom knižníc pre strojové učenie, na ktoré sa odkazujeme v týchto lekciách.

---
## Čo je strojové učenie?

Termín 'strojové učenie' je jedným z najpopulárnejších a najčastejšie používaných termínov dneška. Je dosť pravdepodobné, že ste tento termín aspoň raz počuli, ak máte nejakú znalosť technológií, bez ohľadu na oblasť, v ktorej pracujete. Mechanizmy strojového učenia sú však pre väčšinu ľudí záhadou. Pre začiatočníka v strojovom učení môže byť táto téma niekedy ohromujúca. Preto je dôležité pochopiť, čo strojové učenie vlastne je, a učiť sa o ňom krok za krokom, prostredníctvom praktických príkladov.

---
## Krivka nadšenia

![krivka nadšenia pre ML](../../../../1-Introduction/1-intro-to-ML/images/hype.png)

> Google Trends ukazuje nedávnu 'krivku nadšenia' pre termín 'strojové učenie'

---
## Záhadný vesmír

Žijeme vo vesmíre plnom fascinujúcich záhad. Veľkí vedci ako Stephen Hawking, Albert Einstein a mnohí ďalší zasvätili svoje životy hľadaniu zmysluplných informácií, ktoré odhaľujú tajomstvá sveta okolo nás. Toto je ľudská podstata učenia: ľudské dieťa sa učí nové veci a rok čo rok odhaľuje štruktúru svojho sveta, keď dospieva.

---
## Mozog dieťaťa

Mozog a zmysly dieťaťa vnímajú fakty zo svojho okolia a postupne sa učia skryté vzory života, ktoré pomáhajú dieťaťu vytvárať logické pravidlá na identifikáciu naučených vzorov. Proces učenia ľudského mozgu robí z ľudí najsofistikovanejšie živé bytosti na tomto svete. Neustále učenie sa objavovaním skrytých vzorov a následné inovovanie na základe týchto vzorov nám umožňuje zlepšovať sa počas celého života. Táto schopnosť učenia a evolúcie súvisí s konceptom nazývaným [plasticita mozgu](https://www.simplypsychology.org/brain-plasticity.html). Povrchne môžeme nájsť niektoré motivačné podobnosti medzi procesom učenia ľudského mozgu a konceptmi strojového učenia.

---
## Ľudský mozog

[Ľudský mozog](https://www.livescience.com/29365-human-brain.html) vníma veci z reálneho sveta, spracováva vnímané informácie, robí racionálne rozhodnutia a vykonáva určité akcie na základe okolností. Toto nazývame inteligentným správaním. Keď naprogramujeme napodobeninu inteligentného procesu správania do stroja, nazýva sa to umelá inteligencia (AI).

---
## Niektoré pojmy

Aj keď sa pojmy môžu zamieňať, strojové učenie (ML) je dôležitou podmnožinou umelej inteligencie. **ML sa zaoberá používaním špecializovaných algoritmov na odhaľovanie zmysluplných informácií a hľadanie skrytých vzorov z vnímaných dát na podporu procesu racionálneho rozhodovania**.

---
## AI, ML, Hlboké učenie

![AI, ML, hlboké učenie, dátová veda](../../../../1-Introduction/1-intro-to-ML/images/ai-ml-ds.png)

> Diagram ukazujúci vzťahy medzi AI, ML, hlbokým učením a dátovou vedou. Infografika od [Jen Looper](https://twitter.com/jenlooper) inšpirovaná [týmto grafom](https://softwareengineering.stackexchange.com/questions/366996/distinction-between-ai-ml-neural-networks-deep-learning-and-data-mining)

---
## Koncepty, ktoré pokryjeme

V tomto učebnom pláne sa budeme venovať iba základným konceptom strojového učenia, ktoré musí začiatočník poznať. Pokryjeme to, čo nazývame 'klasické strojové učenie', primárne pomocou Scikit-learn, vynikajúcej knižnice, ktorú mnohí študenti používajú na učenie základov. Na pochopenie širších konceptov umelej inteligencie alebo hlbokého učenia je nevyhnutné mať silné základné znalosti strojového učenia, a preto ich chceme ponúknuť tu.

---
## V tomto kurze sa naučíte:

- základné koncepty strojového učenia
- históriu ML
- ML a spravodlivosť
- regresné techniky ML
- klasifikačné techniky ML
- techniky zhlukovania ML
- techniky spracovania prirodzeného jazyka ML
- techniky predpovedania časových radov ML
- posilňovacie učenie
- reálne aplikácie ML

---
## Čo nebudeme pokrývať

- hlboké učenie
- neurónové siete
- AI

Aby sme zabezpečili lepší zážitok z učenia, vyhneme sa zložitostiam neurónových sietí, 'hlbokého učenia' - modelovania s mnohými vrstvami pomocou neurónových sietí - a AI, o ktorých budeme diskutovať v inom učebnom pláne. Taktiež pripravujeme učebný plán dátovej vedy, ktorý sa zameria na tento aspekt širšieho poľa.

---
## Prečo študovať strojové učenie?

Strojové učenie je z pohľadu systémov definované ako tvorba automatizovaných systémov, ktoré dokážu učiť skryté vzory z dát na podporu inteligentného rozhodovania.

Táto motivácia je voľne inšpirovaná tým, ako ľudský mozog učí určité veci na základe dát, ktoré vníma z vonkajšieho sveta.

✅ Zamyslite sa na chvíľu, prečo by firma chcela použiť stratégie strojového učenia namiesto vytvorenia pevne zakódovaného systému založeného na pravidlách.

---
## Aplikácie strojového učenia

Aplikácie strojového učenia sú dnes takmer všade a sú rovnako rozšírené ako dáta, ktoré prúdia našimi spoločnosťami, generované našimi smartfónmi, pripojenými zariadeniami a inými systémami. Vzhľadom na obrovský potenciál najmodernejších algoritmov strojového učenia skúmajú výskumníci ich schopnosť riešiť multidimenzionálne a multidisciplinárne problémy reálneho života s veľkými pozitívnymi výsledkami.

---
## Príklady aplikovaného ML

**Strojové učenie môžete použiť mnohými spôsobmi**:

- Na predpovedanie pravdepodobnosti ochorenia na základe zdravotnej histórie alebo správ pacienta.
- Na využitie údajov o počasí na predpovedanie meteorologických udalostí.
- Na pochopenie sentimentu textu.
- Na detekciu falošných správ a zastavenie šírenia propagandy.

Financie, ekonómia, vedy o Zemi, vesmírny výskum, biomedicínske inžinierstvo, kognitívne vedy a dokonca aj oblasti humanitných vied adaptovali strojové učenie na riešenie náročných problémov spracovania dát vo svojich oblastiach.

---
## Záver

Strojové učenie automatizuje proces objavovania vzorov tým, že nachádza zmysluplné poznatky z reálnych alebo generovaných dát. Ukázalo sa, že je mimoriadne hodnotné v podnikaní, zdravotníctve a finančných aplikáciách, medzi inými.

V blízkej budúcnosti bude pochopenie základov strojového učenia nevyhnutné pre ľudí z akejkoľvek oblasti vzhľadom na jeho široké prijatie.

---
# 🚀 Výzva

Nakreslite na papier alebo pomocou online aplikácie ako [Excalidraw](https://excalidraw.com/) vaše pochopenie rozdielov medzi AI, ML, hlbokým učením a dátovou vedou. Pridajte niekoľko nápadov na problémy, ktoré sú každá z týchto techník dobré pri riešení.

# [Kvíz po prednáške](https://ff-quizzes.netlify.app/en/ml/)

---
# Prehľad a samostatné štúdium

Ak sa chcete dozvedieť viac o tom, ako môžete pracovať s ML algoritmami v cloude, sledujte tento [učebný plán](https://docs.microsoft.com/learn/paths/create-no-code-predictive-models-azure-machine-learning/?WT.mc_id=academic-77952-leestott).

Absolvujte [učebný plán](https://docs.microsoft.com/learn/modules/introduction-to-machine-learning/?WT.mc_id=academic-77952-leestott) o základoch ML.

---
# Zadanie

[Začnite](assignment.md)

---

**Upozornenie**:  
Tento dokument bol preložený pomocou služby AI prekladu [Co-op Translator](https://github.com/Azure/co-op-translator). Hoci sa snažíme o presnosť, prosím, berte na vedomie, že automatizované preklady môžu obsahovať chyby alebo nepresnosti. Pôvodný dokument v jeho pôvodnom jazyku by mal byť považovaný za autoritatívny zdroj. Pre kritické informácie sa odporúča profesionálny ľudský preklad. Nie sme zodpovední za žiadne nedorozumenia alebo nesprávne interpretácie vyplývajúce z použitia tohto prekladu.