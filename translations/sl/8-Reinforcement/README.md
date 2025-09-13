<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "20ca019012b1725de956681d036d8b18",
  "translation_date": "2025-09-05T13:32:10+00:00",
  "source_file": "8-Reinforcement/README.md",
  "language_code": "sl"
}
-->
# Uvod v učenje z okrepitvami

Učenje z okrepitvami, RL, velja za enega osnovnih paradigm strojnega učenja, poleg nadzorovanega in nenadzorovanega učenja. RL se osredotoča na sprejemanje odločitev: sprejemanje pravih odločitev ali vsaj učenje iz njih.

Predstavljajte si simulirano okolje, kot je borza. Kaj se zgodi, če uvedete določeno regulacijo? Ali ima pozitiven ali negativen učinek? Če se zgodi nekaj negativnega, morate to _negativno okrepitev_ uporabiti, se iz nje naučiti in spremeniti smer. Če je rezultat pozitiven, morate graditi na tej _pozitivni okrepitvi_.

![Peter in volk](../../../8-Reinforcement/images/peter.png)

> Peter in njegovi prijatelji morajo pobegniti lačnemu volku! Slika: [Jen Looper](https://twitter.com/jenlooper)

## Regionalna tema: Peter in volk (Rusija)

[Peter in volk](https://en.wikipedia.org/wiki/Peter_and_the_Wolf) je glasbena pravljica, ki jo je napisal ruski skladatelj [Sergej Prokofjev](https://en.wikipedia.org/wiki/Sergei_Prokofiev). Gre za zgodbo o mladem pionirju Petru, ki pogumno zapusti svojo hišo in se odpravi na gozdno jaso, da bi ujel volka. V tem poglavju bomo trenirali algoritme strojnega učenja, ki bodo Petru pomagali:

- **Raziskovati** okolico in zgraditi optimalen navigacijski zemljevid
- **Naučiti se** uporabljati rolko in ohranjati ravnotežje, da se bo lahko hitreje premikal.

[![Peter in volk](https://img.youtube.com/vi/Fmi5zHg4QSM/0.jpg)](https://www.youtube.com/watch?v=Fmi5zHg4QSM)

> 🎥 Kliknite zgornjo sliko, da poslušate Peter in volk, skladbo Prokofjeva

## Učenje z okrepitvami

V prejšnjih poglavjih ste videli dva primera problemov strojnega učenja:

- **Nadzorovano učenje**, kjer imamo podatkovne nabore, ki predlagajo vzorčne rešitve za problem, ki ga želimo rešiti. [Klasifikacija](../4-Classification/README.md) in [regresija](../2-Regression/README.md) sta nalogi nadzorovanega učenja.
- **Nenadzorovano učenje**, pri katerem nimamo označenih podatkov za učenje. Glavni primer nenadzorovanega učenja je [Gručenje](../5-Clustering/README.md).

V tem poglavju vam bomo predstavili nov tip problema učenja, ki ne zahteva označenih podatkov za učenje. Obstaja več vrst takšnih problemov:

- **[Polnadzorovano učenje](https://wikipedia.org/wiki/Semi-supervised_learning)**, kjer imamo veliko neoznačenih podatkov, ki jih lahko uporabimo za predhodno treniranje modela.
- **[Učenje z okrepitvami](https://wikipedia.org/wiki/Reinforcement_learning)**, pri katerem agent uči, kako se obnašati, z izvajanjem eksperimentov v simuliranem okolju.

### Primer - računalniška igra

Recimo, da želite naučiti računalnik igrati igro, kot sta šah ali [Super Mario](https://wikipedia.org/wiki/Super_Mario). Da bi računalnik igral igro, mora napovedati, katero potezo naj izvede v vsakem stanju igre. Čeprav se to morda zdi kot problem klasifikacije, ni - ker nimamo podatkovnega nabora s stanji in ustreznimi akcijami. Čeprav imamo morda nekaj podatkov, kot so obstoječe šahovske partije ali posnetki igralcev, ki igrajo Super Mario, je verjetno, da ti podatki ne bodo zadostno pokrili velikega števila možnih stanj.

Namesto iskanja obstoječih podatkov o igri se **učenje z okrepitvami** (RL) opira na idejo, da *računalnik večkrat igra igro* in opazuje rezultate. Tako za uporabo učenja z okrepitvami potrebujemo dve stvari:

- **Okolje** in **simulator**, ki nam omogočata, da igro večkrat igramo. Ta simulator bi določal vsa pravila igre ter možna stanja in akcije.

- **Funkcijo nagrajevanja**, ki nam pove, kako dobro smo se odrezali med posamezno potezo ali igro.

Glavna razlika med drugimi vrstami strojnega učenja in RL je, da pri RL običajno ne vemo, ali zmagamo ali izgubimo, dokler ne končamo igre. Tako ne moremo reči, ali je določena poteza sama po sebi dobra ali ne - nagrado prejmemo šele na koncu igre. Naš cilj je oblikovati algoritme, ki nam omogočajo treniranje modela v negotovih razmerah. Spoznali bomo en RL-algoritem, imenovan **Q-učenje**.

## Lekcije

1. [Uvod v učenje z okrepitvami in Q-učenje](1-QLearning/README.md)
2. [Uporaba simulacijskega okolja Gym](2-Gym/README.md)

## Zasluge

"Uvod v učenje z okrepitvami" je bilo napisano z ♥️ avtorja [Dmitry Soshnikov](http://soshnikov.com)

---

**Omejitev odgovornosti**:  
Ta dokument je bil preveden z uporabo storitve za prevajanje z umetno inteligenco [Co-op Translator](https://github.com/Azure/co-op-translator). Čeprav si prizadevamo za natančnost, vas prosimo, da upoštevate, da lahko avtomatizirani prevodi vsebujejo napake ali netočnosti. Izvirni dokument v njegovem izvirnem jeziku je treba obravnavati kot avtoritativni vir. Za ključne informacije priporočamo profesionalni človeški prevod. Ne prevzemamo odgovornosti za morebitna napačna razumevanja ali napačne interpretacije, ki bi nastale zaradi uporabe tega prevoda.