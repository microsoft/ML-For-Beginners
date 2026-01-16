<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "20ca019012b1725de956681d036d8b18",
  "translation_date": "2025-10-11T11:14:48+00:00",
  "source_file": "8-Reinforcement/README.md",
  "language_code": "et"
}
-->
# Sissejuhatus tugevdusõppesse

Tugevdusõpe, RL, on üks põhilisi masinõppe paradigmasid, kõrvuti juhendatud ja juhendamata õppega. RL keskendub otsustele: õigete otsuste tegemisele või vähemalt nende õppimisele.

Kujutlege, et teil on simuleeritud keskkond, näiteks aktsiaturg. Mis juhtub, kui kehtestate teatud regulatsiooni? Kas sellel on positiivne või negatiivne mõju? Kui juhtub midagi negatiivset, peate võtma selle _negatiivse tugevduse_, sellest õppima ja suunda muutma. Kui tulemus on positiivne, peate sellele _positiivsele tugevdusele_ tuginedes edasi liikuma.

![Peeter ja hunt](../../../translated_images/et/peter.779730f9ba3a8a8d.png)

> Peeter ja tema sõbrad peavad põgenema näljase hundi eest! Pildi autor [Jen Looper](https://twitter.com/jenlooper)

## Regionaalne teema: Peeter ja hunt (Venemaa)

[Peeter ja hunt](https://en.wikipedia.org/wiki/Peter_and_the_Wolf) on muinasjutt, mille kirjutas vene helilooja [Sergei Prokofjev](https://en.wikipedia.org/wiki/Sergei_Prokofiev). See on lugu noorest pioneerist Peetrist, kes julgesti lahkub oma kodust, et metsas hundiga silmitsi seista. Selles osas treenime masinõppe algoritme, mis aitavad Peetril:

- **Avastada** ümbritsevat ala ja koostada optimaalne navigeerimiskaart
- **Õppida** kasutama rulaga tasakaalu hoidmist, et kiiremini liikuda.

[![Peeter ja hunt](https://img.youtube.com/vi/Fmi5zHg4QSM/0.jpg)](https://www.youtube.com/watch?v=Fmi5zHg4QSM)

> 🎥 Klõpsake ülaloleval pildil, et kuulata Prokofjevi "Peeter ja hunt"

## Tugevdusõpe

Eelnevates osades olete näinud kahte masinõppe probleemi näidet:

- **Juhendatud õpe**, kus meil on andmekogumid, mis pakuvad näidislahendusi probleemile, mida soovime lahendada. [Klassifikatsioon](../4-Classification/README.md) ja [regressioon](../2-Regression/README.md) on juhendatud õppe ülesanded.
- **Juhendamata õpe**, kus meil ei ole märgistatud treeningandmeid. Juhendamata õppe peamine näide on [klasterdamine](../5-Clustering/README.md).

Selles osas tutvustame teile uut tüüpi õppeprobleemi, mis ei vaja märgistatud treeningandmeid. Selliseid probleeme on mitut tüüpi:

- **[Pooljuhendatud õpe](https://wikipedia.org/wiki/Semi-supervised_learning)**, kus meil on palju märgistamata andmeid, mida saab kasutada mudeli eeltreenimiseks.
- **[Tugevdusõpe](https://wikipedia.org/wiki/Reinforcement_learning)**, kus agent õpib käitumist, tehes katseid mingis simuleeritud keskkonnas.

### Näide - arvutimäng

Oletame, et soovite õpetada arvutit mängima mängu, näiteks malet või [Super Mario](https://wikipedia.org/wiki/Super_Mario). Selleks, et arvuti mängu mängiks, peame õpetama seda ennustama, millist käiku teha igas mängu seisus. Kuigi see võib tunduda klassifikatsiooniprobleemina, ei ole see nii - sest meil ei ole andmekogumit, mis sisaldaks seisusid ja vastavaid tegevusi. Kuigi meil võib olla mõningaid andmeid, nagu olemasolevad malemängud või Super Mario mängijate salvestused, ei kata need andmed tõenäoliselt piisavalt suurt hulka võimalikke seisusid.

Selle asemel, et otsida olemasolevaid mänguandmeid, põhineb **Tugevdusõpe** (RL) ideel *lasta arvutil mängida* mitu korda ja jälgida tulemust. Seega, et rakendada tugevdusõpet, vajame kahte asja:

- **Keskkonda** ja **simulaatorit**, mis võimaldavad meil mängu mitu korda mängida. See simulaator määratleks kõik mängureeglid, samuti võimalikud seisud ja tegevused.

- **Tasu funktsiooni**, mis ütleks meile, kui hästi meil iga käigu või mängu ajal läks.

Peamine erinevus teiste masinõppe tüüpide ja RL vahel on see, et RL-is me tavaliselt ei tea, kas võidame või kaotame, kuni mäng on lõppenud. Seega ei saa me öelda, kas teatud käik iseenesest on hea või mitte - me saame tasu alles mängu lõpus. Meie eesmärk on kujundada algoritme, mis võimaldavad meil treenida mudelit ebakindlates tingimustes. Õpime tundma ühte RL algoritmi, mida nimetatakse **Q-õppeks**.

## Õppetunnid

1. [Sissejuhatus tugevdusõppesse ja Q-õppesse](1-QLearning/README.md)
2. [Simulatsioonikeskkonna kasutamine Gymis](2-Gym/README.md)

## Autorid

"Sissejuhatus tugevdusõppesse" on kirjutatud ♥️ poolt [Dmitry Soshnikov](http://soshnikov.com)

---

**Lahtiütlus**:  
See dokument on tõlgitud AI tõlketeenuse [Co-op Translator](https://github.com/Azure/co-op-translator) abil. Kuigi püüame tagada täpsust, palume arvestada, et automaatsed tõlked võivad sisaldada vigu või ebatäpsusi. Algne dokument selle algses keeles tuleks pidada autoriteetseks allikaks. Olulise teabe puhul soovitame kasutada professionaalset inimtõlget. Me ei vastuta selle tõlke kasutamisest tulenevate arusaamatuste või valesti tõlgenduste eest.