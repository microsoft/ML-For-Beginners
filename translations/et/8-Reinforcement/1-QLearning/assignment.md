<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "68394b2102d3503882e5e914bd0ff5c1",
  "translation_date": "2025-10-11T11:21:08+00:00",
  "source_file": "8-Reinforcement/1-QLearning/assignment.md",
  "language_code": "et"
}
-->
# Realistlikum maailm

Meie olukorras suutis Peeter liikuda peaaegu väsimata või nälga tundmata. Realistlikumas maailmas peab ta aeg-ajalt istuma ja puhkama ning ka end toitma. Muudame oma maailma realistlikumaks, rakendades järgmisi reegleid:

1. Liikudes ühest kohast teise kaotab Peeter **energiat** ja kogub **väsimust**.
2. Peeter saab energiat juurde, süües õunu.
3. Peeter saab väsimusest lahti, puhates puu all või murul (st liikudes laua asukohta, kus on puu või muru - roheline ala).
4. Peeter peab leidma ja tapma hundi.
5. Hundi tapmiseks peab Peeter omama teatud tasemel energiat ja väsimust, vastasel juhul kaotab ta lahingu.

## Juhised

Kasuta algset [notebook.ipynb](notebook.ipynb) märkmikku oma lahenduse lähtepunktina.

Muuda ülaltoodud tasustamisfunktsiooni vastavalt mängu reeglitele, käivita tugevdusõppe algoritm, et õppida parim strateegia mängu võitmiseks, ja võrdle juhusliku liikumise tulemusi oma algoritmiga mängude võitude ja kaotuste arvu osas.

> **Note**: Uues maailmas on olek keerulisem ja lisaks inimese positsioonile hõlmab see ka väsimuse ja energiatasemeid. Võid valida oleku esitlemise tuplina (Board, energy, fatigue), defineerida oleku jaoks klassi (võid selle ka tuletada `Board`-ist) või isegi muuta algset `Board` klassi failis [rlboard.py](../../../../8-Reinforcement/1-QLearning/rlboard.py).

Oma lahenduses säilita juhusliku liikumise strateegia eest vastutav kood ja võrdle oma algoritmi tulemusi juhusliku liikumisega lõpus.

> **Note**: Võid vajada hüperparameetrite kohandamist, et see töötaks, eriti epohhide arvu osas. Kuna mängu edu (hundi vastu võitlemine) on harv sündmus, võid oodata palju pikemat treeninguaega.

## Hindamiskriteeriumid

| Kriteerium | Silmapaistev                                                                                                                                                                                             | Piisav                                                                                                                                                                                | Vajab parandamist                                                                                                                          |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
|            | Märkmik sisaldab uue maailma reeglite määratlust, Q-õppe algoritmi ja mõningaid tekstilisi selgitusi. Q-õpe suudab juhusliku liikumisega võrreldes tulemusi märkimisväärselt parandada.                   | Märkmik on esitatud, Q-õpe on rakendatud ja parandab juhusliku liikumisega võrreldes tulemusi, kuid mitte märkimisväärselt; või märkmik on halvasti dokumenteeritud ja kood pole hästi struktureeritud | On tehtud katseid maailma reeglite ümberdefineerimiseks, kuid Q-õppe algoritm ei tööta või tasustamisfunktsioon pole täielikult määratletud |

---

**Lahtiütlus**:  
See dokument on tõlgitud AI tõlketeenuse [Co-op Translator](https://github.com/Azure/co-op-translator) abil. Kuigi püüame tagada täpsust, palume arvestada, et automaatsed tõlked võivad sisaldada vigu või ebatäpsusi. Algne dokument selle algses keeles tuleks pidada autoriteetseks allikaks. Olulise teabe puhul soovitame kasutada professionaalset inimtõlget. Me ei vastuta selle tõlke kasutamisest tulenevate arusaamatuste või valesti tõlgenduste eest.