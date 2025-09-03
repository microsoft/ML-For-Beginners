<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "68394b2102d3503882e5e914bd0ff5c1",
  "translation_date": "2025-09-03T18:36:12+00:00",
  "source_file": "8-Reinforcement/1-QLearning/assignment.md",
  "language_code": "lt"
}
-->
# Realistiškesnis pasaulis

Mūsų situacijoje Peteris galėjo judėti beveik nepavargdamas ar nejausdamas alkio. Realistiškesniame pasaulyje jis turėtų kartais atsisėsti ir pailsėti, taip pat pasimaitinti. Padarykime mūsų pasaulį realistiškesnį, įgyvendindami šias taisykles:

1. Judėdamas iš vienos vietos į kitą, Peteris praranda **energiją** ir įgyja **nuovargį**.
2. Peteris gali gauti daugiau energijos valgydamas obuolius.
3. Peteris gali atsikratyti nuovargio ilsėdamasis po medžiu arba ant žolės (t. y. įžengdamas į lentos vietą su medžiu arba žole - žalią lauką).
4. Peteris turi surasti ir nužudyti vilką.
5. Kad nužudytų vilką, Peteris turi turėti tam tikrą energijos ir nuovargio lygį, kitaip jis pralaimės kovą.

## Instrukcijos

Naudokite originalų [notebook.ipynb](notebook.ipynb) užrašų knygelės failą kaip pradinį tašką savo sprendimui.

Pakeiskite aukščiau pateiktą atlygio funkciją pagal žaidimo taisykles, paleiskite stiprinamojo mokymosi algoritmą, kad išmoktumėte geriausią strategiją žaidimui laimėti, ir palyginkite atsitiktinio vaikščiojimo rezultatus su savo algoritmu pagal laimėtų ir pralaimėtų žaidimų skaičių.

> **Note**: Jūsų naujame pasaulyje būsena yra sudėtingesnė ir, be žmogaus pozicijos, taip pat apima nuovargio ir energijos lygius. Galite pasirinkti atvaizduoti būseną kaip tuple (Board,energy,fatigue), arba apibrėžti klasę būsenai (taip pat galite norėti ją išvesti iš `Board`), arba netgi modifikuoti originalią `Board` klasę faile [rlboard.py](../../../../8-Reinforcement/1-QLearning/rlboard.py).

Savo sprendime prašome išlaikyti kodą, atsakingą už atsitiktinio vaikščiojimo strategiją, ir palyginti savo algoritmo rezultatus su atsitiktiniu vaikščiojimu pabaigoje.

> **Note**: Jums gali tekti koreguoti hiperparametrus, kad algoritmas veiktų, ypač epochų skaičių. Kadangi žaidimo sėkmė (kova su vilku) yra retas įvykis, galite tikėtis daug ilgesnio mokymosi laiko.

## Vertinimo kriterijai

| Kriterijai | Puikiai                                                                                                                                                                                             | Pakankamai                                                                                                                                                                              | Reikia patobulinimų                                                                                                                        |
| ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
|            | Užrašų knygelė pateikta su naujo pasaulio taisyklių apibrėžimu, Q-Learning algoritmu ir tam tikrais tekstiniais paaiškinimais. Q-Learning ženkliai pagerina rezultatus, palyginti su atsitiktiniu vaikščiojimu. | Užrašų knygelė pateikta, Q-Learning įgyvendintas ir pagerina rezultatus, palyginti su atsitiktiniu vaikščiojimu, bet ne ženkliai; arba užrašų knygelė prastai dokumentuota ir kodas nėra gerai struktūruotas | Bandymas perkurti pasaulio taisykles atliktas, tačiau Q-Learning algoritmas neveikia arba atlygio funkcija nėra pilnai apibrėžta |

---

**Atsakomybės apribojimas**:  
Šis dokumentas buvo išverstas naudojant AI vertimo paslaugą [Co-op Translator](https://github.com/Azure/co-op-translator). Nors siekiame tikslumo, prašome atkreipti dėmesį, kad automatiniai vertimai gali turėti klaidų ar netikslumų. Originalus dokumentas jo gimtąja kalba turėtų būti laikomas autoritetingu šaltiniu. Kritinei informacijai rekomenduojama naudoti profesionalų žmogaus vertimą. Mes neprisiimame atsakomybės už nesusipratimus ar klaidingus interpretavimus, atsiradusius dėl šio vertimo naudojimo.