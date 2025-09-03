<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "20ca019012b1725de956681d036d8b18",
  "translation_date": "2025-09-03T18:25:56+00:00",
  "source_file": "8-Reinforcement/README.md",
  "language_code": "lt"
}
-->
# Įvadas į stiprinamąjį mokymąsi

Stiprinamasis mokymasis (RL) laikomas vienu iš pagrindinių mašininio mokymosi paradigmų, greta prižiūrimo mokymosi ir neprižiūrimo mokymosi. RL yra susijęs su sprendimais: priimti tinkamus sprendimus arba bent jau mokytis iš jų.

Įsivaizduokite, kad turite simuliuotą aplinką, pavyzdžiui, akcijų rinką. Kas nutiks, jei įvesite tam tikrą reguliavimą? Ar tai turės teigiamą ar neigiamą poveikį? Jei nutiks kažkas neigiamo, turite priimti šį _neigiamą stiprinimą_, pasimokyti iš jo ir pakeisti kryptį. Jei rezultatas yra teigiamas, turite remtis tuo _teigiamu stiprinimu_.

![peter ir vilkas](../../../translated_images/peter.779730f9ba3a8a8d9290600dcf55f2e491c0640c785af7ac0d64f583c49b8864.lt.png)

> Petras ir jo draugai turi pabėgti nuo alkano vilko! Vaizdas sukurtas [Jen Looper](https://twitter.com/jenlooper)

## Regioninė tema: Petras ir Vilkas (Rusija)

[Petras ir Vilkas](https://en.wikipedia.org/wiki/Peter_and_the_Wolf) yra muzikinė pasaka, kurią parašė rusų kompozitorius [Sergejus Prokofjevas](https://en.wikipedia.org/wiki/Sergei_Prokofiev). Tai pasakojimas apie jauną pionierių Petrą, kuris drąsiai išeina iš namų į miško laukymę, kad sugautų vilką. Šioje dalyje mes treniruosime mašininio mokymosi algoritmus, kurie padės Petrui:

- **Tyrinėti** aplinką ir sukurti optimizuotą navigacijos žemėlapį
- **Išmokti** naudotis riedlente ir išlaikyti pusiausvyrą, kad galėtų greičiau judėti.

[![Petras ir Vilkas](https://img.youtube.com/vi/Fmi5zHg4QSM/0.jpg)](https://www.youtube.com/watch?v=Fmi5zHg4QSM)

> 🎥 Spustelėkite aukščiau esantį vaizdą, kad išklausytumėte Prokofjevo kūrinį „Petras ir Vilkas“

## Stiprinamasis mokymasis

Ankstesnėse dalyse matėte du mašininio mokymosi problemų pavyzdžius:

- **Prižiūrimas mokymasis**, kai turime duomenų rinkinius, kurie siūlo pavyzdinius sprendimus problemai, kurią norime išspręsti. [Klasifikacija](../4-Classification/README.md) ir [regresija](../2-Regression/README.md) yra prižiūrimo mokymosi užduotys.
- **Neprižiūrimas mokymasis**, kai neturime pažymėtų mokymo duomenų. Pagrindinis neprižiūrimo mokymosi pavyzdys yra [Grupavimas](../5-Clustering/README.md).

Šioje dalyje mes supažindinsime jus su naujo tipo mokymosi problema, kuriai nereikia pažymėtų mokymo duomenų. Yra keletas tokių problemų tipų:

- **[Pusiau prižiūrimas mokymasis](https://wikipedia.org/wiki/Semi-supervised_learning)**, kai turime daug nepažymėtų duomenų, kuriuos galima naudoti modelio išankstiniam mokymui.
- **[Stiprinamasis mokymasis](https://wikipedia.org/wiki/Reinforcement_learning)**, kai agentas mokosi elgtis atlikdamas eksperimentus tam tikroje simuliuotoje aplinkoje.

### Pavyzdys - kompiuterinis žaidimas

Tarkime, norite išmokyti kompiuterį žaisti žaidimą, pavyzdžiui, šachmatus ar [Super Mario](https://wikipedia.org/wiki/Super_Mario). Kad kompiuteris galėtų žaisti žaidimą, reikia, kad jis numatytų, kokį ėjimą atlikti kiekvienoje žaidimo būsenoje. Nors tai gali atrodyti kaip klasifikacijos problema, taip nėra - nes neturime duomenų rinkinio su būsenomis ir atitinkamais veiksmais. Nors galime turėti duomenų, tokių kaip esamos šachmatų partijos ar žaidėjų „Super Mario“ žaidimo įrašai, tikėtina, kad tie duomenys nepakankamai apims didelį galimų būsenų skaičių.

Užuot ieškoję esamų žaidimo duomenų, **Stiprinamasis mokymasis** (RL) remiasi idėja, kad *kompiuteris žaistų* daug kartų ir stebėtų rezultatą. Taigi, norint taikyti stiprinamąjį mokymąsi, mums reikia dviejų dalykų:

- **Aplinkos** ir **simuliatoriaus**, kurie leistų mums žaisti žaidimą daug kartų. Šis simuliatorius apibrėžtų visas žaidimo taisykles, galimas būsenas ir veiksmus.

- **Atlygio funkcijos**, kuri nurodytų, kaip gerai pasirodėme kiekvieno ėjimo ar žaidimo metu.

Pagrindinis skirtumas tarp kitų mašininio mokymosi tipų ir RL yra tas, kad RL dažniausiai nežinome, ar laimime, ar pralaimime, kol nebaigiame žaidimo. Taigi, negalime pasakyti, ar tam tikras ėjimas vienas pats yra geras, ar ne - atlygio gauname tik žaidimo pabaigoje. Mūsų tikslas yra sukurti algoritmus, kurie leistų mums treniruoti modelį esant neapibrėžtoms sąlygoms. Mes išmoksime apie vieną RL algoritmą, vadinamą **Q-mokymusi**.

## Pamokos

1. [Įvadas į stiprinamąjį mokymąsi ir Q-mokymąsi](1-QLearning/README.md)
2. [Simuliacinės aplinkos naudojimas su „Gym“](2-Gym/README.md)

## Kreditas

„Įvadas į stiprinamąjį mokymąsi“ buvo parašytas su ♥️ [Dmitrijaus Sošnikovo](http://soshnikov.com)

---

**Atsakomybės apribojimas**:  
Šis dokumentas buvo išverstas naudojant AI vertimo paslaugą [Co-op Translator](https://github.com/Azure/co-op-translator). Nors siekiame tikslumo, prašome atkreipti dėmesį, kad automatiniai vertimai gali turėti klaidų ar netikslumų. Originalus dokumentas jo gimtąja kalba turėtų būti laikomas autoritetingu šaltiniu. Kritinei informacijai rekomenduojama naudoti profesionalų žmogaus vertimą. Mes neprisiimame atsakomybės už nesusipratimus ar klaidingus interpretavimus, atsiradusius dėl šio vertimo naudojimo.