<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "1f2b7441745eb52e25745423b247016b",
  "translation_date": "2025-09-05T01:19:02+00:00",
  "source_file": "8-Reinforcement/2-Gym/assignment.md",
  "language_code": "fi"
}
-->
# Kouluta Mountain Car

[OpenAI Gym](http://gym.openai.com) on suunniteltu siten, että kaikki ympäristöt tarjoavat saman API:n - eli samat metodit `reset`, `step` ja `render`, sekä samat abstraktiot **toimintatilasta** ja **havaintotilasta**. Näin ollen pitäisi olla mahdollista soveltaa samoja vahvistusoppimisalgoritmeja eri ympäristöihin vähäisin koodimuutoksin.

## Mountain Car -ympäristö

[Mountain Car -ympäristö](https://gym.openai.com/envs/MountainCar-v0/) sisältää auton, joka on jumissa laaksossa:

Tavoitteena on päästä ulos laaksosta ja napata lippu tekemällä jokaisella askeleella yksi seuraavista toimista:

| Arvo | Merkitys |
|---|---|
| 0 | Kiihdytä vasemmalle |
| 1 | Älä kiihdytä |
| 2 | Kiihdytä oikealle |

Tämän ongelman päätemppu on kuitenkin se, että auton moottori ei ole tarpeeksi voimakas kiivetäkseen vuoren huipulle yhdellä kerralla. Siksi ainoa tapa onnistua on ajaa edestakaisin keräten vauhtia.

Havaintotila koostuu vain kahdesta arvosta:

| Num | Havainto       | Min   | Max   |
|-----|---------------|-------|-------|
|  0  | Auton sijainti | -1.2  | 0.6   |
|  1  | Auton nopeus   | -0.07 | 0.07  |

Mountain Car -ympäristön palkkiojärjestelmä on melko haastava:

 * Palkkio 0 annetaan, jos agentti saavuttaa lipun (sijainti = 0.5) vuoren huipulla.
 * Palkkio -1 annetaan, jos agentin sijainti on alle 0.5.

Episodi päättyy, jos auton sijainti on yli 0.5 tai episodin pituus ylittää 200.

## Ohjeet

Sovella vahvistusoppimisalgoritmiamme ratkaistaksesi Mountain Car -ongelman. Aloita olemassa olevasta [notebook.ipynb](../../../../8-Reinforcement/2-Gym/notebook.ipynb) -koodista, korvaa uusi ympäristö, muuta tilan diskretisointifunktioita ja yritä saada olemassa oleva algoritmi toimimaan vähäisin koodimuutoksin. Optimoi tulos säätämällä hyperparametreja.

> **Huom**: Hyperparametrien säätöä tarvitaan todennäköisesti, jotta algoritmi konvergoituu.

## Arviointikriteerit

| Kriteeri | Erinomainen | Riittävä | Parannettavaa |
| -------- | ----------- | -------- | ------------- |
|          | Q-Learning -algoritmi on onnistuneesti sovitettu CartPole-esimerkistä vähäisin koodimuutoksin, ja se pystyy ratkaisemaan lipun nappaamisen alle 200 askeleessa. | Uusi Q-Learning -algoritmi on otettu käyttöön Internetistä, mutta se on hyvin dokumentoitu; tai olemassa oleva algoritmi on sovitettu, mutta ei saavuta toivottuja tuloksia. | Opiskelija ei ole onnistunut soveltamaan mitään algoritmia, mutta on tehnyt merkittäviä edistysaskeleita ratkaisun suuntaan (toteuttanut tilan diskretisoinnin, Q-Table -tietorakenteen jne.). |

---

**Vastuuvapauslauseke**:  
Tämä asiakirja on käännetty käyttämällä tekoälypohjaista käännöspalvelua [Co-op Translator](https://github.com/Azure/co-op-translator). Vaikka pyrimme tarkkuuteen, huomioithan, että automaattiset käännökset voivat sisältää virheitä tai epätarkkuuksia. Alkuperäinen asiakirja sen alkuperäisellä kielellä tulisi pitää ensisijaisena lähteenä. Kriittisen tiedon osalta suositellaan ammattimaista ihmiskäännöstä. Emme ole vastuussa väärinkäsityksistä tai virhetulkinnoista, jotka johtuvat tämän käännöksen käytöstä.