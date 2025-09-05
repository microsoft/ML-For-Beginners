<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "68394b2102d3503882e5e914bd0ff5c1",
  "translation_date": "2025-09-05T01:13:23+00:00",
  "source_file": "8-Reinforcement/1-QLearning/assignment.md",
  "language_code": "fi"
}
-->
# Realistisempi maailma

Meidän tilanteessamme Peter pystyi liikkumaan lähes väsymättä tai tuntematta nälkää. Realistisemmassa maailmassa hänen täytyy välillä istua alas ja levätä, sekä syödä jotain. Tehdään maailmastamme realistisempi toteuttamalla seuraavat säännöt:

1. Liikkuessaan paikasta toiseen Peter menettää **energiaa** ja kerää **väsymystä**.
2. Peter voi saada lisää energiaa syömällä omenoita.
3. Peter voi päästä eroon väsymyksestä lepäämällä puun alla tai ruohikolla (eli siirtymällä pelilaudan ruutuun, jossa on puu tai ruoho - vihreä kenttä).
4. Peterin täytyy löytää ja tappaa susi.
5. Jotta Peter voi tappaa suden, hänen täytyy olla tietyllä energian ja väsymyksen tasolla, muuten hän häviää taistelun.

## Ohjeet

Käytä alkuperäistä [notebook.ipynb](../../../../8-Reinforcement/1-QLearning/notebook.ipynb) -tiedostoa ratkaisusi lähtökohtana.

Muokkaa yllä olevaa palkkiofunktiota pelin sääntöjen mukaisesti, suorita vahvistusoppimisalgoritmi löytääksesi parhaan strategian pelin voittamiseen, ja vertaa satunnaiskävelyn tuloksia algoritmisi tuloksiin pelien voittamisen ja häviämisen osalta.

> **Note**: Uudessa maailmassasi tila on monimutkaisempi, ja ihmisen sijainnin lisäksi siihen sisältyy myös väsymyksen ja energian tasot. Voit valita, että esität tilan tuple-muodossa (Board,energy,fatigue), tai määritellä tilalle luokan (voit myös halutessasi johdattaa sen `Board`-luokasta), tai jopa muokata alkuperäistä `Board`-luokkaa tiedostossa [rlboard.py](../../../../8-Reinforcement/1-QLearning/rlboard.py).

Ratkaisussasi säilytä satunnaiskävelystrategiasta vastaava koodi ja vertaa algoritmisi tuloksia satunnaiskävelyyn lopuksi.

> **Note**: Saatat joutua säätämään hyperparametreja, jotta algoritmi toimii, erityisesti epochien määrää. Koska pelin onnistuminen (suden voittaminen) on harvinainen tapahtuma, voit odottaa paljon pidempää koulutusaikaa.

## Arviointikriteerit

| Kriteeri | Erinomainen                                                                                                                                                                                             | Riittävä                                                                                                                                                                                | Parannettavaa                                                                                                                          |
| -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
|          | Notebook sisältää uuden maailman sääntöjen määrittelyn, Q-Learning-algoritmin ja tekstuaalisia selityksiä. Q-Learning pystyy merkittävästi parantamaan tuloksia verrattuna satunnaiskävelyyn.           | Notebook on esitetty, Q-Learning on toteutettu ja parantaa tuloksia verrattuna satunnaiskävelyyn, mutta ei merkittävästi; tai notebook on huonosti dokumentoitu ja koodi ei ole hyvin jäsennelty | Yrityksiä maailman sääntöjen uudelleenmäärittelyyn on tehty, mutta Q-Learning-algoritmi ei toimi, tai palkkiofunktiota ei ole täysin määritelty |

---

**Vastuuvapauslauseke**:  
Tämä asiakirja on käännetty käyttämällä tekoälypohjaista käännöspalvelua [Co-op Translator](https://github.com/Azure/co-op-translator). Vaikka pyrimme tarkkuuteen, huomioithan, että automaattiset käännökset voivat sisältää virheitä tai epätarkkuuksia. Alkuperäinen asiakirja sen alkuperäisellä kielellä tulisi pitää ensisijaisena lähteenä. Kriittisen tiedon osalta suositellaan ammattimaista ihmiskäännöstä. Emme ole vastuussa väärinkäsityksistä tai virhetulkinnoista, jotka johtuvat tämän käännöksen käytöstä.