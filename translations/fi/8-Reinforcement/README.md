<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "20ca019012b1725de956681d036d8b18",
  "translation_date": "2025-09-05T01:04:31+00:00",
  "source_file": "8-Reinforcement/README.md",
  "language_code": "fi"
}
-->
# Johdatus vahvistusoppimiseen

Vahvistusoppiminen, RL, nähdään yhtenä koneoppimisen perusparadigmoista, yhdessä ohjatun oppimisen ja ohjaamattoman oppimisen kanssa. RL keskittyy päätöksentekoon: oikeiden päätösten tekemiseen tai ainakin oppimiseen niistä.

Kuvittele, että sinulla on simuloitu ympäristö, kuten osakemarkkinat. Mitä tapahtuu, jos asetat tietyn sääntelyn? Onko sillä positiivinen vai negatiivinen vaikutus? Jos jotain negatiivista tapahtuu, sinun täytyy ottaa tämä _negatiivinen vahvistus_, oppia siitä ja muuttaa suuntaa. Jos tulos on positiivinen, sinun täytyy rakentaa sen _positiivisen vahvistuksen_ varaan.

![peter ja susi](../../../8-Reinforcement/images/peter.png)

> Peter ja hänen ystävänsä yrittävät paeta nälkäistä sutta! Kuva: [Jen Looper](https://twitter.com/jenlooper)

## Alueellinen aihe: Peter ja susi (Venäjä)

[Peter ja susi](https://en.wikipedia.org/wiki/Peter_and_the_Wolf) on venäläisen säveltäjän [Sergei Prokofjevin](https://en.wikipedia.org/wiki/Sergei_Prokofiev) kirjoittama musiikillinen satu. Se kertoo nuoresta pioneeri Peteristä, joka rohkeasti lähtee talostaan metsän aukealle jahtamaan sutta. Tässä osiossa koulutamme koneoppimisalgoritmeja, jotka auttavat Peteriä:

- **Tutkimaan** ympäröivää aluetta ja rakentamaan optimaalisen navigointikartan
- **Oppimaan** käyttämään skeittilautaa ja tasapainottamaan sillä, jotta hän voi liikkua nopeammin.

[![Peter ja susi](https://img.youtube.com/vi/Fmi5zHg4QSM/0.jpg)](https://www.youtube.com/watch?v=Fmi5zHg4QSM)

> 🎥 Klikkaa yllä olevaa kuvaa kuunnellaksesi Prokofjevin Peter ja susi

## Vahvistusoppiminen

Aiemmissa osioissa olet nähnyt kaksi esimerkkiä koneoppimisongelmista:

- **Ohjattu oppiminen**, jossa meillä on datakokonaisuuksia, jotka ehdottavat esimerkkiratkaisuja ongelmaan, jonka haluamme ratkaista. [Luokittelu](../4-Classification/README.md) ja [regressio](../2-Regression/README.md) ovat ohjatun oppimisen tehtäviä.
- **Ohjaamaton oppiminen**, jossa meillä ei ole merkittyä harjoitusdataa. Tärkein esimerkki ohjaamattomasta oppimisesta on [Klusterointi](../5-Clustering/README.md).

Tässä osiossa esittelemme uuden tyyppisen oppimisongelman, joka ei vaadi merkittyä harjoitusdataa. Tällaisia ongelmia on useita:

- **[Puoliohjattu oppiminen](https://wikipedia.org/wiki/Semi-supervised_learning)**, jossa meillä on paljon merkitsemätöntä dataa, jota voidaan käyttää mallin esikoulutukseen.
- **[Vahvistusoppiminen](https://wikipedia.org/wiki/Reinforcement_learning)**, jossa agentti oppii käyttäytymään tekemällä kokeita jossain simuloidussa ympäristössä.

### Esimerkki - tietokonepeli

Oletetaan, että haluat opettaa tietokoneen pelaamaan peliä, kuten shakkia tai [Super Mario](https://wikipedia.org/wiki/Super_Mario). Jotta tietokone voisi pelata peliä, sen täytyy ennustaa, mikä siirto tehdään kussakin pelitilanteessa. Vaikka tämä saattaa vaikuttaa luokitteluongelmalta, se ei ole sitä - koska meillä ei ole datakokonaisuutta, jossa olisi tilat ja vastaavat toiminnot. Vaikka meillä saattaisi olla dataa, kuten olemassa olevia shakkimatseja tai pelaajien Super Mario -pelitallenteita, on todennäköistä, että tämä data ei riittävästi kata suurta määrää mahdollisia tiloja.

Sen sijaan, että etsisimme olemassa olevaa pelidataa, **Vahvistusoppiminen** (RL) perustuu ideaan, että *tietokone pelaa* monta kertaa ja tarkkailee tulosta. Näin ollen vahvistusoppimisen soveltamiseen tarvitaan kaksi asiaa:

- **Ympäristö** ja **simulaattori**, jotka mahdollistavat pelin pelaamisen monta kertaa. Tämä simulaattori määrittelisi kaikki pelisäännöt sekä mahdolliset tilat ja toiminnot.

- **Palkintofunktio**, joka kertoo, kuinka hyvin suoriuduimme kunkin siirron tai pelin aikana.

Suurin ero muiden koneoppimisen tyyppien ja RL:n välillä on se, että RL:ssä emme yleensä tiedä, voitammeko vai häviämme, ennen kuin peli on ohi. Näin ollen emme voi sanoa, onko tietty siirto yksinään hyvä vai ei - saamme palkinnon vasta pelin lopussa. Tavoitteemme on suunnitella algoritmeja, jotka mahdollistavat mallin kouluttamisen epävarmoissa olosuhteissa. Opimme yhdestä RL-algoritmista nimeltä **Q-oppiminen**.

## Oppitunnit

1. [Johdatus vahvistusoppimiseen ja Q-oppimiseen](1-QLearning/README.md)
2. [Simulaatioympäristön käyttö Gymissä](2-Gym/README.md)

## Tekijät

"Johdatus vahvistusoppimiseen" on kirjoitettu ♥️:lla [Dmitry Soshnikov](http://soshnikov.com)

---

**Vastuuvapauslauseke**:  
Tämä asiakirja on käännetty käyttämällä tekoälypohjaista käännöspalvelua [Co-op Translator](https://github.com/Azure/co-op-translator). Vaikka pyrimme tarkkuuteen, huomioithan, että automaattiset käännökset voivat sisältää virheitä tai epätarkkuuksia. Alkuperäinen asiakirja sen alkuperäisellä kielellä tulisi pitää ensisijaisena lähteenä. Kriittisen tiedon osalta suositellaan ammattimaista ihmiskäännöstä. Emme ole vastuussa väärinkäsityksistä tai virhetulkinnoista, jotka johtuvat tämän käännöksen käytöstä.