<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "b28a3a4911584062772c537b653ebbc7",
  "translation_date": "2025-10-11T12:05:01+00:00",
  "source_file": "5-Clustering/README.md",
  "language_code": "et"
}
-->
# Klasterdamise mudelid masinõppes

Klasterdamine on masinõppe ülesanne, mille eesmärk on leida objekte, mis sarnanevad üksteisele, ja rühmitada need klastriteks. Mis eristab klasterdamist teistest masinõppe lähenemistest, on see, et protsess toimub automaatselt – tegelikult võib öelda, et see on vastand juhendatud õppimisele.

## Regionaalne teema: klasterdamise mudelid Nigeeria publiku muusikamaitse jaoks 🎧

Nigeeria mitmekesine publik eelistab mitmekesist muusikat. Kasutades Spotifyst kogutud andmeid (inspireerituna [sellest artiklist](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421)), vaatame mõningaid Nigeerias populaarseid lugusid. See andmestik sisaldab teavet erinevate laulude kohta, nagu nende 'tantsitavuse' skoor, 'akustilisus', valjus, 'kõnelemise' määr, populaarsus ja energia. On huvitav avastada mustreid nendes andmetes!

![Plaadimängija](../../../translated_images/turntable.f2b86b13c53302dc106aa741de9dc96ac372864cf458dd6f879119857aab01da.et.jpg)

> Foto autor <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Marcela Laskoski</a> lehel <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
Selles õppetundide sarjas avastate uusi viise andmete analüüsimiseks klasterdamistehnikate abil. Klasterdamine on eriti kasulik, kui teie andmestikul puuduvad sildid. Kui andmestikul on sildid, siis võivad klassifitseerimistehnikad, mida õppisite eelnevates tundides, olla kasulikumad. Kuid juhtudel, kus soovite rühmitada sildistamata andmeid, on klasterdamine suurepärane viis mustrite avastamiseks.

> On olemas kasulikke vähese koodikirjutamisega tööriistu, mis aitavad teil klasterdamise mudelitega töötamist õppida. Proovige [Azure ML selleks ülesandeks](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

## Õppetunnid

1. [Sissejuhatus klasterdamisse](1-Visualize/README.md)
2. [K-Means klasterdamine](2-K-Means/README.md)

## Autorid

Need õppetunnid on kirjutatud 🎶 poolt [Jen Looper](https://www.twitter.com/jenlooper) koos kasulike ülevaadetega [Rishit Dagli](https://rishit_dagli) ja [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan) poolt.

[Nigeeria laulude](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) andmestik pärineb Kaggle'ist ja on kogutud Spotifyst.

Kasulikud K-Meansi näited, mis aitasid selle õppetunni loomisel, hõlmavad seda [iiriste analüüsi](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering), seda [sissejuhatavat märkmikku](https://www.kaggle.com/prashant111/k-means-clustering-with-python) ja seda [hüpoteetilist MTÜ näidet](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering).

---

**Vastutusest loobumine**:  
See dokument on tõlgitud AI tõlketeenuse [Co-op Translator](https://github.com/Azure/co-op-translator) abil. Kuigi püüame tagada täpsust, palume arvestada, et automaatsed tõlked võivad sisaldada vigu või ebatäpsusi. Algne dokument selle algses keeles tuleks pidada autoriteetseks allikaks. Olulise teabe puhul soovitame kasutada professionaalset inimtõlget. Me ei vastuta selle tõlke kasutamisest tulenevate arusaamatuste või valesti tõlgenduste eest.