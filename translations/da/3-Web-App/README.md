<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9836ff53cfef716ddfd70e06c5f43436",
  "translation_date": "2025-09-05T00:36:28+00:00",
  "source_file": "3-Web-App/README.md",
  "language_code": "da"
}
-->
# Byg en webapp til at bruge din ML-model

I denne del af pensum vil du blive introduceret til et anvendt ML-emne: hvordan du gemmer din Scikit-learn-model som en fil, der kan bruges til at lave forudsigelser i en webapplikation. Når modellen er gemt, lærer du, hvordan du bruger den i en webapp bygget i Flask. Først opretter du en model ved hjælp af nogle data, der handler om UFO-observationer! Derefter bygger du en webapp, der giver dig mulighed for at indtaste et antal sekunder sammen med en bredde- og længdegradsværdi for at forudsige, hvilket land der rapporterede at have set en UFO.

![UFO Parkering](../../../3-Web-App/images/ufo.jpg)

Foto af <a href="https://unsplash.com/@mdherren?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Michael Herren</a> på <a href="https://unsplash.com/s/photos/ufo?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>

## Lektioner

1. [Byg en Webapp](1-Web-App/README.md)

## Krediteringer

"Byg en Webapp" blev skrevet med ♥️ af [Jen Looper](https://twitter.com/jenlooper).

♥️ Quizzerne blev skrevet af Rohan Raj.

Datasættet er hentet fra [Kaggle](https://www.kaggle.com/NUFORC/ufo-sightings).

Webapp-arkitekturen blev delvist foreslået af [denne artikel](https://towardsdatascience.com/how-to-easily-deploy-machine-learning-models-using-flask-b95af8fe34d4) og [dette repo](https://github.com/abhinavsagar/machine-learning-deployment) af Abhinav Sagar.

---

**Ansvarsfraskrivelse**:  
Dette dokument er blevet oversat ved hjælp af AI-oversættelsestjenesten [Co-op Translator](https://github.com/Azure/co-op-translator). Selvom vi bestræber os på nøjagtighed, skal du være opmærksom på, at automatiserede oversættelser kan indeholde fejl eller unøjagtigheder. Det originale dokument på dets oprindelige sprog bør betragtes som den autoritative kilde. For kritisk information anbefales professionel menneskelig oversættelse. Vi er ikke ansvarlige for eventuelle misforståelser eller fejltolkninger, der måtte opstå som følge af brugen af denne oversættelse.