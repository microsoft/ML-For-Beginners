<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9836ff53cfef716ddfd70e06c5f43436",
  "translation_date": "2025-09-05T16:12:10+00:00",
  "source_file": "3-Web-App/README.md",
  "language_code": "sk"
}
-->
# Vytvorte webovú aplikáciu na použitie vášho ML modelu

V tejto časti kurzu sa zoznámite s praktickou témou strojového učenia: ako uložiť váš Scikit-learn model ako súbor, ktorý môže byť použitý na predikcie v rámci webovej aplikácie. Keď je model uložený, naučíte sa, ako ho použiť v webovej aplikácii postavenej vo Flasku. Najskôr vytvoríte model pomocou dát, ktoré sa týkajú pozorovaní UFO! Potom vytvoríte webovú aplikáciu, ktorá vám umožní zadať počet sekúnd spolu s hodnotami zemepisnej šírky a dĺžky na predpovedanie, ktorá krajina nahlásila pozorovanie UFO.

![UFO Parkovanie](../../../3-Web-App/images/ufo.jpg)

Foto od <a href="https://unsplash.com/@mdherren?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Michael Herren</a> na <a href="https://unsplash.com/s/photos/ufo?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>

## Lekcie

1. [Vytvorte webovú aplikáciu](1-Web-App/README.md)

## Kredity

"Vytvorte webovú aplikáciu" bolo napísané s ♥️ od [Jen Looper](https://twitter.com/jenlooper).

♥️ Kvízy boli napísané Rohanom Rajom.

Dataset pochádza z [Kaggle](https://www.kaggle.com/NUFORC/ufo-sightings).

Architektúra webovej aplikácie bola čiastočne navrhnutá podľa [tohto článku](https://towardsdatascience.com/how-to-easily-deploy-machine-learning-models-using-flask-b95af8fe34d4) a [tohto repozitára](https://github.com/abhinavsagar/machine-learning-deployment) od Abhinava Sagara.

---

**Upozornenie**:  
Tento dokument bol preložený pomocou služby AI prekladu [Co-op Translator](https://github.com/Azure/co-op-translator). Aj keď sa snažíme o presnosť, prosím, berte na vedomie, že automatizované preklady môžu obsahovať chyby alebo nepresnosti. Pôvodný dokument v jeho rodnom jazyku by mal byť považovaný za autoritatívny zdroj. Pre kritické informácie sa odporúča profesionálny ľudský preklad. Nie sme zodpovední za akékoľvek nedorozumenia alebo nesprávne interpretácie vyplývajúce z použitia tohto prekladu.