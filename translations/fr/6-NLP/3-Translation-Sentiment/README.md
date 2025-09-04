<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "6396d5d8617572cd2ac1de74fb0deb22",
  "translation_date": "2025-09-04T00:49:36+00:00",
  "source_file": "6-NLP/3-Translation-Sentiment/README.md",
  "language_code": "fr"
}
-->
# Traduction et analyse de sentiment avec ML

Dans les leçons précédentes, vous avez appris à créer un bot basique en utilisant `TextBlob`, une bibliothèque qui intègre l'apprentissage automatique en coulisses pour effectuer des tâches NLP simples comme l'extraction de syntagmes nominaux. Un autre défi important en linguistique computationnelle est la _traduction_ précise d'une phrase d'une langue parlée ou écrite à une autre.

## [Quiz avant la leçon](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/35/)

La traduction est un problème très complexe, aggravé par le fait qu'il existe des milliers de langues, chacune ayant des règles grammaticales très différentes. Une approche consiste à convertir les règles grammaticales formelles d'une langue, comme l'anglais, en une structure indépendante de la langue, puis à les traduire en les convertissant dans une autre langue. Cette approche implique les étapes suivantes :

1. **Identification**. Identifier ou étiqueter les mots dans la langue source comme noms, verbes, etc.
2. **Créer une traduction**. Produire une traduction directe de chaque mot dans le format de la langue cible.

### Exemple de phrase, de l'anglais à l'irlandais

En 'anglais', la phrase _I feel happy_ contient trois mots dans l'ordre suivant :

- **sujet** (I)
- **verbe** (feel)
- **adjectif** (happy)

Cependant, en 'irlandais', la même phrase a une structure grammaticale très différente - les émotions comme "*happy*" ou "*sad*" sont exprimées comme étant *sur* vous.

La phrase anglaise `I feel happy` en irlandais serait `Tá athas orm`. Une traduction *littérale* serait `Happy is upon me`.

Un locuteur irlandais traduisant en anglais dirait `I feel happy`, et non `Happy is upon me`, car il comprend le sens de la phrase, même si les mots et la structure de la phrase sont différents.

L'ordre formel de la phrase en irlandais est :

- **verbe** (Tá ou is)
- **adjectif** (athas, ou happy)
- **sujet** (orm, ou upon me)

## Traduction

Un programme de traduction naïf pourrait traduire uniquement les mots, en ignorant la structure de la phrase.

✅ Si vous avez appris une deuxième (ou troisième ou plus) langue à l'âge adulte, vous avez peut-être commencé par penser dans votre langue maternelle, en traduisant un concept mot par mot dans votre tête vers la deuxième langue, puis en exprimant votre traduction. C'est similaire à ce que font les programmes de traduction naïfs. Il est important de dépasser cette phase pour atteindre la fluidité !

La traduction naïve conduit à des mauvaises (et parfois hilarantes) erreurs de traduction : `I feel happy` se traduit littéralement par `Mise bhraitheann athas` en irlandais. Cela signifie (littéralement) `me feel happy` et ce n'est pas une phrase valide en irlandais. Bien que l'anglais et l'irlandais soient des langues parlées sur deux îles voisines, ce sont des langues très différentes avec des structures grammaticales distinctes.

> Vous pouvez regarder des vidéos sur les traditions linguistiques irlandaises comme [celle-ci](https://www.youtube.com/watch?v=mRIaLSdRMMs)

### Approches par apprentissage automatique

Jusqu'à présent, vous avez appris l'approche des règles formelles pour le traitement du langage naturel. Une autre approche consiste à ignorer le sens des mots et _à utiliser l'apprentissage automatique pour détecter des motifs_. Cela peut fonctionner en traduction si vous disposez de beaucoup de texte (un *corpus*) ou de textes (*corpora*) dans les langues source et cible.

Par exemple, prenons le cas de *Orgueil et Préjugés*, un roman anglais bien connu écrit par Jane Austen en 1813. Si vous consultez le livre en anglais et une traduction humaine du livre en *français*, vous pourriez détecter des phrases dans l'un qui sont traduites _idiomatiquement_ dans l'autre. Vous allez le faire dans un instant.

Par exemple, lorsqu'une phrase anglaise comme `I have no money` est traduite littéralement en français, elle pourrait devenir `Je n'ai pas de monnaie`. "Monnaie" est un faux ami français délicat, car 'money' et 'monnaie' ne sont pas synonymes. Une meilleure traduction qu'un humain pourrait faire serait `Je n'ai pas d'argent`, car elle transmet mieux l'idée que vous n'avez pas d'argent (plutôt que 'petite monnaie', qui est le sens de 'monnaie').

![monnaie](../../../../translated_images/monnaie.606c5fa8369d5c3b3031ef0713e2069485c87985dd475cd9056bdf4c76c1f4b8.fr.png)

> Image par [Jen Looper](https://twitter.com/jenlooper)

Si un modèle ML dispose de suffisamment de traductions humaines pour construire un modèle, il peut améliorer la précision des traductions en identifiant des motifs communs dans des textes qui ont été précédemment traduits par des experts humains parlant les deux langues.

### Exercice - traduction

Vous pouvez utiliser `TextBlob` pour traduire des phrases. Essayez la célèbre première phrase de **Orgueil et Préjugés** :

```python
from textblob import TextBlob

blob = TextBlob(
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife!"
)
print(blob.translate(to="fr"))

```

`TextBlob` fait un très bon travail de traduction : "C'est une vérité universellement reconnue, qu'un homme célibataire en possession d'une bonne fortune doit avoir besoin d'une femme!". 

On peut dire que la traduction de TextBlob est bien plus exacte, en fait, que la traduction française de 1932 du livre par V. Leconte et Ch. Pressoir :

"C'est une vérité universelle qu'un célibataire pourvu d'une belle fortune doit avoir envie de se marier, et, si peu que l'on sache de son sentiment à cet égard, lorsqu'il arrive dans une nouvelle résidence, cette idée est si bien fixée dans l'esprit de ses voisins qu'ils le considèrent sur-le-champ comme la propriété légitime de l'une ou l'autre de leurs filles."

Dans ce cas, la traduction informée par ML fait un meilleur travail que le traducteur humain qui ajoute inutilement des mots dans la bouche de l'auteur original pour plus de 'clarté'.

> Pourquoi cela se produit-il ? Et pourquoi TextBlob est-il si bon en traduction ? Eh bien, en coulisses, il utilise Google Translate, une IA sophistiquée capable d'analyser des millions de phrases pour prédire les meilleures chaînes pour la tâche en cours. Il n'y a rien de manuel ici et vous avez besoin d'une connexion Internet pour utiliser `blob.translate`.

✅ Essayez d'autres phrases. Quelle est la meilleure, la traduction ML ou humaine ? Dans quels cas ?

## Analyse de sentiment

Un autre domaine où l'apprentissage automatique peut très bien fonctionner est l'analyse de sentiment. Une approche non-ML pour le sentiment consiste à identifier les mots et phrases qui sont 'positifs' et 'négatifs'. Ensuite, étant donné un nouveau texte, calculer la valeur totale des mots positifs, négatifs et neutres pour identifier le sentiment global. 

Cette approche est facilement trompée, comme vous avez pu le voir dans la tâche Marvin - la phrase `Great, that was a wonderful waste of time, I'm glad we are lost on this dark road` est une phrase sarcastique, à sentiment négatif, mais l'algorithme simple détecte 'great', 'wonderful', 'glad' comme positifs et 'waste', 'lost' et 'dark' comme négatifs. Le sentiment global est influencé par ces mots contradictoires.

✅ Arrêtez-vous un instant et réfléchissez à la façon dont nous transmettons le sarcasme en tant que locuteurs humains. L'inflexion du ton joue un rôle important. Essayez de dire la phrase "Well, that film was awesome" de différentes manières pour découvrir comment votre voix transmet le sens.

### Approches ML

L'approche ML consisterait à rassembler manuellement des corpus de textes négatifs et positifs - tweets, critiques de films, ou tout autre texte où l'humain a donné une note *et* une opinion écrite. Ensuite, des techniques NLP peuvent être appliquées aux opinions et aux notes, afin que des motifs émergent (par exemple, les critiques de films positives ont tendance à contenir la phrase 'Oscar worthy' plus que les critiques négatives, ou les critiques de restaurants positives disent 'gourmet' beaucoup plus que 'disgusting').

> ⚖️ **Exemple** : Si vous travailliez dans le bureau d'un politicien et qu'une nouvelle loi était débattue, les électeurs pourraient écrire au bureau avec des emails en faveur ou contre cette nouvelle loi. Disons que vous êtes chargé de lire les emails et de les trier en 2 piles, *pour* et *contre*. S'il y avait beaucoup d'emails, vous pourriez être submergé en essayant de tous les lire. Ne serait-il pas agréable qu'un bot puisse tous les lire pour vous, les comprendre et vous dire dans quelle pile chaque email appartient ? 
> 
> Une façon d'y parvenir est d'utiliser l'apprentissage automatique. Vous entraîneriez le modèle avec une partie des emails *contre* et une partie des emails *pour*. Le modèle aurait tendance à associer des phrases et des mots au côté contre et au côté pour, *mais il ne comprendrait aucun contenu*, seulement que certains mots et motifs étaient plus susceptibles d'apparaître dans un email *contre* ou *pour*. Vous pourriez le tester avec des emails que vous n'avez pas utilisés pour entraîner le modèle, et voir s'il arrive à la même conclusion que vous. Ensuite, une fois satisfait de la précision du modèle, vous pourriez traiter les emails futurs sans avoir à lire chacun d'eux.

✅ Ce processus vous semble-t-il similaire à des processus que vous avez utilisés dans des leçons précédentes ?

## Exercice - phrases sentimentales

Le sentiment est mesuré avec une *polarité* de -1 à 1, où -1 est le sentiment le plus négatif et 1 le plus positif. Le sentiment est également mesuré avec un score de 0 à 1 pour l'objectivité (0) et la subjectivité (1).

Reprenez *Orgueil et Préjugés* de Jane Austen. Le texte est disponible ici sur [Project Gutenberg](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm). L'exemple ci-dessous montre un court programme qui analyse le sentiment des premières et dernières phrases du livre et affiche sa polarité de sentiment et son score d'objectivité/subjectivité.

Vous devez utiliser la bibliothèque `TextBlob` (décrite ci-dessus) pour déterminer le `sentiment` (vous n'avez pas besoin d'écrire votre propre calculateur de sentiment) dans la tâche suivante.

```python
from textblob import TextBlob

quote1 = """It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife."""

quote2 = """Darcy, as well as Elizabeth, really loved them; and they were both ever sensible of the warmest gratitude towards the persons who, by bringing her into Derbyshire, had been the means of uniting them."""

sentiment1 = TextBlob(quote1).sentiment
sentiment2 = TextBlob(quote2).sentiment

print(quote1 + " has a sentiment of " + str(sentiment1))
print(quote2 + " has a sentiment of " + str(sentiment2))
```

Vous voyez le résultat suivant :

```output
It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want # of a wife. has a sentiment of Sentiment(polarity=0.20952380952380953, subjectivity=0.27142857142857146)

Darcy, as well as Elizabeth, really loved them; and they were
     both ever sensible of the warmest gratitude towards the persons
      who, by bringing her into Derbyshire, had been the means of
      uniting them. has a sentiment of Sentiment(polarity=0.7, subjectivity=0.8)
```

## Défi - vérifier la polarité du sentiment

Votre tâche est de déterminer, en utilisant la polarité du sentiment, si *Orgueil et Préjugés* contient plus de phrases absolument positives que de phrases absolument négatives. Pour cette tâche, vous pouvez supposer qu'un score de polarité de 1 ou -1 est absolument positif ou négatif respectivement.

**Étapes :**

1. Téléchargez une [copie d'Orgueil et Préjugés](https://www.gutenberg.org/files/1342/1342-h/1342-h.htm) depuis Project Gutenberg en tant que fichier .txt. Supprimez les métadonnées au début et à la fin du fichier, en ne laissant que le texte original.
2. Ouvrez le fichier en Python et extrayez le contenu sous forme de chaîne.
3. Créez un TextBlob en utilisant la chaîne du livre.
4. Analysez chaque phrase du livre dans une boucle.
   1. Si la polarité est 1 ou -1, stockez la phrase dans un tableau ou une liste de messages positifs ou négatifs.
5. À la fin, affichez toutes les phrases positives et négatives (séparément) et leur nombre respectif.

Voici une [solution exemple](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/3-Translation-Sentiment/solution/notebook.ipynb).

✅ Vérification des connaissances

1. Le sentiment est basé sur les mots utilisés dans la phrase, mais le code *comprend-il* les mots ?
2. Pensez-vous que la polarité du sentiment est précise, ou en d'autres termes, êtes-vous *d'accord* avec les scores ?
   1. En particulier, êtes-vous d'accord ou en désaccord avec la polarité **positive** absolue des phrases suivantes ?
      * “What an excellent father you have, girls!” said she, when the door was shut.
      * “Your examination of Mr. Darcy is over, I presume,” said Miss Bingley; “and pray what is the result?” “I am perfectly convinced by it that Mr. Darcy has no defect.
      * How wonderfully these sort of things occur!
      * I have the greatest dislike in the world to that sort of thing.
      * Charlotte is an excellent manager, I dare say.
      * “This is delightful indeed!
      * I am so happy!
      * Your idea of the ponies is delightful.
   2. Les 3 phrases suivantes ont été évaluées avec une polarité positive absolue, mais à une lecture attentive, elles ne sont pas des phrases positives. Pourquoi l'analyse de sentiment a-t-elle pensé qu'elles étaient des phrases positives ?
      * Happy shall I be, when his stay at Netherfield is over!” “I wish I could say anything to comfort you,” replied Elizabeth; “but it is wholly out of my power.
      * If I could but see you as happy!
      * Our distress, my dear Lizzy, is very great.
   3. Êtes-vous d'accord ou en désaccord avec la polarité **négative** absolue des phrases suivantes ?
      - Everybody is disgusted with his pride.
      - “I should like to know how he behaves among strangers.” “You shall hear then—but prepare yourself for something very dreadful.
      - The pause was to Elizabeth’s feelings dreadful.
      - It would be dreadful!

✅ Tout amateur de Jane Austen comprendra qu'elle utilise souvent ses livres pour critiquer les aspects les plus ridicules de la société anglaise de la Régence. Elizabeth Bennett, le personnage principal de *Orgueil et Préjugés*, est une observatrice sociale perspicace (comme l'auteur) et son langage est souvent très nuancé. Même M. Darcy (l'intérêt amoureux dans l'histoire) note l'utilisation ludique et taquine du langage par Elizabeth : "J'ai eu le plaisir de faire votre connaissance assez longtemps pour savoir que vous trouvez beaucoup de plaisir à professer occasionnellement des opinions qui en fait ne sont pas les vôtres."

---

## 🚀Défi

Pouvez-vous rendre Marvin encore meilleur en extrayant d'autres caractéristiques des entrées utilisateur ?

## [Quiz après la leçon](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/36/)

## Révision et étude personnelle
Il existe de nombreuses façons d'extraire le sentiment d'un texte. Pensez aux applications commerciales qui pourraient utiliser cette technique. Réfléchissez à la manière dont cela peut mal tourner. Lisez davantage sur les systèmes sophistiqués prêts pour l'entreprise qui analysent les sentiments, tels que [Azure Text Analysis](https://docs.microsoft.com/azure/cognitive-services/Text-Analytics/how-tos/text-analytics-how-to-sentiment-analysis?tabs=version-3-1?WT.mc_id=academic-77952-leestott). Testez certaines des phrases de "Orgueil et Préjugés" ci-dessus et voyez si elles peuvent détecter les nuances.

## Devoir 

[Licence poétique](assignment.md)

---

**Avertissement** :  
Ce document a été traduit à l'aide du service de traduction automatique [Co-op Translator](https://github.com/Azure/co-op-translator). Bien que nous nous efforcions d'assurer l'exactitude, veuillez noter que les traductions automatisées peuvent contenir des erreurs ou des inexactitudes. Le document original dans sa langue d'origine doit être considéré comme la source faisant autorité. Pour des informations critiques, il est recommandé de recourir à une traduction professionnelle réalisée par un humain. Nous déclinons toute responsabilité en cas de malentendus ou d'interprétations erronées résultant de l'utilisation de cette traduction.