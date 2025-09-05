<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "5f3cb462e3122e1afe7ab0050ccf2bd3",
  "translation_date": "2025-09-04T23:06:17+00:00",
  "source_file": "6-NLP/2-Tasks/README.md",
  "language_code": "fr"
}
-->
# Tâches et techniques courantes en traitement du langage naturel

Pour la plupart des tâches de *traitement du langage naturel*, le texte à traiter doit être décomposé, analysé, et les résultats doivent être stockés ou croisés avec des règles et des ensembles de données. Ces tâches permettent au programmeur de déduire le _sens_, l'_intention_ ou simplement la _fréquence_ des termes et des mots dans un texte.

## [Quiz avant le cours](https://ff-quizzes.netlify.app/en/ml/)

Découvrons les techniques courantes utilisées pour traiter le texte. Combinées avec l'apprentissage automatique, ces techniques vous aident à analyser efficacement de grandes quantités de texte. Avant d'appliquer l'IA à ces tâches, comprenons les problèmes rencontrés par un spécialiste du NLP.

## Tâches courantes en NLP

Il existe différentes façons d'analyser un texte sur lequel vous travaillez. Il y a des tâches que vous pouvez effectuer et, grâce à ces tâches, vous pouvez comprendre le texte et en tirer des conclusions. Ces tâches sont généralement réalisées dans un ordre précis.

### Tokenisation

La première étape que la plupart des algorithmes de NLP doivent effectuer est probablement de diviser le texte en tokens, ou mots. Bien que cela semble simple, tenir compte de la ponctuation et des délimiteurs de mots et de phrases propres à chaque langue peut rendre cette tâche complexe. Vous pourriez avoir besoin de différentes méthodes pour déterminer les délimitations.

![tokenisation](../../../../6-NLP/2-Tasks/images/tokenization.png)
> Tokenisation d'une phrase tirée de **Pride and Prejudice**. Infographie par [Jen Looper](https://twitter.com/jenlooper)

### Embeddings

[Les embeddings de mots](https://wikipedia.org/wiki/Word_embedding) sont une méthode pour convertir vos données textuelles en valeurs numériques. Les embeddings sont réalisés de manière à ce que les mots ayant un sens similaire ou utilisés ensemble se regroupent.

![embeddings de mots](../../../../6-NLP/2-Tasks/images/embedding.png)
> "I have the highest respect for your nerves, they are my old friends." - Embeddings de mots pour une phrase tirée de **Pride and Prejudice**. Infographie par [Jen Looper](https://twitter.com/jenlooper)

✅ Essayez [cet outil intéressant](https://projector.tensorflow.org/) pour expérimenter avec les embeddings de mots. En cliquant sur un mot, vous verrez des regroupements de mots similaires : 'toy' se regroupe avec 'disney', 'lego', 'playstation', et 'console'.

### Analyse syntaxique et étiquetage des parties du discours

Chaque mot qui a été tokenisé peut être étiqueté comme une partie du discours - un nom, un verbe ou un adjectif. La phrase `the quick red fox jumped over the lazy brown dog` pourrait être étiquetée comme suit : fox = nom, jumped = verbe.

![analyse syntaxique](../../../../6-NLP/2-Tasks/images/parse.png)

> Analyse syntaxique d'une phrase tirée de **Pride and Prejudice**. Infographie par [Jen Looper](https://twitter.com/jenlooper)

L'analyse syntaxique consiste à reconnaître quels mots sont liés les uns aux autres dans une phrase - par exemple, `the quick red fox jumped` est une séquence adjectif-nom-verbe distincte de la séquence `lazy brown dog`.

### Fréquences des mots et des phrases

Une procédure utile lors de l'analyse d'un grand corpus de texte est de construire un dictionnaire de chaque mot ou phrase d'intérêt et de la fréquence à laquelle il apparaît. La phrase `the quick red fox jumped over the lazy brown dog` a une fréquence de 2 pour le mot "the".

Prenons un exemple de texte où nous comptons la fréquence des mots. Le poème The Winners de Rudyard Kipling contient le vers suivant :

```output
What the moral? Who rides may read.
When the night is thick and the tracks are blind
A friend at a pinch is a friend, indeed,
But a fool to wait for the laggard behind.
Down to Gehenna or up to the Throne,
He travels the fastest who travels alone.
```

Comme les fréquences des phrases peuvent être sensibles ou non à la casse selon les besoins, la phrase `a friend` a une fréquence de 2, `the` a une fréquence de 6, et `travels` une fréquence de 2.

### N-grams

Un texte peut être divisé en séquences de mots d'une longueur définie : un seul mot (unigramme), deux mots (bigrammes), trois mots (trigrammes) ou tout autre nombre de mots (n-grams).

Par exemple, `the quick red fox jumped over the lazy brown dog` avec un score de n-gram de 2 produit les n-grams suivants :

1. the quick 
2. quick red 
3. red fox
4. fox jumped 
5. jumped over 
6. over the 
7. the lazy 
8. lazy brown 
9. brown dog

Il peut être plus facile de le visualiser comme une fenêtre glissante sur la phrase. Voici un exemple pour des n-grams de 3 mots, le n-gram est en gras dans chaque phrase :

1.   <u>**the quick red**</u> fox jumped over the lazy brown dog
2.   the **<u>quick red fox</u>** jumped over the lazy brown dog
3.   the quick **<u>red fox jumped</u>** over the lazy brown dog
4.   the quick red **<u>fox jumped over</u>** the lazy brown dog
5.   the quick red fox **<u>jumped over the</u>** lazy brown dog
6.   the quick red fox jumped **<u>over the lazy</u>** brown dog
7.   the quick red fox jumped over <u>**the lazy brown**</u> dog
8.   the quick red fox jumped over the **<u>lazy brown dog</u>**

![fenêtre glissante des n-grams](../../../../6-NLP/2-Tasks/images/n-grams.gif)

> Valeur de n-gram de 3 : Infographie par [Jen Looper](https://twitter.com/jenlooper)

### Extraction de syntagmes nominaux

Dans la plupart des phrases, il y a un nom qui est le sujet ou l'objet de la phrase. En anglais, il est souvent identifiable par la présence de 'a', 'an' ou 'the' avant lui. Identifier le sujet ou l'objet d'une phrase en "extrayant le syntagme nominal" est une tâche courante en NLP lorsqu'on tente de comprendre le sens d'une phrase.

✅ Dans la phrase "I cannot fix on the hour, or the spot, or the look or the words, which laid the foundation. It is too long ago. I was in the middle before I knew that I had begun.", pouvez-vous identifier les syntagmes nominaux ?

Dans la phrase `the quick red fox jumped over the lazy brown dog`, il y a 2 syntagmes nominaux : **quick red fox** et **lazy brown dog**.

### Analyse de sentiment

Une phrase ou un texte peut être analysé pour déterminer son sentiment, c'est-à-dire à quel point il est *positif* ou *négatif*. Le sentiment est mesuré en termes de *polarité* et d'*objectivité/subjectivité*. La polarité est mesurée de -1.0 à 1.0 (négatif à positif) et de 0.0 à 1.0 (le plus objectif au plus subjectif).

✅ Plus tard, vous apprendrez qu'il existe différentes façons de déterminer le sentiment en utilisant l'apprentissage automatique, mais une méthode consiste à avoir une liste de mots et de phrases catégorisés comme positifs ou négatifs par un expert humain et à appliquer ce modèle au texte pour calculer un score de polarité. Pouvez-vous voir comment cela fonctionnerait dans certains cas et moins bien dans d'autres ?

### Inflection

L'inflexion vous permet de prendre un mot et d'obtenir sa forme singulière ou plurielle.

### Lemmatisation

Un *lemme* est la racine ou le mot de base d'un ensemble de mots, par exemple *flew*, *flies*, *flying* ont pour lemme le verbe *fly*.

Il existe également des bases de données utiles pour les chercheurs en NLP, notamment :

### WordNet

[WordNet](https://wordnet.princeton.edu/) est une base de données de mots, synonymes, antonymes et de nombreux autres détails pour chaque mot dans de nombreuses langues différentes. Elle est incroyablement utile lorsqu'on tente de créer des traductions, des correcteurs orthographiques ou des outils linguistiques de tout type.

## Bibliothèques NLP

Heureusement, vous n'avez pas besoin de construire toutes ces techniques vous-même, car il existe d'excellentes bibliothèques Python qui rendent le NLP beaucoup plus accessible aux développeurs qui ne sont pas spécialisés en traitement du langage naturel ou en apprentissage automatique. Les prochaines leçons incluent davantage d'exemples, mais ici vous apprendrez quelques exemples utiles pour vous aider dans la prochaine tâche.

### Exercice - utiliser la bibliothèque `TextBlob`

Utilisons une bibliothèque appelée TextBlob, car elle contient des API utiles pour aborder ces types de tâches. TextBlob "s'appuie sur les épaules des géants [NLTK](https://nltk.org) et [pattern](https://github.com/clips/pattern), et fonctionne bien avec les deux." Elle intègre une quantité considérable d'IA dans son API.

> Note : Un [guide de démarrage rapide](https://textblob.readthedocs.io/en/dev/quickstart.html#quickstart) utile est disponible pour TextBlob et est recommandé pour les développeurs Python expérimentés.

Lorsqu'on tente d'identifier des *syntagmes nominaux*, TextBlob offre plusieurs options d'extracteurs pour trouver ces syntagmes.

1. Regardez `ConllExtractor`.

    ```python
    from textblob import TextBlob
    from textblob.np_extractors import ConllExtractor
    # import and create a Conll extractor to use later 
    extractor = ConllExtractor()
    
    # later when you need a noun phrase extractor:
    user_input = input("> ")
    user_input_blob = TextBlob(user_input, np_extractor=extractor)  # note non-default extractor specified
    np = user_input_blob.noun_phrases                                    
    ```

    > Que se passe-t-il ici ? [ConllExtractor](https://textblob.readthedocs.io/en/dev/api_reference.html?highlight=Conll#textblob.en.np_extractors.ConllExtractor) est "un extracteur de syntagmes nominaux qui utilise l'analyse par chunking entraînée avec le corpus d'entraînement ConLL-2000." ConLL-2000 fait référence à la Conférence de 2000 sur l'apprentissage automatique du langage naturel. Chaque année, la conférence organisait un atelier pour résoudre un problème complexe de NLP, et en 2000, il s'agissait du chunking des syntagmes nominaux. Un modèle a été entraîné sur le Wall Street Journal, avec "les sections 15-18 comme données d'entraînement (211727 tokens) et la section 20 comme données de test (47377 tokens)". Vous pouvez consulter les procédures utilisées [ici](https://www.clips.uantwerpen.be/conll2000/chunking/) et les [résultats](https://ifarm.nl/erikt/research/np-chunking.html).

### Défi - améliorer votre bot avec le NLP

Dans la leçon précédente, vous avez construit un bot de questions-réponses très simple. Maintenant, vous allez rendre Marvin un peu plus sympathique en analysant votre entrée pour le sentiment et en imprimant une réponse adaptée au sentiment. Vous devrez également identifier un `syntagme_nominal` et poser une question à ce sujet.

Vos étapes pour construire un bot conversationnel amélioré :

1. Imprimez des instructions conseillant à l'utilisateur comment interagir avec le bot
2. Démarrez une boucle 
   1. Acceptez l'entrée utilisateur
   2. Si l'utilisateur demande à quitter, alors quittez
   3. Traitez l'entrée utilisateur et déterminez une réponse sentimentale appropriée
   4. Si un syntagme nominal est détecté dans le sentiment, mettez-le au pluriel et demandez plus d'informations à ce sujet
   5. Imprimez la réponse
3. Revenez à l'étape 2

Voici un extrait de code pour déterminer le sentiment en utilisant TextBlob. Notez qu'il n'y a que quatre *gradients* de réponse sentimentale (vous pourriez en avoir plus si vous le souhaitez) :

```python
if user_input_blob.polarity <= -0.5:
  response = "Oh dear, that sounds bad. "
elif user_input_blob.polarity <= 0:
  response = "Hmm, that's not great. "
elif user_input_blob.polarity <= 0.5:
  response = "Well, that sounds positive. "
elif user_input_blob.polarity <= 1:
  response = "Wow, that sounds great. "
```

Voici un exemple de sortie pour vous guider (l'entrée utilisateur est sur les lignes commençant par >) :

```output
Hello, I am Marvin, the friendly robot.
You can end this conversation at any time by typing 'bye'
After typing each answer, press 'enter'
How are you today?
> I am ok
Well, that sounds positive. Can you tell me more?
> I went for a walk and saw a lovely cat
Well, that sounds positive. Can you tell me more about lovely cats?
> cats are the best. But I also have a cool dog
Wow, that sounds great. Can you tell me more about cool dogs?
> I have an old hounddog but he is sick
Hmm, that's not great. Can you tell me more about old hounddogs?
> bye
It was nice talking to you, goodbye!
```

Une solution possible à la tâche est [ici](https://github.com/microsoft/ML-For-Beginners/blob/main/6-NLP/2-Tasks/solution/bot.py)

✅ Vérification des connaissances

1. Pensez-vous que les réponses sympathiques pourraient "tromper" quelqu'un en lui faisant croire que le bot le comprend réellement ?
2. Identifier le syntagme nominal rend-il le bot plus "crédible" ?
3. Pourquoi l'extraction d'un "syntagme nominal" d'une phrase serait-elle utile ?

---

Implémentez le bot dans la vérification des connaissances précédente et testez-le sur un ami. Peut-il les tromper ? Pouvez-vous rendre votre bot plus "crédible" ?

## 🚀Défi

Prenez une tâche dans la vérification des connaissances précédente et essayez de l'implémenter. Testez le bot sur un ami. Peut-il les tromper ? Pouvez-vous rendre votre bot plus "crédible" ?

## [Quiz après le cours](https://ff-quizzes.netlify.app/en/ml/)

## Révision et auto-apprentissage

Dans les prochaines leçons, vous en apprendrez davantage sur l'analyse de sentiment. Faites des recherches sur cette technique intéressante dans des articles tels que ceux sur [KDNuggets](https://www.kdnuggets.com/tag/nlp)

## Devoir 

[Faire parler un bot](assignment.md)

---

**Avertissement** :  
Ce document a été traduit à l'aide du service de traduction automatique [Co-op Translator](https://github.com/Azure/co-op-translator). Bien que nous nous efforcions d'assurer l'exactitude, veuillez noter que les traductions automatisées peuvent contenir des erreurs ou des inexactitudes. Le document original dans sa langue d'origine doit être considéré comme la source faisant autorité. Pour des informations critiques, il est recommandé de faire appel à une traduction professionnelle humaine. Nous déclinons toute responsabilité en cas de malentendus ou d'interprétations erronées résultant de l'utilisation de cette traduction.