# Contribuer en traduisant des leçons

Nous accueillons les traductions des leçons de ce programme !
## Directives

Il y a des dossiers dans chaque dossier de leçon et dans le dossier d'introduction aux leçons qui contiennent les fichiers markdown traduits.

> Remarque, veuillez ne pas traduire de code dans les fichiers d'exemple de code ; les seules choses à traduire sont le README, les devoirs et les quiz. Merci !

Les fichiers traduits doivent suivre cette convention de nommage :

**README._[langue]_.md**

où _[langue]_ est une abréviation de deux lettres suivant la norme ISO 639-1 (par exemple `README.es.md` pour l'espagnol et `README.nl.md` pour le néerlandais).

**assignment._[langue]_.md**

Comme pour les README, veuillez également traduire les devoirs.

> Important : lorsque vous traduisez du texte dans ce dépôt, veuillez vous assurer de ne pas utiliser de traduction automatique. Nous vérifierons les traductions via la communauté, donc veuillez ne vous porter volontaire pour des traductions que dans les langues où vous êtes compétent.

**Quiz**

1. Ajoutez votre traduction à l'application quiz en ajoutant un fichier ici : https://github.com/microsoft/ML-For-Beginners/tree/main/quiz-app/src/assets/translations, avec la convention de nommage appropriée (en.json, fr.json). **Veuillez ne pas localiser les mots 'true' ou 'false', cependant. merci !**

2. Ajoutez votre code de langue dans le menu déroulant du fichier App.vue de l'application quiz.

3. Modifiez le fichier [translations index.js de l'application quiz](https://github.com/microsoft/ML-For-Beginners/blob/main/quiz-app/src/assets/translations/index.js) pour ajouter votre langue.

4. Enfin, modifiez TOUS les liens de quiz dans vos fichiers README.md traduits pour pointer directement vers votre quiz traduit : https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/1 devient https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/1?loc=id

**MERCI**

Nous apprécions vraiment vos efforts !

**Avertissement** :  
Ce document a été traduit à l'aide de services de traduction automatique basés sur l'IA. Bien que nous nous efforçons d'assurer l'exactitude, veuillez noter que les traductions automatisées peuvent contenir des erreurs ou des inexactitudes. Le document original dans sa langue native doit être considéré comme la source autorisée. Pour des informations critiques, une traduction humaine professionnelle est recommandée. Nous ne sommes pas responsables des malentendus ou des erreurs d'interprétation résultant de l'utilisation de cette traduction.