# Quiz

Ces quiz sont les quiz pré- et post-conférence pour le programme ML sur https://aka.ms/ml-beginners

## Configuration du projet

```
npm install
```

### Compile et recharge à chaud pour le développement

```
npm run serve
```

### Compile et minifie pour la production

```
npm run build
```

### Lint et corrige les fichiers

```
npm run lint
```

### Personnaliser la configuration

Voir [Référence de configuration](https://cli.vuejs.org/config/).

Crédits : Merci à la version originale de cette application de quiz : https://github.com/arpan45/simple-quiz-vue

## Déploiement sur Azure

Voici un guide étape par étape pour vous aider à démarrer :

1. Forkez un dépôt GitHub  
Assurez-vous que le code de votre application web statique est dans votre dépôt GitHub. Forkez ce dépôt.

2. Créez une application web statique Azure  
- Créez un [compte Azure](http://azure.microsoft.com)  
- Allez sur le [portail Azure](https://portal.azure.com)  
- Cliquez sur « Créer une ressource » et recherchez « Application web statique ».  
- Cliquez sur « Créer ».

3. Configurez l'application web statique  
- Bases : Abonnement : Sélectionnez votre abonnement Azure.  
- Groupe de ressources : Créez un nouveau groupe de ressources ou utilisez un existant.  
- Nom : Fournissez un nom pour votre application web statique.  
- Région : Choisissez la région la plus proche de vos utilisateurs.

- #### Détails du déploiement :  
- Source : Sélectionnez « GitHub ».  
- Compte GitHub : Autorisez Azure à accéder à votre compte GitHub.  
- Organisation : Sélectionnez votre organisation GitHub.  
- Dépôt : Choisissez le dépôt contenant votre application web statique.  
- Branche : Sélectionnez la branche à partir de laquelle vous souhaitez déployer.

- #### Détails de la construction :  
- Préréglages de construction : Choisissez le framework avec lequel votre application est construite (par exemple, React, Angular, Vue, etc.).  
- Emplacement de l'application : Spécifiez le dossier contenant le code de votre application (par exemple, / s'il est à la racine).  
- Emplacement de l'API : Si vous avez une API, spécifiez son emplacement (optionnel).  
- Emplacement de sortie : Spécifiez le dossier où la sortie de la construction est générée (par exemple, build ou dist).

4. Examinez et créez  
Examinez vos paramètres et cliquez sur « Créer ». Azure mettra en place les ressources nécessaires et créera un workflow GitHub Actions dans votre dépôt.

5. Workflow GitHub Actions  
Azure créera automatiquement un fichier de workflow GitHub Actions dans votre dépôt (.github/workflows/azure-static-web-apps-<name>.yml). Ce workflow gérera le processus de construction et de déploiement.

6. Surveillez le déploiement  
Allez dans l'onglet « Actions » de votre dépôt GitHub.  
Vous devriez voir un workflow en cours d'exécution. Ce workflow construira et déploiera votre application web statique sur Azure.  
Une fois le workflow terminé, votre application sera en ligne à l'URL Azure fournie.

### Exemple de fichier de workflow

Voici un exemple de ce à quoi le fichier de workflow GitHub Actions pourrait ressembler :  
name: Azure Static Web Apps CI/CD  
```
on:
  push:
    branches:
      - main
  pull_request:
    types: [opened, synchronize, reopened, closed]
    branches:
      - main

jobs:
  build_and_deploy_job:
    runs-on: ubuntu-latest
    name: Build and Deploy Job
    steps:
      - uses: actions/checkout@v2
      - name: Build And Deploy
        id: builddeploy
        uses: Azure/static-web-apps-deploy@v1
        with:
          azure_static_web_apps_api_token: ${{ secrets.AZURE_STATIC_WEB_APPS_API_TOKEN }}
          repo_token: ${{ secrets.GITHUB_TOKEN }}
          action: "upload"
          app_location: "/quiz-app" # App source code path
          api_location: ""API source code path optional
          output_location: "dist" #Built app content directory - optional
```

### Ressources supplémentaires  
- [Documentation des applications web statiques Azure](https://learn.microsoft.com/azure/static-web-apps/getting-started)  
- [Documentation des actions GitHub](https://docs.github.com/actions/use-cases-and-examples/deploying/deploying-to-azure-static-web-app)

**Avertissement** :  
Ce document a été traduit à l'aide de services de traduction automatisée par IA. Bien que nous visons à garantir l'exactitude, veuillez noter que les traductions automatiques peuvent contenir des erreurs ou des inexactitudes. Le document original dans sa langue native doit être considéré comme la source autorisée. Pour des informations critiques, une traduction humaine professionnelle est recommandée. Nous ne sommes pas responsables des malentendus ou des interprétations erronées résultant de l'utilisation de cette traduction.