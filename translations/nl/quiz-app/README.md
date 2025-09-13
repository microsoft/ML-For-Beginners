<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "6d130dffca5db70d7e615f926cb1ad4c",
  "translation_date": "2025-09-05T19:47:13+00:00",
  "source_file": "quiz-app/README.md",
  "language_code": "nl"
}
-->
# Quizzen

Deze quizzen zijn de pre- en post-lezing quizzen voor het ML-curriculum op https://aka.ms/ml-beginners

## Projectinstelling

```
npm install
```

### Compileert en herlaadt automatisch voor ontwikkeling

```
npm run serve
```

### Compileert en minimaliseert voor productie

```
npm run build
```

### Controleert en corrigeert bestanden

```
npm run lint
```

### Configuratie aanpassen

Zie [Configuratiereferentie](https://cli.vuejs.org/config/).

Credits: Dank aan de originele versie van deze quiz-app: https://github.com/arpan45/simple-quiz-vue

## Deployen naar Azure

Hier is een stapsgewijze handleiding om je op weg te helpen:

1. Fork een GitHub-repository  
Zorg ervoor dat de code van je statische webapp in je GitHub-repository staat. Fork deze repository.

2. Maak een Azure Static Web App  
- Maak een [Azure-account](http://azure.microsoft.com)  
- Ga naar het [Azure-portaal](https://portal.azure.com)  
- Klik op "Create a resource" en zoek naar "Static Web App".  
- Klik op "Create".  

3. Configureer de Static Web App  
- Basics:  
  - Abonnement: Selecteer je Azure-abonnement.  
  - Resourcegroep: Maak een nieuwe resourcegroep of gebruik een bestaande.  
  - Naam: Geef een naam op voor je statische webapp.  
  - Regio: Kies de regio die het dichtst bij je gebruikers ligt.  

- #### Deployment Details:  
  - Bron: Selecteer "GitHub".  
  - GitHub-account: Autoriseer Azure om toegang te krijgen tot je GitHub-account.  
  - Organisatie: Selecteer je GitHub-organisatie.  
  - Repository: Kies de repository die je statische webapp bevat.  
  - Branch: Selecteer de branch waarvan je wilt deployen.  

- #### Build Details:  
  - Build Presets: Kies het framework waarmee je app is gebouwd (bijv. React, Angular, Vue, etc.).  
  - App-locatie: Geef de map op die je app-code bevat (bijv. / als het in de root staat).  
  - API-locatie: Als je een API hebt, geef de locatie op (optioneel).  
  - Output-locatie: Geef de map op waar de build-output wordt gegenereerd (bijv. build of dist).  

4. Review en aanmaken  
Controleer je instellingen en klik op "Create". Azure zal de benodigde resources instellen en een GitHub Actions-workflow in je repository aanmaken.

5. GitHub Actions-workflow  
Azure zal automatisch een GitHub Actions-workflowbestand aanmaken in je repository (.github/workflows/azure-static-web-apps-<name>.yml). Deze workflow zal het build- en deployproces afhandelen.

6. Monitor de deployment  
Ga naar het tabblad "Actions" in je GitHub-repository.  
Je zou een workflow moeten zien draaien. Deze workflow zal je statische webapp bouwen en deployen naar Azure.  
Zodra de workflow is voltooid, is je app live op de verstrekte Azure-URL.

### Voorbeeld Workflow-bestand

Hier is een voorbeeld van hoe het GitHub Actions-workflowbestand eruit zou kunnen zien:  
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

### Aanvullende bronnen  
- [Azure Static Web Apps Documentatie](https://learn.microsoft.com/azure/static-web-apps/getting-started)  
- [GitHub Actions Documentatie](https://docs.github.com/actions/use-cases-and-examples/deploying/deploying-to-azure-static-web-app)  

---

**Disclaimer**:  
Dit document is vertaald met behulp van de AI-vertalingsservice [Co-op Translator](https://github.com/Azure/co-op-translator). Hoewel we ons best doen voor nauwkeurigheid, dient u zich ervan bewust te zijn dat geautomatiseerde vertalingen fouten of onnauwkeurigheden kunnen bevatten. Het originele document in zijn oorspronkelijke taal moet worden beschouwd als de gezaghebbende bron. Voor cruciale informatie wordt professionele menselijke vertaling aanbevolen. Wij zijn niet aansprakelijk voor misverstanden of verkeerde interpretaties die voortvloeien uit het gebruik van deze vertaling.