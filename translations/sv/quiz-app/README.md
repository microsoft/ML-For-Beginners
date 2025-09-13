<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "6d130dffca5db70d7e615f926cb1ad4c",
  "translation_date": "2025-09-05T21:48:33+00:00",
  "source_file": "quiz-app/README.md",
  "language_code": "sv"
}
-->
# Quizzer

Dessa quizzer är för- och efterföreläsningsquizzer för ML-kursplanen på https://aka.ms/ml-beginners

## Projektinställning

```
npm install
```

### Kompilerar och laddar om för utveckling

```
npm run serve
```

### Kompilerar och minimerar för produktion

```
npm run build
```

### Lintar och fixar filer

```
npm run lint
```

### Anpassa konfiguration

Se [Konfigurationsreferens](https://cli.vuejs.org/config/).

Krediter: Tack till den ursprungliga versionen av denna quizapp: https://github.com/arpan45/simple-quiz-vue

## Distribuera till Azure

Här är en steg-för-steg-guide för att hjälpa dig komma igång:

1. Forka ett GitHub-repository  
Se till att din statiska webbappkod finns i ditt GitHub-repository. Forka detta repository.

2. Skapa en Azure Static Web App  
- Skapa ett [Azure-konto](http://azure.microsoft.com)  
- Gå till [Azure-portalen](https://portal.azure.com)  
- Klicka på "Skapa en resurs" och sök efter "Static Web App".  
- Klicka på "Skapa".  

3. Konfigurera den statiska webbappen  
- Grundläggande:  
  - Prenumeration: Välj din Azure-prenumeration.  
  - Resursgrupp: Skapa en ny resursgrupp eller använd en befintlig.  
  - Namn: Ange ett namn för din statiska webbapp.  
  - Region: Välj den region som är närmast dina användare.  

- #### Distributionsdetaljer:  
  - Källa: Välj "GitHub".  
  - GitHub-konto: Auktorisera Azure att få åtkomst till ditt GitHub-konto.  
  - Organisation: Välj din GitHub-organisation.  
  - Repository: Välj det repository som innehåller din statiska webbapp.  
  - Gren: Välj den gren du vill distribuera från.  

- #### Byggdetaljer:  
  - Byggförinställningar: Välj det ramverk din app är byggd med (t.ex. React, Angular, Vue, etc.).  
  - Appens plats: Ange mappen som innehåller din appkod (t.ex. / om den är i roten).  
  - API-plats: Om du har ett API, ange dess plats (valfritt).  
  - Utdata-plats: Ange mappen där byggutdata genereras (t.ex. build eller dist).  

4. Granska och skapa  
Granska dina inställningar och klicka på "Skapa". Azure kommer att ställa in de nödvändiga resurserna och skapa ett GitHub Actions-arbetsflöde i ditt repository.

5. GitHub Actions-arbetsflöde  
Azure kommer automatiskt att skapa en GitHub Actions-arbetsflödesfil i ditt repository (.github/workflows/azure-static-web-apps-<name>.yml). Detta arbetsflöde hanterar bygg- och distributionsprocessen.

6. Övervaka distributionen  
Gå till fliken "Actions" i ditt GitHub-repository.  
Du bör se ett arbetsflöde som körs. Detta arbetsflöde kommer att bygga och distribuera din statiska webbapp till Azure.  
När arbetsflödet är klart kommer din app att vara live på den angivna Azure-URL:en.

### Exempel på arbetsflödesfil

Här är ett exempel på hur GitHub Actions-arbetsflödesfilen kan se ut:  
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

### Ytterligare resurser  
- [Azure Static Web Apps-dokumentation](https://learn.microsoft.com/azure/static-web-apps/getting-started)  
- [GitHub Actions-dokumentation](https://docs.github.com/actions/use-cases-and-examples/deploying/deploying-to-azure-static-web-app)  

---

**Ansvarsfriskrivning**:  
Detta dokument har översatts med hjälp av AI-översättningstjänsten [Co-op Translator](https://github.com/Azure/co-op-translator). Även om vi strävar efter noggrannhet, vänligen notera att automatiska översättningar kan innehålla fel eller felaktigheter. Det ursprungliga dokumentet på dess originalspråk bör betraktas som den auktoritativa källan. För kritisk information rekommenderas professionell mänsklig översättning. Vi ansvarar inte för eventuella missförstånd eller feltolkningar som uppstår vid användning av denna översättning.