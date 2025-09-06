<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "6d130dffca5db70d7e615f926cb1ad4c",
  "translation_date": "2025-09-05T21:48:45+00:00",
  "source_file": "quiz-app/README.md",
  "language_code": "no"
}
-->
# Quizer

Disse quizene er forhånds- og etterforelesningsquizene for ML-læreplanen på https://aka.ms/ml-beginners

## Prosjektoppsett

```
npm install
```

### Kompilerer og oppdaterer automatisk for utvikling

```
npm run serve
```

### Kompilerer og minimerer for produksjon

```
npm run build
```

### Linter og fikser filer

```
npm run lint
```

### Tilpass konfigurasjon

Se [Konfigurasjonsreferanse](https://cli.vuejs.org/config/).

Kreditering: Takk til den originale versjonen av denne quiz-appen: https://github.com/arpan45/simple-quiz-vue

## Distribuere til Azure

Her er en steg-for-steg guide for å komme i gang:

1. Fork en GitHub-repositorium  
Sørg for at koden til din statiske webapp er i ditt GitHub-repositorium. Fork dette repositoriet.

2. Opprett en Azure Static Web App  
- Opprett en [Azure-konto](http://azure.microsoft.com)  
- Gå til [Azure-portalen](https://portal.azure.com)  
- Klikk på "Opprett en ressurs" og søk etter "Static Web App".  
- Klikk "Opprett".  

3. Konfigurer den statiske webappen  
- Grunnleggende: Abonnement: Velg ditt Azure-abonnement.  
- Ressursgruppe: Opprett en ny ressursgruppe eller bruk en eksisterende.  
- Navn: Gi et navn til din statiske webapp.  
- Region: Velg regionen nærmest brukerne dine.  

- #### Distribusjonsdetaljer:  
- Kilde: Velg "GitHub".  
- GitHub-konto: Autoriser Azure til å få tilgang til din GitHub-konto.  
- Organisasjon: Velg din GitHub-organisasjon.  
- Repositorium: Velg repositoriet som inneholder din statiske webapp.  
- Gren: Velg grenen du vil distribuere fra.  

- #### Byggdetaljer:  
- Byggforhåndsinnstillinger: Velg rammeverket appen din er bygget med (f.eks. React, Angular, Vue, osv.).  
- App-plassering: Angi mappen som inneholder koden til appen din (f.eks. / hvis den er i roten).  
- API-plassering: Hvis du har en API, spesifiser dens plassering (valgfritt).  
- Utdata-plassering: Angi mappen der byggeutdataene genereres (f.eks. build eller dist).  

4. Gjennomgå og opprett  
Gjennomgå innstillingene dine og klikk "Opprett". Azure vil sette opp nødvendige ressurser og opprette en GitHub Actions-arbeidsflyt i ditt repositorium.  

5. GitHub Actions-arbeidsflyt  
Azure vil automatisk opprette en GitHub Actions-arbeidsflytfil i ditt repositorium (.github/workflows/azure-static-web-apps-<name>.yml). Denne arbeidsflyten vil håndtere bygge- og distribusjonsprosessen.  

6. Overvåk distribusjonen  
Gå til "Actions"-fanen i ditt GitHub-repositorium.  
Du bør se en arbeidsflyt som kjører. Denne arbeidsflyten vil bygge og distribuere din statiske webapp til Azure.  
Når arbeidsflyten er fullført, vil appen din være live på den oppgitte Azure-URL-en.  

### Eksempel på arbeidsflytfil

Her er et eksempel på hvordan GitHub Actions-arbeidsflytfilen kan se ut:  
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

### Tilleggsressurser  
- [Azure Static Web Apps Dokumentasjon](https://learn.microsoft.com/azure/static-web-apps/getting-started)  
- [GitHub Actions Dokumentasjon](https://docs.github.com/actions/use-cases-and-examples/deploying/deploying-to-azure-static-web-app)  

---

**Ansvarsfraskrivelse**:  
Dette dokumentet er oversatt ved hjelp av AI-oversettelsestjenesten [Co-op Translator](https://github.com/Azure/co-op-translator). Selv om vi tilstreber nøyaktighet, vennligst vær oppmerksom på at automatiske oversettelser kan inneholde feil eller unøyaktigheter. Det originale dokumentet på sitt opprinnelige språk bør anses som den autoritative kilden. For kritisk informasjon anbefales profesjonell menneskelig oversettelse. Vi er ikke ansvarlige for eventuelle misforståelser eller feiltolkninger som oppstår ved bruk av denne oversettelsen.