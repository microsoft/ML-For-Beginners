# Quiz

Questi quiz sono i quiz pre e post-lezione per il curriculum di ML su https://aka.ms/ml-beginners

## Configurazione del progetto

```
npm install
```

### Compilazione e ricaricamento automatico per lo sviluppo

```
npm run serve
```

### Compilazione e minificazione per la produzione

```
npm run build
```

### Lint e correzione dei file

```
npm run lint
```

### Personalizza la configurazione

Consulta [Configuration Reference](https://cli.vuejs.org/config/).

Crediti: Grazie alla versione originale di questa app quiz: https://github.com/arpan45/simple-quiz-vue

## Distribuzione su Azure

Ecco una guida passo-passo per aiutarti a iniziare:

1. Fai un fork del repository GitHub
Assicurati che il codice della tua app web statica sia nel tuo repository GitHub. Fai un fork di questo repository.

2. Crea una Azure Static Web App
- Crea un [account Azure](http://azure.microsoft.com)
- Vai al [portale di Azure](https://portal.azure.com) 
- Clicca su “Crea una risorsa” e cerca “Static Web App”.
- Clicca su “Crea”.

3. Configura la Static Web App
- Base: Sottoscrizione: Seleziona la tua sottoscrizione Azure.
- Gruppo di risorse: Crea un nuovo gruppo di risorse o usa uno esistente.
- Nome: Fornisci un nome per la tua app web statica.
- Regione: Scegli la regione più vicina ai tuoi utenti.

- #### Dettagli di distribuzione:
- Sorgente: Seleziona “GitHub”.
- Account GitHub: Autorizza Azure ad accedere al tuo account GitHub.
- Organizzazione: Seleziona la tua organizzazione GitHub.
- Repository: Scegli il repository contenente la tua app web statica.
- Branch: Seleziona il branch da cui vuoi distribuire.

- #### Dettagli di build:
- Preimpostazioni di build: Scegli il framework con cui è costruita la tua app (es. React, Angular, Vue, ecc.).
- Posizione dell'app: Specifica la cartella contenente il codice della tua app (es. / se è nella radice).
- Posizione API: Se hai un'API, specifica la sua posizione (opzionale).
- Posizione output: Specifica la cartella in cui viene generato l'output della build (es. build o dist).

4. Rivedi e crea
Rivedi le tue impostazioni e clicca su “Crea”. Azure configurerà le risorse necessarie e creerà un workflow di GitHub Actions nel tuo repository.

5. Workflow di GitHub Actions
Azure creerà automaticamente un file di workflow di GitHub Actions nel tuo repository (.github/workflows/azure-static-web-apps-<nome>.yml). Questo workflow gestirà il processo di build e distribuzione.

6. Monitora la distribuzione
Vai alla scheda “Actions” nel tuo repository GitHub.
Dovresti vedere un workflow in esecuzione. Questo workflow costruirà e distribuirà la tua app web statica su Azure.
Una volta completato il workflow, la tua app sarà live sull'URL fornito da Azure.

### Esempio di file Workflow

Ecco un esempio di come potrebbe apparire il file di workflow di GitHub Actions:
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

### Risorse aggiuntive
- [Documentazione di Azure Static Web Apps](https://learn.microsoft.com/azure/static-web-apps/getting-started)
- [Documentazione di GitHub Actions](https://docs.github.com/actions/use-cases-and-examples/deploying/deploying-to-azure-static-web-app)

**Disclaimer**: 
Questo documento è stato tradotto utilizzando servizi di traduzione automatica basati su AI. Anche se ci sforziamo di garantire l'accuratezza, si prega di essere consapevoli che le traduzioni automatiche possono contenere errori o imprecisioni. Il documento originale nella sua lingua nativa dovrebbe essere considerato la fonte autorevole. Per informazioni critiche, si raccomanda una traduzione professionale umana. Non siamo responsabili per eventuali malintesi o interpretazioni errate derivanti dall'uso di questa traduzione.