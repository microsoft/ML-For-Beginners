# Maswali

Haya maswali ni ya kabla na baada ya mihadhara ya mtaala wa ML kwenye https://aka.ms/ml-beginners

## Kuanzisha Mradi

```
npm install
```

### Hukusanya na kupakia upya kwa maendeleo

```
npm run serve
```

### Hukusanya na kupunguza kwa uzalishaji

```
npm run build
```

### Hufanya lint na kurekebisha faili

```
npm run lint
```

### Kubadilisha usanidi

Tazama [Marejeleo ya Usanidi](https://cli.vuejs.org/config/).

Shukrani: Asante kwa toleo la awali la programu hii ya maswali: https://github.com/arpan45/simple-quiz-vue

## Kuweka kwenye Azure

Hapa kuna mwongozo wa hatua kwa hatua kukusaidia kuanza:

1. Fork Repositori ya GitHub
Hakikisha msimbo wa programu yako ya wavuti ya static uko kwenye repositori yako ya GitHub. Fork repositori hii.

2. Unda Azure Static Web App
- Unda [akaunti ya Azure](http://azure.microsoft.com)
- Nenda kwenye [portal ya Azure](https://portal.azure.com) 
- Bonyeza “Create a resource” na tafuta “Static Web App”.
- Bonyeza “Create”.

3. Sanidi Static Web App
- Msingi: Usajili: Chagua usajili wako wa Azure.
- Kikundi cha Rasilimali: Unda kikundi kipya cha rasilimali au tumia kilichopo.
- Jina: Toa jina kwa programu yako ya wavuti ya static.
- Kanda: Chagua kanda iliyo karibu na watumiaji wako.

- #### Maelezo ya Uwekaji:
- Chanzo: Chagua “GitHub”.
- Akaunti ya GitHub: Ruhusu Azure kufikia akaunti yako ya GitHub.
- Shirika: Chagua shirika lako la GitHub.
- Repositori: Chagua repositori inayoshikilia programu yako ya wavuti ya static.
- Tawi: Chagua tawi unalotaka kuweka kutoka.

- #### Maelezo ya Ujenzi:
- Presets za Ujenzi: Chagua mfumo ambao programu yako imejengwa (mfano, React, Angular, Vue, nk.).
- Mahali pa Programu: Eleza folda inayoshikilia msimbo wa programu yako (mfano, / ikiwa iko kwenye mzizi).
- Mahali pa API: Ikiwa una API, eleza mahali pake (hiari).
- Mahali pa Matokeo: Eleza folda ambapo matokeo ya ujenzi yanazalishwa (mfano, build au dist).

4. Kagua na Unda
Kagua mipangilio yako na bonyeza “Create”. Azure itaweka rasilimali zinazohitajika na kuunda mtiririko wa kazi wa GitHub Actions kwenye repositori yako.

5. Mtiririko wa Kazi wa GitHub Actions
Azure itaweka faili ya mtiririko wa kazi wa GitHub Actions kwenye repositori yako (.github/workflows/azure-static-web-apps-<name>.yml). Mtiririko huu utashughulikia mchakato wa ujenzi na uwekaji.

6. Fuata Uwekaji
Nenda kwenye kichupo cha “Actions” kwenye repositori yako ya GitHub.
Unapaswa kuona mtiririko wa kazi unaoendesha. Mtiririko huu utajenga na kuweka programu yako ya wavuti ya static kwenye Azure.
Baada ya mtiririko wa kazi kukamilika, programu yako itakuwa hai kwenye URL iliyotolewa ya Azure.

### Faili ya Mfano ya Mtiririko wa Kazi

Hapa kuna mfano wa jinsi faili ya mtiririko wa kazi wa GitHub Actions inaweza kuonekana:
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

### Rasilimali za Ziada
- [Nyaraka za Azure Static Web Apps](https://learn.microsoft.com/azure/static-web-apps/getting-started)
- [Nyaraka za GitHub Actions](https://docs.github.com/actions/use-cases-and-examples/deploying/deploying-to-azure-static-web-app)

**Kanusho**:
Hati hii imetafsiriwa kwa kutumia huduma za kutafsiri za AI za mashine. Ingawa tunajitahidi kwa usahihi, tafadhali fahamu kwamba tafsiri za kiotomatiki zinaweza kuwa na makosa au kutokubaliana. Hati ya asili katika lugha yake ya kiasili inapaswa kuzingatiwa kama chanzo sahihi. Kwa taarifa muhimu, tafsiri ya kibinadamu ya kitaalamu inapendekezwa. Hatutawajibika kwa kutoelewana au kutafsiri vibaya kunakotokana na matumizi ya tafsiri hii.