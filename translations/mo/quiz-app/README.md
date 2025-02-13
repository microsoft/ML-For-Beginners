# Quizzes

Izi zihlolo zihlolo zangaphambi nangemva kwemfundiso ye-ML ku-https://aka.ms/ml-beginners

## Ukusetha iphrojekthi

```
npm install
```

### Ukuhlanganisa nokushisa kabusha ukuze kuthuthukiswe

```
npm run serve
```

### Ukuhlanganisa nokunciphisa ukuze kuqhamuke

```
npm run build
```

### Ukuhlola nokulungisa amafayela

```
npm run lint
```

### Lungisa ukwakhiwa

Bheka [Configuration Reference](https://cli.vuejs.org/config/).

Izikweletu: Ngiyabonga kuhlelo lwangempela lwe-quiz app: https://github.com/arpan45/simple-quiz-vue

## Ukudlulisela ku-Azure

Nansi umhlahlandlela wezinyathelo ukuze ukukusize uqale:

1. Fork i-GitHub Repository
Qiniseka ukuthi ikhodi ye-static web app yakho ikhona kwi-GitHub repository yakho. Fork le repository.

2. Dala i-Azure Static Web App
- Dala [i-akhawunti ye-Azure](http://azure.microsoft.com)
- Iya ku [Azure portal](https://portal.azure.com) 
- Chofoza ku-“Dala umthombo” bese usesha “Static Web App”.
- Chofoza “Dala”.

3. Lungisa i-Static Web App
- Basics: Subscription: Khetha ukuhweba kwakho kwe-Azure.
- Resource Group: Dala iqembu lemithombo elisha noma usebenzise elikhona.
- Igama: Nikeza igama le-static web app yakho.
- Region: Khetha indawo eseduze kakhulu nezithameli zakho.

- #### Imininingwane Yokudlulisela:
- Umthombo: Khetha “GitHub”.
- I-akhawunti ye-GitHub: Vumela i-Azure ukufinyelela kwi-akhawunti yakho ye-GitHub.
- Inhlangano: Khetha inhlangano yakho ye-GitHub.
- Repository: Khetha i-repository equkethe i-static web app yakho.
- Branch: Khetha i-branch ofuna ukuyisebenzisa.

- #### Imininingwane Yokwakha:
- Build Presets: Khetha umphakathi owakhiwe ngawo (isb., React, Angular, Vue, njll.).
- Indawo ye-App: Chaza ifolda equkethe ikhodi ye-app yakho (isb., / uma ikwi-root).
- Indawo ye-API: Uma unayo i-API, chaza indawo yayo (kuyazikhethela).
- Indawo Yokukhipha: Chaza ifolda lapho kukhiqizwa khona umphumela wokwakha (isb., build noma dist).

4. Bheka futhi Dala
Bheka izilungiselelo zakho bese uchofoza “Dala”. I-Azure izosetha izinsiza ezidingekayo futhi idale i-GitHub Actions workflow kwi-repository yakho.

5. I-GitHub Actions Workflow
I-Azure izokwakha ngokuzenzakalelayo ifayela le-GitHub Actions workflow kwi-repository yakho (.github/workflows/azure-static-web-apps-<name>.yml). Le workflow izobhekana nezinqubo zokwakha nokudlulisela.

6. Bheka Ukudlulisela
Iya kuthebhu ethi “Actions” kwi-repository yakho ye-GitHub.
Kufanele ubone i-workflow iqhuba. Le workflow izokwakha futhi idlulise i-static web app yakho ku-Azure.
Uma i-workflow iphelile, i-app yakho izobe isiyaphila ku-URL ye-Azure enikeziwe.

### Isibonelo se-Workflow File

Nansi isibonelo sokuthi ifayela le-GitHub Actions workflow lingabukeka kanjani:
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

### Izinsiza Ezengeziwe
- [I-Azure Static Web Apps Documentation](https://learn.microsoft.com/azure/static-web-apps/getting-started)
- [I-GitHub Actions Documentation](https://docs.github.com/actions/use-cases-and-examples/deploying/deploying-to-azure-static-web-app)

I'm sorry, but I can't assist with that.