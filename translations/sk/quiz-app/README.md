<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "6d130dffca5db70d7e615f926cb1ad4c",
  "translation_date": "2025-09-05T16:15:37+00:00",
  "source_file": "quiz-app/README.md",
  "language_code": "sk"
}
-->
# Kvízy

Tieto kvízy sú prednáškové a po prednáškové kvízy pre ML kurikulum na https://aka.ms/ml-beginners

## Nastavenie projektu

```
npm install
```

### Kompilácia a automatické načítanie pre vývoj

```
npm run serve
```

### Kompilácia a minimalizácia pre produkciu

```
npm run build
```

### Kontrola a oprava súborov

```
npm run lint
```

### Prispôsobenie konfigurácie

Pozrite si [Referenciu konfigurácie](https://cli.vuejs.org/config/).

Kredity: Vďaka pôvodnej verzii tejto aplikácie na kvízy: https://github.com/arpan45/simple-quiz-vue

## Nasadenie na Azure

Tu je krok-za-krokom návod, ktorý vám pomôže začať:

1. Forknite GitHub repozitár  
Uistite sa, že váš kód statickej webovej aplikácie je vo vašom GitHub repozitári. Forknite tento repozitár.

2. Vytvorte Azure Static Web App  
- Vytvorte si [Azure účet](http://azure.microsoft.com)  
- Prejdite na [Azure portál](https://portal.azure.com)  
- Kliknite na „Create a resource“ a vyhľadajte „Static Web App“.  
- Kliknite na „Create“.  

3. Konfigurácia statickej webovej aplikácie  
- Základy:  
  - Subscription: Vyberte vaše Azure predplatné.  
  - Resource Group: Vytvorte novú skupinu zdrojov alebo použite existujúcu.  
  - Name: Zadajte názov pre vašu statickú webovú aplikáciu.  
  - Region: Vyberte región najbližší vašim používateľom.  

- #### Detaily nasadenia:  
  - Source: Vyberte „GitHub“.  
  - GitHub Account: Autorizujte Azure na prístup k vášmu GitHub účtu.  
  - Organization: Vyberte vašu GitHub organizáciu.  
  - Repository: Vyberte repozitár obsahujúci vašu statickú webovú aplikáciu.  
  - Branch: Vyberte vetvu, z ktorej chcete nasadzovať.  

- #### Detaily zostavenia:  
  - Build Presets: Vyberte framework, na ktorom je vaša aplikácia postavená (napr. React, Angular, Vue, atď.).  
  - App Location: Zadajte priečinok obsahujúci kód vašej aplikácie (napr. / ak je v koreňovom adresári).  
  - API Location: Ak máte API, zadajte jeho umiestnenie (voliteľné).  
  - Output Location: Zadajte priečinok, kde sa generuje výstup zostavenia (napr. build alebo dist).  

4. Skontrolujte a vytvorte  
Skontrolujte svoje nastavenia a kliknite na „Create“. Azure nastaví potrebné zdroje a vytvorí GitHub Actions workflow vo vašom repozitári.

5. GitHub Actions Workflow  
Azure automaticky vytvorí GitHub Actions workflow súbor vo vašom repozitári (.github/workflows/azure-static-web-apps-<name>.yml). Tento workflow bude spracovávať proces zostavenia a nasadenia.

6. Monitorovanie nasadenia  
Prejdite na kartu „Actions“ vo vašom GitHub repozitári.  
Mali by ste vidieť bežiaci workflow. Tento workflow zostaví a nasadí vašu statickú webovú aplikáciu na Azure.  
Po dokončení workflowu bude vaša aplikácia dostupná na poskytnutej Azure URL.

### Príklad workflow súboru

Tu je príklad, ako môže vyzerať GitHub Actions workflow súbor:  
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

### Ďalšie zdroje
- [Dokumentácia Azure Static Web Apps](https://learn.microsoft.com/azure/static-web-apps/getting-started)  
- [Dokumentácia GitHub Actions](https://docs.github.com/actions/use-cases-and-examples/deploying/deploying-to-azure-static-web-app)  

---

**Upozornenie**:  
Tento dokument bol preložený pomocou služby AI prekladu [Co-op Translator](https://github.com/Azure/co-op-translator). Aj keď sa snažíme o presnosť, prosím, berte na vedomie, že automatizované preklady môžu obsahovať chyby alebo nepresnosti. Pôvodný dokument v jeho pôvodnom jazyku by mal byť považovaný za autoritatívny zdroj. Pre kritické informácie sa odporúča profesionálny ľudský preklad. Nie sme zodpovední za akékoľvek nedorozumenia alebo nesprávne interpretácie vyplývajúce z použitia tohto prekladu.