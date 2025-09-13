<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "6d130dffca5db70d7e615f926cb1ad4c",
  "translation_date": "2025-09-05T13:01:57+00:00",
  "source_file": "quiz-app/README.md",
  "language_code": "hr"
}
-->
# Kvizovi

Ovi kvizovi su uvodni i završni kvizovi za ML kurikulum na https://aka.ms/ml-beginners

## Postavljanje projekta

```
npm install
```

### Kompajlira i automatski učitava za razvoj

```
npm run serve
```

### Kompajlira i minimizira za produkciju

```
npm run build
```

### Provjerava i popravlja datoteke

```
npm run lint
```

### Prilagodba konfiguracije

Pogledajte [Referencu za konfiguraciju](https://cli.vuejs.org/config/).

Zasluge: Zahvala originalnoj verziji ove aplikacije za kviz: https://github.com/arpan45/simple-quiz-vue

## Postavljanje na Azure

Evo korak-po-korak vodiča koji će vam pomoći da započnete:

1. Forkajte GitHub repozitorij
Osigurajte da je kod vaše statične web aplikacije u vašem GitHub repozitoriju. Forkajte ovaj repozitorij.

2. Kreirajte Azure Static Web App
- Kreirajte [Azure račun](http://azure.microsoft.com)
- Idite na [Azure portal](https://portal.azure.com) 
- Kliknite na “Create a resource” i potražite “Static Web App”.
- Kliknite “Create”.

3. Konfigurirajte Static Web App
- Osnovno: Pretplata: Odaberite svoju Azure pretplatu.
- Resource Group: Kreirajte novu grupu resursa ili koristite postojeću.
- Naziv: Dodijelite naziv svojoj statičnoj web aplikaciji.
- Regija: Odaberite regiju najbližu vašim korisnicima.

- #### Detalji o implementaciji:
- Izvor: Odaberite “GitHub”.
- GitHub račun: Autorizirajte Azure za pristup vašem GitHub računu.
- Organizacija: Odaberite svoju GitHub organizaciju.
- Repozitorij: Odaberite repozitorij koji sadrži vašu statičnu web aplikaciju.
- Grana: Odaberite granu iz koje želite implementirati.

- #### Detalji o izgradnji:
- Predlošci za izgradnju: Odaberite okvir na kojem je vaša aplikacija izgrađena (npr. React, Angular, Vue, itd.).
- Lokacija aplikacije: Navedite mapu koja sadrži kod vaše aplikacije (npr. / ako je u korijenu).
- Lokacija API-ja: Ako imate API, navedite njegovu lokaciju (opcionalno).
- Lokacija izlaza: Navedite mapu u kojoj se generira izlaz izgradnje (npr. build ili dist).

4. Pregled i kreiranje
Pregledajte svoje postavke i kliknite “Create”. Azure će postaviti potrebne resurse i kreirati GitHub Actions workflow u vašem repozitoriju.

5. GitHub Actions Workflow
Azure će automatski kreirati GitHub Actions workflow datoteku u vašem repozitoriju (.github/workflows/azure-static-web-apps-<name>.yml). Ovaj workflow će upravljati procesom izgradnje i implementacije.

6. Praćenje implementacije
Idite na karticu “Actions” u vašem GitHub repozitoriju.
Trebali biste vidjeti workflow koji se pokreće. Ovaj workflow će izgraditi i implementirati vašu statičnu web aplikaciju na Azure.
Nakon što workflow završi, vaša aplikacija će biti dostupna na dodijeljenom Azure URL-u.

### Primjer datoteke workflowa

Evo primjera kako bi GitHub Actions workflow datoteka mogla izgledati:
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

### Dodatni resursi
- [Dokumentacija za Azure Static Web Apps](https://learn.microsoft.com/azure/static-web-apps/getting-started)
- [Dokumentacija za GitHub Actions](https://docs.github.com/actions/use-cases-and-examples/deploying/deploying-to-azure-static-web-app)

---

**Odricanje od odgovornosti**:  
Ovaj dokument je preveden korištenjem AI usluge za prevođenje [Co-op Translator](https://github.com/Azure/co-op-translator). Iako nastojimo osigurati točnost, imajte na umu da automatski prijevodi mogu sadržavati pogreške ili netočnosti. Izvorni dokument na izvornom jeziku treba smatrati mjerodavnim izvorom. Za ključne informacije preporučuje se profesionalni prijevod od strane stručnjaka. Ne preuzimamo odgovornost za bilo kakve nesporazume ili pogrešne interpretacije proizašle iz korištenja ovog prijevoda.