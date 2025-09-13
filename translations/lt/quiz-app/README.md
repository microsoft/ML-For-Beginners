<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "6d130dffca5db70d7e615f926cb1ad4c",
  "translation_date": "2025-09-03T17:58:04+00:00",
  "source_file": "quiz-app/README.md",
  "language_code": "lt"
}
-->
# Testai

Šie testai yra skirti prieš ir po paskaitų ML mokymo programai adresu https://aka.ms/ml-beginners

## Projekto nustatymas

```
npm install
```

### Kompiliavimas ir automatinis perkrovimas vystymui

```
npm run serve
```

### Kompiliavimas ir minimizavimas produkcijai

```
npm run build
```

### Failų tikrinimas ir taisymas

```
npm run lint
```

### Konfigūracijos pritaikymas

Žiūrėkite [Konfigūracijos nuorodą](https://cli.vuejs.org/config/).

Kreditai: Dėkojame originaliai šio testo programėlės versijai: https://github.com/arpan45/simple-quiz-vue

## Diegimas į Azure

Štai žingsnis po žingsnio vadovas, kuris padės jums pradėti:

1. Fork'inkite GitHub saugyklą  
Įsitikinkite, kad jūsų statinio tinklalapio programos kodas yra jūsų GitHub saugykloje. Fork'inkite šią saugyklą.

2. Sukurkite Azure statinį tinklalapį  
- Susikurkite [Azure paskyrą](http://azure.microsoft.com)  
- Eikite į [Azure portalą](https://portal.azure.com)  
- Spustelėkite „Sukurti resursą“ ir ieškokite „Static Web App“.  
- Spustelėkite „Sukurti“.

3. Konfigūruokite statinį tinklalapį  
- Pagrindai: Prenumerata: Pasirinkite savo Azure prenumeratą.  
- Resursų grupė: Sukurkite naują resursų grupę arba naudokite esamą.  
- Pavadinimas: Nurodykite savo statinio tinklalapio pavadinimą.  
- Regionas: Pasirinkite regioną, artimiausią jūsų vartotojams.

- #### Diegimo detalės:  
- Šaltinis: Pasirinkite „GitHub“.  
- GitHub paskyra: Suteikite Azure leidimą pasiekti jūsų GitHub paskyrą.  
- Organizacija: Pasirinkite savo GitHub organizaciją.  
- Saugykla: Pasirinkite saugyklą, kurioje yra jūsų statinio tinklalapio kodas.  
- Šaka: Pasirinkite šaką, iš kurios norite diegti.

- #### Kūrimo detalės:  
- Kūrimo šablonai: Pasirinkite sistemą, su kuria jūsų programėlė sukurta (pvz., React, Angular, Vue ir kt.).  
- Programos vieta: Nurodykite aplanką, kuriame yra jūsų programos kodas (pvz., / jei jis yra šaknyje).  
- API vieta: Jei turite API, nurodykite jo vietą (nebūtina).  
- Išvesties vieta: Nurodykite aplanką, kuriame generuojamas kūrimo išvestis (pvz., build arba dist).

4. Peržiūra ir sukūrimas  
Peržiūrėkite savo nustatymus ir spustelėkite „Sukurti“. Azure sukurs reikalingus resursus ir sukurs GitHub Actions darbo eigą jūsų saugykloje.

5. GitHub Actions darbo eiga  
Azure automatiškai sukurs GitHub Actions darbo eigos failą jūsų saugykloje (.github/workflows/azure-static-web-apps-<name>.yml). Ši darbo eiga tvarkys kūrimo ir diegimo procesą.

6. Diegimo stebėjimas  
Eikite į „Actions“ skirtuką savo GitHub saugykloje.  
Turėtumėte matyti vykdomą darbo eigą. Ši darbo eiga sukurs ir įdiegs jūsų statinį tinklalapį į Azure.  
Kai darbo eiga bus baigta, jūsų programėlė bus pasiekiama nurodytu Azure URL.

### Pavyzdinis darbo eigos failas

Štai pavyzdys, kaip gali atrodyti GitHub Actions darbo eigos failas:  
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

### Papildomi ištekliai  
- [Azure Static Web Apps dokumentacija](https://learn.microsoft.com/azure/static-web-apps/getting-started)  
- [GitHub Actions dokumentacija](https://docs.github.com/actions/use-cases-and-examples/deploying/deploying-to-azure-static-web-app)  

---

**Atsakomybės apribojimas**:  
Šis dokumentas buvo išverstas naudojant AI vertimo paslaugą [Co-op Translator](https://github.com/Azure/co-op-translator). Nors siekiame tikslumo, prašome atkreipti dėmesį, kad automatiniai vertimai gali turėti klaidų ar netikslumų. Originalus dokumentas jo gimtąja kalba turėtų būti laikomas autoritetingu šaltiniu. Kritinei informacijai rekomenduojama naudoti profesionalų žmogaus vertimą. Mes neprisiimame atsakomybės už nesusipratimus ar klaidingus interpretavimus, atsiradusius dėl šio vertimo naudojimo.