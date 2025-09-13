<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "6d130dffca5db70d7e615f926cb1ad4c",
  "translation_date": "2025-09-05T13:02:09+00:00",
  "source_file": "quiz-app/README.md",
  "language_code": "sl"
}
-->
# Kvizi

Ti kvizi so predhodni in zaključni kvizi za učni načrt strojnega učenja na https://aka.ms/ml-beginners

## Nastavitev projekta

```
npm install
```

### Kompilacija in samodejno osveževanje za razvoj

```
npm run serve
```

### Kompilacija in minimizacija za produkcijo

```
npm run build
```

### Preverjanje kode in odpravljanje napak

```
npm run lint
```

### Prilagoditev konfiguracije

Glej [Referenca konfiguracije](https://cli.vuejs.org/config/).

Zasluge: Zahvala za izvirno različico te aplikacije za kvize: https://github.com/arpan45/simple-quiz-vue

## Namestitev na Azure

Tukaj je korak-po-korak vodič, ki vam bo pomagal začeti:

1. Forkajte GitHub repozitorij  
Poskrbite, da je koda vaše statične spletne aplikacije v vašem GitHub repozitoriju. Forkajte ta repozitorij.

2. Ustvarite Azure Static Web App  
- Ustvarite [Azure račun](http://azure.microsoft.com)  
- Pojdite na [Azure portal](https://portal.azure.com)  
- Kliknite “Create a resource” in poiščite “Static Web App”.  
- Kliknite “Create”.  

3. Konfigurirajte Static Web App  
- Osnovno: Naročnina: Izberite svojo Azure naročnino.  
- Skupina virov: Ustvarite novo skupino virov ali uporabite obstoječo.  
- Ime: Dajte ime svoji statični spletni aplikaciji.  
- Regija: Izberite regijo, ki je najbližja vašim uporabnikom.  

- #### Podrobnosti o namestitvi:  
- Vir: Izberite “GitHub”.  
- GitHub račun: Avtorizirajte Azure za dostop do vašega GitHub računa.  
- Organizacija: Izberite svojo GitHub organizacijo.  
- Repozitorij: Izberite repozitorij, ki vsebuje vašo statično spletno aplikacijo.  
- Branch: Izberite branch, iz katerega želite namestiti.  

- #### Podrobnosti o gradnji:  
- Prednastavitve gradnje: Izberite okvir, na katerem je vaša aplikacija zgrajena (npr. React, Angular, Vue itd.).  
- Lokacija aplikacije: Določite mapo, ki vsebuje kodo vaše aplikacije (npr. /, če je v korenu).  
- Lokacija API-ja: Če imate API, določite njegovo lokacijo (neobvezno).  
- Lokacija izhoda: Določite mapo, kjer je generiran izhod gradnje (npr. build ali dist).  

4. Pregled in ustvarjanje  
Preglejte svoje nastavitve in kliknite “Create”. Azure bo nastavil potrebne vire in ustvaril GitHub Actions workflow v vašem repozitoriju.

5. GitHub Actions Workflow  
Azure bo samodejno ustvaril GitHub Actions workflow datoteko v vašem repozitoriju (.github/workflows/azure-static-web-apps-<ime>.yml). Ta workflow bo upravljal proces gradnje in namestitve.

6. Spremljanje namestitve  
Pojdite na zavihek “Actions” v vašem GitHub repozitoriju.  
Videti bi morali, da se izvaja workflow. Ta workflow bo zgradil in namestil vašo statično spletno aplikacijo na Azure.  
Ko se workflow zaključi, bo vaša aplikacija na voljo na dodeljenem Azure URL-ju.

### Primer datoteke workflow

Tukaj je primer, kako bi lahko izgledala GitHub Actions workflow datoteka:  
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

### Dodatni viri  
- [Dokumentacija Azure Static Web Apps](https://learn.microsoft.com/azure/static-web-apps/getting-started)  
- [Dokumentacija GitHub Actions](https://docs.github.com/actions/use-cases-and-examples/deploying/deploying-to-azure-static-web-app)  

---

**Omejitev odgovornosti**:  
Ta dokument je bil preveden z uporabo storitve za strojno prevajanje [Co-op Translator](https://github.com/Azure/co-op-translator). Čeprav si prizadevamo za natančnost, vas prosimo, da upoštevate, da lahko avtomatizirani prevodi vsebujejo napake ali netočnosti. Izvirni dokument v njegovem izvirnem jeziku je treba obravnavati kot avtoritativni vir. Za ključne informacije priporočamo strokovno človeško prevajanje. Ne prevzemamo odgovornosti za morebitna nesporazumevanja ali napačne razlage, ki izhajajo iz uporabe tega prevoda.