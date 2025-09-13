<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "6d130dffca5db70d7e615f926cb1ad4c",
  "translation_date": "2025-09-05T00:40:57+00:00",
  "source_file": "quiz-app/README.md",
  "language_code": "fi"
}
-->
# Visailut

Nämä visailut ovat ML-kurssin ennakko- ja jälkiluennon visailuja osoitteessa https://aka.ms/ml-beginners

## Projektin asennus

```
npm install
```

### Kääntää ja päivittää automaattisesti kehitystä varten

```
npm run serve
```

### Kääntää ja minimoi tuotantoa varten

```
npm run build
```

### Tarkistaa ja korjaa tiedostoja

```
npm run lint
```

### Mukauta asetuksia

Katso [Asetusten viite](https://cli.vuejs.org/config/).

Kiitokset: Kiitos tämän visailusovelluksen alkuperäiselle versiolle: https://github.com/arpan45/simple-quiz-vue

## Julkaisu Azureen

Tässä vaiheittainen opas, joka auttaa sinua alkuun:

1. Haarauta GitHub-repositorio  
Varmista, että staattisen verkkosovelluksesi koodi on GitHub-repositoriossasi. Haarauta tämä repositorio.

2. Luo Azure Static Web App  
- Luo [Azure-tili](http://azure.microsoft.com)  
- Siirry [Azure-portaaliin](https://portal.azure.com)  
- Klikkaa “Luo resurssi” ja etsi “Static Web App”.  
- Klikkaa “Luo”.  

3. Määritä Static Web App  
- Perustiedot: Tilaukset: Valitse Azure-tilauksesi.  
- Resurssiryhmä: Luo uusi resurssiryhmä tai käytä olemassa olevaa.  
- Nimi: Anna staattiselle verkkosovelluksellesi nimi.  
- Alue: Valitse käyttäjiäsi lähinnä oleva alue.  

- #### Julkaisun tiedot:  
- Lähde: Valitse “GitHub”.  
- GitHub-tili: Valtuuta Azure käyttämään GitHub-tiliäsi.  
- Organisaatio: Valitse GitHub-organisaatiosi.  
- Repositorio: Valitse repositorio, joka sisältää staattisen verkkosovelluksesi.  
- Haara: Valitse haara, josta haluat julkaista.  

- #### Rakennuksen tiedot:  
- Rakennusasetukset: Valitse kehys, jolla sovelluksesi on rakennettu (esim. React, Angular, Vue jne.).  
- Sovelluksen sijainti: Määritä kansio, joka sisältää sovelluskoodisi (esim. / jos se on juurihakemistossa).  
- API:n sijainti: Jos sinulla on API, määritä sen sijainti (valinnainen).  
- Tulosteen sijainti: Määritä kansio, johon rakennuksen tuloste luodaan (esim. build tai dist).  

4. Tarkista ja luo  
Tarkista asetuksesi ja klikkaa “Luo”. Azure luo tarvittavat resurssit ja luo GitHub Actions -työnkulun repositoriossasi.

5. GitHub Actions -työnkulku  
Azure luo automaattisesti GitHub Actions -työnkulun tiedoston repositoriossasi (.github/workflows/azure-static-web-apps-<nimi>.yml). Tämä työnkulku hoitaa rakennus- ja julkaisuprosessin.

6. Seuraa julkaisua  
Siirry GitHub-repositoriosi “Toiminnot”-välilehteen.  
Näet työnkulun käynnissä. Tämä työnkulku rakentaa ja julkaisee staattisen verkkosovelluksesi Azureen.  
Kun työnkulku on valmis, sovelluksesi on käytettävissä annetussa Azure-URL-osoitteessa.

### Esimerkki työnkulun tiedostosta

Tässä esimerkki siitä, miltä GitHub Actions -työnkulun tiedosto voi näyttää:  
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

### Lisäresurssit  
- [Azure Static Web Apps -dokumentaatio](https://learn.microsoft.com/azure/static-web-apps/getting-started)  
- [GitHub Actions -dokumentaatio](https://docs.github.com/actions/use-cases-and-examples/deploying/deploying-to-azure-static-web-app)  

---

**Vastuuvapauslauseke**:  
Tämä asiakirja on käännetty käyttämällä tekoälypohjaista käännöspalvelua [Co-op Translator](https://github.com/Azure/co-op-translator). Vaikka pyrimme tarkkuuteen, huomioithan, että automaattiset käännökset voivat sisältää virheitä tai epätarkkuuksia. Alkuperäistä asiakirjaa sen alkuperäisellä kielellä tulisi pitää ensisijaisena lähteenä. Kriittisen tiedon osalta suositellaan ammattimaista ihmiskäännöstä. Emme ole vastuussa väärinkäsityksistä tai virhetulkinnoista, jotka johtuvat tämän käännöksen käytöstä.