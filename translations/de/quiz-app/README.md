<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "6d130dffca5db70d7e615f926cb1ad4c",
  "translation_date": "2025-09-03T21:53:28+00:00",
  "source_file": "quiz-app/README.md",
  "language_code": "de"
}
-->
# Quizfragen

Diese Quizfragen sind die Vor- und Nachbereitungsquiz für das ML-Curriculum unter https://aka.ms/ml-beginners

## Projektsetup

```
npm install
```

### Kompiliert und lädt für die Entwicklung neu

```
npm run serve
```

### Kompiliert und minimiert für die Produktion

```
npm run build
```

### Überprüft und behebt Dateien

```
npm run lint
```

### Konfiguration anpassen

Siehe [Konfigurationsreferenz](https://cli.vuejs.org/config/).

Credits: Dank an die ursprüngliche Version dieser Quiz-App: https://github.com/arpan45/simple-quiz-vue

## Deployment auf Azure

Hier ist eine Schritt-für-Schritt-Anleitung, um Ihnen den Einstieg zu erleichtern:

1. Forken Sie ein GitHub-Repository  
Stellen Sie sicher, dass der Code Ihrer statischen Web-App in Ihrem GitHub-Repository liegt. Forken Sie dieses Repository.

2. Erstellen Sie eine Azure Static Web App  
- Erstellen Sie ein [Azure-Konto](http://azure.microsoft.com)  
- Gehen Sie zum [Azure-Portal](https://portal.azure.com)  
- Klicken Sie auf „Create a resource“ und suchen Sie nach „Static Web App“.  
- Klicken Sie auf „Create“.  

3. Konfigurieren Sie die Static Web App  
- Grundlagen:  
  - Abonnement: Wählen Sie Ihr Azure-Abonnement aus.  
  - Ressourcengruppe: Erstellen Sie eine neue Ressourcengruppe oder verwenden Sie eine bestehende.  
  - Name: Geben Sie Ihrer statischen Web-App einen Namen.  
  - Region: Wählen Sie die Region, die Ihren Nutzern am nächsten liegt.  

- #### Deployment-Details:  
  - Quelle: Wählen Sie „GitHub“.  
  - GitHub-Konto: Autorisieren Sie Azure, auf Ihr GitHub-Konto zuzugreifen.  
  - Organisation: Wählen Sie Ihre GitHub-Organisation aus.  
  - Repository: Wählen Sie das Repository, das Ihre statische Web-App enthält.  
  - Branch: Wählen Sie den Branch, von dem Sie deployen möchten.  

- #### Build-Details:  
  - Build-Voreinstellungen: Wählen Sie das Framework, mit dem Ihre App erstellt wurde (z. B. React, Angular, Vue usw.).  
  - App-Standort: Geben Sie den Ordner an, der Ihren App-Code enthält (z. B. /, wenn er sich im Root befindet).  
  - API-Standort: Falls Sie eine API haben, geben Sie deren Standort an (optional).  
  - Output-Standort: Geben Sie den Ordner an, in dem die Build-Ausgabe generiert wird (z. B. build oder dist).  

4. Überprüfen und Erstellen  
Überprüfen Sie Ihre Einstellungen und klicken Sie auf „Create“. Azure wird die notwendigen Ressourcen einrichten und einen GitHub Actions-Workflow in Ihrem Repository erstellen.

5. GitHub Actions Workflow  
Azure erstellt automatisch eine GitHub Actions-Workflow-Datei in Ihrem Repository (.github/workflows/azure-static-web-apps-<name>.yml). Dieser Workflow übernimmt den Build- und Deployment-Prozess.

6. Überwachen Sie das Deployment  
Gehen Sie zum Tab „Actions“ in Ihrem GitHub-Repository.  
Sie sollten einen laufenden Workflow sehen. Dieser Workflow wird Ihre statische Web-App auf Azure erstellen und deployen.  
Sobald der Workflow abgeschlossen ist, ist Ihre App unter der bereitgestellten Azure-URL live.

### Beispiel-Workflow-Datei

Hier ist ein Beispiel, wie die GitHub Actions-Workflow-Datei aussehen könnte:  
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

### Zusätzliche Ressourcen  
- [Azure Static Web Apps Dokumentation](https://learn.microsoft.com/azure/static-web-apps/getting-started)  
- [GitHub Actions Dokumentation](https://docs.github.com/actions/use-cases-and-examples/deploying/deploying-to-azure-static-web-app)  

---

**Haftungsausschluss**:  
Dieses Dokument wurde mit dem KI-Übersetzungsdienst [Co-op Translator](https://github.com/Azure/co-op-translator) übersetzt. Obwohl wir uns um Genauigkeit bemühen, beachten Sie bitte, dass automatisierte Übersetzungen Fehler oder Ungenauigkeiten enthalten können. Das Originaldokument in seiner ursprünglichen Sprache sollte als maßgebliche Quelle betrachtet werden. Für kritische Informationen wird eine professionelle menschliche Übersetzung empfohlen. Wir übernehmen keine Haftung für Missverständnisse oder Fehlinterpretationen, die sich aus der Nutzung dieser Übersetzung ergeben.