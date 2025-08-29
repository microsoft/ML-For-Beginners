# Quizze

Diese Quizze sind die Vor- und Nachlese-Quizze für das ML-Curriculum unter https://aka.ms/ml-beginners

## Projektsetup

```
npm install
```

### Kompiliert und lädt für die Entwicklung neu

```
npm run serve
```

### Kompiliert und minifiziert für die Produktion

```
npm run build
```

### Überprüft und behebt Dateien

```
npm run lint
```

### Konfiguration anpassen

Siehe [Konfigurationsreferenz](https://cli.vuejs.org/config/).

Credits: Danke an die Originalversion dieser Quiz-App: https://github.com/arpan45/simple-quiz-vue

## Bereitstellung auf Azure

Hier ist eine Schritt-für-Schritt-Anleitung, um Ihnen den Einstieg zu erleichtern:

1. Forken Sie ein GitHub-Repository
Stellen Sie sicher, dass Ihr Code für die statische Web-App in Ihrem GitHub-Repository ist. Forken Sie dieses Repository.

2. Erstellen Sie eine Azure Static Web App
- Erstellen Sie ein [Azure-Konto](http://azure.microsoft.com)
- Gehen Sie zum [Azure-Portal](https://portal.azure.com) 
- Klicken Sie auf „Ressource erstellen“ und suchen Sie nach „Static Web App“.
- Klicken Sie auf „Erstellen“.

3. Konfigurieren Sie die Static Web App
- Grundlagen: Abonnement: Wählen Sie Ihr Azure-Abonnement aus.
- Ressourcengruppe: Erstellen Sie eine neue Ressourcengruppe oder verwenden Sie eine vorhandene.
- Name: Geben Sie einen Namen für Ihre statische Web-App an.
- Region: Wählen Sie die Region, die Ihren Benutzern am nächsten ist.

- #### Bereitstellungsdetails:
- Quelle: Wählen Sie „GitHub“.
- GitHub-Konto: Autorisieren Sie Azure, auf Ihr GitHub-Konto zuzugreifen.
- Organisation: Wählen Sie Ihre GitHub-Organisation aus.
- Repository: Wählen Sie das Repository aus, das Ihre statische Web-App enthält.
- Branch: Wählen Sie den Branch aus, von dem Sie bereitstellen möchten.

- #### Build-Details:
- Build-Voreinstellungen: Wählen Sie das Framework, mit dem Ihre App erstellt wurde (z. B. React, Angular, Vue usw.).
- App-Standort: Geben Sie den Ordner an, der Ihren App-Code enthält (z. B. / wenn es im Stammverzeichnis ist).
- API-Standort: Wenn Sie eine API haben, geben Sie deren Standort an (optional).
- Ausgabestandort: Geben Sie den Ordner an, in dem die Build-Ausgabe generiert wird (z. B. build oder dist).

4. Überprüfen und Erstellen
Überprüfen Sie Ihre Einstellungen und klicken Sie auf „Erstellen“. Azure richtet die erforderlichen Ressourcen ein und erstellt einen GitHub Actions-Workflow in Ihrem Repository.

5. GitHub Actions Workflow
Azure erstellt automatisch eine GitHub Actions-Workflow-Datei in Ihrem Repository (.github/workflows/azure-static-web-apps-<name>.yml). Dieser Workflow kümmert sich um den Build- und Bereitstellungsprozess.

6. Überwachen der Bereitstellung
Gehen Sie zum Tab „Aktionen“ in Ihrem GitHub-Repository.
Sie sollten einen laufenden Workflow sehen. Dieser Workflow wird Ihre statische Web-App auf Azure erstellen und bereitstellen.
Sobald der Workflow abgeschlossen ist, wird Ihre App unter der angegebenen Azure-URL live sein.

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
- [Dokumentation zu Azure Static Web Apps](https://learn.microsoft.com/azure/static-web-apps/getting-started)
- [Dokumentation zu GitHub Actions](https://docs.github.com/actions/use-cases-and-examples/deploying/deploying-to-azure-static-web-app)

**Haftungsausschluss**:  
Dieses Dokument wurde mit maschinellen KI-Übersetzungsdiensten übersetzt. Obwohl wir uns um Genauigkeit bemühen, beachten Sie bitte, dass automatisierte Übersetzungen Fehler oder Ungenauigkeiten enthalten können. Das Originaldokument in seiner ursprünglichen Sprache sollte als die maßgebliche Quelle betrachtet werden. Für kritische Informationen wird eine professionelle menschliche Übersetzung empfohlen. Wir übernehmen keine Verantwortung für Missverständnisse oder Fehlinterpretationen, die aus der Verwendung dieser Übersetzung entstehen.