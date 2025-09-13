<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "6d130dffca5db70d7e615f926cb1ad4c",
  "translation_date": "2025-09-03T23:48:12+00:00",
  "source_file": "quiz-app/README.md",
  "language_code": "es"
}
-->
# Cuestionarios

Estos cuestionarios son los cuestionarios previos y posteriores a las clases del currículo de ML en https://aka.ms/ml-beginners

## Configuración del proyecto

```
npm install
```

### Compila y recarga automáticamente para desarrollo

```
npm run serve
```

### Compila y minimiza para producción

```
npm run build
```

### Revisa y corrige archivos

```
npm run lint
```

### Personaliza la configuración

Consulta [Referencia de Configuración](https://cli.vuejs.org/config/).

Créditos: Gracias a la versión original de esta aplicación de cuestionarios: https://github.com/arpan45/simple-quiz-vue

## Desplegando en Azure

Aquí tienes una guía paso a paso para ayudarte a comenzar:

1. Haz un fork de un repositorio de GitHub  
Asegúrate de que el código de tu aplicación web estática esté en tu repositorio de GitHub. Haz un fork de este repositorio.

2. Crea una Aplicación Web Estática en Azure  
- Crea una [cuenta de Azure](http://azure.microsoft.com)  
- Ve al [portal de Azure](https://portal.azure.com)  
- Haz clic en “Crear un recurso” y busca “Aplicación Web Estática”.  
- Haz clic en “Crear”.  

3. Configura la Aplicación Web Estática  
- Básicos:  
  - Suscripción: Selecciona tu suscripción de Azure.  
  - Grupo de Recursos: Crea un nuevo grupo de recursos o utiliza uno existente.  
  - Nombre: Proporciona un nombre para tu aplicación web estática.  
  - Región: Elige la región más cercana a tus usuarios.  

- #### Detalles de Despliegue:  
  - Fuente: Selecciona “GitHub”.  
  - Cuenta de GitHub: Autoriza a Azure para acceder a tu cuenta de GitHub.  
  - Organización: Selecciona tu organización de GitHub.  
  - Repositorio: Elige el repositorio que contiene tu aplicación web estática.  
  - Rama: Selecciona la rama desde la que deseas desplegar.  

- #### Detalles de Construcción:  
  - Presets de Construcción: Elige el framework con el que está construida tu aplicación (por ejemplo, React, Angular, Vue, etc.).  
  - Ubicación de la Aplicación: Especifica la carpeta que contiene el código de tu aplicación (por ejemplo, / si está en la raíz).  
  - Ubicación de la API: Si tienes una API, especifica su ubicación (opcional).  
  - Ubicación de Salida: Especifica la carpeta donde se genera la salida de la construcción (por ejemplo, build o dist).  

4. Revisa y Crea  
Revisa tu configuración y haz clic en “Crear”. Azure configurará los recursos necesarios y creará un archivo de flujo de trabajo de GitHub Actions en tu repositorio.  

5. Flujo de Trabajo de GitHub Actions  
Azure creará automáticamente un archivo de flujo de trabajo de GitHub Actions en tu repositorio (.github/workflows/azure-static-web-apps-<nombre>.yml). Este flujo de trabajo manejará el proceso de construcción y despliegue.  

6. Monitorea el Despliegue  
Ve a la pestaña “Actions” en tu repositorio de GitHub.  
Deberías ver un flujo de trabajo en ejecución. Este flujo de trabajo construirá y desplegará tu aplicación web estática en Azure.  
Una vez que el flujo de trabajo se complete, tu aplicación estará en vivo en la URL proporcionada por Azure.  

### Ejemplo de Archivo de Flujo de Trabajo  

Aquí tienes un ejemplo de cómo podría verse el archivo de flujo de trabajo de GitHub Actions:  
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

### Recursos Adicionales  
- [Documentación de Aplicaciones Web Estáticas en Azure](https://learn.microsoft.com/azure/static-web-apps/getting-started)  
- [Documentación de GitHub Actions](https://docs.github.com/actions/use-cases-and-examples/deploying/deploying-to-azure-static-web-app)  

---

**Descargo de responsabilidad**:  
Este documento ha sido traducido utilizando el servicio de traducción automática [Co-op Translator](https://github.com/Azure/co-op-translator). Si bien nos esforzamos por lograr precisión, tenga en cuenta que las traducciones automáticas pueden contener errores o imprecisiones. El documento original en su idioma nativo debe considerarse como la fuente autorizada. Para información crítica, se recomienda una traducción profesional realizada por humanos. No nos hacemos responsables de malentendidos o interpretaciones erróneas que puedan surgir del uso de esta traducción.