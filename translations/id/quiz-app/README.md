<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "6d130dffca5db70d7e615f926cb1ad4c",
  "translation_date": "2025-09-05T19:47:53+00:00",
  "source_file": "quiz-app/README.md",
  "language_code": "id"
}
-->
# Kuis

Kuis ini adalah kuis sebelum dan sesudah kuliah untuk kurikulum ML di https://aka.ms/ml-beginners

## Pengaturan Proyek

```
npm install
```

### Kompilasi dan pemuatan ulang untuk pengembangan

```
npm run serve
```

### Kompilasi dan minimisasi untuk produksi

```
npm run build
```

### Linting dan perbaikan file

```
npm run lint
```

### Sesuaikan konfigurasi

Lihat [Referensi Konfigurasi](https://cli.vuejs.org/config/).

Kredit: Terima kasih kepada versi asli dari aplikasi kuis ini: https://github.com/arpan45/simple-quiz-vue

## Penerapan ke Azure

Berikut panduan langkah demi langkah untuk membantu Anda memulai:

1. Fork Repository GitHub  
Pastikan kode aplikasi web statis Anda ada di repository GitHub Anda. Fork repository ini.

2. Buat Azure Static Web App  
- Buat [akun Azure](http://azure.microsoft.com)  
- Pergi ke [portal Azure](https://portal.azure.com)  
- Klik “Create a resource” dan cari “Static Web App”.  
- Klik “Create”.  

3. Konfigurasi Static Web App  
- Dasar:  
  - Subscription: Pilih langganan Azure Anda.  
  - Resource Group: Buat grup sumber daya baru atau gunakan yang sudah ada.  
  - Name: Berikan nama untuk aplikasi web statis Anda.  
  - Region: Pilih wilayah yang paling dekat dengan pengguna Anda.  

- #### Detail Penerapan:  
  - Source: Pilih “GitHub”.  
  - GitHub Account: Otorisasi Azure untuk mengakses akun GitHub Anda.  
  - Organization: Pilih organisasi GitHub Anda.  
  - Repository: Pilih repository yang berisi aplikasi web statis Anda.  
  - Branch: Pilih cabang yang ingin Anda gunakan untuk penerapan.  

- #### Detail Build:  
  - Build Presets: Pilih framework yang digunakan untuk membangun aplikasi Anda (misalnya, React, Angular, Vue, dll.).  
  - App Location: Tentukan folder yang berisi kode aplikasi Anda (misalnya, / jika berada di root).  
  - API Location: Jika Anda memiliki API, tentukan lokasinya (opsional).  
  - Output Location: Tentukan folder tempat output build dihasilkan (misalnya, build atau dist).  

4. Tinjau dan Buat  
Tinjau pengaturan Anda dan klik “Create”. Azure akan menyiapkan sumber daya yang diperlukan dan membuat file workflow GitHub Actions di repository Anda.

5. Workflow GitHub Actions  
Azure akan secara otomatis membuat file workflow GitHub Actions di repository Anda (.github/workflows/azure-static-web-apps-<name>.yml). Workflow ini akan menangani proses build dan penerapan.

6. Pantau Penerapan  
Pergi ke tab “Actions” di repository GitHub Anda.  
Anda akan melihat workflow yang sedang berjalan. Workflow ini akan membangun dan menerapkan aplikasi web statis Anda ke Azure.  
Setelah workflow selesai, aplikasi Anda akan aktif di URL Azure yang diberikan.

### Contoh File Workflow

Berikut adalah contoh file workflow GitHub Actions:  
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

### Sumber Daya Tambahan  
- [Dokumentasi Azure Static Web Apps](https://learn.microsoft.com/azure/static-web-apps/getting-started)  
- [Dokumentasi GitHub Actions](https://docs.github.com/actions/use-cases-and-examples/deploying/deploying-to-azure-static-web-app)  

---

**Penafian**:  
Dokumen ini telah diterjemahkan menggunakan layanan penerjemahan AI [Co-op Translator](https://github.com/Azure/co-op-translator). Meskipun kami berusaha untuk memberikan hasil yang akurat, harap diingat bahwa terjemahan otomatis mungkin mengandung kesalahan atau ketidakakuratan. Dokumen asli dalam bahasa aslinya harus dianggap sebagai sumber yang otoritatif. Untuk informasi yang bersifat kritis, disarankan menggunakan jasa penerjemahan profesional oleh manusia. Kami tidak bertanggung jawab atas kesalahpahaman atau penafsiran yang keliru yang timbul dari penggunaan terjemahan ini.