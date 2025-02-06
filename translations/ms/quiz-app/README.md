# Kuiz

Kuiz-kuiz ini adalah kuiz sebelum dan selepas kuliah untuk kurikulum ML di https://aka.ms/ml-beginners

## Persediaan Projek

```
npm install
```

### Kompil dan muat semula secara langsung untuk pembangunan

```
npm run serve
```

### Kompil dan kecilkan untuk produksi

```
npm run build
```

### Lint dan betulkan fail

```
npm run lint
```

### Sesuaikan konfigurasi

Lihat [Rujukan Konfigurasi](https://cli.vuejs.org/config/).

Kredit: Terima kasih kepada versi asal aplikasi kuiz ini: https://github.com/arpan45/simple-quiz-vue

## Melancarkan ke Azure

Berikut adalah panduan langkah demi langkah untuk membantu anda memulakan:

1. Fork Repositori GitHub
Pastikan kod aplikasi web statik anda berada dalam repositori GitHub anda. Fork repositori ini.

2. Buat Aplikasi Web Statik Azure
- Buat akaun [Azure](http://azure.microsoft.com)
- Pergi ke [portal Azure](https://portal.azure.com) 
- Klik "Create a resource" dan cari "Static Web App".
- Klik "Create".

3. Konfigurasikan Aplikasi Web Statik
- Asas: Langganan: Pilih langganan Azure anda.
- Kumpulan Sumber: Buat kumpulan sumber baru atau gunakan yang sedia ada.
- Nama: Berikan nama untuk aplikasi web statik anda.
- Wilayah: Pilih wilayah yang paling dekat dengan pengguna anda.

- #### Butiran Pelancaran:
- Sumber: Pilih "GitHub".
- Akaun GitHub: Benarkan Azure mengakses akaun GitHub anda.
- Organisasi: Pilih organisasi GitHub anda.
- Repositori: Pilih repositori yang mengandungi aplikasi web statik anda.
- Cabang: Pilih cabang yang anda ingin lancarkan.

- #### Butiran Pembinaan:
- Pratetap Pembinaan: Pilih rangka kerja yang digunakan oleh aplikasi anda (contoh: React, Angular, Vue, dsb.).
- Lokasi Aplikasi: Nyatakan folder yang mengandungi kod aplikasi anda (contoh: / jika berada di akar).
- Lokasi API: Jika anda mempunyai API, nyatakan lokasinya (pilihan).
- Lokasi Output: Nyatakan folder di mana output pembinaan dijana (contoh: build atau dist).

4. Semak dan Buat
Semak tetapan anda dan klik "Create". Azure akan menyediakan sumber yang diperlukan dan membuat aliran kerja GitHub Actions dalam repositori anda.

5. Aliran Kerja GitHub Actions
Azure akan secara automatik membuat fail aliran kerja GitHub Actions dalam repositori anda (.github/workflows/azure-static-web-apps-<name>.yml). Aliran kerja ini akan mengendalikan proses pembinaan dan pelancaran.

6. Pantau Pelancaran
Pergi ke tab "Actions" dalam repositori GitHub anda.
Anda sepatutnya melihat aliran kerja sedang berjalan. Aliran kerja ini akan membina dan melancarkan aplikasi web statik anda ke Azure.
Setelah aliran kerja selesai, aplikasi anda akan hidup di URL Azure yang disediakan.

### Fail Aliran Kerja Contoh

Berikut adalah contoh bagaimana fail aliran kerja GitHub Actions mungkin kelihatan:
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

### Sumber Tambahan
- [Dokumentasi Azure Static Web Apps](https://learn.microsoft.com/azure/static-web-apps/getting-started)
- [Dokumentasi GitHub Actions](https://docs.github.com/actions/use-cases-and-examples/deploying/deploying-to-azure-static-web-app)

**Penafian**:
Dokumen ini telah diterjemahkan menggunakan perkhidmatan terjemahan AI berasaskan mesin. Walaupun kami berusaha untuk ketepatan, sila maklum bahawa terjemahan automatik mungkin mengandungi kesilapan atau ketidaktepatan. Dokumen asal dalam bahasa asalnya harus dianggap sebagai sumber yang berwibawa. Untuk maklumat penting, terjemahan manusia profesional disyorkan. Kami tidak bertanggungjawab atas sebarang salah faham atau salah tafsir yang timbul daripada penggunaan terjemahan ini.