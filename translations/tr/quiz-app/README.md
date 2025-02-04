# Quizler

Bu quizler, https://aka.ms/ml-beginners adresindeki ML müfredatının ders öncesi ve sonrası quizleridir.

## Proje Kurulumu

```
npm install
```

### Geliştirme için derler ve sıcak yükler

```
npm run serve
```

### Üretim için derler ve küçültür

```
npm run build
```

### Dosyaları kontrol eder ve düzeltir

```
npm run lint
```

### Yapılandırmayı Özelleştir

[Configuration Reference](https://cli.vuejs.org/config/) adresine bakın.

Teşekkürler: Bu quiz uygulamasının orijinal versiyonuna teşekkürler: https://github.com/arpan45/simple-quiz-vue

## Azure'a Dağıtım

Başlamanıza yardımcı olacak adım adım bir rehber:

1. Bir GitHub Deposunu Çatallayın
Statik web uygulamanızın kodunun GitHub deponuzda olduğundan emin olun. Bu depoyu çatallayın.

2. Bir Azure Statik Web Uygulaması Oluşturun
- [Azure hesabı](http://azure.microsoft.com) oluşturun
- [Azure portalına](https://portal.azure.com) gidin
- "Kaynak oluştur" seçeneğine tıklayın ve "Statik Web Uygulaması" arayın.
- "Oluştur" butonuna tıklayın.

3. Statik Web Uygulamasını Yapılandırın
- Temel Bilgiler: Abonelik: Azure aboneliğinizi seçin.
- Kaynak Grubu: Yeni bir kaynak grubu oluşturun veya mevcut birini kullanın.
- Ad: Statik web uygulamanız için bir ad girin.
- Bölge: Kullanıcılarınıza en yakın bölgeyi seçin.

- #### Dağıtım Detayları:
- Kaynak: "GitHub"ı seçin.
- GitHub Hesabı: Azure'un GitHub hesabınıza erişmesine izin verin.
- Organizasyon: GitHub organizasyonunuzu seçin.
- Depo: Statik web uygulamanızı içeren depoyu seçin.
- Dal: Hangi daldan dağıtım yapacağınızı seçin.

- #### Yapı Detayları:
- Yapı Ön Ayarları: Uygulamanızın hangi çerçeve ile oluşturulduğunu seçin (örneğin, React, Angular, Vue, vb.).
- Uygulama Konumu: Uygulama kodunuzu içeren klasörü belirtin (örneğin, kökteyse /).
- API Konumu: Bir API'niz varsa, konumunu belirtin (isteğe bağlı).
- Çıktı Konumu: Yapı çıktısının oluşturulduğu klasörü belirtin (örneğin, build veya dist).

4. Gözden Geçirin ve Oluşturun
Ayarlarınızı gözden geçirin ve "Oluştur" butonuna tıklayın. Azure gerekli kaynakları ayarlayacak ve deponuza bir GitHub Actions iş akışı oluşturacaktır.

5. GitHub Actions İş Akışı
Azure, deponuzda otomatik olarak bir GitHub Actions iş akışı dosyası oluşturacaktır (.github/workflows/azure-static-web-apps-<name>.yml). Bu iş akışı yapı ve dağıtım sürecini yönetecektir.

6. Dağıtımı İzleyin
GitHub deponuzdaki "Actions" sekmesine gidin.
Bir iş akışının çalıştığını görmelisiniz. Bu iş akışı, statik web uygulamanızı Azure'a yapılandıracak ve dağıtacaktır.
İş akışı tamamlandığında, uygulamanız sağlanan Azure URL'sinde canlı olacaktır.

### Örnek İş Akışı Dosyası

İşte GitHub Actions iş akışı dosyasının nasıl görünebileceğine dair bir örnek:
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

### Ek Kaynaklar
- [Azure Statik Web Uygulamaları Dokümantasyonu](https://learn.microsoft.com/azure/static-web-apps/getting-started)
- [GitHub Actions Dokümantasyonu](https://docs.github.com/actions/use-cases-and-examples/deploying/deploying-to-azure-static-web-app)

**Feragatname**: 
Bu belge, makine tabanlı yapay zeka çeviri hizmetleri kullanılarak çevrilmiştir. Doğruluk için çaba göstersek de, otomatik çevirilerin hata veya yanlışlıklar içerebileceğini lütfen unutmayın. Belgenin orijinal diliyle yazılmış hali, yetkili kaynak olarak kabul edilmelidir. Kritik bilgiler için profesyonel insan çevirisi önerilir. Bu çevirinin kullanımından kaynaklanan herhangi bir yanlış anlama veya yanlış yorumlamadan sorumlu değiliz.