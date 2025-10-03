<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "93fdaa0fd38836e50c4793e2f2f25e8b",
  "translation_date": "2025-10-03T11:08:08+00:00",
  "source_file": "AGENTS.md",
  "language_code": "tr"
}
-->
# AGENTS.md

## Proje Genel Bakış

Bu, **Yeni Başlayanlar için Makine Öğrenimi** adlı, Python (özellikle Scikit-learn) ve R kullanarak klasik makine öğrenimi kavramlarını kapsayan 12 haftalık, 26 derslik kapsamlı bir müfredattır. Depo, kendi hızınızda öğrenebileceğiniz, uygulamalı projeler, testler ve ödevler içeren bir kaynak olarak tasarlanmıştır. Her ders, dünya çapındaki farklı kültürlerden ve bölgelerden alınan gerçek dünya verileriyle ML kavramlarını keşfeder.

Ana bileşenler:
- **Eğitim İçeriği**: ML'ye giriş, regresyon, sınıflandırma, kümeleme, NLP, zaman serileri ve pekiştirmeli öğrenmeyi kapsayan 26 ders
- **Test Uygulaması**: Ders öncesi ve sonrası değerlendirmelerle Vue.js tabanlı test uygulaması
- **Çok Dilli Destek**: GitHub Actions aracılığıyla 40'tan fazla dile otomatik çeviri
- **Çift Dil Desteği**: Dersler hem Python (Jupyter not defterleri) hem de R (R Markdown dosyaları) olarak sunuluyor
- **Proje Tabanlı Öğrenme**: Her konu pratik projeler ve ödevler içerir

## Depo Yapısı

```
ML-For-Beginners/
├── 1-Introduction/         # ML basics, history, fairness, techniques
├── 2-Regression/          # Regression models with Python/R
├── 3-Web-App/            # Flask web app for ML model deployment
├── 4-Classification/      # Classification algorithms
├── 5-Clustering/         # Clustering techniques
├── 6-NLP/               # Natural Language Processing
├── 7-TimeSeries/        # Time series forecasting
├── 8-Reinforcement/     # Reinforcement learning
├── 9-Real-World/        # Real-world ML applications
├── quiz-app/           # Vue.js quiz application
├── translations/       # Auto-generated translations
└── sketchnotes/       # Visual learning aids
```

Her ders klasörü genellikle şunları içerir:
- `README.md` - Ana ders içeriği
- `notebook.ipynb` - Python Jupyter not defteri
- `solution/` - Çözüm kodu (Python ve R versiyonları)
- `assignment.md` - Uygulama alıştırmaları
- `images/` - Görsel kaynaklar

## Kurulum Komutları

### Python Dersleri İçin

Çoğu ders Jupyter not defterlerini kullanır. Gerekli bağımlılıkları yükleyin:

```bash
# Install Python 3.8+ if not already installed
python --version

# Install Jupyter
pip install jupyter

# Install common ML libraries
pip install scikit-learn pandas numpy matplotlib seaborn

# For specific lessons, check lesson-specific requirements
# Example: Web App lesson
pip install flask
```

### R Dersleri İçin

R dersleri `solution/R/` klasörlerinde `.rmd` veya `.ipynb` dosyaları olarak bulunur:

```bash
# Install R and required packages
# In R console:
install.packages(c("tidyverse", "tidymodels", "caret"))
```

### Test Uygulaması İçin

Test uygulaması `quiz-app/` dizininde bulunan bir Vue.js uygulamasıdır:

```bash
cd quiz-app
npm install
```

### Dokümantasyon Sitesi İçin

Dokümantasyonu yerel olarak çalıştırmak için:

```bash
# Install Docsify
npm install -g docsify-cli

# Serve from repository root
docsify serve

# Access at http://localhost:3000
```

## Geliştirme İş Akışı

### Ders Not Defterleriyle Çalışma

1. Ders dizinine gidin (ör. `2-Regression/1-Tools/`)
2. Jupyter not defterini açın:
   ```bash
   jupyter notebook notebook.ipynb
   ```
3. Ders içeriği ve alıştırmalar üzerinde çalışın
4. Gerekirse `solution/` klasöründeki çözümleri kontrol edin

### Python Geliştirme

- Dersler standart Python veri bilimi kütüphanelerini kullanır
- Etkileşimli öğrenme için Jupyter not defterleri
- Her dersin `solution/` klasöründe çözüm kodu mevcuttur

### R Geliştirme

- R dersleri `.rmd` formatındadır (R Markdown)
- Çözümler `solution/R/` alt dizinlerinde bulunur
- RStudio veya R çekirdeği ile Jupyter kullanarak R not defterlerini çalıştırabilirsiniz

### Test Uygulaması Geliştirme

```bash
cd quiz-app

# Start development server
npm run serve
# Access at http://localhost:8080

# Build for production
npm run build

# Lint and fix files
npm run lint
```

## Test Talimatları

### Test Uygulaması Testi

```bash
cd quiz-app

# Lint code
npm run lint

# Build to verify no errors
npm run build
```

**Not**: Bu öncelikle bir eğitim müfredatı deposudur. Ders içeriği için otomatik testler yoktur. Doğrulama şu yollarla yapılır:
- Ders alıştırmalarını tamamlama
- Not defteri hücrelerini başarıyla çalıştırma
- Çözümlerdeki beklenen sonuçlarla çıktıyı karşılaştırma

## Kod Stili Yönergeleri

### Python Kodu
- PEP 8 stil yönergelerini takip edin
- Açık ve açıklayıcı değişken adları kullanın
- Karmaşık işlemler için yorumlar ekleyin
- Jupyter not defterlerinde kavramları açıklayan markdown hücreleri bulunmalıdır

### JavaScript/Vue.js (Test Uygulaması)
- Vue.js stil rehberini takip eder
- `quiz-app/package.json` içinde ESLint yapılandırması
- Sorunları kontrol etmek ve otomatik düzeltmek için `npm run lint` çalıştırın

### Dokümantasyon
- Markdown dosyaları açık ve iyi yapılandırılmış olmalıdır
- Çitlenmiş kod bloklarında kod örnekleri ekleyin
- Dahili referanslar için göreceli bağlantılar kullanın
- Mevcut biçimlendirme kurallarını takip edin

## Derleme ve Dağıtım

### Test Uygulaması Dağıtımı

Test uygulaması Azure Static Web Apps'e dağıtılabilir:

1. **Ön Koşullar**:
   - Azure hesabı
   - GitHub deposu (zaten çatallanmış)

2. **Azure'a Dağıtım**:
   - Azure Static Web App kaynağı oluşturun
   - GitHub deposuna bağlanın
   - Uygulama konumunu ayarlayın: `/quiz-app`
   - Çıktı konumunu ayarlayın: `dist`
   - Azure otomatik olarak GitHub Actions iş akışı oluşturur

3. **GitHub Actions İş Akışı**:
   - İş akışı dosyası `.github/workflows/azure-static-web-apps-*.yml` konumunda oluşturulur
   - Ana dalda yapılan değişikliklerde otomatik olarak derlenir ve dağıtılır

### Dokümantasyon PDF

Dokümantasyondan PDF oluşturun:

```bash
npm install
npm run convert
```

## Çeviri İş Akışı

**Önemli**: Çeviriler GitHub Actions aracılığıyla Co-op Translator kullanılarak otomatik yapılır.

- Çeviriler `main` dalına yapılan değişikliklerde otomatik olarak oluşturulur
- **İçeriği manuel olarak çevirmeyin** - sistem bunu otomatik olarak yapar
- İş akışı `.github/workflows/co-op-translator.yml` içinde tanımlanmıştır
- Çeviri için Azure AI/OpenAI hizmetlerini kullanır
- 40'tan fazla dili destekler

## Katkı Yönergeleri

### İçerik Katkıcıları İçin

1. **Depoyu çatallayın** ve bir özellik dalı oluşturun
2. **Ders içeriğinde değişiklik yapın** (ders ekliyorsanız/güncelliyorsanız)
3. **Çevrilmiş dosyaları değiştirmeyin** - bunlar otomatik olarak oluşturulur
4. **Kodunuzu test edin** - tüm not defteri hücrelerinin başarıyla çalıştığından emin olun
5. **Bağlantıların ve görsellerin** doğru çalıştığını doğrulayın
6. **Açıklayıcı bir açıklama ile bir çekme isteği gönderin**

### Çekme İsteği Yönergeleri

- **Başlık formatı**: `[Bölüm] Değişikliklerin kısa açıklaması`
  - Örnek: `[Regression] Ders 5'teki yazım hatasını düzelt`
  - Örnek: `[Quiz-App] Bağımlılıkları güncelle`
- **Göndermeden önce**:
  - Tüm not defteri hücrelerinin hatasız çalıştığından emin olun
  - Test uygulamasını değiştiriyorsanız `npm run lint` çalıştırın
  - Markdown biçimlendirmesini doğrulayın
  - Yeni kod örneklerini test edin
- **Çekme isteği şunları içermelidir**:
  - Değişikliklerin açıklaması
  - Değişikliklerin nedeni
  - UI değişiklikleri varsa ekran görüntüleri
- **Davranış Kuralları**: [Microsoft Açık Kaynak Davranış Kuralları](CODE_OF_CONDUCT.md) takip edin
- **CLA**: Katkı Sağlayıcı Lisans Sözleşmesini imzalamanız gerekecek

## Ders Yapısı

Her ders tutarlı bir deseni takip eder:

1. **Ders öncesi test** - Temel bilgileri test edin
2. **Ders içeriği** - Yazılı talimatlar ve açıklamalar
3. **Kod gösterimleri** - Not defterlerinde uygulamalı örnekler
4. **Bilgi kontrolleri** - Anlamayı doğrulama
5. **Zorluk** - Kavramları bağımsız olarak uygulama
6. **Ödev** - Genişletilmiş pratik
7. **Ders sonrası test** - Öğrenme sonuçlarını değerlendirme

## Yaygın Komutlar Referansı

```bash
# Python/Jupyter
jupyter notebook                    # Start Jupyter server
jupyter notebook notebook.ipynb     # Open specific notebook
pip install -r requirements.txt     # Install dependencies (where available)

# Quiz App
cd quiz-app
npm install                        # Install dependencies
npm run serve                      # Development server
npm run build                      # Production build
npm run lint                       # Lint and fix

# Documentation
docsify serve                      # Serve documentation locally
npm run convert                    # Generate PDF

# Git workflow
git checkout -b feature/my-change  # Create feature branch
git add .                         # Stage changes
git commit -m "Description"       # Commit changes
git push origin feature/my-change # Push to remote
```

## Ek Kaynaklar

- **Microsoft Learn Koleksiyonu**: [Yeni Başlayanlar için ML modülleri](https://learn.microsoft.com/en-us/collections/qrqzamz1nn2wx3?WT.mc_id=academic-77952-bethanycheum)
- **Test Uygulaması**: [Çevrimiçi testler](https://ff-quizzes.netlify.app/en/ml/)
- **Tartışma Panosu**: [GitHub Tartışmaları](https://github.com/microsoft/ML-For-Beginners/discussions)
- **Video Anlatımları**: [YouTube Oynatma Listesi](https://aka.ms/ml-beginners-videos)

## Temel Teknolojiler

- **Python**: ML dersleri için birincil dil (Scikit-learn, Pandas, NumPy, Matplotlib)
- **R**: tidyverse, tidymodels, caret kullanarak alternatif uygulama
- **Jupyter**: Python dersleri için etkileşimli not defterleri
- **R Markdown**: R dersleri için belgeler
- **Vue.js 3**: Test uygulaması çerçevesi
- **Flask**: ML modeli dağıtımı için web uygulama çerçevesi
- **Docsify**: Dokümantasyon sitesi oluşturucu
- **GitHub Actions**: CI/CD ve otomatik çeviriler

## Güvenlik Dikkatleri

- **Kodda gizli bilgiler yok**: API anahtarlarını veya kimlik bilgilerini asla kodda paylaşmayın
- **Bağımlılıklar**: npm ve pip paketlerini güncel tutun
- **Kullanıcı girdisi**: Flask web uygulama örnekleri temel giriş doğrulaması içerir
- **Hassas veri**: Örnek veri setleri kamuya açık ve hassas olmayan verilerdir

## Sorun Giderme

### Jupyter Not Defterleri

- **Çekirdek sorunları**: Hücreler takılırsa çekirdeği yeniden başlatın: Çekirdek → Yeniden Başlat
- **İçe aktarma hataları**: Gerekli tüm paketlerin pip ile yüklü olduğundan emin olun
- **Yol sorunları**: Not defterlerini bulundukları dizinden çalıştırın

### Test Uygulaması

- **npm install başarısız**: npm önbelleğini temizleyin: `npm cache clean --force`
- **Port çakışmaları**: Portu şu şekilde değiştirin: `npm run serve -- --port 8081`
- **Derleme hataları**: `node_modules` klasörünü silin ve yeniden yükleyin: `rm -rf node_modules && npm install`

### R Dersleri

- **Paket bulunamadı**: Şu komutla yükleyin: `install.packages("package-name")`
- **RMarkdown renderleme**: rmarkdown paketinin yüklü olduğundan emin olun
- **Çekirdek sorunları**: Jupyter için IRkernel'i yüklemeniz gerekebilir

## Proje Özel Notları

- Bu öncelikle bir **öğrenme müfredatı**, üretim kodu değil
- Odak noktası **ML kavramlarını anlamak** ve uygulamalı pratik yapmak
- Kod örnekleri **açıklık öncelikli**, optimizasyon değil
- Çoğu ders **bağımsızdır** ve tek başına tamamlanabilir
- **Çözümler sağlanır**, ancak öğrenciler önce alıştırmaları denemelidir
- Depo **Docsify** kullanır, web dokümantasyonu için derleme adımı gerekmez
- **Sketchnotes** kavramların görsel özetlerini sağlar
- **Çok dilli destek** içeriği küresel olarak erişilebilir kılar

---

**Feragatname**:  
Bu belge, AI çeviri hizmeti [Co-op Translator](https://github.com/Azure/co-op-translator) kullanılarak çevrilmiştir. Doğruluk için çaba göstersek de, otomatik çevirilerin hata veya yanlışlık içerebileceğini lütfen unutmayın. Belgenin orijinal dili, yetkili kaynak olarak kabul edilmelidir. Kritik bilgiler için profesyonel insan çevirisi önerilir. Bu çevirinin kullanımından kaynaklanan yanlış anlamalar veya yanlış yorumlamalar için sorumluluk kabul etmiyoruz.