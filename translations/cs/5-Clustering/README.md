<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "b28a3a4911584062772c537b653ebbc7",
  "translation_date": "2025-09-04T23:57:41+00:00",
  "source_file": "5-Clustering/README.md",
  "language_code": "cs"
}
-->
# Modely shlukování pro strojové učení

Shlukování je úloha strojového učení, která se snaží najít objekty, jež si jsou navzájem podobné, a seskupit je do skupin nazývaných shluky. Co odlišuje shlukování od jiných přístupů ve strojovém učení, je to, že vše probíhá automaticky. Ve skutečnosti lze říci, že jde o opak učení s učitelem.

## Regionální téma: modely shlukování pro hudební vkus nigerijského publika 🎧

Nigerijské publikum je velmi rozmanité a má různorodý hudební vkus. Pomocí dat získaných ze Spotify (inspirováno [tímto článkem](https://towardsdatascience.com/country-wise-visual-analysis-of-music-taste-using-spotify-api-seaborn-in-python-77f5b749b421)) se podíváme na některé populární skladby v Nigérii. Tento dataset obsahuje údaje o různých skladbách, jako je skóre 'tanečnosti', 'akustičnosti', hlasitosti, 'mluvnosti', oblíbenosti a energie. Bude zajímavé objevit v těchto datech určité vzory!

![Gramofon](../../../5-Clustering/images/turntable.jpg)

> Foto od <a href="https://unsplash.com/@marcelalaskoski?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Marcely Laskoski</a> na <a href="https://unsplash.com/s/photos/nigerian-music?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
  
V této sérii lekcí objevíte nové způsoby analýzy dat pomocí technik shlukování. Shlukování je obzvláště užitečné, pokud váš dataset postrádá štítky. Pokud však štítky má, mohou být užitečnější klasifikační techniky, které jste se naučili v předchozích lekcích. Ale v případech, kdy chcete seskupit neoznačená data, je shlukování skvělým způsobem, jak objevit vzory.

> Existují užitečné nástroje s nízkým kódem, které vám mohou pomoci naučit se pracovat s modely shlukování. Vyzkoušejte [Azure ML pro tento úkol](https://docs.microsoft.com/learn/modules/create-clustering-model-azure-machine-learning-designer/?WT.mc_id=academic-77952-leestott)

## Lekce

1. [Úvod do shlukování](1-Visualize/README.md)
2. [Shlukování metodou K-Means](2-K-Means/README.md)

## Poděkování

Tyto lekce byly napsány s 🎶 od [Jen Looper](https://www.twitter.com/jenlooper) s užitečnými recenzemi od [Rishit Dagli](https://rishit_dagli) a [Muhammad Sakib Khan Inan](https://twitter.com/Sakibinan).

Dataset [Nigerijské skladby](https://www.kaggle.com/sootersaalu/nigerian-songs-spotify) byl získán z Kaggle a pochází ze Spotify.

Užitečné příklady K-Means, které pomohly při tvorbě této lekce, zahrnují tuto [analýzu kosatců](https://www.kaggle.com/bburns/iris-exploration-pca-k-means-and-gmm-clustering), tento [úvodní notebook](https://www.kaggle.com/prashant111/k-means-clustering-with-python) a tento [hypotetický příklad NGO](https://www.kaggle.com/ankandash/pca-k-means-clustering-hierarchical-clustering).

---

**Prohlášení**:  
Tento dokument byl přeložen pomocí služby pro automatický překlad [Co-op Translator](https://github.com/Azure/co-op-translator). Ačkoli se snažíme o přesnost, mějte prosím na paměti, že automatické překlady mohou obsahovat chyby nebo nepřesnosti. Původní dokument v jeho původním jazyce by měl být považován za autoritativní zdroj. Pro důležité informace doporučujeme profesionální lidský překlad. Neodpovídáme za žádné nedorozumění nebo nesprávné interpretace vyplývající z použití tohoto překladu.