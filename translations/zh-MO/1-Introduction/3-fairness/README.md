# 使用負責任的 AI 建構機器學習解決方案
 
![機器學習中負責任 AI 的概要以手繪筆記呈現](../../../../translated_images/zh-MO/ml-fairness.ef296ebec6afc98a.webp)
> 手繪筆記作者：[Tomomi Imura](https://www.twitter.com/girlie_mac)

## [課前測驗](https://ff-quizzes.netlify.app/en/ml/)
 
## 簡介

在這個課程中，你將開始了解機器學習如何影響我們的日常生活。即使在現在，系統和模型已經參與到日常決策任務中，例如健康診斷、貸款審核或詐騙偵測。因此，這些模型必須運作良好，才能提供值得信賴的結果。就像任何軟體應用一樣，AI 系統有時會未達期望或產生不理想的結果。這就是為什麼理解並解釋 AI 模型的行為至關重要的原因。

想像當你用來建構這些模型的數據缺少某些族群資料，比如種族、性別、政治觀點、宗教，或者不成比例地代表某些族群時，會發生什麼事？如果模型的輸出被解讀為偏好某些族群呢？這對應用會帶來什麼後果？此外，當模型產生不良結果且對人們有害時會怎樣？誰要為 AI 系統的行為負責？這些都是本課程將探討的問題。

本課程中，你將：

- 提高對機器學習中公平性及與公平相關傷害重要性的認知。
- 熟悉探索異常值和異常情境的實踐，以確保可靠性和安全性。
- 了解通過設計包容性系統來讓每個人都能受益的必要性。
- 探討保護資料及個人隱私與安全的重要性。
- 看見採用透明機制解釋 AI 模型行為的重要性。
- 注意建立 AI 系統信任所需的問責機制。

## 先修知識

作為先修，請先完成「負責任 AI 原則」學習路徑，並觀看以下相關主題影片：

透過此[學習路徑](https://docs.microsoft.com/learn/modules/responsible-ai-principles/?WT.mc_id=academic-77952-leestott)進一步了解負責任 AI

[![Microsoft 的負責任 AI 方法](https://img.youtube.com/vi/dnC8-uUZXSc/0.jpg)](https://youtu.be/dnC8-uUZXSc "Microsoft 的負責任 AI 方法")

> 🎥 點擊上方圖片觀看影片：Microsoft 的負責任 AI 方法

## 公平性

AI 系統應公平對待每個人，避免對類似群體產生不同影響。例如，當 AI 系統提供醫療治療建議、貸款申請或就業指導時，應對具有相似症狀、財務狀況或專業資格的每個人給出相同建議。我們每個人都帶有影響決策的天生偏見，而這些偏見也可能反映在用於訓練 AI 系統的數據中。這種偏差有時是無意中產生的，通常很難有意識地察覺到數據中引入了偏見。

**「不公平」** 指的是對某族群造成的負面影響，或「傷害」，例如基於種族、性別、年齡或殘障狀態所劃定的群體。主要的公平性相關傷害可分為：

- <strong>分配不公</strong>，例如性別或族裔之一相較於另一方被偏袒。
- <strong>服務質量</strong>。如果訓練數據只涵蓋某特定情境，然而現實更複雜，則會導致服務表現不佳。例如，有手用肥皂機無法感應深色皮膚的人。 [參考](https://gizmodo.com/why-cant-this-soap-dispenser-identify-dark-skin-1797931773)
- <strong>誹謗</strong>。不公正地批評和標籤某事物或某人。例如，某影像標籤技術惡名昭彰地將深膚色人的圖像標記為黑猩猩。
- <strong>過度或不足的代表性</strong>。某群體在某職業中不被看到，而任何持續推動此狀況的服務或功能都是有害的。
- <strong>刻板印象化</strong>。將特定群體與預設屬性連結。例如，英語與土耳其語的翻譯系統，因為詞彙帶有性別刻板印象而造成不準確。

![翻譯成土耳其語](../../../../translated_images/zh-MO/gender-bias-translate-en-tr.f185fd8822c2d437.webp)
> 翻譯成土耳其語

![再翻譯回英語](../../../../translated_images/zh-MO/gender-bias-translate-tr-en.4eee7e3cecb8c70e.webp)
> 再翻譯回英語

在設計與測試 AI 系統時，我們需要確保 AI 是公平的，不會被編程成做出偏見或歧視性的決定，而人類也被禁止做這類決策。確保 AI 及機器學習的公平性仍是一個複雜的社會技術挑戰。

### 可靠性與安全性

為了建立信任，AI 系統需要在正常及異常狀況下都能可靠、安全且一致地運作。重要的是要知道 AI 系統在各種情境下如何表現，特別是異常狀況。在建構 AI 解決方案時，必須大量關注如何處理 AI 解決方案可能遇到的多樣化狀況。例如，一輛自駕車必須將人的安全列為首要任務。因此，推動該車的 AI 需考慮所有可能遇到的場景，如夜間、雷雨或暴風雪、小孩奔跑穿越街道、寵物、道路施工等。AI 系統能多可靠安全地處理廣泛條件，反映了資料科學家或 AI 開發者在設計或測試時的預見程度。

> [🎥 點擊此處觀看影片：](https://www.microsoft.com/videoplayer/embed/RE4vvIl)

### 包容性

AI 系統應設計成能夠讓每個人都能參與和賦能。在設計與部署 AI 系統時，資料科學家與 AI 開發者會辨識並解決系統中可能無意排除人群的障礙。例如，全球有 10 億殘疾人士。隨著 AI 的進步，他們能更輕鬆地在日常生活中獲取廣泛的資訊與機會。通過排除障礙，創造了創新並開發出更佳用戶體驗的 AI 產品，有益於所有人。

> [🎥 點擊此處觀看影片：AI 包容性](https://www.microsoft.com/videoplayer/embed/RE4vl9v)

### 安全與隱私

AI 系統應安全且尊重個人隱私。人們對危及其隱私、資訊或生命安全的系統信任度較低。訓練機器學習模型時，我們依賴資料以取得最佳結果。這過程中必須考慮資料的來源和正確性。例如，資料是用戶提交或公開取得的？同時，使用資料時，開發能保護機密資訊且具抵抗攻擊能力的 AI 系統十分關鍵。隨著 AI 越來越普及，保護隱私並確保重要個人與商業資訊安全變得更為重要且複雜。隱私與資料安全問題在 AI 領域尤其需要嚴密關注，因為 AI 系統需存取資料才能對人民作出準確且明智的預測與決策。

> [🎥 點擊此處觀看影片：AI 的安全性](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- 身為產業，我們在隱私與安全方面已取得重大進展，特別是受到 GDPR（一般資料保護規則）等法規的推動。
- 然而，針對 AI 系統，我們必須承認在需要更多個人資料以提升系統個人化與效能與隱私之間的矛盾。
- 正如互聯網連結電腦的誕生，我們也看到 AI 相關的安全問題急速上升。
- 同時，AI 也被用於提升安全性。例如，現代多數防毒掃描器現在多由 AI 偵測法則驅動。
- 我們需確保資料科學流程與最新隱私及安全實務和諧融合。


### 透明度
AI 系統應該是可理解的。透明度的重要部分是解釋 AI 系統及其組件的行為。提升對 AI 系統的理解需要相關利害關係者明白系統如何及為何運作，以便辨識潛在的效能問題、安全及隱私疑慮、偏見、排他性措施或非預期結果。我們也相信使用 AI 系統的人應誠實且公開何時、為何及如何選擇部署，並說明所使用系統的限制。例如，銀行若用 AI 系統支持其消費貸款決策，就必須檢視結果並理解哪些數據影響系統建議。政府開始對產業中的 AI 進行規管，因此資料科學家與組織必須說明 AI 系統是否符合監管要求，尤其在出現不良結果時。

> [🎥 點擊此處觀看影片：AI 的透明度](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- 由於 AI 系統非常複雜，很難理解其運作方式及解讀結果。
- 這種缺乏理解影響了系統管理、運作和文件記錄的方式。
- 更重要的是，這種缺乏理解影響了根據系統輸出結果所做的決策。

### 問責制
 
設計與部署 AI 系統的人必須對系統的運作負起責任。問責制的必要性在敏感的使用技術如人臉識別中特別重要。近來，人臉識別技術的需求不斷增加，尤其是執法機構看到該技術在尋找失蹤兒童等用途的潛力。然而，這些技術也可能被政府用來危害公民的基本自由，例如對特定個體進行持續監視。因此，資料科學家與組織需要對其 AI 系統對個人或社會的影響負責。

[![領先 AI 研究人員警告人臉識別造成大規模監控](../../../../translated_images/zh-MO/accountability.41d8c0f4b85b6231.webp)](https://www.youtube.com/watch?v=Wldt8P5V6D0 "Microsoft 的負責任 AI 方法")

> 🎥 點擊上方圖片觀看影片：關於人臉識別引起的大規模監控警告

對我們這一代而言，作為帶來 AI 給社會的第一代，最大的問題之一是如何確保電腦能對人負責，以及如何確保設計電腦的人對其他所有人負責。

## 影響評估

在訓練機器學習模型之前，進行影響評估非常重要，藉此了解 AI 系統的目的；其預期用途；部署地點；以及誰會與系統互動。這些對評審者或測試者評估系統時了解哪些因素需考慮，以辨識潛在風險及預期後果很有幫助。

進行影響評估時的重點領域包括：

* <strong>對個人的不良影響</strong>。了解任何限制或要求、不支持的用途或已知限制可能影響系統效能，對防止系統以可能對個人造成傷害的方式被使用至關重要。
* <strong>資料需求</strong>。了解系統如何及在哪裡使用資料，讓評審者能探討需要注意的資料要求（如 GDPR 或 HIPAA 資料規則）。此外，檢查資料來源或數量是否足以支持訓練。
* <strong>影響摘要</strong>。彙整使用系統可能帶來的潛在傷害清單。在機器學習生命週期中，檢視已辨識問題是否已緩解或處理。
* <strong>每項六核心原則的適用目標</strong>。評估每項原則的目標是否達成及是否存在缺口。


## 負責任 AI 的除錯

與軟體應用除錯類似，除錯 AI 系統是辨識及解決系統問題的必要過程。有許多因素會影響模型未如期或不負責任地運作。多數傳統模型效能指標是模型整體表現的定量彙總，這不足以分析模型如何違反負責任 AI 原則。此外，機器學習模型是黑盒，難以理解其產出動因或錯誤時提供解釋。稍後本課程將學習如何利用負責任 AI 儀表板協助除錯 AI 系統。該儀表板為資料科學家與 AI 開發者提供整合工具，能執行：

* <strong>錯誤分析</strong>。辨識可能影響系統公平性或可靠性的模型錯誤分布。
* <strong>模型概覽</strong>。發現模型在各資料群體間效能差異的位置。
* <strong>資料分析</strong>。理解資料分布及辨識資料中可能導致公平性、包容性和可靠性問題的偏見。
* <strong>模型可解釋性</strong>。理解影響模型預測的因素，協助解釋模型行為，對透明度與問責制重要。


## 🚀 挑戰
 
為防止傷害的產生，應該：

- 讓從事系統開發的人員背景與觀點多元化
- 投資反映我們社會多元性的資料集
- 在整個機器學習生命週期中發展更好的方法來偵測並修正負責任 AI 中的問題

思考真實情境中模型不可信的例子，無論是在模型建構或使用過程中。我們還應該考慮什麼？

## [課後測驗](https://ff-quizzes.netlify.app/en/ml/)

## 複習與自學
 

在本課程中，您已學習了機器學習中公平與不公平概念的一些基礎知識。  
 
觀看此工作坊以更深入探討相關主題： 

- 追求負責任的 AI：將原則付諸實踐，講者 Besmira Nushi、Mehrnoosh Sameki 與 Amit Sharma

[![Responsible AI Toolbox: An open-source framework for building responsible AI](https://img.youtube.com/vi/tGgJCrA-MZU/0.jpg)](https://www.youtube.com/watch?v=tGgJCrA-MZU "RAI Toolbox: An open-source framework for building responsible AI")

> 🎥 點擊上方圖片觀看影片：RAI Toolbox：由 Besmira Nushi、Mehrnoosh Sameki 與 Amit Sharma 分享的開源負責任 AI 框架

另外，請參閱： 

- 微軟的 RAI 資源中心：[Responsible AI Resources – Microsoft AI](https://www.microsoft.com/ai/responsible-ai-resources?activetab=pivot1%3aprimaryr4) 

- 微軟的 FATE 研究團隊：[FATE: Fairness, Accountability, Transparency, and Ethics in AI - Microsoft Research](https://www.microsoft.com/research/theme/fate/) 

RAI 工具箱： 

- [Responsible AI Toolbox GitHub 存儲庫](https://github.com/microsoft/responsible-ai-toolbox)

閱讀關於 Azure 機器學習確保公平性的工具：

- [Azure Machine Learning](https://docs.microsoft.com/azure/machine-learning/concept-fairness-ml?WT.mc_id=academic-77952-leestott) 

## 作業

[探索 RAI 工具箱](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**免責聲明**：
本文件使用 AI 翻譯服務 [Co-op Translator](https://github.com/Azure/co-op-translator) 進行翻譯。雖然我們力求準確，但請注意，自動翻譯可能包含錯誤或不準確之處。原始文件的母語版本應被視為權威來源。對於重要資訊，建議尋求專業人工翻譯。我們不對因使用本翻譯而引起的任何誤解或曲解承擔責任。
<!-- CO-OP TRANSLATOR DISCLAIMER END -->