# 用負責任的 AI 建立機器學習解決方案
 
![機器學習中負責任 AI 的概要示意圖](../../../../translated_images/zh-HK/ml-fairness.ef296ebec6afc98a.webp)
> 示意圖來源：[Tomomi Imura](https://www.twitter.com/girlie_mac)

## [課前測驗](https://ff-quizzes.netlify.app/en/ml/)
 
## 簡介

在本課程中，你將開始了解機器學習如何影響並正在影響我們的日常生活。即使是現在，系統和模型已經參與了每天的決策任務，例如醫療診斷、貸款審批或詐騙偵測。因此，確保這些模型運作良好，提供值得信賴的結果非常重要。就像任何軟件應用程序一樣，AI 系統也可能未達預期或產生不理想的結果。因此，理解並解釋 AI 模型的行為至關重要。

想像一下，當你用來建立這些模型的資料缺乏某些族群（如種族、性別、政治觀點、宗教）或過度代表某些族群時會發生什麼。當模型的輸出被解讀為偏袒某些族群時，應用會有什麼後果？此外，當模型產生不良結果並對人們造成傷害時會怎樣？誰應該對 AI 系統的行為負責？這些都是我們在本課程中會探討的問題。

本課程中，你將：

- 提升對機器學習中公平性及公平性相關傷害重要性的認識。
- 熟悉探索異常值與異常情況的實踐，以確保可靠性和安全性。
- 理解設計包容性系統以使每個人都能受惠的必要性。
- 探討保護資料及人的隱私與安全的重要性。
- 了解採用透明方法解釋 AI 模型行為的重要性。
- 謹記責任制對建立 AI 系統信任的重要性。

## 先決條件

請先完成「負責任 AI 原則」學習路徑並觀看以下主題相關影片：

透過此[學習路徑](https://docs.microsoft.com/learn/modules/responsible-ai-principles/?WT.mc_id=academic-77952-leestott)深入了解負責任的 AI。

[![Microsoft's Approach to Responsible AI](https://img.youtube.com/vi/dnC8-uUZXSc/0.jpg)](https://youtu.be/dnC8-uUZXSc "Microsoft's Approach to Responsible AI")

> 🎥 點擊上方圖片觀賞影片：Microsoft 的負責任 AI 方法

## 公平性

AI 系統應公平對待每個人，避免以不同方式影響相似群體。例如，當 AI 系統提供醫療治療建議、貸款申請或就業建議時，應對所有具有相似症狀、財務狀況或專業資格的人做出相同的建議。我們每個人身上皆帶有影響決策與行動的固有偏見。這些偏見可能反映在用來訓練 AI 系統的資料中。這種偏見有時是無意中造成的，往往難以有意識地察覺自己在資料中引入了偏見。

<strong>「不公平」</strong>涵蓋對特定族群（如基於種族、性別、年齡或殘疾狀況定義的群體）造成的負面影響或「傷害」。主要的公平性相關傷害可分為：

- <strong>分配不公</strong>，例如偏袒某性別或族裔。
- <strong>服務品質</strong>。若訓練數據只涵蓋特定場景，但現實更複雜，會導致服務表現不佳。例如一款洗手液分配器無法感應深色皮膚的人。[參考](https://gizmodo.com/why-cant-this-soap-dispenser-identify-dark-skin-1797931773)
- <strong>污名化</strong>。不公平地批評和標籤某人或某物。例如，有影像標籤技術惡名昭彰地將深色皮膚人的影像誤標為猩猩。
- <strong>過度或不足代表</strong>。例如某團體在某職業中鮮少出現，而任何持續促進該情況的服務或功能都會造成傷害。
- <strong>刻板印象</strong>。將特定群體與預設屬性聯結。例如，英土語言翻譯系統可能因有男女刻板聯想的詞語而產生不準確。

![翻譯成土耳其語](../../../../translated_images/zh-HK/gender-bias-translate-en-tr.f185fd8822c2d437.webp)
> 翻譯成土耳其語

![再翻譯回英文](../../../../translated_images/zh-HK/gender-bias-translate-tr-en.4eee7e3cecb8c70e.webp)
> 再翻譯回英文

在設計和測試 AI 系統時，我們需要確保 AI 公平且未被編程以做出偏見或歧視決策，這些也是人類被禁止的行為。保證 AI 和機器學習的公平性依然是個複雜的社會技術難題。

### 可靠性與安全性

為建立信任，AI 系統需在正常與異常條件下均可靠、安全且一致。了解 AI 系統在各種情況，尤其是異常值下的行為至關重要。建立 AI 解決方案時，需高度關注如何處理 AI 解決方案可能遇到的各種情況。例如，自駕車必須將人員安全擺在首位。因此，驅動該車的 AI 需要考慮所有可能遇到的場景，如夜間、雷暴大風雪、街上奔跑的孩童、寵物、道路施工等。AI 系統能可靠且安全處理多樣情況，反映出資料科學家或 AI 開發者在設計或測試系統時的預期程度。

> [🎥 點擊此處觀看影片：](https://www.microsoft.com/videoplayer/embed/RE4vvIl)

### 包容性

AI 系統應設計為能讓每個人參與並獲得賦能。資料科學家與 AI 開發者在設計和實作 AI 系統時會辨識並解決可能無意中排除某些人的潛在障礙。例如，全球有十億名身心障礙者，隨著 AI 的進步，他們能更輕鬆取得資訊和機會。克服障礙能創造創新機會，開發出改善體驗且惠及所有人的 AI 產品。

> [🎥 點擊此處觀看影片：AI 的包容性](https://www.microsoft.com/videoplayer/embed/RE4vl9v)

### 安全與隱私

AI 系統應安全且尊重個人隱私。人們對那些危及隱私、資訊或生命的系統信任度較低。訓練機器學習模型時，我們仰賴資料來產生最佳結果，因此需考慮資料的來源與完整性。例如，資料是用戶提交的還是公開可用？在處理資料時，也必須開發能保護機密資訊並抵禦攻擊的 AI 系統。隨著 AI 越來越普及，保護隱私和資訊安全變得更為重要且複雜。對 AI 而言，隱私與資料安全議題需特別注意，因為 AI 依賴資料來做出準確且有根據的預測和決策。

> [🎥 點擊此處觀看影片：AI 中的安全](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- 我們產業在隱私及安全方面已取得重大進展，受 GDPR（一般資料保護條例）等規範大力推動。
- 然而在 AI 系統中，我們必須承認在需要更多個人資料以提升系統個人化與效能與隱私之間的矛盾。
- 正如連結電腦和網際網路誕生時產生大量安全問題，我們也看到與 AI 相關的安全問題急劇增加。
- 同時，我們也看到 AI 被用來提升安全。例如，現代多數防毒掃描器均由 AI 啟發式技術驅動。
- 我們需要確保資料科學流程與最新隱私及安全實務和諧融合。


### 透明度
AI 系統應易於理解。透明度的關鍵部分是解釋 AI 系統及其組件的行為。提升對 AI 系統的理解，需要持份者明白系統如何運作以及背後原因，以識別潛在的性能問題、安全與隱私疑慮、偏見、排他性慣例或非預期結果。我們也相信 AI 使用者應誠實並公開說明何時、為何及如何選擇部署 AI 系統，以及他們使用系統的限制。例如，銀行使用 AI 系統支援消費貸款決策時，檢視結果並了解哪些資料影響系統建議非常重要。政府正開始跨產業對 AI 進行監管，資料科學家和組織必須說明 AI 系統是否符合監管要求，尤其是在出現不理想結果時。

> [🎥 點擊此處觀看影片：AI 中的透明度](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- 由於 AI 系統非常複雜，很難理解其運作方式及結果解讀。
- 這種理解不足影響系統的管理、運營和紀錄。
- 更重要的是，這種缺乏理解影響使用這些系統產出結果時的決策。

### 責任制
 
設計和部署 AI 系統的人必須對系統的運作負責。責任制尤其在敏感技術（如臉部辨識）中尤為重要。近年來，臉部辨識技術需求日增，尤其是執法機構視其可用於尋找失蹤兒童等用途。然而，這項技術可能被政府用來危害公民的基本自由，例如對特定個體實施持續監控。因此，資料科學家與組織需要對 AI 系統對個人或社會的影響負責。

[![前沿 AI 研究者警告臉部辨識造成大規模監控](../../../../translated_images/zh-HK/accountability.41d8c0f4b85b6231.webp)](https://www.youtube.com/watch?v=Wldt8P5V6D0 "Microsoft's Approach to Responsible AI")

> 🎥 點擊上方圖片觀看影片：臉部辨識帶來大規模監控的警告

對我們這代人來說（首批將 AI 帶入社會的一代），最大問題之一就是如何確保電腦對人負責，以及設計電腦的人對所有人負責。

## 影響評估

在訓練機器學習模型之前，進行影響評估非常重要，以了解 AI 系統的目的、預期用途、部署地點及使用者。這有助於審查者或測試者在評估系統時，了解辨識潛在風險及預期後果時應考慮的因素。

進行影響評估時的重點領域包括：

* <strong>對個人的不利影響</strong>。了解任何限制或需求、不支持的用途或已知限制，對確保系統不以可能傷害個人的方式使用至關重要。
* <strong>資料需求</strong>。了解系統如何以及在哪裡使用資料，能讓審查者檢視需留意的資料需求（例如 GDPR 或 HIPAA 規定）。此外，也要審查資料來源及量是否足夠訓練。
* <strong>影響摘要</strong>。蒐集使用系統可能產生的潛在傷害清單。於機器學習生命週期中檢視所辨識問題是否被緩解或解決。
* <strong>適用目標</strong>，針對六項核心原則，評估是否達到各目標，並找出任何缺口。


## 負責任的 AI 除錯

如同除錯軟體應用程序，AI 系統除錯是識別及解決系統問題的必要過程。多種因素會影響模型表現不符合預期或負責任。大多數傳統模型性能指標是模型表現的量化彙總，無法充分分析模型如何違反負責任 AI 原則。此外，機器學習模型是黑盒，難以理解驅動結果的原因或當出錯時解釋原因。本課程稍後我們將學習如何使用負責任 AI 儀表板協助除錯 AI 系統。該儀表板為資料科學家和 AI 開發者提供全面工具，以進行：

* <strong>錯誤分析</strong>。識別模型錯誤分佈，可能影響系統公平性或可靠性。
* <strong>模型總覽</strong>。發現不同資料群體中模型表現差異。
* <strong>資料分析</strong>。了解資料分布並辨識可能導致公平性、包容性和可靠性問題的偏見。
* <strong>模型可解釋性</strong>。了解影響模型預測的因素，協助解釋模型行為，對透明度和責任制很重要。


## 🚀 挑戰
 
為避免傷害的產生，我們應：

- 團隊成員背景與觀點多元化
- 投資反映社會多樣性的資料集
- 在整個機器學習生命週期內，開發更佳方法檢測並糾正責任 AI 相關問題

思考現實情境中模型不可靠的具體表現及其帶來的影響。我們還應考慮什麼？

## [課後測驗](https://ff-quizzes.netlify.app/en/ml/)

## 回顧與自學
 

在本課程中，您已經學習了一些機器學習中公平與不公平概念的基礎知識。  
 
觀看此工作坊以深入了解這些主題： 

- 追求負責任的 AI：將原則付諸實踐，講者 Besmira Nushi、Mehrnoosh Sameki 和 Amit Sharma

[![Responsible AI Toolbox: An open-source framework for building responsible AI](https://img.youtube.com/vi/tGgJCrA-MZU/0.jpg)](https://www.youtube.com/watch?v=tGgJCrA-MZU "RAI Toolbox: An open-source framework for building responsible AI")

> 🎥 點擊上方圖片觀看影片：RAI Toolbox：一個用於構建負責任 AI 的開源框架，講者 Besmira Nushi、Mehrnoosh Sameki 與 Amit Sharma

另外，請閱讀： 

- 微軟的 RAI 資源中心：[Responsible AI Resources – Microsoft AI](https://www.microsoft.com/ai/responsible-ai-resources?activetab=pivot1%3aprimaryr4) 

- 微軟的 FATE 研究組：[FATE: Fairness, Accountability, Transparency, and Ethics in AI - Microsoft Research](https://www.microsoft.com/research/theme/fate/) 

RAI Toolbox： 

- [Responsible AI Toolbox GitHub repository](https://github.com/microsoft/responsible-ai-toolbox)

閱讀關於 Azure Machine Learning 的工具以確保公平性：

- [Azure Machine Learning](https://docs.microsoft.com/azure/machine-learning/concept-fairness-ml?WT.mc_id=academic-77952-leestott) 

## 作業

[探索 RAI Toolbox](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**免責聲明**：
本文件由 AI 翻譯服務 [Co-op Translator](https://github.com/Azure/co-op-translator) 翻譯而成。雖然我們致力於確保準確性，但請注意，機器自動翻譯可能包含錯誤或不準確之處。原始文件的母語版本應被視為權威來源。對於重要資訊，建議進行專業人工翻譯。我們不對因使用本翻譯而產生的任何誤解或誤釋承擔責任。
<!-- CO-OP TRANSLATOR DISCLAIMER END -->