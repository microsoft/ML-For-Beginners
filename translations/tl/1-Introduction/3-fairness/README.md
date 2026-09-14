# Pagtatayo ng mga Solusyon sa Machine Learning gamit ang responsable na AI
 
![Buod ng responsable na AI sa Machine Learning sa isang sketchnote](../../../../translated_images/tl/ml-fairness.ef296ebec6afc98a.webp)
> Sketchnote ni [Tomomi Imura](https://www.twitter.com/girlie_mac)

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)
 
## Panimula

Sa kurikulung ito, sisimulan mong tuklasin kung paano at paano naaapektuhan ng machine learning ang ating pang-araw-araw na buhay. Kahit ngayon, ang mga sistema at modelo ay kasangkot sa mga gawaing pang-araw-araw na pagdedesisyon, tulad ng mga diagnosis sa pangangalagang pangkalusugan, pag-apruba ng pautang o pagtuklas ng panlilinlang. Kaya mahalagang ang mga modelong ito ay gumana nang mabuti upang magbigay ng mga resulta na mapagkakatiwalaan. Tulad ng anumang aplikasyon ng software, ang mga AI system ay maaaring hindi maabot ang inaasahan o magkaroon ng hindi kanais-nais na kinalabasan. Kaya naman mahalagang maunawaan at maipaliwanag ang kilos ng isang AI model.

Isipin kung ano ang maaaring mangyari kapag ang data na iyong ginagamit upang buuin ang mga modelong ito ay kulang sa ilang demograpiko, tulad ng lahi, kasarian, pananaw sa politika, relihiyon, o hindi pantay na kinakatawan ang mga ganitong demograpiko. Paano naman kapag ang output ng modelo ay ininterpretang pabor sa ilang demograpiko? Ano ang magiging epekto nito sa aplikasyon? Bukod dito, ano ang mangyayari kapag ang modelo ay nagkaroon ng masamang kinalabasan at nakasakit sa mga tao? Sino ang mananagot sa kilos ng sistema ng AI? Ito ang ilan sa mga tanong na susuriin natin sa kurikulung ito.

Sa araling ito, matututuhan mo:

- Pataasin ang iyong kamalayan sa kahalagahan ng pagiging patas sa machine learning at mga pinsalang may kinalaman sa pagiging patas.
- Maging pamilyar sa pagsasanay ng pagtuklas ng mga outliers at hindi pangkaraniwang mga sitwasyon upang matiyak ang pagiging maaasahan at kaligtasan
- Maunawaan ang pangangailangan na bigyang-kapangyarihan ang lahat sa pamamagitan ng pagdidisenyo ng mga inklusibong sistema
- Tuklasin kung gaano kahalaga ang pagprotekta sa privacy at seguridad ng data at mga tao
- Makita ang kahalagahan ng pagkakaroon ng glass box na pamamaraan upang maipaliwanag ang kilos ng mga AI model
- Maging maingat sa kung paano mahalaga ang pananagutan upang makabuo ng tiwala sa mga sistema ng AI

## Kinakailangan

Bilang isang kinakailangan, pakiusap na kunin ang "Responsible AI Principles" Learn Path at panoorin ang video sa ibaba tungkol sa paksa:

Matuto pa tungkol sa Responsible AI sa pamamagitan ng pagsunod sa [Learning Path](https://docs.microsoft.com/learn/modules/responsible-ai-principles/?WT.mc_id=academic-77952-leestott)

[![Microsoft's Approach to Responsible AI](https://img.youtube.com/vi/dnC8-uUZXSc/0.jpg)](https://youtu.be/dnC8-uUZXSc "Microsoft's Approach to Responsible AI")

> 🎥 I-click ang larawan sa itaas para sa isang video: Microsoft's Approach to Responsible AI

## Pagiging patas

Dapat tratuhin ng mga sistema ng AI ang lahat nang patas at iwasang makaapekto sa magkatulad na mga grupo ng tao sa magkakaibang paraan. Halimbawa, kapag nagbibigay ang mga sistema ng AI ng patnubay sa medikal na paggamot, aplikasyon ng pautang, o trabaho, dapat nilang gawin ang parehong mga rekomendasyon sa lahat na may magkatulad na sintomas, kalagayan sa pananalapi, o propesyonal na kwalipikasyon. Ang bawat isa sa atin bilang mga tao ay nagdadala ng mga likas na pagkiling na nakakaapekto sa ating mga desisyon at aksyon. Ang mga pagkiling na ito ay maaaring makita sa data na ginagamit natin upang sanayin ang mga sistema ng AI. Minsan nangyayari ang ganitong manipulasyon nang hindi sinasadya. Madalas mahirap bigyang-alam sa sarili kung kailan ka nagpapasok ng pagkiling sa data.

**“Hindi pagiging patas”** ay sumasaklaw sa mga negatibong epekto, o “mga pinsala”, para sa isang grupo ng mga tao, tulad ng mga tinukoy batay sa lahi, kasarian, edad, o status ng kapansanan. Ang mga pangunahing pinsalang may kinalaman sa pagiging patas ay maaring iklasipika bilang:

- **Alokasyon**, kung tulad ng kasarian o etnisidad ay pabor sa iba.
- **Kalidad ng serbisyo**. Kung sanayin mo ang data para sa isang partikular na sitwasyon ngunit ang realidad ay mas kumplikado, magreresulta ito sa isang mahina ang pagganap na serbisyo. Halimbawa, isang hand soap dispenser na hindi tila maramdaman ang mga taong may maitim na balat. [Reference](https://gizmodo.com/why-cant-this-soap-dispenser-identify-dark-skin-1797931773)
- **Panlilibak**. Hindi patas na pag-kritika at pag-label sa isang bagay o tao. Halimbawa, isang teknolohiya sa pag-label ng imahe ay kilalang mali ang pag-label ng mga larawan ng mga taong may maitim na balat bilang mga gorilya.
- **Sobra o kulang na kinatawan**. Ang ideya ay ang isang tiyak na grupo ay hindi nakikita sa isang partikular na propesyon, at anumang serbisyo o tungkulin na patuloy na ipinapromote ito ay nakakadagdag sa pinsala.
- **Stereotyping**. Pag-uugnay sa isang grupo sa mga nakatalaga nang mga katangian. Halimbawa, isang sistema ng pagsasalin ng wika sa pagitan ng Ingles at Turkish ay maaaring may mga kamalian dahil sa mga salita na may stereotypical na mga ugnayan sa kasarian.

![translation to Turkish](../../../../translated_images/tl/gender-bias-translate-en-tr.f185fd8822c2d437.webp)
> pagsasalin sa Turkish

![translation back to English](../../../../translated_images/tl/gender-bias-translate-tr-en.4eee7e3cecb8c70e.webp)
> pagsasalin pabalik sa Ingles

Kapag nagdidisenyo at sumusubok ng mga sistema ng AI, kailangan nating matiyak na ang AI ay patas at hindi naka-program upang gumawa ng mga kinikilingang o diskriminatibong desisyon, na ipinagbabawal din sa mga tao. Ang pagtiyak ng pagiging patas sa AI at machine learning ay nananatiling isang komplikadong hamon sa lipunan at teknikal.

### Pagkakatiwalaan at kaligtasan

Upang makabuo ng tiwala, ang mga sistema ng AI ay kailangang maging maaasahan, ligtas, at consistent sa normal at hindi inaasahang mga kondisyon. Mahalagang malaman kung paano kikilos ang mga sistema ng AI sa iba't ibang sitwasyon, lalo na kapag sila ay mga outliers. Kapag bumubuo ng mga solusyon sa AI, kailangan ng malaking pokus kung paano haharapin ang malawak na hanay ng mga kalagayan na maaaring maranasan ng mga solusyon sa AI. Halimbawa, ang isang self-driving na sasakyan ay kailangang unahin ang kaligtasan ng mga tao. Kaya, ang AI na nagpapatakbo sa sasakyan ay kailangang isaalang-alang ang lahat ng posibleng senaryo na maaaring harapin ng sasakyan tulad ng gabi, mga bagyo o snowstorm, mga batang tumatakbo sa kalsada, mga alagang hayop, mga konstruksyon sa kalsada atbp. Kung gaano kahusay hawakan ng isang AI system ang malawak na hanay ng mga kondisyon nang maaasahan at ligtas ay nagpapakita ng antas ng pag-anticipate ng data scientist o developer ng AI nang idinisenyo o sinubukan ang sistema.

> [🎥 I-click dito para sa isang video: Pagkakatiwalaan at kaligtasan sa AI](https://www.microsoft.com/videoplayer/embed/RE4vvIl)

### Inklusibidad

Ang mga sistema ng AI ay dapat idisenyo upang makilahok at bigyang kapangyarihan ang lahat. Kapag nagdidisenyo at nagpapatupad ng mga sistema ng AI, ang mga data scientist at AI developer ay tumutukoy at tinutugunan ang posibleng hadlang sa sistema na maaaring hindi sinasadya na mag-alis ng mga tao. Halimbawa, mayroong 1 bilyong tao sa buong mundo na may kapansanan. Sa pag-unlad ng AI, mas madali na nilang ma-access ang malawak na hanay ng mga impormasyon at oportunidad sa kanilang pang-araw-araw na buhay. Sa pagtugon sa mga hadlang, lumilikha ito ng mga pagkakataon upang mag-inobasyon at bumuo ng mga produktong AI na may mas mahusay na karanasan na nakikinabang sa lahat.

> [🎥 I-click dito para sa isang video: Inklusibidad sa AI](https://www.microsoft.com/videoplayer/embed/RE4vl9v)

### Seguridad at privacy

Ang mga sistema ng AI ay dapat maging ligtas at igalang ang privacy ng mga tao. Mas kaunti ang tiwala ng mga tao sa mga sistema na naglalagay sa panganib ng kanilang privacy, impormasyon, o buhay. Kapag nagsasanay ng mga modelong machine learning, umaasa tayo sa data upang makagawa ng pinakamahusay na resulta. Sa paggawa nito, kailangang isaalang-alang ang pinagmulan ng data at integridad. Halimbawa, ang data ba ay isinumite ng user o pampublikong available? Susunod, habang nagtatrabaho sa data, mahalagang bumuo ng mga sistema ng AI na makakaprotekta ng kumpidensyal na impormasyon at makatiis sa mga pag-atake. Habang lumalaganap ang AI, ang pagprotekta sa privacy at pagseseguro ng mahalagang personal at negosyo ay nagiging mas kritikal at komplikado. Ang mga isyu sa privacy at seguridad ng data ay nangangailangan ng espesyal na pansin para sa AI dahil ang akses sa data ay mahalaga upang makagawa ang AI ng mga tumpak at napag-alamang mga prediksyon at desisyon tungkol sa mga tao.

> [🎥 I-click dito para sa isang video: Seguridad sa AI](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Bilang isang industriya, nakagawa tayo ng makabuluhang pag-unlad sa Privacy at seguridad, na lubos na pinatibay ng mga regulasyon tulad ng GDPR (General Data Protection Regulation).
- Ngunit sa mga sistema ng AI, kailangang kilalanin ang tensiyon sa pagitan ng pangangailangan para sa mas maraming personal na data upang gawing mas personal at epektibo ang mga sistema – at privacy.
- Tulad ng pagsilang ng mga konektadong kompyuter sa internet, nakikita rin natin ang malaking pagtaas sa bilang ng mga isyu sa seguridad na kaugnay ng AI.
- Kasabay nito, nakita natin ang paggamit ng AI upang mapabuti ang seguridad. Halimbawa, karamihan sa mga modernong anti-virus scanner ay pinalakas ng AI heuristics ngayon.
- Kailangang tiyakin na ang aming mga proseso sa Data Science ay nagkakaisa nang maayos sa mga pinakabagong praktis sa privacy at seguridad.


### Transparency
Dapat maintindihan ang mga sistema ng AI. Isang mahalagang bahagi ng transparency ay ang pagpapaliwanag sa kilos ng mga sistema ng AI at kanilang mga bahagi. Ang pagpapabuti ng pag-unawa sa mga sistema ng AI ay nangangailangan na maunawaan ng mga stakeholder kung paano at bakit sila gumagana upang matukoy ang mga posibleng problema sa pagganap, mga isyu sa kaligtasan at privacy, mga pagkiling, mga pagsasanay na naglalagay ng ibang tao sa layo, o hindi inaasahang mga resulta. Naniniwala rin kami na ang mga gumagamit ng mga sistema ng AI ay dapat maging tapat at bukas tungkol kung kailan, bakit, at paano nila pinipili ang pag-deploy ng mga ito. Gayundin ang mga limitasyon ng mga sistemang kanilang ginagamit. Halimbawa, kung ang isang bangko ay gumagamit ng AI sistema upang suportahan ang kanilang mga desisyon sa pautang sa mga consumer, mahalagang suriin ang mga resulta at intindihin kung aling data ang nakaimpluwensya sa mga rekomendasyon ng sistema. Nagsisimula nang mag-regulate ang mga gobyerno sa AI sa iba't ibang industriya, kaya ang mga data scientist at organisasyon ay dapat ipaliwanag kung ang isang AI sistema ay nakakatugon sa mga regulasyon, lalo na kapag may masamang kinalabasan.

> [🎥 I-click dito para sa isang video: Transparency sa AI](https://www.microsoft.com/videoplayer/embed/RE4voJF)

- Dahil napaka-komplikado ng mga sistema ng AI, mahirap maintindihan kung paano sila gumagana at i-interpret ang mga resulta.
- Ang kakulangan ng pag-unawang ito ay nakaapekto sa paraan kung paano pinamamahalaan, pinapatawag sa operasyon, at dinodokumento ang mga sistemang ito.
- Mas mahalaga, ang kakulangan ng pag-unawa na ito ay nakaapekto sa mga desisyon na ginagawa gamit ang mga resulta ng mga sistemang ito.

### Pananagutan
 
Ang mga taong nagdidisenyo at nagde-deploy ng mga sistema ng AI ay dapat managot sa kung paano gumagana ang kanilang mga sistema. Ang pangangailangan para sa pananagutan ay partikular na mahalaga sa mga sensitibong teknolohiya tulad ng facial recognition. Kamakailan lamang, lumalago ang demand para sa teknolohiyang facial recognition, lalo na mula sa mga ahensya ng pagpapatupad ng batas na nakikita ang potensyal ng teknolohiya sa mga gamit tulad ng paghahanap sa mga nawalang bata. Gayunpaman, ang mga teknolohiyang ito ay posibleng magamit ng isang gobyerno upang ilagay sa panganib ang mga pangunahing kalayaan ng kanilang mga mamamayan sa pamamagitan, halimbawa, ng pagpapatupad ng tuloy-tuloy na surveillance sa mga partikular na indibidwal. Kaya, kailangang maging responsable ang mga data scientist at organisasyon sa kung paano naapektuhan ng kanilang sistema ng AI ang mga indibidwal o lipunan.

[![Pangunahing Mananaliksik sa AI Nagbabala ng Mass Surveillance Sa Pamamagitan ng Facial Recognition](../../../../translated_images/tl/accountability.41d8c0f4b85b6231.webp)](https://www.youtube.com/watch?v=Wldt8P5V6D0 "Microsoft's Approach to Responsible AI")

> 🎥 I-click ang larawan sa itaas para sa isang video: Mga Babala tungkol sa Mass Surveillance Sa Pamamagitan ng Facial Recognition

Sa huli, isa sa pinakamalalaking tanong para sa aming henerasyon, bilang unang henerasyon na nagdadala ng AI sa lipunan, ay kung paano matitiyak na mananatiling may pananagutan sa mga tao ang mga kompyuter at kung paano matitiyak na ang mga taong nagdidisenyo ng mga kompyuter ay mananatiling may pananagutan sa lahat ng iba pa.

## Pagsusuri sa Epekto

Bago sanayin ang isang machine learning model, mahalagang magsagawa ng pagsusuri sa epekto upang maunawaan ang layunin ng AI system; kung ano ang inaasahang gamit nito; saan ito ide-deploy; at sino ang makikialam sa sistema. Nakakatulong ito sa reviewer(s) o mga sumusuri sa sistema na malaman kung anu-anong mga salik ang dapat isaalang-alang kapag tinutukoy ang mga posibleng panganib at inaasahang mga epekto.

Ang mga sumusunod ay mga lugar ng pokus kapag nagsasagawa ng pagsusuri sa epekto:

* **Masamang epekto sa mga indibidwal**. Mahalaga ang kamalayan sa anumang mga restriksiyon o pangangailangan, di-suportadong paggamit o anumang kilalang limitasyon na pumipigil sa pagganap ng sistema upang matiyak na hindi ito magagamit sa paraang makasasakit sa mga indibidwal.
* **Mga pangangailangan sa data**. Ang pag-unawa kung paano at saan gagamitin ng sistema ang data ay nagpapahintulot sa mga reviewer na tuklasin ang anumang mga kinakailangan sa data na dapat mong tandaan (hal., GDPR o HIPAA na mga regulasyon sa data). Bukod dito, suriin kung sapat ba ang pinagmulan o dami ng data para sa pagsasanay.
* **Buod ng epekto**. Kolektahin ang listahan ng mga posibleng pinsalang maaaring lumitaw mula sa paggamit ng sistema. Sa buong lifecycle ng ML, suriin kung naibsan o natugunan ang mga isyung natukoy.
* **Mga naaangkop na layunin** para sa bawat isa sa anim na pangunahing prinsipyo. Suriin kung natutugunan ang mga layunin mula sa bawat prinsipyo at kung may mga kakulangan.


## Pag-debug gamit ang responsable na AI

Tulad ng pag-debug ng isang software application, ang pag-debug ng isang AI system ay isang kinakailangang proseso ng pagtukoy at paglutas ng mga isyu sa sistema. Maraming salik ang maaaring makaapekto sa hindi inaasahang pagganap o hindi responsableng pagganap ng isang modelo. Karamihan sa mga tradisyunal na metric sa pagganap ng modelo ay mga quantitative na kabuuan ng pagganap ng modelo, na hindi sapat upang suriin kung paano nilalabag ng isang modelo ang mga prinsipyo ng responsable na AI. Bukod pa rito, ang isang machine learning model ay isang black box na nagpapahirap na maunawaan kung ano ang nagtutulak ng kanyang kinalabasan o magbigay paliwanag kapag nagkamali. Sa susunod na bahagi ng kurso, matututuhan natin kung paano gamitin ang Responsible AI dashboard upang makatulong sa pag-debug ng mga sistema ng AI. Nagbibigay ang dashboard ng isang holistikong kagamitan para sa mga data scientist at AI developer upang maisagawa ang:

* **Pagsusuri ng error**. Para matukoy ang distribusyon ng error ng modelo na maaaring makaapekto sa pagiging patas o pagiging maaasahan ng sistema.
* **Pangkalahatang-ideya ng modelo**. Upang matuklasan kung saan may mga disparity sa pagganap ng modelo sa mga data cohorts.
* **Pagsusuri ng data**. Upang maunawaan ang distribusyon ng data at matukoy ang anumang posibleng bias sa data na maaaring magdulot ng mga isyu sa pagiging patas, inklusibidad, at pagiging maaasahan.
* **Interpretabilidad ng modelo**. Upang maunawaan kung ano ang nakakaapekto o nakakaimpluwensya sa mga prediksyon ng modelo. Nakakatulong ito sa pagpapaliwanag ng kilos ng modelo, na mahalaga para sa transparency at pananagutan.


## 🚀 Hamon
 
Upang mapigilan ang pagpasok ng mga pinsala mula sa simula, dapat nating:

- magkaroon ng pagkakaiba-iba ng mga pinanggalingan at pananaw sa mga taong nagtatrabaho sa mga sistema
- mamuhunan sa mga dataset na sumasalamin sa pagkakaiba-iba ng ating lipunan
- bumuo ng mas mahusay na mga pamamaraan sa buong lifecycle ng machine learning para matukoy at maitama ang hindi responsableng AI kapag nangyari ito

Isipin ang mga totoong buhay na scenario kung saan maliwanag ang hindi pagiging mapagkakatiwalaan ng isang modelo sa paggawa at paggamit ng modelo. Ano pa ang dapat nating isaalang-alang?

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Balik-aral at Sariling Pag-aaral
 
Sa araling ito, natutunan mo ang ilang mga pundasyon ng mga konsepto ng pagiging patas at hindi pagiging patas sa machine learning.
 
Panoorin ang workshop na ito upang mas malaliman pa ang mga paksa:

- Sa paghahangad ng responsable na AI: Pagsasabuhay ng mga prinsipyo ni Besmira Nushi, Mehrnoosh Sameki at Amit Sharma

[![Responsible AI Toolbox: Isang open-source na balangkas para sa paggawa ng responsable na AI](https://img.youtube.com/vi/tGgJCrA-MZU/0.jpg)](https://www.youtube.com/watch?v=tGgJCrA-MZU "RAI Toolbox: Isang open-source na balangkas para sa paggawa ng responsable na AI")

> 🎥 I-click ang imahe sa itaas para sa isang video: RAI Toolbox: Isang open-source na balangkas para sa paggawa ng responsable na AI ni Besmira Nushi, Mehrnoosh Sameki, at Amit Sharma

Basahin din: 

- Sentro ng mga RAI resource ng Microsoft: [Responsible AI Resources – Microsoft AI](https://www.microsoft.com/ai/responsible-ai-resources?activetab=pivot1%3aprimaryr4) 

- Pangkat ng pananaliksik ng FATE ng Microsoft: [FATE: Fairness, Accountability, Transparency, and Ethics in AI - Microsoft Research](https://www.microsoft.com/research/theme/fate/) 

RAI Toolbox: 

- [Responsible AI Toolbox GitHub repository](https://github.com/microsoft/responsible-ai-toolbox)

Basahin ang tungkol sa mga kasangkapan ng Azure Machine Learning upang matiyak ang katarungan:

- [Azure Machine Learning](https://docs.microsoft.com/azure/machine-learning/concept-fairness-ml?WT.mc_id=academic-77952-leestott) 

## Takdang Aralin

[Galugarin ang RAI Toolbox](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Pagtatanggi**:
Ang dokumentong ito ay isinalin gamit ang serbisyo ng AI translation na [Co-op Translator](https://github.com/Azure/co-op-translator). Bagama't nagsusumikap kami para sa katumpakan, pakatandaan na ang awtomatikong pagsasalin ay maaaring maglaman ng mga pagkakamali o hindi pagkakatugma. Ang orihinal na dokumento sa orihinal nitong wika ang dapat ituring na pangunahing sanggunian. Para sa mahahalagang impormasyon, inirerekomenda ang propesyonal na pagsasalin ng tao. Hindi kami mananagot sa anumang maling pagkakaintindi o maling interpretasyon na nagmula sa paggamit ng pagsasaling ito.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->