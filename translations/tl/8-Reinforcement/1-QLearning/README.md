<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "911efd5e595089000cb3c16fce1beab8",
  "translation_date": "2025-09-05T18:23:22+00:00",
  "source_file": "8-Reinforcement/1-QLearning/README.md",
  "language_code": "tl"
}
-->
# Panimula sa Reinforcement Learning at Q-Learning

![Buod ng reinforcement sa machine learning sa isang sketchnote](../../../../sketchnotes/ml-reinforcement.png)
> Sketchnote ni [Tomomi Imura](https://www.twitter.com/girlie_mac)

Ang reinforcement learning ay may tatlong mahalagang konsepto: ang agent, ilang estado, at isang hanay ng mga aksyon sa bawat estado. Sa pamamagitan ng pagsasagawa ng isang aksyon sa isang tiyak na estado, binibigyan ang agent ng gantimpala. Isipin muli ang laro sa computer na Super Mario. Ikaw si Mario, nasa isang antas ng laro, nakatayo sa gilid ng bangin. Sa itaas mo ay may barya. Ikaw bilang si Mario, nasa isang antas ng laro, sa isang tiyak na posisyon ... iyon ang iyong estado. Ang paggalaw ng isang hakbang pakanan (isang aksyon) ay magdadala sa iyo sa gilid, at magbibigay iyon sa iyo ng mababang numerong puntos. Gayunpaman, ang pagpindot sa pindutan ng pagtalon ay magbibigay-daan sa iyo na makakuha ng puntos at manatiling buhay. Iyon ay isang positibong resulta at dapat kang gantimpalaan ng positibong numerong puntos.

Sa pamamagitan ng paggamit ng reinforcement learning at isang simulator (ang laro), maaari mong matutunan kung paano laruin ang laro upang mapakinabangan ang gantimpala, na ang pananatiling buhay at pagkamit ng pinakamaraming puntos hangga't maaari.

[![Panimula sa Reinforcement Learning](https://img.youtube.com/vi/lDq_en8RNOo/0.jpg)](https://www.youtube.com/watch?v=lDq_en8RNOo)

> 🎥 I-click ang imahe sa itaas upang marinig si Dmitry na talakayin ang Reinforcement Learning

## [Pre-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Mga Paunang Kailangan at Setup

Sa araling ito, mag-eeksperimento tayo gamit ang ilang code sa Python. Dapat mong magawang patakbuhin ang Jupyter Notebook code mula sa araling ito, alinman sa iyong computer o sa cloud.

Maaari mong buksan ang [notebook ng aralin](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/notebook.ipynb) at sundan ang araling ito upang bumuo.

> **Note:** Kung binubuksan mo ang code na ito mula sa cloud, kailangan mo ring kunin ang file na [`rlboard.py`](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/rlboard.py), na ginagamit sa notebook code. Idagdag ito sa parehong direktoryo ng notebook.

## Panimula

Sa araling ito, susuriin natin ang mundo ng **[Peter and the Wolf](https://en.wikipedia.org/wiki/Peter_and_the_Wolf)**, na inspirasyon ng isang musikal na kwento ng isang Russian composer, [Sergei Prokofiev](https://en.wikipedia.org/wiki/Sergei_Prokofiev). Gagamit tayo ng **Reinforcement Learning** upang tulungan si Peter na tuklasin ang kanyang kapaligiran, mangolekta ng masasarap na mansanas, at iwasan ang makaharap ang lobo.

Ang **Reinforcement Learning** (RL) ay isang teknik sa pag-aaral na nagbibigay-daan sa atin na matutunan ang optimal na pag-uugali ng isang **agent** sa isang **environment** sa pamamagitan ng pagsasagawa ng maraming eksperimento. Ang agent sa environment na ito ay dapat mayroong **layunin**, na tinutukoy ng isang **reward function**.

## Ang Kapaligiran

Para sa pagiging simple, isipin natin ang mundo ni Peter bilang isang square board na may sukat na `width` x `height`, tulad nito:

![Kapaligiran ni Peter](../../../../8-Reinforcement/1-QLearning/images/environment.png)

Ang bawat cell sa board na ito ay maaaring:

* **lupa**, kung saan maaaring maglakad si Peter at iba pang nilalang.
* **tubig**, kung saan malinaw na hindi ka maaaring maglakad.
* isang **puno** o **damo**, isang lugar kung saan maaari kang magpahinga.
* isang **mansanas**, na kumakatawan sa isang bagay na ikatutuwa ni Peter na makita upang pakainin ang sarili.
* isang **lobo**, na mapanganib at dapat iwasan.

May hiwalay na Python module, [`rlboard.py`](https://github.com/microsoft/ML-For-Beginners/blob/main/8-Reinforcement/1-QLearning/rlboard.py), na naglalaman ng code upang magtrabaho sa kapaligirang ito. Dahil ang code na ito ay hindi mahalaga para sa pag-unawa sa ating mga konsepto, i-import natin ang module at gamitin ito upang lumikha ng sample board (code block 1):

```python
from rlboard import *

width, height = 8,8
m = Board(width,height)
m.randomize(seed=13)
m.plot()
```

Ang code na ito ay dapat mag-print ng larawan ng kapaligiran na katulad ng nasa itaas.

## Mga Aksyon at Patakaran

Sa ating halimbawa, ang layunin ni Peter ay makahanap ng mansanas, habang iniiwasan ang lobo at iba pang hadlang. Upang magawa ito, maaari siyang maglakad-lakad hanggang sa makita niya ang mansanas.

Samakatuwid, sa anumang posisyon, maaari siyang pumili sa isa sa mga sumusunod na aksyon: pataas, pababa, pakaliwa, at pakanan.

Itatakda natin ang mga aksyon bilang isang diksyunaryo, at i-map ang mga ito sa mga pares ng kaukulang pagbabago sa coordinate. Halimbawa, ang paggalaw pakanan (`R`) ay tumutugma sa pares `(1,0)`. (code block 2):

```python
actions = { "U" : (0,-1), "D" : (0,1), "L" : (-1,0), "R" : (1,0) }
action_idx = { a : i for i,a in enumerate(actions.keys()) }
```

Sa kabuuan, ang estratehiya at layunin ng senaryong ito ay ang mga sumusunod:

- **Ang estratehiya**, ng ating agent (Peter) ay tinutukoy ng tinatawag na **policy**. Ang policy ay isang function na nagbabalik ng aksyon sa anumang ibinigay na estado. Sa ating kaso, ang estado ng problema ay kinakatawan ng board, kabilang ang kasalukuyang posisyon ng player.

- **Ang layunin**, ng reinforcement learning ay sa huli matutunan ang isang mahusay na policy na magpapahintulot sa atin na malutas ang problema nang mahusay. Gayunpaman, bilang baseline, isaalang-alang natin ang pinakasimpleng policy na tinatawag na **random walk**.

## Random walk

Unahin nating lutasin ang ating problema sa pamamagitan ng pagpapatupad ng random walk strategy. Sa random walk, pipiliin natin nang random ang susunod na aksyon mula sa mga pinapayagang aksyon, hanggang sa maabot natin ang mansanas (code block 3).

1. Ipatupad ang random walk gamit ang code sa ibaba:

    ```python
    def random_policy(m):
        return random.choice(list(actions))
    
    def walk(m,policy,start_position=None):
        n = 0 # number of steps
        # set initial position
        if start_position:
            m.human = start_position 
        else:
            m.random_start()
        while True:
            if m.at() == Board.Cell.apple:
                return n # success!
            if m.at() in [Board.Cell.wolf, Board.Cell.water]:
                return -1 # eaten by wolf or drowned
            while True:
                a = actions[policy(m)]
                new_pos = m.move_pos(m.human,a)
                if m.is_valid(new_pos) and m.at(new_pos)!=Board.Cell.water:
                    m.move(a) # do the actual move
                    break
            n+=1
    
    walk(m,random_policy)
    ```

    Ang tawag sa `walk` ay dapat magbalik ng haba ng kaukulang landas, na maaaring mag-iba mula sa isang run patungo sa isa pa.

1. Patakbuhin ang walk experiment nang ilang beses (sabihin, 100), at i-print ang mga resulta ng istatistika (code block 4):

    ```python
    def print_statistics(policy):
        s,w,n = 0,0,0
        for _ in range(100):
            z = walk(m,policy)
            if z<0:
                w+=1
            else:
                s += z
                n += 1
        print(f"Average path length = {s/n}, eaten by wolf: {w} times")
    
    print_statistics(random_policy)
    ```

    Tandaan na ang average na haba ng landas ay nasa paligid ng 30-40 hakbang, na medyo marami, kung isasaalang-alang ang katotohanan na ang average na distansya sa pinakamalapit na mansanas ay nasa paligid ng 5-6 hakbang.

    Maaari mo ring makita kung ano ang hitsura ng galaw ni Peter sa random walk:

    ![Random Walk ni Peter](../../../../8-Reinforcement/1-QLearning/images/random_walk.gif)

## Reward function

Upang gawing mas matalino ang ating policy, kailangan nating maunawaan kung aling mga galaw ang "mas mabuti" kaysa sa iba. Upang magawa ito, kailangan nating tukuyin ang ating layunin.

Ang layunin ay maaaring tukuyin sa mga tuntunin ng isang **reward function**, na magbabalik ng ilang halaga ng puntos para sa bawat estado. Ang mas mataas na numero, mas maganda ang reward function. (code block 5)

```python
move_reward = -0.1
goal_reward = 10
end_reward = -10

def reward(m,pos=None):
    pos = pos or m.human
    if not m.is_valid(pos):
        return end_reward
    x = m.at(pos)
    if x==Board.Cell.water or x == Board.Cell.wolf:
        return end_reward
    if x==Board.Cell.apple:
        return goal_reward
    return move_reward
```

Isang kawili-wiling bagay tungkol sa reward functions ay sa karamihan ng mga kaso, *binibigyan lamang tayo ng makabuluhang gantimpala sa dulo ng laro*. Nangangahulugan ito na ang ating algorithm ay dapat na alalahanin ang "magagandang" hakbang na humantong sa positibong gantimpala sa dulo, at dagdagan ang kanilang kahalagahan. Katulad nito, ang lahat ng galaw na humantong sa masamang resulta ay dapat na iwasan.

## Q-Learning

Ang algorithm na tatalakayin natin dito ay tinatawag na **Q-Learning**. Sa algorithm na ito, ang policy ay tinutukoy ng isang function (o isang data structure) na tinatawag na **Q-Table**. Itinatala nito ang "kabutihan" ng bawat aksyon sa isang ibinigay na estado.

Tinatawag itong Q-Table dahil madalas na maginhawa itong i-representa bilang isang table, o multi-dimensional array. Dahil ang ating board ay may sukat na `width` x `height`, maaari nating i-representa ang Q-Table gamit ang isang numpy array na may hugis na `width` x `height` x `len(actions)`: (code block 6)

```python
Q = np.ones((width,height,len(actions)),dtype=np.float)*1.0/len(actions)
```

Pansinin na ini-initialize natin ang lahat ng mga halaga ng Q-Table sa pantay na halaga, sa ating kaso - 0.25. Ito ay tumutugma sa "random walk" policy, dahil ang lahat ng galaw sa bawat estado ay pantay na mabuti. Maaari nating ipasa ang Q-Table sa `plot` function upang i-visualize ang table sa board: `m.plot(Q)`.

![Kapaligiran ni Peter](../../../../8-Reinforcement/1-QLearning/images/env_init.png)

Sa gitna ng bawat cell ay may "arrow" na nagpapahiwatig ng ginustong direksyon ng paggalaw. Dahil pantay ang lahat ng direksyon, isang tuldok ang ipinapakita.

Ngayon kailangan nating patakbuhin ang simulation, tuklasin ang ating kapaligiran, at matutunan ang mas mahusay na distribusyon ng mga halaga ng Q-Table, na magpapahintulot sa atin na mahanap ang landas patungo sa mansanas nang mas mabilis.

## Ang Esensya ng Q-Learning: Bellman Equation

Kapag nagsimula na tayong gumalaw, ang bawat aksyon ay magkakaroon ng kaukulang gantimpala, ibig sabihin maaari nating teoretikal na piliin ang susunod na aksyon batay sa pinakamataas na agarang gantimpala. Gayunpaman, sa karamihan ng mga estado, ang galaw ay hindi makakamit ang ating layunin na maabot ang mansanas, at sa gayon hindi natin agad matutukoy kung aling direksyon ang mas mabuti.

> Tandaan na hindi ang agarang resulta ang mahalaga, kundi ang panghuling resulta, na makukuha natin sa dulo ng simulation.

Upang isaalang-alang ang naantalang gantimpala, kailangan nating gamitin ang mga prinsipyo ng **[dynamic programming](https://en.wikipedia.org/wiki/Dynamic_programming)**, na nagbibigay-daan sa atin na isipin ang ating problema nang recursive.

Ipagpalagay na tayo ay nasa estado *s* ngayon, at nais nating lumipat sa susunod na estado *s'*. Sa paggawa nito, makakatanggap tayo ng agarang gantimpala *r(s,a)*, na tinutukoy ng reward function, kasama ang ilang hinaharap na gantimpala. Kung ipagpalagay natin na ang ating Q-Table ay tama na sumasalamin sa "kaakit-akit" ng bawat aksyon, kung gayon sa estado *s'* pipili tayo ng aksyon *a* na tumutugma sa maximum na halaga ng *Q(s',a')*. Kaya, ang pinakamahusay na posibleng hinaharap na gantimpala na maaari nating makuha sa estado *s* ay tinutukoy bilang `max`

## Pagsusuri ng polisiya

Dahil ang Q-Table ay naglilista ng "kaakit-akit" ng bawat aksyon sa bawat estado, madali itong gamitin upang tukuyin ang epektibong pag-navigate sa ating mundo. Sa pinakasimpleng kaso, maaari nating piliin ang aksyon na may pinakamataas na Q-Table value: (code block 9)

```python
def qpolicy_strict(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = list(actions)[np.argmax(v)]
        return a

walk(m,qpolicy_strict)
```

> Kapag sinubukan mo ang code sa itaas nang ilang beses, mapapansin mo na minsan ito ay "naghahang", at kailangan mong pindutin ang STOP button sa notebook upang ihinto ito. Nangyayari ito dahil maaaring may mga sitwasyon kung saan ang dalawang estado ay "nagtuturo" sa isa't isa batay sa optimal na Q-Value, na nagreresulta sa agent na paulit-ulit na gumagalaw sa pagitan ng mga estadong iyon.

## 🚀Hamunin

> **Gawain 1:** Baguhin ang `walk` function upang limitahan ang maximum na haba ng landas sa isang tiyak na bilang ng mga hakbang (halimbawa, 100), at panoorin ang code sa itaas na bumalik sa halagang ito paminsan-minsan.

> **Gawain 2:** Baguhin ang `walk` function upang hindi ito bumalik sa mga lugar na napuntahan na nito dati. Maiiwasan nito ang `walk` na mag-loop, ngunit maaari pa ring ma-"trap" ang agent sa isang lokasyon na hindi nito matakasan.

## Pag-navigate

Mas magandang polisiya sa pag-navigate ang ginamit natin noong training, na pinagsasama ang exploitation at exploration. Sa polisiya na ito, pipiliin natin ang bawat aksyon na may tiyak na probabilidad, na proporsyonal sa mga halaga sa Q-Table. Ang estratehiyang ito ay maaaring magresulta pa rin sa agent na bumalik sa posisyon na napuntahan na nito, ngunit, tulad ng makikita mo sa code sa ibaba, nagreresulta ito sa napakaikling average na landas patungo sa nais na lokasyon (tandaan na ang `print_statistics` ay nagpapatakbo ng simulation nang 100 beses): (code block 10)

```python
def qpolicy(m):
        x,y = m.human
        v = probs(Q[x,y])
        a = random.choices(list(actions),weights=v)[0]
        return a

print_statistics(qpolicy)
```

Pagkatapos patakbuhin ang code na ito, dapat kang makakuha ng mas maliit na average na haba ng landas kaysa dati, sa saklaw na 3-6.

## Pagsisiyasat sa proseso ng pag-aaral

Tulad ng nabanggit, ang proseso ng pag-aaral ay balanse sa pagitan ng exploration at paggamit ng nakuha nang kaalaman tungkol sa istruktura ng problem space. Nakita natin na ang resulta ng pag-aaral (ang kakayahang tulungan ang agent na makahanap ng maikling landas patungo sa layunin) ay bumuti, ngunit interesante ring obserbahan kung paano nagbabago ang average na haba ng landas sa panahon ng proseso ng pag-aaral:

## Mga natutunan

- **Tumataas ang average na haba ng landas**. Sa simula, ang average na haba ng landas ay tumataas. Malamang ito ay dahil kapag wala tayong alam tungkol sa kapaligiran, mas malamang na ma-trap tayo sa masamang estado, tulad ng tubig o lobo. Habang natututo tayo at nagsisimulang gamitin ang kaalamang ito, mas matagal nating ma-explore ang kapaligiran, ngunit hindi pa rin natin alam nang maayos kung nasaan ang mga mansanas.

- **Bumaba ang haba ng landas habang natututo**. Kapag sapat na ang natutunan, mas madali para sa agent na maabot ang layunin, at nagsisimulang bumaba ang haba ng landas. Gayunpaman, bukas pa rin tayo sa exploration, kaya madalas tayong lumihis mula sa pinakamahusay na landas at mag-explore ng mga bagong opsyon, na nagpapahaba sa landas kaysa sa optimal.

- **Biglang tumataas ang haba**. Mapapansin din natin sa graph na sa isang punto, biglang tumataas ang haba. Ipinapakita nito ang stochastic na kalikasan ng proseso, at na maaari nating "masira" ang mga coefficient ng Q-Table sa pamamagitan ng pag-overwrite ng mga ito ng mga bagong halaga. Dapat itong mabawasan sa pamamagitan ng pagpapababa ng learning rate (halimbawa, sa pagtatapos ng training, ina-adjust lang natin ang mga halaga ng Q-Table nang kaunti).

Sa kabuuan, mahalagang tandaan na ang tagumpay at kalidad ng proseso ng pag-aaral ay malaki ang nakadepende sa mga parameter, tulad ng learning rate, learning rate decay, at discount factor. Ang mga ito ay madalas tawaging **hyperparameters**, upang maiba sa **parameters**, na ina-optimize natin sa panahon ng training (halimbawa, mga coefficient ng Q-Table). Ang proseso ng paghahanap ng pinakamahusay na halaga ng hyperparameters ay tinatawag na **hyperparameter optimization**, at ito ay karapat-dapat sa hiwalay na paksa.

## [Post-lecture quiz](https://ff-quizzes.netlify.app/en/ml/)

## Takdang-Aralin 
[A Mas Realistikong Mundo](assignment.md)

---

**Paunawa**:  
Ang dokumentong ito ay isinalin gamit ang AI translation service na [Co-op Translator](https://github.com/Azure/co-op-translator). Bagama't sinisikap naming maging tumpak, tandaan na ang mga awtomatikong pagsasalin ay maaaring maglaman ng mga pagkakamali o hindi pagkakatugma. Ang orihinal na dokumento sa kanyang katutubong wika ang dapat ituring na opisyal na sanggunian. Para sa mahalagang impormasyon, inirerekomenda ang propesyonal na pagsasalin ng tao. Hindi kami mananagot sa anumang hindi pagkakaunawaan o maling interpretasyon na maaaring magmula sa paggamit ng pagsasaling ito.