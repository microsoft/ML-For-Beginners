<!--
CO_OP_TRANSLATOR_METADATA:
{
  "original_hash": "9660fbd80845c59c15715cb418cd6e23",
  "translation_date": "2025-08-29T14:17:06+00:00",
  "source_file": "8-Reinforcement/2-Gym/README.md",
  "language_code": "tl"
}
-->
## Mga Paunang Kaalaman

Sa araling ito, gagamit tayo ng library na tinatawag na **OpenAI Gym** upang mag-simulate ng iba't ibang **kapaligiran**. Maaari mong patakbuhin ang code ng araling ito sa lokal na makina (halimbawa, gamit ang Visual Studio Code), kung saan magbubukas ang simulation sa isang bagong window. Kapag pinapatakbo ang code online, maaaring kailanganin mong baguhin ang ilang bahagi ng code, tulad ng ipinaliwanag [dito](https://towardsdatascience.com/rendering-openai-gym-envs-on-binder-and-google-colab-536f99391cc7).

## OpenAI Gym

Sa nakaraang aralin, ang mga patakaran ng laro at ang estado ay ibinigay ng `Board` class na tayo mismo ang nagtakda. Dito, gagamit tayo ng espesyal na **simulation environment**, na magsi-simulate ng pisika sa likod ng balancing pole. Isa sa mga pinakasikat na simulation environment para sa pagsasanay ng reinforcement learning algorithms ay tinatawag na [Gym](https://gym.openai.com/), na pinapanatili ng [OpenAI](https://openai.com/). Sa paggamit ng gym na ito, maaari tayong lumikha ng iba't ibang **kapaligiran** mula sa cartpole simulation hanggang sa mga laro ng Atari.

> **Note**: Makikita mo ang iba pang mga kapaligiran na available mula sa OpenAI Gym [dito](https://gym.openai.com/envs/#classic_control). 

Una, mag-install ng gym at i-import ang mga kinakailangang library (code block 1):

```python
import sys
!{sys.executable} -m pip install gym 

import gym
import matplotlib.pyplot as plt
import numpy as np
import random
```

## Ehersisyo - i-initialize ang isang cartpole environment

Upang magtrabaho sa problema ng cartpole balancing, kailangan nating i-initialize ang kaukulang kapaligiran. Ang bawat kapaligiran ay may kaugnayan sa:

- **Observation space** na tumutukoy sa istruktura ng impormasyon na natatanggap natin mula sa kapaligiran. Para sa problema ng cartpole, natatanggap natin ang posisyon ng pole, bilis, at iba pang mga halaga.

- **Action space** na tumutukoy sa mga posibleng aksyon. Sa ating kaso, ang action space ay discrete, at binubuo ng dalawang aksyon - **kaliwa** at **kanan**. (code block 2)

1. Upang mag-initialize, i-type ang sumusunod na code:

    ```python
    env = gym.make("CartPole-v1")
    print(env.action_space)
    print(env.observation_space)
    print(env.action_space.sample())
    ```

Upang makita kung paano gumagana ang kapaligiran, magpatakbo ng maikling simulation para sa 100 hakbang. Sa bawat hakbang, nagbibigay tayo ng isa sa mga aksyon na gagawin - sa simulation na ito, random nating pinipili ang isang aksyon mula sa `action_space`. 

1. Patakbuhin ang code sa ibaba at tingnan ang resulta.

    ✅ Tandaan na mas mainam na patakbuhin ang code na ito sa lokal na Python installation! (code block 3)

    ```python
    env.reset()
    
    for i in range(100):
       env.render()
       env.step(env.action_space.sample())
    env.close()
    ```

    Dapat kang makakita ng isang bagay na katulad ng larawang ito:

    ![non-balancing cartpole](../../../../8-Reinforcement/2-Gym/images/cartpole-nobalance.gif)

1. Sa panahon ng simulation, kailangan nating makakuha ng mga obserbasyon upang magdesisyon kung paano kumilos. Sa katunayan, ang step function ay nagbabalik ng kasalukuyang obserbasyon, isang reward function, at ang done flag na nagpapahiwatig kung may saysay pa bang ipagpatuloy ang simulation o hindi: (code block 4)

    ```python
    env.reset()
    
    done = False
    while not done:
       env.render()
       obs, rew, done, info = env.step(env.action_space.sample())
       print(f"{obs} -> {rew}")
    env.close()
    ```

    Makakakita ka ng isang bagay na ganito sa output ng notebook:

    ```text
    [ 0.03403272 -0.24301182  0.02669811  0.2895829 ] -> 1.0
    [ 0.02917248 -0.04828055  0.03248977  0.00543839] -> 1.0
    [ 0.02820687  0.14636075  0.03259854 -0.27681916] -> 1.0
    [ 0.03113408  0.34100283  0.02706215 -0.55904489] -> 1.0
    [ 0.03795414  0.53573468  0.01588125 -0.84308041] -> 1.0
    ...
    [ 0.17299878  0.15868546 -0.20754175 -0.55975453] -> 1.0
    [ 0.17617249  0.35602306 -0.21873684 -0.90998894] -> 1.0
    ```

    Ang observation vector na ibinabalik sa bawat hakbang ng simulation ay naglalaman ng mga sumusunod na halaga:
    - Posisyon ng cart
    - Bilis ng cart
    - Anggulo ng pole
    - Bilis ng pag-ikot ng pole

1. Kunin ang minimum at maximum na halaga ng mga numerong ito: (code block 5)

    ```python
    print(env.observation_space.low)
    print(env.observation_space.high)
    ```

    Mapapansin mo rin na ang reward value sa bawat simulation step ay palaging 1. Ito ay dahil ang layunin natin ay magtagal hangga't maaari, ibig sabihin, panatilihin ang pole sa isang makatwirang patayong posisyon sa pinakamahabang panahon.

    ✅ Sa katunayan, ang CartPole simulation ay itinuturing na nalutas kung makakakuha tayo ng average reward na 195 sa loob ng 100 sunud-sunod na pagsubok.

## Pag-discretize ng Estado

Sa Q-Learning, kailangan nating bumuo ng Q-Table na tumutukoy kung ano ang gagawin sa bawat estado. Upang magawa ito, kailangan ang estado ay **discreet**, mas partikular, dapat itong maglaman ng limitadong bilang ng mga discrete na halaga. Kaya, kailangan nating **i-discretize** ang ating mga obserbasyon, i-map ang mga ito sa isang limitadong set ng mga estado.

May ilang paraan upang magawa ito:

- **Hatiin sa mga bins**. Kung alam natin ang interval ng isang tiyak na halaga, maaari nating hatiin ang interval na ito sa ilang **bins**, at pagkatapos ay palitan ang halaga ng numero ng bin kung saan ito kabilang. Magagawa ito gamit ang numpy [`digitize`](https://numpy.org/doc/stable/reference/generated/numpy.digitize.html) method. Sa ganitong paraan, tiyak nating malalaman ang laki ng estado, dahil ito ay depende sa bilang ng bins na pinili natin para sa digitalization.
  
✅ Maaari nating gamitin ang linear interpolation upang dalhin ang mga halaga sa isang limitadong interval (halimbawa, mula -20 hanggang 20), at pagkatapos ay i-convert ang mga numero sa integers sa pamamagitan ng pag-round. Nagbibigay ito sa atin ng kaunting kontrol sa laki ng estado, lalo na kung hindi natin alam ang eksaktong saklaw ng mga input na halaga. Halimbawa, sa ating kaso, 2 sa 4 na halaga ay walang upper/lower bounds sa kanilang mga halaga, na maaaring magresulta sa walang katapusang bilang ng mga estado.

Sa ating halimbawa, gagamitin natin ang pangalawang paraan. Tulad ng mapapansin mo mamaya, sa kabila ng hindi tinukoy na upper/lower bounds, ang mga halagang iyon ay bihirang lumampas sa ilang limitadong interval, kaya ang mga estado na may extreme values ay magiging napakabihira.

1. Narito ang function na kukuha ng obserbasyon mula sa ating modelo at gagawa ng tuple ng 4 na integer values: (code block 6)

    ```python
    def discretize(x):
        return tuple((x/np.array([0.25, 0.25, 0.01, 0.1])).astype(np.int))
    ```

1. Tuklasin din natin ang isa pang paraan ng discretization gamit ang bins: (code block 7)

    ```python
    def create_bins(i,num):
        return np.arange(num+1)*(i[1]-i[0])/num+i[0]
    
    print("Sample bins for interval (-5,5) with 10 bins\n",create_bins((-5,5),10))
    
    ints = [(-5,5),(-2,2),(-0.5,0.5),(-2,2)] # intervals of values for each parameter
    nbins = [20,20,10,10] # number of bins for each parameter
    bins = [create_bins(ints[i],nbins[i]) for i in range(4)]
    
    def discretize_bins(x):
        return tuple(np.digitize(x[i],bins[i]) for i in range(4))
    ```

1. Ngayon, magpatakbo ng maikling simulation at obserbahan ang mga discrete environment values. Subukan ang parehong `discretize` at `discretize_bins` at tingnan kung may pagkakaiba.

    ✅ Ang `discretize_bins` ay nagbabalik ng bin number, na 0-based. Kaya para sa mga halaga ng input variable na malapit sa 0, nagbabalik ito ng numero mula sa gitna ng interval (10). Sa `discretize`, hindi natin inintindi ang saklaw ng output values, pinapayagan ang mga ito na maging negatibo, kaya ang mga estado ay hindi na-shift, at ang 0 ay tumutugma sa 0. (code block 8)

    ```python
    env.reset()
    
    done = False
    while not done:
       #env.render()
       obs, rew, done, info = env.step(env.action_space.sample())
       #print(discretize_bins(obs))
       print(discretize(obs))
    env.close()
    ```

    ✅ I-uncomment ang linya na nagsisimula sa env.render kung nais mong makita kung paano isinasagawa ang kapaligiran. Kung hindi, maaari mo itong patakbuhin sa background, na mas mabilis. Gagamitin natin ang "invisible" execution na ito sa panahon ng proseso ng Q-Learning.

## Ang Istruktura ng Q-Table

Sa nakaraang aralin, ang estado ay isang simpleng pares ng mga numero mula 0 hanggang 8, kaya't maginhawa itong i-representa ang Q-Table gamit ang isang numpy tensor na may hugis na 8x8x2. Kung gagamit tayo ng bins discretization, ang laki ng ating state vector ay kilala rin, kaya maaari nating gamitin ang parehong paraan at i-representa ang estado gamit ang array na may hugis na 20x20x10x10x2 (dito ang 2 ay ang dimensyon ng action space, at ang mga unang dimensyon ay tumutugma sa bilang ng bins na pinili natin para sa bawat parameter sa observation space).

Gayunpaman, minsan ang eksaktong dimensyon ng observation space ay hindi alam. Sa kaso ng `discretize` function, hindi tayo sigurado na ang ating estado ay mananatili sa loob ng ilang limitasyon, dahil ang ilan sa mga orihinal na halaga ay hindi bound. Kaya, gagamit tayo ng bahagyang naiibang paraan at i-representa ang Q-Table gamit ang dictionary. 

1. Gamitin ang pares *(state,action)* bilang key ng dictionary, at ang value ay tumutugma sa Q-Table entry value. (code block 9)

    ```python
    Q = {}
    actions = (0,1)
    
    def qvalues(state):
        return [Q.get((state,a),0) for a in actions]
    ```

    Dito, nagtakda rin tayo ng function na `qvalues()`, na nagbabalik ng listahan ng Q-Table values para sa isang estado na tumutugma sa lahat ng posibleng aksyon. Kung ang entry ay wala sa Q-Table, magbabalik tayo ng 0 bilang default.

## Simulan ang Q-Learning

Ngayon, handa na nating turuan si Peter na magbalanse!

1. Una, magtakda ng ilang hyperparameters: (code block 10)

    ```python
    # hyperparameters
    alpha = 0.3
    gamma = 0.9
    epsilon = 0.90
    ```

    Dito, ang `alpha` ay ang **learning rate** na tumutukoy kung hanggang saan natin dapat i-adjust ang kasalukuyang mga halaga ng Q-Table sa bawat hakbang. Sa nakaraang aralin, nagsimula tayo sa 1, at pagkatapos ay binawasan ang `alpha` sa mas mababang mga halaga sa panahon ng pagsasanay. Sa halimbawang ito, panatilihin natin itong constant para sa kasimplehan, at maaari kang mag-eksperimento sa pag-adjust ng mga halaga ng `alpha` sa ibang pagkakataon.

    Ang `gamma` ay ang **discount factor** na nagpapakita kung hanggang saan natin dapat bigyang-priyoridad ang hinaharap na reward kaysa sa kasalukuyang reward.

    Ang `epsilon` ay ang **exploration/exploitation factor** na tumutukoy kung mas pipiliin natin ang exploration kaysa exploitation o vice versa. Sa ating algorithm, sa `epsilon` na porsyento ng mga kaso, pipiliin natin ang susunod na aksyon ayon sa Q-Table values, at sa natitirang bilang ng mga kaso, magsasagawa tayo ng random na aksyon. Papayagan tayo nitong tuklasin ang mga bahagi ng search space na hindi pa natin nakikita.

    ✅ Sa usapin ng pagbalanse - ang pagpili ng random na aksyon (exploration) ay kikilos bilang random na suntok sa maling direksyon, at ang pole ay kailangang matutong bumawi ng balanse mula sa mga "pagkakamali."

### Pagbutihin ang Algorithm

Maaari rin nating gawin ang dalawang pagpapabuti sa ating algorithm mula sa nakaraang aralin:

- **Kalkulahin ang average cumulative reward**, sa loob ng ilang simulation. Ipi-print natin ang progreso bawat 5000 iterations, at i-average natin ang cumulative reward sa panahong iyon. Nangangahulugan ito na kung makakakuha tayo ng higit sa 195 puntos - maaari nating ituring na nalutas ang problema, na may mas mataas na kalidad kaysa sa kinakailangan.
  
- **Kalkulahin ang maximum average cumulative result**, `Qmax`, at itatabi natin ang Q-Table na tumutugma sa resulta na iyon. Kapag pinatakbo mo ang training, mapapansin mo na minsan ang average cumulative result ay nagsisimulang bumaba, at gusto nating panatilihin ang mga halaga ng Q-Table na tumutugma sa pinakamahusay na modelo na naobserbahan sa panahon ng pagsasanay.

1. Kolektahin ang lahat ng cumulative rewards sa bawat simulation sa `rewards` vector para sa karagdagang pag-plot. (code block 11)

    ```python
    def probs(v,eps=1e-4):
        v = v-v.min()+eps
        v = v/v.sum()
        return v
    
    Qmax = 0
    cum_rewards = []
    rewards = []
    for epoch in range(100000):
        obs = env.reset()
        done = False
        cum_reward=0
        # == do the simulation ==
        while not done:
            s = discretize(obs)
            if random.random()<epsilon:
                # exploitation - chose the action according to Q-Table probabilities
                v = probs(np.array(qvalues(s)))
                a = random.choices(actions,weights=v)[0]
            else:
                # exploration - randomly chose the action
                a = np.random.randint(env.action_space.n)
    
            obs, rew, done, info = env.step(a)
            cum_reward+=rew
            ns = discretize(obs)
            Q[(s,a)] = (1 - alpha) * Q.get((s,a),0) + alpha * (rew + gamma * max(qvalues(ns)))
        cum_rewards.append(cum_reward)
        rewards.append(cum_reward)
        # == Periodically print results and calculate average reward ==
        if epoch%5000==0:
            print(f"{epoch}: {np.average(cum_rewards)}, alpha={alpha}, epsilon={epsilon}")
            if np.average(cum_rewards) > Qmax:
                Qmax = np.average(cum_rewards)
                Qbest = Q
            cum_rewards=[]
    ```

Ano ang mapapansin mo mula sa mga resulta:

- **Malapit sa ating layunin**. Malapit na nating maabot ang layunin na makakuha ng 195 cumulative rewards sa loob ng 100+ sunud-sunod na pagtakbo ng simulation, o maaaring naabot na natin ito! Kahit na makakuha tayo ng mas mababang mga numero, hindi pa rin natin alam, dahil nag-a-average tayo sa loob ng 5000 pagtakbo, at 100 pagtakbo lamang ang kinakailangan sa pormal na pamantayan.
  
- **Reward nagsisimulang bumaba**. Minsan ang reward ay nagsisimulang bumaba, na nangangahulugang maaari nating "sirain" ang mga natutunang halaga sa Q-Table gamit ang mga bago na nagpapalala sa sitwasyon.

Mas malinaw na makikita ang obserbasyong ito kung i-plot natin ang progreso ng pagsasanay.

## Pag-plot ng Progreso ng Pagsasanay

Sa panahon ng pagsasanay, nakolekta natin ang cumulative reward value sa bawat iteration sa `rewards` vector. Ganito ang hitsura nito kapag na-plot laban sa iteration number:

```python
plt.plot(rewards)
```

![raw progress](../../../../translated_images/train_progress_raw.2adfdf2daea09c596fc786fa347a23e9aceffe1b463e2257d20a9505794823ec.tl.png)

Mula sa graph na ito, hindi posible na makapagsabi ng kahit ano, dahil sa likas na katangian ng stochastic training process, ang haba ng mga training session ay lubos na nag-iiba. Upang mas maunawaan ang graph na ito, maaari nating kalkulahin ang **running average** sa isang serye ng mga eksperimento, sabihin nating 100. Magagawa ito nang maginhawa gamit ang `np.convolve`: (code block 12)

```python
def running_average(x,window):
    return np.convolve(x,np.ones(window)/window,mode='valid')

plt.plot(running_average(rewards,100))
```

![training progress](../../../../translated_images/train_progress_runav.c71694a8fa9ab35935aff6f109e5ecdfdbdf1b0ae265da49479a81b5fae8f0aa.tl.png)

## Pag-iiba ng Hyperparameters

Upang gawing mas matatag ang pag-aaral, makatuwiran na i-adjust ang ilan sa ating mga hyperparameters sa panahon ng pagsasanay. Partikular:

- **Para sa learning rate**, `alpha`, maaari tayong magsimula sa mga halaga na malapit sa 1, at pagkatapos ay patuloy na bawasan ang parameter. Sa paglipas ng panahon, makakakuha tayo ng magagandang probability values sa Q-Table, kaya't dapat nating i-adjust ang mga ito nang bahagya, at hindi ganap na i-overwrite gamit ang mga bagong halaga.

- **Dagdagan ang epsilon**. Maaaring gusto nating unti-unting taasan ang `epsilon`, upang mas kaunti ang exploration at mas marami ang exploitation. Marahil ay makatuwiran na magsimula sa mas mababang halaga ng `epsilon`, at pataasin ito hanggang halos 1.
> **Gawain 1**: Subukan ang iba't ibang halaga ng hyperparameter at tingnan kung makakamit mo ang mas mataas na kabuuang gantimpala. Nakakakuha ka ba ng higit sa 195?
> **Gawain 2**: Upang pormal na malutas ang problema, kailangan mong makakuha ng 195 average na gantimpala sa loob ng 100 magkakasunod na takbo. Sukatin ito habang nagsasanay at tiyaking nalutas mo na ang problema nang pormal!

## Panoorin ang resulta sa aksyon

Magiging interesante na makita kung paano kumikilos ang sinanay na modelo. Patakbuhin natin ang simulation at sundin ang parehong estratehiya sa pagpili ng aksyon tulad ng sa panahon ng pagsasanay, gamit ang sampling ayon sa probability distribution sa Q-Table: (code block 13)

```python
obs = env.reset()
done = False
while not done:
   s = discretize(obs)
   env.render()
   v = probs(np.array(qvalues(s)))
   a = random.choices(actions,weights=v)[0]
   obs,_,done,_ = env.step(a)
env.close()
```

Makikita mo ang ganito:

![isang balancing cartpole](../../../../8-Reinforcement/2-Gym/images/cartpole-balance.gif)

---

## 🚀Hamunin

> **Gawain 3**: Dito, ginagamit natin ang huling kopya ng Q-Table, na maaaring hindi ang pinakamainam. Tandaan na iniimbak natin ang pinakamahusay na Q-Table sa `Qbest` na variable! Subukan ang parehong halimbawa gamit ang pinakamahusay na Q-Table sa pamamagitan ng pagkopya ng `Qbest` papunta sa `Q` at tingnan kung may mapapansin kang pagkakaiba.

> **Gawain 4**: Dito, hindi natin pinipili ang pinakamahusay na aksyon sa bawat hakbang, kundi nagsa-sample tayo gamit ang kaukulang probability distribution. Mas may saysay kaya na palaging piliin ang pinakamahusay na aksyon, na may pinakamataas na halaga sa Q-Table? Magagawa ito gamit ang `np.argmax` na function upang malaman ang numero ng aksyon na may pinakamataas na halaga sa Q-Table. Ipatupad ang estratehiyang ito at tingnan kung mas mapapabuti nito ang balanse.

## [Post-lecture quiz](https://gray-sand-07a10f403.1.azurestaticapps.net/quiz/48/)

## Takdang-Aralin
[Sanayin ang Mountain Car](assignment.md)

## Konklusyon

Natutuhan na natin ngayon kung paano sanayin ang mga ahente upang makamit ang magagandang resulta sa pamamagitan lamang ng pagbibigay sa kanila ng reward function na naglalarawan ng nais na estado ng laro, at pagbibigay sa kanila ng pagkakataong matalinong galugarin ang search space. Matagumpay nating naipatupad ang Q-Learning algorithm sa mga kaso ng discrete at continuous na kapaligiran, ngunit may discrete na mga aksyon.

Mahalaga ring pag-aralan ang mga sitwasyon kung saan ang estado ng aksyon ay tuloy-tuloy din, at kung kailan mas kumplikado ang observation space, tulad ng imahe mula sa screen ng laro ng Atari. Sa mga problemang ito, madalas nating kailangang gumamit ng mas makapangyarihang mga teknik sa machine learning, tulad ng neural networks, upang makamit ang magagandang resulta. Ang mga mas advanced na paksang ito ay tatalakayin sa paparating nating mas advanced na kurso sa AI.

---

**Paunawa**:  
Ang dokumentong ito ay isinalin gamit ang AI translation service na [Co-op Translator](https://github.com/Azure/co-op-translator). Bagama't sinisikap naming maging tumpak, pakitandaan na ang mga awtomatikong pagsasalin ay maaaring maglaman ng mga pagkakamali o hindi pagkakatugma. Ang orihinal na dokumento sa orihinal nitong wika ang dapat ituring na opisyal na sanggunian. Para sa mahalagang impormasyon, inirerekomenda ang propesyonal na pagsasalin ng tao. Hindi kami mananagot sa anumang hindi pagkakaunawaan o maling interpretasyon na dulot ng paggamit ng pagsasaling ito.