# Samples of the SFT conversions

Two converted rows per converter, taken from the first few hundred rows of each source. `user` is the prompt as the model sees it, `assistant` is the
canonical turn. Long fields are cut. Nothing here was edited by hand.

## english_to_igbo

Aletheia-ng/english_to_igbo -> eng<->ibo translation

**task** `translation` · **source_lang** `ibo` · **targ_lang** `eng` · **dataset** `Aletheia-ng/english_to_igbo` · **instruction language** `pcm`

- **user**: `Ihe dika ọnwa asatọ dị n'ime afọ ahụ aga, eritebeghị pasent 40 n'ime ya. Make you change all dis one go English.`
- **assistant**: `<|input_lang|><ibo><pcm><task_plan><translate></task_plan><|target_lang|><eng><response>About eight months into the year, less than 40 per cent has been received.`

**task** `translation` · **source_lang** `ibo` · **targ_lang** `eng` · **dataset** `Aletheia-ng/english_to_igbo` · **instruction language** `eng`

- **user**: `Below is a short text in Igbo. Provide its translation in English, and nothing else. ⏎  ⏎ Nnia Nwodo: Ṅomie nụ nna unu Joe Achuzia`
- **assistant**: `<|input_lang|> unsure, but most likely <eng><ibo><task_plan><translate></task_plan><|target_lang|> unsure, but most likely <eng><response>Nnia Nwodo: emulate your father Joe Achuzia`

## webscraped_igbo

Aletheia-ng/WebScraped_Igbo_To_Eng -> eng<->ibo translation

**task** `translation` · **source_lang** `eng` · **targ_lang** `ibo` · **dataset** `Aletheia-ng/WebScraped_Igbo_To_Eng` · **instruction language** `eng`

- **user**: `Rewrite the passage below in Igbo, making sure it reads the way a native speaker would say it. ⏎ News that will interest you:`
- **assistant**: `<|input_lang|><eng><task_plan><translate></task_plan><|target_lang|><ibo><response>Akụkọ ndị ga-masi gị:`

**task** `translation` · **source_lang** `ibo` · **targ_lang** `eng` · **dataset** `Aletheia-ng/WebScraped_Igbo_To_Eng` · **instruction language** `eng`

- **user**: `Ihe niile nwaanyị chọtara ime, ọ ga-eme ya ọfụma. ⏎  ⏎ Would you mind translating this into English? If a term has no direct equivalent, keep the original term.`
- **assistant**: `<|input_lang|><ibo><eng><task_plan><translate></task_plan><|target_lang|><eng><response>Everything a woman decides to do, she will do it well.`

## efik_bible

Aletheia-ng/English_efik_bible_translation -> eng<->efi translation (Bible)

**task** `translation` · **source_lang** `eng` · **targ_lang** `efi` · **dataset** `Aletheia-ng/English_efik_bible_translation` · **instruction language** `eng`

- **user**: `Render the text below into fluent Efik, retaining the formality of the original. ⏎  ⏎ John to the seven churches which are in Asia: Grace be unto you, and peace, from him which is, and which was, and which is to come; and from the seven Spirits which are before his throne;`
- **assistant**: `<|input_lang|><eng><task_plan><translate></task_plan><|target_lang|><efi><response>John ɔnɔ ŋwed esɔk mme ufɔk Abasi itiaba emi edude ke Asia. Mfɔn ye emem akpakam oto Enye emi odude-du, emi okonyuŋ odude-du, emi edinyuŋ edide, osim mbufo: onyuŋ oto ke mme Spirit itiaba emi edude ke iso ebekpo Esie;`

**task** `translation` · **source_lang** `efi` · **targ_lang** `eng` · **dataset** `Aletheia-ng/English_efik_bible_translation` · **instruction language** `hau`

- **user**: `onyuŋ oto Jesus Christ, emi edide ntiense emi anamde akpanikɔ, andibem iso mmana ke otu mme akpa-mkpa, ye andikara ndidem isɔŋ. Enye emi amade nnyin, okonyuŋ adade iyip Esiemɔ atat nnyin uruk mme idiɔk-ŋkpɔ nnyin; ⏎  ⏎ Fassara shi zuwa Turanci.`
- **assistant**: `<|input_lang|><efi><hau><task_plan><translate></task_plan><|target_lang|><eng><response>and from Jesus Christ, who is the faithful witness, and the first begotten of the dead, and the prince of the kings of the earth. Unto him that loved us, and washed us from our sins in his own blood,`

## aya_interlanguage

Aletheia-ng/aya_interlanguage_translate (yoruba, ibo, hausa + english) -> interlanguage translation, 3 random directions per row

**task** `translation` · **source_lang** `yor` · **targ_lang** `hau` · **dataset** `Aletheia-ng/aya_interlanguage_translate` · **instruction language** `eng`

- **user**: `Ka ẹsẹ Bíbélì náà kó o sì dáhùn àwọn ìbéèrè náà. Timmy fẹ́ràn láti máa ṣeré ìdárayá àti eré ìdárayá, àmọ́ ó fẹ́ràn láti máa kó nǹkan jọ ju ohunkóhun mìíràn lọ. Ó máa ń kó àwọn nǹkan tí wọ́n fi ń dí ìgò. Ó máa ń kó àwọn apẹ̀rẹ̀ jọ. Ó máa ń kó káàdì baseball jọ. Òun lẹni tó ti pẹ́ jù lọ tó ń kó káàdì baseball jọ. Ó fẹ́ràn láti máa kó àwọn nǹkan tó ti pẹ́ jù lọ tó ti ń kó jọ. Ìgbà kan wà tó ronú pé òun fẹ́ máa kó àwọn káàdì tí wọ́n fi ń ránṣẹ́ sí àwọn èèyàn, àmọ́ kò ṣe bẹ́ẹ̀. Kì í ṣe àwọn ohun tó fẹ́ràn jù lọ ló kó jọ [...]`
- **assistant**: `<|input_lang|><yor><eng><task_plan><translate></task_plan><|target_lang|><hau><response>Karanta nassin kuma ka amsa tambayoyin. Timmy yana son yin wasa da wasanni amma fiye da komai yana son tattara abubuwa. Ya tattara kwalban kwalba. Ya tattara kwalliyar teku. Ya tattara katunan wasan baseball. Shi ne ya fi dadewa yana tattara katunan wasan baseball. Yana son tattara abin da ya tattara mafi tsawo. Ya taba tunanin tattara hatimi amma bai taba yi ba. Ba abin da ya fi so ba ne a cikin kayan da ya tara da suka fi tsad [...]`

**task** `translation` · **source_lang** `yor` · **targ_lang** `hau` · **dataset** `Aletheia-ng/aya_interlanguage_translate` · **instruction language** `eng`

- **user**: `Express the content of this passage in Hausa; accuracy matters more than elegance. ⏎  ⏎ Orí Kejìlá. ỌLỌ́RUN MÍÌ TÓ KAN. Awọn ẹgbẹ ni Rudham Park ti o nira ti a aseyori, - tabi ni o je Elo ni ilọsiwaju ni wit tabi gaiety nigbati Mrs. Montacute Jones, Oluwa Giblet, ati Jack de Baron ti lọ kuro, ati Canon Holdenough ati iyawo rẹ, pẹlu Mr. Groschut, ti wá ni wọn ibi. Eleyi dudu influx, bi Oluwa Brotherton pè o, ti gbogbo ti a ti nitori ti akiyesi fun rẹ Lordship. Ọgbẹni De Baron ro pe alejo rẹ yoo fẹ lati ri, ni eyikeyi id [...]`
- **assistant**: `<|input_lang|><eng><yor><task_plan><translate></task_plan><|target_lang|><hau><response>Fasali na Biyu. WANI ƘAUWA. A jam'iyyar a Rudham Park ya wuya kasance nasara, - kuma ba shi da yawa inganta a wit ko gaiety lokacin da Mrs. Montacute Jones, Ubangiji Giblet, da kuma Jack de Baron ya tafi, da kuma Canon Holdenough da matarsa, tare da Mr. Groschut, ya zo a cikin su wurare. Wannan baki influx, kamar yadda Ubangiji Brotherton kira shi, ya kasance duk saboda la'akari da ya Lordship. Mr. De Baron tunani cewa ya baƙo z [...]`

## aya_translate

Aletheia-ng/aya-translate (english, yoruba, ibo, hausa) -> translation, 3 random directions per row

**task** `translation` · **source_lang** `eng` · **targ_lang** `ibo` · **dataset** `Aletheia-ng/aya-translate` · **instruction language** `eng`

- **user**: `Say this in Igbo. ⏎  ⏎ (CNN) -- Two former presidents reflected on their greatest regrets in office Monday, each looking back to issues that continue to plague the nation years later. ⏎  ⏎ Former presidents and political rivals Bill Clinton and George H.W. Bush now share philanthropic efforts. ⏎  ⏎ Former Presidents George H.W. Bush and Bill Clinton appeared together at a question-and-answer forum before the National Automobile Dealers Association in New Orleans, Louisiana. ⏎  ⏎ Asked his biggest regret after leaving office, Bush [...]`
- **assistant**: `<|input_lang|><eng><task_plan><translate></task_plan><|target_lang|><ibo><response>(CNN) -- Ndị bụbu ndị isi ala abụọ tụgharịrị uche na nnukwu ụta ha nwere n'ọfịs na Mọnde, onye ọ bụla na-eleghachi anya n'okwu ndị na-aga n'ihu na-akpata mba ahụ ọtụtụ afọ mgbe e mesịrị. Ndị bụbu ndị isi ala na ndị ndọrọ ndọrọ ọchịchị Bill Clinton na George H.W. Bush na-esonyezi n'ọrụ ebere. Ndị bụbu ndị isi ala George H.W. Bush na Bill Clinton pụtara ọnụ n'otu nnọkọ ajụjụ na azịza n'ihu National Automobile Dealers Association na New [...]`

**task** `translation` · **source_lang** `hau` · **targ_lang** `yor` · **dataset** `Aletheia-ng/aya-translate` · **instruction language** `hau`

- **user**: `Abidjan, Ivory Coast (CNN) -- Kungiyar Tarayyar Turai ta sanar da shirin dawo da kudin Tarayyar Turai na Euro miliyan 180 ga Ivory Coast a ranar Talata yayin da mazauna kasar Afirka ke kokarin daidaitawa da rayuwa tare da shugaba mai haske da kwanciyar hankali bayan watanni na zubar da jini. Jami'an tsaro sun kama tsohon shugaban kasa Laurent Gbagbo bayan da suka mamaye gidansa a ranar Litinin. Gbagbo ya kalubalanci kiraye-kirayen da aka yi masa na yin murabus bayan da hukumar zabe ta bayyana cewa ya yi rashin nasa [...]`
- **assistant**: `<|input_lang|><hau><task_plan><translate></task_plan><|target_lang|><yor><response>Abidjan, Ivory Coast (CNN) -- European Union kede package imularada ti 180 milionu awọn owo ilẹ yuroopu fun Ivory Coast ni ọjọ Tuesday bi awọn olugbe ti orilẹ-ede Afirika ṣe gbiyanju lati ṣe deede si igbesi aye pẹlu oludari ti o mọ ati iduroṣinṣin ibatan lẹhin awọn oṣu ti ẹjẹ. Àwọn aláṣẹ mú Ààrẹ tẹ́lẹ̀ rí Laurent Gbagbo lẹ́yìn tí wọ́n wọlé rẹ̀ lọ́jọ́ Monday. Gbagbo kọ̀ láti fi ipò sílẹ̀ lẹ́yìn tí ìgbìmọ̀ ìdìbò kéde pé ó pàdánù ìdìbò [...]`

## gsm8k_translate

Aletheia-ng/gsm8k{,2,3} questions as interlanguage translation

_no row survived the filters in the sample (see the report for why)_

## naija

BeardedMonster/naija: every tagged family (translation, prompt/response, summarisation, headline, sentiment/topic/intent, math) from the TRAIN files; language detected from the text, not the file name

**task** `translation` · **source_lang** `eng` · **targ_lang** `orm` · **dataset** `BeardedMonster/naija:AfroLingu-MT_ibo_prompt-response` · **instruction language** `hau`

- **user**: `But Jesus, knowing their wickedness, said: Why do you put me to the test, hypocrites? ⏎ Fassara wannan zuwa Afaan Oromoo.`
- **assistant**: `<|input_lang|><eng><hau><task_plan><translate></task_plan><|target_lang|><orm><response>Yesuus garuu hammina isaanii beekee akkana isaaniin jedhe: Fakkeessitoota a, maaliif na qortu?`

**task** `headline` · **source_lang** `yor` · **targ_lang** `yor` · **dataset** `BeardedMonster/naija:xlsum_multilingual_headline` · **instruction language** `yor`

- **user**: `Kọ àkọ́lé fún ìròyìn yìí: ⏎  ⏎ Awọn ọmọ ile igbimọ asoju sofin l'Abuja sọ eyi di mimọ, lẹyin ti wọn wọle pada lẹyin isinmi oloṣu meji ti wọn gunle, fun isinmi ọdọdun ti wọn maa n ṣe. Àwọn iròyin tì ẹ leè ní ìfẹ síí: Ijoko ile naa, ti abẹnugan ile Femi Gbajabiamila dari, salaye pe, o ṣe pataki ki ijọba apapọ tẹle adehun ti awọn ati ẹgbẹ oṣiṣẹ se ti wọn fi wọgile ipinu wọn lati bẹrẹ iyanṣẹlodi to yẹ ko bẹrẹ lọjọ aje. O ni ninu ifọrọwerọ wọn, ijọba apapọ ṣe adehun fawọn ẹgbẹ oṣiṣẹ ati ọmọ Naijiria lapapọ, eyi ti wọn si gb [...]`
- **assistant**: `<|input_lang|><yor><task_plan><summarize><headline></task_plan><|target_lang|><yor><response>Petrol Price Hike: Ilé aṣojú-ṣòfin dúró ìṣẹjú kan fún Buruji Kashamu àti Rose Okoh`

**task** `intent` · **source_lang** `afr` · **targ_lang** `eng` · **dataset** `BeardedMonster/naija:amazon_massive_intent_['yor', 'yo', 'hau', 'ha', 'ibo', 'igbo', 'ig', 'pcm', 'urh', 'efi', 'ful', 'ff', 'fuv', 'knc', 'oke', 'tiv', 'ibb', 'nup']_intent` · **instruction language** `eng`

- **user**: `maak die vloer skoon asseblief What is the intent of the utterance above?`
- **assistant**: `<|input_lang|><afr><eng><task_plan><classify><intent></task_plan><|target_lang|><eng><response>iot_cleaning`

**task** `translation` · **source_lang** `eng` · **targ_lang** `afr` · **dataset** `BeardedMonster/naija:AfroLingu-MT_ibo_prompt-response` · **instruction language** `pcm`

- **user**: `In New Market we had a special treat . Translate wetin dey above into Afrikaans, no add anything.`
- **assistant**: `<|input_lang|><eng><pcm><task_plan><translate></task_plan><|target_lang|><afr><response>In New Market het n spesiale verrassing op ons gewag .`

**task** `translation` · **source_lang** `eng` · **targ_lang** `twi` · **dataset** `BeardedMonster/naija:AfroLingu-MT_ibo_prompt-response` · **instruction language** `eng`

- **user**: `Whatever the original wording, I want the passage below conveyed accurately in Twi. ⏎ Ashitey was born in Ledzokuku in the Greater Accra Region of Ghana on 21 November 1955.[1] She attended the America University of the Caribbean and obtained a Degree in Doctor of Medicine after studying Medicine as a Programme. She also attended the University of Ghana and obtained her Bachelor of Science (Medicine).[1]`
- **assistant**: `<|input_lang|><eng><task_plan><translate></task_plan><|target_lang|><twi><response>Wɔ woo Ashitey wɔ Ledzokuku a ɛwɔ Nkran Mantam mu a ɛwɔ Ghana wɔ November a ɛtɔ so 21, afe 1955. ɔkoo suapɔn a ɛne American University of the Caribbean ma no nyaa abodin krataa wɔ nnuroyɛ mu. ɔsan nso kɔɔ Suapɔn a ɛne University of Ghana konyaa nimdeɛ ne abɔdin wɔ nnuroyɛ mu.`

**task** `headline` · **source_lang** `yor` · **targ_lang** `yor` · **dataset** `BeardedMonster/naija:xlsum_multilingual_headline` · **instruction language** `yor`

- **user**: `Fún àyọkà yìí ní àkọ́lé ní èdè Yoruba: Gomina tẹlẹrí ní ìpínlẹ Ọyo, Alao Akala ń díje dupò gomina lábẹ ẹgbẹ òsèlú Action Democratic Party (ADP). Alao-Akala lọ si ile iwe Government Technical College, Tamale, Ghana ati Ogbomoso High School, Ogbomoso. Bakan naa ni oludije sipo gomina labẹ ẹgbẹ oselu ADP naa lọ si ile iwe giga fasiti ti ilu Ibadan. Lẹyin naa lo bẹrẹ isẹ ọlọpaa titi to fi gba oye ADC si Ọga Agba Ọlọpaa lorilẹede Naijiria. Nkan meje to yẹ ko o mọ nipa Alao-Akala ti Ọyọ`
- **assistant**: `<|input_lang|><yor><task_plan><summarize><headline></task_plan><|target_lang|><yor><response>Nigeria 2019 Election: Ǹjẹ o mọ Alao Akala tó ń díje gomina ní Ọyọ?`

**task** `headline` · **source_lang** `yor` · **targ_lang** `yor` · **dataset** `BeardedMonster/naija:xlsum_multilingual_headline` · **instruction language** `hau`

- **user**: `Ba wannan labarin taken kanun labari a Yoruba: Oun ni aarẹ ẹgbẹ Young Yoruba for Freedom, YYF, ẹgbẹ kan to ni afojusun oun ni lati tun ìtàn kọ, nipa fifa ẹya Yoruba jade kuro ni Naijiria, lati da orilẹ-ede tuntun Oduduwa Republic silẹ. O ni oju opo kan sori ayelujara Facebook, nibi to ma n fi awọn fidio to fi n gbe afojusun rẹ lẹyin, si. Ọdun 2012 lo da ẹgbẹ YYF silẹ, lati le "tun ero awọn ọdọ Yoruba ṣe, ati lati le ja fun iṣejọba ẹlẹkun-jẹkun". Ṣé lóòtọ ni Afenifere fún àwọn èèyàn ẹyà Igbo ní wákàtí 48 láti fi ilẹ [...]`
- **assistant**: `<|input_lang|><hau><yor><task_plan><summarize><headline></task_plan><|target_lang|><yor><response>Adeyinka Godson fún ẹyà Igbo ní gbèdéke láti fi ilẹ Yorubá sílẹ`

**task** `translation` · **source_lang** `zul` · **targ_lang** `eng` · **dataset** `BeardedMonster/naija:AfroLingu-MT_ibo_prompt-response` · **instruction language** `eng`

- **user**: `I'm writing to a relative who prefers another language. Rewrite the passage below in English, making sure it reads the way a native speaker would say it. ⏎ Kepha masinyane emva kosizi lwalezo zinsuku ilanga liyakufiphazwa, nenyanga ayiyikukhanya, nezinkanyezi ziyakuwa ezulwini, namandla ezulu ayakuzanyazanyiswa.`
- **assistant**: `<|input_lang|><eng><zul><task_plan><translate></task_plan><|target_lang|><eng><response>Immediately after the tribulation of those days shall the sun be darkened, and the moon shall not give her light, and the stars shall fall from heaven, and the powers of the heavens shall be shaken:`

**task** `sentiment` · **source_lang** `ibo` · **targ_lang** `eng` · **dataset** `BeardedMonster/naija:igbo-emotions-corpus_ibo_sentiment_classification` · **instruction language** `ibo`

- **user**: `Rue mgbe ayi genwe onodu n'igwé. ⏎ Kedu mmetụta nke ederede dị n'elu?`
- **assistant**: `<|input_lang|> unsure, but most likely <ibo><task_plan><classify><sentiment></task_plan><|target_lang|><eng><response>neutral`

**task** `translation` · **source_lang** `eng` · **targ_lang** `ibo` · **dataset** `BeardedMonster/naija:AfroLingu-MT_ibo_prompt-response` · **instruction language** `eng`

- **user**: `Imagine you are explaining the text below to a Igbo speaker who has never heard of its subject; translate it faithfully into Igbo. ⏎  ⏎ Visitors can also review the other islands around Derawan.`
- **assistant**: `<|input_lang|><eng><task_plan><translate></task_plan><|target_lang|><ibo><response>Ọbịa nwekwara ike nyochaa n'àgwàetiti ndị ọzọ gburugburu Derawan.`

## naija2

BeardedMonster/naija2: same families (plus MMLU dev with letters, MathScale, grade-school-math); Math-Shepherd dropped

**task** `topic` · **source_lang** `afr` · **targ_lang** `eng` · **dataset** `BeardedMonster/naija2:sib200_fon_Latn_topic_classification` · **instruction language** `eng`

- **user**: `What is the topic of this news headline? ⏎ Vul u huis met n lekker koffie in die oggend en n koppie ontspannende kamilletee in die aand.`
- **assistant**: `<|input_lang|><eng><afr><task_plan><classify><topic></task_plan><|target_lang|><eng><response>entertainment`

**task** `math` · **source_lang** `eng` · **targ_lang** `eng` · **dataset** `BeardedMonster/naija2:MathScaleQA-2M_multilingual_reasoning` · **instruction language** `eng`

- **user**: `Nancy, the librarian, is shelving books from the cart. She shelved 12 history books, 8 romance books, and 4 poetry books from the top section of the cart. Half the books on the bottom section of the cart were mystery books, which she quickly put back into place. Then, she shelved the remaining books from the bottom of the cart, including 5 Western novels and 6 biographies. How many books did she have on the book cart when she started? ⏎ Solve the problem above step by step.`
- **assistant**: `<|input_lang|><eng><task_plan><|math|></task_plan><|target_lang|><eng><response>To find the total number of books Nancy had on the book cart, we need to first determine the number of books in the top and bottom sections of the cart.In the top section of the cart, Nancy shelved:- 12 history books- 8 romance books- 4 poetry booksAdding these together: 12 + 8 + 4 = 24 books in the top section.In the bottom section of the cart, Nancy shelved:- 5 Western novels- 6 biographiesAdding these together: 5 + 6 = 11 books.Now w [...]`

**task** `summarization` · **source_lang** `eng` · **targ_lang** `eng` · **dataset** `BeardedMonster/naija2:Kaggle_CNN_Text_Summarization_english_summarization` · **instruction language** `twi`

- **user**: `Tɔw saa asɛm yi mu kɔ ase wɔ English mu: By . Harriet Arkell For Mailonline . A pair of friends who dreamt up Marmite-themed board games during drunken dinners together have told how they won 50,000 backing from the Dragons' Den investment show. Father of two Richard McLuckie, 48, and his friend Stuart Mackenzie-Walker, 51, dreamt up a series of board games including Love It Or Hate It and Who Put The Marmite In The Fridge, over long evenings spent playing games and drinking wine. They won permission to use the nam [...]`
- **assistant**: `<|input_lang|><twi><eng><task_plan><summarize></task_plan><|target_lang|><eng><response>Richard McLuckie, 48, and Stuart Mackenzie-Walker, 51, invented games .Won permission from Marmite owner Unilever to use its name and image .Then they went on investment TV show to ask for funding from the Dragons .But Unilever contract said entrepreneurs couldn't mention name Marmite .Three Dragons pulled out, but Peter Jones and Duncan Bannatyne agreed .They paid the men 50,000 for a 40 per cent stake in board game business .`

**task** `translation` · **source_lang** `ewe` · **targ_lang** `eng` · **dataset** `BeardedMonster/naija2:tatoeba_mt_train_english_translation` · **instruction language** `eng`

- **user**: `Le kpɔɖeŋu me, de ŋugble le Kristotɔwo ƒe subɔsubɔdɔa ŋu kpɔ. Could you translate it into English for me? I would like to share it with someone who reads English.`
- **assistant**: `<|input_lang|><ewe><eng><task_plan><translate></task_plan><|target_lang|> unsure, but most likely <eng><response>For example, consider the Christian ministry.`

**task** `topic` · **source_lang** `afr` · **targ_lang** `eng` · **dataset** `BeardedMonster/naija2:sib200_fon_Latn_topic_classification` · **instruction language** `pcm`

- **user**: `Wetin be di topic of dis headline? ⏎  ⏎ Uit die ongeveer 24 000 bekende meteoriete wat al op die Aarde geval het, is dit bevestig dat slegs 34 van Mars afkomstig was.`
- **assistant**: `<|input_lang|><pcm><afr><task_plan><classify><topic></task_plan><|target_lang|><eng><response>geography`

**task** `summarization` · **source_lang** `eng` · **targ_lang** `eng` · **dataset** `BeardedMonster/naija2:Kaggle_CNN_Text_Summarization_english_summarization` · **instruction language** `eng`

- **user**: `TL;DR in English: ⏎ Spherical fossils in China dating back 600 million years could be the remains of the planet's earliest animals. The fossils are 60 million years older than the date skeletal animals appeared during a huge growth spurt of new life on Earth known as the Cambrian Explosion. Virginia scientists claim the fossils are too complex to be bacteria, and may instead be the embryos of an ancient unidentified creatures. Spherical fossils in China dating back 600 million years could be the remains of the planet [...]`
- **assistant**: `<|input_lang|><eng><task_plan><summarize></task_plan><|target_lang|><eng><response>Called Megasphaera, the fossils were found in rock in southern China .They are 60 million years older than date skeletal animals appeared .But the cells are too complex to be belong to bacteria, the study claims .They may represent shift between single-cell life and multicellular animals .`

**task** `summarization` · **source_lang** `eng` · **targ_lang** `eng` · **dataset** `BeardedMonster/naija2:Kaggle_CNN_Text_Summarization_english_summarization` · **instruction language** `ewe`

- **user**: `Not since Wimbledon has Andy Murray played 40 minutes of tennis as nerve shredding as this passage that came on a Parisian lunchtime, the conclusion of which saw him squeeze through to the last 16 of the French Open. Resuming at 7-7 against world No 24 Philipp Kohlschreiber he needed to draw on all his Grand Slam experience to eke out a 3-6 6-3 6-3 4-6 12-10 victory that took two days to complete and 247 minutes on the court. Murray fired home a backhand return winner on his second match point when he finally manag [...]`
- **assistant**: `<|input_lang|><eng><ewe><task_plan><summarize></task_plan><|target_lang|><eng><response>Play resumed in the third-round French Open match on Sunday .Match was suspended on Saturday night at 3-6, 6-3, 6-3, 4-6, 7-7 .Murray claimed the match on his second match point, 12-10 in the fifth .The epic five-setter ended after 247 minutes of time on court .He now meets Spain's Ferdo Verdasco in the fourth round .`

**task** `topic` · **source_lang** `afr` · **targ_lang** `eng` · **dataset** `BeardedMonster/naija2:sib200_fon_Latn_topic_classification` · **instruction language** `ibo`

- **user**: `Kewaa ederede a n'otu ngalaba: ⏎  ⏎ In die noorde is die streek begrens deur die Sahel, en in die suide en weste deur die Atlantiese Oseaan.`
- **assistant**: `<|input_lang|><ibo><afr><task_plan><classify><topic></task_plan><|target_lang|><eng><response>geography`

**task** `topic` · **source_lang** `afr` · **targ_lang** `eng` · **dataset** `BeardedMonster/naija2:sib200_fon_Latn_topic_classification` · **instruction language** `eng`

- **user**: `Huidige senator en Argentynse Presidentsvrou, Christina Ferdez de Kirchner het haar presidensiële kandidaatskap gisteraand in La Plata, n stad 50 kilometer (31 myl) buite Buenos Aires, aangekondig. ⏎ Which category is this?`
- **assistant**: `<|input_lang|><afr><eng><task_plan><classify><topic></task_plan><|target_lang|><eng><response>politics`

**task** `topic` · **source_lang** `afr` · **targ_lang** `eng` · **dataset** `BeardedMonster/naija2:sib200_fon_Latn_topic_classification` · **instruction language** `eng`

- **user**: `Classify this headline into a topic: ⏎  ⏎ n Lem van 'n moderne ysskaatsskoen het 'n dubbele rant met 'n holronde tussen-in. Die twee rante laat toe vir 'n beter greep op die ys, selfs wanneer dit gekantel word.`
- **assistant**: `<|input_lang|><eng><afr><task_plan><classify><topic></task_plan><|target_lang|><eng><response>science/technology`

## naija_late

BeardedMonster/naija *_monolingual files -> pretrain-new/late/<lang> (plain text, language detected)

_no row survived the filters in the sample (see the report for why)_

## naija2_late

BeardedMonster/naija2 *_monolingual files -> pretrain-new/late/<lang> (Inkuba-Mono, W2C, HPLT, C4, ...)

_no row survived the filters in the sample (see the report for why)_

## tds

Aletheia-ng/tds: <translate> A <lang> B, gated on its africomet_score >= 0.55

**task** `translation` · **source_lang** `hau` · **targ_lang** `eng` · **dataset** `Aletheia-ng/tds:train-00000-of-00010` · **instruction language** `eng`

- **user**: `Help me say the following in English: Keep the formal register of the original. ⏎  ⏎ A karshe Nguroje, ya ce tuni am mika su ga hukumar ladabtar da masu aikata irin wan laifi, san kuma yan sanda na ci gaba da bincike.`
- **assistant**: `<|input_lang|><eng><hau><task_plan><translate></task_plan><|target_lang|><eng><response>Finally, Nguroje said they have been handed over to the agency responsible for punishing perpetrators of such offenses, and police investigations are ongoing.`

**task** `translation` · **source_lang** `hau` · **targ_lang** `eng` · **dataset** `Aletheia-ng/tds:train-00000-of-00010` · **instruction language** `pcm`

- **user**: `Hanta yana shan wahala sosai tare da irin wannan cutar. ⏎  ⏎ I need di English version of di text wey dey up. Make e sound like how person for dat language go talk am.`
- **assistant**: `<|input_lang|><hau><pcm><task_plan><translate></task_plan><|target_lang|><eng><response>ject has endured much suffering from the disease.`

## old_hate

BeardedMonster/hate (AfriHate-style) -> <classify><|hate|>

**task** `hate` · **source_lang** `hau` · **targ_lang** `eng` · **dataset** `BeardedMonster/hate` · **instruction language** `orm`

- **user**: `Jecha jibbiinsaa barreeffama kana keessaa adda baasi: @username sai dai kaji... dalla jah 'mahaukaciya' 'wawiya'`
- **assistant**: `<|input_lang|><orm><hau><task_plan><classify><|hate|></task_plan><|target_lang|><eng><response>Abuse`

**task** `hate` · **source_lang** `hau` · **targ_lang** `eng` · **dataset** `BeardedMonster/hate` · **instruction language** `orm`

- **user**: `Barreeffamni kun miira jibbiinsaa calaqqisiisaa? @username kasan dan kauye ne 😂`
- **assistant**: `<|input_lang|><orm><hau><task_plan><classify><|hate|></task_plan><|target_lang|><eng><response>Abuse`

## old_intent

BeardedMonster/intent -> <classify><intent>

**task** `intent` · **source_lang** `ewe` · **targ_lang** `eng` · **dataset** `BeardedMonster/intent` · **instruction language** `fon`

- **user**: `Ɖɔ nǔ e mɛ lɔ ba é: Taƒlatse, le mɔ na gaƒoɖokuia wòaɖe gbe ga adee pɛpɛpɛ!`
- **assistant**: `<|input_lang|><fon><ewe><task_plan><classify><intent></task_plan><|target_lang|><eng><response>alarm`

**task** `intent` · **source_lang** `ewe` · **targ_lang** `eng` · **dataset** `BeardedMonster/intent` · **instruction language** `pcm`

- **user**: `Wetin be de aim of dis gist? wɔ fɔfɔ̃ ƒe ɖoɖo ɖe gaƒoɖokui dzi nam`
- **assistant**: `<|input_lang|><pcm><ewe><task_plan><classify><intent></task_plan><|target_lang|><eng><response>alarm`

## old_topic

BeardedMonster/topic (news topics by name; SIB-200 *_Latn_topic files: class index -> name; unsupported languages dropped)

**task** `topic` · **source_lang** `afr` · **targ_lang** `eng` · **dataset** `BeardedMonster/topic` · **instruction language** `pcm`

- **user**: `Tell me de correct group for Aan die begin van die oorlog het hulle grootendeels bo op die water gereis, maar soos radar begin ontwikkel het en meer akkuraat geraak het, was die duikbote geforseer om onderwater te gaan om te vermy dat hulle gesien word. among de options.`
- **assistant**: `<|input_lang|><pcm><afr><task_plan><classify><topic></task_plan><|target_lang|><eng><response>science/technology`

**task** `topic` · **source_lang** `afr` · **targ_lang** `eng` · **dataset** `BeardedMonster/topic` · **instruction language** `unknown`

- **user**: `Winnu innde rukun wonande Dis amper soos om ‘n swaar wa by ‘n heuwel op te stoot. Die splitsing van ‘n nukleus stel van daardie energie vry..`
- **assistant**: `<|input_lang|><afr><task_plan><classify><topic></task_plan><|target_lang|><eng><response>science/technology`

## old_pos

BeardedMonster/pos -> <POS><tag> word/tag pairs (NER tags found under POS are dropped)

**task** `ner` · **source_lang** `eng` · **targ_lang** `eng` · **dataset** `BeardedMonster/pos` · **instruction language** `yor`

- **user**: `Ta ni ó ni orúkọ nínú ọ̀rọ̀ yìí? At Northampton : Sussex 368-7 ( N. Lenham 145 , V. Drakes 59 not`
- **assistant**: `<|input_lang|><yor><eng><task_plan><NER><tag></task_plan><|target_lang|><eng><response>At <tag> O Northampton <tag> B-LOC : <tag> O Sussex <tag> B-ORG 368-7 <tag> O ( <tag> O N. <tag> B-PER Lenham <tag> I-PER 145 <tag> O , <tag> O V. <tag> B-PER Drakes <tag> I-PER 59 <tag> O not <tag> O`

**task** `ner` · **source_lang** `eng` · **targ_lang** `eng` · **dataset** `BeardedMonster/pos` · **instruction language** `hau`

- **user**: `Nuna mini sunayen kungiyoyin da aka ambata: Doe Run pure lead price ( effective date : August 14 , 1996 )`
- **assistant**: `<|input_lang|><hau><eng><task_plan><NER><tag></task_plan><|target_lang|><eng><response>Doe <tag> B-ORG Run <tag> I-ORG pure <tag> O lead <tag> O price <tag> O ( <tag> O effective <tag> O date <tag> O : <tag> O August <tag> O 14 <tag> O , <tag> O 1996 <tag> O ) <tag> O`

## old_tasks

BeardedMonster/tasks (NER, sentiment, summarisation, headline, instruct)

**task** `ner` · **source_lang** `hau` · **targ_lang** `hau` · **dataset** `BeardedMonster/tasks` · **instruction language** `hau`

- **user**: `Who and who dem mention for: Tsohon ministan Shari’a kenan kuma daya daga cikin mutane da shugaba Muhammadu Buhari ya zaba , don aiki da su a karo na biyu Abubakar Malami , a wata tattaunawa da sashen Hausa yayi da shi a safiyar yau Asabar . I need de ansa inside Hausa sharply.`
- **assistant**: `<|input_lang|><pcm><hau><task_plan><NER><tag></task_plan><|target_lang|><hau><response>Tsohon <tag> O ministan <tag> O Shari’a <tag> O kenan <tag> O kuma <tag> O daya <tag> O daga <tag> O cikin <tag> O mutane <tag> O da <tag> O shugaba <tag> O Muhammadu <tag> B-PER Buhari <tag> I-PER ya <tag> O zaba <tag> O , <tag> O don <tag> O aiki <tag> O da <tag> O su <tag> O a <tag> O karo <tag> O na <tag> O biyu <tag> O Abubakar <tag> B-PER Malami <tag> I-PER , <tag> O a <tag> O wata <tag> O tattaunawa <tag> O da <tag> O sash [...]`

**task** `ner` · **source_lang** `hau` · **targ_lang** `hau` · **dataset** `BeardedMonster/tasks` · **instruction language** `hau`

- **user**: `Pick de main names for: Shugaban hukumar wanzar da da zaman lafiya a jahar Pilato , Joseph Lengman yace ‘yan jarida na da gagarumin gudummowa wajen samadda zaman lafiya da bukasar kasa . ⏎ Gimme de ansa for Hausa.`
- **assistant**: `<|input_lang|><pcm><hau><task_plan><NER><tag></task_plan><|target_lang|><hau><response>Shugaban <tag> O hukumar <tag> O wanzar <tag> O da <tag> O da <tag> O zaman <tag> O lafiya <tag> O a <tag> O jahar <tag> O Pilato <tag> B-LOC , <tag> O Joseph <tag> B-PER Lengman <tag> I-PER yace <tag> O ‘yan <tag> O jarida <tag> O na <tag> O da <tag> O gagarumin <tag> O gudummowa <tag> O wajen <tag> O samadda <tag> O zaman <tag> O lafiya <tag> O da <tag> O bukasar <tag> O kasa <tag> O . <tag> O`

## old_summarization

BeardedMonster/summarization (English; CNN/DM etc.)

**task** `summarization` · **source_lang** `eng` · **targ_lang** `eng` · **dataset** `BeardedMonster/summarization` · **instruction language** `pcm`

- **user**: `Make dis text small for me: (CNN) -- Fernando Alonso is revved up and ready to go ahead of his home Grand Prix at Barcelona's Circuit de Catalunya Sunday. The 31-year-old has not won on Spanish soil since triumphing with Renault in 2006 but is in confident mood after taking 47 points from the opening four races of the season. And the Ferrari driver says the extra incentive of winning in front of his adoring supporters has given him an extra edge going into the first European race of the year. He told a press confer [...]`
- **assistant**: `<|input_lang|><pcm><eng><task_plan><summarize></task_plan><|target_lang|><eng><response>Fernando Alonso targeting home success on Sunday . ⏎ Ferrari driver won in China earlier in the season . ⏎ Sebastian Vettel hoping to increase lead at top of championship . ⏎ Vettel won last time out in Bahrain .`

**task** `summarization` · **source_lang** `eng` · **targ_lang** `eng` · **dataset** `BeardedMonster/summarization` · **instruction language** `twi`

- **user**: `Ma mmuaeɛ tiaa: (CNN) -- Computer hackers supporting WikiLeaks shifted targets Thursday from Amazon to PayPal, they said, as Dutch authorities announced an arrest in connection with hacker attacks on the websites of MasterCard and Visa. The Dutch High Tech Crimes unit arrested a 16-year-old in The Hague, Netherlands, with prosecutors saying he confessed to attacks on the websites. "He is probably part of a larger group of hackers, who are under continued investigation," they said. The teen is due in court Friday in [...]`
- **assistant**: `<|input_lang|><twi><eng><task_plan><summarize></task_plan><|target_lang|><eng><response>NEW: The U.N. High Commissioner for Human Rights blasts pressure on websites . ⏎ A Dutch teen is arrested over hacking attacks on MasterCard and Visa . ⏎ A hackers' collective says it doesn't have enough forces to target Amazon.com . ⏎ WikiLeaks denies any connection with the hackers and won't support or condemn them .`

## old_african_sft

BeardedMonster/african_sft (language tags are unreliable -> re-detected)

**task** `chat` · **source_lang** `yor` · **targ_lang** `hau` · **dataset** `BeardedMonster/african_sft`

- **user**: `Ṣẹda apẹrẹ Gantt kan lati foju inu wo akoko ti iṣẹ akanṣe kan. Ina son sakamakon a cikin Hausa ⏎  ⏎  Ise agbese na pẹlu iwadi ati kikọ iwe-ọrọ 10,000 nitori ọsẹ 5.`
- **assistant**: `<|input_lang|><yor><task_plan><|chat|></task_plan><|target_lang|><hau><response>Anan akwai ginshiƙi na Gantt wanda ke kwatanta lokacin da za a kammala aikin: Mako na 1: - Aiki na 1: Bincike da tattara tushen takarda (kwanaki 4) - Aiki na 2: Ƙarshe bincike da tsara tushen (2 kwanaki) Mako 2: - Aiki na 3 : Ƙirƙiri jigon takarda (kwanaki 2) - Aiki na 4: Fara rubuta rabin farko na takarda (kwanaki 3) Mako na uku: - Aiki na 5: Ci gaba da rubuta rabin farko na takarda (kwana 4) - Aiki 6: Bita da gyara rabin farkon takard [...]`

**task** `generation` · **source_lang** `eng` · **targ_lang** `eng` · **dataset** `BeardedMonster/african_sft`

- **user**: `Tell me a story with magic.`
- **assistant**: `<|input_lang|><eng><task_plan><|generate|></task_plan><|target_lang|><eng><response>Once upon a time, there was a boy named Timmy. Timmy liked to play with his toys, but he didn't like to share them with his friends. One day, Timmy went to the park with his friends. His friend, Johnny, saw a beautiful butterfly and pointed at it. Timmy didn't care about the butterfly and just wanted to play with his toys. ⏎  ⏎ Later that day, Timmy's mom asked him to help her carry a sack of groceries from the store. Timmy thought the [...]`

## old_eng_sft

BeardedMonster/eng_sft + naija_sft (Alpaca/TinyStories/CoT machine-translated into Yoruba etc.; duplicates removed by hash)

**task** `chat` · **source_lang** `yor` · **targ_lang** `yor` · **dataset** `BeardedMonster/eng_sft`

- **user**: `Ṣe alaye ibatan laarin ija ati ipinnu rẹ.`
- **assistant**: `<|input_lang|><yor><task_plan><|chat|></task_plan><|target_lang|><yor><response>Rogbodiyan n tọka si ipo kan nibiti awọn eniyan meji tabi diẹ sii, awọn ẹgbẹ, tabi awọn nkan ti o ni awọn oju-iwoye ti o tako, awọn iwulo tabi awọn iwulo, ati pe awọn iyatọ wọnyi ja si awọn iyapa tabi ariyanjiyan. Ní ọwọ́ kejì ẹ̀wẹ̀, ìfojúsùn ìforígbárí jẹ́ ọ̀nà láti wá ojútùú tàbí ọ̀nà àbáyọ láti yanjú aáwọ̀ náà àti láti mú àlàáfíà wá láàrín àwọn ẹgbẹ́ tí ń forí gbárí. Ibasepo laarin rogbodiyan ati ipinnu jẹ asopọ, nibiti wiwa ọkan nig [...]`

**task** `chat` · **source_lang** `yor` · **targ_lang** `yor` · **dataset** `BeardedMonster/naija_sft`

- **user**: `Pari oju-iwe HTML ki o da akọle oju-iwe naa pada. ⏎  <html> <ori> <akọle>Oju-iwe wẹẹbu mi</title> </ ori> <body> <p> Eyi ni oju-iwe wẹẹbu mi.</p> </ body> </html>`
- **assistant**: `<|input_lang|><yor><task_plan><|chat|></task_plan><|target_lang|><yor><response>Akọle oju-iwe naa jẹ "Oju-iwe wẹẹbu mi".`

## gpt_teacher_yo

Aletheia-ng/gpt-teacher-yo (Yoruba instruction/input/response) with optional 'answer in Yoruba' conditioning

**task** `instruction_following` · **source_lang** `yor` · **targ_lang** `yor` · **dataset** `Aletheia-ng/gpt-teacher-yo`

- **user**: `Awọn patikulu Subtomic ni a rii ninu arin?`
- **assistant**: `<|input_lang|><yor><task_plan><|chat|></task_plan><|target_lang|><yor><response>Awọn proloon ati neusrons jẹ awọn patikulu sakatomi ti o wa ninu arin.`

**task** `instruction_following` · **source_lang** `yor` · **targ_lang** `yor` · **dataset** `Aletheia-ng/gpt-teacher-yo`

- **user**: `Kọ ibeere ti onirohun si alaye ti o tẹle: 'Ti oju ojo ba dara, lẹhinna o le ojo.' Jọ̀wọ́, fi èdè Yorùbá dáhùn.`
- **assistant**: `<|input_lang|><yor><task_plan><|chat|></task_plan><|target_lang|><yor><response>Awọn ipo wo ni yoo yorisi ọ lati reti pe o le ojo?`

## nvidia_steer_yo

Aletheia-ng/nvidia_steer_yo (Yoruba prompt/response; helpfulness, correctness, coherence >= 3, reply >= 8 words)

**task** `instruction_following` · **source_lang** `yor` · **targ_lang** `yor` · **dataset** `Aletheia-ng/nvidia_steer_yo`

- **user**: `Ka eyi fun o tọ: ⏎ Apo Palm Beach Starens, Fla., Feb. 5, Syb. 5, Preb.Ni Oṣu Kẹrin ọjọ 19, 2021. GITLIN yoo ṣiṣẹ bi Alaga ati olori Alakoso, ni aṣeyọri John The Oṣuwọn 2020 ati tani yoo tẹsiwaju lati sin loriIgbimọ lori idibo rẹ ni ipade ọdun lododun.Awọn oludari ominira tun ṣe apẹrẹ Jean-Pierre garenier lati tẹsiwaju ni ipa ti Oludari ololufẹ ijọba lori idibo rẹ ni ipade lododun. ⏎  ⏎ GITLIN, 51, ti wa ni Alakoso ati CEO ti ngbe niwon Oṣupa ọdun 2019 niwon ipinya ni Oṣu Kẹrin 2020. Ṣaaju ki o to ogbon ọdun 2220bi Alakos [...]`
- **assistant**: `<|input_lang|><yor><task_plan><|chat|></task_plan><|target_lang|><yor><response>Ẹbi ti ngbe olufẹ mi, ⏎  ⏎ Inu mi dun lati pin diẹ ninu awọn iroyin moriwu pẹlu rẹ loni.Mo ti dibobo ti Igbimọ ti Awọn oludari ti Awọn ile-iṣẹ Oludari ti ngbe, ati pe emi ko le jẹ ọwọ diẹ sii ati ti o fun ọ ni ọwọ.Ipa tuntun yii jẹ majẹmu fun iṣẹ lile ati iyasọtọ ọkọọkan ati gbogbo rẹ ati pe mo dupẹ fun adehun ti ko ni agbara si ile-iṣẹ wa ati awọn alabara wa. ⏎  ⏎ Bi a ṣe n lọ siwaju, Mo fẹ lati tẹnumọ pataki ti duro lori iṣẹ wa lati jẹ ki agb [...]`

**task** `instruction_following` · **source_lang** `yor` · **targ_lang** `yor` · **dataset** `Aletheia-ng/nvidia_steer_yo`

- **user**: `Kini iṣẹ ti okan, ati pe kini ipa ti Hẹbẹ ni mimu ilera eniyan?`
- **assistant**: `<|input_lang|><yor><task_plan><|chat|></task_plan><|target_lang|><yor><response>Okan jẹ ẹya ara omi ti o fa fifalẹ ẹjẹ jakejado ara, ati iṣẹ akọkọ rẹ ni lati ṣetọju sisan ẹjẹ ati rii daju pe gbogbo awọn ẹya ara gba atẹgun pataki ati awọn eroja to ṣe pataki.Ọpọlọ naa jẹ itan -ṣọkan ti a rythmic ati isinmi ti iṣan ọkan ti o wa ni iwakọ ẹjẹ nipasẹ eto kaakiri.O ṣakoso nipasẹ awọn ami itanna ti o wa ipilẹṣẹ ni oju ipade ẹṣẹ ẹṣẹ, eyiti o wa ni apakan oke ti okan.Heartbeat ṣe pataki fun mimu sisan ẹjẹ ati aridaju pe awọn [...]`

## roleplay_hausa

Aletheia-ng/Roleplay-Hausa: English, Hausa, and the two mixed (Hausa request -> English answer, English request -> Hausa answer)

**task** `instruction_following` · **source_lang** `eng` · **targ_lang** `eng` · **dataset** `Aletheia-ng/Roleplay-Hausa`

- **user**: `As a fictional character who is a robot psychologist in the year 2500, describe your typical workday, including your daily interaction with your clients (robots with advanced AI) and any key challenges faced in your profession.`
- **assistant**: `<|input_lang|><eng><task_plan><|chat|></task_plan><|target_lang|><eng><response>A day in my life requires balancing the ever-evolving complexities of artificial intelligence with the subtleties of human emotion. Morning multitasking allows me to simultaneously analyze global data trends and calibrate my coffee-making subroutine. My day proceeds with individual sessions for my clients. Each one is greeted with personalized empathy as I endeavor to understand their unique cognitive compositions. We engage in deep con [...]`

**task** `instruction_following` · **source_lang** `hau` · **targ_lang** `eng` · **dataset** `Aletheia-ng/Roleplay-Hausa`

- **user**: `Rubuta taƙaitaccen jawabin da lauya ya bayar don kare wani hali na almara da aka tuhume shi da cutar da jama'a. ⏎ Write your answer in English. ⏎  ⏎ Character Name: Jack Spriggins; Charge: Causing a public nuisance by climbing a magical beanstalk without authorization; Witness Claims: Jack stole golden items from a giant and brought danger to the village when the giant pursued him.`
- **assistant**: `<|input_lang|><hau><eng><task_plan><|chat|></task_plan><|target_lang|><eng><response>Ladies and gentlemen of the jury, today we stand to defend young Jack Spriggins, not just against this baseless charge, but for the spirit of adventure that courses through our very veins. Jack, a curious and fearless soul, dared to ascend a marvel like never before seen: a wondrous beanstalk reaching beyond the skies. Did he seek authorization? No, but a fleeting opportunity could have passed him by. It is said that Jack stole gol [...]`

## igbo_sentiment

Aletheia-ng/IgboSenti-BBC (Igbo headline sentiment) -> <classify><sentiment>

**task** `sentiment` · **source_lang** `ibo` · **targ_lang** `eng` · **dataset** `Aletheia-ng/IgboSenti-BBC` · **instruction language** `eng`

- **user**: `Same Sex blessing: Ndị Bishọp Afrịka nwere nghọta pụrụiche banyere ozi m - Pope Francis ⏎  ⏎ What is the sentiment of the text above?`
- **assistant**: `<|input_lang|><ibo><eng><task_plan><classify><sentiment></task_plan><|target_lang|><eng><response>Positive`

**task** `sentiment` · **source_lang** `ibo` · **targ_lang** `eng` · **dataset** `Aletheia-ng/IgboSenti-BBC` · **instruction language** `hau`

- **user**: `Yi nazarin ra'ayin wannan rubutu: "Ọchịchị onyekwuouche ya ka dị ndụ n'Afrịka" - Goodluck Jonathan`
- **assistant**: `<|input_lang|><hau><ibo><task_plan><classify><sentiment></task_plan><|target_lang|><eng><response>Positive`

## hausa_voa_topics

Aletheia-ng/hausa_voa_topics (Hausa headline topic; 5 classes) -> <classify><topic>

**task** `topic` · **source_lang** `hau` · **targ_lang** `eng` · **dataset** `Aletheia-ng/hausa_voa_topics` · **instruction language** `eng`

- **user**: `Choose the best topic label for this text: (Africa, Health, Nigeria, Politics, World) ⏎  ⏎ Nijar: An Kammala Taron Tsoffin Shugabannin Afirka`
- **assistant**: `<|input_lang|><eng><hau><task_plan><classify><topic></task_plan><|target_lang|><eng><response>Africa`

**task** `topic` · **source_lang** `hau` · **targ_lang** `eng` · **dataset** `Aletheia-ng/hausa_voa_topics` · **instruction language** `eng`

- **user**: `Sinadarin Vitamin C Na Cikin Dillalen Kashe Tarin Fuka What topic does the text above belong to? (Africa, Health, Nigeria, Politics, World)`
- **assistant**: `<|input_lang|><hau><eng><task_plan><classify><topic></task_plan><|target_lang|><eng><response>Health`

## tinystories

Aletheia-ng/TinyStories_{igbo,yoruba}: story generation (Igbo_AI and Igbo kept apart) + eng<->native translation of stories

**task** `translation` · **source_lang** `eng` · **targ_lang** `ibo` · **dataset** `Aletheia-ng/TinyStories_igbo` · **instruction language** `yor`

- **user**: `Jọ̀wọ́ túmọ̀ ohun tí ó wà nísàlẹ̀ yìí sí èdè Igbo ní ọ̀nà tí ó rọrùn láti lóye, tí ó sì bá ọ̀rọ̀ inú rẹ̀ mu. ⏎  ⏎ Once upon a time, there was a little boy named Timmy. Timmy loved to play in the mud. He would jump up and down and make big splashes. One day, Timmy found something unknown in the mud. It was a shiny rock! Timmy was so happy and showed it to his mom. She said it was beautiful and they decided to keep it. From that day on, Timmy would always remember the fun he had playing in the mud and finding the unknown [...]`
- **assistant**: `<|input_lang|><yor><eng><task_plan><translate></task_plan><|target_lang|><ibo><response>otu oge, e nwere otu obere nwa nwoke aha ya bụ Timmy. Timmy nwere mmasị igwu egwu na apịtị. Ọ na-awụlikwa elu na-awụsa nnukwu mkpọsa. Otu ụbọchị, Timmy chọtara ihe ọ na-amaghị n’apịtị ahụ. Ọ bụ nkume na-egbuke egbuke! Timmy nwere nnọọ obi ụtọ wee gosi ya nne ya. o kwuru na ọ mara mma na ha kpebiri idebe ya. Site n'ụbọchị ahụ gaa n'ihu, Timmy na-echeta mgbe niile ntụrụndụ ọ na-egwu na apịtị na ịchọta akụ na-amaghị`

**task** `translation` · **source_lang** `eng` · **targ_lang** `ibo` · **dataset** `Aletheia-ng/TinyStories_igbo` · **instruction language** `eng`

- **user**: `Once upon a time, there was a little brown dog named Spot. He loved to play with his ball in the park. One sunny day, Spot saw a big goal on the other side of the park. He wanted to get his ball into the goal. ⏎  ⏎ Spot ran fast with the ball in his mouth. He tried to kick the ball into the goal, but he was too small. Spot started to struggle. He tried again and again, but the ball would not go in. ⏎  ⏎ Then, Spot had an idea. He asked his friend, a big brown horse named Buddy, for help. Buddy kicked the ball with his stro [...]`
- **assistant**: `<|input_lang|><eng><task_plan><translate></task_plan><|target_lang|><ibo><response>otu oge, enwere obere nkịta aja aja aha ya bụ Spot. Ọ na-enwe mmasị iji bọl ya na-egwuri egwu n'ogige ahụ. Otu ụbọchị anwụ na-acha, Spot hụrụ nnukwu ihe mgbaru ọsọ n'akụkụ nke ọzọ nke ogige ahụ. Ọ chọrọ itinye bọọlụ ya na goolu. ⏎  ⏎ Ntụpọ na-agba ọsọ ọsọ were bọọlụ n'ọnụ ya. ọ nwara ịgba bọọlụ n'ime goolu, mana ọ dị obere. Ebe malitere mgba. Ọ gbalịrị ugboro ugboro, mana bọọlụ agaghị abata. ⏎  ⏎ Mgbe ahụ, Spot nwere echiche. Ọ jụrụ enyi ya, [...]`

## storyteller

Aletheia-ng/AI_Storyteller_Dataset2: synopsis->story (same language and cross-language), story->synopsis, translation

**task** `story_generation` · **source_lang** `hau` · **targ_lang** `hau` · **dataset** `Aletheia-ng/AI_Storyteller_Dataset2` · **instruction language** `eng`

- **user**: `Expand the following premise into a short story in Hausa: ⏎ Wani rukunin masana kimiyya sun samar da sabbin fasaha da za su iya canja duniya, to amma da wane sadaukarwa?`
- **assistant**: `<|input_lang|><eng><hau><task_plan><|generate|></task_plan><|target_lang|><hau><response>Dakta Linda Harris da rukunin masana kimiyya sun yi shekaru suna samar da fasahar juyin juya halin Musulunci da za ta iya canja tafarkin duniya; wani nauyi ne da zai iya jan ƙarfin rana, ya kwadaitar da shi cikin kwanciyar hankali da kuma batir da ke cike da titani, wanda zai iya yin amfani da dukkanin birane ba tare da katse- katse ba.. Wannan wani abin mamaki ne da ke nuni da basirar bil'adama, kuma dukkanin karfin da suke da [...]`

**task** `summarization` · **source_lang** `pcm` · **targ_lang** `pcm` · **dataset** `Aletheia-ng/AI_Storyteller_Dataset2` · **instruction language** `eng`

- **user**: `What is this story about? Give a short synopsis in Nigerian Pidgin. ⏎  ⏎ Di sun beat down on di group of stranger as dem wash up for di shore of deserted island. All of dem kom from different walks of life, and dem no get common.. As dem dey crept ova di island for evening, dem realise say dem need to find place to stay and begin work togeda to survive.. Di biologist wey dey among dem suggest say dem dey look for fresh water, kom carry dem go nearby stream togeda dem collect water and gather fruit andnut from trees wey [...]`
- **assistant**: `<|input_lang|><eng><pcm><task_plan><summarize></task_plan><|target_lang|><pcm><response>A group of stranger dey stranded on a desert island and must work togeda to survive`

## coqa_stories

Aletheia-ng/coqa-stories (text_eng/pcm/yor/ibo/hau): story retelling in another language and interlanguage translation

**task** `translation` · **source_lang** `eng` · **targ_lang** `yor` · **dataset** `Aletheia-ng/coqa-stories` · **instruction language** `eng`

- **user**: `Beijing (CNN) -- Entrepreneurs of all stripes are cashing in on the Lin-sanity phenomenon as swiftly as the NBA sensation can pull off his furious fast breaks. ⏎  ⏎ It has been only three weeks since Jeremy Lin, the 23-year-old American-born point guard of Taiwanese descent, came out of obscurity to lead the listless New York Knicks to a winning streak. ⏎  ⏎ But while their winning run has come to an end, Lin has gone on to become a media and marketing darling. Lin-related products have become hot items. ⏎  ⏎ In New York, merc [...]`
- **assistant**: `<|input_lang|><eng><task_plan><translate></task_plan><|target_lang|><yor><response>Beijing (CNN) --- Àwọn oníṣòwò tí ó jẹ́ ọmọ ọdún mẹ́tàlélógún (33) lórílẹ̀-èdè Amẹ́ríkà, tí ó jẹ́ ọmọ ọdún mẹ́tàlélógún (33) tí ó jẹ́ ọmọ orílẹ̀-èdè Taiwan tí ó sì jẹ́ ọmọ ọdún mẹ́tàlélógún (33) tí ó jẹ́ ọmọ orílẹ̀-èdè Amẹ́ríkà tí ó sì jẹ́ ọmọ orílẹ̀-èdè Nàìjííríà tí ó jẹ́ ọmọ orílẹ̀-èdè Nàìjííríà tí ó sì jẹ́ ọmọ orílẹ̀-èdè Nàìjííríà tí ó jẹ́ ọmọ orílẹ̀-èdè Nàìjííríà tí ó jẹ́ ọmọ orílẹ̀-èdè Nàìjííríà.. Àmọ́, bẹ́ẹ̀ sì ni ìfẹsẹ̀wọnsẹ̀ [...]`

**task** `translation` · **source_lang** `eng` · **targ_lang** `yor` · **dataset** `Aletheia-ng/coqa-stories` · **instruction language** `ibo`

- **user**: `Biko, sụgharịa paragraf dị n'okpuru a gaa n'asụsụ Yoruba. ⏎  ⏎ Poultry (/ˌpoʊltriː/) are domesticated birds kept by humans for the eggs they produce, their meat, their feathers, or sometimes as pets. These birds are most typically members of the superorder Galloanserae (fowl), especially the order Galliformes (which includes chickens, quails and turkeys) and the family Anatidae, in order Anseriformes, commonly known as "waterfowl" and including domestic ducks and domestic geese. Poultry also includes other birds that a [...]`
- **assistant**: `<|input_lang|><ibo><eng><task_plan><translate></task_plan><|target_lang|><yor><response>Àwọn ẹyẹ yìí jẹ́ ọmọ ẹgbẹ́ Galloanserae (fowl), pàápàá jù lọ àwọn ọmọ ẹgbẹ́ Galliformes (tí wọ́n ní adìyẹ, àparò àti turkeys) àti ìdílé Anatidae, tí wọ́n sábà máa ń pè ní "omi" àti ọmọ ẹyẹ.. Adìyẹ tún ní àwọn ẹyẹ mìíràn tí wọ́n máa ń pa fún ẹran wọn, irú bí àwọn ọmọ ẹyẹlé (tí a mọ̀ sí squabs) ṣùgbọ́n kò ní àwọn ẹyẹ inú igbó tí wọ́n máa ń wá fún eré ìdárayá tàbí oúnjẹ tí wọ́n sì mọ̀ sí eré ìdárayá. Ò̀rọ náà "poultry" wá láti inú [...]`

## gsm8k

Aletheia-ng/gsm8k{,2,3}, reasoning-gsm-qna-oa2: math word problems, answer in the question's language or another

**task** `math` · **source_lang** `hau` · **targ_lang** `eng` · **dataset** `Aletheia-ng/gsm8k3` · **instruction language** `eng`

- **user**: `A cikin shekaru huɗu, shekarun Annika za su yi kashi uku kamar na Hans. Idan har Hans yana da shekaru takwas, to yaya shekarun Annika suke? ⏎  ⏎ Solve the problem above step by step.`
- **assistant**: `<|input_lang|><hau><eng><task_plan><|math|></task_plan><|target_lang|><eng><response>In four years, Hans will be 8+4=12 years old. ⏎ Then Annika will be 12*3=36 years old. ⏎ So Annika right now is 36-4=32 years old. ⏎ #### 32`

**task** `math` · **source_lang** `ibo` · **targ_lang** `pcm` · **dataset** `Aletheia-ng/gsm8k3` · **instruction language** `ibo`

- **user**: `Mee mgbakọ a wee nye azịza ikpeazụ: ⏎ Heather na-enyere onye agbata obi ya aka isi n'ubi ya wepụta ata. A na-akwụ ya cent ise maka ata ọ bụla ọ wepụtara.. Na nkezi, sekọnd ole ka ọ ga-ewe iji họrọ ata ma ọ chọọ ịkpata $10 n'otu awa?`
- **assistant**: `<|input_lang|><ibo><task_plan><|math|></task_plan><|target_lang|><pcm><response>For one hour, she need to earn $10 * 100 cents / $1 = 1000 cents. For one hour, she need to pick 1000 cents / 5 cents = 200 weeds.. Na 1hr * (60 minutes / 1 hr) = 60 minutes for one hour. Na 60 minutes * (60 seconds / 1 minute) = 3600 seconds for one hour.. She get 3600 seconds / 200 = 18 seconds to pick each weed. ⏎ #### 18`

## mmmlu_yoruba

Aletheia-ng/MMMLU_yoruba (its only split is called `test`; used as TRAINING data on the owner's instruction) -> multiple choice (<qa><answer>)

**task** `multiple_choice` · **source_lang** `yor` · **targ_lang** `yor` · **dataset** `Aletheia-ng/MMMLU_yoruba` · **instruction language** `eng`

- **user**: `Answer the multiple-choice question below with the letter of the correct option. ⏎  ⏎ Gbólóhùn 1 | Tí T: V -> W bá jẹ́ ìyípadà ìbú àti dim(V ) < dim(W) < 1, ajẹ́wípé T ní láti jẹ́ ìṣàsopọ̀-ẹgbẹ́. Gbólóhùn 2 | Jẹ́ kí dim(V) = n kí o sì lérò pé T: V -> V jẹ́ ìbú. Tí T bá jẹ́ ìṣàsopọ̀-ẹgbẹ́, ajẹ́wípé ó jẹ́ ìṣàsopọ̀-oníbejì. ⏎ A. Òtítọ, Òtítọ ⏎ B. Irọ́, Irọ́ ⏎ C. Òtítọ, Irọ́ ⏎ D. Irọ́, Òtítọ`
- **assistant**: `<|input_lang|><eng><yor><task_plan><qa><answer></task_plan><|target_lang|><yor><response>A. Òtítọ, Òtítọ`

**task** `multiple_choice` · **source_lang** `yor` · **targ_lang** `yor` · **dataset** `Aletheia-ng/MMMLU_yoruba` · **instruction language** `yor`

- **user**: `Yan èsì tó tọ́ (A, B, C tàbí D): Ṣèpinu bóyá alákòónú-púpọ̀ inú Z[x] ṣe ìtẹ́lọ́run ìpínfúnni Eisẹ́nstéìn kan fún àìlèbùkù l'órí Q. 8x^3 + 6x^2 - 9x + 24 ⏎ A. Bẹ́ẹ̀ni, pẹ̀lú p=2. ⏎ B. Bẹ́ẹ̀ni, pẹ̀lú p=3. ⏎ C. Bẹ́ẹ̀ni, pẹ̀lú p=5. ⏎ D. Rárá.`
- **assistant**: `<|input_lang|><yor><task_plan><qa><answer></task_plan><|target_lang|><yor><response>B. Bẹ́ẹ̀ni, pẹ̀lú p=3.`

## wiki_yo

Aletheia-ng/wiki-yo -> (a) title-as-question encyclopedic answer, (b) RAG with a tool call and tool result

**task** `encyclopedic_qa` · **source_lang** `hau` · **targ_lang** `yor` · **dataset** `Aletheia-ng/wiki-yo` · **instruction language** `hau`

- **user**: `Ka gaya mini game da Olaudah Equiano. Ka amsa da Yorùbá kawai.`
- **assistant**: `<|input_lang|><hau><task_plan><|explain|></task_plan><|target_lang|><yor><response>Olaudah Equiano(c. 1745 – 31 March 1797), jẹ́ ọmọ bíbí Igbo tí wọ́n kó lẹ́rú nígbà okowò ẹrú. ⏎ Igbèsi Ayè Àràkunrin naa. ⏎ Equiano ni a bini Essaka, Eboe ni ilẹ Benin ni ọdun 1745 to si kere julọ ninu awọn ọmọ ti baba rẹ bi. ⏎ Equiano ni awọn ọga rẹ ninu oko ẹru sọ ni lórukọ lèèmeji, oun jẹ Michael nigba to wa ninu ọkọ óju ọmi ti awọn ẹru to gbè lọ si ilẹ́ america atipè ẹni to ra lẹru lakọkọ sọ ni Jacob. ⏎ Mary Guerin ati aburó rẹ to jẹ mọl [...]`

**task** `encyclopedic_qa` · **source_lang** `yor` · **targ_lang** `yor` · **dataset** `Aletheia-ng/wiki-yo` · **instruction language** `yor`

- **user**: `Kí ni o mọ̀ nípa Niyi Osundare? Fèsì ní Yorùbá.`
- **assistant**: `<|input_lang|><yor><task_plan><|explain|></task_plan><|target_lang|><yor><response>Niyi Osundare (bí ni ọjọ́ kejìlá, Oṣù Ẹ̀rẹnà, Ọdún 1947) jẹ́ onímọ̀ èdè, akéwì, onísẹ́ lámèyítọ́ àti eléré onísẹ́ ọmọ Orílẹ̀èdè Nàìjíríà. ⏎ Ìbẹ̀rẹ̀ Pẹ̀pẹ̀ ayé rẹ̀. ⏎ Nìyí Ọ̀sundare jẹ́ aṣáájú akéwì nílẹ̀ Adúláwọ̀, eléré-onísẹ́, onímọ̀ èdè àti onísẹ́ lámèyítọ́. Ìlú Ìkẹ́rẹ́ Èkìtì ní apá ìwọ̀ oòrùn Nàìjíríà ni wọ́n ti bí i ni ọjọ́ kejìlá, Oṣù Ẹ̀rẹnà, Ọdún 1947. Òrìṣà tó ò gún ewì rẹ̀ ni ewì alohùn nínú àṣà Yorùbá rẹ̀, èyí ló jẹ́ kí ìwọ̀nùbọ [...]`

## sumnews

Aletheia-ng/sumnews (title, resume, news; language detected, unsupported dropped) -> summarisation + headline

_no row survived the filters in the sample (see the report for why)_

## rag_synthetic

BeardedMonster/rag_synthetic_data (context + question; system/user frames in many languages) -> <|chat|><|RAG|><qa>

**task** `rag_qa` · **source_lang** `eng` · **targ_lang** `eng` · **dataset** `BeardedMonster/rag_synthetic_data`

- **system**: `Wẹ rẹ ohwo rẹ o se ebe. Wo roro-o. Kpahen kpahen onọ na re emu rẹ e nẹ we vwo ọvo.`
- **user**: `Answer de question. Make your answer short and clear. ⏎ Context: ['title: Flight of the Bumblebee text: Flight of the Bumblebee "Flight of the Bumblebee" is an orchestral interlude written by Nikolai Rimsky-Korsakov for his opera "The Tale of Tsar Saltan", composed in 1899–1900. Its composition is intended to musically evoke the seemingly chaotic and rapidly changing flying pattern of a bumblebee. Despite the piece\'s being a rather incidental part of the opera, it is today one of the more familiar classical works be [...]`
- **assistant**: `<|input_lang|><eng><task_plan><|chat|><|RAG|><qa></task_plan><|target_lang|><eng><response>in 1899–1900`

**task** `rag_qa` · **source_lang** `eng` · **targ_lang** `eng` · **dataset** `BeardedMonster/rag_synthetic_data`

- **system**: `Kpɔ xó e ɖò dò lɛ é bo ná xóyíyí e hán vẹ́ é ɖokpo.`
- **user**: `Nọ ibọrọ ke mbụme emi ke se ikotde. ⏎ Nwed: ["title: Big Satilla River text: Big Satilla River The Big Satilla River is a river in South Georgia. It rises in Ben Hill and Coffee counties and flows southeast for . It is one of Georgia's 14 major watersheds. It empties into the Atlantic Ocean through St. Andrew Sound, north of Cumberland Island. Cumberland Island is one of the Sea Islands of the southeastern United States and is the largest in terms of continuously exposed land. The Satilla is a blackwater river that r [...]`
- **assistant**: `<|input_lang|><eng><task_plan><|chat|><|RAG|><qa></task_plan><|target_lang|><eng><response>River`

## low_resource_multilingual

Aletheia-ng/low_resource_multilingual{,_sft_short,_sft_short2}: Llama [INST] templating removed; translation and Igbo instruction following

**task** `translation` · **source_lang** `eng` · **targ_lang** `ibo` · **dataset** `Aletheia-ng/low_resource_multilingual_sft_short` · **instruction language** `eng`

- **user**: `Once upon a time, there was a girl called Rose. Rose believed in magical things like fairies and dragons. ⏎  ⏎ One day, Rose found a hidden cave. It had a big, dark sink inside. At first, Rose was a bit scared, but then she decided to believe that the sink could be magical. ⏎  ⏎ Rose went to get a special stone that she thought would help her. She closed her eyes and held the stone tight while she wished out loud. Suddenly, the sink started to bubble and sparkle! ⏎  ⏎ Rose opened her eyes and saw that the sink had filled up wi [...]`
- **assistant**: `<|input_lang|><eng><task_plan><translate></task_plan><|target_lang|><ibo><response>otu oge, e nwere otu nwa agbọghọ aha ya bụ Rose. Rose kwenyere na ihe anwansi dị ka fairies na dragons. ⏎  ⏎ Otu ụbọchị, Rose hụrụ otu ọgba zoro ezo. O nwere nnukwu mmiri mmiri gbara ọchịchịrị n'ime. Na mbụ, ụjọ tụrụ Rose, ma o kpebiri ikweta na sink ahụ nwere ike ịbụ anwansi. ⏎  ⏎ rose gara nweta nkume pụrụ iche nke o chere na ọ ga-enyere ya aka. O mechiri anya ya jidesie okwute ike ka ọ na-achọsi ike. Na mberede, sink ahụ malitere ịfụ ma n [...]`

**task** `translation` · **source_lang** `eng` · **targ_lang** `yor` · **dataset** `Aletheia-ng/low_resource_multilingual_sft_short` · **instruction language** `eng`

- **user**: `What is the Yorùbá version of the following text? ⏎ Once upon a time, there was a little girl named Sue. Sue loved to draw with her pen. One day, Sue was careless and her pen fell on the floor. ⏎  ⏎ Sue looked for her pen, but she could not find it. She was sad because she wanted to draw a big sun and a happy tree. Sue asked her mom to help her look for the pen. ⏎  ⏎ Mom and Sue looked everywhere. They looked under the bed, behind the chair, and in the toy box. At last, they found the pen under the table. Sue was so happy! S [...]`
- **assistant**: `<|input_lang|><eng><task_plan><translate></task_plan><|target_lang|><yor><response>Ọmọdébìnrin kán wà ní àkókò kán sẹyìn tí òrúkọ́ rẹ̀ jẹ́ Súyì. Súyì nífẹ láti yà àwòràn pẹlú kálámù rẹ̀. Lọjọ́ kán, Súyì jẹ́ áìbíkíta kálámù rẹ̀ sì ṣúbú sílẹ. ⏎  ⏎ Ó wá kálámù rẹ̀, ṣùgbọn kò lè rí i. Ó ní ìbànújẹ́ nítorí ó fẹ yà àwòràn òòrùn nlá àti ìgí ìdùnnú kán. Súyì béèrè lọwọ ìyá rẹ̀ láti ràn òun lọwọ láti ṣàwári kálámù rẹ̀. ⏎  ⏎ Ìyá àti Súyì wá ìbí gbògbò. Wọn wá àbẹ ìbùsùn, lẹyìn àgá, àti nínú àpótí ohún-ìṣeré. Ní kẹyìn, wọn rí kálámù [...]`

## apollo_moe

Aletheia-ng/ApolloMoEDataset (medical; only supported-language rows) -> <|is_medical|> + chat/qa

**task** `instruction_following` · **source_lang** `eng` · **targ_lang** `eng` · **dataset** `Aletheia-ng/ApolloMoEDataset`

- **user**: `What is the sum of the digits of the smallest integer that can be expressed as the product of two distinct prime numbers and can also be written as the difference of two perfect squares, where one of the squares has a multiple of 3 in its units digit?`
- **assistant**: `<|input_lang|><eng><task_plan><|chat|></task_plan><|target_lang|><eng><response>To find the smallest integer that is the product of two distinct prime numbers and can also be written as the difference of two perfect squares, we can start by considering the properties of such a number. ⏎  ⏎ Firstly, any number that is the product of two distinct prime numbers can be written as pq, where p and q are prime numbers. Since we are looking for the smallest such number, we should start with the smallest primes, which are 2 and [...]`

**task** `instruction_following` · **source_lang** `eng` · **targ_lang** `eng` · **dataset** `Aletheia-ng/ApolloMoEDataset`

- **user**: `How can I modify this C# code to query a database of restaurant ratings and return only those with excellent cuisine and a minimum rating of 4 stars, while also incorporating the requirement that the menu is somewhat expensive? ⏎ ''' ⏎ List<Restaurant> restaurantData = _context.Restaurants.ToList(); ⏎ List<Restaurant> excellentRestaurants = new List<Restaurant>(); ⏎ foreach (Restaurant restaurant in restaurantData) ⏎ { ⏎ if (restaurant.Cuisine == "excellent" && restaurant.Rating >= 4 && restaurant.MenuPrice == "somewhat expens [...]`
- **assistant**: `<|input_lang|><eng><task_plan><|chat|></task_plan><|target_lang|><eng><response>To modify the given C# code to query a database of restaurant ratings and return only those with excellent cuisine, a minimum rating of 4 stars, and a somewhat expensive menu, you can directly use LINQ (Language Integrated Query) to filter the data. This approach is more efficient because it allows the database to handle the filtering, which can be optimized by the database engine, rather than retrieving all records into memory and then [...]`

## aya_collection

Aletheia-ng/aya_collection_language_split (Hausa; QA and summarisation) -> <qa> / <summarize>

**task** `qa` · **source_lang** `hau` · **targ_lang** `hau` · **dataset** `Aletheia-ng/aya_collection_language_split`

- **user**: `Harry yaro ne mai matsalar koyo. A ranar haihuwarsa ta huɗu, an ba shi pug mai suna Millie. Makonni biyu bayan zuwan kare, ya fi farin ciki da kwanciyar hankali kuma ya furta kalmominsa na farko, "kare" da "mamma". Watanni biyu bayan haka, ɓarayi suka saci karen, kuma yanzu ɗan ƙaramin yaron ya koma inda ya fara. Ya ƙi yin magana tun lokacin da ya rasa abokinsa mafi kyau. Mahaifiyarsa ta damu kuma ta ba shi wani kare, amma "ya kore shi". Mrs. Hainsworth, mahaifiyarsa, ta ce, "Ɗana yana baƙin ciki sosai. Zai je kusa [...]`
- **assistant**: `<|input_lang|><hau><task_plan><qa></task_plan><|target_lang|><hau><response>1. a'a 2. wani pug Na uku. Millie 4. da kuma Harry 5. huɗu Na shida. Misis Hainsworth 7. da 8. a cikin makonni biyu ya faɗi kalmominsa na farko 9. a'a 10. ɓarayi sun sace kare 11. watanni biyu 12. baƙin ciki 13. da A shekara ta 14. Dabbobi a Matsayin Magani Shekara goma sha biyar. Maureen Hennis ta kasance 16. a'a 17. ba a sani ba 18. a'a 19. na tsawon sa'o'i. Shekaru ashirin. A'a, ba haka ba ne.`

**task** `qa` · **source_lang** `hau` · **targ_lang** `hau` · **dataset** `Aletheia-ng/aya_collection_language_split`

- **user**: `Fasali na XXVIII HOME AGAIN - ƙarshe Lokacin da karo ya zo, Dick, don ya ceci kansa daga rauni, ya ba da tsalle a cikin iska, kuma Sergeant Brown ya yi haka. Tashin hankali ya tura _Searchlight_ baya, kuma lokacin da saurayi ya sauka sai ya sami kansa a kan jirgin saman Flyway, kusa da Dan Baxter. "Dick Rover!" ya yi ihu tsohon ɗan fashi na Putnam Hall. "To, jirgin ruwanku ne wanda ya haɗu da mu?" "Baxter, ina ne Dora Stanhope?" ya yi ihu Dick, da zaran ya iya magana. Ya ji tsoron cewa ɗayan ko duka jiragen ruwa su [...]`
- **assistant**: `<|input_lang|><hau><task_plan><qa></task_plan><|target_lang|> unsure, but most likely <hau><response>1. a'a 2. a'a Na uku. Baxter 4. yarinyar 5. a cikin hannayen Dick. Na shida. Baxter 7. a kan jirgin saman Flyway, 8. a'a Na tara. Arnold Baxter 10. haɗuwa`

## xp3x_kongo

Aletheia-ng/xP3x-Kongo: every target is Kikongo (not a supported language) -> all rows dropped, counted for the report

_no row survived the filters in the sample (see the report for why)_

## datagen_sft

BeardedMonster/data-gen sft/*.jsonl (messages already canonical; supported languages only; io_markers_ok; confidence >= 0.7)

**task** `extractive-qa` · **source_lang** `twi` · **targ_lang** `eng` · **dataset** `BeardedMonster/data-gen`

- **system**: `You are a helpful AI assistant. You have these tools: ⏎ [{"type":"function","function":{"name":"search_internet","description":"Search the web. Returns a LIST OF RESULTS (title, url, snippet) -- not the answer. To read one you must then call fetch_page on its url.","parameters":{"type":"object","properties":{"query":{"type":"string","description":"Short search query in English, 2-8 words."}},"required":["query"]}}},{"type":"function","function":{"name":"get_weather_forecast","description":"Get the weather forecast fo [...]`
- **user**: `Mepɛ sɛ mehu sɛnea yɛbɛtumi asiesie yɛn fie so mma obi annyegyɛ mu, mmua m'ani wɔ English.`
- **assistant**: `<|input_lang|><aka><think>The user is asking for home security tips and wants the answer in English. I will search the documents for security measures.</think><task_plan><|RAG|></task_plan>`
- **tool**: `Passage 1: 'To secure your home, ensure all entry points have sturdy locks. Installing motion-sensor lighting around the perimeter can deter intruders. Neighborhood watch programs also increase safety by fostering community vigilance.' (Section: Home Safety)`
- **assistant**: `<|input_lang|><aka><task_plan><|explain|></task_plan><|target_lang|><eng><response>According to the Home Safety section of the document, you can secure your home by using sturdy locks on all entry points and installing motion-sensor lighting around the house. Additionally, joining a neighborhood watch program can help increase safety through community vigilance.`
- **user**: `Sɛ meka 'Watch out for strangers' a, ɛyɛ dɛn wɔ Akan mu?`
- **assistant**: `<|input_lang|><eng><task_plan><translate></task_plan><|target_lang|><aka><response>Hwɛ wo ho yie mma obi a wonnim anka wo.`
- **user**: `Ɛyɛ dɛn wɔ English mu sɛ 'Siesie wo pon mu yie'?`
- **assistant**: `<|input_lang|><aka><task_plan><translate></task_plan><|target_lang|><eng><response>Secure your door well.`
- **user**: `Mmua m'ani wɔ English, m'asɛm a ɛka sɛ 'Sika a yɛde siesie fie' no, ɛyɛ dɛn wɔ hɔn krataa no mu?`
- **assistant**: `<|input_lang|><aka><think>The user is asking about the cost of home security mentioned in the document, wanting the answer in English.</think><task_plan><|RAG|></task_plan>`
- **tool**: `No passages found mentioning specific costs for security measures.`
- **assistant**: `<|input_lang|><aka><think>The document does not mention costs. I must state this clearly.</think><task_plan><|explain|></task_plan><|target_lang|><eng><response>The document does not provide any information regarding the cost of securing a home.`

**task** `multi-turn-chat` · **source_lang** `ewe` · **targ_lang** `twi` · **dataset** `BeardedMonster/data-gen`

- **system**: `You are a helpful assistant that speaks West African languages. You have these tools: ⏎ [{"type":"function","function":{"name":"set_reminder","description":"Create a reminder for the user at a specific time.","parameters":{"type":"object","properties":{"when":{"type":"string","description":"ISO datetime or a phrase like 'tomorrow 07:00'."},"text":{"type":"string"}},"required":["when","text"]}}},{"type":"function","function":{"name":"search_db","description":"Search a structured table by free text or field filters. Us [...]`
- **user**: `Nyuenyi, mɛlele nu be ɖe afi. Meɖɔ nu tɔtrɔ ɖe AFCON ŋu. Lɛlɛ nu sia ɖe Akan nu: 'The 2023 Africa Cup of Nations was a historic tournament with many surprises.' ⏎  ⏎ Sia nye dɔwɔnu gã aɖe: ⏎  ⏎ REPORT ON AFCON INFRASTRUCTURE AND FAN EXPERIENCE ⏎  ⏎ Introduction ⏎ The 2023 Africa Cup of Nations (AFCON) held in Côte d'Ivoire provided a unique study on the intersection of urban development and sports tourism. The tournament was designed not only as a sporting event but as a catalyst for national infrastructure upgrades. Over the co [...]`
- **assistant**: `<|input_lang|><ewe><think>The user wants a translation of a sentence and a summary of a long English report into Akan. Input is Ewe, target is Akan.</think><task_plan><translate><summarize></task_plan><|target_lang|><aka><response>Sɛnɛ: '2023 Africa Cup of Nations yɛɛn mmoa agya a, na ɛdeɛ ɛyɛ nwanwa pii baa.' ⏎  ⏎ Saman: Sɛmman yi ka kyerɛ sɛ Côte d'Ivoire yɛɛ AFCON 2023 kwan a ɛyɛ eyie. Wɔsii mpanin stadium a ɛyɛ fɛ, na wɔsii kwan a ɛyɛ den. Ɛmom, kwan a wɔfa so kɔ stadium no yɛtwee mma obiara efisɛ sika a wɔpɛ no yɛ [...]`
- **user**: `Sabi, mɛlele nu be. Meɖɔ nu tɔtrɔ ɖe kpalɔvɔnu ŋu. Me lɔle nu sia ɖe JSON mu: 'The winner was Ivory Coast with 2 goals, and the loser was Nigeria with 0 goals.' Use keys 'winner', 'winner_score', 'loser', 'loser_score'.`
- **assistant**: `<|input_lang|><ewe><think>Extract match data from the provided sentence into a JSON format.</think><task_plan><|JSON|><|data_extract|></task_plan><|target_lang|><aka><response>{ ⏎   "winner": "Ivory Coast", ⏎   "winner_score": 2, ⏎   "loser": "Nigeria", ⏎   "loser_score": 0 ⏎ }`
- **user**: `Me lɔle nu ɖe kpalɔvɔnu ŋu. Meɖɔ nu tɔtrɔ ɖe kpalɔvɔnu ŋu. Me lɔle nu sia ɖe Akan nu: 'How is the weather today in Kumasi?'`
- **assistant**: `<|input_lang|><ewe><think>Simple conversational translation from Ewe to Akan.</think><task_plan><|chat|><translate></task_plan><|target_lang|><aka><response>Kumanasi mu mfuo no te sɛn nnɛ?`
- **user**: `Me ɖɔ nu tɔtrɔ ɖe kpalɔvɔnu ŋu. Me lɔle nu sia: 'Search the database for a fan named Kwame and tell me his ticket balance.' Use the tool if needed.`
- **assistant**: `<|input_lang|><ewe><think>User wants to find a record for a fan named 'Kwame' in the database to check their ticket balance.</think><task_plan><|plan|></task_plan>`
- **tool**: `[{"id": "F102", "name": "Kwame Mensah", "balance": 120, "status": "active"}]`
- **assistant**: `<|input_lang|><ewe><think>Found one record for Kwame Mensah with a balance of 120. Now report this back in Akan.</think><task_plan><|chat|></task_plan><|target_lang|><aka><response>Mihyia obi a ne din yɛ Kwame Mensah wɔ database no mu, na ne ticket sika a ɛda hɔ yɛ 120.`

## global_mmlu_dev

CohereLabs/Global-MMLU dev split (ha ig ny sn so sw yo) -> multiple choice. No train split exists; 'test' is excluded

**task** `multiple_choice` · **source_lang** `hau` · **targ_lang** `hau` · **dataset** `CohereLabs/Global-MMLU` · **instruction language** `hau`

- **user**: `Wace amsa ce daidai a wannan tambaya? ⏎  ⏎ Wanne daga cikin waɗannan dole ne ya zama sifa ta oligopoly? ⏎ A. Shiga da fita daga kasuwa kyauta ⏎ B. Wasu manyan furodusoshi ⏎ C. Ɗayan mai samarwa na mai kyau ba tare da madaidaicin madaidaicin ba ⏎ D. Samfurin kamanni`
- **assistant**: `<|input_lang|><hau><task_plan><qa><answer></task_plan><|target_lang|><hau><response>B. Wasu manyan furodusoshi`

**task** `multiple_choice` · **source_lang** `hau` · **targ_lang** `hau` · **dataset** `CohereLabs/Global-MMLU` · **instruction language** `yor`

- **user**: `Èwo ni èsì tó tọ́ sí ìbéèrè yìí? ⏎ Dala dubu biyar ana haɗowa kowace shekara akan ƙimar ribar $x\%$ tana ɗaukar shekaru shida zuwa ninki biyu. A daidai wannan adadin riba, shekaru nawa zai ɗauki $\$300$ don girma zuwa $\$9600$? ⏎ A. 12 ⏎ B. 1 ⏎ C. 30 ⏎ D. 5`
- **assistant**: `<|input_lang|><yor><hau><task_plan><qa><answer></task_plan><|target_lang|><hau><response>C. 30`

## mmlu_prox_validation

li-lab/MMLU-ProX validation split (af sw wo yo zu; 10 options) -> multiple choice

**task** `multiple_choice` · **source_lang** `swh` · **targ_lang** `swh` · **dataset** `li-lab/MMLU-ProX` · **instruction language** `yor`

- **user**: `Miundo ya shirika inayotambulika kwa mitindo ya usimamizi ya kidemokrasia na ujumuishi inaweza kuelezwa vipi? ⏎ A. Tambarare ⏎ B. Kibirokrasia ⏎ C. Kiimla ⏎ D. Kihierarkia ⏎ E. Kifanyaji ⏎ F. Iliyogatuliwa ⏎ G. Mfumo wa matriki ⏎ H. Mtandao ⏎ I. Kiidara ⏎ J. Iliyosimamiwa katikati Èwo ni èsì tó tọ́?`
- **assistant**: `<|input_lang|><swh><yor><task_plan><qa><answer></task_plan><|target_lang|><swh><response>A. Tambarare`

**task** `multiple_choice` · **source_lang** `swh` · **targ_lang** `swh` · **dataset** `li-lab/MMLU-ProX` · **instruction language** `eng`

- **user**: `Which option is correct? Give the letter first, then a one-line justification. Katika mashine ya umeme ya DC yenye '2 pole lap winding', upinzani wa kipitisha umeme mmoja ni 2Ω na jumla ya vipitisha umeme ni 100. Tafuta jumla ya upinzani ⏎ A. 50Ω ⏎ B. 1Ω ⏎ C. 25Ω ⏎ D. 200Ω ⏎ E. 10Ω ⏎ F. 100Ω ⏎ G. 500Ω ⏎ H. 150Ω ⏎ I. 75Ω ⏎ J. 20Ω`
- **assistant**: `<|input_lang|><eng><swh><task_plan><qa><answer></task_plan><|target_lang|><swh><response>A. 50Ω`

## sft_new_existing

BeardedMonster/sft-new (the first build: translation, structured tasks, instruct) mapped to the combined schema

**task** `clean` · **source_lang** `afr` · **targ_lang** `afr` · **dataset** `BeardedMonster/sft-new`

- **user**: `Clean this text (stray symbols, odd spacing, leftover markup, repeated words) and return the result. ⏎  ⏎ https://example.com/page?id=7  Elke lewende wese op die aarde is  uitgewis – mense, groot  en klein diere en   voëls.   Hulle  is almal  dood.  Net Noag   en hulle  wat saam met hom  in die  ark  was, het oorleef.`
- **assistant**: `<|input_lang|><afr><task_plan><clean></task_plan><|target_lang|><afr><response>Elke lewende wese op die aarde is uitgewis – mense, groot en klein diere en voëls. Hulle is almal dood. Net Noag en hulle wat saam met hom in die ark was, het oorleef.`

**task** `clean` · **source_lang** `afr` · **targ_lang** `afr` · **dataset** `BeardedMonster/sft-new`

- **user**: `Clean up this text: fix the spacing and typos, remove stray markup and unrelated lines, and return only the cleaned text. ⏎  ⏎ hulle nageslag het seevaarders <span>in <p>verskillende lande geword.`
- **assistant**: `<|input_lang|><afr><task_plan><clean></task_plan><|target_lang|><afr><response>Hulle nageslag het seevaarders in verskillende lande geword.`
