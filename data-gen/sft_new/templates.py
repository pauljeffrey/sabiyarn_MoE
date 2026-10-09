"""Instruction pools for the non-translation tasks, in the same shape as africomet_translation/instructions.py.

pool(task) -> {instruction language: {"pre": [...], "post": [...], "any": [...]}}
  pre   placed before the input, post after it, any works in either place.
  Slots: {target} = name of the language the answer must be in, {source} = name of the input's language, {n} = a number.

Written by hand for seven instruction languages (eng pcm yor hau ibo twi ewe); `confidence` records how much to trust the
non-English wording (same convention as the translation bank). Language is chosen per sample by `choose_language`.
"""

from __future__ import annotations

import random

from africomet_translation.instructions import CONFIDENCE, DISPLAY_NAMES, Instruction, _name

_POOLS: dict[str, dict[str, dict[str, list[str]]]] = {}


def _reg(task: str, lang: str, pre=(), post=(), any_=()):
    _POOLS.setdefault(task, {})[lang] = {"pre": list(pre), "post": list(post), "any": list(any_)}


# ---------------------------------------------------------------- summarize
_reg("summarize", "eng",
     pre=["Summarize the following text.", "Summarize this in {target}:", "Give me a short summary of the text below.",
          "Please write a brief summary of the following article in {target}.", "TL;DR in {target}:",
          "Read the passage below and summarise it in two or three sentences.", "Condense the following text into a short summary.",
          "Write a concise {target} summary of this news story:", "What is the main point of the text below? Summarize it in {target}.",
          "Summarize the key points of the following in {target}, without adding anything that is not in the text."],
     post=["Summarize the text above.", "Give a short summary of the above in {target}.", "Please summarise this in {target}.",
           "Can you condense this into a few sentences?", "Write a brief {target} summary of the passage above.",
           "What are the key points? Reply with a summary in {target}."],
     any_=["Summarize in {target}.", "Summary in {target}, please."])
_reg("summarize", "pcm",
     pre=["Summarize dis tori for me:", "Make short summary of wetin dey inside dis text:", "Abeg summarize dis one for {target}:",
          "Tell me di main point of dis text for short:"],
     post=["Summarize wetin dey up there for me.", "Abeg give me short summary of dis one for {target}.", "Make e short, summarize am."],
     any_=["Summarize am for {target}."])
_reg("summarize", "yor",
     pre=["Ṣe àkópọ̀ ọ̀rọ̀ yìí ní ṣókí:", "Kọ àkótán ìròyìn yìí ní èdè {target}:", "Fún mi ní àkótán kúkúrú ti àyọkà yìí:",
          "Jọ̀wọ́ ṣe àkótán àyọkà tó wà nísàlẹ̀ yìí:"],
     post=["Ṣe àkótán ohun tí ó wà lókè yìí.", "Fún mi ní àkótán rẹ̀ ní ṣókí ní èdè {target}."], any_=["Ṣe àkótán rẹ̀ ní èdè {target}."])
_reg("summarize", "hau",
     pre=["Takaita wannan rubutun:", "Ka wannan labarin ka takaita shi a harshen {target}:", "Ka ba ni taƙaitaccen bayani game da wannan rubutun:",
          "Don Allah, taƙaita wannan labari:"],
     post=["Takaita rubutun da ke sama.", "Ka taƙaita wannan a cikin jimloli kaɗan a harshen {target}."], any_=["Taƙaita shi a {target}."])
_reg("summarize", "ibo", pre=["Chịkọta ederede a n'ụzọ dị mkpụmkpụ:", "Dee nchịkọta nke akụkọ a n'asụsụ {target}:", "Nye m nchịkọta nkenke nke ihe e dere n'okpuru:"],
     post=["Chịkọta ihe dị n'elu a.", "Biko, nye m nchịkọta ya n'asụsụ {target}."], any_=["Chịkọta ya n'asụsụ {target}."])
_reg("summarize", "twi", pre=["Kyerɛw saa nsɛm yi ho nsɛm tiawa:", "Tɔw saa asɛm yi mu kɔ ase wɔ {target} mu:"], post=["Tɔw nea ɛwɔ soro no mu kɔ ase."])
_reg("summarize", "ewe", pre=["Ŋlɔ nyatakaka sue le nya sia ŋu:", "Ɖe nyatakakaa nu kpui le {target} me:"], post=["Ɖe nu si le dziƒe la nu kpui."])

# ---------------------------------------------------------------- headline
_reg("headline", "eng",
     pre=["Write a headline for the following article.", "Give this news story a title in {target}:", "Suggest a short headline for this text.",
          "What would be a good headline for the text below? Reply in {target}."],
     post=["Write a headline for the text above.", "Give the article above a title in {target}.", "Come up with a catchy headline for this."],
     any_=["Headline in {target}, please."])
_reg("headline", "pcm", pre=["Give dis news one headline:", "Wetin be good headline for dis story?"], post=["Give di story wey dey up there headline for {target}."])
_reg("headline", "yor", pre=["Kọ àkọ́lé fún ìròyìn yìí:", "Fún àyọkà yìí ní àkọ́lé ní èdè {target}:"], post=["Kọ àkọ́lé fún ohun tí ó wà lókè yìí."])
_reg("headline", "hau", pre=["Rubuta kanun labari na wannan rubutu:", "Ba wannan labarin taken kanun labari a {target}:"], post=["Rubuta kanun labarin da ke sama."])
_reg("headline", "ibo", pre=["Dee isiokwu maka akụkọ a:", "Nye akụkọ a aha isiokwu n'asụsụ {target}:"], post=["Dee isiokwu maka ihe dị n'elu."])

# ---------------------------------------------------------------- sentiment (labels are English words: Positive / Negative / Neutral)
_reg("sentiment", "eng",
     pre=["What is the sentiment of the following text? Answer Positive, Negative or Neutral.", "Classify the sentiment of this sentence:",
          "Is the following sentence positive, negative or neutral?", "Do sentiment analysis on this text:", "Tell me the sentiment (Positive, Negative or Neutral):",
          "Read the sentence below and decide whether the writer feels positively, negatively or neutrally about it."],
     post=["What is the sentiment of the text above?", "Is that positive, negative or neutral?", "Label the sentiment of this text."])
_reg("sentiment", "pcm", pre=["Wetin be di sentiment of dis sentence? Positive, Negative or Neutral:", "Dis text good, bad or e just dey normal? Tell me:"],
     post=["Wetin be di sentiment of wetin dey up there?"])
_reg("sentiment", "yor", pre=["Kí ni ìmọ̀lára tó wà nínú gbólóhùn yìí? Rere, Búburú tàbí Aláìṣègbè:", "Ṣe ìtúpalẹ̀ ìmọ̀lára ọ̀rọ̀ yìí:"], post=["Kí ni ìmọ̀lára ọ̀rọ̀ tó wà lókè yìí?"])
_reg("sentiment", "hau", pre=["Menene yanayin ra'ayin wannan jimla? Tabbatacce, Korau ko Tsaka-tsaki:", "Yi nazarin ra'ayin wannan rubutu:"], post=["Menene ra'ayin da ke cikin rubutun da ke sama?"])
_reg("sentiment", "ibo", pre=["Kedu mmetụta dị n'ahịrịokwu a? Ọma, Njọ ma ọ bụ Ọnọdụ etiti:", "Nyochaa mmetụta nke ederede a:"], post=["Kedu mmetụta nke ederede dị n'elu?"])

# ---------------------------------------------------------------- topic
_reg("topic", "eng",
     pre=["What is the topic of this news headline?", "Classify this headline into a topic:", "Which category does the following text belong to?",
          "Choose the best topic label for this text:"],
     post=["What topic does the text above belong to?", "Which category is this?"])
_reg("topic", "pcm", pre=["Which topic dis news belong to?", "Wetin be di topic of dis headline?"], post=["Which topic di one wey dey up there belong to?"])
_reg("topic", "yor", pre=["Ẹ̀ka wo ni àkọ́lé ìròyìn yìí wà?", "Kí ni kókó ọ̀rọ̀ ìròyìn yìí?"], post=["Ẹ̀ka wo ni ohun tó wà lókè yìí wà?"])
_reg("topic", "hau", pre=["Wane nau'in labari ne wannan kanun labari?", "Ka rarraba wannan rubutu zuwa rukuni:"], post=["Wane rukuni ne rubutun da ke sama ya ke ciki?"])
_reg("topic", "ibo", pre=["Kedu ụdị akụkọ bụ isiokwu a?", "Kewaa ederede a n'otu ngalaba:"], post=["Kedu ngalaba ederede dị n'elu bụ?"])

# ---------------------------------------------------------------- hate / abuse / toxicity, intent
_reg("hate", "eng", pre=["Does this text contain hate speech, abuse or neither? Answer Hate, Abuse or Normal.", "Classify this text as Normal, Abuse or Hate:",
                         "Is the following message offensive? Label it Normal, Abuse or Hate."],
     post=["Is the text above hateful or abusive?", "Label the message above: Normal, Abuse or Hate."])
_reg("hate", "pcm", pre=["Dis text get hate speech or abuse inside? Talk am:", "Wetin dis message be: Normal, Abuse or Hate?"], post=["Di message wey dey up there na Normal, Abuse or Hate?"])
_reg("hate", "yor", pre=["Ṣé ọ̀rọ̀ ìkórìíra tàbí ìwà ìbàjẹ́ wà nínú ọ̀rọ̀ yìí?", "Pín ọ̀rọ̀ yìí sí Déédé, Ìbàjẹ́ tàbí Ìkórìíra:"], post=["Ṣé ọ̀rọ̀ tó wà lókè yìí jẹ́ ti ìkórìíra?"])
_reg("hate", "hau", pre=["Shin wannan rubutun yana ɗauke da kalaman ƙiyayya ko cin zarafi?", "Rarraba wannan saƙon: Na al'ada, Cin zarafi ko Ƙiyayya:"], post=["Shin rubutun da ke sama na ƙiyayya ne?"])
_reg("hate", "ibo", pre=["Ederede a nwere okwu ịkpọasị ma ọ bụ mkparị?", "Kewaa ozi a: Nkịtị, Mkparị ma ọ bụ Ịkpọasị:"], post=["Ederede dị n'elu ọ bụ okwu ịkpọasị?"])
_reg("intent", "eng", pre=["What does the speaker want? Name the intent.", "Identify the intent of this request:", "Classify the intent of the following utterance:",
                           "What is the user trying to do with this message?"], post=["What is the intent of the utterance above?", "Name the intent of that request."])
_reg("intent", "pcm", pre=["Wetin di person want for dis message?", "Tell me di intent of dis request:"], post=["Wetin be di intent of wetin dey up there?"])
_reg("intent", "yor", pre=["Kí ni ète ẹni tó sọ̀rọ̀ yìí?", "Dárúkọ ìdí tí a fi kọ ọ̀rọ̀ yìí:"], post=["Kí ni ète ọ̀rọ̀ tó wà lókè yìí?"])
_reg("intent", "hau", pre=["Menene manufar mai magana a nan?", "Gano dalilin wannan buƙata:"], post=["Menene manufar maganar da ke sama?"])
_reg("intent", "ibo", pre=["Kedu ihe onye na-ekwu chọrọ?", "Chọpụta ebumnuche nke arịrịọ a:"], post=["Kedu ebumnuche nke okwu dị n'elu?"])

# ---------------------------------------------------------------- story generation
_reg("story", "eng",
     pre=["Write a short story in {target} based on this synopsis:", "Tell a story in {target} from the following idea:", "Turn this synopsis into a short story:",
          "Using the synopsis below, write a story. Write it in {target}.", "Expand the following premise into a short story in {target}:"],
     post=["Write a story in {target} based on the synopsis above.", "Now tell the full story in {target}.", "Please turn that into a short story."],
     any_=["Write a {target} story."])
_reg("story", "pcm", pre=["Write short story for {target} from dis idea:", "Tell me story for {target} wey follow dis synopsis:"], post=["Now tell di full story for {target}."])
_reg("story", "yor", pre=["Kọ ìtàn kúkúrú ní èdè {target} láti inú àkótán yìí:", "Sọ ìtàn kan ní èdè {target} tó dá lórí èrò yìí:"], post=["Kọ ìtàn náà ní kíkún ní èdè {target}."])
_reg("story", "hau", pre=["Rubuta gajeren labari a {target} bisa wannan taƙaitaccen bayani:", "Ba da labari a {target} daga wannan ra'ayi:"], post=["Yanzu ka ba da cikakken labarin a {target}."])
_reg("story", "ibo", pre=["Dee obere akụkọ n'asụsụ {target} site na nchịkọta a:", "Kọọ akụkọ n'asụsụ {target} gbasara echiche a:"], post=["Ugbu a kọọ akụkọ ahụ n'ụzọ zuru ezu n'asụsụ {target}."])

# ---------------------------------------------------------------- stories without a synopsis
_reg("story_prompt", "eng", pre=["Write a short story in {target}.", "Tell me a story in {target}.", "Write a simple children's story in {target}.",
                                 "Please write a short, easy story for kids in {target}.", "I would like a short story in {target}, with a beginning, a middle and an end."],
     post=[], any_=["A short story in {target}, please."])
_reg("story_prompt", "pcm", pre=["Write short story for {target}.", "Tell me one story for {target}.", "Abeg write simple story for pikin for {target}."])
_reg("story_prompt", "yor", pre=["Kọ ìtàn kúkúrú kan ní èdè {target}.", "Sọ ìtàn kan fún mi ní èdè {target}.", "Jọ̀wọ́ kọ ìtàn ọmọdé kan ní èdè {target}."])
_reg("story_prompt", "hau", pre=["Rubuta gajeren labari a harshen {target}.", "Ba ni labari a {target}.", "Don Allah, rubuta labarin yara a {target}."])
_reg("story_prompt", "ibo", pre=["Dee obere akụkọ n'asụsụ {target}.", "Kọọrọ m akụkọ n'asụsụ {target}.", "Biko, dee akụkọ ụmụaka n'asụsụ {target}."])
_reg("story_seed", "eng", pre=["Write a story in {target} that begins with this sentence: {seed}", "Continue from this opening and tell the whole story in {target}: {seed}",
                              "Use this first sentence and write a short story in {target}: {seed}"])
_reg("story_seed", "pcm", pre=["Use dis first sentence write story for {target}: {seed}"])
_reg("story_seed", "yor", pre=["Lo gbólóhùn àkọ́kọ́ yìí láti kọ ìtàn ní èdè {target}: {seed}"])
_reg("story_seed", "hau", pre=["Yi amfani da wannan jimla ta farko ka rubuta labari a {target}: {seed}"])
_reg("story_seed", "ibo", pre=["Jiri ahịrịokwu mbụ a dee akụkọ n'asụsụ {target}: {seed}"])

# ---------------------------------------------------------------- generate the synopsis from the story / retell
_reg("synopsis", "eng", pre=["Write a one-sentence synopsis of the following story, in {target}:", "What is this story about? Give a short synopsis in {target}."],
     post=["Write a one-sentence synopsis of the story above in {target}."])
_reg("synopsis", "pcm", pre=["Wetin dis story be about? Write am short for {target}:"], post=["Write one-line synopsis of di story wey dey up there for {target}."])
_reg("synopsis", "yor", pre=["Kọ àkótán ìtàn yìí ní gbólóhùn kan ní èdè {target}:"], post=["Kọ àkótán ìtàn tó wà lókè yìí ní èdè {target}."])
_reg("synopsis", "hau", pre=["Rubuta taƙaitaccen bayani na wannan labari a {target}:"], post=["Rubuta taƙaitaccen bayanin labarin da ke sama a {target}."])
_reg("synopsis", "ibo", pre=["Dee nchịkọta nke akụkọ a n'otu ahịrịokwu n'asụsụ {target}:"], post=["Dee nchịkọta nke akụkọ dị n'elu n'asụsụ {target}."])

# ---------------------------------------------------------------- answer-in suffixes: used to condition an existing instruction on an output language
_reg("answer_in", "eng", post=["Provide your response in {target}.", "Answer in {target} only.", "Please respond in {target}.", "Write your answer in {target}.",
                               "Reply in {target}.", "Your response must be in {target}.", "Respond only in {target}."])
_reg("answer_in", "pcm", post=["Answer for {target} only.", "Abeg reply for {target}.", "Give me di answer for {target}."])
_reg("answer_in", "yor", post=["Dáhùn ní èdè {target}.", "Jọ̀wọ́, fi èdè {target} dáhùn.", "Kọ ìdáhùn rẹ ní èdè {target} nìkan."])
_reg("answer_in", "hau", post=["Ka ba da amsarka da harshen {target}.", "Amsa a {target} kawai.", "Don Allah ka amsa da {target}."])
_reg("answer_in", "ibo", post=["Zaa n'asụsụ {target}.", "Biko, jiri asụsụ {target} zaghachi.", "Nyefee azịza gị n'asụsụ {target} naanị."])
_reg("answer_in", "twi", post=["Fa {target} bua."])
_reg("answer_in", "ewe", post=["Ɖo eŋu le {target} me."])

# ---------------------------------------------------------------- explain a topic / wiki article (title -> request)
_reg("wiki_ask", "eng",
     pre=["Tell me about {title}. Give your answer in {target} only.", "Who or what is {title}? Answer in {target}.", "Write an encyclopedia-style article about {title} in {target}.",
          "Explain {title} to me in {target}.", "Give me a detailed overview of {title}, written in {target}.", "What do you know about {title}? Reply in {target}."],
     post=[], any_=["{title} - give me the facts, in {target}.", "Tell me everything important about {title} in {target}."])
_reg("wiki_ask", "pcm", pre=["Tell me about {title}. Answer for {target} only.", "Who be {title}? Explain am for {target}."], any_=["Wetin you know about {title}? Talk am for {target}."])
_reg("wiki_ask", "yor", pre=["Sọ fún mi nípa {title}. Dáhùn ní èdè {target} nìkan.", "Ta ni tàbí kí ni {title}? Ṣàlàyé ní èdè {target}."], any_=["Kí ni o mọ̀ nípa {title}? Fèsì ní {target}."])
_reg("wiki_ask", "hau", pre=["Ka gaya mini game da {title}. Ka amsa da {target} kawai.", "Wane ne ko menene {title}? Bayyana a {target}."], any_=["Me ka sani game da {title}? Ka amsa a {target}."])
_reg("wiki_ask", "ibo", pre=["Gwa m banyere {title}. Zaa n'asụsụ {target} naanị.", "Onye ma ọ bụ gịnị bụ {title}? Kọwaa n'asụsụ {target}."], any_=["Gịnị ka ị maara gbasara {title}? Zaa n'asụsụ {target}."])

# ---------------------------------------------------------------- multiple choice
_reg("mcq", "eng", pre=["Answer the multiple-choice question below with the letter of the correct option.", "Choose the correct answer (A, B, C or D):",
                        "Pick the right option for the following question.", "Which option is correct? Give the letter first, then a one-line justification."],
     post=["Which option is correct?", "Answer with A, B, C or D."])
_reg("mcq", "pcm", pre=["Pick di correct answer (A, B, C or D):", "Which option correct for dis question?"], post=["Which one correct?"])
_reg("mcq", "yor", pre=["Yan èsì tó tọ́ (A, B, C tàbí D):", "Èwo ni èsì tó tọ́ sí ìbéèrè yìí?"], post=["Èwo ni èsì tó tọ́?"])
_reg("mcq", "hau", pre=["Zaɓi amsar da ta dace (A, B, C ko D):", "Wace amsa ce daidai a wannan tambaya?"], post=["Wace amsa ce daidai?"])
_reg("mcq", "ibo", pre=["Họrọ azịza ziri ezi (A, B, C ma ọ bụ D):", "Kedu azịza ziri ezi maka ajụjụ a?"], post=["Kedu azịza ziri ezi?"])

# ---------------------------------------------------------------- math word problems
_reg("math", "eng", pre=["Solve the following problem step by step, then state the final answer.", "Work through this math word problem and give the final answer.",
                         "Solve:", "Show your reasoning and then give the answer in {target}."], post=["Solve the problem above step by step.", "Show your working and give the final answer."])
_reg("math", "pcm", pre=["Solve dis wahala step by step, den tell me di final answer:", "Do dis maths and show how you take do am:"], post=["Solve di one wey dey up there step by step."])
_reg("math", "yor", pre=["Yanjú ìṣòro ìṣirò yìí ní ìgbésẹ̀ ní ìgbésẹ̀:", "Ṣe ìṣirò yìí kí o sì fún mi ní èsì ìkẹyìn:"], post=["Yanjú ìṣòro tó wà lókè yìí ní ìgbésẹ̀ ní ìgbésẹ̀."])
_reg("math", "hau", pre=["Warware wannan matsalar lissafi mataki-mataki:", "Yi wannan lissafi ka ba da amsar ƙarshe:"], post=["Warware matsalar da ke sama mataki-mataki."])
_reg("math", "ibo", pre=["Dozie nsogbu mgbakọ a nzọụkwụ site na nzọụkwụ:", "Mee mgbakọ a wee nye azịza ikpeazụ:"], post=["Dozie nsogbu dị n'elu nzọụkwụ site na nzọụkwụ."])

# ---------------------------------------------------------------- reading comprehension / grounded QA
_reg("qa_context", "eng", pre=["Answer the question using only the passage below.", "Read the passage and answer the question that follows.", "Use the context to answer:"],
     post=["Answer the question using only the context above.", "Based on the passage above, what is the answer?"])
_reg("qa_context", "pcm", pre=["Use dis passage answer di question:"], post=["Use wetin dey up there answer di question."])
_reg("qa_context", "yor", pre=["Lo àyọkà yìí láti dáhùn ìbéèrè náà:"], post=["Lo ohun tó wà lókè yìí láti dáhùn ìbéèrè náà."])
_reg("qa_context", "hau", pre=["Yi amfani da wannan rubutu ka amsa tambayar:"], post=["Yi amfani da abin da ke sama ka amsa tambayar."])
_reg("qa_context", "ibo", pre=["Jiri ederede a zaa ajụjụ ahụ:"], post=["Jiri ihe dị n'elu zaa ajụjụ ahụ."])

# ---------------------------------------------------------------- sequence labelling instructions (when the source gives none)
_reg("ner", "eng", pre=["Tag the named entities in this sentence. Write each word followed by <tag> and its label (PER, ORG, LOC, DATE or O).",
                        "Do named-entity recognition on the sentence below, giving word <tag> LABEL pairs."])
_reg("pos", "eng", pre=["Give the part-of-speech tag of every word in this sentence as word <tag> TAG pairs.", "POS-tag the following sentence. Write each word followed by <tag> and its tag."])


def languages(task: str) -> list[str]:
    return list(_POOLS.get(task, {}))


# weights for the instruction language of a sample: English is the lingua franca of the corpus; the rest follow the input's language
def choose_language(task: str, rng: random.Random, prefer: str | None = None) -> str:
    have = languages(task)
    if prefer in have and rng.random() < 0.35:
        return prefer
    if rng.random() < 0.55 or len(have) == 1:
        return "eng"
    return rng.choice([l for l in have if l != "eng"] or ["eng"])


def make(task: str, rng: random.Random, target: str = "eng", source: str = "eng", prefer: str | None = None,
         position: str | None = None, lang: str | None = None, **slots) -> Instruction:
    il = lang or choose_language(task, rng, prefer)
    bank = _POOLS[task][il]
    want = position or rng.choice(["prepend", "append"])
    pool = bank["pre" if want == "prepend" else "post"] + bank["any"]
    if not pool:                                              # a task with only a prefix pool (or only a suffix pool)
        want = "prepend" if bank["pre"] else "append"
        pool = bank["pre" if want == "prepend" else "post"] + bank["any"]
    tpl = rng.choice(pool)
    text = tpl.format(target=_name(target, il, rng), source=_name(source, il, rng), n=slots.get("n", ""), **{k: v for k, v in slots.items() if k != "n"})
    return Instruction(text, want, il, CONFIDENCE.get(il, "low"))
