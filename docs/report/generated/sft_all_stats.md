## A.7 Result of the `sft-all` build (7 steps finished)

**Rows kept: 6,025,319**; rows routed to late pretraining folders: 0; total build time of the finished steps: 0.4 h.

### By converter

| converter | kept | late rows | seconds |
|---|---|---|---|
| sft_new_existing | 5,627,582 | 0 | 531 |
| old_tasks | 245,460 | 0 | 613 |
| old_intent | 56,434 | 0 | 80 |
| old_hate | 45,821 | 0 | 75 |
| old_pos | 27,212 | 0 | 56 |
| old_topic | 18,768 | 0 | 115 |
| datagen_sft | 4,042 | 0 | 12 |

### By task

| task | rows | share |
|---|---|---|
| unknown | 5,171,887 | 85.8% |
| sentiment | 236,595 | 3.9% |
| intent | 111,857 | 1.9% |
| hate | 80,631 | 1.3% |
| clean | 66,000 | 1.1% |
| identify | 65,922 | 1.1% |
| ner | 63,653 | 1.1% |
| topic | 47,848 | 0.8% |
| structured_output | 34,164 | 0.6% |
| translation_score | 32,399 | 0.5% |
| correct | 29,999 | 0.5% |
| diacritize | 22,560 | 0.4% |
| summarization | 22,388 | 0.4% |
| pos | 15,676 | 0.3% |
| toxic | 6,192 | 0.1% |
| chat | 4,838 | 0.1% |
| qa | 4,774 | 0.1% |
| summarize | 2,046 | 0.0% |
| headline | 2,046 | 0.0% |
| insufficient-context | 509 | 0.0% |
| extractive-qa | 492 | 0.0% |
| knowledge-boundary | 469 | 0.0% |
| content-writing | 408 | 0.0% |
| financial-analysis | 333 | 0.0% |
| health-education | 320 | 0.0% |
| health-advice | 297 | 0.0% |
| multi-turn-chat | 296 | 0.0% |
| health-triaging | 257 | 0.0% |
| structured-output | 138 | 0.0% |
| intent-detection | 114 | 0.0% |
| language-identification | 86 | 0.0% |
| pos-tagging | 42 | 0.0% |
| sentiment-analysis | 34 | 0.0% |
| tool-calling | 20 | 0.0% |
| toxicity-spam-detection | 13 | 0.0% |
| topic-classification | 12 | 0.0% |
| translation | 4 | 0.0% |

### By language (as source / as target)

| language | as source | as target |
|---|---|---|
| eng | 2,549,809 | 3,115,608 |
| swh | 682,950 | 460,105 |
| yor | 585,805 | 477,447 |
| afr | 540,518 | 512,086 |
| hau | 257,411 | 241,493 |
| xho | 238,653 | 220,569 |
| ibo | 176,716 | 164,852 |
| zul | 126,646 | 107,284 |
| som | 117,016 | 109,793 |
| kin | 108,762 | 87,929 |
| sot | 96,663 | 85,087 |
| sna | 94,587 | 81,930 |
| nya | 83,303 | 79,521 |
| twi | 67,712 | 56,777 |
| wol | 59,512 | 39,295 |
| fon | 51,474 | 47,208 |
| orm | 55,335 | 35,406 |
| tsn | 48,014 | 38,980 |
| pcm | 34,741 | 30,650 |
| ewe | 28,384 | 21,025 |
| efi | 9,138 | 6,113 |
| urh | 6,177 | 3,161 |
| ibb | 5,993 | 3,000 |

### Languages per input (`<|input_lang|>` size)

| languages in input | rows |
|---|---|
| 1 | 5,700,926 |
| 2 | 324,393 |

### Top 40 language pairs

| src>tgt | rows |
|---|---|
| swh>eng | 679,616 |
| afr>eng | 527,703 |
| eng>afr | 499,271 |
| eng>swh | 456,771 |
| yor>eng | 389,613 |
| eng>yor | 281,252 |
| xho>eng | 229,256 |
| hau>eng | 217,100 |
| eng>xho | 211,172 |
| eng>hau | 201,186 |
| yor>yor | 196,131 |
| ibo>eng | 138,182 |
| eng>ibo | 126,304 |
| zul>eng | 111,308 |
| som>eng | 111,158 |
| kin>eng | 104,434 |
| eng>som | 103,935 |
| sot>eng | 93,663 |
| eng>zul | 91,946 |
| sna>eng | 87,704 |
| eng>kin | 83,601 |
| eng>sot | 82,087 |
| nya>eng | 79,641 |
| eng>nya | 75,859 |
| eng>sna | 75,047 |
| eng>eng | 67,302 |
| twi>eng | 60,560 |
| orm>eng | 52,335 |
| eng>twi | 49,684 |
| wol>eng | 48,729 |
| fon>eng | 44,424 |
| hau>hau | 40,247 |
| eng>fon | 40,145 |
| ibo>ibo | 38,487 |
| tsn>eng | 37,694 |
| eng>orm | 32,406 |
| pcm>pcm | 29,987 |
| eng>tsn | 28,660 |
| eng>wol | 28,512 |
| ewe>eng | 21,391 |

### Most frequent drop reasons

| reason | rows |
|---|---|
| config_skipped | 654,303 |
| pos_labels_invalid | 97,427 |
| blank_row | 35,379 |
| ner_labels_invalid | 31,545 |
| duplicate | 21,391 |
| plan_token:<chunk> | 20,744 |
| empty_response | 5,734 |
| unsupported_language_config | 2,400 |
| language_not_supported | 568 |
| assistant_language_not_supported | 157 |
| low_confidence_or_markers | 73 |
