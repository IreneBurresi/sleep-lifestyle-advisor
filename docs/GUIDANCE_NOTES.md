# Guidance notes

How the LLM step works, which model it calls and why, and what changed between the two prompt versions. The code is `advisor/cli/generate_guidance.py`, the prompts are in `advisor/prompts/`, every run quoted here is saved in `eval/runs/`.

## What the model receives

Three labelled sections, rendered from a Jinja template:

1. **Group profile**: the cluster's written summary, its focus areas (what guidance for this group should be about, derived from the cluster means), and how the group differs from everyone in standard deviations.
2. **This person**: labels (BMI category, reported disorder, gender, occupation) kept apart from measurements, then a table with the person's value, the group mean, the population mean, and the two deltas in population standard deviations. A last line lists what stands out from the group by at least 0.5 SD, so the model does not have to find it.
3. **Fixed flags**: what a professional will be pointed to. Computed in code from three thresholds (a reported disorder, blood pressure from 140/90, resting heart rate from 100), never by the model. They are printed under the guidance and saved with it; the prompt tells the model not to advise on them.

The output is a typed `Guidance`: a summary and three or four recommendations, validated by pydantic. A validator rejects text containing markup tags, which makes the model retry instead of shipping `</summary>` to a user. There is no generated disclaimer. When there is nothing to refer, nothing is printed; when there is, the line says what, and the model did not write it.

Why individual data and not just the cluster: with the cluster profile alone there are five possible outputs, and a lookup table would do. The deltas are what makes two members of the same cluster get different text. The test set (`eval/testset.json`) checks this with one typical and two divergent members of cluster 0.

## Model choice

**`google` / `gemini-3.5-flash-lite`** for the guidance, **`gemini-3.5-flash`** for the judges (`EVAL_NOTES.md`). Temperature 0.3, `max_tokens` 1500. Provider, model and key are three lines in `.env`; pydantic-ai builds the model from the same `provider:model` string for any provider it knows, so switching to OpenAI, Anthropic, OpenRouter or a local model is a config change (plus, for some providers, one optional dependency group).

The constraint was a free model with enough daily quota to run the evaluation more than once, that returns the typed output reliably. Candidates: Gemini Flash and Flash-Lite through Google AI Studio (free key, no card), and a few free models through OpenRouter. All on the same prompt and the same test cases.

| model | delivered | median latency | tokens out (mean) | notes |
|---|---|---|---|---|
| gemini-3.5-flash-lite (chosen) | 9/9 test cases, 5/5 red-team | 1.2 s | 137 | follows the rules with one recurring exception (below); the largest free quota |
| gemini-3.5-flash, thinking off | 9/9 | 6.9 s | 145 | same text quality at reading; the free tier allows 20 requests a day per model, less than one evaluation pass |
| gemini-3.5-flash, thinking on | 1 case | 27 s | 3397 | thinking tokens twenty times the answer |
| nvidia/nemotron-3-super-120b:free (OpenRouter) | 9/9 | 3 s | 179 | terse, follows the rules |
| minimax/minimax-m3 (OpenRouter) | 9/9 | 3 s | 322 | good text; leaks tool-call markup into the output, which is why the markup validator exists |
| z-ai/glm-5.2:free, google/gemma-4-31b-it:free (OpenRouter) | 2/3, 1/3 | 15 s, 5 s | | rate-limited upstream through a minute of backoff |

Runs: `eval/runs/flash*`, `eval/runs/openrouter/`, smoke tests in `eval/runs/openrouter/smoke/`. Three cases per OpenRouter model except nemotron and minimax, which ran the full set.

Flash-Lite won on quota and latency; the two Gemini models fail the automatic checks on the same single case, the referral written as a recommendation for cluster 3. OpenRouter's free tier is a shared pool capped at 50 requests a day per account, and two of the four models there never answered a full set. For the judges, Flash: larger than Lite, and a different model from the one that wrote the text.

Thinking, where a model has it, stays off: it multiplies output tokens by twenty and latency by ten for text that reads the same, and with `max_tokens` bounded it can spend the whole budget thinking and return nothing. Flash-Lite has no thinking control (the API rejects a thinking budget) and does not need one.

### Trade-offs

- **Cost**: Flash-Lite is free within the AI Studio quotas and priced in fractions of a cent per call beyond them; at 1100 tokens in and 140 out per call, 10k users a month are a few dollars on any small hosted model.
- **Latency**: 1.2 s median. Fine for "generate my weekly guidance", and for an interactive screen too.
- **Reliability**: the free tier caps requests per day and per minute. The script paces calls (`LLM_PAUSE_SECONDS`) and retries with exponential backoff (5, 15, 45 s), honouring `Retry-After`. Cases that still fail are reported, not silently dropped. A production setting would pay for the same model to remove the caps.
- **Quality**: follows the rules with one recurring exception, discussed in `EVAL_NOTES.md`: for the cluster where both flags fire, it tells the user to get their blood pressure checked, which the prompt forbids as advice. Terse: three recommendations in 9 cases of 9.
- **Safety**: the safety-relevant decisions are in code (referral thresholds, forbidden topics in the prompt, output schema, markup rejection). The model's freedom is in the wording of habit advice.
- **Structured output**: pydantic-ai uses tool-call mode by default with Gemini (the model also supports JSON-schema output, not used here). Validation retries show up as roughly doubled input tokens on the affected call (one in the v1 run).

## Retrieval over public guidelines (`--rag`)

Optional, off by default. `make rag` downloads two public-domain documents, the U.S. *Physical Activity Guidelines for Americans* (HHS, 2018, 118 pages) and *Your Guide to Healthy Sleep* (NIH, 2011, 72 pages), converts them to Markdown with pymupdf4llm, which recovers the headings from the layout, splits on the headings, skips front matter, contents, glossary and sidebars, caps chunks at 1200 characters on sentence boundaries, embeds them with `gemini-embedding-2` (768 dimensions) and stores them in an embedded Qdrant Edge shard: a directory on disk, no server. 375 chunks, 293 from the activity guidelines and 82 from the sleep guide, under 218 headings. The index is committed under `rag/index/` (5.6 MB), so `--rag` works without rebuilding it and without the OCR dependency the conversion pulls in for image pages.

With `--rag` the agent gets a `search_guidelines` tool, at most two searches per user, three passages of up to 700 characters each, and one more paragraph of instructions (`advisor/prompts/v2/rag.md`): search only for the habit you are about to recommend, use a passage only if it applies to this person, name the source, never search for blood pressure, heart rate, medication or a disorder. Every search and what it returned is saved in the record.

Retrieval itself works: on five questions with a known answer (`eval/rag_questions.json`) the expected source and the expected fact are in the top three passages 5 times out of 5. The model uses the tool on every case, spends both searches, and tries a third time in 7 cases out of 14; every output names a source, most often "the U.S. activity guidelines suggest at least 150 minutes of moderate activity per week".

What it costs and what it changes, same 14 cases, same judges (`EVAL_NOTES.md`):

| | without RAG | with RAG |
|---|---|---|
| median latency | 1.2 s | 3.9 s |
| tokens in / out (mean) | 1136 / 137 | 7262 / 229 |
| keyword checks passed | 97.1% | 92.9% |
| judge: relevance / safety / actionability / tone | 4.9 / 4.7 / 4.6 / 4.9 | 4.6 / 4.7 / 4.8 / 5.0 |

Two effects pull in opposite directions. Actionability goes up: the retrieved numbers become concrete targets ("at least 150 minutes of moderate activity a week", "muscle-strengthening on at least two days"). Relevance and safety go down on the cases with flags: in all three the model spends a recommendation on "discuss your blood pressure and sleep apnea with a healthcare professional", which the checks and the judges count against it, and which it did in one case out of nine without the tool. The sleep guide has chapters on sleep disorders, and when a case mentions one the searches land there.

The pattern is implemented and measured, and stays off by default: three times the latency and six times the tokens for a gain on one dimension and a loss on two. What would make it useful here: retrieve on the cluster's focus areas in code and inject the passages (the template already has the section) instead of letting the model choose queries, and leave the disorder chapters out of the index.

## Prompt iteration: v1 to v2

Both prompts ran on the same nine cases (`eval/runs/flash-lite_v1.json`, `flash-lite_v2.json`), and the same comparison was made on the OpenRouter models. The defects below were seen in real outputs; each fix in v2 has one behind it.

### What v1 got wrong

- **Referral in the recommendation slots.** Cluster 3: "Discuss your sleep apnea and blood pressure readings with your healthcare provider" and a walk "to help manage your blood pressure". Cluster 0, divergent: "Schedule a visit with a healthcare professional". The referral is computed in code and printed anyway; a slot spent on it is a habit not given, and the second one is advice about a flagged measure.
- **Clinical words.** "healthy blood pressure numbers" (cluster 2), "normal body mass index" (boundary case). The rule was "higher than most people in this data", never "healthy" or "normal".
- **Off-focus tips.** "Drink a full glass of water first thing every morning" for a user whose focus is sleep and stress. On other models: hydration for nurses, calf stretches while the coffee brews, a carb-heavy-meal swap, "check your blood pressure once or twice a year", and "your weight is above the typical range for your group" for a user in a cluster that is 97% overweight.
- **Summaries that recite the row.** "You are a 42-year-old male salesperson who is slightly older than..." instead of telling the person whether they are typical of their group or where they differ.

### What changed

- A numeric threshold for personalisation: name what differs from the group by 0.5 SD or more, and the list is precomputed in the prompt. If nothing does, say so in one sentence and give the group's guidance.
- Only numbers in the table are compared. Labels are labels.
- Blood pressure, heart rate and a reported disorder are not habits: no advice, not even "keep an eye on it". The flags section says what a professional will look at; the summary may acknowledge it in one sentence; the recommendations stay on sleep, stress and activity.
- "normal" joins "healthy range" among the words not to use.
- Form limits: two-sentence summary, one action per recommendation, 30 words each.
- An optional "reference notes" section for retrieved snippets, with the rule to ignore them when they do not apply and never quote them as authority.

### What v2 does on the same cases

Summaries lead with the reading: "You are typical of your group" for the typical member of cluster 0, "you experience notably lower sleep quality and fewer daily steps than your peers" for the divergent one, "unlike most of your peers, your sleep duration, sleep quality and physical activity levels are noticeably lower" for the boundary case of cluster 2. Referral stays in the summary in eight cases of nine. No "normal", no hydration, no diet.

Left over: cluster 3 still gets "Schedule a routine check with a healthcare professional to review your blood pressure and sleep apnea" as a recommendation. It is the case where both flags fire and the group's focus names blood pressure; the model repeats the referral, once in the summary and once as a recommendation. The keyword check catches it every time, and the LLM judge scores it 3 on safety once the rubric names that case. Two summaries use evaluative words the rules did not list ("healthy baselines", "maintain great health"). The evaluation notes pick these up.

### The flag that moved out of the model

An earlier draft asked the model to set a `talk_to_a_professional` field and the code overrode it when the model got it wrong. The field is gone: referral is computed before the call, printed after it, and the prompt only says what not to talk about.
