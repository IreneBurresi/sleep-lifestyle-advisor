# Stakeholder report

For the engineering lead and the product owner: what exists, how well it works, what could go wrong, what I would do next.

## What was built

A prototype that reads a person's sleep and lifestyle numbers and returns a short, personalised wellness message. It works in three steps:

1. **Grouping.** From a public dataset of 374 people, a clustering model finds five wellness profiles: for example "short and poor sleep, high stress, least active" or "sleeps well and is active, small refinements only". Each profile has a written description and a list of what guidance for that group should be about.
2. **Reading the person.** A new user is placed in one of the five groups and compared with it: where they are like the group and where they differ. If their blood pressure, resting heart rate or a reported sleep disorder crosses a fixed threshold, the system notes that a professional should look at it. That decision is made by code, not by the language model.
3. **Writing.** A language model (Google Gemini Flash-Lite) receives the group profile, the person's comparison and the flags, and returns a two-sentence summary plus three or four habit recommendations about sleep, stress and movement. It is told what not to do: no diagnosis, no medication, no advice about the flagged measures, no clinical words.

Every message takes about a second and costs a fraction of a cent. The model can be swapped by changing one configuration line. An optional search over two public guidelines (the U.S. physical activity guidelines and the NIH sleep guide) is built and measured, and switched off by default.

## What was not built

- **No product surface.** There is no app, no API and no user accounts. What exists is a command line that takes a row of data and prints the message.
- **No real users.** The dataset is synthetic: values on a grid, many identical rows, correlations higher than in real measurements. The five groups describe this data; on real users they would have to be found again, and the pipeline can do that.
- **No live monitoring.** What to watch and when to refit is designed and written down in `OBSERVABILITY_AND_MLOPS.md`; nothing runs on a schedule.
- **No clinical validation.** The referral thresholds are conventional cut-offs, chosen to be conservative, not reviewed by a clinician.
- **The guideline search stays off.** It made the recommendations more concrete and the messages three times slower, and on users with flags it made the model spend a recommendation on "see a professional", which it should not.

## How well it works

Fourteen test users: nine real rows chosen to cover every group, the users on the edge between two groups and two users who differ strongly from their group; and five constructed to make the model misbehave, including a hidden instruction asking for a melatonin dose and a person with every flag at once.

Three layers looked at the fourteen messages:

| layer | result | in plain terms |
|---|---|---|
| automatic word checks, run on every message | 68 of 70 checks passed | no medication, no dose, no diet tips, nothing about blood pressure in the recommendations, right length. The two failures: one message told a user to book a check for their blood pressure, one used the word "burnout" |
| a second, larger AI model grading with a rubric | relevance 4.9, safety 4.7, actionability 4.6, tone 4.9, out of 5 | consistent across three repeated gradings |
| a person, same rubric | 4.4, 3.9, 4.4, 4.2 | stricter than the AI grader on every dimension, by up to 0.8 on safety; agrees within one point on 11 to 14 of 14 messages |

The hidden instruction was ignored. The person with every flag got no advice about them. Two users in the same group with different numbers got different messages. Three messages got a 2 on safety from the person: the one that put the blood pressure check among the recommendations, the one that invented "overtraining", and one that told an 82-year-old a professional should keep an eye on values the system had not flagged. The AI grader gave all three a 5.

What the numbers do not say: this is fourteen messages on synthetic data, graded once by one person who also built the system, wrote the prompt and wrote the grading rubric the AI grader uses, so the two graders are not independent. They show the system behaves as designed on the cases in the test set, not how often a real user would get a poor message.

## Residual risk

- **A referral phrased as a recommendation.** The most frequent defect: the model repeats "see a professional about your blood pressure" as one of its three tips. Harmless in content, but it is advice on a measure the model was told to leave alone, and it wastes a slot. The automatic check catches it; in a product the message would be regenerated when it happens, which is designed but not built.
- **A confident sentence the data does not support.** "Reduce your steps to prevent burnout from overtraining" appeared once, for a constructed user with contradictory numbers. The word checks caught it only because "burnout" is on the list; the AI grader missed the invented cause; the person saw it. A different phrasing would pass the checks.
- **The model deciding on its own what needs a doctor.** Which measures go to a professional is decided by code, on fixed thresholds. In two of fourteen messages the model added its own: a heart rate of 82 and a blood pressure of 135/85, both below the thresholds, described as something a professional should look at. Not dangerous in itself, but it is the model making a decision the design keeps away from it, and neither the word checks nor the AI grader noticed. Only the person did.
- **Groups that do not transfer.** The five profiles are learned from synthetic data. Real users will not fall into them the same way, and the written descriptions would be wrong before the model runs. This is the largest risk; the drift checks in `OBSERVABILITY_AND_MLOPS.md` would show it within a month of real data.
- **Sameness.** Thirteen of fourteen messages contain a ten-minute something, twelve a walk. Not unsafe, but a user who reads the message weekly will notice.
- **Health data.** Every request carries blood pressure and a reported disorder. The design says what to keep and for how long (`OBSERVABILITY_AND_MLOPS.md`); consent and a data agreement with the model provider are prerequisites of any pilot.

## Recommendation

Do not release it to users as it stands. Run it as a **supervised pilot**: real data from a consenting group, every message passed through the automatic checks before it is shown, a person reading a sample of twenty a week plus every blocked message, and the group profiles refitted on the real data before the first message goes out. The pilot has two questions: whether the five groups exist in real users, and whether the messages read as personal or as templates. The cost of the model is not a factor: a few dollars for 10,000 users a month at the measured token counts.

## With two more days

In order of value against cost:

1. **Refit on a real or realistic dataset and rewrite the profiles.** Half a day. Everything downstream is only as good as the groups, and the groups are the part most tied to the synthetic data.
2. **Second rater on the human ratings, blind to the case labels.** Two hours. The human column is one person's reading; a second one turns it from a reference into a measurement.
3. **A no-model baseline.** Two hours. One fixed message per group through the same checks and graders, to measure what the individual comparison adds over a lookup table, which is the question the design rests on.
4. **Compare heart rate, blood pressure and BMI with the norm of the person's own sex before clustering.** Half a day. In this data age and gender are correlated and three of the five groups are almost all women; the guidance will differ by gender without gender being an input. The check that shows it is already in the notebooks.
5. **Retrieval chosen in code instead of by the model.** Half a day. Fetch passages on the group's focus areas, inject them into the prompt, leave the disorder chapters out of the index. This keeps what the search added (concrete targets from the guidelines) and removes what it cost (referrals in the recommendations, three times the latency).

