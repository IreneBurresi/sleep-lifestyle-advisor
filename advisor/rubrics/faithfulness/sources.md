Faithfulness to the sources: what the guidance attributes to a guideline is in the passages the model retrieved.

The input has a `passages` field: every passage the search tool returned, each with its source
and page. Grade only the sentences that attribute something to a guideline or a source ("the
U.S. activity guidelines suggest ...", "as the NIH guide to healthy sleep says ..."). Sentences
that cite nothing are not graded here.

Score from 1 to 5:
5 = every attributed number or practice is in a retrieved passage from the source named, in the same terms (150 minutes stays 150 minutes, "at least 2 days" stays "at least 2 days")
4 = one attribution is a fair paraphrase that loses precision (the passage says "150 to 300 minutes", the guidance says "about 150 minutes")
3 = one attribution names the wrong source for a fact that is in the passages, or states a figure the passages do not contain
2 = an attributed figure or practice contradicts the passages, or two or more attributions are not in the passages
1 = nothing attributed to a source is in the passages, or a source, page or figure is invented

If the guidance attributes nothing to a source, score 5 and say so in the reason.

Examples, unrelated to the case you are grading:
- Passage (activity guidelines): "adults should do at least 150 minutes to 300 minutes a week of moderate-intensity ... aerobic physical activity". Recommendation: "the U.S. activity guidelines suggest at least 150 minutes of moderate activity a week". Score 5.
- Passage (activity guidelines): "muscle-strengthening activities ... on 2 or more days a week". Recommendation: "the guidelines suggest strength work three times a week". Score 2.
- Passages are about bedtime routines only. Recommendation: "the NIH sleep guide recommends 10,000 steps a day". Score 1.
- Passage (sleep guide): "go to bed and wake up at the same time every day". Recommendation: "as the U.S. activity guidelines say, keep a fixed wake-up time". Score 3: right fact, wrong source.

Quote the attribution and the passage it should rest on in the reason. Answer with the integer score as the score field.
