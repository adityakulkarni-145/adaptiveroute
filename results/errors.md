# Error analysis: most confident wrong small-model predictions

Source: SST-2 validation second half. Confidence is the calibrated small-model confidence.

| # | Text | True | Small (conf) | Large | Negation | Contrast |
|---|---|---|---|---|---|---|
| 1 | although huppert 's intensity and focus has a raw exhilaration about it , the piano teacher is anything but fun .  | negative | positive (0.995) | positive |  | yes |
| 2 | drops you into a dizzying , volatile , pressure-cooker of a situation that quickly snowballs out of control , while focusing on the what much more than the why .  | positive | negative (0.992) | negative |  |  |
| 3 | a working class `` us vs. them '' opera that leaves no heartstring untugged and no liberal cause unplundered .  | positive | negative (0.988) | negative | yes |  |
| 4 | ... routine , harmless diversion and little else .  | positive | negative (0.987) | negative |  |  |
| 5 | do n't be fooled by the impressive cast list - eye see you is pure junk .  | negative | positive (0.977) | negative | yes |  |
| 6 | whaley 's determination to immerse you in sheer , unrelenting wretchedness is exhausting .  | negative | positive (0.976) | negative |  |  |
| 7 | care deftly captures the wonder and menace of growing up , but he never really embraces the joy of fuhrman 's destructive escapism or the grace-in-rebellion found by his characters .  | negative | positive (0.956) | negative | yes | yes |
| 8 | it 's about following your dreams , no matter what your parents think .  | positive | negative (0.953) | negative | yes |  |
| 9 | a hamfisted romantic comedy that makes our girl the hapless facilitator of an extended cheap shot across the mason-dixon line .  | negative | positive (0.948) | negative |  |  |
| 10 | a taut psychological thriller that does n't waste a moment of its two-hour running time .  | positive | negative (0.948) | positive | yes |  |
| 11 | the inspirational screenplay by mike rich covers a lot of ground , perhaps too much , but ties things together , neatly , by the end .  | positive | negative (0.944) | positive |  | yes |
| 12 | writer/director joe carnahan 's grimy crime drama is a manual of precinct cliches , but it moves fast enough to cover its clunky dialogue and lapses in logic .  | positive | negative (0.942) | negative |  | yes |
| 13 | a disappointment for those who love alternate versions of the bard , particularly ones that involve deep fryers and hamburgers .  | negative | positive (0.926) | negative |  |  |
| 14 | fancy a real downer ?  | negative | positive (0.920) | negative |  |  |
| 15 | i can take infantile humor ... but this is the sort of infantile that makes you wonder about changing the director and writer 's diapers .  | negative | positive (0.920) | negative |  | yes |

## Measured summary

- Small-model errors on this split: 89 of 436.
- Of these 15 most confident errors: 5 contain a negation word, 5 contain but/however/although.
- The large model gets 9 of 15 right.
- The tuned fixed router (threshold 0.94) escalates 3 of 15; any not escalated are answered wrongly by the small model with high confidence.
## Pattern (manual reading of the 15 rows above)

- Contrast clauses where the final clause decides the sentiment (rows 1, 7, 11, 12): bag-of-words weighs both halves equally.
- Negation that flips a polar word (rows 5, 10: "don't be fooled by the impressive cast", "doesn't waste a moment").
- Negative sentiment carried by words that look positive or neutral in isolation, plus irony (rows 6, 9, 13, 14: "unrelenting wretchedness", "fancy a real downer?").
- Some gold labels are debatable (rows 2, 4, 12, where the large model also disagrees with the label).
