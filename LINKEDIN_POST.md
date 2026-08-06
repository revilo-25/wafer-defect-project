I built a semiconductor manufacturing data science project on the real WM-811K wafer defect dataset (811,457 wafer maps from actual fabs) — going past "trained a CNN, got good accuracy" into three things fab data teams actually deal with.

𝗧𝗵𝗲 𝗶𝗺𝗯𝗮𝗹𝗮𝗻𝗰𝗲 𝗽𝗿𝗼𝗯𝗹𝗲𝗺, 𝗶𝗻 𝗼𝗻𝗲 𝗻𝘂𝗺𝗯𝗲𝗿:
785,936 wafers have no defect. Near-full defects — one of the 8 classes — appear only 149 times in the entire dataset. Naive random sampling for training would barely see the rare classes at all, so I built a stratified sampling strategy that keeps every single labeled defect wafer and only subsamples the dominant "None" class.

𝗪𝗵𝗮𝘁 𝗜 𝗯𝘂𝗶𝗹𝘁 — 𝗳𝗼𝘂𝗿 𝗰𝗼𝗻𝗻𝗲𝗰𝘁𝗲𝗱 𝗽𝗶𝗲𝗰𝗲𝘀:

𝟭. 𝗕𝗮𝘀𝗲𝗹𝗶𝗻𝗲 𝗖𝗡𝗡 𝗰𝗹𝗮𝘀𝘀𝗶𝗳𝗶𝗲𝗿 (macro F1 = 0.74) with class-weighted loss. Edge-Ring hit F1=0.96, Center 0.87 — classes with distinctive large-scale shapes. Scratch bottomed out at F1=0.21 — thin, low-contrast line defects are genuinely hard for a plain CNN at low resolution, and the confusion matrix shows exactly why: it gets confused with Local and misclassified as None more than any other class.

𝟮. 𝗠𝘂𝗹𝘁𝗶-𝗹𝗮𝗯𝗲𝗹 𝗿𝗲𝗳𝗿𝗮𝗺𝗶𝗻𝗴 + 𝗳𝗼𝗰𝗮𝗹 𝗹𝗼𝘀𝘀 — because real wafers can show compound defects (edge-ring AND scratch on the same wafer), and single-label classification throws that signal away entirely.

𝟯. 𝗔𝘂𝘁𝗼𝗲𝗻𝗰𝗼𝗱𝗲𝗿-𝗯𝗮𝘀𝗲𝗱 𝗻𝗼𝘃𝗲𝗹𝘁𝘆 𝗱𝗲𝘁𝗲𝗰𝘁𝗶𝗼𝗻 (ROC-AUC = 0.78) — trained only on normal wafers, flags anomalies including defect types never in the training labels. A supervised classifier can only recognize what it's been shown; this catches what it hasn't.

𝟰. 𝗔 𝗰𝗮𝘂𝘀𝗮𝗹 𝗰𝗼𝗻𝗳𝗼𝘂𝗻𝗱𝗶𝗻𝗴 𝗱𝗲𝗺𝗼 — the part I think matters most and is missing from almost every ML portfolio: a synthetic fab dataset with a known ground-truth cause, showing how naive correlation flags the WRONG variable (a decoy sensor, r=0.60, p<0.001) while the TRUE cause is nearly invisible to plain correlation. Stratifying by the confounder recovers the correct answer. This is the gap between "what correlates with yield loss" and "what would we still see if we intervened" — exactly the kind of mistake that costs fabs real money when engineers chase the wrong sensor.

𝗢𝗻𝗲 𝗳𝗶𝗻𝗱𝗶𝗻𝗴 𝗜 𝗱𝗶𝗱𝗻'𝘁 𝗲𝘅𝗽𝗲𝗰𝘁: multi-label Scratch detection dropped to F1=0.000 even though single-label Scratch scored 0.21-0.27 across runs — a specific, diagnosable threshold issue with focal loss on an already-ambiguous class, not just "the model got worse." Debugging that taught me more about loss function behavior than a clean result would have.

Code + full writeup on GitHub: [link]

#DataScience #Semiconductors #MachineLearning #Manufacturing
