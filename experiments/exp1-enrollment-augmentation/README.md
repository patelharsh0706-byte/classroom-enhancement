# exp1 — Enrollment Augmentation

**Started:** 2026-08-19
**Status:** Phase 0 (baseline)

## The defect

Student voiceprints were built from clean, close-mic phone clips. The lecture is
far-field room audio. ECAPA-TDNN (VoxCeleb-trained, near-field) cannot bridge
that domain gap.

Evidence from the 2026-08-16 pilot run (`pilot_15min.wav`, 164 turns):

| Result | Count |
|---|---|
| Matched to **Prof** | 82 turns, confidence 0.82–0.87 |
| Matched to a **student** | 0 |
| **Unknown** | 82 turns, confidence 0.17–0.42 |

The professor matches because his enrollment conditions happen to resemble his
lecture conditions. No student clears the 0.7 threshold; the highest student
similarity anywhere in the recording was 0.418.

## The hypothesis

Degrade the clean enrollment clips until they sound like the room — room impulse
response, background noise at matched SNR, attenuation, band-limiting — then
build voiceprints from the degraded audio. This converts a cross-channel
verification problem (FFSVC-style, 6–7% EER baselines) into a same-channel one,
where ECAPA is strong.

## Honest prior

**10–15%** likelihood of producing usable per-student scoring.

Established science: FFSVC 2020/2022 benchmark this exact setup (close-talk
cellphone enrollment vs far-field test) at 6.27–7.18% EER, versus ~0.92% for the
same model class on clean matched audio — a 7x degradation on a problem with a
dedicated challenge series that remains unsolved. Classroom-specific numbers
(Demszky group, Stanford, EDM 2024/2025): ~17% DER teacher-vs-student, ~34–45%
student-vs-student.

This setup is *worse* than those papers on five axes: single mic (no array/
beamforming), no labeled adaptation data, open conversation (not text-dependent),
open set (unenrolled speakers in the room), and many sub-second utterances.

The competing approach — roll-call enrollment through the room mic, paired to the
phone clips by Hungarian assignment — is estimated at **40–50%** for less compute.
exp1 runs first because it is cheap and it settles definitively whether the phone
clips are usable at all.

## Arms

| Arm | Description |
|---|---|
| `baseline` | Current voiceprints, no modification — the number to beat |
| `room` | Augmented with background noise sampled from the lecture's own quiet stretches |
| `synthetic` | Augmented with `AddColoredNoise` (generic hiss/hum) |
| `+centering` | Domain-shift subtraction applied to any of the above |
| `+asnorm` | Adaptive score normalization applied to any of the above |

`room` vs `synthetic` is a deliberate control: if room-sampled noise beats
generic, the domain-matching thesis has legs. If they are identical, the
augmentation route is dead and roll-call is the answer.

## Phases

| Phase | Script | Writes |
|---|---|---|
| 0 | `p0_baseline.py` | `results/baseline.json` |
| 1 | `p1_noise_bank.py` | `noise_bank/` |
| 2 | `p2_augment.py` | `audio_aug/`, `voiceprints/` |
| 3 | `p3_normalize.py` | `results/scores_*.json` |
| 4 | `p4_compare.py` | `results/comparison.md`, `results/listen_list.md` |

## Success criterion

Not "a score went up." A student must clear the **impostor floor** by a real
margin — the professor's 0.58 margin over his runner-up is the reference shape.

Because there is no ground truth, the winning arm's top predictions require a
**listening check** before any result is believed. Without that, this experiment
produces numbers, not an answer.

## Read-only inputs

Nothing outside this directory is modified.

- `audio/enrollment/*.wav` — source clips
- `audio/lectures/pilot_15min.wav` — noise extraction + evaluation
- `data/turns.json`, `data/named_turns.json` — cluster segments
- `data/voiceprints.json` — baseline embeddings (**29 speakers as of 2026-08-30**:
  28 students + Prof; was 15)
- `pipeline/common.py`, `match.py` — imported, never edited

## Log

- **2026-08-19** — Pre-step: Clive processed. He was in the roster but had no
  audio, so enrollment silently produced 14 voiceprints for a 14-student roster.
  His `_originals` file is **corrupt** (decodes to 0 samples — the likely reason
  he was skipped on 2026-08-16); a re-export dated 2026-08-18 decodes fine and was
  converted to `clive_1.wav` (pcm_s16le/16k/mono/30.29s, matching the other 14).
  Enrollment rerun → 15 voiceprints. Prior file backed up to `voiceprints.json.bak`.
  Caveat: Clive's clip is phone audio like the rest, so he inherits the same domain
  mismatch — this makes the set complete, not better.

## Result

_(appended after Phase 4)_

- **2026-08-30** — Roster completed. A newer Google Form export
  (`File upload (Voice responses)…-2`) contained 28 submissions vs the 14 held
  previously. The 14 missing students were converted and validated (all decode
  clean, 13.6–39.1s, RMS 0.014–0.170); enrollment rerun → **29 voiceprints**
  (28 students + Prof).

  **This moved the impostor floor and made the task harder**, which must be
  carried into every comparison below:

  | Cohort | pairs | mean | p95 | p99 | **max** |
  |---|---|---|---|---|---|
  | 13 students | 78 | 0.129 | 0.352 | — | **0.444** |
  | 28 students | 378 | 0.133 | 0.374 | 0.458 | **0.536** |

  The mean is flat but the tail grew: more enrolled speakers means more chances
  for a spurious high score. Six *different-person* pairs now score above the
  best student match ever seen in the pilot (0.401) — the worst being
  Andrea/Xing at 0.536 and Yihong/Yulu at 0.529. These cluster among the
  Mandarin-speaking students, consistent with ECAPA's English-heavy VoxCeleb
  training. Any pre-2026-08-30 threshold reasoning is invalid at this cohort size.

## 2026-08-30 — Segment 54:00–56:00: first confirmed student identifications

Ad-hoc analysis outside the phase plan, run on a 2-minute window of
`lecture_01.wav` (not the pilot slice). Diarization found 3 speakers / 24
segments. Matched against all 29 **unaugmented** voiceprints — i.e. this is the
`baseline` arm, on different audio.

| Cluster | Speech | Top-1 | Top-2 | Margin | vs impostor max (0.536) |
|---|---|---|---|---|---|
| SPEAKER_00 | 41s | **Prof 0.831** | Preethi 0.331 | +0.500 | above |
| SPEAKER_02 | 33s | **Clive 0.654** | Justin 0.190 | +0.464 | above |
| SPEAKER_01 | 41s | **Jayden 0.541** | Justin 0.390 | +0.152 | above |

**User confirmed these identifications as accurate** (by ear / knowledge of the
class, 2026-08-30). Not a systematic evaluation — n=3 speakers, one window — but
it is the first ground truth this project has had, and it is positive.

### Why SPEAKER_01's thin margin is not ambiguity

Split into its two independent sub-segments, both rank Jayden first:

| Window | Duration | Top-1 | Top-2 |
|---|---|---|---|
| 54:56–55:30 | 34.1s | Jayden 0.551 | Justin 0.410 |
| 55:34–55:41 | 7.2s | Jayden 0.449 | Justin 0.292 |

So it is one speaker, not a merged cluster. Justin's elevated score is
**leakage**: the Jayden/Justin *enrollment* voiceprints score 0.421 against each
other (above impostor p95), so they are an intrinsically confusable pair. A
runner-up that is a known-confusable partner of the winner is not an independent
competing hypothesis.

Incidental but valuable: the same speaker loses **0.10 similarity** going from a
34.1s segment to a 7.2s one. That is the short-utterance penalty measured
directly in this project's own data, under otherwise identical conditions.

### Hypothesis revision — this weakens exp1's premise

exp1 assumes the phone clips are unusable because of channel mismatch. **Three
speakers were just identified from those same unaugmented phone clips against
that same room audio.** The domain gap is therefore *bridgeable*, and the pilot's
total failure has a better explanation than channel:

| | Pilot (0 students found) | Segment 54–56 (2 students found) |
|---|---|---|
| Cluster speech | 7–68s, many sub-second turns | 33–41s, long turns |
| Content | interjections ("Yeah.", "Oh.", "Okay.") | sustained argument |
| Clive enrolled? | **no** — corrupt file | yes |

At least one pilot Unknown was Unknown because the speaker was not in the
database at all. The rest is consistent with **utterance duration**, not channel,
being the dominant term.

**Implication for the arms:** augmentation targets channel. If duration is the
real bottleneck, `room` vs `synthetic` may both fail while the true fix is
turn-level aggregation (pool each speaker's turns across the whole lecture before
embedding) or simply declining to score short turns. The `room`/`synthetic`
control still settles the channel question cleanly, so exp1 remains worth
running — but its **10–15% prior should now be read as a prior on the wrong
variable**, and a duration-focused arm deserves to exist.

### Suggested additional arm

| Arm | Description |
|---|---|
| `mindur` | Score only clusters with ≥20s of pooled speech; leave the rest Unknown |

Cheap, needs no augmentation, and is directly supported by the 0.10-per-27s
degradation measured above.

## 2026-08-31 — Ground truth: confirmed TRUE NEGATIVE (open-set failure)

Window 1:05:30–1:08:00, cluster SPEAKER_02 (32s, 9 turns), scored
**Richard 0.143 / margin +0.006** → below impostor floor → declined.

**User identified the actual speaker as STUDENT-Z, who has no
voiceprint** (she did not submit to the Google Form; the 28-student roster is
therefore incomplete relative to the real class).

### Why this matters more than the true positives

The correct answer did not exist in the database. There was no threshold, model,
or augmentation that could have produced it. The system's low score and refusal
to name was the **correct behaviour**, and it is the first direct confirmation
that the impostor-floor gate separates "I don't know" from a real match.

Running tally of confirmed ground truth:

| Window | Cluster | System output | Truth | Correct? |
|---|---|---|---|---|
| 54–56 | SPEAKER_00 | Prof 0.831 | Prof | ✅ |
| 54–56 | SPEAKER_02 | Clive 0.654 | Clive | ✅ |
| 54–56 | SPEAKER_01 | Jayden 0.541 | Jayden | ✅ |
| 1:05–1:08 | SPEAKER_04 | Prof 0.753 | Prof | ✅ |
| 1:05–1:08 | SPEAKER_00 | Saieshwar 0.624 | Saieshwar | ✅ |
| 1:05–1:08 | SPEAKER_02 | *declined* (0.143) | Student-Z (**unenrolled**) | ✅ (true negative) |

6/6 correct decisions so far — 5 identifications above the floor, 1 correct
refusal below it. Small n, one recording, but zero false positives to date.

### The real defect this exposes

**The class is larger than the enrolled set.** Every unenrolled student is a
guaranteed miss, and with "best guess" display enabled they are actively
mislabelled with an enrolled student's name. This is a data-collection problem,
not a modelling one, and no arm of exp1 addresses it.

Two consequences for the design:

1. **Open-set is the operating condition, not an edge case.** Any evaluation that
   assumes every speaker is enrolled will overstate accuracy. The low-score
   population contains both "enrolled but hard" and "not enrolled at all", and
   these are indistinguishable from score alone.
2. **"Best guess" labels are unsafe below the floor.** A sub-floor score may mean
   the true speaker is absent from the database entirely. Best-guess names must
   never be exported, scored, or shown without that caveat attached.

### Revised interpretation of the pilot

The 2026-08-16 pilot's 82 Unknown turns were previously attributed to channel
mismatch. Two confirmed causes now compete:
- speaker not enrolled (Clive then; Student-Z now — **confirmed**)
- short/fragmented utterances (**measured**: 0.10 similarity lost, 34.1s → 7.2s)

Channel mismatch remains **unconfirmed as a cause of any specific failure**, and
five students were identified from unaugmented phone clips. exp1's premise is
weaker than when it was written.

### Actions

- [ ] Collect voice recording from Student-Z
- [ ] Reconcile Form submissions (28) against the real class roster / attendance
      workbook — find every student with no voiceprint
- [ ] Add persistent "Unknown Speaker A/B/C" identities so unenrolled students'
      participation is still counted and can be named retroactively

## 2026-08-31 — Set 2 (1:05:30–1:08:00) fully verified — THRESHOLD WAS WRONG

User confirmed **every** identification in this window, including the two that
the impostor-floor gate had rejected as unreliable. Full verified timeline:

| Time | Speaker | Source |
|---|---|---|
| 1:05:30–1:05:36 | Prof | system, confirmed |
| 1:05:37–1:06:01 | Saieshwar | system, confirmed |
| 1:05:39–1:05:40 | Prof | system, confirmed |
| 1:06:01–1:06:35 | **Student-Z** | **user only — unenrolled, system declined** |
| 1:06:35–1:06:59 | Prof | system, confirmed |
| 1:06:59–1:07:21 | **Kaiting** | system @ 0.340 — **was rejected**, confirmed correct |
| 1:07:35–1:07:59 | **Rija** | system @ 0.225 — **was rejected**, confirmed correct |

Set 1 (54:00–56:00) for comparison: Prof / Clive / Jayden, all confirmed, all
above the old floor.

### The methodological error

The 0.536 gate was derived from similarities **between enrollment voiceprints**
— phone vs phone, *same channel*. But the operational comparison is
cluster-vs-enrollment, which is *cross-channel*, and cross-channel comparison
compresses every score downward. Same-channel null applied to cross-channel
scores = apples to oranges. The gate discarded correct answers.

The correct null is **per-cluster**: score the cluster against all candidates and
measure how far the top one departs from that cluster's own score distribution
(AS-Norm, as flagged in the original arms table but never applied).

### Full result against ground truth (n=8, zero errors)

| Cluster | Top-1 | raw | margin | z | Truth | |
|---|---|---|---|---|---|---|
| 54–56 / S00 | Prof | 0.831 | 0.500 | 8.56 | Prof | ✅ |
| 54–56 / S02 | Clive | 0.654 | 0.464 | 8.31 | Clive | ✅ |
| 54–56 / S01 | Jayden | 0.541 | 0.152 | 6.07 | Jayden | ✅ |
| 1:05 / S04 | Prof | 0.753 | 0.366 | 6.87 | Prof | ✅ |
| 1:05 / S00 | Saieshwar | 0.624 | 0.358 | 7.29 | Saieshwar | ✅ |
| 1:05 / S03 | Kaiting | 0.340 | 0.026 | 2.44 | Kaiting | ✅ |
| 1:05 / S01 | Rija | 0.225 | 0.012 | 2.20 | Rija | ✅ |
| 1:05 / S02 | Richard | 0.143 | 0.006 | 1.97 | **unenrolled** | ✅ declined |

```
enrolled & correct : raw 0.225–0.831    z 2.20–8.56
unenrolled         : raw 0.143          z 1.97
```

**8/8 correct decisions. Zero false positives, zero false negatives.**

### Revised decision rule (provisional)

| Old | New |
|---|---|
| accept if raw ≥ 0.536 (enrollment-pair impostor max) | accept top-1 if raw ≳ 0.18 **or** z ≳ 2.1 |
| margin used as confidence signal | **margin is useless** — Rija correct at margin 0.012 |

Caveats, stated plainly:
- The boundary rests on **one** negative example (z 1.97 vs 2.20 — a 0.23 gap).
  One more unenrolled speaker could land above it. This is a hypothesis, not a
  calibrated threshold.
- n=8, one recording, discussion segments with 22–41s clusters. Says nothing yet
  about sub-second interjections.
- Set-2 clusters that were correct at low raw scores (Kaiting 22s/1 turn, Rija
  25s/4 turns) had *ample* duration. Low raw score ≠ short utterance here.

### Impact on exp1

The pilot's "0 students identified" was **largely a gating artefact**, not a
model failure — compounded by missing enrollments (Clive corrupt, Student-Z and
others never submitted). Under the corrected rule, the pilot's clusters scoring
0.17–0.42 were plausibly mostly correct and were thrown away.

exp1's premise (channel mismatch makes phone clips unusable) is now **contradicted
by direct evidence**: seven speakers identified from unaugmented phone clips
against room audio, one at raw 0.225. Augmentation is no longer the obvious first
move.

**Recommended reprioritisation:**
1. Re-run the 2026-08-16 pilot under the corrected rule, sample the output, get
   user ground truth → convert n=8 into a real accuracy figure. Zero new compute
   on enrollment, highest information gain.
2. Close the enrollment gap (Student-Z + roster reconciliation) — bounds the
   open-set error rate, which is currently the only confirmed failure mode.
3. Collect more unenrolled examples to actually calibrate the reject boundary.
4. exp1 augmentation arms — **demoted**. Run only if 1–3 leave a channel-shaped
   residual.
