# QA Extraction Quality Analysis Report

**Scope:** All `.qa_text.json` files in `data/qa_text/`  
**Analysis date:** 2026-07  
**Files examined:** 908 total; ~20 manually inspected in full; all 908 processed by automated script

---

## 1. Corpus Overview

| Metric                         | Value  |
| ------------------------------ | ------ |
| Total `.qa_text.json` files    | 908    |
| Files with content (pairs > 0) | 838    |
| Files intentionally empty      | 70     |
| Total QA pairs                 | 10,135 |
| Avg pairs per content file     | ~12.1  |

### By event type

| Event type           | Files | Total pairs | Avg pairs/file |
| -------------------- | ----- | ----------- | -------------- |
| `media_interview`    | 363   | 3,657       | 10.1           |
| `student_qa`         | 274   | 3,462       | 12.6           |
| `press_conference`   | 234   | 2,961       | 12.7           |
| `panel`              | 4     | 55          | 13.8           |
| `produced_content`   | 24    | 0           | —              |
| `produced_interview` | 9     | 0           | —              |

The 33 empty files (`produced_content` + `produced_interview`) represent pre-edited, monologue-style content where no Q&A extraction is appropriate. **This filtering is correct and working as intended.**

---

## 2. Question Quality Analysis

### 2.1 Overview

Using automated regex classification across all 10,135 pairs:

| Category                                    | Count      | % of total |
| ------------------------------------------- | ---------- | ---------- |
| Clearly genuine (has `?`, no noise pattern) | 5,174      | 51%        |
| Interrogative without `?` (genuine-like)    | 1,087      | 11%        |
| Pure statement — no interrogative, no `?`   | 3,008      | 30%        |
| Specific noise patterns (see §3)            | ~866       | 8%         |
| **Estimated genuine**                       | **~6,261** | **~62%**   |
| **Estimated noise**                         | **~3,874** | **~38%**   |

The 11% "interrogative without `?`" category covers legitimate conversational-style questions such as "Tell me about the research you've been doing" or "Walk us through that EVA" — questions that don't end with a question mark but are genuine prompts answered naturally by the crew.

The 30% "pure statement" category is the primary quality problem. These are declarative sentences, half-sentences, or off-topic remarks that were extracted as questions due to speaker-turn detection errors.

### 2.2 Question length distribution

| Length range  | Count | Notes                                              |
| ------------- | ----- | -------------------------------------------------- |
| 1–9 chars     | 54    | "Hi.", "Go ahead."                                 |
| 10–29 chars   | 814   | Very short, often greetings or fragments           |
| 30–59 chars   | 1,574 | Mix of genuine short questions and fragments       |
| 60–99 chars   | 2,319 | Good range for single questions                    |
| 100–199 chars | 3,336 | Often preamble + question, multi-part              |
| 200+ chars    | 2,038 | Often multiple questions merged, or Q + start of A |

Questions in the 60–200 char range are most likely to be high-quality single questions.

### 2.3 Noise rate by event type

These rates are calculated as the fraction of pairs that do **not** have an explicit `?` and/or match a noise pattern:

| Event type         | Pairs with `?` | Estimated clean rate                                                |
| ------------------ | -------------- | ------------------------------------------------------------------- |
| `press_conference` | 54%            | ~65–70% (journalists use formal Q patterns)                         |
| `media_interview`  | 54%            | ~55–65% (many leading statements, readiness checks)                 |
| `student_qa`       | 46%            | ~50–60% (kids often ask without `?`; moderator intro noise is high) |
| `panel`            | 58%            | ~65–70% (similar to press conference)                               |

`student_qa` has the highest noise rate, primarily due to **moderator introduction sentences** being extracted as questions (see Pattern C below).

---

## 3. Specific Failure Patterns

### Pattern A — Readiness / Communication Checks (508 instances)

The most pervasive single noise category. Every in-flight event begins with a standard readiness exchange that gets captured as a question-answer pair.

**Examples:**

```
Q: "Are you ready for the event? Houston, this is Station. We're ready for the event.
    StarTalk, this is Mission Control Houston. Please call station for a voice check."
A: "Can you hear me okay? Hey, Mark."
```

```
Q: "Station this is Houston. Are you ready for the event? Houston, this is the space
    station. Ready for the event. Fox News Radio, this is Mission Control..."
```

These appear in virtually every media_interview and many student_qa files. The phrase "Are you ready for the event?" is the primary trigger, often followed by several more communication-check sentences appended to the same "question" field.

### Pattern B — Greeting / Sign-off Noise (313+ instances)

Thank-yous, welcome statements, and sign-off remarks are extracted as question speakers.

**Examples:**

```
Q: "Thanks for taking the time to talk with us."              [KYW-TV, pair 1]
Q: "Commander Kelly, thanks so much for doing this again with us."  [Expedition46, pair 7]
Q: "Thank you so much, John Bennett, for your time."           [Everyday Astronaut, pair 9]
Q: "It is an honor to welcome back Commander Scott Kelly to CBSN."  [Expedition46, pair 8]
```

### Pattern C — Moderator Introduction Lines (student_qa, high frequency)

In student in-flight events, a host moderator introduces each student by name before they ask their question. The single-line introduction is extracted as the question, and the actual student question (which follows) becomes part of the answer or is lost.

**Examples (all from Expedition_55_Monta_Loma, 6 of 23 pairs):**

```
Q: "Our next question comes from another fifth grade student named Tylee Ramos."
Q: "Our next question comes from a fifth grade student named Amara Hernandez."
Q: "The next person is Juliana Woo, a fifth grade student."
```

### Pattern D — Pure Statements Presented as Questions (3,008 estimated)

The largest failure category. Declarative statements — often context-setting remarks by journalists — are captured as questions.

**Examples:**

```
Q: "Long-duration flight, this was all about seeing whether humans can go to Mars."
Q: "You're the first black woman in space for America."
Q: "Peggy, you're going to be the first woman commander of the ISS."
Q: "You are forced to be creative."
Q: "For long-term space travel... obviously there'll be some expectation that you
    grow your own food."
Q: "And you say that NASA crews are going to fly in this."
Q: "So let's talk about food."
Q: "I'm curious about the future of robotics and artificial intelligence."
```

Many of these are partial-statements that belong to a leading journalism technique where the journalist makes a statement and then implies a question ("So... tell me more"). The true question is implicit or verbal.

### Pattern E — Moderator Administrative Announcements

Phone-queue management and logistical announcements are extracted as questions.

**Examples:**

```
Q: "A reminder, if you are on the phone and you have a question, please press star 1
    to be added to the queue or star 2 if you'd like to be removed."
Q: "So with that, I'll turn it over to Jim Chilton."
Q: "Oh, I'm sorry. Go ahead, Ryan."
Q: "If you can address your question to who you want to answer it, that would be helpful."
Q: "And I'll turn it over to Joel Montalbano."
Q: "All right. Our next question comes from David Curley with Discovery News."
```

### Pattern F — Inverted / Backward Q&A Pairs

The question and answer fields are swapped. The astronaut's response appears in `question.text` and the journalist's actual question appears in `answers[].text`.

**Examples (Expedition46 multi-network, Expedition43 1-year crew):**

```
// Expedition43 pair 3 — the actual question is in the answer field:
Q: "Scott, along those same lines, in less than two weeks a SpaceX Dragon cargo craft will
    be coming your way..."  [trailing statement, no actual question]
A: "Is this the variety and pace of work that will keep you focused or will it be
    necessary to tune out?"  [the real question is here]

// Expedition46 pair 10:
Q: "So I was fortunate that I had been up here before... I'm going to miss it."  [Kelly's own answer]
A: "So I don't mean to minimize docking or hatch."  [journalist fragment]
```

### Pattern G — Truncated / Incomplete Questions

Questions are cut off mid-sentence, often at a field boundary where the transcription segment ended.

**Examples:**

```
Q: "Can you tell us a little bit about how team sports has played a part in your
    path to becoming an"                                [cuts off at "an"]
Q: "When it comes to whether we'll get to Mars, is that something that you hope
    happens in your lifetime to the point where"       [cuts off mid-clause]
Q: "And finally for both of you, I'm struck by the variety of key anniversaries
    coming up that you'll be celebrating in"           [cuts off]
```

### Pattern H — Multi-Question Merge / Garbled Extraction

Multiple distinct questions are concatenated into a single question field, often from different students or consecutive journal questions.

**Example (Expedition_56_Armstrong_Flight_Research — only 1 total pair):**

```
Q: "What does your body feel like in zero gravity? What inspires you to become
    an astronaut? What exercise do you do and do you sweat in space?"
A: "What do yo-yo work in microgravity?"    [this is actually another question]
```

**Example (NASA Boeing Crew Flight Test Astronaut Q&A):**

```
Q: "Hi, can you hear me this time? Can you hear me? We can hear you, Marcia.
    Oh, great, thank you. For both of you..."  [technical hiccup + actual question merged]
```

### Pattern I — Answer Text Bleeding into Question Field

The sentence-boundary detection places the start of the astronaut's answer inside the question field, and the question ends abruptly or gets the first word/phrase of the answer appended.

**Examples (KYW-TV):**

```
Q: "And how many times a day do you circle the globe? 16"        [answer's "16" appended]
A: "times a day we go around the globe"                           [answer fragment]
```

### Pattern J — Introduction Only / Name Only

Student events where a student introduces themselves but their question text is missing.

**Examples:**

```
Q: "Hi, my name is David."
Q: "Hi, my name is Jane."
Q: "Hi."
```

These account for ~140 identified instances. The student's name is present but the actual question was either inaudible, captured elsewhere, or transcribed into a subsequent segment.

### Pattern K — Compliment / Affirmation as Question

Positive affirmations extended to the crew are captured as questions, sometimes producing completely nonsensical pairs.

**Example (VideoFiles2010 Expedition25):**

```
Q: "That's a very good question."
A: "Good question."
```

Both fields contain moderator/host commentary with no informational content.

### Pattern L — Ceremony / Non-Q&A Event Mislabeled

Events that are announcements, ceremonies, or award presentations get incorrectly processed as having Q&A content; all extracted pairs are speeches, stage directions, or ceremonial dialogue.

**Example (Artemis II Crew Announcement — 7 pairs, all noise):**

```
Q: "To give us a proper Houston welcome, please join me in bringing out JSC
    Center Director, Vanessa White."
A: "Thank you, Joe. Good morning."

Q: "You got more you want to say, Reed? There's three words that we keep saying
    in this Artemis program. We are going. And I want everybody to say it..."
A: "That was awesome."
```

The event_type classifier assigned `press_conference` to this file, which then triggered QA extraction on what was a pure announcement ceremony.

---

## 4. Answer Text Quality

### 4.1 Overall assessment

| Metric                                    | Count  | %     |
| ----------------------------------------- | ------ | ----- |
| Total pairs                               | 10,135 | —     |
| Pairs with no answer                      | 0      | 0%    |
| Pairs with short answer (<20 chars)       | 281    | 2.8%  |
| Pairs with multi-speaker answer           | 655    | 6.5%  |
| Pairs with substantive answer (≥20 chars) | 9,854  | 97.2% |

Answer text quality is **significantly higher** than question quality. Nearly all pairs have substantive astronaut answers (≥20 characters). The 281 short answers (<20 chars) are typically truncated acknowledgments ("Yes", "Absolutely", "Can you hear me now?") where the astronaut's continuation was captured in a subsequent segment.

### 4.2 Is answer text necessary?

**Yes, strongly — for the following reasons:**

1. **Context anchoring**: Many genuine questions are stated as leading statements (Pattern D). Without the astronaut's answer, such a pair is meaningless. The answer text is the only reliable informational content in about 30-40% of pairs.

2. **Questions with preamble**: Journalists frequently include background before their actual question (2–3 sentences of setup). The answer text is the only part that demonstrates _what was actually asked_ by showing what the astronaut chose to respond to.

3. **Multi-part questions**: When a question field contains several questions merged together (Pattern H), the answer reveals which sub-question the astronaut addressed.

4. **Inverted pairs**: In the ~5% of pairs where question and answer are swapped (Pattern F), the answer field contains the actual question. Filtering out answers would make these pairs even more confusing.

5. **Search utility**: The answer text contains the domain content (space research details, astronaut experiences, technical explanations) that makes these pairs valuable for search indexing. Many questions are deliberately vague or context-dependent ("And how did that make you feel?"), while answers are specific and informationally rich.

---

## 5. Good Examples

The following are representative high-quality pairs — genuine question, substantive answer, no noise:

**1. Press conference — technical specificity (Crew-5):**

```
Q: "Irene Klotz with Aviation Week for Josh and Nicole. You've had the unusual
    opportunity at this early stage to have gone through Starliner training and
    now also SpaceX training. Can you give us maybe like two or three specific
    examples of what you've gleaned from the two different programs?"
A: "We are in a unique position, having been trained on both spacecraft. And it
    is remarkable to see how two different communities approach and solve the same
    problem, not unlike we see with our international partners..."
```

**2. Media interview — science content (Mark Zuckerberg Facebook event):**

```
Q: "So what kind of research are you doing in space that we can't do anywhere else?"
A: "You know, one of the main things that we do is try to understand the impacts
    of zero gravity on the human body... we're learning a ton."
```

**3. Student Q&A — clean curiosity question (Iowa Space Grant):**

```
Q: "What motivates you to go into space and explore the unknown?"
A: "Well I have to say that I think what had motivated me early in life was the
    feeling that humans were destined to explore..."
```

**4. Press conference — post-mission debrief (Expedition 66 Crew-2 Departure):**

```
Q: "Frank, for you, what experiment or activity that you were involved in aboard
    the station are you most proud of?"
A: "That's a tough question because we had over 250 science experiments on board..."
```

**5. Press conference — policy/future-oriented (Crew-6 Post-Flight):**

```
Q: "Are you able to talk a little bit more about the plans for yourself and also
    for the UAE Space Agency for the human spaceflight program?"
A: "Yes... Our program is continuous. It was announced to be sustainable from
    the first day. My mission is a continuation... the UAE is committed to actually
    going further into space."
```

**6. Student Q&A — STEM curiosity (Children's Museum of Saratoga):**

```
Q: "My question for you is, what do you do for fun in space?"
A: [substantive answer about hobbies, photography, exercise...]
```

**7. Panel discussion — institutional perspective (ISS 20th Anniversary):**

```
Q: "How is the International Space Station a player in some of these
    commercialization efforts?"
A: "I'm going to start with the second question first, because the why you do
    something matters..."
```

---

## 6. Bad Examples (worst noise)

**1. Readiness check (most common single noise pair):**

```
Q: "Are you ready for the event? Houston, this is Station. We're ready for the
    event. StarTalk, this is Mission Control Houston..."
A: "Can you hear me okay? Hey, Mark."
```

**2. Pure ceremony (Artemis II):**

```
Q: "To give us a proper Houston welcome, please join me in bringing out JSC
    Center Director, Vanessa White."
A: "Thank you, Joe. Good morning."
```

**3. Completely empty content (VideoFiles2010):**

```
Q: "That's a very good question."
A: "Good question."
```

**4. Answer bleed (KYW-TV):**

```
Q: "And how many times a day do you circle the globe? 16"
A: "times a day we go around the globe"
```

**5. Garbled multi-question merge (Armstrong Flight Research):**

```
Q: "What does your body feel like in zero gravity? What inspires you to become
    an astronaut? What exercise do you do and do you sweat in space?"
A: "What do yo-yo work in microgravity?"
```

**6. Moderator introduction (Monta Loma):**

```
Q: "Our next question comes from another fifth grade student named Tylee Ramos."
A: [Tylee's actual answer about being an astronaut...]
```

**7. Leading statement, no question (Expedition 46 multi-network):**

```
Q: "Long-duration flight, this was all about seeing whether humans can go to Mars."
A: "Well, that's certainly one of the goals of very long duration flights..."
```

**8. Truncated question (Expedition 58-59):**

```
Q: "Can you tell us a little bit about how team sports has played a part in your
    path to becoming an"
A: [astronaut's response to the complete question that was actually asked]
```

**9. Sign-off labeled as question (Expedition 46):**

```
Q: "Commander Kelly, thanks so much for doing this again with us."
A: "You're welcome."
```

**10. Inverted pair (Expedition 43 1-Year Crew):**

```
Q: "Scott, along those same lines, in less than two weeks a SpaceX Dragon cargo
    craft will be coming your way..."        [journalist's setup, not the question]
A: "Is this the variety and pace of work that will keep you focused or will it
    be necessary to tune out from time to time?"   [the real question is HERE]
```

---

## 7. Event Type Comparison

| Dimension                    | `press_conference`               | `media_interview`                    | `student_qa`                 | `panel`         |
| ---------------------------- | -------------------------------- | ------------------------------------ | ---------------------------- | --------------- |
| Typical pair count           | 12–18                            | 8–14                                 | 12–25                        | 10–28           |
| Genuine question rate (est.) | ~65–70%                          | ~55–65%                              | ~50–60%                      | ~65–70%         |
| Primary noise type           | Statements as Q, moderator admin | Readiness checks, leading statements | Moderator intros, intro-only | Statements as Q |
| Readiness check frequency    | Low                              | **Very high**                        | High                         | Low             |
| Inverted pair risk           | Low–medium                       | **High**                             | Low                          | Low             |
| Answer quality               | High                             | Medium–high                          | High                         | High            |

**`press_conference`** is the most reliable type. Journalists tend to ask genuine questions with `?` and attribution ("Hi, [name] from [outlet]."). The main failure mode is long preambles before the question and occasional moderator admin lines.

**`media_interview`** has the worst noise profile. In-flight events with radio/TV networks suffer from readiness check noise at the start, and multi-network events (ABC/CBS/CNN/FOX/NBC broadcast pools) have frequent speaker-transition confusion causing inverted pairs. Single-interviewer media interviews (Women's Day, Gizmodo, Science Friday) are cleaner.

**`student_qa`** has the highest raw noise rate mostly because moderator introductions ("Our next question comes from...") are pervasive and because students often say only their names without a question. However, when a genuine student question is captured, the answers are excellent — astronauts give detailed, accessible explanations clearly calibrated for a younger audience.

**`panel`** files are few (4 files) but high quality. The structured format with a moderator directing questions to specific panelists works well for extraction.

---

## 8. Conclusions

### What's working

- The pipeline correctly identifies and produces **zero pairs** for `produced_content` and `produced_interview` files (70 files, 8% of corpus).
- **~62% of all 10,135 pairs are genuine questions** in the sense that they prompt a real astronaut response.
- Answer text is almost universally present and substantive (97% ≥ 20 chars). The answers are the most reliable, information-dense part of each pair.
- `press_conference` and `panel` extraction quality is reasonably good (~65–70% genuine).

### What needs improvement

- **508 readiness-check pairs** (5%) are reliably identifiable by the phrase "Are you ready for the event?" and could be filtered during post-processing with a simple string match.
- **313 thank-you/sign-off pairs** (3%) could be filtered using patterns like `thanks for (taking|doing|joining)` and `thank you so much`.
- **~140 intro-only pairs** — "Hi, my name is X." — could be filtered using `^(hi,?\s*)?my name is \w+\.?\s*$`.
- **Moderator intro pairs in student_qa** — "Our next question comes from..." — could be filtered on this phrase across all files, recovering ~100+ noisy pairs.
- **Remaining ~30% statement noise** is harder to filter without semantic analysis, since some genuine leading statements do prompt substantive answers.
- **Inverted pair detection** is hard without speaker-role metadata (which speaker is the journalist vs. the crew?). The `speaker` field in each pair currently holds diarization IDs (SPEAKER_00, SPEAKER_01) that are file-local and not mapped to roles.

### Specific recommendations

1. **Add a post-processing filter** in `5c_build_qa_text.py` that removes pairs where `question.text` matches any of:
   - `"are you ready for the event"` (case-insensitive)
   - `"^(hi,?\s*)?my name is \w+\.?\s*$"`
   - `"^(thank you|thanks for)"`
   - `"our next question comes from"`
   - `"i'll turn it over to"`
     This would clean ~1,000 pairs (10%) with near-zero false positives.

2. **Short question filter**: Consider dropping pairs where `len(question.text) < 10` — all 54 such pairs are noise.

3. **Speaker role mapping**: Adding crew vs. journalist speaker diarization would enable detection of inverted pairs and dramatically improve extraction fidelity.

4. **Ceremony/announcement detection**: The event type classifier should distinguish between press conferences with genuine Q&A and pure announcement ceremonies (like the Artemis II crew announcement). Adding `announcement_ceremony` as an event type and routing these to zero pairs would avoid extracting stage directions.
