# QA Recall Analysis
**Date:** 2026-02-24  
**Script assessed:** `scripts/5b_extract_qa.py`  
**Sample size:** 43 transcript-QA pairs (stratified by event type, random seed 42)

## Methodology

A transcript segment is classified as a **candidate question** if its `text` field:
- Contains a `?` character, OR
- Starts with an interrogative word/phrase (`what`, `who`, `when`, `where`, `how`, `why`, `which`, `could you`, `would you`, `can you`, `do you`, `is there`, `are there`, `did`, `does`, `will you`, `have you`, `is it`, `are you`, etc.)

A candidate segment is **captured** if its `[start, end]` time range overlaps with any QA pair's `[question_start - 10s, question_end + 10s]` window.  
A candidate segment is **missed** (false negative) if no such overlap exists.

**Recall estimate** = captured / total candidates

---

## Aggregate Results

| Metric | Value |
|--------|-------|
| Files sampled | 43 |
| Total candidate question segments | 1121 |
| Captured (overlaps QA pair window) | 850 |
| Missed (no overlap) | 271 |
| Of which: contain literal `?` | 212 |
| **Estimated recall** | **75.8%** |

### By Event Type

| Event Type | Candidates | Captured | Missed | Recall |
|------------|-----------|---------|--------|--------|
| media_interview | 229 | 197 | 32 | 86.0% |
| panel | 180 | 107 | 73 | 59.4% |
| press_conference | 375 | 299 | 76 | 79.7% |
| produced_interview | 46 | 15 | 31 | 32.6% |
| student_qa | 291 | 232 | 59 | 79.7% |

### Per-File Breakdown

| File (truncated) | Event Type | Candidates | Captured | Missed | Recall |
|------------------|-----------|-----------|---------|--------|--------|
| `iss074m260231458__NASA_Astronaut_Discusses_Life_In_Space_Wit` | media_interview | 17 | 14 | 3 | 82.4% |
| `Expedition47ResourceReel__Inflight-Event_European-Space-Educ` | media_interview | 36 | 30 | 6 | 83.3% |
| `EXP_61_InFlight_Luca_Parmitano_UN_COP25_2019_1211_1280905__E` | media_interview | 11 | 8 | 3 | 72.7% |
| `Week-of-April-10-2017__Inflight-Event_CBS-Radio-News_CNN_502` | media_interview | 41 | 36 | 5 | 87.8% |
| `InflightEvent-WomansDayAugust152017__Inflight-Event_Womans-D` | media_interview | 31 | 24 | 7 | 77.4% |
| `Inflight-Event_KMA-Radio_2017_221_1400_550746.mxf__Inflight-` | media_interview | 28 | 25 | 3 | 89.3% |
| `Expedition48ResourceReel__Inflight-Event_Westwood-One_San-Di` | media_interview | 31 | 29 | 2 | 93.5% |
| `Expedition46ResourceReel__Interview_Mark Kelly_160225_352079` | media_interview | 4 | 4 | 0 | 100.0% |
| `jsc2021m000108_Quick_Questions_With_Crew-2__jsc2021m000108_Q` | media_interview | 1 | 0 | 1 | 0.0% |
| `iss071m261941724_Expedition71_NASA_Astronaut_Matt_Dominick_T` | media_interview | 29 | 27 | 2 | 93.1% |
| `jsc2020m001435_iss_20th_anniversary_panel_iss_and_beyond_202` | panel | 49 | 17 | 32 | 34.7% |
| `jsc2021m000211_Pittsburgh_Steelers_Quarterback_Talks_With_NA` | panel | 34 | 23 | 11 | 67.6% |
| `jsc2020m001434_ISS_20th_Anniversary_Panel_Trailblazing_Inter` | panel | 40 | 39 | 1 | 97.5% |
| `jsc2020m001425_ISS_20th_Anniversary_Panel_Expanding_the_Mark` | panel | 57 | 28 | 29 | 49.1% |
| `Expedition45ResourceReel__Crew-News-Conference_2015_306_1500` | press_conference | 41 | 29 | 12 | 70.7% |
| `iss065m262651659_Expedition_65_PAO_ESA_Radio_France_210922_l` | press_conference | 26 | 20 | 6 | 76.9% |
| `iss054m261101059_Expedition_65_Inflight_with_Prime_Minister_` | press_conference | 12 | 8 | 4 | 66.7% |
| `Crew-5_Preflight-Briefings__Crew-5_Crew_News_Conference_2208` | press_conference | 24 | 21 | 3 | 87.5% |
| `Crew-4_Mission_Overview_Briefing_220331_1612883__Crew-4_Miss` | press_conference | 59 | 36 | 23 | 61.0% |
| `Expedition46ResourceReel__Expedition-46-Crew-News-Conference` | press_conference | 76 | 64 | 12 | 84.2% |
| `Expedition_58_State_Commission_and_Crew_News_Conference-2018` | press_conference | 32 | 23 | 9 | 71.9% |
| `Expedition_65_Recorded_Soyuz_Crew_News_Conference_210324_145` | press_conference | 57 | 52 | 5 | 91.2% |
| `iss063m262111933_Expedition_63_Post_Flight_Readiness_Review_` | press_conference | 16 | 15 | 1 | 93.8% |
| `iss066m260731759_Expedition_66_Spacewalk_79-80_Preview_Brief` | press_conference | 32 | 31 | 1 | 96.9% |
| `jsc2020m000128_Bob_Behnken_discusses_his_background_as_a_fli` | produced_interview | 9 | 1 | 8 | 11.1% |
| `jsc2018m000967-CCP_Interview_Behnken.mxf__jsc2018m000967-CCP` | produced_interview | 2 | 1 | 1 | 50.0% |
| `Expedition_56_Education_Interview_Goddard_Space_Flight_Ctr_S` | produced_interview | 7 | 2 | 5 | 28.6% |
| `jsc2021m000180-Crew-3_Interview_Reel_Matthias_Maurer__jsc202` | produced_interview | 8 | 7 | 1 | 87.5% |
| `Expedition48ResourceReel__Interview_Bowman_160706_398469_low` | produced_interview | 2 | 0 | 2 | 0.0% |
| `jsc2021m000181-crew-3_interview_reel_thomas_marshburn_202109` | produced_interview | 8 | 1 | 7 | 12.5% |
| `jsc2021m000093_Crew2_Interview_Reel_Shane_Kimbrough__jsc2021` | produced_interview | 6 | 2 | 4 | 33.3% |
| `jsc2018m000971-CCP_Interview_Cassada.mxf__jsc2018m000971-CCP` | produced_interview | 1 | 1 | 0 | 100.0% |
| `jsc2020m000132_Bob_Behnken_discusses_training_and_developmen` | produced_interview | 3 | 0 | 3 | 0.0% |
| `Expedition50ResourceReel01__Inflight-Educational-Interview_N` | student_qa | 26 | 21 | 5 | 80.8% |
| `iss060m262211714_Expedition_60_Inflight_with_Slover_Library_` | student_qa | 34 | 26 | 8 | 76.5% |
| `iss070m263261553_Expedition_70_Astronaut_Mogensen_Talks_with` | student_qa | 34 | 31 | 3 | 91.2% |
| `iss062m260581739-Expedition_62_Inflight_with_Lee_County_Scho` | student_qa | 21 | 20 | 1 | 95.2% |
| `iss072m260651529_Space_Station_Crew_Talks_with_the_Saint_Amb` | student_qa | 37 | 29 | 8 | 78.4% |
| `iss065m262071459_Expedition_65_Education_Inflight_with_U.S._` | student_qa | 27 | 19 | 8 | 70.4% |
| `Expedition-56-Inflight-Event_Future-Engineers_Smithsonian_Ju` | student_qa | 42 | 33 | 9 | 78.6% |
| `Inflight-Event_Educational_Puerto_Rico_Institute_of_Robotics` | student_qa | 28 | 19 | 9 | 67.9% |
| `iss071m261091400_Astronaut_Jeanette_Epps_Answers_Syracuse_St` | student_qa | 24 | 20 | 4 | 83.3% |
| `iss067m262341759_Expedition_67_Challenger_Center_edu_220822_` | student_qa | 18 | 14 | 4 | 77.8% |

---

## Examples of Likely Missed Questions

These are transcript segments containing `?` that do **not** overlap with any extracted QA pair's question window.

**File:** `Crew-4_Mission_Overview_Briefing_220331_1612883__Crew-4_Mission_Overvi`  
**Event:** press_conference  
**Timestamp:** 24:24 (1464.0s)  
**Text:** Yeah, I would say from a NASA perspective and CCP, we've been working hand-in-hand with SpaceX to look at the sparing strategy, right?

**File:** `Crew-4_Mission_Overview_Briefing_220331_1612883__Crew-4_Mission_Overvi`  
**Event:** press_conference  
**Timestamp:** 35:13 (2113.9s)  
**Text:** We went through a very specific analysis for this flight to get to that reuse, and we're going to work with SpaceX to try to figure out, you know, can we go more than one reuse?

**File:** `Crew-4_Mission_Overview_Briefing_220331_1612883__Crew-4_Mission_Overvi`  
**Event:** press_conference  
**Timestamp:** 36:20 (2180.1s)  
**Text:** You want me to speak to Jessica?

**File:** `Crew-4_Mission_Overview_Briefing_220331_1612883__Crew-4_Mission_Overvi`  
**Event:** press_conference  
**Timestamp:** 37:20 (2240.0s)  
**Text:** Want to talk to crew assignments?

**File:** `Crew-4_Mission_Overview_Briefing_220331_1612883__Crew-4_Mission_Overvi`  
**Event:** press_conference  
**Timestamp:** 41:54 (2514.1s)  
**Text:** we will basically evaluate over the next few years, should that be a Dragon mission or would that be a Starship mission?

**File:** `Crew-4_Mission_Overview_Briefing_220331_1612883__Crew-4_Mission_Overvi`  
**Event:** press_conference  
**Timestamp:** 42:49 (2569.5s)  
**Text:** And then once you have Starliner flying, are you, like, alternating Crew Dragons and Starliners one each a year, or what is the cadence once they're both operational?

**File:** `Crew-4_Mission_Overview_Briefing_220331_1612883__Crew-4_Mission_Overvi`  
**Event:** press_conference  
**Timestamp:** 46:15 (2775.4s)  
**Text:** How do we incorporate it into spacecraft?

**File:** `Crew-4_Mission_Overview_Briefing_220331_1612883__Crew-4_Mission_Overvi`  
**Event:** press_conference  
**Timestamp:** 46:18 (2778.3s)  
**Text:** Do we really have growing chambers?

**File:** `Crew-4_Mission_Overview_Briefing_220331_1612883__Crew-4_Mission_Overvi`  
**Event:** press_conference  
**Timestamp:** 46:20 (2780.2s)  
**Text:** Do we have plants all over the place?

**File:** `Crew-4_Mission_Overview_Briefing_220331_1612883__Crew-4_Mission_Overvi`  
**Event:** press_conference  
**Timestamp:** 46:21 (2781.5s)  
**Text:** Does it look like a farm?

**File:** `Crew-4_Mission_Overview_Briefing_220331_1612883__Crew-4_Mission_Overvi`  
**Event:** press_conference  
**Timestamp:** 46:53 (2813.3s)  
**Text:** What can we do better to understand the situation on board and the health of our crew members?

**File:** `Crew-4_Mission_Overview_Briefing_220331_1612883__Crew-4_Mission_Overvi`  
**Event:** press_conference  
**Timestamp:** 47:42 (2862.9s)  
**Text:** Stephen?

**File:** `Crew-4_Mission_Overview_Briefing_220331_1612883__Crew-4_Mission_Overvi`  
**Event:** press_conference  
**Timestamp:** 47:48 (2868.7s)  
**Text:** Can you hear me, Stephen?

**File:** `Crew-4_Mission_Overview_Briefing_220331_1612883__Crew-4_Mission_Overvi`  
**Event:** press_conference  
**Timestamp:** 48:17 (2897.4s)  
**Text:** Is that still a plan for a joint spacewalk with the Russian cosmonauts?

**File:** `Crew-4_Mission_Overview_Briefing_220331_1612883__Crew-4_Mission_Overvi`  
**Event:** press_conference  
**Timestamp:** 48:22 (2902.3s)  
**Text:** Can you just talk about generally?

**File:** `Crew-4_Mission_Overview_Briefing_220331_1612883__Crew-4_Mission_Overvi`  
**Event:** press_conference  
**Timestamp:** 48:23 (2903.9s)  
**Text:** The status of activating that, has that been put on hold or is it still going forward?

**File:** `Crew-4_Mission_Overview_Briefing_220331_1612883__Crew-4_Mission_Overvi`  
**Event:** press_conference  
**Timestamp:** 54:26 (3266.6s)  
**Text:** Was there anything that we saw that was anomalous?

**File:** `Crew-4_Mission_Overview_Briefing_220331_1612883__Crew-4_Mission_Overvi`  
**Event:** press_conference  
**Timestamp:** 54:42 (3282.1s)  
**Text:** Was there anything anomalous with the GNC that might have had some certain rate to cause a slow inflation?

**File:** `Crew-4_Mission_Overview_Briefing_220331_1612883__Crew-4_Mission_Overvi`  
**Event:** press_conference  
**Timestamp:** 57:38 (3458.6s)  
**Text:** the ability to grow things, but what's different?

**File:** `Crew-5_Preflight-Briefings__Crew-5_Crew_News_Conference_220804_1673970`  
**Event:** press_conference  
**Timestamp:** 17:55 (1075.3s)  
**Text:** So on the SpaceX side, of course, they have the heritage of having done a lot of cargo missions, right?

*(... and 192 more missed `?` segments not shown)*

---

## Suspicious QA Pairs (No `?` in Question Window)

These QA pairs have a `question_start`/`question_end` window that contains **no transcript segment with `?`**. This suggests the 'question' may be a preamble, hand-off, or misidentified segment.

**File:** `iss074m260231458__NASA_Astronaut_Discusses_Life_In_Space_With_Columbia`  
**Event:** media_interview  
**Window:** 5:52–5:56 (352.3s–356.4s)  
**Window text:** That's a really good question as well. Yeah, there's something that pops to my mind.

**File:** `iss074m260231458__NASA_Astronaut_Discusses_Life_In_Space_With_Columbia`  
**Event:** media_interview  
**Window:** 8:45–8:47 (525.2s–527.9s)  
**Window text:** Hey, Ash, that's a really, really great question. Yeah, and I did make a pretty big career shift a couple times in my career.

**File:** `iss074m260231458__NASA_Astronaut_Discusses_Life_In_Space_With_Columbia`  
**Event:** media_interview  
**Window:** 16:21–16:35 (981.6s–995.1s)  
**Window text:** Hey, Astronaut Williams. I'm wondering how your team-building experience with scouts when you were younger impacted your career as an astronaut. Yeah, that's a great question. I was involved in the Sc

**File:** `Expedition47ResourceReel__Inflight-Event_European-Space-Education-Reso`  
**Event:** media_interview  
**Window:** 8:16–8:17 (496.8s–497.6s)  
**Window text:** Hi, Tim. Milana Kołodziejczyk, primary school number 94 in Warsaw.

**File:** `Expedition47ResourceReel__Inflight-Event_European-Space-Education-Reso`  
**Event:** media_interview  
**Window:** 21:59–22:04 (1319.1s–1324.1s)  
**Window text:** And I think you can sustain as long as you need to, as long as we can break the code on some of the physiological aspects of zero gravity. For Tim Peake, I'm wondering how the carbo loading is going u

**File:** `EXP_61_InFlight_Luca_Parmitano_UN_COP25_2019_1211_1280905__EXP_61_InFl`  
**Event:** media_interview  
**Window:** 12:16–12:33 (736.4s–753.1s)  
**Window text:** And the only way to do it right now, today, is for everybody to come together, despite our differences. The other way to join people is to find one common enemy. And today we have the number one enemy

**File:** `EXP_61_InFlight_Luca_Parmitano_UN_COP25_2019_1211_1280905__EXP_61_InFl`  
**Event:** media_interview  
**Window:** 13:27–13:42 (807.0s–822.1s)  
**Window text:** All these things can be revolutionized using technology and what we know today to make a better world and to join everybody into the fight against a changing world. I must say that I am really impress

**File:** `EXP_61_InFlight_Luca_Parmitano_UN_COP25_2019_1211_1280905__EXP_61_InFl`  
**Event:** media_interview  
**Window:** 16:11–16:24 (971.9s–984.8s)  
**Window text:** So there is more than tomorrow's election. There is more to the world than a seat on a panel. What we need is the vision to understand that there are things that we need to sacrifice today so that we 

**File:** `EXP_61_InFlight_Luca_Parmitano_UN_COP25_2019_1211_1280905__EXP_61_InFl`  
**Event:** media_interview  
**Window:** 17:19–17:23 (1039.7s–1043.5s)  
**Window text:** And it's high time we pulled our head out of the sand because that's just when we refuse to see what's happening. And we need to look at the problem in the eyes. We need to see the problem in order to

**File:** `EXP_61_InFlight_Luca_Parmitano_UN_COP25_2019_1211_1280905__EXP_61_InFl`  
**Event:** media_interview  
**Window:** 17:56–18:13 (1076.7s–1093.9s)  
**Window text:** Lucas, one of the—let me explain a little bit the surroundings that people are seeing here. So, Lucas is standing—not standing, it's holding with his feet onto one side of the space station in order n

*(... and 478 more suspicious pairs not shown)*

---

## Large Missed Q&A Sections (Gap Clusters)

Clusters of 3+ consecutive missed `?`-containing segments within 120 seconds of each other — suggesting entire Q&A sections were not extracted.

**File:** `jsc2020m001425_ISS_20th_Anniversary_Panel_Expanding_the_Market_in_Low-`  
**Event:** panel  
**Span:** 31:47–35:42 (235s window)  
**Missed `?` segments in cluster:** 14  
**Sample segments:**
- `31:47` — So how to do that, so it's kind of a practical set of steps as you think through the problem, right?
- `31:57` — And what are the barriers to that?
- `32:06` — The cost is the problem, right?

**File:** `jsc2020m001435_iss_20th_anniversary_panel_iss_and_beyond_202101__jsc20`  
**Event:** panel  
**Span:** 17:20–24:38 (439s window)  
**Missed `?` segments in cluster:** 11  
**Sample segments:**
- `17:20` — And what does that mean?
- `18:25` — Well, right now, you know, we use predominantly exercise for the microgravity, the altered gravity hazard, right?
- `18:47` — Like how is the human going to affect the vehicle?

**File:** `Crew-4_Mission_Overview_Briefing_220331_1612883__Crew-4_Mission_Overvi`  
**Event:** press_conference  
**Span:** 46:15–48:29 (134s window)  
**Missed `?` segments in cluster:** 10  
**Sample segments:**
- `46:15` — How do we incorporate it into spacecraft?
- `46:18` — Do we really have growing chambers?
- `46:20` — Do we have plants all over the place?

**File:** `jsc2020m001435_iss_20th_anniversary_panel_iss_and_beyond_202101__jsc20`  
**Event:** panel  
**Span:** 10:36–12:12 (96s window)  
**Missed `?` segments in cluster:** 7  
**Sample segments:**
- `10:36` — And it's not very hard work physically to live in microgravity, right?
- `11:04` — So the question has been, what value is going to a planetary surface from the human standpoint of getting exposed to som
- `11:24` — We're always concerned, like, how much do we really replicate the actual experience of spaceflight?

**File:** `Inflight-Event_Educational_Puerto_Rico_Institute_of_Robotics_012_1025C`  
**Event:** student_qa  
**Span:** 4:09–5:35 (86s window)  
**Missed `?` segments in cluster:** 6  
**Sample segments:**
- `4:09` — Are you ready for the event?
- `4:37` — Hello?
- `4:57` — Hello?

**File:** `Week-of-April-10-2017__Inflight-Event_CBS-Radio-News_CNN_502657_lowres`  
**Event:** media_interview  
**Span:** 3:00–4:50 (109s window)  
**Missed `?` segments in cluster:** 5  
**Sample segments:**
- `3:00` — Are you ready for the event?
- `3:17` — How do you hear me?
- `3:23` — How are you doing?

**File:** `Expedition45ResourceReel__Crew-News-Conference_2015_306_1500_314360_lo`  
**Event:** press_conference  
**Span:** 21:14–23:35 (141s window)  
**Missed `?` segments in cluster:** 5  
**Sample segments:**
- `21:14` — What exactly could you compare it to what a human is experiencing when in outer space?
- `21:28` — And what would you say about the suit and the tools that you used when in space?
- `23:27` — Are you going to celebrate it?

**File:** `iss065m262651659_Expedition_65_PAO_ESA_Radio_France_210922_lowres`  
**Event:** press_conference  
**Span:** 20:13–22:14 (121s window)  
**Missed `?` segments in cluster:** 5  
**Sample segments:**
- `20:13` — So inside this capsule, are you following the news?
- `20:17` — So if you're like in a submarine, do you realize what I'm talking about?
- `20:55` — Is there any diplomatic?

**File:** `iss065m262071459_Expedition_65_Education_Inflight_with_U.S._Embassy_of`  
**Event:** student_qa  
**Span:** 0:23–2:26 (123s window)  
**Missed `?` segments in cluster:** 5  
**Sample segments:**
- `0:23` — Are you ready for the event?
- `0:39` — How do you hear me?
- `0:45` — How about us?

**File:** `jsc2020m001435_iss_20th_anniversary_panel_iss_and_beyond_202101__jsc20`  
**Event:** panel  
**Span:** 29:30–33:58 (267s window)  
**Missed `?` segments in cluster:** 5  
**Sample segments:**
- `29:30` — variety through other means?
- `29:32` — Can we introduce variety through, as Jennifer mentioned, a supplemental crop system, right?
- `31:27` — So imagine you've never experienced a hill or rain or a pothole, and you don't have any gas stations or service stations

---

## Patterns Observed

1. **Best recall: `media_interview`** (197/229 = 86.0%); **worst: `produced_interview`** (15/46 = 32.6%).

2. **Suspicious QA pairs** (no `?` in question window): 488 out of 989 total QA pairs = 49.3%. These are likely preambles, moderator hand-offs, or misattributed segments.

3. **Interrogative-word candidates without `?`:** 59 missed segments were identified by interrogative words alone (no `?`). Some of these may be false positives (statements starting with 'How great it is...' etc.), but others represent real questions missed by 5b.

4. **Gap clusters:** 20 clusters of 3+ consecutive missed questions detected. These represent contiguous Q&A sections completely absent from the extraction.

5. **Files with 0% recall** (had candidates but none captured): 3 files.

   - `jsc2021m000108_Quick_Questions_With_Crew-2__jsc2021m000108_Quick_Questions_With_` (media_interview, 1 candidates, 3 QA pairs)
   - `Expedition48ResourceReel__Interview_Bowman_160706_398469_lowres` (produced_interview, 2 candidates, 1 QA pairs)
   - `jsc2020m000132_Bob_Behnken_discusses_training_and_development__jsc2020m000132_Bo` (produced_interview, 3 candidates, 6 QA pairs)

---

*Analysis generated by `scripts/_qa_recall_analysis.py`*
