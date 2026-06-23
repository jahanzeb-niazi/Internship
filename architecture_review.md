# Architecture Review

## 1. What is the most fragile part of this system and why?
The most fragile part of the system is agent reasoning and deciding whether to qualify, schedule, notify, or escalate by interpreting its current state against a text-based rule list, If a candidate's profile is highly ambiguous or missing expected keys, the agent might get stuck in a reasoning loop or hallucinate tool calls 

## 2. Where would it break first under 100x the current load?
Under 100x the current load, the system would immediately break at the LLM API rate limits and the blocking CLI-based human-in-the-loop approval gate.

## 3. What is the one prompt decision you are least confident about, and what would you test to resolve that uncertainty?
I am least confident about the `parser.py` prompt that extracts `CandidateProfile` from text. Parsing years of experience as a precise float is unreliable due to overlapping roles, part-time work, and ambiguous dates.I would evaluate it on 100+ edge-case resumes and measure precision/recall of years_of_experience against human-annotated ground truth.

## 4. What would you do differently if you started over today?
If I were to start over today, I would shift from a purely prompt-driven agent loop to state driven hard coded agent, and the LLM is only invoked for isolated cognitive tasks. Relying on llm for standard hiring tasks increase latency and cost (do not add value)

## 5. If a client asked you to estimate building this from scratch, what is your hour estimate and what are your biggest unknowns?
I would estimate approximately 40–50 hours to build this into a production-ready system from scratch. My biggest unknowns would be the messiness of real-world resumes.
