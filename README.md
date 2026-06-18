# AI Engineering Bootcamp (Intern Edition)

Welcome to the AI Engineering Bootcamp repository! This 5-day intensive program is designed for undergraduate Computer Science and Software Engineering interns to transition from vibe-coding to building production-ready, structured, and reliable AI systems.

## 📋 Course Overview
- **Duration:** 5 Days + Day 0 Prep
- **Level:** Undergrad CS/SE
- **Model:** Gemini 2.5 Flash (Free Tier)
- **Objective:** Go from basic ChatGPT prompting to engineering robust AI systems including Structured Outputs, RAG (Retrieval-Augmented Generation), and Autonomous AI Agents.

---

## 🌿 Repository Branch Structure
Each day's hands-on work and deliverables are organized into separate branches for clean tracking:
- `day-0-prep` — Pre-requisites, research, and API configuration.
- `day-1-fundamentals` — LLM foundations, role-based prompting, and `AlService.js`.
- `day-2-reliability` — Structured JSON outputs, schema validation, and retry logic.
- `day-3-rag` — Embeddings, local vector search with ChromaDB, and a mini RAG system.
- `day-4-agents` — Tool calling, autonomous ReAct loops, and human-in-the-loop gates.
- `day-5-capstone` — Final architecture diagrams, project implementations, and demos.

---

## 📅 Bootcamp Curriculum

### 📍 Day 0: Before You Start (Pre-Work)
*Spend ~1 hour tonight so you are ready for Day 1.*

#### 🔍 Research Homework
1. **Watch:** "But what is a GPT?" by 3Blue1Brown on YouTube (~26 min). *The best visual explanation of how LLMs work without heavy math.*
2. **Read:** Search "what is prompt engineering" and read a clear article (~10 min). Focus on understanding *system prompt vs. user prompt*, and *few-shot prompting*.
3. **Get Your API Key:** Head to [aistudio.google.com/api-keys](https://aistudio.google.com/api-keys), sign in with a Google account, and create a free key. Store it safely.
4. **Explore:** Spend 10 minutes talking to Claude.ai or ChatGPT. Try to intentionally make it refuse a prompt and observe what happens.

---

### 🚀 Day 1: LLM Fundamentals + Your First API Call
*Understand what LLMs actually are and make your first programmatic API call.*

#### 📘 Topics Covered
- AI vs. ML vs. GenAI vs. LLM hierarchy
- How LLMs work (Tokens, temperature, context window)
- System prompt vs. user prompt & role-based prompting
- Prompt templates with placeholders & few-shot prompting
- Why LLMs have no memory between calls (statelessness)
- Where AI fits into a traditional backend architecture

#### 💻 Hands-on Exercises
- Make your first API call and get a response in the terminal.
- Build an **Email Drafter** and a **Data Extractor** from raw text.
- Create an `AlService.js` file with 4 specific functions:
  - Lead qualification prompt
  - Support ticket classifier
  - Email drafter
  - Data extractor
- **Break it on purpose:** Intentionally trigger hallucinations, prompt injections, and edge cases.

> 💡 **Intern Note:** The "break it on purpose" task at the end of the day is not optional—it's the most important exercise. Document every single failure with notes to discuss during the End-of-Day (EOD) sync.

#### 📦 Deliverable
- `AlService` file with 4 working prompt functions + failure notes from the break session.

---

### 🛡️ Day 2: Structured Outputs + Making AI Production-Ready
*Tame unpredictable AI responses into validated, consistent data structures.*

#### 📘 Topics Covered
- Why free-text AI responses break in real applications
- Prompting for strict JSON output & schema-based validation (Zod/Pydantic)
- Streaming responses & retry handling for invalid outputs
- Guardrails, fallback handling, and hallucination control basics
- Confidence scores, human review flags, and prompt versioning basics
- Token cost optimization

#### 💻 Hands-on Exercises
- Refactor Day 1 functions to return structured, validated JSON.
- Add robust retry logic (up to 3 attempts before failing).
- Append a `confidence_score` field to each response and automatically flag low-confidence results for human review.

> 💡 **Intern Note:** "Works in the playground" ≠ "Works in production." Day 2 bridges this gap. This is the stage most beginners skip and later regret.

#### 📦 Deliverable
- Production-style AI feature module with validated JSON output, built-in retry logic, and human review flags.

---

### 🧠 Day 3: Embeddings + Vector Search + RAG
*Connect LLMs to external data sources to generate grounded, factual answers.*

#### 📘 Topics Covered
- What embeddings are and how numbers represent semantic meaning
- Semantic search vs. keyword search (and Cosine Similarity)
- Vector databases: Local setup with ChromaDB, overview of Pinecone/pgvector
- Document chunking strategies & building a RAG pipeline from scratch
- Why RAG reduces hallucinations and how to implement source attribution

#### 💻 Hands-on Exercises
- Chunk a raw PDF document into manageable pieces.
- Generate embeddings for each chunk and store them locally in ChromaDB.
- Query the database with a question to retrieve the top 3 relevant chunks.
- Pass those chunks into the LLM to generate a grounded answer.

> 💡 **Intern Note:** This is the day things click. Once you understand RAG, you understand how every "chat with your PDF" product actually works. It's not magic—it's 50 lines of code.

#### 📦 Deliverable
- A mini RAG system capable of answering questions fully sourced and grounded in a real document.

---

### 🤖 Day 4: AI Agents + Tool Calling + Workflow Automation
*Move from responsive chatbots to autonomous agents that can act and execute workflows.*

#### 📘 Topics Covered
- Chatbots vs. Agents (responding vs. acting)
- How an LLM decides which tool to call & Tool/Function calling mechanics
- The ReAct loop (Reason ➔ Act ➔ Observe ➔ Repeat)
- Connecting mock APIs and managing state between multi-step execution flows
- Logging agent actions, safe execution practices, and human-in-the-loop approval gates
- Model Context Protocol (MCP) overview

#### 💻 Hands-on Exercises
- Build an autonomous tool-calling agent integrated with 3 mock tools.
- Implement a human approval gate before any destructive action is executed.
- Log every tool call along with its exact inputs and outputs.
- Build graceful error handling for failed tool executions.

> 💡 **Intern Note:** The hardest part of agents is not writing the code—it's understanding that the LLM is autonomously deciding what to call and in what order. You aren't writing a standard script; you're writing constraints for an autonomous decision-maker.

#### 📦 Deliverable
- A working tool-calling agent prototype with comprehensive action logging and human approval logic.

---

### 🏛️ Day 5: Architecture, Real Projects + Capstone Demo
*Tie everything together, learn real-world architecture design, and present.*

#### 📘 Topics Covered
- Common production AI architectures (RAG chatbots, multi-agent frameworks, LLM service layers)
- Security & privacy basics (what sensitive data *not* to send to LLMs)
- Cost, latency, scaling optimizations, and estimating AI project resource requirements
- Explaining complex AI concepts to non-technical stakeholders

#### 🛠️ Capstone Options (Pick One to Build & Present)
- **AI Lead Qualification Agent**
- **RAG Knowledge Base Chatbot**
- **Support Ticket Router** with priority scoring
- **AI Email Drafting Assistant**
- **CRM Automation Agent**

> 💡 **Intern Note:** The capstone demo is not about reaching flawless perfection. It's about explaining what you built, justifying your technical choices, and detailing what you would do differently next time. Treat it like a professional project review.

#### 📦 Deliverable
- A working live demo + a clear system architecture diagram + a 5-minute walkthrough presentation.
