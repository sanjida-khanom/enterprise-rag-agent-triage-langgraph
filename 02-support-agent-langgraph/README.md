# Project 2 — Customer Support Agent (LangGraph)

A tool-using agent built as an **explicit state machine**, which looks up a customer's account, finds the policy that applies to their situation, and combines the two into a specific answer.

The job description names LangGraph and AI agents directly. This is the project that covers that ground.

---

## What makes it more than a chatbot

A chatbot answers from what it was given. This agent decides, at runtime, what information it needs and goes and gets it — sometimes in several steps, where step two depends on what step one returned.

The clearest example is scenario 2. A customer asks *"can I upgrade my plan today?"* The agent must:

1. Look up the subscriber → discovers they are **postpaid**
2. Search policy for plan migration rules → mid-cycle changes not permitted for postpaid
3. Notice the account has **BDT 450 outstanding** → a second, independent rule blocks migration while dues are unpaid
4. Combine both into one answer with a next step

No fixed chain could do that, because the second lookup depends on the first result.

## The graph

```
                  START
                    |
                    v
             ┌─────────────┐
             │   reason    │  <───────────┐   LLM: answer, or call a tool?
             └─────────────┘              │
                    |                     │
            should_continue               │
             /            \               │
        "tools"          "end"            │
            |               |             │
            v               |      ┌─────────────┐
      ┌─────────────┐       |      │   execute   │  run tools, append results
      │   execute   │───────┼─────>└─────────────┘
      └─────────────┘       |
                            v
                           END
```

Three components, and you should be able to point at each:
- **Nodes** (`reason`, `execute`) — units of work that read state and return updates
- **Conditional edge** (`should_continue`) — the runtime decision that makes this an agent rather than a chain
- **State** (`AgentState`) — carried along every edge; `add_messages` is a reducer that appends rather than overwrites

**I deliberately did not use `create_react_agent`.** It's one line and hides the entire loop. Building it explicitly means you can explain what happens, which is the whole point of having it in a portfolio.

---

## Setup

```bash
cd 02-support-agent-langgraph
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # add GOOGLE_API_KEY
python llm_setup.py         # verify
```

## Run

```bash
python agent.py --scenario 2 --trace     # the multi-hop showcase
python agent.py --scenario all --trace   # all six
python agent.py                          # interactive
```

`--trace` prints the reasoning loop — which tool the model chose and what came back. **Always demo with `--trace` on.** The loop is the interesting part; without it you're just showing a chatbot.

## The six scenarios

| # | Tests | Expected behaviour |
|---|---|---|
| 1 | Single tool | `search_policy` only |
| 2 | **Multi-hop** | Lookup → postpaid → policy → dues → combined answer |
| 3 | Threshold reasoning | 730-day tenure vs the 90-day eligibility rule |
| 4 | Write action | Duplicate charge → `create_ticket` |
| 5 | Bangla input | Retrieves English policy, answers in Bangla |
| 6 | Out of scope | Refuses; does not invent a rule |

---

## The 3-minute demo script

**Open with scenario 2 and `--trace`.** Let them watch the loop.

> "Watch the trace. First it calls lookup_subscriber and finds this is a postpaid account with 450 taka outstanding. Then — and this is the part a fixed chain couldn't do — it searches policy for the rule that applies to *postpaid* migration specifically, because it now knows the account type. It finds two independent blockers: postpaid can't change mid-cycle, and dues over 300 taka block migration entirely. The answer combines both and tells the customer what to actually do."

Then scenario 6:
> "And here it refuses. Tool descriptions tell it to consult policy before asserting a rule, and when nothing matches, the tool returns text explicitly instructing it not to invent one. Failure handling lives in the tool's return value, not in a try/except — the model has to be able to read and recover from it."

If you have time, scenario 4 (creates a ticket) — good hook for the human-in-the-loop conversation below.

---

## Questions you will be asked

### The big one

**"When would you NOT use an agent?"**

This is the question that separates people who understand agents from people who discovered them last week. Answer it confidently:

> "Most of the time, honestly. If the sequence of steps is known in advance, a fixed chain is cheaper, faster, and far easier to debug and test. An agent adds a model call per reasoning step, non-determinism, and a class of failure — tool-selection loops — that chains simply don't have.
>
> The case for an agent is when the path genuinely depends on what you find. Scenario 2 qualifies: you can't know which policy to look up until you know the account type. Scenario 1 doesn't — that's a single lookup, and I'd serve it with a plain RAG chain in production.
>
> The pattern I'd actually deploy is a cheap classifier routing most traffic to fixed chains and only ambiguous multi-step cases to the agent."

Volunteering this unprompted is a strong signal. A lot of candidates over-apply agents.

### On design

**"How does the model know which tool to call?"**
> "From the tool name, the signature, and the docstring — those get serialised into the schema sent with the request. So the docstring is a prompt, not documentation. Mine say *when* to use each tool, not just what it does: 'Use this whenever the user mentions a phone number or asks about my account.' Vague descriptions are the most common cause of wrong tool selection."

**"What if a tool fails?"**
> "It returns an error message as text rather than raising. 'No subscriber found with number ending 9999, ask the user to confirm' lets the model recover and ask a sensible follow-up. An exception ends the run. I wrap execution in try/except and feed the exception back as a ToolMessage for the same reason — the model is the error handler."

**"Why the step limit?"**
> "Cost and safety. Every loop iteration is a paid model call. A model that gets confused can call the same tool repeatedly forever — I've seen it happen with ambiguous tool descriptions. MAX_STEPS = 6 caps the damage. In production I'd also cap wall-clock time and total tokens, and alert on runs that hit the ceiling, because hitting it is a signal that something is wrong upstream."

**"What is `add_messages` doing?"**
> "It's a reducer on the state field. Nodes return only the new messages and LangGraph appends them to the existing list. Without it each node would have to reconstruct the full history and return it, which is verbose and easy to get wrong. It also handles message ID deduplication."

### On safety and production

**"This agent can create tickets. What if it creates the wrong one?"**
> "Then it should be behind a human confirmation, and in production it would be. Read tools are safe to run autonomously; write tools change state in a system of record. The pattern is human-in-the-loop — LangGraph supports interrupting before a node, so the agent proposes the ticket and a human approves it. I'd let it run autonomously only after the approval rate stayed high for a sustained period on real traffic.
>
> The general principle: the blast radius of a wrong action determines how much autonomy it gets. Creating a ticket is recoverable. Applying a credit to a bill is not."

**"How would you test an agent? It's non-deterministic."**
> "You can't assert on exact strings, so you assert on behaviour. Three levels. Tool selection — given this input, was `lookup_subscriber` called at all? That's deterministic enough to unit test. Trajectory — did it call the tools in a sensible order without redundant calls? Final answer — LLM-as-judge against a rubric, plus a keyword check for facts that must appear. The scenario file is the skeleton of that suite: each one names the behaviour it tests. I'd also run each case several times, because a case that passes four times in five is a flaky agent, and you only find that by repeating."

**"What about prompt injection? A customer could type instructions."**
> "Real risk, and it gets worse as tools gain permissions. Mitigations: never let retrieved content or user text be treated as system instructions; scope tools tightly so `lookup_subscriber` can only read the account the session is authenticated for, not any arbitrary MSISDN; keep write actions behind confirmation; and log everything. The structural point is that permissions should live in the tool layer, not the prompt. A prompt saying 'don't look up other customers' accounts' is a suggestion. A tool that physically cannot is a control."

**"How would this integrate with Robi's real systems?"**
> "The tools are the integration boundary — that's why they're in a separate file. `lookup_subscriber` reads a CSV here; in production it's an authenticated call to the CRM or billing API with the session's identity attached. Nothing in the graph changes. That separation is the reason the demo is honest: the reasoning layer is real, only the data source is mocked."

### On the honest limits

**"What's weak about this?"**
> "No conversation memory across turns, so it can't handle 'and what about my other number?'. No streaming, so the user waits through the whole loop with no feedback — a real UX problem when a run takes eight seconds. The policy search is keyword overlap, not embeddings; I'd swap in the retriever from project 1, and the two projects are meant to compose that way. And I haven't measured it properly — I have six scenarios I check by eye, which is a smoke test, not an evaluation."

---

## Files

| File | What it is |
|---|---|
| `agent.py` | The state graph: nodes, conditional edge, step limit, scenarios |
| `tools.py` | Four tools — three read, one write |
| `data/subscribers.csv` | Six accounts with deliberately varied edge cases |
| `data/plans.csv` | Tariff catalogue |
| `data/policy_kb.json` | Seven business rules |
| `llm_setup.py` | Provider abstraction (shared with project 1) |
