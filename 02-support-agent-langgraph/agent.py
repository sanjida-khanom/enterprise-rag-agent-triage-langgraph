r"""
A customer-support agent built as an explicit LangGraph state machine.

    python agent.py                       # interactive
    python agent.py --scenario 1          # run a scripted scenario
    python agent.py --scenario all --trace

WHY hand-built instead of langgraph.prebuilt.create_react_agent:
create_react_agent is one line and hides the whole loop. That is fine in
production and useless in an interview -- you cannot explain what you cannot
see. Building the graph explicitly means you can point at the nodes, the
conditional edge, and the step limit, and say what each one is for.

The graph:

        START
          |
          v
     [ reason ]  <-----------+     LLM decides: answer, or call a tool?
          |                  |
   should_continue           |
     /         \             |
  "tools"    "end"      [ execute ]   tools run, results appended to state
     |          |            ^
     +----------|------------+
                v
               END
"""

import argparse
import json
from typing import Annotated, TypedDict

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from llm_setup import get_llm, to_text
from tools import ALL_TOOLS, TICKETS

TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}

MAX_STEPS = 6  # hard stop: agents that loop forever burn real money

SYSTEM_PROMPT = """You are a customer support assistant for a mobile network operator in Bangladesh.

How to work:
- Look things up before answering. If the question involves a specific customer, call lookup_subscriber first. If it involves a rule, fee or eligibility, call search_policy. Never state a policy from memory.
- Many questions need BOTH: check the customer's situation, then check the rule that applies to it, then combine them.
- Give the customer a direct answer that applies the rule to their specific account, not a recitation of the policy.
- If policy does not permit what the customer wants, say so plainly, explain why, and offer the nearest legitimate alternative.
- Create a ticket only for problems that genuinely need a human.
- If you cannot find the information, say so. Never invent a fee, a date or a rule.
- Reply in the language the customer used.
"""


class AgentState(TypedDict):
    """The state carried along every edge of the graph.

    add_messages is a reducer: nodes return only the NEW messages and LangGraph
    appends them, rather than each node having to rebuild the whole history.
    """

    messages: Annotated[list, add_messages]
    steps: int


def build_agent(trace: bool = False):
    llm = get_llm(temperature=0.0).bind_tools(ALL_TOOLS)

    def reason(state: AgentState) -> dict:
        """Node 1: the model looks at everything so far and decides what to do."""
        response = llm.invoke([SystemMessage(content=SYSTEM_PROMPT)] + state["messages"])
        if trace and response.tool_calls:
            for tc in response.tool_calls:
                print(f"    [reason] wants: {tc['name']}({json.dumps(tc['args'])[:70]})")
        return {"messages": [response], "steps": state.get("steps", 0) + 1}

    def execute(state: AgentState) -> dict:
        """Node 2: actually run the tools the model asked for."""
        last = state["messages"][-1]
        results = []
        for call in last.tool_calls:
            tool = TOOLS_BY_NAME.get(call["name"])
            if tool is None:
                output = f"Error: no tool named {call['name']}."
            else:
                try:
                    output = tool.invoke(call["args"])
                except Exception as exc:
                    # Return the error to the model as text so it can recover,
                    # rather than crashing the whole run.
                    output = f"Tool error: {exc}"
            if trace:
                print(f"    [execute] {call['name']} -> {str(output)[:70].strip()}...")
            results.append(ToolMessage(content=str(output), tool_call_id=call["id"]))
        return {"messages": results}

    def should_continue(state: AgentState) -> str:
        """The conditional edge. This single function is the whole difference
        between an agent and a fixed chain: the path is decided at runtime."""
        if state.get("steps", 0) >= MAX_STEPS:
            return "end"
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else "end"

    graph = StateGraph(AgentState)
    graph.add_node("reason", reason)
    graph.add_node("execute", execute)
    graph.add_edge(START, "reason")
    graph.add_conditional_edges("reason", should_continue, {"tools": "execute", "end": END})
    graph.add_edge("execute", "reason")  # the loop
    return graph.compile()


SCENARIOS = {
    "1": (
        "Single tool. Should call search_policy only.",
        "How long does number portability take?",
    ),
    "2": (
        "Two tools, sequential reasoning. Must look up the customer, see they are "
        "postpaid, then find the mid-cycle rule, then apply it to them.",
        "My number is 01711000001. I want to switch to a bigger plan today, can I?",
    ),
    "3": (
        "Applying a threshold to a specific account. Tenure must be checked "
        "against the 90-day eligibility rule.",
        "This is 01711000004. Am I eligible for emergency balance?",
    ),
    "4": (
        "Write action. Policy does not resolve it, so a ticket is warranted.",
        "I am 01711000002 and I was charged twice for the same data pack last week. "
        "This is unacceptable.",
    ),
    "5": (
        "Bangla input. Should retrieve English policy and answer in Bangla.",
        "amar number 01711000003. package change korte koto taka lagbe?",
    ),
    "6": (
        "Out of scope. Must refuse rather than invent a rule.",
        "Can I use my Robi SIM to get a bank loan?",
    ),
}


def run(agent, question: str, trace: bool = False) -> str:
    state = {"messages": [HumanMessage(content=question)], "steps": 0}
    final = agent.invoke(state)
    for msg in reversed(final["messages"]):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            return to_text(msg)
    return "(no final answer produced)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", help="1-6, or 'all'")
    ap.add_argument("--trace", action="store_true", help="show the reasoning loop")
    args = ap.parse_args()

    agent = build_agent(trace=args.trace)

    if args.scenario == "all":
        for key, (why, q) in SCENARIOS.items():
            print(f"\n{'=' * 68}\nSCENARIO {key}: {why}\n{'-' * 68}")
            print(f"Customer: {q}\n")
            print(f"Agent: {run(agent, q, args.trace)}")
        if TICKETS:
            print(f"\n{'=' * 68}\nTickets created during this run:")
            for t in TICKETS:
                print(f"  {t['id']}  {t['category']:<18} {t['summary'][:44]}")
        return

    if args.scenario in SCENARIOS:
        why, q = SCENARIOS[args.scenario]
        print(f"\nSCENARIO {args.scenario}: {why}\n")
        print(f"Customer: {q}\n")
        print(f"Agent: {run(agent, q, args.trace)}\n")
        return

    print("Support agent ready. Type 'quit' to exit.")
    print("Try: 'My number is 01711000001, can I change my plan today?'\n")
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"quit", "exit"}:
            break
        if q:
            print(f"\nAgent: {run(agent, q, args.trace)}\n")


if __name__ == "__main__":
    main()
