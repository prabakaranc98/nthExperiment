# Planning & Task Decomposition

**One-liner:** Explicitly emit a plan or tree/graph of subgoals before (plan-and-solve) or during (interleaved, self-revising) execution, then dispatch them to an executor — the lever that buys long-horizon reliability by separating the "what/order" decision from the "do" step.

## The pattern (planner-executor split)

```
plan        = Planner(task)                 # ordered list / DAG of subgoals g_1..g_k
for g in plan:                              # or topological order over the DAG
    result  = Executor(g, context)          # tool call, sub-agent, code, retrieval
    context = update(context, result)
    if Monitor(context, plan):              # replanning trigger
        plan = Planner(task, context)       # self-revision: re-plan from new state
```

Two axes: (1) **static vs. interleaved** — generate the whole plan up front vs. plan-step-act-observe loops; (2) **linear vs. tree/graph** — a chain of subgoals vs. branching search over decompositions. Decomposition cuts a horizon-H task into k subtasks of horizon ~H/k, so per-step error ε compounds over fewer dependent steps; the executor's context per subgoal stays bounded.

## Where it appears

- **Plan-and-Solve (Wang et al., 2023)** — "devise a plan, then carry it out" prompt; the zero-shot precursor that separates planning from solving to cut CoT calculation errors.
- **ReAct / Reflexion** — interleaved plan-act-observe; Reflexion adds verbal self-revision of the plan after failures.
- **Tree-of-Thought / LATS / MCTS-style search** — decomposition becomes an explicit search tree with a value/verifier scoring partial plans (deliberate search vs. greedy chaining).
- **Frontier agents (Claude/Codex/Devin-class, deep-research agents)** — an explicit planner emits a todo/task list, then a coding or browsing executor (often a sub-agent with isolated context) works each item; orchestrators replan on tool failures.

## Common mistake

Treating the plan as a contract instead of a hypothesis. Static up-front plans rot the moment the environment returns something unexpected; without a replanning/monitor trigger the executor faithfully marches a stale plan off a cliff. The reliability win comes from the *revision loop*, not from the act of planning once.

## See also
- [[react]] — interleaved reason-act-observe; the canonical executor loop a planner feeds
- [[multi-agent-orchestration]] — planner-as-orchestrator dispatching subgoals to executor sub-agents
- [[tree-of-thought]] — decomposition lifted into explicit branching search over subgoals
