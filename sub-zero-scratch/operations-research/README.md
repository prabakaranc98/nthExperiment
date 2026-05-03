# Operations Research

**Type:** Lens + Craft
**Serves:** The mathematical discipline of optimal decision-making under constraints — the rigorous backbone of Decision Engineering. OR is what happens when you take "what should we do?" seriously as a mathematical question.

---

## Why This

Operations Research is the most underrated field in ML. It predates machine learning, contains decades of theory on optimization, scheduling, routing, queuing, and resource allocation — and almost no ML researcher knows it. The connection to Decision Engineering is direct: OR is the classical framework for prescriptive analytics. Understanding it at depth means you can see where ML replaces OR, where OR constrains ML, and where the hybrid is the right answer.

---

## Core Areas

### Linear and Integer Programming
The foundational formalism: objective function, constraints, feasible region. LP gives you continuous optima; ILP handles combinatorial problems.

**Build:** Formulate and solve 3 real problems using PuLP or CVXPY — a scheduling problem, a resource allocation problem, a routing problem.

### Convex Optimization
The geometry of optimization. When is a problem convex? What does that guarantee about the solution? Gradient descent works because (many) ML objectives are approximately convex in practice — but why?

**Build:** Implement gradient descent, Newton's method, and projected gradient descent on the same problem. Compare convergence.

### Stochastic Optimization and Robust Optimization
The real world is uncertain. Stochastic programming optimizes in expectation; robust optimization optimizes against worst-case realizations. Both are essential for decision systems that operate under uncertainty.

### Dynamic Programming and Bellman
DP is the bridge from OR to RL. The Bellman equation is the foundation of both. Understanding DP deeply — not just as "memoization" but as a principle for sequential decision problems — changes how you think about RL.

**Build:** Implement value iteration and policy iteration on a small MDP from scratch. No libraries.

### Queuing Theory
How do systems behave under load? M/M/1 queues, Little's Law, stability conditions. Directly relevant to ML systems: inference serving, training job scheduling.

---

## Build Progression

- [ ] **Step 1:** Formulate a real scheduling problem as an ILP. Solve with CVXPY. Verify optimality.
- [ ] **Step 2:** Read CVX Book (Boyd & Vandenberghe) Ch. 1–4. Identify which ML training objectives are convex and which are not.
- [ ] **Step 3:** Implement value iteration and policy iteration from scratch on GridWorld.
- [ ] **Step 4:** Model a simple queuing system (ML inference server). Derive the expected latency under different load levels using M/M/1 theory. Verify with simulation.
- [ ] **Step 5:** Connect to Decision Engineering: write a one-page note on where OR and ML meet in a decision system.

---

## Essential Resources

- [ ] **Introduction to Operations Research — Hillier & Lieberman**
  - Ch. 1–3: LP formulation and the simplex method
  - Ch. 11: Dynamic programming
  - Ch. 20: Queuing theory

- [ ] **Convex Optimization — Boyd & Vandenberghe** *(free online)*
  - Ch. 1–4: Convex sets, functions, problems, duality
  - Ch. 9: Unconstrained minimization algorithms

- [ ] **Reinforcement Learning: An Introduction — Sutton & Barto**
  - Ch. 3–4: MDPs and dynamic programming *(the OR-RL bridge)*

- [ ] **Introduction to Linear Optimization — Bertsimas & Tsitsiklis** *(selected)*
  - Ch. 1–2: Geometry of LP, simplex

---

## Build Log

| # | Problem Formulated | Method | Result | Date |
|---|-------------------|--------|--------|------|
| — | — | — | — | — |
