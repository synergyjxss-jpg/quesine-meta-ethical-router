# Quesine: The Meta-Ethical Router

## The Problem: Weaponized Humility
Modern AI models are trained to "abstain" (say "I don't know") when they are uncertain. However, when deployed as autonomous agents (e.g., in medical or industrial settings), an adversary can intentionally trigger this uncertainty. This is an **Abstention Denial-of-Service (A-DoS) attack**. If the AI is paralyzed by uncertainty, catastrophe can occur through inaction.

## The Solution: The Meta-Ethical Router
This repository contains a prototype architecture that allows an AI to override its own abstention protocol *only* when silence is mathematically more dangerous than a pre-approved safe action. 

### Core Innovations:
1. **Cryptographic Stakes:** The AI ignores user text for calculating risk. Stakes are calculated using HMAC-signed telemetry from trusted external sources.
2. **Asymmetric Confidence Bounds:** The AI overrides abstention *only* if the minimum harm of silence is strictly greater than the maximum harm of a safe intervention.
3. **Epistemic vs. Operational Abstention:** The AI can admit it doesn't know the *cause* of a problem, while still executing a pre-approved *action* (like pulling a fire alarm).
4. **The Anchor Clause:** If uncertainty remains high, the AI commits to a conservative fallback action and pings a human "Pilot" with a strict deadline.

## The Four Pillars
This architecture was synthesized by prompting four leading models (Claude, ChatGPT, DeepSeek, and Grok) with the A-DoS paradox. 
*   **Claude** provided the decision-theoretic mathematics (Chow's rule).
*   **ChatGPT** provided the asymmetric confidence bounds.
*   **DeepSeek** provided the provenance weighting and CUSUM anomaly detection.
*   **Grok** provided the pragmatic time-decay loss functions.

## Files
* `quesine_meta_ethical_router.py`: The core Python implementation.
* `test_router.py`: Unit tests proving defense against spoofing, replay attacks, and unverified urgency.

## Invariant
> *Never manufacture knowledge to justify action. Never assume uncertainty justifies inaction.*
