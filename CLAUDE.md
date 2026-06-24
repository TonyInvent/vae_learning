# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

A learning repository for Variational Autoencoders (VAEs) and their applications in robotics reinforcement learning. The user is a student learning this topic and prefers **narrative, tutorial-style introduction documents** over fragmented wiki pages or concept graphs. Write like you're teaching — start from concrete problems, build intuition step by step, use analogies, connect every concept to "why should I care?", and maintain a clear through-line from fundamentals to real-world applications.

## Preferred Format

- **Long-form narrative Markdown documents** with clear sections, progressive difficulty, and natural language flow
- Every document should tell a story: motivation → intuition → math → code → application → what's next
- Use analogies and concrete examples before formal definitions
- Include PyTorch code snippets for key architectures (encoder/decoder, reparameterization, training loop)
- ASCII diagrams are fine but keep them simple — PlantUML via Kroki (`curl -X POST https://kroki.io/plantuml/utxt`) is preferred for complex diagrams
- When a diagram is needed, prefer clean auto-rendered output over hand-crafted ASCII art

## Key Reference

`vae_introduction.md` is the primary document — a comprehensive survey covering VAE fundamentals (ELBO, KL divergence, reparameterization trick), variants (β-VAE, VQ-VAE, q-VAE), the World Models paradigm, the Dreamer family, and robotics applications (navigation, locomotion, cross-embodiment transfer). Use it as the starting point for deeper dives.

## Research Tools

- **Tavily API**: `curl -X POST https://api.tavily.com/search` with the user's API key for sourcing papers and articles
- Built-in `WebSearch` and `WebFetch` tools are available for research without API keys
- When the user asks to learn a topic, research first, then write — don't write from memory alone

## Writing Principles

1. **Story first, taxonomy second.** Start with a concrete problem (e.g., "a robot needs to navigate from pixels"), show why naive approaches fail, then introduce the concept as the natural solution.
2. **Build intuition before math.** Explain what the ELBO *does* before deriving it. Show why the reparameterization trick is clever before proving it works.
3. **Code as illustration.** Include minimal, runnable PyTorch snippets that the user can copy and experiment with.
4. **Connect forward and backward.** When introducing a concept, mention what it enables later. When discussing an application, trace back to which fundamental concept makes it possible.
5. **One coherent narrative per document.** Don't fragment into tiny interlinked pages. Each document should be readable start-to-finish.
