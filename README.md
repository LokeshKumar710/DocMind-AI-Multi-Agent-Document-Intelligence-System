# Multi-Agent Document Intelligence System

A multi-agent AI system designed to analyze long unstructured documents such as meeting transcripts, policy documents, legal drafts, and project briefs. The system uses autonomous agents coordinated by an orchestration layer to extract structured insights from complex text.

## Features

- Context-aware document summarization
- Action item extraction with metadata
- Task dependency detection
- Risk and open-issue identification
- Structured JSON output for downstream processing
- Designed to handle long documents using chunking and context sharing

## Architecture

The system consists of multiple specialized AI agents:

### 1. Summary Agent
Generates a concise context-aware summary while preserving key decisions, constraints, and intent.

### 2. Action & Dependency Agent
Extracts actionable tasks along with:
- Task description
- Owners (if mentioned)
- Deadlines
- Dependencies

### 3. Risk & Open Issues Agent
Identifies:
- Unresolved questions
- Missing information
- Assumptions
- Potential risks

### 4. Orchestrator
Coordinates communication between agents and manages context sharing across them.

## Input
Long unstructured document (500+ words recommended)

## Output
Structured JSON containing:
- Summary
- Action Items
- Dependencies
- Risks
- Open Questions

## Use Cases
- Meeting transcript analysis
- Project planning documents
- Legal document review
- Policy analysis
- Enterprise knowledge extraction

## Tech Stack
- Python
- LLMs
- Multi-Agent Orchestration
- Prompt Engineering

## Hackathon
Built for **AidenAI Hackathon 2026**.
