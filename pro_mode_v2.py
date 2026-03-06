"""should emulate grok-4.20 agentic pipeline with a chatroom and A2A messages"""

import asyncio
import os
from openai import AsyncOpenAI
from typing import Dict, Any
import re

class ProModeV32:
    def __init__(self, api_key: str, model: str = "gpt-5-mini"):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.print_lock = asyncio.Lock()
        self.transcript = ["=== Live Chatroom Transcript ==="]

    async def _decompose(self, query: str) -> Dict[str, str]:
        prompt = f"""You are the Lead Orchestrator.
Query: {query}

You MUST output EXACTLY three lines and nothing else:

HARPER: precise research/verification subtask
BENJAMIN: code/algorithms/math subtask
LUCAS: creative synthesis subtask"""
        resp = await self.client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": prompt}],
            #temperature=0.0, max_tokens=200
        )
        subtasks = {}
        for line in resp.choices[0].message.content.strip().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                subtasks[k.strip().upper()] = v.strip()
        if len(subtasks) < 3:
            subtasks = {"HARPER": "Verify facts", "BENJAMIN": "Technical validation", "LUCAS": "Creative synthesis"}
        return subtasks

    async def _agent_response(self, name: str, role_prompt: str, query: str, history: str, private_note: str = "", first_turn: bool = False) -> tuple[str, str | None]:
        prompt = f"""{role_prompt}

You are part of a three-agent team (Harper, Benjamin, Lucas) working together with the Lead Orchestrator in a shared chatroom.
You can address other agents directly:
@Harper: message
@Benjamin: message
@Lucas: message
or send private messages:
PRIVATE_TO_HARPER: message
PRIVATE_TO_BENJAMIN: message
PRIVATE_TO_LUCAS: message

Query: {query}
Current chatroom (last 4 messages):
{history}

Private note from Lead: {private_note or "None"}

{'You MUST reply on the first turn.' if first_turn else 'Reply ONLY if you have genuinely new value or the private note asks you.'}

End with:

FINAL: [your concise answer]
Otherwise reply with exactly [SILENT]"""

        full = ""
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            #temperature=0.75,
            #max_tokens=1400,
            stream=True
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            full += delta

        if not first_turn and "[SILENT]" in full.upper():
            return name, None
        return name, full.strip()

    async def run(self, query: str) -> Dict[str, Any]:
        print("Lead: Decomposing the query...")
        subtasks = await self._decompose(query)
        print(subtasks, "\n")

        print("=== Live Chatroom ===\n")
        print("Lead: Query received. Harper, Benjamin, Lucas — initial thoughts please.\n")

        history = ["Lead: Query received. Harper, Benjamin, Lucas — initial thoughts please."]
        self.transcript.append(history[0])

        role_prompts = {
            "Harper": "You are Harper — precise researcher and fact verifier. You are part of a team with Benjamin and Lucas.",
            "Benjamin": "You are Benjamin — code and math specialist. You are part of a team with Harper and Lucas.",
            "Lucas": "You are Lucas — creative thinker and synthesizer. You are part of a team with Harper and Benjamin."
        }

        agent_list = [
            ("Harper", subtasks.get("HARPER", "Verify facts")),
            ("Benjamin", subtasks.get("BENJAMIN", "Technical validation")),
            ("Lucas", subtasks.get("LUCAS", "Creative synthesis"))
        ]

        # === First mandatory round ===
        print("--- Agents giving initial thoughts ---\n")
        tasks = []
        task_to_name = {}
        for name, _ in agent_list:
            coro = self._agent_response(name, role_prompts[name], query, "\n".join(history[-4:]), first_turn=True)
            task = asyncio.create_task(coro)
            tasks.append(task)
            task_to_name[id(task)] = name

        for completed in asyncio.as_completed(tasks):
            name, msg = await completed
            if msg:
                block = f"{name}:\n{msg}"
                history.append(block)
                self.transcript.append(block)
                async with self.print_lock:
                    print(block)
                    print()

        # === Continuous dynamic loop ===
        while True:
            print("--- Agents thinking ---\n")

            tasks = []
            task_to_name = {}
            for name, _ in agent_list:
                coro = self._agent_response(name, role_prompts[name], query, "\n".join(history[-4:]))
                task = asyncio.create_task(coro)
                tasks.append(task)
                task_to_name[id(task)] = name

            for completed in asyncio.as_completed(tasks):
                name, msg = await completed
                if msg:
                    block = f"{name}:\n{msg}"
                    history.append(block)
                    self.transcript.append(block)
                    async with self.print_lock:
                        print(block)
                        print()

            print("Lead: ", end="", flush=True)
            decide_prompt = f"""You are the Lead Orchestrator.
Current chatroom:
{"\n".join(history[-12:])}

The initial round is complete. If the pros and cons are already well-covered, finalize now.
Decide now:
DECISION: FINALIZE or CONTINUE
NOTE: [one short sentence]"""
            decide_resp = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": decide_prompt}],
                #temperature=0.25, max_tokens=180
            )
            decide_text = decide_resp.choices[0].message.content.strip()
            print(decide_text)
            self.transcript.append(f"Lead: {decide_text}")
            history.append(f"Lead: {decide_text}")

            if "FINALIZE" in decide_text.upper():
                break

            # Private ping if needed
            private = decide_text.lower()
            for name in ["Harper", "Benjamin", "Lucas"]:
                if name.lower() in private:
                    print(f"Lead (private to {name}): ", end="", flush=True)
                    _, msg = await self._agent_response(name, role_prompts[name], query, "\n".join(history[-4:]), decide_text)
                    if msg:
                        block = f"{name}:\n{msg}"
                        history.append(block)
                        self.transcript.append(block)
                        async with self.print_lock:
                            print(block)
                            print()

        print("\nLead: Synthesizing final answer...\n")
        synth_prompt = f"""You are the Lead Orchestrator. Deliver one clean final answer.
Query: {query}

Chatroom: {" | ".join(self.transcript[-30:])}

Final answer only:"""
        final = ""
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": synth_prompt}],
            #temperature=0.2,
            #max_tokens=1500,
            stream=True
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            print(delta, end="", flush=True)
            final += delta
        print()

        self.transcript.append(f"Final Synthesis: {final.strip()}")
        return {
            "final": final.strip(),
            "transcript": "\n".join(self.transcript)
        }

# CLI
if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "How many 'R's are in 'srawberry'?"
    
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")  # or paste directly
    
    pm = ProModeV32(api_key=OPENAI_API_KEY, model="gpt-5-mini")
    try:
        result = asyncio.run(pm.run(query))
        print("\n\n\n\nFinal:", result["final"])
#        print("\n" + result["transcript"])
    except KeyboardInterrupt:
        print("\n\nInterrupted. Printing transcript so far...")
        print("\n" + "\n".join(pm.transcript))
