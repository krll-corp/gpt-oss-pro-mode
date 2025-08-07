# gpt-oss-pro-mode

**Multi‑agent answer generation & synthesis for any OpenAI‑compatible backend.**

`pro_mode.py` spins up multiple "agents" (candidate completions) at a creative temperature, then runs a low‑temperature synthesis pass that merges the best ideas into one decisive answer — all with real‑time streaming. Use it as a **command‑line tool** or import it as a **Python library**.

---

## Features

| Feature                | Details                                                                  |
| ---------------------- | ------------------------------------------------------------------------ |
| Multi‑agent generation | `n_agents` independent candidates (temperature 0.9)                      |
| Expert synthesis       | Low‑temperature editor merges answers, fixes errors, removes duplication |
| Token streaming        | See reasoning & answer tokens live (`stdout`)                            |
| OpenAI‑compatible      | Works with local Ollama, OpenAI API, or any drop‑in replacement          |
| Library‑friendly       | Safe to `import pro_mode` without triggering CLI code                    |

---

## Installation

AN OPENAI COMPATIBLE SEVER IS REQUIRED

eg. ollama, LM Studio, vLLM, etc

```bash
# clone the repo
 git clone https://github.com/krll-corp/gpt-oss-pro-mode.git
 cd gpt-oss-pro-mode

# create a venv (optional)
 python -m venv .venv && source .venv/bin/activate

# install requirements
 pip install openai
```

> **Backend**: Point the OpenAI client to your inference server via the `OPENAI_BASE_URL` env var **or** edit the `client` line in `pro_mode.py`.
> My setup runs with ollama:
> ```
> ollama pull gpt-oss:20b # download a model (do it once)
> ollama serve # launch completions server, you can also set OLLAMA_CONTEXT_LENGTH=XXX before if the task requires extensive reasoning.
> ```

---

## Command‑line usage

```bash
# simplest invocation
python pro_mode.py "How many R's are in strawberry?"

# advanced
python pro_mode.py "Explain self‑play in reinforcement learning" \
  --model llama3:70b \
  --n_agents 7 # more creativity / more intelligence
  --max_tokens 4096 \
  --no_stream # disable token streaming
```

### CLI flags

| Flag           | Default       | Description                           |
| -------------- | ------------- | ------------------------------------- |
| `prompt`       |  —            | The question / task (positional)      |
| `--model`      | `gpt-oss:20b` | Model name or id                      |
| `--max_tokens` |  30 000       | Per‑request token limit               |
| `--n_agents`   |  5            | Number of candidate answers           |
| `--no_stream`  |  —            | Disable streaming of candidate tokens |

---

## Library usage

```python
from pro_mode import pro_mode

result = pro_mode(
    "How many 'R's are in 'strawberry'?",
    n_agents=3,
    model="gpt-oss:20b",
    stream_candidates=False,
)
print(result["final"])  # -> "There are 3 R's."
```

You receive a dict:

```python
{
  "final": "<synthesised answer>",
  "candidates": ["agent1 text", "agent2 text", ...]
}
```

---

## How it works (under the hood)

1. **Candidate phase** – For each agent, a creative request (`temperature=0.9`) is sent. Tokens stream live: reasoning first, then a `---` separator, then the answer content.
2. **Synthesis phase** – A system prompt appoints the model as an "expert editor". It receives all candidate answers and must return **one** concise, corrected answer (`temperature=0.2`).
3. **Streaming** – The synthesis answer also streams, so you can watch the final wording form in real time.

---

## Configuration tips

* **Retry logic** – `_one_completion` auto‑retries (exponential back‑off) up to three times on API errors.
* **Custom separators** – Hack `_stream_print` if you want a different reasoning/content delimiter.
* **Model agnostic** – Works with GPT‑4o, Mistral, Llama 3, etc., as long as the server speaks the OpenAI REST spec (current implemantation uses gpt-oss:).

---

## Roadmap / ideas

* [ ] Parallel candidate generation (asyncio)
* [ ] Optional JSON mode for structured prompts
* [ ] Confidence scores based on voting / RLAIF

PRs are always welcome!

---

## Special thanks to OpenAI, and ollama for making this project possible!

---

## License

MIT © 2025 Kyryll Kochkin / \$]-\[∆ÐØVV
See [LICENSE](LICENSE) for details.
