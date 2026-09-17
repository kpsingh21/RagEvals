import os, re, glob, json, random
from dotenv import load_dotenv
from deepeval.synthesizer import Synthesizer
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from deepeval.synthesizer import Synthesizer
from deepeval.models import DeepEvalBaseLLM
from langchain_ollama import ChatOllama

load_dotenv()



class OllamaDeepEvalModel(DeepEvalBaseLLM):

    def __init__(self, model_name):
        self.model = ChatOllama(
            model=model_name,
            temperature=0
        )

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        response = self.model.invoke(prompt)
        return response.content

    async def a_generate(self, prompt: str) -> str:
        response = await self.model.ainvoke(prompt)
        return response.content

    def get_model_name(self):
        return "Ollama"

ollama_model = OllamaDeepEvalModel(
    "gpt-oss:120b-cloud"
)

synthesizer = Synthesizer(
    model=ollama_model
)

# --- reuse your own VTT cleaning + chunking (same as the retriever) ---
def load_chunks():
    texts = []
    for path in glob.glob("data/*.vtt"):
        with open(path) as f:
            lines = [ln.strip() for ln in f
                     if ln.strip() and ln.strip() != "WEBVTT" and "-->" not in ln]
        texts.append(" ".join(lines))
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    return splitter.split_text("\n\n".join(texts))


# --- generate ---
chunks = load_chunks()
sample = random.sample(chunks, min(15, len(chunks)))     # ~12 chunks -> keep the set small
contexts = [[c] for c in sample]                          # each context = one chunk


# deepeval set-ollama --model=gpt-oss:120b-cloud
# synthesizer = Synthesizer(model="gpt-oss:120b-cloud")                # the generator/critic model -- pin it
# synthesizer = Synthesizer(model=MyCustomLLM())               

goldens = synthesizer.generate_goldens_from_contexts(
    contexts=contexts,
    include_expected_output=True,       # <-- THIS gives you the ideal_answer
    max_goldens_per_context=1,          # 1 question per chunk -> ~12 goldens
)


# --- convert to YOUR schema (id / query / ideal_answer / source) ---
rows = []
for i, g in enumerate(goldens, 1):
    rows.append({
        "id": f"g{i:03d}",
        "query": g.input,
        "ideal_answer": g.expected_output,
        "source": "TODO-verify",        # Synthesizer won't know the session -- you fill this
    })

with open("goldens/retriever_deepeval_goldens.json", "w") as f:
    json.dump(rows, f, indent=2, ensure_ascii=False)

print(f"wrote {len(rows)} DRAFT goldens -> goldens/component_goldens_draft.json")
print("!! REVIEW EVERY ONE before using: check grounding, trim padding, fix leading questions.")