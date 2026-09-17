"""Lazy public API: importing it does not load a model or touch CUDA/DB."""


def generate_draft(title, topic=None, research_question=None, paper_evidence=None,
                   news_evidence=None, instruction=None, *, evidence=None,
                   model_path=None, device="auto") -> str:
    from transformer.inference.generate import generate_draft as implementation
    return implementation(title, topic, research_question, paper_evidence, news_evidence,
                          instruction, evidence=evidence, model_path=model_path, device=device)
