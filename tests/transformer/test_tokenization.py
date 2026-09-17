import pytest

from transformer.configs.config import TrainingConfig
from transformer.preprocessing.prompts import encode_input, encode_target, make_input, parse_sections


@pytest.fixture(scope="module")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(TrainingConfig().model_name, local_files_only=True)
    except OSError:
        pytest.skip("Tokenizer not cached; the real training dry-run covers tokenization after download")


def test_long_input_keeps_news_and_grounding_instructions(tokenizer):
    value = make_input("A study " * 100, "Topic " * 100, "Question " * 100,
                       ["PaperStart " + "paper text " * 200],
                       ["NewsStart " + "news text " * 200])
    item = encode_input(tokenizer, value, 256)
    decoded = tokenizer.decode(item["input_ids"], skip_special_tokens=True)
    assert len(item["input_ids"]) <= 256
    assert "Do not invent citations" in decoded
    assert "PaperStart" in decoded and "NewsStart" in decoded
    assert "Research Question:" in decoded and "Instruction:" in decoded


def test_long_target_preserves_all_three_sections(tokenizer):
    text = "\n\n".join(name + ":\n" + "Evidence supports collective participation. " * 50 for name in ("Introduction", "Body", "Conclusion"))
    ids = encode_target(tokenizer, text, 96)
    assert len(ids) <= 96
    assert len(parse_sections(tokenizer.decode(ids, skip_special_tokens=True), strict=True)) == 3
