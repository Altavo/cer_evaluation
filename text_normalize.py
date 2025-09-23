from num2words import num2words
import torch
import jiwer

ALLOWED_LANGUAGES = ["de-DE", "en-US"]


class NormalizeSentence(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: list[str]):
        if not isinstance(x, list):
            x = [x]

        return [" ".join(normalize_sentence(t)) for t in x]


ALLOWED_LANGUAGES = ["de-DE", "en-US"]


cer_transform = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.RemoveWhiteSpace(replace_by_space=True),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveEmptyStrings(),
        jiwer.ReduceToListOfListOfChars(),
    ]
)

wer_transform = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveWhiteSpace(replace_by_space=True),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.RemoveEmptyStrings(),
        jiwer.ReduceToListOfListOfWords(word_delimiter=" "),
    ]
)

special_char_replace = {
    "u\u0308": "ü",
    "a\u0308": "ä",
    "o\u0308": "ö",
    "\xa0": " ",
    "\u00BD": "einhalb",
}


def normalize_sentence(
    sentence: str,
    language: str = "de",
) -> str:
    """
    Normalize sentence.
    :param sentence: Sentence to normalize.
    :param language: Language of sentence.
    :return: Normalized sentence.
    """

    # Replace special characters
    for (
        k,
        v,
    ) in special_char_replace.items():
        sentence = sentence.replace(
            k,
            v,
        )

    # Convert numbers to words
    sentence_norm = wer_transform(sentence)[
        0
    ]  # TODO: This strips punctuation, numbers like "19.1" will be wrong
    for (
        idx,
        word,
    ) in enumerate(sentence_norm):
        if word.isnumeric():
            try:
                sentence_norm[idx] = num2words(
                    word,
                    lang=language,
                )
            except Exception as e:
                raise Exception(
                    f"Error in sentence: '{sentence_norm}', number '{word}' to words: {e}"
                )

    return sentence_norm