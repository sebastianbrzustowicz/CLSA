from typing import TypedDict, Annotated
import operator

class InputText(TypedDict):
    language: str
    text: str

class RawArticle(TypedDict):
    article_id: int
    language: str
    text: str

class TranslatedArticles(TypedDict):
    article_id: int
    source_language: str
    text_en: str

class ModelResult(TypedDict):
    article_id: int
    source_language: str
    model: str
    score: float

class GraphState(TypedDict):
    selected_languages: list[str]
    num_articles: int
    input_text: list[InputText]
    raw_articles: Annotated[list[RawArticle], operator.add]
    translated_articles: Annotated[list[TranslatedArticles], operator.add]
    results: Annotated[list[ModelResult], operator.add]
    summary: str
