from langchain.schema.output_parser import BaseTransformOutputParser
from enum import Enum
from typing import Any


class Category(Enum):
    GENERAL = "general"
    CONTENT = "content"
    KEYWORD = "keyword"
    SIMILAR = "similar"
    TRENDING = "trending"
    ERROR = "error"


class CustomCategoryParser(BaseTransformOutputParser[str]):
    def __init__(self):
        super().__init__()

    def parse(self, text: str) -> Any:
        return_val = None
        try:
            ans = text.split(":")[-1]
            ans = ans.lower().strip()
            return_val = Category(ans).value
        except ValueError:
            return_val = None
        finally:
            return return_val


class CustomKeywordParser(BaseTransformOutputParser[str]):
    def __init__(self):
        super().__init__()

    def parse(self, text: str) -> Any:
        return_val = None
        try:
            ans = text.split(":")[-1]
            ans = ans.split(",")
            return_val = [a.strip() for a in ans]
        except ValueError:
            return_val = None
        finally:
            return return_val
