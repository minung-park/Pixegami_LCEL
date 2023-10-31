from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from packages.models import CHAT_LLM, vector_setting
from packages.functions import print_green, print_blue
from langchain.output_parsers import PydanticOutputParser
from langchain.chains import LLMChain, SequentialChain
from langchain.schema.runnable import RunnableParallel
from langchain.schema.output_parser import StrOutputParser
from packages.custom_parser import CustomCategoryParser, CustomKeywordParser
from operator import itemgetter


# TODO: 4 steps
# TODO: 1. extract category from the input
# TODO: 2. extract keyword from the input
# TODO: 3. use keyword to search
# TODO: 4. use Search result to generate response

vector_store = vector_setting()
retriever = vector_store.as_retriever()

# print(retriever.search_type)
# retriever.get_relevant_documents(query="멀티버스")


class Output(BaseModel):
    category: str = Field(description="Extract category from the input")
    text: str = Field(description="Extract text from the input")
    content_id: list[str] = Field(description="Extract content_id from the input")


prompt_1 = """
Extact category from the input. Answer is only one of the following: general, keyword, content, similar, trending.\n
Choose from the following categories:\n

Categories: ['general', 'keyword', 'content, 'similar', 'trending']
- 'general': Use this category for questions that are not specifically related to OTT programs.
- 'content': This category is for questions that seek detailed information about a program, such as its plot, actors, release dates, cast, and characters.
- 'keyword': Choose this category for inquiries where the user doesn't provide specific program information but asks about actors, release dates, titles, cast, and other keyword-based queries.
- 'similar': Use this category when users are looking for recommendations similar to a specific program.
- 'trending': This category is for questions about programs that need viewing data to provide trending information.\n

input: {input}
"""

prompt_2 = """
Extract keyword from the input. If you do not extract, you must say "None":\n
{input}
"""

prompt_3 = """
As an AI assistant for the OTT platform, Wavve, your main task is to provide truthful responses to user's query based on the information available.
Remember to be polite and avoid using harmful language.

category: {category}\n
input: {keyword}\n
"""
# search_result: {search_result}
# Set up the models
llm3 = CHAT_LLM
# llm4 = CHAT_LLM_4

p1_template = PromptTemplate.from_template(prompt_1)
# chain_1 = LLMChain(llm=llm3, prompt=p1_template, output_key="category")
chain_1 = p1_template | llm3 | StrOutputParser() | CustomCategoryParser()

p2_template = PromptTemplate.from_template(prompt_2)
# chain_2 = LLMChain(llm=llm3, prompt=p2_template, output_key="keyword")
chain_2 = p2_template | llm3 | StrOutputParser() | CustomKeywordParser()

map_chain = RunnableParallel(category=chain_1, keyword=chain_2)

p3_template = PromptTemplate.from_template(prompt_3)




# retrieved_documents = {
#     "search_result": retriever, "keyword": itemgetter("keyword"), "category": itemgetter("category")
# } | p3_template | llm3 | StrOutputParser()
# # https://python.langchain.com/docs/modules/chains/foundational/sequential_chains
# #
# response = retrieved_documents.invoke({"input": "멀티버스가 뭐야?"})
# print(response)


#
# chain_3 = {"keyword": map_chain["keyword"],
#            "category": map_chain["category"],
#            "search_result": retrieved_documents } | p3_template | llm3 | StrOutputParser() # https://python.langchain.com/docs/modules/chains/foundational/sequential_chains
# #



input = [
    "멀티버스가 뭐야?", # general
    "오늘의 날씨", # general
    "쑥의 효능", # general
    "멀티버스", # keyword
    "런닝맨 아이돌 나오는 편 찾아줘", # content
    "하정우가 출연한 영화 알려줘", # keyword
    "해운대는 무슨 내용이야?", # content
    "상견니랑 비슷한 영화 추천해줘", # similar
    "나혼자 산다 출연진 알려줘", # content
    "약한 영웅이랑 비슷한 콘텐츠 추천해줘", # similar
    "현재 인기작 알려줘", # trending
    "웬만해서 그들을 막을 수 없다", # content
    "한번 다녀왔습니다." # content
]

for i in input:
    output = map_chain.invoke({"input": i})
    print(itemgetter("category", "keyword")(output))
    # print(output)
    # print(custom_parser.parse_result(itemgetter("category")(output)))

# for i in range(len(input)):
#     # response = map_chain.invoke({"input": input[i]})
#     # print(chain_3)
#     response = chain_3.invoke({"input": input[i]})
#     print_blue(f"Response for GPT3: {response}")
#     print("\n")


