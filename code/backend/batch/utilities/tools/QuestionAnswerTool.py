import json
from .AnsweringToolBase import AnsweringToolBase

from langchain.chains.llm import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    AIMessagePromptTemplate,
    FewShotChatMessagePromptTemplate,
)
from langchain_core.messages import SystemMessage
from langchain_community.callbacks import get_openai_callback

from ..helpers.AzureSearchHelper import AzureSearchHelper
from ..helpers.ConfigHelper import ConfigHelper
from ..helpers.LLMHelper import LLMHelper
from ..helpers.EnvHelper import EnvHelper
from ..common.Answer import Answer
from ..common.SourceDocument import SourceDocument


class QuestionAnswerTool(AnsweringToolBase):
    def __init__(self) -> None:
        self.name = "QuestionAnswer"
        self.vector_store = AzureSearchHelper().get_vector_store()
        self.verbose = True
        self.env_helper = EnvHelper()

    def answer_question(self, question: str, chat_history: list[dict], **kwargs: dict):
        config = ConfigHelper.get_active_config_or_default()

        examples = [config.prompts.answering_prompt_example]

        example_prompt = ChatPromptTemplate.from_messages(
            [
                HumanMessagePromptTemplate.from_template(
                    config.prompts.answering_user_prompt
                ),
                AIMessagePromptTemplate.from_template("{answer}"),
            ]
        )

        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=examples,
        )

        answering_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=config.prompts.answering_system_prompt),
                few_shot_prompt,
                SystemMessage(content=self.env_helper.AZURE_OPENAI_SYSTEM_MESSAGE),
                MessagesPlaceholder("chat_history"),
                HumanMessagePromptTemplate.from_template(
                    config.prompts.answering_user_prompt
                ),
            ]
        )

        llm_helper = LLMHelper()

        # Retrieve documents as sources
        sources = self.vector_store.similarity_search(
            query=question, k=4, search_type="hybrid"
        )

        # Generate answer from sources
        answer_generator = LLMChain(
            llm=llm_helper.get_llm(), prompt=answering_prompt, verbose=self.verbose
        )
        documents = json.dumps(
            {
                "retrieved_documents": [
                    {f"[doc{i+1}]": {"content": source.page_content}}
                    for i, source in enumerate(sources)
                ],
            }
        )

        with get_openai_callback() as cb:
            result = answer_generator(
                {
                    "user_question": question,
                    "documents": documents,
                    "chat_history": chat_history,
                }
            )

        answer = result["text"]
        print(f"Answer: {answer}")

        # Generate Answer Object
        source_documents = []
        for source in sources:
            source_document = SourceDocument(
                id=source.metadata["id"],
                content=source.page_content,
                title=source.metadata["title"],
                source=source.metadata["source"],
                chunk=source.metadata["chunk"],
                offset=source.metadata["offset"],
                page_number=source.metadata["page_number"],
            )
            source_documents.append(source_document)

        clean_answer = Answer(
            question=question,
            answer=answer,
            source_documents=source_documents,
            prompt_tokens=cb.prompt_tokens,
            completion_tokens=cb.completion_tokens,
        )
        return clean_answer
