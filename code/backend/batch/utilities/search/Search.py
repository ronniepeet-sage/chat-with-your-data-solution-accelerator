from ..search.AzureSearchHandler import AzureSearchHandler
from ..search.IntegratedVectorizationSearchHandler import (
    IntegratedVectorizationSearchHandler,
)
from ..common.SourceDocument import SourceDocument


class Search:
    @staticmethod
    def get_search_handler(env_helper):
        if env_helper.AZURE_SEARCH_USE_INTEGRATED_VECTORIZATION:
            return IntegratedVectorizationSearchHandler(env_helper)
        else:
            return AzureSearchHandler(env_helper)

    @staticmethod
    def get_source_documents(search_handler, question) -> list[SourceDocument]:
        return search_handler.query_search(question)
