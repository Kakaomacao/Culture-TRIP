from langchain_community.utilities import WikipediaAPIWrapper

class CustomWikipediaAPI(WikipediaAPIWrapper):
    top_k_results: int = 1 # 이 부분으로 찾아오는 개수 조절

    def run(self, query: str) -> str:
        """Run Wikipedia search and get page summaries."""
        page_titles = self.wiki_client.search(
            query[:100], results=self.top_k_results
        )
        summaries = []
        for page_title in page_titles[: self.top_k_results]:
            if wiki_page := self._fetch_page(page_title):
                if summary := self._formatted_page_summary(page_title, wiki_page):
                    summaries.append(summary)
        if not summaries:
            return "No good Wikipedia Search Result was found"
        return "\n".join(summaries)[: self.doc_content_chars_max]

    @staticmethod
    def _formatted_page_summary(page_title, wiki_page):
        return f"<document><title>{page_title}</title><content>{wiki_page.summary[:3000]}</content></document>"
