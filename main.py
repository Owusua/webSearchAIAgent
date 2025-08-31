import os
import requests
import json
from typing import List, Dict, Any
import google.generativeai as genai
from datetime import datetime
from dotenv import load_dotenv
from httplib2.auth import params

load_dotenv()

class WebSearchAgent:
    def __init__(self, gemini_api_key: str, search_api_key: str = None, search_engine_id: str = None):

        # Configuring Gemini
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')

        # Search configuration
        self.search_api_key = search_api_key
        self.search_engine_id = search_engine_id

        print("Web Search AI Agent is initialized")

    def search_web(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:

        if self.search_api_key and self.search_engine_id:
            return self._google_search(query, num_results)
        else:
            return self._duckduckgo_search(query, num_results)

    def _google_search(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        try:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                'key': self.search_api_key,
                'cx': self.search_engine_id,
                'q': query,
                'num': min(num_results, 10)
            }

            response = requests.get(url, params=params)
            response.raise_for_status()

            data = response.json()
            results = []

            for item in data.get('items', []):
                results.append({
                    'title': item.get('title', ''),
                    'snippet': item.get('snippet', ''),
                    'link': item.get('link', ''),
                    'source': 'Google'
                })

            return results
        except Exception as e:
            print(f"❌ Google search failed: {e}")
            return self._duckduckgo_search(query, num_results)


    def _duckduckgo_search(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        try:
            url = "https://api.duckduckgo.com/"
            params = {
                'q': query,
                'format': 'json',
                'no_html': '1',
                'skip_disambig': '1'
            }

            response = requests.get(url, params=params)
            response.raise_for_status()

            data = response.json()
            results = []

            if data.get('Abstract'):
                results.append({
                    'title': data.get('Heading', query),
                    'snippet': data.get('Abstract', '')[:200] + '...',
                    'link': data.get('AbstractURL', ''),
                })

            for topic in data.get('RelatedTopics', [])[:num_results - 1]:
                if isinstance(topic, dict) and topic.get('Text'):
                    results.append({
                        'title': topic.get('Text', '')[:50] + '...',
                        'snippet': topic.get('Text', '')[:200] + '...',
                        'link': topic.get('FirstURL', ''),
                        'source': 'DuckDuckGo'
                    })

            if not results:
                results.append({
                    'title': f"Search results for: {query}",
                    'snippet': f"Found search query about {query}. More detailed results may require additional search APIs.",
                    'link': f"https://api.duckduckgo.com/?q={query.replace(' ', '+')}",
                    'source':'DuckDuckGo'
                })
            return results
        except Exception as e:
            print(f"❌ DuckDuckGo search failed: {e}")
            return [{
                'title': f"Search: {query}",
                'snippet': f"Unable to fetch live search results. This would normally search for: {query}",
                'link': f"https://www.google.com/search?q={query.replace(' ', '+')}",
                'source': 'Fallback'
            }]

    def generate_response(self, query: str, search_results: List[Dict[str, Any]]) -> str:
        try:
            context = "Here are the search results: \n\n"
            for i, result in enumerate(search_results, 1):
                context += f"{i}, **{result['title']}**\n"
                context += f"   {result['snippet']}\n"
                context += f"   Source: {result['link']}\n\n"

            prompt = f"""You are helpful AI assistant. Based on the following search results, provide a comprehensive and accurate answer to the user's question.
                     User Question: {query}
                     {context}
                     Please provide a well-structured response that:
                     1. Directly answers the user's question
                     2. Synthesizes information from the search results
                     3. Mentions sources when relevant
                     4. Is clear and easy to understand
                     5. Acknowledges if information is limited or uncertain
                     
                     Response:"""
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"❌ Error generating response: {e}\n\nHere are the raw search results:\n" + \
                "\n".join([f"• {r['title']}: {r['snippet']}" for r in search_results])

    def search_and_answer(self, query: str, num_results: int = 5) -> Dict[str, Any]:
        print(f"🔍 Searching for: {query}")
        # search the web
        search_results = self.search_web(query, num_results)
        print(f"✅ Found {len(search_results)} results")

        # Generate AI response
        print("🤖 Generating AI response...")
        ai_response = self.generate_response(query, search_results)

        return {
            'query': query,
            'search_results': search_results,
            'ai_response': ai_response,
            'timestamp': datetime.now().isoformat()
        }

def main():
    GEMINI_API_KEY = os.getenv('GOOGLE_API_KEY')

    GOOGLE_SEARCH_API_KEY = os.getenv('GOOGLE_SEARCH_API_KEY')
    GOOGLE_SEARCH_ENGINE_ID = os.getenv('GOOGLE_SEARCH_ENGINE_ID')

    try:
        agent = WebSearchAgent(gemini_api_key=GEMINI_API_KEY, search_api_key=GOOGLE_SEARCH_API_KEY, search_engine_id=GOOGLE_SEARCH_ENGINE_ID)

        # Interactive search loop
        print("\n" + "=" * 50)
        print("🚀 Web Search AI Agent Ready!")
        print("Type your questions or 'quit' to exit")
        print("=" * 50 + "\n")

        while True:
            query = input("❓ Ask me anything: ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            if not query:
                continue
            try:
                result = agent.search_and_answer(query)
                print(f"\n🤖 **AI Response:**")
                print("-" * 30)
                print(result['ai_response'])

                print(f"\n📋 **Search Results Used:**")
                print("-" * 30)
                for i, search_result in enumerate(result['search_results'], 1):
                    print(f"{i}. {search_result['title']}")
                    print(f"     {search_result['link']}")
                print("\n" + "=" * 50 + "\n")
            except Exception as e:
                print(f"❌ Error: {e}\n")

    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        print("\n💡 Make sure to:")
        print("1. Install required packages: pip install google-generativeai requests")
        print("2. Set your GEMINI_API_KEY in the code")


if __name__ == "__main__":
    main()










