"""
CrewAI Agent Crew Module
Defines agents and tasks for the study assistant
"""

from crewai import Agent, Task, Crew
from langchain.llms.ollama import Ollama
from vector_store import VectorStore
from config import OLLAMA_BASE_URL, OLLAMA_MODEL
from typing import List, Dict


class StudyAssistantCrew:
    """Manages CrewAI agents for document analysis and question answering"""

    def __init__(self):
        self.vector_store = VectorStore()
        self.llm = Ollama(
            base_url=OLLAMA_BASE_URL,
            model=OLLAMA_MODEL
        )
        self.agents = self._create_agents()

    def _create_agents(self):
        """Create specialized agents"""
        
        # Research Agent - Searches knowledge base
        research_agent = Agent(
            role="Research Agent",
            goal="Search and retrieve relevant information from the knowledge base to answer questions",
            backstory="An expert research analyst who excels at finding relevant information from documents",
            llm=self.llm,
            verbose=True
        )

        # Analysis Agent - Analyzes retrieved information
        analysis_agent = Agent(
            role="Analysis Agent",
            goal="Analyze and synthesize retrieved information to create comprehensive answers",
            backstory="A skilled analyst who can process complex information and create clear, coherent responses",
            llm=self.llm,
            verbose=True
        )

        # Review Agent - Validates answers
        review_agent = Agent(
            role="Review Agent",
            goal="Review and validate answers to ensure they are supported by the retrieved content",
            backstory="A critical reviewer who ensures accuracy and reliability of answers",
            llm=self.llm,
            verbose=True
        )

        return {
            "research": research_agent,
            "analysis": analysis_agent,
            "review": review_agent
        }

    def create_crew(self, query: str, retrieved_docs: List[Dict]):
        """Create a crew with tasks for processing a query"""
        
        # Format retrieved documents
        docs_context = self._format_documents(retrieved_docs)

        # Research Task
        research_task = Task(
            description=f"""
            Search and extract relevant information from the knowledge base for the following query:
            Query: {query}
            
            Retrieved Documents:
            {docs_context}
            
            Extract key points and relevant information that could answer the query.
            """,
            agent=self.agents["research"],
            expected_output="Relevant information extracted from documents"
        )

        # Analysis Task
        analysis_task = Task(
            description=f"""
            Analyze the retrieved information and create a comprehensive answer to the query.
            Query: {query}
            
            Use the information provided by the research agent to:
            1. Synthesize the information
            2. Create a clear and coherent answer
            3. Organize the response logically
            """,
            agent=self.agents["analysis"],
            expected_output="A comprehensive answer to the query"
        )

        # Review Task
        review_task = Task(
            description=f"""
            Review the generated answer to ensure:
            1. It directly addresses the query
            2. It is supported by the retrieved documents
            3. It is accurate and reliable
            
            Query: {query}
            """,
            agent=self.agents["review"],
            expected_output="A validated and verified answer with confidence assessment"
        )

        # Create crew
        crew = Crew(
            agents=[
                self.agents["research"],
                self.agents["analysis"],
                self.agents["review"]
            ],
            tasks=[
                research_task,
                analysis_task,
                review_task
            ],
            verbose=True
        )

        return crew

    def process_query(self, query: str) -> Dict:
        """Process a query through the crew"""
        try:
            # Search vector store
            retrieved_docs = self.vector_store.search(query, top_k=5)

            if not retrieved_docs:
                return {
                    "success": False,
                    "answer": "No relevant documents found in the knowledge base.",
                    "retrieved_docs": []
                }

            # Create and run crew
            crew = self.create_crew(query, retrieved_docs)
            result = crew.kickoff()

            return {
                "success": True,
                "answer": result,
                "retrieved_docs": retrieved_docs
            }

        except Exception as e:
            return {
                "success": False,
                "answer": f"Error processing query: {str(e)}",
                "retrieved_docs": []
            }

    @staticmethod
    def _format_documents(docs: List[Dict]) -> str:
        """Format retrieved documents for context"""
        formatted = ""
        for idx, doc in enumerate(docs, 1):
            formatted += f"\n{idx}. [Score: {doc['score']:.2f}]\n{doc['text']}\n"
        return formatted


if __name__ == "__main__":
    crew = StudyAssistantCrew()
    print("Study Assistant Crew initialized successfully")
