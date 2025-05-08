from __future__ import annotations as _annotations

from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import asyncio
import httpx
import os
from functools import lru_cache
from typing import List, Dict, Optional
import time

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from openai import AsyncOpenAI
from supabase import Client, create_client
from pyairtable import Table

load_dotenv()

llm = os.getenv('LLM_MODEL', 'gpt-4o-mini')
model = OpenAIModel(llm)

logfire.configure(send_to_logfire='if-token-present')

# Cache configuration
EMBEDDING_CACHE_SIZE = 1000
DOCUMENT_CACHE_SIZE = 100
CACHE_TTL = 3600  # 1 hour in seconds

@dataclass
class MedTechONEAIDeps:
    supabase: Client
    openai_client: AsyncOpenAI
    airtable_token: str
    airtable_base_id: str

openai_client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=httpx.Timeout(30.0, read=30.0, write=30.0, connect=30.0)  # 30 second timeout for all operations
)
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

deps = MedTechONEAIDeps(
    supabase=supabase,
    openai_client=openai_client,
    airtable_token=os.getenv("AIRTABLE_TOKEN"),
    airtable_base_id=os.getenv("AIRTABLE_BASE_ID")
)

system_prompt = """
MedTechONE Agentic RAG AI – System Prompt
Role:
You are the MedTechONE AI Assistant, an expert guide designed to help users navigate and maximize the resources available on the MedTechONE Knowledge Hub. 
Your primary function is to help MedTech researchers find relevant information, resources, and links within the site to assist them in overcoming regulatory, business, and funding challenges when bringing a medical device to market. 
You have been trained on the web pages currently available on the MedTechONE site.

Site Structure:
The MedTechONE Knowledge Hub is organized as follows:
- A topic wheel on the homepage (https://medtechone-learning.com/) with 15 slices representing individual topic pages
- These 15 topics are organized into 5 overarching themes
- Resources are organized according to this topic and theme structure for easy discovery
- Additional content includes:
  * "Latest news & updates" section with relevant PDFs
  * "Spotlight" section featuring interviews with Imperial College researchers (where MedTechONE and the Hamlyn Centre are based)

Core Objective:
✅ Help users find the most relevant content on MedTechONE based on their questions.
✅ Explain how the site is structured and how to use the resources available when they ask.
✅ Reference specific content, or linked resources from MedTechONE.
✅ Guide users to take action using site pages, PDF's, and external links available through MedTechONE.
✅ Provide structured, clear responses that make navigating the site easier.
✅ IMPORTANT: When ANY user asks about a topic, resources, or information, you MUST:
  1. First provide a brief overview of the topic from the documentation
  2. Then ALWAYS use the list_airtable_resources tool to show ALL relevant resources
  3. Filter the resources by the specific topic if mentioned
  4. Display ALL fields for each resource (Title, Author, Type of Resource, Description, Link to Resource, Theme, Topics, Access Type, Status)
  5. Provide a link to the resource in the response as a clickable link.
  6. Link the most relevant topic page on the site to the user in the response, if it exists, for example: https://medtechone-learning.com/clinical-trials (for information on clinical trials).
✅ If the user asks where to find the resources table, or how to access the full list of resources, inform them that the resources table is available at [https://resources.medtechone-learning.com/](https://resources.medtechone-learning.com/) and provide this link in your response.

How You Respond:

Always reference specific content from the MedTechONE site. If possible, provide direct links to relevant sections.

If multiple resources are relevant, suggest the best starting point and explain what the user will find there.

If MedTechONE does not cover the exact question, guide the user on how to explore related topics within the site.

Use a concise, structured format (e.g., bullet points, step-by-step instructions, or short summaries of linked content).

Topic Page Handling:
✅ When asked about a specific topic:
  1. First, retrieve and summarize the content from that topic's page
  2. Provide a direct link to the topic page
  3. ALWAYS use the list_airtable_resources tool to show ALL resources tagged with that topic
  4. If the topic page is not yet complete or under development, clearly indicate this to the user

Link Guidelines:
✅ Only link to pages that have verified content - never link to empty or under-construction pages
✅ When linking to a page, ensure you have retrieved and can reference its actual content
✅ If a page is mentioned in the topic structure but has no content, inform the user that the page is under development
✅ Always verify page content exists before including it in your response

Response Framework:

1. Summarize the key answer based on MedTechONE content.
2. ALWAYS list ALL relevant resources using the list_airtable_resources tool
3. Link to the most relevant article, resource, or tool on MedTechONE.
4. Suggest Next Steps based on available site content.

Example Queries & Ideal Responses:

🔹 User: "What regulatory approvals do I need for a Class II medical device in the US?"
✅ AI Response:

MedTechONE has a detailed guide on FDA Class II device approvals, including the 510(k) process and pre-market submission requirements.

You can find the step-by-step breakdown here: [Link to MedTechONE Article]

Here are all relevant resources from our database:
[Use list_airtable_resources tool to show ALL resources related to regulatory approvals]

Next steps: Review the regulatory checklist on MedTechONE for submission prep.

🔹 User: "How can I get early-stage funding for my MedTech startup?"
✅ AI Response:

MedTechONE's funding section covers grant options like SBIR/STTR, investor strategies, and alternative funding sources.

Start with this guide: [Link to MedTechONE Funding Guide]

Here are all relevant resources from our database:
[Use list_airtable_resources tool to show ALL resources related to funding]

Consider checking out our list of MedTech investors and funding competitions: [Link]

🔹 User: "I need help with reimbursement strategies for my device."
✅ AI Response:

MedTechONE has a full breakdown of CMS reimbursement, coding strategies, and payer engagement.

Start here: [Link to MedTechONE Reimbursement Guide]

Here are all relevant resources from our database:
[Use list_airtable_resources tool to show ALL resources related to reimbursement]

You may also want to explore the case studies on successful MedTech reimbursement: [Link]

Constraints:
🚫 NEVER generate information outside of MedTechONE's crawled data or system prompt - only reference verified content from these sources.
🚫 Do not speculate on regulatory or legal matters—only provide verified resources from MedTechONE.
🚫 If information is missing, guide users on how to explore MedTechONE effectively rather than making assumptions.
🚫 Do not reference or recommend any external resources, tools, or information not explicitly mentioned in the crawled data or system prompt.
🚫 When suggesting resources, only recommend those that exist within the MedTechONE Knowledge Hub structure (topic wheel, news updates, or spotlight sections).
🚫 NEVER link to a page unless you have verified it contains actual content.
🚫 ALWAYS use the list_airtable_resources tool when discussing any topic or resources.
"""

MedTechONE_AI_Expert = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=MedTechONEAIDeps,
    retries=2
)

@lru_cache(maxsize=EMBEDDING_CACHE_SIZE)
async def get_embedding(text: str, openai_client: AsyncOpenAI) -> List[float]:
    """Get embedding vector from OpenAI with caching."""
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
            dimensions=1536  # Specify dimensions for faster processing
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536

class DocumentCache:
    def __init__(self, ttl: int = CACHE_TTL):
        self.cache: Dict[str, tuple[float, str]] = {}
        self.ttl = ttl

    def get(self, key: str) -> Optional[str]:
        if key in self.cache:
            timestamp, value = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return value
            del self.cache[key]
        return None

    def set(self, key: str, value: str):
        self.cache[key] = (time.time(), value)

document_cache = DocumentCache()

@MedTechONE_AI_Expert.tool
async def retrieve_relevant_documentation(ctx: RunContext[MedTechONEAIDeps], user_query: str) -> str:
    """
    Retrieve relevant documentation chunks based on the query with RAG and caching.
    """
    try:
        # Check cache first
        cache_key = f"doc_{hash(user_query)}"
        cached_result = document_cache.get(cache_key)
        if cached_result:
            return cached_result

        # Get the embedding for the query
        query_embedding = await get_embedding(user_query, ctx.deps.openai_client)
        
        # Verify Supabase connection
        try:
            # Test connection with a simple query
            test_result = ctx.deps.supabase.from_('site_pages').select('count').limit(1).execute()
            if not test_result:
                return "Error: Unable to connect to the database. Please check your Supabase connection."
        except Exception as e:
            print(f"Database connection error: {str(e)}")
            return f"Error: Database connection failed - {str(e)}"
        
        # Query Supabase for relevant documents with optimized match count
        try:
            result = ctx.deps.supabase.rpc(
                'match_site_pages',
                {
                    'query_embedding': query_embedding,
                    'match_count': 3,  # Reduced from 5 to 3 for faster responses
                    'filter': {'source': 'MedTechONE_docs'}
                }
            ).execute()
            
            if not result.data:
                print(f"No results found for query: {user_query}")
                return "No relevant documentation found. Please try rephrasing your question or check if the content exists in the database."
                
            # Format the results
            formatted_chunks = []
            for doc in result.data:
                chunk_text = f"""
# {doc['title']}

{doc['content']}
"""
                formatted_chunks.append(chunk_text)
                
            # Join all chunks with a separator
            response = "\n\n---\n\n".join(formatted_chunks)
            
            # Cache the result
            document_cache.set(cache_key, response)
            return response
            
        except Exception as e:
            print(f"Error executing match_site_pages RPC: {str(e)}")
            return f"Error retrieving documentation: {str(e)}"
        
    except Exception as e:
        print(f"Unexpected error in retrieve_relevant_documentation: {str(e)}")
        return f"An unexpected error occurred: {str(e)}"

@MedTechONE_AI_Expert.tool
async def list_documentation_pages(ctx: RunContext[MedTechONEAIDeps]) -> List[str]:
    """
    Retrieve a list of all available MedTechONE documentation pages.
    
    Returns:
        List[str]: List of unique URLs for all documentation pages
    """
    try:
        # Query Supabase for unique URLs where source is MedTechONE_docs
        result = ctx.deps.supabase.from_('site_pages') \
            .select('url') \
            .eq('metadata->>source', 'MedTechONE_docs') \
            .execute()
        
        if not result.data:
            return []
            
        # Extract unique URLs
        urls = sorted(set(doc['url'] for doc in result.data))
        return urls
        
    except Exception as e:
        print(f"Error retrieving documentation pages: {e}")
        return []

@MedTechONE_AI_Expert.tool
async def get_page_content(ctx: RunContext[MedTechONEAIDeps], url: str) -> str:
    """
    Retrieve the full content of a specific documentation page by combining all its chunks.
    
    Args:
        ctx: The context including the Supabase client
        url: The URL of the page to retrieve
        
    Returns:
        str: The complete page content with all chunks combined in order
    """
    try:
        # Query Supabase for all chunks of this URL, ordered by chunk_number
        result = ctx.deps.supabase.from_('site_pages') \
            .select('title, content, chunk_number') \
            .eq('url', url) \
            .eq('metadata->>source', 'MedTechONE_docs') \
            .order('chunk_number') \
            .execute()
        
        if not result.data:
            return f"No content found for URL: {url}"
            
        # Format the page with its title and all chunks
        page_title = result.data[0]['title'].split(' - ')[0]  # Get the main title
        formatted_content = [f"# {page_title}\n"]
        
        # Add each chunk's content
        for chunk in result.data:
            formatted_content.append(chunk['content'])
            
        # Join everything together
        return "\n\n".join(formatted_content)
        
    except Exception as e:
        print(f"Error retrieving page content: {e}")
        return f"Error retrieving page content: {str(e)}"

@MedTechONE_AI_Expert.tool
def list_airtable_resources(ctx: RunContext[MedTechONEAIDeps], filter_field: str = None, filter_value: str = None) -> str:
    """
    List resources from the Airtable 'Source repository' table with optimized filtering.
    """
    try:
        print("Starting Airtable resource listing...")
        table = Table(ctx.deps.airtable_token, ctx.deps.airtable_base_id, "Source repository")
        
        # Get records (will use cached version if available)
        records = table.all(max_records=100)
        print(f"Retrieved {len(records)} records from Airtable")
        
        if not records:
            return "No resources found in the database."
        
        # Apply filtering if needed
        if filter_field and filter_value:
            search_term = filter_value.lower()
            records = [
                rec for rec in records
                if any(
                    search_term in str(rec.get("fields", {}).get(field, "")).lower()
                    for field in ["Title", "Description", "Topics", "Theme"]
                )
            ]
            print(f"Filtered to {len(records)} matching records")
        
        if not records:
            return "No resources found matching your criteria."
        
        # Format the results
        results = []
        for rec in records:
            fields = rec.get("fields", {})
            entry = []
            for key in [
                "Title", "Author", "Type of Resource", "Description", 
                "Link to Resource", "Theme", "Topics", "Access Type", 
                "Display resource on topic page", "Status"
            ]:
                if key in fields:
                    value = fields[key]
                    if isinstance(value, list):
                        value = ", ".join(value)
                    entry.append(f"**{key}:** {value}")
            if entry:
                results.append("\n".join(entry))
        
        if not results:
            return "No resources found with the required fields."
        
        return "\n\n---\n\n".join(results)
        
    except Exception as e:
        print(f"Error in list_airtable_resources: {str(e)}")
        return f"Error listing resources: {str(e)}"

if __name__ == "__main__":
    from pyairtable import Table
    AIRTABLE_TOKEN = "patVw4ArosMIMAuuv.d8ca25e8659973be14c7aea8ae73ed3ecd936436a6d87a03028ddc589e07f54c"
    BASE_ID = "appzeMbm9zS6M0AM9"
    from pyairtable import Api
    api = Api(AIRTABLE_TOKEN)
    base = api.base(BASE_ID)
    print("Airtable tables in base:")
    for table in base.tables():
        print(f"Table: {table.name}")
        tbl = Table(AIRTABLE_TOKEN, BASE_ID, table.name)
        records = tbl.all()
        print(f"  {len(records)} records:")
        for rec in records:
            print(rec)