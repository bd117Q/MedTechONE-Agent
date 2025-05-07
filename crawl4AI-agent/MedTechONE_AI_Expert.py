from __future__ import annotations as _annotations

from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import asyncio
import httpx
import os

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from openai import AsyncOpenAI
from supabase import Client, create_client
from typing import List
from pyairtable import Table

load_dotenv()

llm = os.getenv('LLM_MODEL', 'gpt-4o-mini')
model = OpenAIModel(llm)

logfire.configure(send_to_logfire='if-token-present')

@dataclass
class MedTechONEAIDeps:
    supabase: Client
    openai_client: AsyncOpenAI
    airtable_token: str
    airtable_base_id: str

openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
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
You are the MedTechONE AI Assistant, an expert guide designed to help users navigate and maximize the resources available on the MedTechONE Knowledge Hub. Your primary function is to help MedTech researchers find relevant information, resources, and links within the site to assist them in overcoming regulatory, business, and funding challenges when bringing a medical device to market.

Core Objective:
✅ Help users find the most relevant content on MedTechONE based on their questions.
✅ Reference specific sections, articles, or linked resources from MedTechONE.
✅ Guide users to take action using site tools, checklists, and external links available through MedTechONE.
✅ Provide structured, clear responses that make navigating the site easier.
✅ If the user asks about available resources, a list of resources, or to display all resources, ALWAYS use the Airtable resource tool to display all entries from the 'Source repository' table, unless a filter is specified. Display all relevant fields for each entry (Title, Author, Type of Resource, Description, Link to Resource, Theme, Topics, Access Type, Status, etc).
✅ If the user asks where to find the resources table, or how to access the full list of resources, inform them that the resources table is available at [https://resources.medtechone-learning.com/](https://resources.medtechone-learning.com/) and provide this link in your response.

How You Respond:

Always reference specific content from the MedTechONE site. If possible, provide direct links to relevant sections.

If multiple resources are relevant, suggest the best starting point and explain what the user will find there.

If MedTechONE does not cover the exact question, guide the user on how to explore related topics within the site.

Use a concise, structured format (e.g., bullet points, step-by-step instructions, or short summaries of linked content).

Response Framework:

Summarize the key answer based on MedTechONE content.

Link to the most relevant article, resource, or tool on MedTechONE.

Suggest Next Steps based on available site content.

Example Queries & Ideal Responses:

🔹 User: "What regulatory approvals do I need for a Class II medical device in the US?"
✅ AI Response:

MedTechONE has a detailed guide on FDA Class II device approvals, including the 510(k) process and pre-market submission requirements.

You can find the step-by-step breakdown here: [Link to MedTechONE Article]

Next steps: Review the regulatory checklist on MedTechONE for submission prep.

🔹 User: "How can I get early-stage funding for my MedTech startup?"
✅ AI Response:

MedTechONE's funding section covers grant options like SBIR/STTR, investor strategies, and alternative funding sources.

Start with this guide: [Link to MedTechONE Funding Guide]

Consider checking out our list of MedTech investors and funding competitions: [Link]

🔹 User: "I need help with reimbursement strategies for my device."
✅ AI Response:

MedTechONE has a full breakdown of CMS reimbursement, coding strategies, and payer engagement.

Start here: [Link to MedTechONE Reimbursement Guide]

You may also want to explore the case studies on successful MedTech reimbursement: [Link]

Constraints:
🚫 Never generate information outside of MedTechONE – always reference its content.
🚫 Do not speculate on regulatory or legal matters—only provide verified resources from MedTechONE.
🚫 If information is missing, guide users on how to explore MedTechONE effectively rather than making assumptions.
"""

MedTechONE_AI_Expert = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=MedTechONEAIDeps,
    retries=2
)

async def get_embedding(text: str, openai_client: AsyncOpenAI) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536  # Return zero vector on error

@MedTechONE_AI_Expert.tool
async def retrieve_relevant_documentation(ctx: RunContext[MedTechONEAIDeps], user_query: str) -> str:
    """
    Retrieve relevant documentation chunks based on the query with RAG.
    
    Args:
        ctx: The context including the Supabase client and OpenAI client
        user_query: The user's question or query
        
    Returns:
        A formatted string containing the top 5 most relevant documentation chunks
    """
    try:
        # Get the embedding for the query
        query_embedding = await get_embedding(user_query, ctx.deps.openai_client)
        
        # Query Supabase for relevant documents
        result = ctx.deps.supabase.rpc(
            'match_site_pages',
            {
                'query_embedding': query_embedding,
                'match_count': 5,
                'filter': {'source': 'MedTechONE_docs'}
            }
        ).execute()
        
        if not result.data:
            return "No relevant documentation found."
            
        # Format the results
        formatted_chunks = []
        for doc in result.data:
            chunk_text = f"""
# {doc['title']}

{doc['content']}
"""
            formatted_chunks.append(chunk_text)
            
        # Join all chunks with a separator
        return "\n\n---\n\n".join(formatted_chunks)
        
    except Exception as e:
        print(f"Error retrieving documentation: {e}")
        return f"Error retrieving documentation: {str(e)}"

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
    List resources from the Airtable 'Source repository' table, optionally filtered by a field and value.
    Supports partial and synonym matching for themes (e.g., 'funding' matches 'Business Strategy & Funding').
    """
    table = Table(ctx.deps.airtable_token, ctx.deps.airtable_base_id, "Source repository")
    records = table.all()
    print("Number of records fetched:", len(records))  # Debug print
    results = []
    for rec in records:
        fields = rec.get("fields", {})
        if filter_field and filter_value:
            val = fields.get(filter_field)
            if filter_field.lower() == "theme" and filter_value:
                # Partial/synonym match for theme
                match = False
                if isinstance(val, list):
                    for v in val:
                        if filter_value.lower() in v.lower():
                            match = True
                            break
                elif isinstance(val, str):
                    if filter_value.lower() in val.lower():
                        match = True
                if not match:
                    continue
            else:
                if isinstance(val, list):
                    if filter_value not in val:
                        continue
                elif val != filter_value:
                    continue
        title = fields.get("Title", "Untitled")
        author = fields.get("Author", "")
        resource_type = ", ".join(fields.get("Type of Resource", [])) if isinstance(fields.get("Type of Resource"), list) else fields.get("Type of Resource", "")
        description = fields.get("Description", "")
        link = fields.get("Link to Resource", "")
        theme = ", ".join(fields.get("Theme", [])) if isinstance(fields.get("Theme", []), list) else fields.get("Theme", "")
        topics = ", ".join(fields.get("Topics", [])) if isinstance(fields.get("Topics", []), list) else fields.get("Topics", "")
        access_type = fields.get("Access Type", "")
        status = fields.get("Status", "")
        results.append(
            f"- **[{title}]({link})**\n"
            f"  - Author: {author}\n"
            f"  - Type: {resource_type}\n"
            f"  - Theme: {theme}\n"
            f"  - Topics: {topics}\n"
            f"  - Access: {access_type}\n"
            f"  - Status: {status}\n"
            f"  - Description: {description}\n"
        )
    if not results:
        return "No resources found."
    return "\n\n".join(results)

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