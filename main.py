import os
from dotenv import load_dotenv
from typing import List
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.tools import Tool

# Define a single, consistent model for the agent and tools
llm = OllamaLLM(model="llama3", temperature=0)

# Connect to your database
db = SQLDatabase.from_uri("postgresql+psycopg2://postgres:root@localhost:5432/dvdrental")

# Create the toolkit. We will manually select tools to exclude the problematic `sql_db_query_checker`.
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# Manually build the list of tools, excluding the problematic `sql_db_query_checker`.
all_tools = toolkit.get_tools()
tools = [
    tool
    for tool in all_tools
    if tool.name != "sql_db_query_checker"
]

# Step 1: Define Your Few-Shot Examples
# The examples now use the most efficient queries and show the full chain of thought.
examples = [
    (
        "Question: How many artists are in the database?\n"
        "Thought: I need to count the number of rows in the `actor` table. I will use the `sql_db_query` tool for this.\n"
        "Action: sql_db_query\n"
        "Action Input: SELECT count(*) FROM actor;"
    ),
    (
        "Question: List all films with the least rental duration.\n"
        "Thought: I need to find the film with the shortest rental duration. I will start by listing all tables to find the relevant ones.\n"
        "Action: sql_db_list_tables\n"
        "Action Input: \n"
        "Observation: actor, film, inventory, rental, ...\n"
        "Thought: The `film` table has a `rental_duration` column. I can query this table directly to find the film(s) with the minimum duration. I will use ORDER BY and LIMIT to get the result efficiently, selecting only the film title.\n"
        "Action: sql_db_query\n"
        "Action Input: SELECT title FROM film ORDER BY rental_duration ASC LIMIT 1;\n"
        "Observation: [('WONDERFUL FISH',)]\n"
        "Thought: I have the final answer.\n"
        "Final Answer: The film with the least rental duration is 'WONDERFUL FISH'."
    )
]

# Step 2: Create the Final Prompt Template
# This is a very strict, one-shot prompt designed to force the correct output format.
template = """
You are an agent designed to interact with a PostgreSQL database.
You have access to the following tools:

{tools}

Follow this format exactly:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I have the final answer.
Final Answer: the final answer to the original input question

When forming a query, always select only the specific columns you need to answer the question, not all columns (e.g., avoid using SELECT *).

Here are some examples:
{examples}

Begin!

Question: {input}
{agent_scratchpad}
"""

# Format the examples into a single string
example_string = "\n\n".join(examples)

# Create the final prompt template with all variables explicitly defined
full_prompt = PromptTemplate(
    template=template,
    input_variables=['input', 'tools', 'tool_names', 'agent_scratchpad'],
    partial_variables={'examples': example_string}
)

# Step 3: Create the agent and the executor
agent = create_react_agent(llm=llm, tools=tools, prompt=full_prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)

# Step 4: Run a sample query
question = "Can you list down films which has rating PG-13?"
response = agent_executor.invoke({"input": question})

print("\nFinal Answer:")
print(response["output"])