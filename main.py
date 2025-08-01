import os
from dotenv import load_dotenv

from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_ollama import OllamaLLM

from langchain_core.prompts import PromptTemplate
from langchain.agents import create_react_agent, AgentExecutor

# Define a single, consistent model for the agent and tools
llm = OllamaLLM(model="llama3", temperature=0)

# Connect to your database
db = SQLDatabase.from_uri("postgresql+psycopg2://postgres:root@localhost:5432/dvdrental")

# Create the toolkit and tools
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()

# Step 1: Define Your Few-Shot Examples
# We'll use a simple string format for maximum compatibility.
examples = [
    (
        "Question: How many artists are in the database?\n"
        "Thought: I need to count the number of rows in the `actor` table. I will use the `sql_db_query` tool for this.\n"
        "Action: sql_db_query\n"
        "Action Input: SELECT count(*) FROM actor;"
    ),
    (
        "Question: List all films in the comedy category.\n"
        "Thought: I need to find the film titles for films in the 'Comedy' category. I'll need to join the `film`, `film_category`, and `category` tables. First, I'll list the tables to confirm their existence.\n"
        "Action: sql_db_list_tables\n"
        "Action Input: "
    )
]

# Step 2: Create the Final Prompt Template (Corrected)
# Step 2: Create the Final Prompt Template
# This is a very strict, one-shot prompt designed to force the correct output format.
template = """
You are an agent designed to interact with a SQL database.
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

Begin!

Question: {input}
{agent_scratchpad}
"""

# Create the final prompt template with all variables explicitly defined
full_prompt = PromptTemplate(
    template=template,
    input_variables=['input', 'tools', 'tool_names', 'agent_scratchpad']
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
# question = "How many artists are there in the database?"
question = "Can you list down films with least rental duration?"
response = agent_executor.invoke({"input": question})

print("\nFinal Answer:")
print(response["output"])