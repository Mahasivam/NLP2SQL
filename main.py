import os
from dotenv import load_dotenv
from langchain_community.llms import Ollama  # Import the Ollama class
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain.agents import AgentType # This import is still valid


# Set up the LLM to use the local Ollama service
# The 'llama3' model should already be downloaded from the previous step
llm = Ollama(model="llama3", temperature=0)

# The rest of the code remains the same as before
# Set up the database connection
db = SQLDatabase.from_uri("postgresql+psycopg2://postgres:root@localhost:5432/dvdrental")

# Create the toolkit for interacting with the database
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# Create the SQL agent
agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True, # Set to True to see the agent's thought process
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    # You might need to experiment with the 'agent_type' for local models
)

# Run a sample query
question = "How many artists are there in the database?"
response = agent_executor.invoke({"input": question})

print("\nFinal Answer:")
print(response["output"])