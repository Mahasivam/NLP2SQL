from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain.agents import create_react_agent, AgentExecutor
from flask import Flask, request, jsonify
from flask_cors import CORS

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app)  # This enables CORS for all routes

# Define a single, consistent model for the agent and tools
llm = OllamaLLM(model="llama3", temperature=0.3)

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
        "Question: What is the total number of actors?\n"
        "Thought: The user wants to count the total number of actors. The `actor` table is the correct place to find this. I will count all rows in the table.\n"
        "Action: sql_db_query\n"
        "Action Input: SELECT COUNT(*) FROM actor;\n"
        "Observation: [(200,)]\n"
        "Thought: I have the final count from the query result.\n"
        "Final Answer: There are 200 actors in the database."
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

**IMPORTANT: Once you have the information needed, you must provide the final answer and stop the process. Do not loop.**

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

# # Step 4: Run a sample query
# question = "how many films are there in database?"
# response = agent_executor.invoke({"input": question})
#
# print("\nFinal Answer:")
# print(response["output"])

# Step 4: Define the Flask route for the API endpoint
@app.route('/query', methods=['POST'])
def handle_query():
    try:
        data = request.get_json()
        question = data.get('question')

        if not question:
            return jsonify({
                'status': 'error',
                'answer': 'Question field is missing in the request body.'
            }), 400

        # Invoke the agent executor with the user's question
        response = agent_executor.invoke({"input": question})
        final_answer = response["output"]

        return jsonify({
            'status': 'success',
            'answer': final_answer
        })

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({
            'status': 'error',
            'answer': f'An internal server error occurred: {str(e)}'
        }), 500


# Step 5: Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
