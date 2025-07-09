#!/usr/bin/env python3

from rich.console import *
from rich.prompt import Prompt
from rich.padding import Padding
from rich.pretty import Pretty
from rich.markdown import Markdown
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents import AgentType
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from datetime import datetime, timedelta


load_dotenv()
console = Console()

llm = ChatOpenAI(
    model_name=os.environ.get('OPENAI_MODEL_NAME'),
    temperature=0
)

# Set a random seed for reproducibility
np.random.seed(42)
# Generate sample data
n_rows = 1000

# Generate dates
start_date = datetime(2022, 1, 1)
dates = [start_date + timedelta(days=i) for i in range(n_rows)]

# Define data categories
makes = ['Toyota', 'Honda', 'Ford', 'Chevrolet', 'Nissan', 'BMW', 'Mercedes', 'Audi', 'Hyundai', 'Kia']
models = ['Sedan', 'SUV', 'Truck', 'Hatchback', 'Coupe', 'Van']
colors = ['Red', 'Blue', 'Black', 'White', 'Silver', 'Gray', 'Green']

# Create the dataset
data = {
    'Date': dates,
    'Make': np.random.choice(makes, n_rows),
    'Model': np.random.choice(models, n_rows),
    'Color': np.random.choice(colors, n_rows),
    'Year': np.random.randint(2015, 2023, n_rows),
    'Price': np.random.uniform(20000, 80000, n_rows).round(2),
    'Mileage': np.random.uniform(0, 100000, n_rows).round(0),
    'EngineSize': np.random.choice([1.6, 2.0, 2.5, 3.0, 3.5, 4.0], n_rows),
    'FuelEfficiency': np.random.uniform(20, 40, n_rows).round(1),
    'SalesPerson': np.random.choice(['Alice', 'Bob', 'Charlie', 'David', 'Eva'], n_rows)
}

# Create DataFrame and sort by date
df = pd.DataFrame(data).sort_values('Date')
console.print(df)

# Create the Pandas DataFrame agent
agent = create_pandas_dataframe_agent(
    llm,
    df,
    return_intermediate_steps=True,
    allow_dangerous_code=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)

console.print("""
[green]Data Analysis Agent is ready. You can now ask questions about the data.[/]
""")

try:
    while True:
        user_input = Prompt.ask("[bold yellow]You[/]")

        console.print()
        console.print("[bold green]AI:[/]")
        with console.status("calling the LLM"):
            steps = agent.iter(
                {
                    "input": user_input,
                    "agent_scratchpad": f"""
                        Human: {user_input}\nAI: To answer this question, I need to use Python to analyze the dataframe.
                        I'll use the python_repl_ast tool and pandas lib only.

                        Action: python_repl_ast

                        Action Input:
                    """,
                }
            )
            for step in steps:
                if output := step.get("intermediate_step"):
                    console.print(Padding(Pretty(output), (0, 0, 0, 2)))
                if output := step.get("output"):
                    console.print(Padding(Markdown(output), (2)))


except KeyboardInterrupt:
    ()
