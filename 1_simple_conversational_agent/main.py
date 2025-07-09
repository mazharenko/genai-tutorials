#!/usr/bin/env python3
"""
A simple conversational agent console application.
This application allows users to interact with a basic chatbot that maintains
context across multiple interactions.

Example human messages:
Hello! How are you?
What was my previous message?
"""

from rich.console import *
from rich.prompt import Prompt
from rich.padding import Padding
from rich.pretty import Pretty
from rich.markdown import Markdown
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
import os
from dotenv import load_dotenv

load_dotenv()
console = Console()

llm = ChatOpenAI(
    model_name=os.environ.get('OPENAI_MODEL_NAME')
)


prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | llm
session_id = "user_123"
history = ChatMessageHistory()
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda _: history,
    input_messages_key="input",
    history_messages_key="history"
)

try:
    while True:
        user_input = Prompt.ask("[bold yellow]You[/]")
        result = chain_with_history.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )
        console.print()
        console.print("[bold green]AI:[/]")
        console.print(Padding(
            Group(
                Markdown(result.content),
                "",
                Pretty(result, max_depth=2, max_string=20),
            ),
            (0, 2, 0, 2))
        )
        console.print()

except KeyboardInterrupt:
    ()
