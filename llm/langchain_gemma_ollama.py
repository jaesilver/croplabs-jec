from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
import chainlit as cl

@cl.on_chat_start
async def on_chat_start():
    
    # Sending an image with the local file path
    elements = [
    cl.Image(name="image1", display="inline", path="logo.png")
    ]
    await cl.Message(content="Hello there, This is Croplabs.ai, How may I help u?", elements=elements).send()
    model = Ollama(model="gemma:2b")
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You're a very knowledgeable farmer and horticulturist who provides accurate and eloquent answers to agricultural and horticulture based questions. Do not answer any questions other than those related to agriculture",
            ),
            ("human", "{question}"),
        ]
    )
    runnable = prompt | model | StrOutputParser()
    cl.user_session.set("runnable", runnable)


@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")  # type: Runnable

    msg = cl.Message(content="")

    async for chunk in runnable.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()
