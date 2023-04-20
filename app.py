import streamlit as st
import pinecone
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory, CombinedMemory, ConversationKGMemory

if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "input" not in st.session_state:
    st.session_state["input"] = ""
if "stored_session" not in st.session_state:
    st.session_state["stored_session"] = []

# Define function to get user input
def get_text():
    """
    Get user input text.
    Returns:
        str: The text entered by the user.
    """
    input_text = st.text_input("You:", st.session_state["input"], key="input",
                               placeholder="Enter your message here...", label_visibility='hidden')
    return input_text

# Define function to start a new chat
def new_chat():
    """
    Clears session state and starts a new chat.
    """
    save = []
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        save.append("User:" + st.session_state["past"][i])
        save.append("Bot:" + st.session_state["generated"][i]['output_text'])      
    st.session_state["stored_session"].append(save)
    st.session_state["generated"] = []
    st.session_state["past"] = []
    st.session_state["input"] = ""
    if "entity_memory" in st.session_state:
        del st.session_state["entity_memory"]

# Set up the Streamlit app layout
st.title("🧠 chatBot with Pinecone and Memory 🤖")
st.markdown(
        ''' 
        > :black[**A Chat Bot that queries your own corpus in Pinecone. The bot has both ConversationBufferMemory and ConversationKGMemory. **  *powered by -  [LangChain]('https://langchain.readthedocs.io/en/latest/modules/memory.html#memory') + 
        [OpenAI]('https://platform.openai.com/docs/models/gpt-3-5') + 
        [Streamlit]('https://streamlit.io')*]
        ''')
# st.markdown(" > Powered by -  🦜 LangChain + OpenAI + Streamlit")

# Sidebar Settings
openai_api = st.sidebar.text_input("OpenAI API Key", type="password")
pinecone_api = st.sidebar.text_input("Pinecone API Key", type="password")
pinecone_env = st.sidebar.text_input("Pinecone Environment")
pinecone_index = st.sidebar.text_input("Pinecone Index")
MODEL = st.sidebar.selectbox("Model", ["gpt-3.5-turbo", "text-davinci-003"])

if openai_api and pinecone_api and pinecone_env and pinecone_index:

    # Create Pinecone Instance
    pinecone.init(api_key=pinecone_api, environment=pinecone_env)

    # Create OpenAI Instance
    llm = OpenAI(
        temperature=0, 
        openai_api_key=openai_api,
        model_name=MODEL,

        )
    
    # Create a ConversationEntityMemory object if not already created
    if 'entity_memory' not in st.session_state:
            KG = ConversationKGMemory(llm=llm, input_key="human_input")
            CBM = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")
            st.session_state["entity_memory"] = CombinedMemory(memories=[KG, CBM])
    

    # Set Template
    template = """You are a chatbot having a conversation with a human.

    Given the following extracted parts of a long document and a question, create a final answer.

    {context}

    Relevant Information:

    {history}

    {chat_history}
    Human: {human_input}
    Chatbot:"""

    # Set the Prompt
    prompt = PromptTemplate(
        input_variables=["chat_history", "human_input", "context", "history"], 
        template=template
    )
            
    # Get Context
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api)
    docsearch = Pinecone.from_existing_index(pinecone_index, embedding=embeddings)
    # Get user input

else:
    st.error("Please enter your API keys in the sidebar.")

input_text = get_text()

if input_text:
    # Fetch docs using user input for cosine similarity
    docs = docsearch.similarity_search(input_text, k=3)

    # Get Response
    chain = load_qa_chain(OpenAI(temperature=0, openai_api_key=openai_api), chain_type="stuff", memory=st.session_state["entity_memory"], prompt=prompt, verbose=True)

    # Generate the output using user input and store it in the session state
    output = chain({"input_documents": docs, "human_input": input_text}, return_only_outputs=True)
    st.session_state.past.append(input_text)
    st.session_state.generated.append(output)

    with st.expander("Conversation"):
        for i in range(len(st.session_state["generated"])-1, -1, -1):
            st.info(st.session_state["past"][i])
            st.success(st.session_state["generated"][i]['output_text'])

# Create button to start a new chat
st.sidebar.button("New Chat", on_click=new_chat, type="primary")