import streamlit as st
from ChatModel import *

st.title("Code Llama Assistant")


@st.cache_resource
def load_model():
    model = ChatModel()
    return model


model = load_model()  # load our ChatModel once and then cache it

with st.sidebar:
    temperature = st.slider("temperature", 0.0, 2.0, 0.1)
    top_p = st.slider("top_p", 0.0, 1.0, 0.9)
    max_new_tokens = st.number_input("max_new_tokens", 128, 4096, 256)
    system_prompt = st.text_area(
        "system prompt", value=model.DEFAULT_SYSTEM_PROMPT, height=500
    )


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask me anything!"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        user_prompt = st.session_state.messages[-1]["content"]
        answer = model.generate(
            user_prompt,
            top_p=top_p,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            system_prompt=system_prompt,
        )
        response = st.write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
