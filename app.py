
import os
import streamlit as st

from dotenv import load_dotenv

from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate

from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables import RunnableParallel

# ---------------------------------------
# 1. LOAD ENVIRONMENT VARIABLES
# ---------------------------------------

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("Groq API key not found. Check your .env file.")
    st.stop()


# ---------------------------------------
# 2. INITIALIZE GROQ LLM
# ---------------------------------------

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.3,
    api_key=api_key
)

parser = StrOutputParser()


# ---------------------------------------
# 3. CREATE PROMPT TEMPLATES
# ---------------------------------------

explanation_prompt = ChatPromptTemplate.from_template("""
You are a technical tutor.

Explain the following topic in simple language.

Topic: {topic}

Include:
1. Definition
2. Why it is used
3. How it works
4. A simple example
""")

application_prompt = ChatPromptTemplate.from_template("""
You are an experienced software engineer.

Explain one real-world application of the following topic.

Topic: {topic}

Include:
1. Application name
2. Problem it solves
3. How the technology is used
4. Practical example
""")

interview_prompt = ChatPromptTemplate.from_template("""
You are a technical interviewer.

Generate 5 important interview questions and answers
about the following topic.

Topic: {topic}

Keep the answers clear and suitable for freshers.
""")


# ---------------------------------------
# 4. CREATE INDIVIDUAL CHAINS
# ---------------------------------------

explanation_chain = (
    explanation_prompt
    | llm
    | parser
)

application_chain = (
    application_prompt
    | llm
    | parser
)

interview_chain = (
    interview_prompt
    | llm
    | parser
)


# ---------------------------------------
# 5. CREATE DAG USING RUNNABLEPARALLEL
# ---------------------------------------

parallel_chain = RunnableParallel(
    {
        "explanation": explanation_chain,
        "application": application_chain,
        "interview": interview_chain
    }
)


# ---------------------------------------
# 6. CREATE FINAL REPORT PROMPT
# ---------------------------------------

final_prompt = ChatPromptTemplate.from_template("""
You are a technical learning assistant.

Create a well-organized learning report
using the following generated content.

TOPIC: {topic}

SIMPLE EXPLANATION:
{explanation}

REAL-WORLD APPLICATION:
{application}

INTERVIEW QUESTIONS:
{interview}

Organize the final report using these headings:

1. Topic Overview
2. Real-World Application
3. Interview Preparation

Do not invent additional technical facts.
Keep the report beginner-friendly.
""")


# ---------------------------------------
# 7. CONNECT DAG TO FINAL LLM
# ---------------------------------------

# RunnablePassthrough.assign preserves the original
# topic while adding the results of parallel execution.

from langchain_core.runnables import RunnablePassthrough

final_chain = (
    RunnablePassthrough.assign(
        **parallel_chain.steps__
    )
    | final_prompt
    | llm
    | parser
)


# ---------------------------------------
# 8. STREAMLIT USER INTERFACE
# ---------------------------------------

st.set_page_config(
    page_title="DAG Learning Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 AI Topic Learning Assistant")

st.write(
    "Learn any technical topic using an AI-powered "
    "LangChain DAG workflow."
)

topic = st.text_input(
    "Enter a technical topic",
    placeholder="Example: LangGraph, RAG, AI Agents"
)

generate = st.button(
    "Generate Learning Report",
    type="primary"
)


# ---------------------------------------
# 9. EXECUTE THE APPLICATION
# ---------------------------------------

if generate:

    if not topic.strip():

        st.warning("Please enter a topic.")

    else:

        try:

            with st.spinner(
                "AI is generating your learning report..."
            ):

                result = final_chain.invoke({
                    "topic": topic
                })

            st.success("Learning report generated!")
            st.markdown(result)

        except Exception as e:

            st.error(f"Something went wrong: {e}")


# ---------------------------------------
# 10. SHOW WORKFLOW INFORMATION
# ---------------------------------------

with st.expander("How does this DAG work?"):

    st.markdown("""
    ### Workflow

    1. User enters a technical topic.
    2. Three independent chains execute in parallel.
    3. First chain generates a simple explanation.
    4. Second chain generates a real-world application.
    5. Third chain generates interview questions.
    6. Results are combined.
    7. Final LLM generates a learning report.

    **Technologies:**
    - Python
    - LangChain
    - Groq API
    - RunnableParallel
    - Streamlit
    """)
