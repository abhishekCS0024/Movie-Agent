"""
Movie Recommendation Agent with Streamlit UI
Run with: streamlit run app.py
"""

import streamlit as st
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# Page configuration
st.set_page_config(
    page_title="Movie Recommendation Agent",
    page_icon="🎬",
    layout="wide"
)

# Initialize session state for storing recommendations
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# Define state structure
class Movie(TypedDict):
    messages: Annotated[list[HumanMessage|AIMessage], "the messages in the conversation"]
    Mood: str
    Genre: List[str]
    Language: str
    Platform: str

# Initialize LLM (with error handling)
@st.cache_resource
def get_llm():
    try:
        # You can also use st.secrets for API key management
        api_key = st.secrets.get("GROQ_API_KEY", "YOUR_GROQ_API_KEY")
        return ChatGroq(
            temperature=0,
            groq_api_key=api_key,
            model_name="qwen/qwen3-32b",
            verbose=False
        )
    except Exception as e:
        st.error(f"Error initializing LLM: {e}")
        return None

llm = get_llm()

# Node functions (modified to not use input())
def mood_node(state: Movie):
    state["messages"] = state["messages"] + [
        HumanMessage(content=f"Your mood is {state['Mood']}")
    ]
    return state

def genre_node(state: Movie):
    state["messages"] = state["messages"] + [
        HumanMessage(content=f"Your favourite genres are: {', '.join(state['Genre'])}")
    ]
    return state

def language_node(state: Movie):
    state["messages"] = state["messages"] + [
        HumanMessage(content=f"Your preferred language is: {state['Language']}")
    ]
    return state

def platform_node(state: Movie):
    state["messages"] = state["messages"] + [
        HumanMessage(content=f"Your preferred platform is: {state['Platform']}")
    ]
    return state

def suggestion_node(state: Movie):
    # Create the prompt with actual state values
    movie_recommendation_prompt = ChatPromptTemplate.from_messages([
        ("system",
         f"You are an intelligent movie recommendation agent. "
         f"Based on the user's mood: {state['Mood']}, preferred genres: {', '.join(state['Genre'])}, "
         f"language: {state['Language']}, and available streaming platform: {state['Platform']}, "
         f"recommend suitable movies available on that platform. "
         f"Provide 3-5 specific movie recommendations with:\n"
         f"1. Movie title and year\n"
         f"2. Brief description (2-3 sentences)\n"
         f"3. Why it matches their preferences\n"
         f"Format each recommendation clearly with bullet points."),
        ("human", "Suggest movies for me to watch.")
    ])
    
    # Format the template into messages before invoking
    messages = movie_recommendation_prompt.format_messages()
    response = llm.invoke(messages)
    
    state["messages"] = state["messages"] + [AIMessage(content=response.content)]
    return state

# Build the workflow graph
@st.cache_resource
def build_workflow():
    workflow = StateGraph(Movie)
    
    # Add nodes
    workflow.add_node("input_mood", mood_node)
    workflow.add_node("input_genre", genre_node)
    workflow.add_node("input_language", language_node)
    workflow.add_node("input_platform", platform_node)
    workflow.add_node("suggestion", suggestion_node)
    
    # Add edges to define flow
    workflow.add_edge(START, "input_mood")
    workflow.add_edge("input_mood", "input_genre")
    workflow.add_edge("input_genre", "input_language")
    workflow.add_edge("input_language", "input_platform")
    workflow.add_edge("input_platform", "suggestion")
    workflow.add_edge("suggestion", END)
    
    return workflow.compile()

app = build_workflow()

# Streamlit UI
st.title("🎬 AI Movie Recommendation Agent")
st.markdown("### Get personalized movie recommendations based on your preferences!")

# Sidebar for user inputs
with st.sidebar:
    st.header("📝 Your Preferences")
    
    # Mood selection
    mood = st.selectbox(
        "What's your mood?",
        ["Feel-good", "Thriller", "Mystery", "Emotional", "Light-hearted", "Dark", "Inspirational"],
        help="Select how you're feeling today"
    )
    
    # Genre selection (multi-select)
    genres = st.multiselect(
        "Preferred Genre(s)",
        ["Action", "Drama", "Comedy", "Thriller", "Sci-Fi", "Romance", 
         "Horror", "Documentary", "Animation", "Fantasy", "Crime", "Adventure"],
        default=["Drama"],
        help="You can select multiple genres"
    )
    
    # Language selection
    language = st.selectbox(
        "Preferred Language",
        ["English", "Hindi", "Spanish", "French", "Japanese", "Korean", 
         "German", "Italian", "Chinese", "Any"],
        help="Select your preferred language"
    )
    
    # Platform selection
    platform = st.selectbox(
        "Streaming Platform",
        ["Netflix", "Amazon Prime Video", "Disney+", "HBO Max", "Hulu", 
         "Apple TV+", "Paramount+", "Any Platform"],
        help="Select your streaming platform"
    )
    
    st.markdown("---")
    
    # Get recommendations button
    get_recommendations = st.button("🎯 Get Recommendations", type="primary", use_container_width=True)
    
    # Clear button
    if st.button("🔄 Clear Results", use_container_width=True):
        st.session_state.recommendations = None
        st.session_state.conversation_history = []
        st.rerun()

# Main content area
if get_recommendations:
    if not genres:
        st.warning("⚠️ Please select at least one genre!")
    elif llm is None:
        st.error("❌ LLM not initialized. Please check your API key.")
    else:
        with st.spinner("🎬 Finding perfect movies for you..."):
            try:
                # Prepare state
                state = {
                    "messages": [HumanMessage(content="I want to watch a movie")],
                    "Mood": mood,
                    "Genre": genres,
                    "Language": language,
                    "Platform": platform,
                }
                
                # Run the agent
                result = None
                for output in app.stream(state):
                    result = output
                
                # Extract the final state
                if result:
                    final_state = list(result.values())[0]
                    # Get the last message (AI response)
                    recommendations = final_state["messages"][-1].content
                    st.session_state.recommendations = recommendations
                    st.session_state.conversation_history.append({
                        "mood": mood,
                        "genres": genres,
                        "language": language,
                        "platform": platform,
                        "recommendations": recommendations
                    })
                    
            except Exception as e:
                st.error(f"❌ An error occurred: {e}")

# Display recommendations
if st.session_state.recommendations:
    st.markdown("## 🎥 Your Personalized Recommendations")
    
    # Display user preferences in a nice format
    col1, col2, col3, col4 = st.columns(4)
    
    latest = st.session_state.conversation_history[-1]
    with col1:
        st.metric("Mood", latest["mood"])
    with col2:
        st.metric("Genres", ", ".join(latest["genres"][:2]) + ("..." if len(latest["genres"]) > 2 else ""))
    with col3:
        st.metric("Language", latest["language"])
    with col4:
        st.metric("Platform", latest["platform"])
    
    st.markdown("---")
    
    # Display recommendations
    st.markdown(st.session_state.recommendations)
    
    # Download button
    st.download_button(
        label="📥 Download Recommendations",
        data=st.session_state.recommendations,
        file_name="movie_recommendations.txt",
        mime="text/plain"
    )
else:
    # Welcome message when no recommendations yet
    st.info("👈 Select your preferences from the sidebar and click 'Get Recommendations' to start!")
    
    # Show example
    with st.expander("ℹ️ How it works"):
        st.markdown("""
        This AI-powered agent helps you find the perfect movie by:
        
        1. **Understanding your mood** - Whether you want something thrilling or feel-good
        2. **Matching your genre preferences** - Action, Drama, Comedy, or any combination
        3. **Respecting language preferences** - Movies in your preferred language
        4. **Platform-specific suggestions** - Only recommends what's available on your platform
        
        The AI analyzes all these factors to give you personalized, relevant recommendations!
        """)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>Powered by LangGraph + Groq + Streamlit</p>",
    unsafe_allow_html=True
)
