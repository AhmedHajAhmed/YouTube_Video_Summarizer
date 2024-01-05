import streamlit as st
from utils import * 
import textwrap

st.title("YouTube Video Summarizer")

with st.sidebar:
    with st.form(key="my_form"):
        youtube_url = st.sidebar.text_area(
            label="YouTube Video URL",
            max_chars=75
        )
        submit_button = st.form_submit_button(label="Submit")


if youtube_url:
    response = get_summary(youtube_url)
    st.subheader("Answer:")
    st.text(textwrap.fill(response, width=80))


