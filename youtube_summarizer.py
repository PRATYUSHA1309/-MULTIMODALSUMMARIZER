import re
import torch
import gradio as gr
from transformers import pipeline
from youtube_transcript_api import YouTubeTranscriptApi
from pathlib import Path


model_path = Path("..") / "models" / "models--sshleifer--distilbart-cnn-12-6" / "snapshots" / "a4f8f3ea906ed274767e9906dbaede7531d660ff"

text_summary = pipeline("summarization", model= model_path, torch_dtype= torch.bfloat16)
# Load the model from Hugging Face hub (use online path for Spaces)
# model_name = "sshleifer/distilbart-cnn-12-6"
# text_summary = pipeline("summarization", model=model_name, torch_dtype=torch.float32)

# Function to summarize text
def summary(input):
    max_input_chars = 3500
    if len(input) > max_input_chars:
        input = input[:max_input_chars]
    output = text_summary(input)
    return output[0]['summary_text']

# Extract the YouTube video ID from various URL formats
def extract_video_id(url):
    regex = r"(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})"
    match = re.search(regex, url)
    return match.group(1) if match else None

# Main summarization function from YouTube transcript
def get_youtube_transcript(video_url):
    video_id = extract_video_id(video_url)
    if not video_id:
        return "Video ID could not be extracted.", None

    try:
        # Fetch transcript
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        text_transcript = " ".join([x["text"] for x in transcript])

        # Get summary
        summary_text = summary(text_transcript)

        # Save summary to a file
        output_path = "summary.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(summary_text)

        return summary_text, output_path
    except Exception as e:
        return f"An error occurred: {str(e)}", None

# Gradio interface
gr.close_all()

demo = gr.Interface(
    fn=get_youtube_transcript,
    inputs=gr.Textbox(label="Input YouTube URL to summarize", lines=1),
    outputs=[
        gr.Textbox(label="Summarized Text", lines=4),
        gr.File(label="Download Summary")
    ],
    title="@GenAILearniverse Project 2: YouTube Script Summarizer",
    description="This application summarizes the transcript of a YouTube video."
)

if __name__ == "__main__":
    demo.launch()
