import torch
import gradio as gr
from pathlib import Path
# Use a pipeline as a high-level helper
from transformers import pipeline
from transformers import AutoTokenizer

text_summary = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", torch_dtype= torch.bfloat16)

# model_path = Path("..") / "models" / "models--sshleifer--distilbart-cnn-12-6" / "snapshots" / "a4f8f3ea906ed274767e9906dbaede7531d660ff"

# text_summary = pipeline("summarization", model= model_path, torch_dtype= torch.bfloat16)


def summarize_text(input):
    output = text_summary(input)
    return output[0]['summary_text']

def split_text(text, max_tokens=1024):
    # A rough way to split text based on words (not tokens, but approximate)
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        if len(current_chunk) >= max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# Load tokenizer (match with your model)
tokenizer = AutoTokenizer.from_pretrained(model_path)


def summarize_file(file):
    try:
        # Read file content
        with open(file.name, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        # Tokenize and chunk text
        max_token_length = 1024
        tokens = tokenizer.encode(content, truncation=False)

        # Split into chunks of max_token_length
        chunks = [tokens[i:i + max_token_length] for i in range(0, len(tokens), max_token_length)]

        summaries = []
        for chunk in chunks:
            # Convert token chunk back to text
            chunk_text = tokenizer.decode(chunk, skip_special_tokens=True)
            # Run summarization
            summary = text_summary(chunk_text)[0]['summary_text']
            summaries.append(summary)

        # Join all summaries
        return "\n\n".join(summaries)

    except Exception as e:
        return f"Error processing file: {str(e)}"



gr.close_all()

# demo = gr.Interface(fn=summary, inputs="text", outputs="text")

demo = gr.Interface(fn=summarize_text, 
                    inputs=[gr.Textbox(label="ENTER YOUR TEXT",lines=6)], 
                    outputs=[gr.Textbox(label="SUMMARIZED TEXT",lines=6)],
                    title="A TEXT-SUMMARIZER",
                    description="THIS APPLICATION SUMMARIZES YOUR TEXT"
                    )

# File upload tab
file_demo = gr.Interface(
    fn=summarize_file,
    inputs=gr.File(label="Upload a .txt file"),
    outputs=gr.Textbox(label="Summary from File"),
    title="📁 File Summarizer",
    description="Upload a .txt file to get a summary."
)


gr.TabbedInterface([demo, file_demo], ["Text Input", "File Upload"]).launch()
