# https://github.com/mahnrws/huggingface-llm/blob/master/hugginfacellm.py

import gradio as gr

def process(pdf_file, question, mode):
    return "Please enter a question or select Summary"

with gr.Blocks() as demo:
    with gr.Row():
        pdf_input = gr.File(type="filepath", file_types=[".pdf"], label="Upload your PDF")
    with gr.Row():
        question = gr.Textbox(label="Enter your question (leave empty for Summary)")
        mode = gr.Radio(["QnA", "Summary"], value="Summary", label="Mode")
    output = gr.Textbox(label="Output")
    submit = gr.Button("Submit")
    submit.click(fn=process, inputs=[pdf_input, question, mode], outputs=output)

demo.launch()
