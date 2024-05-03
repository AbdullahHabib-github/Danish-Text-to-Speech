"""
doc string
"""
import logging
import os
import time
import uuid
import gradio as gr
import soundfile as sf
from model import get_pretrained_model

title = "# Danish Text To Speech"
css = """
.result {display:flex;flex-direction:column}
.result_item {padding:15px;margin-bottom:8px;border-radius:15px;width:100%}
.result_item_success {background-color:mediumaquamarine;color:white;align-self:start}
.result_item_error {background-color:#ff7070;color:white;align-self:start}
"""

def process(text: str, sid: str):
    """
    doc string
    """
    repo_id = "csukuangfj/vits-piper-da_DK-talesyntese-medium"
    speed = 1
    sid = int(sid)
    tts = get_pretrained_model(repo_id, speed)
    start = time.time()
    audio = tts.generate(text, sid = sid)
    if len(audio.samples) == 0:
        raise ValueError(
            "Error in generating audios. Please read previous error messages."
        )
    audio_dir = "audios"  
    os.makedirs(audio_dir, exist_ok=True) 

    # Generate the filename and construct the full path
    filename = str(uuid.uuid4()) + ".wav" 
    filename = os.path.join(audio_dir, filename)

    sf.write(
        filename,
        audio.samples,
        samplerate = audio.sample_rate,
        subtype = "PCM_16",
    )
    return filename

demo = gr.Blocks(css=css)
with demo:
    gr.Markdown(title)
    with gr.Tabs():
        with gr.TabItem("Please input your text"):
            input_text = gr.Textbox(
                label="Input text",
                info="Your text",
                lines=3,
                placeholder="Please input your text here",
            )
            input_sid = gr.Textbox(
                label="Speaker ID",
                info="Speaker ID",
                lines=1,
                max_lines=1,
                value="0",
                placeholder="Speaker ID. Valid only for mult-speaker model",
                visible = False
            )
            input_button = gr.Button("Submit")

            output_audio = gr.Audio(label="Output")

            output_info = gr.HTML(label="Info")
        input_button.click(
            process,
            inputs=[
                input_text,
                input_sid
            ],
            outputs=[
                output_audio
            ],
        )

def download_espeak_ng_data():
    """
    doc string
    """
    os.system(
    """
    cd /tmp
    wget -qq https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/espeak-ng-data.tar.bz2
    tar xf espeak-ng-data.tar.bz2
    """
    )

if __name__ == "__main__":
    download_espeak_ng_data()
    demo.launch(share = True)
