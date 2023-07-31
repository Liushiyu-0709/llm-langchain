import gradio as gr
import time

from vector_tool import insert_db
from fuc_test import my_run, all_init

# -*- coding: utf-8 -*-
response = ''


def add_text(history, text):
    global response
    # 返回结果response
    print(text)
    response = my_run(text)
    print(response)
    history = history + [(text, None)]
    return history, gr.update(value="", interactive=False)


def bot(history):
    history[-1][1] = ""
    for character in response:
        history[-1][1] += character
        history[-1][1] += '\n'
        time.sleep(0.05)
        yield history


def upload_file(files):
    file_paths = [file.name for file in files]
    # 保存文件
    insert_db(file_paths)
    print(file_paths)
    return file_paths


with gr.Blocks(css=".gradio-container {background-image: linear-gradient(120deg, #e0c3fc 0%, #8ec5fc 100%);#input {"
                   "height:30}") as demo:
    gr.Markdown(
        """
        # Knowledge Master
        """)
    with gr.Row():
        with gr.Column(scale=1):
            file_output = gr.File()
            upload_button = gr.UploadButton("点击上传语料（仅支持txt文件）", file_types=["txt"],
                                            file_count="multiple")
            upload_button.upload(upload_file, upload_button, file_output, show_progress=True)

        with gr.Column(scale=3):
            chatbot = gr.Chatbot().style(height=600)
            with gr.Row():
                with gr.Column(scale=9):
                    txt = gr.Textbox(
                        show_label=False,
                        placeholder="请输入问题",
                        elem_id="input"
                    ).style(container=False)
                with gr.Column(scale=1):
                    clear = gr.ClearButton([txt, chatbot], variant='stop')

    txt_msg = txt.submit(add_text, [chatbot, txt], [chatbot, txt], queue=False).then(
        bot, chatbot, chatbot
    )
    txt_msg.then(lambda: gr.update(interactive=True), None, [txt], queue=False)

demo.queue()
if __name__ == "__main__":
    all_init()
    demo.queue().launch()
