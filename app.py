import argparse
import asyncio
import os

import torch

import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)

os.environ['RWKV_JIT_ON'] = '1'
os.environ["RWKV_CUDA_ON"] = '0' # '1' if torch.cuda.is_available() else '0' 

from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal
from textual.reactive import reactive
from textual.widgets import Button, Header, Footer, Markdown, Input, Static

# I fucking hate this library
class OptionsContainer(Container):
    pass

class RWKVApp(App):

    TITLE = "RWKV App"
    CSS_PATH = "rwkv.css"

    md_text = reactive("")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", required=True, type=str)
        # TODO strategy and stuff

        args = parser.parse_args()

        self.model = RWKV(model=args.model, strategy="cuda fp16")
        self.pipeline = PIPELINE(self.model, "20B_tokenizer.json")

    @property
    def markdown_viewer(self) -> Markdown:
        """Get the Markdown widget."""
        return self.query_one("#md-output", Markdown)

    @property
    def input_widget(self) -> Input:
        """Get the Input widget."""
        return self.query_one("#input", Input)

    def on_mount(self) -> None:
        self.input_widget.focus()
        # self.markdown_viewer.focus()
        # self.markdown_viewer.show_table_of_contents = (False)
        # self.markdown_viewer.update("### Test")

    def compose(self) -> ComposeResult:
        yield Header()
        with Container(id="main"):
            yield Input(id="input", placeholder="Prompt")
            # with Horizontal():
            with OptionsContainer():
                yield Button("Generate", id="generate", classes="dock-left", variant="primary")
                yield Button("Clear", id="clear", classes="dock-right", variant="error")
            # with Horizontal():
            with OptionsContainer():
                yield Static("Temperature", classes="label")
                yield Input("1.0", id="temperature")
            with OptionsContainer():
                yield Static("Max tokens", classes="label")
                yield Input("16", id="max_tokens")
            yield Markdown("Output will be here!", id="md-output")
        yield Footer()

    async def watch_md_text(self, md_text: str) -> None:
        await self.markdown_viewer.update(md_text)
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        # global model, pipeline

        button_id = event.button.id
        assert button_id is not None

        # Sever lack of ~~fe..~~ switch case? :monocle:
        if button_id == "generate":
            self.md_text = self.input_widget.value
            args = PIPELINE_ARGS(
                token_ban=[0],
                temperature=float(self.query_one("#temperature", Input).value)
            )
            token_count = int(self.query_one("#max_tokens", Input).value)
            def cb(text: str):
                print(text)
                self.md_text = self.md_text + text
            # Trying to make it async doesn't do anything from my experience
            # async def generate(token_count):
            #     self.pipeline.generate(self.input_widget.value, token_count=token_count, args=args, callback=cb, state=None)
            # asyncio.create_task(generate(token_count))
            self.pipeline.generate(self.input_widget.value, token_count=token_count, args=args, callback=cb, state=None)
        elif button_id == "clear":
            self.input_widget.value = ""

if __name__ == "__main__":
    # global model, pipeline
    app = RWKVApp()
    app.run()