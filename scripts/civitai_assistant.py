import gradio as gr
import os

from lib_civitaiassistant.types import ModelType
from lib_civitaiassistant.helper import update_metadata

from modules.shared import cmd_opts
from modules import script_callbacks


css_path = os.path.join(os.path.dirname(os.path.abspath(__name__)), "styles.css")


def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False, css=css_path) as civitai_assistant_view:
        with gr.Column(variant="compact", elem_id="civitai_assistant_column"):
            with gr.Row(elem_classes="civitai_assistant_row"):
                model_checkboxes = gr.CheckboxGroup(
                    [member.value for member in ModelType],
                    label="Models",
                    info="Select models to update info for",
                    elem_id="ch_model_checkbox_group",
                )
            with gr.Row(elem_classes="civitai_assistant_row"):
                with gr.Row():
                    metadata_btn = gr.Button("Update Metadata", elem_id="update_tags_button")
                    # check_updates_btn = gr.Button("Check For Updates", elem_id="check_updates_button")

                metadata_btn.click(fn=update_metadata, inputs=model_checkboxes)

    return [(civitai_assistant_view, "Civitai Assistant", "civitai_assistant_tab")]


script_callbacks.on_ui_tabs(on_ui_tabs)
