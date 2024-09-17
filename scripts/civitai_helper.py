import gradio as gr

from lib_civitaihelper.types import ModelType
from lib_civitaihelper.helper import update_metadata

from modules.shared import cmd_opts
from modules import script_callbacks

def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False, css="extensions/sd-forge-civitai-helper/html/styles.css") as civitai_helper_view:
        with gr.Column(variant="compact", elem_id="civitai_helper_column"):
            with gr.Row(elem_classes="civitai_helper_row"):
                model_checkboxes = gr.CheckboxGroup(
                    [member.value for member in ModelType],
                    label="Models",
                    info="Select models to update info for",
                    elem_id="ch_model_checkbox_group",
                )
            with gr.Row(elem_classes="civitai_helper_row"):
                with gr.Row():
                    metadata_btn = gr.Button("Update Metadata", elem_id="update_tags_button")
                    # check_updates_btn = gr.Button("Check For Updates", elem_id="check_updates_button")

                metadata_btn.click(fn=update_metadata, inputs=model_checkboxes)

    return [(civitai_helper_view, "Civitai Helper", "civitai_helper_tab")]


script_callbacks.on_ui_tabs(on_ui_tabs)
