import logging
import os

import gradio as gr

from lib_civitaiassistant.types import ModelType
from lib_civitaiassistant.helper import update_metadata, update_preview_images

from modules.shared import opts
from modules import script_callbacks


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

is_debug = getattr(opts, "is_debug", False)

if is_debug:
    logger.setLevel(logging.DEBUG)


css_path = os.path.join(os.path.dirname(os.path.abspath(__name__)), "styles.css")


def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as civitai_assistant_view:
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
                    overwrite_checkbox = gr.Checkbox(
                        label="Overwite Existing Tags/Images",
                        elem_id="civitai_assistant_checkbox",
                    )

            with gr.Row(elem_classes="civitai_assistant_row"):
                with gr.Row():
                    metadata_btn = gr.Button("Update Tags", elem_id="update_tags_button")
                    preview_btn = gr.Button("Update Preview Image", elem_id="update_preview_button")
                    check_updates_btn = gr.Button("Check For Updates", elem_id="check_updates_button")

                metadata_btn.click(fn=update_metadata, inputs=[model_checkboxes, overwrite_checkbox])
                preview_btn.click(fn=update_preview_images, inputs=[model_checkboxes, overwrite_checkbox])

    return [(civitai_assistant_view, "Civitai Assistant", "civitai_assistant_tab")]


script_callbacks.on_ui_tabs(on_ui_tabs)
