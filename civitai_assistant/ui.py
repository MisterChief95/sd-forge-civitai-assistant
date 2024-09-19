from collections.abc import Callable, Generator, Sequence
from inspect import signature
from typing import TypeVar

import gradio as gr


T = TypeVar("T")


def progressify_sequence(
    items: Sequence[T], lower_bound: float = 0.25, upper_bound: float = 0.95
) -> "Generator[tuple[T, float], None, None]":
    """
    Calculate the number of steps and progress step size for updating metadata.
    Args:
        num_descriptors (int): The number of model descriptors.
    Returns:
        tuple[int, float]: A tuple containing the number of steps and the progress step size.
    """
    num_steps = items if isinstance(items, int) else len(items)
    progress_step = (upper_bound - lower_bound) / num_steps

    for i, item in enumerate(items):
        yield item, lower_bound + progress_step * i


def create_progressable_button(
    button_text: str, progressable_fn: Callable[[], None], inputs: list[gr.components.Component] = []
):
    """
    Creates a button that shows a progress label when clicked and executes a given function.
    Args:
        button_text (str): The text to display on the button.
        progressable_fn (Callable[[], None]): A function to execute when the button is clicked.
            This function must have a parameter with a default value of `gr.Progress`.
    Raises:
        AssertionError: If `progressable_fn` does not have a parameter with a default value of `gr.Progress()`.
    Returns:
        None
    """

    assert any(
        isinstance(param.default, gr.Progress) for param in signature(progressable_fn).parameters.values()
    ), f"Function '{progressable_fn.__name__}' must have a parameter with default value of gr.Progress()"

    button = gr.Button(button_text)
    progress_label = gr.Label("", visible=False, label="Processing change", show_label=True)

    button.click(lambda: [gr.Button(visible=False), gr.Label(visible=True)], outputs=[button, progress_label]).then(
        progressable_fn, inputs=inputs, outputs=progress_label
    ).then(lambda: [gr.Button(visible=True), gr.Label(visible=False)], outputs=[button, progress_label])
