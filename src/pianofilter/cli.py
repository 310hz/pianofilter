"""Command-line interface."""

import json
from pathlib import Path

import typer

from pianofilter.analyze import analyze
from pianofilter.features import extract_features
from pianofilter.judge import judge
from pianofilter.piano import PianoScorer

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])
app = typer.Typer(
    add_completion=False,
    context_settings=CONTEXT_SETTINGS,
    no_args_is_help=True,
)

ORIGINAL_ARGUMENT = typer.Argument(..., help="Original song.")
PIANO_ARGUMENT = typer.Argument(..., help="Piano cover.")
AUDIO_ARGUMENT = typer.Argument(..., help="Audio file to score.")
WEIGHTS_OPTION = typer.Option(
    None,
    "--weights",
    "-w",
    help=(
        "Aria-CL large-0.1 checkpoint. Downloaded to "
        "~/.cache/pianofilter when omitted."
    ),
)
DEVICE_OPTION = typer.Option(
    None, "--device", "-d", help="Torch device. Defaults to CUDA if any."
)


@app.command()
def compare(
    fpath_original: Path = ORIGINAL_ARGUMENT,
    fpath_piano: Path = PIANO_ARGUMENT,
    fpath_weights: Path | None = WEIGHTS_OPTION,
    device: str | None = DEVICE_OPTION,
) -> None:
    """Compare a piano cover with its original and print JSON."""
    scorer = PianoScorer(fpath_weights, device)
    analysis = analyze(
        extract_features(fpath_original, scorer),
        extract_features(fpath_piano, scorer),
    )
    result = {"reject": judge(analysis), **analysis.to_dict()}
    typer.echo(json.dumps(result, indent=2))


@app.command()
def piano(
    fpath_audio: Path = AUDIO_ARGUMENT,
    fpath_weights: Path | None = WEIGHTS_OPTION,
    device: str | None = DEVICE_OPTION,
) -> None:
    """Print the solo piano score of each second of a file."""
    scorer = PianoScorer(fpath_weights, device)
    scores = scorer.score_file(fpath_audio)
    for second, score in enumerate(scores):
        typer.echo(f"{second}\t{score:.3f}")
