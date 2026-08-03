#!/usr/bin/env python3
"""Patch heretic master to add --n-top-trials N batch export mode.

When set, heretic automatically saves the top N Pareto-optimal trials as
LoRA adapters (or merged models) after optimization, then exits — no
interactive prompts. Requires --save-directory to also be set.

Patches:
  config.py: adds n_top_trials field to Settings
  main.py:   adds batch export loop after Pareto front computation
"""

import sys
from pathlib import Path


def find_heretic_file(filename: str) -> Path | None:
    for candidate in sys.path:
        p = Path(candidate) / "heretic" / filename
        if p.exists():
            return p
    return None


def patch_config():
    target = find_heretic_file("config.py")
    if target is None:
        print("heretic/config.py not found, skipping")
        return

    code = target.read_text()
    if "n_top_trials" in code:
        print("config.py already patched, skipping")
        return

    old = '''    save_directory: str | None = Field(
        default=None,
        description="Directory to save the model to, or unset to prompt the user.",
        exclude=True,
    )'''

    new = '''    save_directory: str | None = Field(
        default=None,
        description="Directory to save the model to, or unset to prompt the user.",
        exclude=True,
    )
    n_top_trials: PositiveInt | None = Field(
        default=None,
        description=(
            "If set, automatically save the top N Pareto-optimal trials after "
            'optimization, then exit. Requires save_directory and export_strategy. '
            "When N > 1, each trial is saved to a subdirectory trial_0, trial_1, ..."
        ),
        exclude=True,
    )'''

    if old not in code:
        print("save_directory field not found in config.py, skipping")
        return

    code = code.replace(old, new, 1)
    target.write_text(code)
    print(f"Patched {target}: added n_top_trials field")


def patch_main():
    target = find_heretic_file("main.py")
    if target is None:
        print("heretic/main.py not found, skipping")
        return

    code = target.read_text()
    if "_N_TOP_TRIALS_PATCHED" in code:
        print("main.py already patched, skipping")
        return

    # Insert batch export block right before format_trial_title definition.
    marker = "            def format_trial_title(trial: FrozenTrial) -> str:"

    batch_block = '''            # _N_TOP_TRIALS_PATCHED
            # Batch export mode: save top N Pareto trials as adapters/models, then exit.
            if settings.n_top_trials is not None:
                if settings.save_directory is None:
                    print("[red]--save-directory is required when --n-top-trials is set.[/]")
                    return

                n_save = min(settings.n_top_trials, len(sorted_trials))
                print()
                print(f"[bold green]Batch exporting top {n_save} Pareto-optimal trials...[/]")

                for top_i in range(n_save):
                    batch_trial = sorted_trials[top_i]
                    score_str = ", ".join(
                        f"{s['name']}: {s['score']['rich_display']}"
                        for s in batch_trial.user_attrs["scores"]
                    )
                    print()
                    print(f"* [{top_i + 1}/{n_save}] Trial {batch_trial.user_attrs['index']}: {score_str}")

                    model.reset_model()
                    model.abliterate(
                        residual_directions,
                        batch_trial.user_attrs["direction_index"],
                        {
                            k: AbliterationParameters(**v)
                            for k, v in batch_trial.user_attrs["parameters"].items()
                        },
                    )

                    if n_save > 1:
                        batch_dir = os.path.join(settings.save_directory, f"trial_{top_i}")
                    else:
                        batch_dir = settings.save_directory
                    os.makedirs(batch_dir, exist_ok=True)

                    if settings.export_strategy == ExportStrategy.MERGE:
                        print("  Saving merged model...")
                        merged = model.get_merged_model()
                        merged.save_pretrained(
                            batch_dir,
                            max_shard_size=settings.max_shard_size,
                        )
                        del merged
                        empty_cache()
                        model.tokenizer.save_pretrained(batch_dir)
                        if model.processor is not None:
                            model.processor.save_pretrained(batch_dir)
                    else:
                        print("  Saving LoRA adapter...")
                        model.model.save_pretrained(
                            batch_dir,
                            max_shard_size=settings.max_shard_size,
                        )

                    # Generate reproduce.json alongside each saved adapter.
                    print("  Saving reproduce info...")
                    try:
                        from pathlib import Path as _Path
                        from heretic.utils import create_reproduce_folder
                        create_reproduce_folder(
                            _Path(batch_dir),
                            settings,
                            checkpoint_path=study_checkpoint_file,
                            trial=batch_trial,
                            uploaded_model_hashes={},
                            include_system_information=False,
                        )
                    except Exception as e:
                        print(f"  [yellow]Warning: Could not create reproduce folder: {e}[/]")

                print()
                print(f"[bold green]Saved {n_save} models to {settings.save_directory}[/]")
                return

'''

    if marker not in code:
        print("format_trial_title marker not found in main.py, skipping")
        return

    code = code.replace(marker, batch_block + marker, 1)
    target.write_text(code)
    print(f"Patched {target}: added n_top_trials batch export mode")


if __name__ == "__main__":
    patch_config()
    patch_main()
