#!/usr/bin/env python3
"""Collect ViZDoom episodes as observations connected by transitions."""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
import vizdoom as vzd


ACTION_REPEAT = 1
MIN_ACTION_CHUNK_LENGTH = 3
MAX_ACTION_CHUNK_LENGTH = 12
EPISODE_DIRECTORY_PATTERN = re.compile(r"episode_(\d+)")

NAVIGATION_GAME_VARIABLES = (
    ("health", vzd.GameVariable.HEALTH),
    ("position_x", vzd.GameVariable.POSITION_X),
    ("position_y", vzd.GameVariable.POSITION_Y),
    ("angle", vzd.GameVariable.ANGLE),
)

SCENARIO_DEFINITIONS = {
    "basic": {
        "config_filename": "basic.cfg",
        "actions": (
            ("MOVE_LEFT", (vzd.Button.MOVE_LEFT,)),
            ("MOVE_RIGHT", (vzd.Button.MOVE_RIGHT,)),
            ("ATTACK", (vzd.Button.ATTACK,)),
        ),
        "game_variables": NAVIGATION_GAME_VARIABLES
        + (("ammo2", vzd.GameVariable.AMMO2),),
    },
    "my_way_home": {
        "config_filename": "my_way_home.cfg",
        "actions": (
            ("MOVE_FORWARD", (vzd.Button.MOVE_FORWARD,)),
            ("TURN_LEFT", (vzd.Button.TURN_LEFT,)),
            ("TURN_RIGHT", (vzd.Button.TURN_RIGHT,)),
            ("NOOP", ()),
            (
                "MOVE_FORWARD_LEFT",
                (vzd.Button.MOVE_FORWARD, vzd.Button.TURN_LEFT),
            ),
            (
                "MOVE_FORWARD_RIGHT",
                (vzd.Button.MOVE_FORWARD, vzd.Button.TURN_RIGHT),
            ),
        ),
        "game_variables": NAVIGATION_GAME_VARIABLES,
    },
}


def build_action_table(
    action_specs: tuple,
    available_buttons: tuple,
) -> dict[str, list[bool]]:
    available_set = set(available_buttons)
    action_table = {}
    for label, pressed_buttons in action_specs:
        missing = set(pressed_buttons) - available_set
        if missing:
            missing_names = ", ".join(sorted(button.name for button in missing))
            raise ValueError(
                f"action {label} requires unavailable buttons: {missing_names}"
            )
        action_table[label] = [
            button in pressed_buttons for button in available_buttons
        ]
    return action_table


def build_game(
    scenario_name: str,
    visible: bool = False,
    seed: int | None = None,
) -> vzd.DoomGame:
    scenario = SCENARIO_DEFINITIONS[scenario_name]
    config_path = Path(vzd.scenarios_path) / scenario["config_filename"]
    if not config_path.is_file():
        raise FileNotFoundError(f"ViZDoom scenario config not found: {config_path}")

    game = vzd.DoomGame()
    game.load_config(str(config_path))

    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_depth_buffer_enabled(True)
    game.set_available_game_variables(
        [variable for _, variable in scenario["game_variables"]]
    )
    game.set_window_visible(visible)
    game.set_mode(vzd.Mode.PLAYER)

    available_buttons = tuple(game.get_available_buttons())
    build_action_table(scenario["actions"], available_buttons)
    if seed is not None:
        game.set_seed(seed)
    game.init()
    return game


def save_rgb(array: np.ndarray, path: Path) -> None:
    Image.fromarray(array.astype(np.uint8), mode="RGB").save(path)


def save_depth(array: np.ndarray, path: Path) -> None:
    Image.fromarray(array.astype(np.uint8), mode="L").save(path)


def game_variables_from_state(
    state: vzd.GameState,
    game_variables: tuple,
) -> dict[str, float]:
    if state.game_variables is None:
        return {}

    values = state.game_variables.tolist()
    if len(values) != len(game_variables):
        raise RuntimeError(
            "ViZDoom returned an unexpected number of game variables: "
            f"expected {len(game_variables)}, got {len(values)}"
        )
    return {
        name: float(value)
        for (name, _), value in zip(game_variables, values, strict=True)
    }


def save_observation(
    state: vzd.GameState,
    observation_index: int,
    episode_dir: Path,
    game_variables: tuple,
) -> dict:
    rgb_path = Path("rgb") / f"{observation_index:06d}.png"
    depth_path = Path("depth") / f"{observation_index:06d}.png"

    save_rgb(state.screen_buffer, episode_dir / rgb_path)
    saved_depth: str | None = None
    if state.depth_buffer is not None:
        save_depth(state.depth_buffer, episode_dir / depth_path)
        saved_depth = str(depth_path)

    return {
        "observation_index": observation_index,
        "tic": int(state.tic),
        "rgb": str(rgb_path),
        "depth": saved_depth,
        "game_variables": game_variables_from_state(state, game_variables),
    }


def natural_termination_reason(game: vzd.DoomGame) -> str:
    is_timeout_reached = getattr(game, "is_episode_timeout_reached", None)
    if callable(is_timeout_reached) and is_timeout_reached():
        return "environment_timeout"
    return "task_terminal"


def next_episode_index(scenario_output_root: Path) -> int:
    """Return one greater than the highest existing numeric episode ID."""
    existing_indices = []
    for path in scenario_output_root.iterdir():
        if not path.is_dir():
            continue
        match = EPISODE_DIRECTORY_PATTERN.fullmatch(path.name)
        if match is not None:
            existing_indices.append(int(match.group(1)))
    return max(existing_indices, default=-1) + 1


def collect_episode(
    game: vzd.DoomGame,
    scenario_name: str,
    episode_index: int,
    output_root: Path,
    max_steps: int,
    seed: int,
) -> dict:
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")

    scenario = SCENARIO_DEFINITIONS[scenario_name]
    game_variables = scenario["game_variables"]
    available_buttons = tuple(game.get_available_buttons())
    action_table = build_action_table(scenario["actions"], available_buttons)

    episode_dir = output_root / f"episode_{episode_index:04d}"
    if episode_dir.exists():
        shutil.rmtree(episode_dir)
    (episode_dir / "rgb").mkdir(parents=True)
    (episode_dir / "depth").mkdir()

    game.new_episode()
    observations = []
    transitions = []
    termination_reason: str | None = None

    initial_state = game.get_state()
    if initial_state is not None:
        observations.append(
            save_observation(initial_state, 0, episode_dir, game_variables)
        )

        step = 0
        chunk_id = 0
        stop_episode = False
        while step < max_steps:
            action_label = random.choice(list(action_table))
            action_vector = action_table[action_label]
            chunk_length = random.randint(
                MIN_ACTION_CHUNK_LENGTH,
                MAX_ACTION_CHUNK_LENGTH,
            )

            for chunk_step in range(chunk_length):
                reward = float(game.make_action(action_vector, ACTION_REPEAT))

                done = bool(game.is_episode_finished())
                if done:
                    truncated = False
                    termination_reason = natural_termination_reason(game)
                else:
                    truncated = step + 1 >= max_steps
                    if truncated:
                        termination_reason = "collector_truncation"
                next_observation_index: int | None = None
                next_state = None if done else game.get_state()

                if next_state is not None:
                    next_observation_index = len(observations)
                    observations.append(
                        save_observation(
                            next_state,
                            next_observation_index,
                            episode_dir,
                            game_variables,
                        )
                    )
                elif not done:
                    raise RuntimeError(
                        "ViZDoom returned no state for a non-terminal transition"
                    )

                transitions.append(
                    {
                        "step": step,
                        "observation_index": step,
                        "next_observation_index": next_observation_index,
                        "action": action_label,
                        "action_vector": action_vector,
                        "reward": reward,
                        "done": done,
                        "truncated": truncated,
                        "chunk_id": chunk_id,
                        "chunk_step": chunk_step,
                        "chunk_length": chunk_length,
                    }
                )
                step += 1

                if done or truncated:
                    stop_episode = True
                    break

            if stop_episode:
                break
            chunk_id += 1
    elif game.is_episode_finished():
        termination_reason = natural_termination_reason(game)
    else:
        raise RuntimeError("ViZDoom returned no initial state for a running episode")

    if termination_reason is None:
        raise RuntimeError("episode collection stopped without a termination reason")

    config_path = Path(vzd.scenarios_path) / scenario["config_filename"]
    wad_path = game.get_doom_scenario_path()
    metadata = {
        "episode_index": episode_index,
        "termination_reason": termination_reason,
        "initial_state_available": initial_state is not None,
        "num_observations": len(observations),
        "num_transitions": len(transitions),
        "provenance": {
            "seed": seed,
            "scenario_name": scenario_name,
            "config_path": str(config_path),
            "wad_path": str(wad_path) if wad_path else None,
            "map": game.get_doom_map(),
            "vizdoom_version": str(vzd.__version__),
            "max_steps": max_steps,
            "screen_resolution": [
                game.get_screen_width(),
                game.get_screen_height(),
            ],
            "available_buttons": [button.name for button in available_buttons],
            "action_repeat": ACTION_REPEAT,
            "action_chunk_length_range": [
                MIN_ACTION_CHUNK_LENGTH,
                MAX_ACTION_CHUNK_LENGTH,
            ],
            "episode_start_time": game.get_episode_start_time(),
            "episode_timeout": game.get_episode_timeout(),
        },
        "action_space": [
            {
                "label": label,
                "buttons": [button.name for button in pressed_buttons],
                "vector": action_table[label],
            }
            for label, pressed_buttons in scenario["actions"]
        ],
        "observations": observations,
        "transitions": transitions,
    }

    (episode_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        choices=SCENARIO_DEFINITIONS,
        default="my_way_home",
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--output", type=Path, default=Path("data/episodes"))
    parser.add_argument("--visible", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes < 1:
        raise ValueError("episodes must be at least 1")
    if args.max_steps < 1:
        raise ValueError("max_steps must be at least 1")

    random.seed(args.seed)
    np.random.seed(args.seed)

    scenario_output_root = args.output / args.scenario
    scenario_output_root.mkdir(parents=True, exist_ok=True)
    start_episode_index = next_episode_index(scenario_output_root)
    game = build_game(
        scenario_name=args.scenario,
        visible=args.visible,
        seed=args.seed,
    )

    try:
        for episode_index in range(
            start_episode_index,
            start_episode_index + args.episodes,
        ):
            metadata = collect_episode(
                game=game,
                scenario_name=args.scenario,
                episode_index=episode_index,
                output_root=scenario_output_root,
                max_steps=args.max_steps,
                seed=args.seed,
            )
            print(
                f"scenario={args.scenario} episode={episode_index:04d} "
                f"observations={metadata['num_observations']} "
                f"transitions={metadata['num_transitions']} "
                f"saved_to={scenario_output_root / f'episode_{episode_index:04d}'}"
            )
    finally:
        game.close()


if __name__ == "__main__":
    main()
