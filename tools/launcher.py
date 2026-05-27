#!/usr/bin/env python3
"""Native Tk launcher for lerobot scripts.

Pick a script (record / teleop / rollout), fill in the flags via form
widgets, copy or run the constructed command. Per-script state is
persisted to ~/.config/lerobot-launcher/state.json so reopening the app
shows your last values.

Run:
    python tools/launcher.py
"""

from __future__ import annotations

import json
import os
import platform
import shlex
import subprocess
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any, Callable

STATE_PATH = Path.home() / ".config" / "lerobot-launcher" / "state.json"


# ----------------------------------------------------------------------
# Field schema
# ----------------------------------------------------------------------


@dataclass
class Field:
    flag: str  # CLI flag without leading --
    label: str
    kind: str  # "text" | "int" | "float" | "bool" | "select" | "multiline"
    default: Any = ""
    options: tuple[str, ...] | None = None  # for "select"


# Shared option lists
ROBOT_TYPES = ("so110_follower", "so100_follower", "so101_follower", "koch_follower")
TELEOP_TYPES = ("so110_leader", "so_leader", "koch_leader")
STRATEGY_TYPES = ("base", "sentry", "highlight", "dagger")
INFERENCE_TYPES = ("sync", "rtc", "remote")
INPUT_DEVICES = ("keyboard", "pedal")

CAMERAS_DEFAULT = (
    "{\n"
    "    wrist_top:    {type: opencv, index_or_path: 0, width: 480, height: 640, fps: 30, rotation: ROTATE_90, crop_to_square: true},\n"
    "    wrist_bottom: {type: opencv, index_or_path: 1, width: 480, height: 640, fps: 30, rotation: ROTATE_90, crop_to_square: true},\n"
    "    overhead:     {type: opencv, index_or_path: 2, width: 480, height: 640, fps: 30, rotation: ROTATE_90, crop_to_square: true}\n"
    "}"
)

P_COEFFICIENT_DEFAULT = (
    "{shoulder_lift: 16, shoulder_swing: 16, elbow_lift: 10, gripper: 10, "
    "wrist_tilt: 10, wrist_yaw: 10, wrist_roll: 10, shoulder_yaw: 10}"
)

RENAME_MAP_DEFAULT = (
    '{"observation.images.wrist_top": "observation.images.camera1", '
    '"observation.images.wrist_bottom": "observation.images.camera2", '
    '"observation.images.overhead": "observation.images.camera3"}'
)


# ----------------------------------------------------------------------
# Per-script schemas
# ----------------------------------------------------------------------

SCHEMAS: dict[str, dict[str, Any]] = {
    "record": {
        "command": "lerobot-record",
        "fields": [
            Field("dataset.repo_id", "Dataset repo id", "text", "eliasab16/so110_dataset"),
            Field("dataset.single_task", "Task", "text", "Insert the wire tip into the component hole from below."),
            Field("dataset.num_episodes", "Num episodes", "int", 40),
            Field("dataset.episode_time_s", "Episode time (s)", "int", 120),
            Field("dataset.reset_time_s", "Reset time (s)", "int", 120),
            Field("dataset.push_to_hub", "Push to hub", "bool", False),
            Field("robot.type", "Robot type", "select", "so110_follower", ROBOT_TYPES),
            Field("robot.port", "Robot port", "text", "/dev/tty.usbmodem5AE60845471"),
            Field("robot.id", "Robot id", "text", "so110_follower_left"),
            Field("robot.p_coefficient", "Robot p_coefficient", "multiline", P_COEFFICIENT_DEFAULT),
            Field("robot.d_coefficient", "Robot d_coefficient", "int", 64),
            Field("robot.temperature_sample_interval_s", "Temp sample interval (s)", "float", 5.0),
            Field("robot.temperature_warning_c", "Temp warning (°C)", "int", 55),
            Field("robot.cameras", "Cameras (YAML)", "multiline", CAMERAS_DEFAULT),
            Field("teleop.type", "Teleop type", "select", "so110_leader", TELEOP_TYPES),
            Field("teleop.port", "Teleop port", "text", "/dev/tty.usbmodem5A7A0565141"),
            Field("teleop.id", "Teleop id", "text", "so110_leader_left"),
            Field("display_data", "Display data", "bool", True),
            Field("display_compressed_images", "Display compressed", "bool", False),
            Field("resume", "Resume", "bool", False),
            Field("dataset.vcodec", "vcodec", "text", "auto"),
            Field("dataset.streaming_encoding", "Streaming encoding", "bool", True),
            Field("dataset.encoder_threads", "Encoder threads", "int", 2),
            Field("dataset.encoder_queue_maxsize", "Encoder queue maxsize", "int", 30),
            Field("dataset.video_encoding_batch_size", "Video encoding batch size", "int", 1),
        ],
    },
    "teleop": {
        "command": "lerobot-teleoperate",
        "fields": [
            Field("robot.type", "Robot type", "select", "so110_follower", ROBOT_TYPES),
            Field("robot.port", "Robot port", "text", "/dev/tty.usbmodem5AE60845471"),
            Field("robot.id", "Robot id", "text", "so110_follower_left"),
            Field("robot.p_coefficient", "Robot p_coefficient", "multiline", P_COEFFICIENT_DEFAULT),
            Field("robot.d_coefficient", "Robot d_coefficient", "int", 64),
            Field("robot.cameras", "Cameras (YAML)", "multiline", CAMERAS_DEFAULT),
            Field("teleop.type", "Teleop type", "select", "so110_leader", TELEOP_TYPES),
            Field("teleop.port", "Teleop port", "text", "/dev/tty.usbmodem5A7A0565141"),
            Field("teleop.id", "Teleop id", "text", "so110_leader_left"),
            Field("fps", "FPS", "int", 30),
            Field("display_data", "Display data", "bool", True),
        ],
    },
    "rollout": {
        "command": "lerobot-rollout",
        "fields": [
            Field("strategy.type", "Strategy", "select", "dagger", STRATEGY_TYPES),
            Field("strategy.num_episodes", "Num episodes", "int", 5),
            Field("strategy.record_autonomous", "Record autonomous", "bool", False),
            Field("strategy.upload_every_n_episodes", "Upload every N episodes", "int", 999),
            Field("strategy.input_device", "Input device", "select", "keyboard", INPUT_DEVICES),
            Field("inference.type", "Inference backend", "select", "remote", INFERENCE_TYPES),
            Field("inference.server_url", "Server URL", "text", "ws://localhost:9000"),
            Field("inference.fire_after_n_actions", "Fire after N actions", "int", 15),
            Field("inference.rtc.enabled", "RTC enabled", "bool", True),
            Field("inference.rtc.execution_horizon", "Execution horizon", "int", 20),
            Field("inference.log_actions_csv", "Log actions CSV", "text", "outputs/rollout_actions.csv"),
            Field("rename_map", "Rename map (JSON)", "multiline", RENAME_MAP_DEFAULT),
            Field("policy.path", "Policy path", "text", "eliasab16/smolvla_insert_wire_b1_25k"),
            Field("robot.type", "Robot type", "select", "so110_follower", ROBOT_TYPES),
            Field("robot.port", "Robot port", "text", "/dev/tty.usbmodem5AE60845471"),
            Field("robot.id", "Robot id", "text", "so110_follower_left"),
            Field("robot.p_coefficient", "Robot p_coefficient", "multiline", P_COEFFICIENT_DEFAULT),
            Field("robot.d_coefficient", "Robot d_coefficient", "int", 64),
            Field("robot.cameras", "Cameras (YAML)", "multiline", CAMERAS_DEFAULT),
            Field("teleop.type", "Teleop type", "select", "so110_leader", TELEOP_TYPES),
            Field("teleop.port", "Teleop port", "text", "/dev/tty.usbmodem5A7A0565141"),
            Field("teleop.id", "Teleop id", "text", "so110_leader_left"),
            Field("dataset.repo_id", "Dataset repo id", "text", "local-only/dagger_test"),
            Field("dataset.single_task", "Task", "text", "Insert the wire tip into the component hole from below."),
            Field("dataset.push_to_hub", "Push to hub", "bool", False),
            Field("task", "Task (rollout)", "text", "Insert the wire tip into the component hole from below."),
            Field("duration", "Duration (s)", "float", 600.0),
            Field("display_data", "Display data", "bool", True),
        ],
    },
}


# ----------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------


def load_state() -> dict[str, dict[str, Any]]:
    if not STATE_PATH.exists():
        return {}
    try:
        return json.loads(STATE_PATH.read_text())
    except json.JSONDecodeError:
        return {}


def save_state(state: dict[str, dict[str, Any]]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2))


# ----------------------------------------------------------------------
# Command construction
# ----------------------------------------------------------------------


def _quote_value(kind: str, value: Any) -> str:
    """Render a field value as a shell-quoted CLI value."""
    if kind == "bool":
        return "true" if bool(value) else "false"
    if kind in ("int", "float"):
        return str(value)
    # text / multiline / select — single-quote and escape any embedded quotes
    s = str(value)
    return "'" + s.replace("'", "'\\''") + "'"


def build_command(script: str, values: dict[str, Any], extra_flags: str) -> str:
    schema = SCHEMAS[script]
    cmd_parts = [schema["command"]]
    for f in schema["fields"]:
        v = values.get(f.flag, f.default)
        if isinstance(v, str) and not v.strip() and f.kind != "multiline":
            continue
        cmd_parts.append(f"--{f.flag}={_quote_value(f.kind, v)}")
    if extra_flags.strip():
        cmd_parts.append(extra_flags.strip())
    return " \\\n    ".join(cmd_parts)


# ----------------------------------------------------------------------
# Tk UI
# ----------------------------------------------------------------------


class LauncherApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("LeRobot Launcher")
        root.geometry("900x720")

        self.state = load_state()
        self.current_script = tk.StringVar(value=next(iter(SCHEMAS)))
        # Per-field tk variables / widgets keyed by flag
        self._vars: dict[str, Any] = {}
        self._widgets: dict[str, tk.Widget] = {}

        self._build_header()
        self._build_form_container()
        self._build_footer()
        self._render_form()

    # -- header ---------------------------------------------------------

    def _build_header(self) -> None:
        bar = ttk.Frame(self.root, padding=(10, 8))
        bar.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(bar, text="Script:").pack(side=tk.LEFT)
        picker = ttk.Combobox(
            bar,
            textvariable=self.current_script,
            values=list(SCHEMAS.keys()),
            state="readonly",
            width=18,
        )
        picker.pack(side=tk.LEFT, padx=(6, 12))
        picker.bind("<<ComboboxSelected>>", lambda _e: self._render_form())

        ttk.Button(bar, text="Reset to defaults", command=self._reset_defaults).pack(
            side=tk.LEFT
        )

    # -- form container -------------------------------------------------

    def _build_form_container(self) -> None:
        outer = ttk.Frame(self.root, padding=(10, 0))
        outer.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Scrollable canvas
        canvas = tk.Canvas(outer, highlightthickness=0)
        sb = ttk.Scrollbar(outer, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        self._form_frame = ttk.Frame(canvas)
        self._form_window = canvas.create_window(
            (0, 0), window=self._form_frame, anchor="nw"
        )
        self._form_canvas = canvas

        def _on_configure(_e: Any) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_configure(event: Any) -> None:
            canvas.itemconfig(self._form_window, width=event.width)

        self._form_frame.bind("<Configure>", _on_configure)
        canvas.bind("<Configure>", _on_canvas_configure)

        # Mouse wheel scrolling (macOS uses different units)
        def _on_mousewheel(event: Any) -> None:
            delta = -1 if event.delta > 0 else 1
            if platform.system() == "Darwin":
                canvas.yview_scroll(int(-1 * event.delta), "units")
            else:
                canvas.yview_scroll(delta, "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)

    # -- footer ---------------------------------------------------------

    def _build_footer(self) -> None:
        bottom = ttk.Frame(self.root, padding=(10, 8))
        bottom.pack(side=tk.BOTTOM, fill=tk.X)

        # Extra-flags entry
        extra_row = ttk.Frame(bottom)
        extra_row.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(extra_row, text="Extra flags:").pack(side=tk.LEFT)
        self._extra_var = tk.StringVar()
        ttk.Entry(extra_row, textvariable=self._extra_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0)
        )

        # Command preview
        ttk.Label(bottom, text="Generated command:").pack(
            side=tk.TOP, anchor="w", pady=(8, 2)
        )
        self._preview = tk.Text(bottom, height=10, wrap="word")
        self._preview.pack(side=tk.TOP, fill=tk.BOTH, expand=False)

        # Action buttons
        btnrow = ttk.Frame(bottom)
        btnrow.pack(side=tk.TOP, fill=tk.X, pady=(8, 0))
        ttk.Button(btnrow, text="Update preview", command=self._update_preview).pack(
            side=tk.LEFT
        )
        ttk.Button(btnrow, text="Copy", command=self._copy_command).pack(
            side=tk.LEFT, padx=(6, 0)
        )
        ttk.Button(btnrow, text="Run in Terminal", command=self._run_in_terminal).pack(
            side=tk.LEFT, padx=(6, 0)
        )
        ttk.Button(btnrow, text="Save state", command=self._save_now).pack(
            side=tk.RIGHT
        )

    # -- form rendering -------------------------------------------------

    def _render_form(self) -> None:
        for w in self._form_frame.winfo_children():
            w.destroy()
        self._vars.clear()
        self._widgets.clear()

        script = self.current_script.get()
        schema = SCHEMAS[script]
        saved = self.state.get(script, {})

        for row, f in enumerate(schema["fields"]):
            lbl = ttk.Label(self._form_frame, text=f.label, width=28, anchor="w")
            lbl.grid(row=row, column=0, sticky="w", padx=4, pady=3)

            val = saved.get(f.flag, f.default)

            widget = self._make_widget(self._form_frame, f, val)
            widget.grid(row=row, column=1, sticky="ew", padx=4, pady=3)
            self._widgets[f.flag] = widget

        self._form_frame.columnconfigure(1, weight=1)
        # Re-apply extra-flags from saved state
        self._extra_var.set(saved.get("__extra__", ""))
        self._update_preview()

    def _make_widget(self, parent: tk.Widget, f: Field, val: Any) -> tk.Widget:
        if f.kind == "bool":
            var = tk.BooleanVar(value=bool(val))
            self._vars[f.flag] = var
            return ttk.Checkbutton(parent, variable=var, command=self._update_preview)

        if f.kind == "select":
            var = tk.StringVar(value=str(val))
            self._vars[f.flag] = var
            cb = ttk.Combobox(
                parent,
                textvariable=var,
                values=list(f.options or ()),
                state="readonly",
            )
            cb.bind("<<ComboboxSelected>>", lambda _e: self._update_preview())
            return cb

        if f.kind == "multiline":
            txt = tk.Text(parent, height=4, wrap="word")
            txt.insert("1.0", str(val))
            txt.bind("<KeyRelease>", lambda _e: self._update_preview())
            self._vars[f.flag] = txt
            return txt

        # text / int / float — use Entry, validation is loose
        var = tk.StringVar(value=str(val))
        self._vars[f.flag] = var
        var.trace_add("write", lambda *_a: self._update_preview())
        return ttk.Entry(parent, textvariable=var)

    # -- value extraction -----------------------------------------------

    def _collect_values(self) -> dict[str, Any]:
        script = self.current_script.get()
        out: dict[str, Any] = {}
        for f in SCHEMAS[script]["fields"]:
            v = self._vars[f.flag]
            if isinstance(v, tk.BooleanVar):
                out[f.flag] = v.get()
            elif isinstance(v, tk.Text):
                out[f.flag] = v.get("1.0", "end-1c")
            elif isinstance(v, tk.StringVar):
                raw = v.get()
                if f.kind == "int":
                    try:
                        out[f.flag] = int(raw)
                    except ValueError:
                        out[f.flag] = raw
                elif f.kind == "float":
                    try:
                        out[f.flag] = float(raw)
                    except ValueError:
                        out[f.flag] = raw
                else:
                    out[f.flag] = raw
        return out

    # -- actions --------------------------------------------------------

    def _update_preview(self) -> None:
        cmd = build_command(
            self.current_script.get(),
            self._collect_values(),
            self._extra_var.get(),
        )
        self._preview.delete("1.0", "end")
        self._preview.insert("1.0", cmd)

    def _save_now(self) -> None:
        script = self.current_script.get()
        values = self._collect_values()
        values["__extra__"] = self._extra_var.get()
        self.state[script] = values
        save_state(self.state)
        messagebox.showinfo("Saved", f"State saved to {STATE_PATH}")

    def _copy_command(self) -> None:
        self._update_preview()
        cmd = self._preview.get("1.0", "end-1c")
        self.root.clipboard_clear()
        self.root.clipboard_append(cmd)

    def _run_in_terminal(self) -> None:
        self._save_now_silent()
        cmd = self._preview.get("1.0", "end-1c")
        cwd = os.getcwd()
        full = f"cd {shlex.quote(cwd)} && {cmd}"
        try:
            if platform.system() == "Darwin":
                escaped = full.replace("\\", "\\\\").replace('"', '\\"')
                subprocess.Popen(
                    [
                        "osascript",
                        "-e",
                        f'tell application "Terminal" to do script "{escaped}"',
                        "-e",
                        'tell application "Terminal" to activate',
                    ]
                )
            else:
                # Linux: try x-terminal-emulator, fall back to xterm
                term = "x-terminal-emulator"
                subprocess.Popen([term, "-e", "bash", "-c", full])
        except Exception as e:
            messagebox.showerror("Run failed", str(e))

    def _save_now_silent(self) -> None:
        script = self.current_script.get()
        values = self._collect_values()
        values["__extra__"] = self._extra_var.get()
        self.state[script] = values
        save_state(self.state)

    def _reset_defaults(self) -> None:
        script = self.current_script.get()
        self.state.pop(script, None)
        save_state(self.state)
        self._render_form()


def main() -> None:
    root = tk.Tk()
    try:
        # Better widget rendering on macOS
        ttk.Style().theme_use("aqua" if platform.system() == "Darwin" else "clam")
    except tk.TclError:
        pass
    LauncherApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
