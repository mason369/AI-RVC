"""Keep the isolated Python 3.10 audio worker alive across notebook cells."""
import atexit
import json
import os
from pathlib import Path
import subprocess
import sys


class NotebookBatch:
    def __init__(self):
        project = Path(__file__).resolve().parents[1]
        self.worker = subprocess.Popen(
            [str(project / "venv310/bin/python"), "-u", "-m", "tools.notebook_batch"],
            cwd=project, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, encoding="utf-8",
        )
        atexit.register(self.close)

    def close(self):
        if self.worker.poll() is None:
            self.worker.terminate()
        self.worker.wait()
        self.worker.stdin.close()
        self.worker.stdout.close()
        atexit.unregister(self.close)

    def convert(self, inputs, character, mode, pitch, original_vocal_mix=None):
        request = dict(inputs=[str(Path(p).resolve()) for p in inputs],
                       character=character, mode=mode, pitch=pitch, original_vocal_mix=original_vocal_mix)
        try:
            self.worker.stdin.write(json.dumps(request) + "\n")
            self.worker.stdin.flush()
            for line in self.worker.stdout:
                event = json.loads(line)
                if "summary" in event:
                    print(event["summary"], flush=True)
                elif "error" in event:
                    raise RuntimeError(event["error"])
                else:
                    return event["archive"]
        except (KeyboardInterrupt, BrokenPipeError):
            self.close()
            raise
        raise ChildProcessError(self.worker.wait(), self.worker.args)


def serve():
    # Keep progress messages separate from native libraries writing to stdout.
    with os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8") as replies:
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        from tools.character_models import download_character_model, get_character_model_path
        from ui.app import cover_status_text, process_cover_batch, resolve_automatic_cover_settings, tf

        def send(**event):
            print(json.dumps(event), file=replies, flush=True)

        state = {}
        for line in sys.stdin:
            try:
                request = json.loads(line)
                character = request["character"]
                if not get_character_model_path(character) and not download_character_model(character):
                    raise RuntimeError(tf("character_download_failed", "messages", name=character))
                settings = resolve_automatic_cover_settings()
                for key in ("index_ratio", "vocals_volume", "accompaniment_volume",
                            "reverb_amount", "rms_mix_rate", "backing_mix"):
                    settings[key] = round(settings[key] * 100)
                settings["pitch_shift"] = request["pitch"]
                if request["original_vocal_mix"] is not None:
                    settings["backing_mix"] = request["original_vocal_mix"]
                for rows, _, summary, download in process_cover_batch(
                    request["inputs"], character, **settings, input_mode=request["mode"],
                    reuse_state=state, progress=lambda fraction, desc: None,
                ):
                    send(summary=summary)
                for row in rows:
                    if row["status"].get("stages"):
                        send(summary=f'{row["name"]}\n{cover_status_text(row["status"])}')
                if not download.get("value"):
                    raise RuntimeError(summary)
                send(archive=str(Path(download["value"]).resolve()))
            except Exception as exc:
                send(error=str(exc) or type(exc).__name__)


if __name__ == "__main__":
    serve()
