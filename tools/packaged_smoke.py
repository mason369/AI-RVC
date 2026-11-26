"""Exercise the actual frozen app, native audio dependencies and HTTP surface."""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path


def check_audio(output: Path) -> dict:
    import librosa
    import numpy as np
    import soundfile as sf
    from infer.cover_reuse import CoverReuse
    from lib.mixer import _juce_reverb_kernel_float64, apply_reverb, mix_vocals_and_accompaniment

    kernel = _juce_reverb_kernel_float64
    source = Path(kernel.py_func.__code__.co_filename)
    if not source.is_absolute() or not source.is_file():
        raise RuntimeError(f"Numba kernel source is not relocatable: {source}")
    signal = np.zeros((2, 48000), dtype=np.float64)
    signal[:, 0] = [0.200000000000001, 0.100000000000001]
    double = apply_reverb(signal, 48000, room_size=0.3, wet_level=0.2)
    single = apply_reverb(signal.astype(np.float32), 48000, room_size=0.3, wet_level=0.2)
    if double.dtype != np.float64 or double.shape != signal.shape or not np.isfinite(double).all():
        raise RuntimeError("FLOAT64 reverb lost precision, shape or finite samples")
    if not np.any(double[:, 1000:]) or not kernel.nopython_signatures:
        raise RuntimeError("Reverb did not produce a tail using native JIT compilation")
    np.testing.assert_allclose(double, single, rtol=2e-4, atol=1e-7)
    resampled = librosa.resample(double, orig_sr=48000, target_sr=44100, res_type="kaiser_fast")
    if resampled.shape != (2, 44100) or not np.isfinite(resampled).all():
        raise RuntimeError("Bundled resampler produced invalid audio")
    # Exercise another cached librosa kernel, not just its import or soxr path.
    spectrum = np.abs(librosa.stft(resampled[0]))
    librosa.amplitude_to_db(spectrum)
    librosa.util.normalize(resampled)
    output.mkdir(parents=True, exist_ok=True)
    vocals, accompaniment, mixed = (output / name for name in ("vocals.wav", "accompaniment.wav", "mix.wav"))
    sf.write(vocals, signal.T, 48000, subtype="DOUBLE")
    sf.write(accompaniment, signal.T * 0.1, 48000, subtype="DOUBLE")
    mix_vocals_and_accompaniment(str(vocals), str(accompaniment), str(mixed), reverb_amount=0.2)
    audio, rate = sf.read(mixed, dtype="float64", always_2d=True)
    if sf.info(mixed).subtype != "DOUBLE" or rate != 48000 or audio.shape != (48000, 2) or not np.isfinite(audio).all():
        raise RuntimeError("Bundled mixer failed the DOUBLE WAV round trip")
    session = {}
    next_run = output / "remix"
    next_run.mkdir(exist_ok=True)
    key = lambda: ("packaged-smoke",)
    tracks = [vocals, accompaniment]
    CoverReuse(session, mixed, output).save("primary", key, tracks)
    restored = CoverReuse(session, mixed, next_run).restore("primary", key)
    if not restored or [Path(path).read_bytes() for path in restored] != [path.read_bytes() for path in tracks]:
        raise RuntimeError("Bundled stage reuse changed the DOUBLE WAV tracks")
    return {"source": str(source), "cache": kernel._cache.cache_path,
            "cache_hits": sum(kernel.stats.cache_hits.values()), "samples": len(audio)}


def check_ui() -> list[str]:
    from ui import app as ui_app
    from ui.gradio_assets import patch_upload_route
    from ui.server_runtime import configure_browser_server

    configure_browser_server()
    checked = []
    for language in ui_app.SUPPORTED_LANGUAGES:
        ui_app.config = {**ui_app.config, "language": language}
        ui_app.i18n = ui_app.load_i18n(language)
        blocks = ui_app.create_ui()
        try:
            blocks.queue()
            blocks.launch(server_name="127.0.0.1", inbrowser=False, share=False,
                          prevent_thread_lock=True, i18n=ui_app.ui_catalog())
            ui_app._patch_gradio_file_download(blocks)
            patch_upload_route(blocks)
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(blocks.local_url, timeout=30) as response:
                if response.status != 200 or b"gradio" not in response.read().lower():
                    raise RuntimeError("Bundled UI HTML is unavailable")
            with opener.open(blocks.local_url + "config", timeout=30) as response:
                configuration = json.load(response)
                handlers = {dep.get("api_name") for dep in configuration["dependencies"]}
                has_multi_upload = any(component["type"] == "file" and
                                       component["props"].get("file_count") == "multiple"
                                       for component in configuration["components"])
                if not has_multi_upload or not {"process_cover_batch", "select_batch_result",
                                                "process_cover_with_downloads"} <= handlers:
                    raise RuntimeError("Bundled UI is missing a cover workflow")
            checked.append(language)
        finally:
            blocks.close()
    return checked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audio-only", action="store_true")
    args = parser.parse_args()
    report = {"frozen": bool(getattr(sys, "frozen", False)), "audio": check_audio(args.output)}
    if not args.audio_only:
        report["ui_languages"] = check_ui()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
