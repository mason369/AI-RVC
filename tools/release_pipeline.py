"""Publish one exact Actions build, with packages and images checked before release."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.stage_release_draft import verified_package_files


REPORT_ASSETS = {"RELEASE-VALIDATION.md", "RELEASE-VALIDATION.json",
                 "PACKAGE-CONTENT-COMPARISON.json", "DRAFT-ASSET-PROVENANCE.json"}
PACKAGE_NAME = re.compile(r"AI-RVC-(?:Windows|Linux)-(?:CPU|GPU)-Portable(?:\.SHA256SUMS|\.7z\.[0-9]{3}|\.tar\.gz(?:\.part[a-z]{2})?)")


def api(path, method="GET", payload=None, missing_ok=False):
    with tempfile.TemporaryDirectory(prefix="ai-rvc-release-request-") as directory:
        arguments = ["gh", "api", path, "--method", method]
        if payload is not None:
            request = Path(directory) / "request.json"
            request.write_text(json.dumps(payload), encoding="utf-8")
            arguments += ["--input", str(request)]
        result = subprocess.run(arguments, text=True, encoding="utf-8", capture_output=True, timeout=120)
        if result.returncode:
            if missing_ok and "HTTP 404" in result.stderr:
                return None
            raise RuntimeError(f"GitHub {method} failed: {result.stderr.strip()}")
        return json.loads(result.stdout) if result.stdout.strip() else None


@dataclass(frozen=True)
class Release:
    repository: str
    source: str
    run_id: str
    version: str
    notes: str

    @property
    def tag(self):
        return "v" + self.version

    @property
    def prefix(self):
        return "repos/" + self.repository


def context():
    source = os.environ["GITHUB_SHA"]
    run_id = os.environ["GITHUB_RUN_ID"]
    repository = os.environ["GITHUB_REPOSITORY"]
    version = Path("VERSION").read_text(encoding="utf-8").strip()
    if not re.fullmatch(r"[0-9a-f]{40}", source) or not re.fullmatch(r"[1-9][0-9]*", run_id):
        raise ValueError("Exact source commit and Actions run ID are required")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository) or not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version):
        raise ValueError("Invalid repository or release version")
    if os.environ["GITHUB_REF"] != "refs/heads/master":
        raise RuntimeError("Run the release workflow from master")
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if actual != source:
        raise RuntimeError("The checked-out commit differs from the release run")
    notes = Path(f"docs/releases/v{version}.md").read_text(encoding="utf-8").strip()
    if not notes or "## English" not in notes or not re.search(r"[\u3400-\u9fff]", notes):
        raise RuntimeError("Release notes must contain both supported languages")
    return Release(repository, source, run_id, version, notes)


def find_release(release):
    for page in range(1, 101):
        records = api(f"{release.prefix}/releases?per_page=100&page={page}")
        matches = [item for item in records if item["tag_name"] == release.tag]
        if matches:
            if len(matches) != 1:
                raise RuntimeError("The release destination is ambiguous")
            return matches[0]
        if len(records) < 100:
            return None
    raise RuntimeError("Release list exceeded the pagination limit")


def check_tag(release):
    ref = api(f"{release.prefix}/git/ref/tags/{release.tag}", missing_ok=True)
    if ref is None:
        return False
    obj = ref["object"]
    for _ in range(5):
        if obj["type"] == "commit":
            if obj["sha"] != release.source:
                raise RuntimeError("The existing tag points to another commit")
            return True
        if obj["type"] != "tag":
            break
        obj = api(f"{release.prefix}/git/tags/{obj['sha']}")["object"]
    raise RuntimeError("Unable to resolve the release tag to a commit")


def prepare(release):
    check_tag(release)
    current = find_release(release)
    if current and not current["draft"]:
        raise RuntimeError("This version is already public; choose a new VERSION")
    print(f"Preparing {release.tag} at {release.source}", flush=True)


def verify_workflow(release, include_images=False):
    run = api(f"{release.prefix}/actions/runs/{release.run_id}")
    if (run["head_sha"] != release.source or run["path"] != ".github/workflows/release.yml"
            or run["head_repository"]["full_name"] != release.repository or run["event"] != "workflow_dispatch"
            or run["status"] != "in_progress" or run["conclusion"] is not None):
        raise RuntimeError("Release artifacts must belong to this exact release workflow and commit")
    response = api(f"{release.prefix}/actions/runs/{release.run_id}/jobs?per_page=100")
    if response["total_count"] > 100:
        raise RuntimeError("Unexpected release job count")
    required = [f"Build {platform}-{variant}" for platform in ("Windows", "Linux") for variant in ("CPU", "GPU")]
    required += [f"{platform} / Python 3.10" for platform in ("windows-latest", "ubuntu-latest", "macos-15")]
    required += ["Verify Windows portable browser"]
    if include_images:
        required += ["Docker cpu", "Docker cuda"]
    for name in required:
        jobs = [item for item in response["jobs"] if item["name"] == name or item["name"].endswith(" / " + name)]
        if len(jobs) != 1 or (jobs[0]["status"], jobs[0]["conclusion"]) != ("completed", "success"):
            raise RuntimeError(f"Required job has not succeeded: {name}")


def draft(release, release_id):
    item = api(f"{release.prefix}/releases/{release_id}")
    if not item["draft"] or item["target_commitish"] != release.source or item["tag_name"] != release.tag:
        raise RuntimeError("The destination must remain the exact unpublished release")
    return item


def verify_assets(assets, expected):
    if len(assets) != len(expected) or {item["name"] for item in assets} != set(expected):
        raise RuntimeError("The release must contain only the complete package volumes and checksums")
    for item in assets:
        entry = expected[item["name"]]
        if item.get("state") != "uploaded" or item.get("digest") != "sha256:" + entry["sha256"] or item["size"] != entry["bytes"]:
            raise RuntimeError(f"Release asset verification failed: {item['name']}")


def upload(release, release_id, path, entry):
    for attempt in range(3):
        current = draft(release, release_id)
        existing = next((item for item in current["assets"] if item["name"] == path.name), None)
        if existing:
            try:
                verify_assets([existing], {path.name: entry})
                return
            except RuntimeError:
                # Only this exact draft's generated package asset can be replaced.
                api(f"{release.prefix}/releases/assets/{existing['id']}", "DELETE")
        endpoint = current["upload_url"].split("{", 1)[0] + "?name=" + urllib.parse.quote(path.name)
        expected_prefix = f"https://uploads.github.com/{release.prefix}/releases/{release_id}/assets?"
        if not endpoint.startswith(expected_prefix):
            raise RuntimeError("Unexpected release upload destination")
        try:
            result = subprocess.check_output(["gh", "api", endpoint, "--method", "POST", "-H",
                                              "Content-Type: application/octet-stream", "--input", str(path)],
                                             text=True, encoding="utf-8", timeout=1800)
            verify_assets([json.loads(result)], {path.name: entry})
            print(f"Uploaded {path.name}", flush=True)
            return
        except (subprocess.SubprocessError, RuntimeError, json.JSONDecodeError):
            # A lost response may still have uploaded successfully; inspect it first.
            current = draft(release, release_id)
            existing = [item for item in current["assets"] if item["name"] == path.name]
            if existing:
                try:
                    verify_assets(existing, {path.name: entry})
                    return
                except RuntimeError:
                    pass
            if attempt == 2:
                raise
            time.sleep(10 * (attempt + 1))


def stage(release, root):
    verify_workflow(release)
    check_tag(release)
    artifacts = api(f"{release.prefix}/actions/runs/{release.run_id}/artifacts?per_page=100")
    package_names = {f"AI-RVC-{platform}-{variant}" for platform in ("Windows", "Linux") for variant in ("CPU", "GPU")}
    expected_artifacts = package_names | {"portable-browser-evidence"}
    records = artifacts["artifacts"]
    if (artifacts["total_count"] != len(expected_artifacts) or len(records) != len(expected_artifacts)
            or {record["name"] for record in records} != expected_artifacts
            or any(record["expired"] for record in records)):
        raise RuntimeError("All four original build artifacts and the browser evidence are required")
    # The downloaded directory contains package artifacts only.
    verified, paths = verified_package_files(root)
    expected = {name: item for entries in verified.values() for name, item in entries.items()}
    current = find_release(release)
    if current:
        if not current["draft"]:
            raise RuntimeError("A public release cannot be replaced")
        if any(not PACKAGE_NAME.fullmatch(item["name"]) and item["name"] not in REPORT_ASSETS for item in current["assets"]):
            raise RuntimeError("The draft contains an unrelated attachment; refusing to delete it")
    payload = dict(tag_name=release.tag, target_commitish=release.source, name=f"AI-RVC {release.tag}",
                   body=release.notes, draft=True, prerelease=False)
    if current:
        current = api(f"{release.prefix}/releases/{current['id']}", "PATCH", payload)
    else:
        current = api(f"{release.prefix}/releases", "POST", payload)
    release_id = current["id"]
    current = draft(release, release_id)
    for item in current["assets"]:
        if item["name"] not in expected:
            api(f"{release.prefix}/releases/assets/{item['id']}", "DELETE")
    for path in paths:
        upload(release, release_id, path, expected[path.name])
    verify_assets(draft(release, release_id)["assets"], expected)
    return dict(source=release.source, run_id=release.run_id, tag=release.tag, release_id=release_id, assets=expected)


def read_url(url, headers=None):
    request = urllib.request.Request(url, headers={"User-Agent": "AI-RVC-release", **(headers or {})})
    with urllib.request.urlopen(request, timeout=120) as response:
        data = response.read(16 * 1024 * 1024 + 1)
        if len(data) > 16 * 1024 * 1024:
            raise RuntimeError("Unexpectedly large release metadata response")
        return data


def check_image_config(config, release):
    labels = config["config"]["Labels"]
    if labels.get("org.opencontainers.image.revision") != release.source or labels.get("org.opencontainers.image.version") != release.version:
        raise RuntimeError("The published image revision or version does not match this release")
    if labels.get("org.opencontainers.image.source", "").lower() != f"https://github.com/{release.repository}".lower():
        raise RuntimeError("Unexpected image source repository")


def verify_public_images(release):
    repository = release.repository.lower()
    query = urllib.parse.urlencode({"service": "ghcr.io", "scope": f"repository:{repository}:pull"})
    token = json.loads(read_url("https://ghcr.io/token?" + query))["token"]
    headers = {"Authorization": "Bearer " + token,
               "Accept": "application/vnd.oci.image.index.v1+json, application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json, application/vnd.docker.distribution.manifest.list.v2+json"}
    result = {}
    for variant in ("cpu", "cuda"):
        base = f"https://ghcr.io/v2/{repository}"
        index_bytes = read_url(f"{base}/manifests/{release.version}-{variant}", headers)
        manifest = json.loads(index_bytes)
        if "manifests" in manifest:
            entries = [item for item in manifest["manifests"] if item.get("platform", {}).get("os") == "linux" and item["platform"].get("architecture") == "amd64"]
            if len(entries) != 1:
                raise RuntimeError("Expected exactly one Linux amd64 image")
            digest = entries[0]["digest"]
            data = read_url(f"{base}/manifests/{digest}", headers)
            if "sha256:" + hashlib.sha256(data).hexdigest() != digest:
                raise RuntimeError("Image manifest digest mismatch")
            manifest = json.loads(data)
        digest = manifest["config"]["digest"]
        config = read_url(f"{base}/blobs/{digest}", headers)
        if "sha256:" + hashlib.sha256(config).hexdigest() != digest:
            raise RuntimeError("Image configuration digest mismatch")
        check_image_config(json.loads(config), release)
        result[variant] = "sha256:" + hashlib.sha256(index_bytes).hexdigest()
        print(f"Verified public ghcr.io/{repository}:{release.version}-{variant} @ {result[variant]}", flush=True)
    return result


def publish(release, receipt):
    if (receipt["source"], receipt["run_id"], receipt["tag"]) != (release.source, release.run_id, release.tag):
        raise RuntimeError("The package receipt belongs to another release run")
    verify_workflow(release, include_images=True)
    current = api(f"{release.prefix}/releases/{receipt['release_id']}")
    if current["tag_name"] != release.tag or current["target_commitish"] != release.source or current["body"] != release.notes:
        raise RuntimeError("Release destination or notes changed after staging")
    verify_assets(current["assets"], receipt["assets"])
    verify_public_images(release)
    if not check_tag(release):
        api(f"{release.prefix}/git/refs", "POST", {"ref": "refs/tags/" + release.tag, "sha": release.source})
    check_tag(release)
    if current["draft"]:
        current = api(f"{release.prefix}/releases/{current['id']}", "PATCH",
                      {"tag_name": release.tag, "target_commitish": release.source, "draft": False, "make_latest": "true"})
    if current["draft"]:
        raise RuntimeError("The release remains unpublished")
    # The public API and four small checksum files are checked without a login.
    public = json.loads(read_url(f"https://api.github.com/{release.prefix}/releases/tags/{release.tag}"))
    verify_assets(public["assets"], receipt["assets"])
    for item in public["assets"]:
        if item["name"].endswith(".SHA256SUMS"):
            data = read_url(item["browser_download_url"])
            if "sha256:" + hashlib.sha256(data).hexdigest() != item["digest"]:
                raise RuntimeError("Public checksum download mismatch")
    print(f"Published {current['html_url']}", flush=True)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        Path(summary).write_text(f"[{release.tag}]({current['html_url']}) — packages, checksums and both container images are published.\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "stage", "publish"))
    phase = parser.parse_args().phase
    release = context()
    if phase == "prepare":
        prepare(release)
    elif phase == "stage":
        receipt = stage(release, Path(os.environ["STAGED_ARTIFACT_DIRECTORY"]))
        with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as stream:
            stream.write("receipt=" + json.dumps(receipt, separators=(",", ":")) + "\n")
    else:
        publish(release, json.loads(os.environ["RELEASE_RECEIPT"]))


if __name__ == "__main__":
    main()
