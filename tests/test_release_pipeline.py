import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import release_pipeline as pipeline


class ReleasePipelineTests(unittest.TestCase):
    def setUp(self):
        self.release = pipeline.Release("owner/repo", "a" * 40, "123", "1.6.0", "更新\n\n## English\nUpdate")
        self.run = dict(head_sha=self.release.source, path=".github/workflows/release.yml",
                        head_repository={"full_name": "owner/repo"}, event="workflow_dispatch",
                        status="in_progress", conclusion=None)
        self.names = [f"AI-RVC-{system}-{variant}" for system in ("Windows", "Linux") for variant in ("CPU", "GPU")]
        job_names = [name.replace("AI-RVC-", "packages / Build ") for name in self.names]
        job_names += [f"contracts / {system} / Python 3.10" for system in ("windows-latest", "ubuntu-latest", "macos-15")]
        job_names += ["browser / Verify Windows portable browser"]
        job_names += ["images / Docker cpu", "images / Docker cuda"]
        self.jobs = [dict(name=name, status="completed", conclusion="success") for name in job_names]
        self.assets = dict(total_count=4, artifacts=[dict(name=name, expired=False) for name in self.names])
        self.current = dict(id=42, draft=True, target_commitish="b" * 40, tag_name="v1.6.0", assets=[],
                            upload_url="https://uploads.github.com/repos/owner/repo/releases/42/assets{?name,label}",
                            html_url="https://github.com/owner/repo/releases/tag/v1.6.0")
        self.tag = None
        self.writes = []
        self.data = {}

    def api(self, path, method="GET", payload=None, missing_ok=False):
        if method != "GET":
            self.writes.append((path, method, copy.deepcopy(payload)))
        if "/git/ref/tags/" in path:
            return copy.deepcopy(self.tag)
        if path.endswith("/git/refs"):
            self.tag = {"object": {"type": "commit", "sha": payload["sha"]}}
            return copy.deepcopy(self.tag)
        if "/jobs?" in path:
            return dict(total_count=len(self.jobs), jobs=copy.deepcopy(self.jobs))
        if "/artifacts?" in path:
            return copy.deepcopy(self.assets)
        if "/actions/runs/" in path:
            return copy.deepcopy(self.run)
        if "/releases?" in path:
            return [copy.deepcopy(self.current)]
        if "/releases/assets/" in path and method == "DELETE":
            identity = int(path.rsplit("/", 1)[1])
            self.current["assets"] = [item for item in self.current["assets"] if item["id"] != identity]
            return None
        if path.endswith("/releases/42"):
            if payload:
                self.current.update(payload)
            return copy.deepcopy(self.current)
        raise AssertionError((path, method, payload))

    def upload(self, release, release_id, path, entry):
        self.data[path.name] = path.read_bytes()
        self.current["assets"] = [item for item in self.current["assets"] if item["name"] != path.name]
        self.current["assets"].append(dict(id=len(self.data), name=path.name, state="uploaded",
                                            digest="sha256:" + entry["sha256"], size=entry["bytes"],
                                            browser_download_url="https://example.test/" + path.name))

    def read_url(self, url, headers=None):
        if "api.github.com" in url:
            return json.dumps(self.current).encode()
        return self.data[url.rsplit("/", 1)[1]]

    def stage(self, corrupt=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in self.names:
                folder = root / name
                folder.mkdir()
                filename = name + ("-Portable.7z.001" if "Windows" in name else "-Portable.tar.gz.partaa")
                data = name.encode()
                (folder / filename).write_bytes(data)
                digest = "0" * 64 if corrupt else hashlib.sha256(data).hexdigest()
                (folder / (name + "-Portable.SHA256SUMS")).write_text(f"{digest}  {filename}\n")
            with patch.object(pipeline, "api", side_effect=self.api), patch.object(pipeline, "upload", side_effect=self.upload):
                return pipeline.stage(self.release, root)

    def test_stage_validates_all_originals_and_excludes_reports(self):
        self.current["assets"] = [dict(id=100 + index, name=name) for index, name in enumerate(pipeline.REPORT_ASSETS)]
        receipt = self.stage()
        self.assertEqual(len(receipt["assets"]), 8)
        self.assertTrue(self.current["draft"])
        self.assertEqual(self.current["target_commitish"], self.release.source)
        self.assertTrue(set(receipt["assets"]).isdisjoint(pipeline.REPORT_ASSETS))
        self.assertEqual(sum(method == "DELETE" for _, method, _ in self.writes), 4)

    def test_failed_portable_job_prevents_any_draft_mutation(self):
        self.jobs[0]["conclusion"] = "failure"
        with self.assertRaisesRegex(RuntimeError, "Required job"):
            self.stage()
        self.assertEqual(self.writes, [])

    def test_skipped_platform_job_is_not_a_pass(self):
        self.jobs[4]["conclusion"] = "skipped"
        with self.assertRaisesRegex(RuntimeError, "Required job"):
            self.stage()
        self.assertEqual(self.writes, [])

    def test_wrong_source_and_cancelled_runs_are_rejected(self):
        for field, value in [("head_sha", "c" * 40), ("status", "completed"), ("event", "push")]:
            with self.subTest(field=field):
                original = self.run[field]
                self.run[field] = value
                with self.assertRaisesRegex(RuntimeError, "exact release workflow"):
                    self.stage()
                self.run[field] = original
        self.assertEqual(self.writes, [])

    def test_corrupt_archive_is_rejected_before_replacing_draft(self):
        with self.assertRaisesRegex(RuntimeError, "mismatched archive"):
            self.stage(corrupt=True)
        self.assertEqual(self.writes, [])

    def test_missing_or_expired_artifact_is_rejected(self):
        self.assets["artifacts"][0]["expired"] = True
        with self.assertRaisesRegex(RuntimeError, "four original"):
            self.stage()
        self.assertEqual(self.writes, [])

    def test_public_release_is_never_replaced(self):
        self.current["draft"] = False
        with self.assertRaisesRegex(RuntimeError, "public release"):
            self.stage()
        self.assertEqual(self.writes, [])

    def test_unrelated_draft_asset_is_never_deleted(self):
        self.current["assets"] = [dict(id=90, name="user-backup.zip")]
        with self.assertRaisesRegex(RuntimeError, "unrelated attachment"):
            self.stage()
        self.assertEqual(self.writes, [])

    def test_existing_tag_at_another_commit_is_never_changed(self):
        self.tag = {"object": {"type": "commit", "sha": "c" * 40}}
        with patch.object(pipeline, "api", side_effect=self.api), self.assertRaisesRegex(RuntimeError, "another commit"):
            pipeline.prepare(self.release)
        self.assertEqual(self.writes, [])

    def test_bad_remote_digest_and_extra_report_are_rejected(self):
        receipt = self.stage()
        bad = copy.deepcopy(self.current["assets"])
        bad[0]["digest"] = "sha256:" + "0" * 64
        with self.assertRaisesRegex(RuntimeError, "verification failed"):
            pipeline.verify_assets(bad, receipt["assets"])
        with self.assertRaisesRegex(RuntimeError, "only the complete"):
            pipeline.verify_assets(self.current["assets"] + [dict(name="RELEASE-VALIDATION.md")], receipt["assets"])

    def test_image_source_and_version_must_match_exactly(self):
        labels = {"org.opencontainers.image.revision": self.release.source,
                  "org.opencontainers.image.version": "1.6.0", "org.opencontainers.image.source": "https://github.com/owner/repo"}
        pipeline.check_image_config({"config": {"Labels": labels}}, self.release)
        for key in labels:
            with self.subTest(key=key):
                bad = dict(labels, **{key: "incorrect"})
                with self.assertRaises(RuntimeError):
                    pipeline.check_image_config({"config": {"Labels": bad}}, self.release)

    def test_container_failure_prevents_tag_and_publication(self):
        receipt = self.stage()
        self.writes.clear()
        self.jobs[-1]["conclusion"] = "failure"
        with patch.object(pipeline, "api", side_effect=self.api), self.assertRaisesRegex(RuntimeError, "Docker cuda"):
            pipeline.publish(self.release, receipt)
        self.assertEqual(self.writes, [])
        self.assertIsNone(self.tag)

    def test_publication_checks_images_before_creating_tag(self):
        receipt = self.stage()
        self.writes.clear()
        with patch.object(pipeline, "api", side_effect=self.api), \
             patch.object(pipeline, "verify_public_images", side_effect=RuntimeError("Image mismatch")), \
             self.assertRaisesRegex(RuntimeError, "Image mismatch"):
            pipeline.publish(self.release, receipt)
        self.assertEqual(self.writes, [])
        self.assertTrue(self.current["draft"])

    def test_success_publishes_matching_tag_and_anonymous_checksums(self):
        receipt = self.stage()
        with patch.object(pipeline, "api", side_effect=self.api), \
             patch.object(pipeline, "verify_public_images", return_value={"cpu": "digest1", "cuda": "digest2"}), \
             patch.object(pipeline, "read_url", side_effect=self.read_url), \
             patch.dict("os.environ", {"GITHUB_STEP_SUMMARY": ""}):
            pipeline.publish(self.release, receipt)
        self.assertFalse(self.current["draft"])
        self.assertEqual(self.tag["object"]["sha"], self.release.source)
        self.assertEqual(self.current["make_latest"], "true")


if __name__ == "__main__":
    unittest.main()
