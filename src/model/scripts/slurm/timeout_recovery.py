#!/usr/bin/env python3.11
"""Persistent timeout-only, two-socket recovery. init never submits jobs.

All pipeline jobs must be adopted: untracked jobs cannot delay singleton dispatch.
An ambiguous sbatch intent deliberately requires operator accounting reconciliation.
"""
import argparse
import contextlib
import fcntl
import getpass
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import uuid

try:
    import tomllib
except ImportError:
    import tomli as tomllib

ACTIVE = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED",
          "RESIZING", "REQUEUED", "REQUEUE_FED", "REQUEUE_HOLD", "SIGNALING", "STAGE_OUT"}
STAGES = ("eigen", "svd")


class Blocked(RuntimeError):
    pass


def clone(value):
    return json.loads(json.dumps(value))


def new_state(root, repo_root, seed_count, points):
    state = {"version": 1, "root": str(Path(root).resolve()), "repo_root": str(Path(repo_root).resolve()),
             "base_seed_count": seed_count, "next_seed": seed_count + 1,
             "selected": {}, "jobs": [], "intents": [], "final_script": None,
             "julia": os.environ.get("JULIA", "/users/ppacchio/.src/juliaup/julia-1.12.6+0.x64.linux.gnu/bin/julia"),
            "ai_reference": os.environ.get("AI_REFERENCE", ""), "status": "initialized"}
    extend_selection(state, points)
    return state


def extend_selection(state, points):
    for point in points:
        for stage in STAGES:
            key = point["point_id"] + ":" + stage
            state["selected"].setdefault(key, {"slot_id": point["point_id"], "stage": stage,
                                               "seed": point["seed"], "directory": state["root"]})


def task_for(key, selected, stage=None):
    return {**selected, "stage": stage or selected["stage"], "targets": [key]}


def adopt(state, records):
    known = {j["job_id"] for j in state["jobs"]}
    for record in records:
        jid = str(record["job_id"])
        if not re.fullmatch(r"\d+(?:_\d+)?", jid):
            raise ValueError("adopt requires a concrete job ID, not an array range")
        if jid in known:
            continue
        stage = record.get("stage")
        if stage not in (*STAGES, "generate", "external"):
            raise ValueError("invalid adopted stage: " + str(stage))
        tasks = []
        for slot in record.get("slot_ids", []):
            for target_stage in STAGES if stage == "generate" else (stage,):
                key = slot + ":" + target_stage
                selected = state["selected"][key]
                if selected["directory"] != state["root"]:
                    raise ValueError("original adoption would overwrite an existing replacement")
                tasks.append(task_for(key, selected, stage))
        if stage != "external" and not tasks:
            raise ValueError("adopted work job needs slot_ids")
        state["jobs"].append({"job_id": jid, "kind": "external" if stage == "external" else "work",
                              "status": "PENDING", "tasks": deduplicate_generation(tasks)})
        known.add(jid)


def deduplicate_generation(tasks):
    result = []
    seen = {}
    for task in tasks:
        identity = (task["slot_id"], task["seed"], task["directory"], task["stage"])
        if identity in seen:
            seen[identity]["targets"].extend(task["targets"])
        else:
            task = clone(task)
            result.append(task)
            seen[identity] = task
    return result


def inventory_key(task):
    return json.dumps([task["slot_id"], task["seed"], task["directory"], task["stage"]], separators=(",", ":"))


class Controller:
    def __init__(self, state, backend, save):
        self.state, self.backend, self.save = state, backend, save

    def block(self, message):
        self.state["status"] = "blocked"
        self.state["error"] = message
        self.save()
        raise Blocked(message)

    def submit(self, kind, tasks=None, dependencies=None):
        intent = {"name": "recovery-" + kind + "-" + uuid.uuid4().hex[:16], "kind": kind,
                  "tasks": tasks or [], "dependencies": sorted(dependencies or []), "status": "submitting"}
        self.state["intents"].append(intent)
        self.save()  # durable write BEFORE the scheduler can accept the request
        try:
            jid = self.backend.submit(intent)
        except Exception as exc:
            intent["status"] = "ambiguous"
            self.block("Submission outcome unknown for " + intent["name"] + ": " + str(exc))
        intent.update(status="submitted", job_id=jid)
        job = {**clone(intent), "status": "PENDING"}
        self.state["jobs"].append(job)
        self.save()
        return job

    def replace(self, key):
        old = self.state["selected"][key]
        seed = self.state["next_seed"]
        self.state["next_seed"] += 1
        self.state["selected"][key] = {**old, "seed": seed,
            "directory": str(Path(self.state["root"]) / "recovery" / "attempts" / ("seed-" + str(seed)))}

    def reconcile(self, current_watch=None):
        if any(i["status"] in {"submitting", "ambiguous"} for i in self.state["intents"]):
            self.block("Unresolved sbatch intent: reconcile its unique name with Slurm before resuming")
        jobs = self.state["jobs"]
        ids = [j["job_id"] for j in jobs if not j.get("handled")]
        statuses = self.backend.accounting(ids) if ids else {}
        for job in jobs:
            if job.get("handled"):
                continue
            if job["job_id"] == current_watch:
                job.update(status="COMPLETED", handled=True)
                continue
            status = statuses.get(job["job_id"])
            if not status:
                self.block("Accounting unavailable for job " + job["job_id"])
            job["status"] = status
        for job in jobs:
            if job["kind"] == "final" and job["status"] not in ACTIVE and job["status"] != "COMPLETED":
                self.block("Final analysis ended " + job["status"] + "; scientific seed replacement is not applicable")
        requests = {}
        for key, selected in self.state["selected"].items():
            for stage in ("generate", selected["stage"]):
                task = task_for(key, selected, stage)
                requests[inventory_key(task)] = {**task, "root": self.state["root"], "key": inventory_key(task)}
        for job in jobs:
            if not job.get("handled"):
                for task in job["tasks"]:
                    requests[inventory_key(task)] = {**task, "root": self.state["root"], "key": inventory_key(task)}
        inventory = self.backend.inventory(list(requests.values()))
        for key in requests:
            if inventory.get(key) not in {"complete", "missing"}:
                self.block("Artifact invalid or unvalidated: " + key)
        # Validate all terminal jobs before mutating any selection.
        for job in jobs:
            if job.get("handled") or job["kind"] == "watch" or job["status"] in ACTIVE:
                continue
            if job["status"] not in {"COMPLETED", "TIMEOUT"}:
                self.block("Job " + job["job_id"] + " ended " + job["status"] + "; only TIMEOUT permits a new seed")
            failures = self.backend.worker_failures(job)
            if failures:
                self.block("Job " + job["job_id"] + " has a non-timeout worker failure: " + "; ".join(failures))
            if job["status"] == "COMPLETED" and any(inventory[inventory_key(t)] != "complete" for t in job["tasks"]):
                self.block("Completed job " + job["job_id"] + " has missing artifacts")
        replaced = set()
        for job in jobs:
            if job.get("handled") or job["status"] in ACTIVE:
                continue
            if job["kind"] == "watch":
                job["handled"] = True
                continue
            for task in job["tasks"]:
                if inventory[inventory_key(task)] == "complete":
                    continue
                for key in task["targets"]:
                    selected = self.state["selected"][key]
                    if (selected["seed"], selected["directory"]) != (task["seed"], task["directory"]):
                        continue
                    # Generation timeout must not replace an already valid spectral result.
                    if inventory.get(inventory_key(selected)) == "complete":
                        continue
                    if key not in replaced:
                        self.replace(key)
                        replaced.add(key)
            job["handled"] = True
            if job["kind"] == "final":
                if job["status"] != "COMPLETED":
                    self.block("Final analysis timed out; scientific seed replacement is not applicable")
                self.state["status"] = "complete"
        self.save()
        active = [j for j in jobs if not j.get("handled") and j["kind"] in {"work", "external"} and j["status"] in ACTIVE]
        busy = {key for j in active for t in j["tasks"] for key in t["targets"]}
        queues = {"generate": [], "eigen": [], "svd": []}
        all_complete = True
        for key, selected in self.state["selected"].items():
            if inventory.get(inventory_key(selected)) == "complete":
                continue
            all_complete = False
            if key in busy:
                continue
            generation = task_for(key, selected, "generate")
            phase = selected["stage"] if inventory.get(inventory_key(generation)) == "complete" else "generate"
            queues[phase].append(task_for(key, selected) if phase != "generate" else generation)
        queues["generate"] = deduplicate_generation(queues["generate"])
        for queue in queues.values():
            while len(queue) >= 2:
                active.append(self.submit("work", tasks=[queue.pop(0), queue.pop(0)]))
        if not active:
            # Dispatch only one residual singleton; the next one waits for its completion.
            for queue in queues.values():
                if queue:
                    active.append(self.submit("work", tasks=[queue.pop(0)]))
                    break
        if all_complete and not active:
            final_jobs = [j for j in jobs if j["kind"] == "final"]
            if self.state.get("final_script") and not final_jobs:
                self.submit("final")
                self.state["status"] = "final_submitted"
            elif not self.state.get("final_script"):
                self.state["status"] = "artifacts_complete"
            else:
                self.state["status"] = "complete" if final_jobs[-1].get("handled") and final_jobs[-1]["status"] == "COMPLETED" else "final_submitted"
        elif active:
            self.state["status"] = "running"
        watched_jobs = active + [j for j in jobs if j["kind"] == "final" and not j.get("handled") and j["status"] in ACTIVE]
        if watched_jobs:
            dependencies = sorted(j["job_id"] for j in watched_jobs)
            watchers = [j for j in jobs if j["kind"] == "watch" and not j.get("handled") and j["status"] in ACTIVE]
            if not watchers:
                self.submit("watch", dependencies=dependencies)
            elif watchers[0]["status"] == "PENDING" and watchers[0]["dependencies"] != dependencies:
                self.backend.update_watch(watchers[0]["job_id"], dependencies)
                watchers[0]["dependencies"] = dependencies
        self.state.pop("error", None)
        self.save()


def write_toml_tables(path, section, rows):
    lines = []
    for row in rows:
        lines.append("[[" + section + "]]")
        for key, value in row.items():
            if isinstance(value, (str, int)):
                lines.append(key + " = " + json.dumps(value, ensure_ascii=False))
        lines.append("")
    atomic_write(path, "\n".join(lines))


def atomic_write(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temp = stream.name
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def save_state(state):
    directory = Path(state["root"]) / "recovery"
    atomic_write(directory / "state.json", json.dumps(state, indent=2) + "\n")
    write_toml_tables(directory / "selection.toml", "replacements",
                      [s for s in state["selected"].values() if s["directory"] != state["root"]])


@contextlib.contextmanager
def locked(root):
    directory = Path(root) / "recovery"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "state.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield directory


def run(command, **kwargs):
    return subprocess.run(command, check=True, text=True, capture_output=True, **kwargs).stdout.strip()


def dependency_expression(ids):
    return "?".join("afterany:" + jid for jid in ids)


class SlurmBackend:
    def __init__(self, state):
        self.state = state
        self.repo = Path(state["repo_root"])
        self.directory = Path(state["root"]) / "recovery"

    def julia(self, *args):
        return run([self.state["julia"], "--project=" + str(self.repo / "src/environment"),
                    str(self.repo / "src/model/scripts/recovery_point.jl"), *map(str, args)])

    def inventory(self, requests):
        request_path, output_path = self.directory / "inventory_request.toml", self.directory / "inventory.toml"
        write_toml_tables(request_path, "requests", requests)
        self.julia("inventory", request_path, output_path)
        with output_path.open("rb") as stream:
            result = tomllib.load(stream)
        rows = result.get("artifacts", [])
        if len({r["key"] for r in rows}) != len(rows):
            raise Blocked("Duplicate inventory keys")
        return {r["key"]: r["status"] for r in rows}

    def accounting(self, ids):
        if not ids:
            return {}
        result = {}
        # sacct can lag the queue: active squeue states take precedence.
        output = run(["sacct", "-n", "-P", "-X", "-j", ",".join(ids), "--format=JobID%40,State%40"])
        for line in output.splitlines():
            fields = line.split("|")
            if len(fields) >= 2 and fields[0] in ids:
                result[fields[0]] = fields[1].split()[0].rstrip("+")
        output = run(["squeue", "-h", "-r", "-u", getpass.getuser(), "-o", "%i|%T"])
        for line in output.splitlines():
            jid, state = line.split("|", 1)
            if jid in ids:
                result[jid] = state
        return result

    def update_watch(self, job_id, dependencies):
        run(["scontrol", "update", "JobId=" + job_id, "Dependency=" + dependency_expression(dependencies)])

    def worker_failures(self, job):
        # Legacy adopted jobs have no markers; new workers record normal exits.
        # A rank killed with its allocation has no normal-return marker.
        failures = []
        if job["kind"] != "work" or not job.get("name"):
            return failures
        for rank, task in enumerate(job["tasks"]):
            marker = self.directory / (job["name"] + ".json.rank" + str(rank) + ".json")
            if not marker.exists():
                continue
            record = json.loads(marker.read_text())
            if any(record.get(k) != task[k] for k in ("slot_id", "seed", "stage", "directory")):
                raise Blocked("Worker exit marker identity mismatch: " + str(marker))
            if record["exit_code"] != 0:
                failures.append("rank " + str(rank) + " exited " + str(record["exit_code"]))
        return failures

    def submit(self, intent):
        state = self.state
        env = os.environ.copy()
        env.update(REPO_ROOT=state["repo_root"], PILOT_ROOT=state["root"], JULIA=state["julia"],
                   OPENBLAS_THREADS_PER_WORKER="64", RECOVERY_SELECTION=str(self.directory / "selection.toml"))
        if state.get("ai_reference"):
            env["AI_REFERENCE"] = state["ai_reference"]
        slurm = self.repo / "src/model/scripts/slurm"
        logs = self.directory / "logs"
        logs.mkdir(exist_ok=True)
        args = ["sbatch", "--parsable", "--job-name=" + intent["name"], "--account=go54", "--partition=normal",
                "--uenv-passthrough=use", "--exclude=nid001738,nid001588", "--export=ALL",
                "--output=" + str(logs / "%x-%j.out"), "--error=" + str(logs / "%x-%j.err")]
        if intent["kind"] == "work":
            generation = all(t["stage"] == "generate" for t in intent["tasks"])
            args += ["--nodes=1", "--ntasks=2", "--ntasks-per-socket=1", "--cpus-per-task=128", "--hint=multithread", "--mem=450G",
                     "--time=" + ("00:30:00" if generation else "24:00:00")]
            job_file = self.directory / (intent["name"] + ".json")
            atomic_write(job_file, json.dumps(intent))
            args += [str(slurm / "recovery_pair.sbatch"), str(job_file)]
        elif intent["kind"] == "watch":
            args += ["--nodes=1", "--ntasks=1", "--cpus-per-task=2", "--mem=4G", "--time=00:10:00",
                     "--dependency=" + dependency_expression(intent["dependencies"]), str(slurm / "recovery_watch.sbatch")]
        else:
            args += [state["final_script"]]
        response = run(args, env=env)
        jid = response.split(";", 1)[0]
        if not jid.isdigit():
            raise RuntimeError("Invalid sbatch response: " + response)
        return jid


def parse_seeds(value):
    if ":" in value:
        first, last = map(int, value.split(":"))
        result = set(range(first, last + 1))
    else:
        result = {int(s) for s in value.split(",")}
    if not result or min(result) < 1:
        raise ValueError("seeds must be positive and nonempty")
    return result


def worker_command(repo, job_file, tasks):
    return ["srun", "--nodes=1", "--ntasks=" + str(len(tasks)), "--ntasks-per-socket=1",
            "--cpus-per-task=128", "--distribution=block:block", "--cpu-bind=verbose,sockets",
            "--mem-bind=verbose,local", "--kill-on-bad-exit=0",
            os.environ.get("PYTHON", "/usr/bin/python3.11"),
            str(Path(repo) / "src/model/scripts/slurm/timeout_recovery.py"),
            "rank", "--root", os.environ.get("PILOT_ROOT", "/unused"), "--job-file", str(job_file)]


def worker(args):
    intent = json.loads(Path(args.job_file).read_text())
    tasks = intent["tasks"]
    if not 1 <= len(tasks) <= 2 or any(t["stage"] not in ("generate", *STAGES) for t in tasks):
        raise ValueError("worker requires one or two generation/spectral tasks")
    if len({t["stage"] for t in tasks}) != 1:
        raise ValueError("generation, eigen and svd tasks require separate jobs")
    repo = os.environ["REPO_ROOT"]
    env = os.environ.copy()
    env.update(JULIA_NUM_THREADS="1", OPENBLAS_NUM_THREADS="64", OPENBLAS_THREADS_PER_WORKER="64", OMP_NUM_THREADS="1")
    if args.command == "worker":
        return subprocess.call(worker_command(repo, args.job_file, tasks), env=env)
    task = tasks[int(os.environ["SLURM_PROCID"])]
    print("Recovery worker: " + json.dumps({"slot_id": task["slot_id"], "seed": task["seed"],
          "stage": task["stage"], "directory": task["directory"], "blas_threads": 64,
          "rank": os.environ["SLURM_PROCID"]}), flush=True)
    try:
        result = subprocess.call([env["JULIA"], "--project=" + str(Path(repo) / "src/environment"),
                                str(Path(repo) / "src/model/scripts/recovery_point.jl"), "run", task["stage"],
                                args.root, task["slot_id"], str(task["seed"]), task["directory"]], env=env)
    except OSError as exc:
        print("Recovery worker launch failed: " + str(exc), file=sys.stderr, flush=True)
        result = 1
    marker = args.job_file + ".rank" + os.environ["SLURM_PROCID"] + ".json"
    atomic_write(marker, json.dumps({**task, "exit_code": result}) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["init", "reconcile", "status", "resolve-intent", "worker", "rank"])
    parser.add_argument("--root", required=True)
    parser.add_argument("--repo-root", default=os.environ.get("REPO_ROOT", str(Path(__file__).resolve().parents[4])))
    parser.add_argument("--seeds", default="1:2")
    parser.add_argument("--adopt", help="JSON array: job_id, stage, slot_ids; external tracks other pipeline jobs")
    parser.add_argument("--final-script")
    parser.add_argument("--intent-name")
    parser.add_argument("--job-id", help="Existing accepted job ID for an ambiguous intent; never resubmits")
    parser.add_argument("--job-file")
    args = parser.parse_args(argv)
    if args.command in {"worker", "rank"}:
        sys.exit(worker(args))
    with locked(args.root) as directory:
        state_file = directory / "state.json"
        state = json.loads(state_file.read_text()) if state_file.exists() else None
        if args.command == "init":
            if state is None:
                provisional = new_state(args.root, args.repo_root, 0, [])
                SlurmBackend(provisional).julia("metadata", provisional["root"])
            with (directory / "base_manifest.toml").open("rb") as stream:
                manifest = tomllib.load(stream)
            seeds = parse_seeds(args.seeds)
            count = manifest["resolved_config"]["grid"]["seed_count"]
            if max(seeds) > count:
                raise ValueError("requested seed exceeds immutable base seed_count")
            points = [p for p in manifest["points"] if p["seed"] in seeds]
            if not points:
                raise ValueError("no points selected")
            if state is None:
                state = new_state(args.root, args.repo_root, count, points)
            else:
                if str(Path(args.root).resolve()) != state["root"] or str(Path(args.repo_root).resolve()) != state["repo_root"]:
                    raise ValueError("existing recovery root/repo identity differs")
                expanding = any(p["point_id"] + ":eigen" not in state["selected"] for p in points)
                if expanding and state.get("final_script") and not args.final_script:
                    raise ValueError("expanding a pilot requires an explicit --final-script for the expanded ensemble")
                finals = [j for j in state["jobs"] if j["kind"] == "final"]
                if any(not j.get("handled") or j["status"] != "COMPLETED" for j in finals):
                    raise ValueError("reconcile final analysis completion before expanding the selection")
                for job in finals:
                    job["kind"] = "final_history"
                extend_selection(state, points)
                state["status"] = "initialized"
            if args.adopt:
                adopt(state, json.loads(Path(args.adopt).read_text()))
            if args.final_script:
                state["final_script"] = str(Path(args.final_script).resolve())
            save_state(state)
        elif state is None:
            raise ValueError("run init first")
        elif args.command == "reconcile":
            Controller(state, SlurmBackend(state), lambda: save_state(state)).reconcile(os.environ.get("SLURM_JOB_ID"))
        elif args.command == "resolve-intent":
            if not args.intent_name or not args.job_id or not args.job_id.isdigit():
                raise ValueError("resolve-intent requires --intent-name and accepted --job-id")
            intent = next(i for i in state["intents"] if i["name"] == args.intent_name)
            if intent["status"] not in {"ambiguous", "submitting"}:
                raise ValueError("intent already resolved")
            # Caller must have verified the unique job name against accounting.
            name = run(["sacct", "-n", "-P", "-X", "-j", args.job_id, "--format=JobName%100"]).strip("|\n ")
            if name != intent["name"]:
                raise ValueError("accounting job name does not match intent")
            intent.update(status="submitted", job_id=args.job_id)
            state["jobs"].append({**clone(intent), "status": "PENDING"})
            save_state(state)
        print(json.dumps({"status": state["status"], "selected_stages": len(state["selected"]),
                          "next_seed": state["next_seed"], "error": state.get("error")}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (Blocked, ValueError, subprocess.CalledProcessError) as exc:
        print("Recovery blocked: " + str(exc), file=sys.stderr)
        sys.exit(2)
