"""Lecture 09 notebook — serving models with FastAPI."""
from nbtools import md, code, section, todo, todo_cell


def cells():
    return [
        md("""
        > **Before you run this notebook**, build the model artefact:
        >
        > ```bash
        > cd lectures/Lecture_09/app
        > python train_and_export.py
        > ```
        >
        > That takes about two minutes and writes `app/artefacts/shape_classifier.pt`
        > plus four sample PNGs. Everything below depends on it.
        """),
        section("1. A checkpoint is not a deployable artefact", """
        The most common production bug in machine learning is not a wrong architecture.
        It is preprocessing that differs between training and serving.

        Nothing raises an exception. The service returns confident, wrong answers, and it
        is often weeks before anyone notices.
        """),
        code("""
        import sys
        APP = ROOT / "lectures" / "Lecture_09" / "app"
        sys.path.insert(0, str(APP))

        from model import ModelBundle, PreprocessConfig, ShapeClassifier

        bundle = ModelBundle.load()
        print("version   :", bundle.version)
        print("classes   :", bundle.classes)
        print("metrics   :", bundle.metrics)
        print("\\npreprocessing shipped with the weights:")
        for k, v in vars(bundle.preprocess).items():
            print(f"  {k:<12} {v}")
        """),
        code("""
        # Demonstrate the failure. Same weights, wrong normalisation statistics.
        from PIL import Image
        import numpy as np

        raw_bytes = (APP / "samples" / "circle.png").read_bytes()

        correct = bundle.bytes_to_tensor(raw_bytes)
        probs_ok, _ = bundle.predict(correct)

        # A plausible mistake: using ImageNet statistics because a tutorial did.
        wrong_bundle = ModelBundle(
            model=bundle.model, classes=bundle.classes,
            preprocess=PreprocessConfig(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            version="wrong-preprocessing", metrics={},
        )
        probs_bad, _ = wrong_bundle.predict(wrong_bundle.bytes_to_tensor(raw_bytes))

        print(f"{'class':<10}{'correct':>10}{'wrong stats':>14}")
        print("-" * 34)
        for c, a, b in zip(bundle.classes, probs_ok[0], probs_bad[0]):
            print(f"{c:<10}{a:>10.4f}{b:>14.4f}")
        print(f"\\ncorrect preprocessing -> {bundle.top_prediction(probs_ok[0])}")
        print(f"wrong   preprocessing -> {bundle.top_prediction(probs_bad[0])}")
        print("\\nNo error was raised. That is what makes this bug expensive.")
        """),
        md("""
        ### Export formats

        | Format | Self-contained? | Use it when |
        |---|---|---|
        | `state_dict` | No — needs the class definition | Default for PyTorch-to-PyTorch |
        | TorchScript | Yes | Serving without the original Python class |
        | ONNX | Yes, cross-runtime | Deploying outside Python |
        | Pickled whole model | Fragile | Never, if you can avoid it |

        Pickling the model object embeds your module paths in the file. Move a file, and
        the checkpoint stops loading.
        """),
        section("2. The service", """
        The full application is in `lectures/Lecture_09/app/`. Read `main.py` alongside
        this — the notebook exercises it, it does not replace it.

        ```
        app/
          model.py              architecture, artefact loading, preprocessing, inference
          schemas.py            Pydantic request/response models
          main.py               the FastAPI application
          train_and_export.py   offline: train and write the artefact
          benchmark.py          latency measurement
          tests/test_api.py     endpoint tests
          Dockerfile            multi-stage build
        ```
        """),
        code("""
        print((APP / "main.py").read_text(encoding="utf-8")[:3000])
        """),
        md("""
        ### Four details that matter more than they look

        **1. Load the model in a lifespan handler, not per request.**
        Loading takes hundreds of milliseconds. Doing it per request makes every request
        pay for it.

        **2. `torch.set_num_threads(1)` in the server.**
        Each request already runs in its own worker. Letting every worker spawn N threads
        oversubscribes the CPU and makes tail latency dramatically worse.

        **3. `torch.inference_mode()` around the forward pass.**
        Without it, PyTorch builds an autograd graph it will never use — wasted time and
        memory that accumulates under load.

        **4. Return the model version with every prediction.**
        When someone asks why a prediction changed, this is the only thing that lets you
        answer.
        """),
        section("3. Exercising the API with TestClient", """
        `TestClient` runs the application in-process — no server, no port, no network. It
        is the right tool for tests and for measuring the service's own cost.

        Note the `with` block: it runs the lifespan handler, which is what loads the
        model. Without it, every request returns 503.
        """),
        code("""
        from fastapi.testclient import TestClient
        from main import app

        client = TestClient(app)
        client.__enter__()          # start lifespan (loads the model)

        print("health:", client.get("/health").json())
        """),
        code("""
        import json
        resp = client.post("/predict",
                           files={"file": ("circle.png", raw_bytes, "image/png")})
        print("status:", resp.status_code)
        print(json.dumps(resp.json(), indent=2))
        """),
        code("""
        # Every sample, through the real endpoint.
        import io

        fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
        for ax, name in zip(axes, bundle.classes):
            data_bytes = (APP / "samples" / f"{name}.png").read_bytes()
            r = client.post("/predict", files={"file": (f"{name}.png", data_bytes, "image/png")})
            body = r.json()
            ax.imshow(np.array(Image.open(io.BytesIO(data_bytes))))
            ax.axis("off")
            correct_mark = "correct" if body["predicted_class"] == name else "WRONG"
            ax.set_title(f"true: {name}\\npred: {body['predicted_class']} "
                         f"({body['confidence']:.3f})\\n{correct_mark}", fontsize=9)
        fig.suptitle("Predictions served over HTTP")
        plt.tight_layout(); plt.show()
        """),
        section("4. Failing correctly", """
        A service is defined as much by its error paths as its happy path. Every one of
        these returns a *useful* status code rather than a 500.
        """),
        code("""
        cases = [
            ("wrong content type",
             {"file": ("notes.txt", b"not an image", "text/plain")}, 415),
            ("oversized file",
             {"file": ("big.png", b"\\x89PNG\\r\\n\\x1a\\n" + b"0" * (6 * 1024 * 1024),
                       "image/png")}, 413),
            ("corrupt image",
             {"file": ("bad.png", b"\\x89PNG\\r\\n\\x1a\\ngarbage", "image/png")}, 422),
            ("empty file",
             {"file": ("empty.png", b"", "image/png")}, 422),
        ]

        print(f"{'case':<22}{'expected':>10}{'actual':>8}   detail")
        print("-" * 78)
        for label, files, expected in cases:
            r = client.post("/predict", files=files)
            detail = r.json().get("detail", "")
            detail = detail if isinstance(detail, str) else str(detail)
            flag = "OK" if r.status_code == expected else "MISMATCH"
            print(f"{label:<22}{expected:>10}{r.status_code:>8}   {detail[:38]}  {flag}")
        """),
        code("""
        # Missing the field entirely — FastAPI's own validation handles this.
        r = client.post("/predict")
        print("no file at all ->", r.status_code)
        print(json.dumps(r.json(), indent=2)[:400])
        """),
        section("5. Batching, and why it is faster", """
        Measure it rather than asserting it.
        """),
        code("""
        import time, statistics

        payload = (APP / "samples" / "circle.png").read_bytes()

        # Warm up first — the earliest requests are always slower.
        for _ in range(10):
            client.post("/predict", files={"file": ("c.png", payload, "image/png")})

        def percentile(xs, p):
            s = sorted(xs)
            return s[min(int(len(s) * p), len(s) - 1)]

        single = []
        for _ in range(150):
            t0 = time.perf_counter()
            client.post("/predict", files={"file": ("c.png", payload, "image/png")})
            single.append((time.perf_counter() - t0) * 1000)

        print("POST /predict, 150 sequential requests")
        print(f"  mean {statistics.mean(single):6.2f} ms")
        print(f"  p50  {percentile(single, .50):6.2f} ms")
        print(f"  p95  {percentile(single, .95):6.2f} ms")
        print(f"  p99  {percentile(single, .99):6.2f} ms")
        print(f"  max  {max(single):6.2f} ms")
        """),
        code("""
        batch_results = {1: statistics.mean(single)}
        for bs in (4, 8, 16, 32):
            files = [("files", (f"{i}.png", payload, "image/png")) for i in range(bs)]
            times = []
            for _ in range(15):
                t0 = time.perf_counter()
                r = client.post("/predict/batch", files=files)
                times.append((time.perf_counter() - t0) * 1000)
                assert r.status_code == 200, r.text
            batch_results[bs] = statistics.mean(times) / bs
            print(f"batch size {bs:>2}: {statistics.mean(times):7.2f} ms total, "
                  f"{batch_results[bs]:6.2f} ms per image")
        """),
        code("""
        sizes = list(batch_results)
        per_image = [batch_results[s] for s in sizes]

        fig, ax = plt.subplots(figsize=(7.5, 4))
        ax.plot(sizes, per_image, marker="o", lw=2, color="#c16a1c")
        ax.set_xlabel("batch size"); ax.set_ylabel("milliseconds per image")
        ax.set_title("Batching amortises the per-request overhead")
        ax.set_xscale("log", base=2); ax.set_xticks(sizes, [str(s) for s in sizes])
        ax.grid(alpha=.3)
        for s, v in zip(sizes, per_image):
            ax.annotate(f"{v:.1f}", (s, v), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=9)
        plt.tight_layout(); plt.show()

        print(f"speed-up from batch 1 -> 32: {per_image[0] / per_image[-1]:.1f}x per image")
        print("\\nThe trade-off: batching adds queuing latency. A real service collects")
        print("requests for a few milliseconds before running them — good for throughput,")
        print("worse for the latency of any single request.")
        """),
        section("6. Monitoring from day one", """
        The `/metrics` endpoint tracks latency percentiles and the distribution of
        predictions. That second one is the cheapest drift signal available: it needs no
        labels, and a sudden shift in it usually precedes a measurable accuracy drop.

        Lecture 16 turns this into proper drift detection.
        """),
        code("""
        m = client.get("/metrics").json()
        print(json.dumps(m, indent=2))

        counts = m["prediction_counts"]
        if counts:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.bar(list(counts), list(counts.values()), color="#c16a1c")
            ax.set_ylabel("predictions served")
            ax.set_title("Prediction distribution — watch this for drift")
            plt.tight_layout(); plt.show()
        """),
        section("7. The test suite", """
        Tests for a model service split into two kinds, and conflating them is a mistake:

        - **Contract tests** — status codes, response schema, error handling. Fast,
          deterministic, run on every commit.
        - **Golden-output tests** — a fixed input must keep producing a fixed prediction.
          This is what catches a preprocessing change that breaks the model without
          breaking any endpoint.
        """),
        code("""
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--no-header", "tests/"],
            cwd=APP, capture_output=True, text=True)
        print(result.stdout[-2500:])
        if result.returncode != 0:
            print("STDERR:", result.stderr[-1000:])
        """),
        code("""
        client.__exit__(None, None, None)     # shut the lifespan down cleanly
        print("service stopped")
        """),
        md("""
        ### Running it for real

        ```bash
        cd lectures/Lecture_09/app
        uvicorn main:app --reload
        ```

        Then:
        - Interactive docs: <http://127.0.0.1:8000/docs>
        - Health: `curl http://127.0.0.1:8000/health`
        - Predict: `curl -F "file=@samples/circle.png" http://127.0.0.1:8000/predict`

        And in a container:

        ```bash
        docker build -t shape-api .
        docker run -p 8000:8000 shape-api
        ```
        """),
        todo("1", "Export a deployable artefact", """
        Save a checkpoint containing the `state_dict`, the class names, the normalisation
        statistics, the input size and a version string. Write `load_model(path)` that
        reconstructs both the model **and** its preprocessing.
        """),
        todo_cell(),
        todo("2", "The predict endpoint", """
        Implement `POST /predict` accepting an image upload and returning JSON with
        predicted class, all class probabilities, model version and inference time in
        milliseconds.
        """),
        todo_cell(),
        todo("3", "Input validation", """
        Return 415 for a non-image content type, 413 for a file over 5 MB, and 422 for a
        corrupt image. Write a test for each case.
        """),
        todo_cell(),
        todo("4", "Batch endpoint", """
        Implement `POST /predict/batch` accepting multiple files and running them as a
        single forward pass. Show it is faster per image than looping over `/predict`.
        """),
        todo_cell(),
        todo("5", "Test suite", """
        Write at least six pytest tests with `TestClient` covering health, a successful
        prediction, each error case, and a golden-output test pinning one fixed image to
        one prediction.
        """),
        todo_cell(),
        todo("6", "Latency benchmark", """
        Measure p50, p95 and p99 over 200 single-image requests. Repeat with the batch
        endpoint at batch size 16 and report per-image latency for both.
        """),
        todo_cell(),
        todo("7", "TorchScript comparison *(stretch)*", """
        Export the model with `torch.jit.script` and serve it. Compare cold-start time and
        p95 latency against the `state_dict` version.
        """),
        todo_cell(),
        todo("8", "Containerise *(stretch)*", """
        Write a Dockerfile using a slim Python base and a multi-stage build. Report the
        final image size and the container cold-start time. A reference Dockerfile is in
        `app/` — try it yourself before reading it.
        """),
        todo_cell(),
    ]
