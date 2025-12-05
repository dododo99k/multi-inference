import argparse
import json
import os
import queue
import time
from typing import List

import numpy as np
import torch
from torch.multiprocessing import Event, Process, Queue, Value, set_start_method, current_process

from functions import Task
from ee_models import *


def load_resnet50ee():
    """Load the EE model from disk."""
    model = torch.load("./weights/resnet50_EE.pth", map_location="cpu", weights_only=False)
    model.eval()
    return model


def batch_worker(model, execute_queue: Queue, permit: Event, done: Event, stop: Event, data_number: Value, ee_head: Value,
                 max_batch: int):
    """
    Worker used for profiling that mirrors single_model_serve synchronization:
    waits on permit -> runs a batch -> stamps tasks -> signals done.
    """
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    try:
        torch.cuda.set_device(0)
        dev = torch.device("cuda:0")
    except Exception:
        dev = torch.device("cpu")
    print(f"{current_process().name} using device {dev}")

    model = model.to(dev).eval()
    random_data = torch.randn(max_batch, 3, 224, 224).to(dev)

    # warm up to stabilize timing
    warmup_max = min(max_batch, 30)
    for datasize in range(1, warmup_max + 1):
        data = random_data[:datasize]
        for _ in range(3):
            for ee in range(0, 14):
                with torch.no_grad():
                    _ = model(data, ee)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    print(f"{current_process().name} warmed up.")

    while True:
        permit.wait()  # block until main process allows execution
        if stop.is_set():
            break

        if dev.type == "cuda":
            torch.cuda.synchronize()
        if execute_queue.qsize() != data_number.value:
            print(f"[WARN] {current_process().name} queue size {execute_queue.qsize()} "
                  f"differs from data_number {data_number.value}")
        data = random_data[: data_number.value]

        start_ns = time.time_ns()
        with torch.no_grad():
            _ = model(data, int(ee_head.value))
        if dev.type == "cuda":
            torch.cuda.synchronize()
        finish_ns = time.time_ns()

        stamped_tasks: List[Task] = []
        while True:
            try:
                task = execute_queue.get_nowait()
            except queue.Empty:
                break
            task.start_time = (start_ns - task.base_time) / 1_000_000
            task.finish_time = (finish_ns - task.base_time) / 1_000_000
            stamped_tasks.append(task)
        for task in stamped_tasks:
            execute_queue.put(task)

        permit.clear()
        done.set()


def drain_results(execute_queue: Queue) -> List[Task]:
    tasks = []
    while True:
        try:
            tasks.append(execute_queue.get_nowait())
        except queue.Empty:
            break
    return tasks


def profile(batch_size: int, ee_head: int, iterations: int, warmup: int) -> dict:
    model = load_resnet50ee()

    permit = Event()
    done = Event()
    stop = Event()
    execute_queue: Queue = Queue()
    data_number = Value("i", 0)
    ee_head_value = Value("i", ee_head)

    max_batch = max(batch_size, 32)
    worker = Process(
        target=batch_worker,
        args=(model, execute_queue, permit, done, stop, data_number, ee_head_value, max_batch),
        name="ProfileWorker",
    )
    worker.start()

    durations_ms: List[float] = []

    # warmup iterations (ignored in stats)
    for _ in range(warmup):
        base_time = time.time_ns()
        for _ in range(batch_size):
            task = Task(model_id=0, tk_id=-1)
            task.base_time = base_time
            execute_queue.put(task)
        data_number.value = batch_size
        permit.set()
        done.wait()
        drain_results(execute_queue)
        done.clear()

    # measured iterations
    for step in range(iterations):
        base_time = time.time_ns()
        for i in range(batch_size):
            task = Task(model_id=0, tk_id=step * batch_size + i)
            task.base_time = base_time
            execute_queue.put(task)
        data_number.value = batch_size
        permit.set()
        done.wait()
        finished_tasks = drain_results(execute_queue)
        for task in finished_tasks:
            durations_ms.append(task.finish_time - task.start_time)
        done.clear()

    stop.set()
    permit.set()  # wake worker if waiting
    worker.join()

    durations_ms = [float(x) for x in durations_ms]
    mean_ms = float(np.mean(durations_ms)) if durations_ms else 0.0
    std_ms = float(np.std(durations_ms)) if durations_ms else 0.0

    return {
        "meta": {
            "batch": batch_size,
            "ee_head": ee_head,
            "iterations": iterations,
            "warmup": warmup,
            "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        },
        "summary": {
            "count": len(durations_ms),
            "mean_ms": mean_ms,
            "std_ms": std_ms,
        },
        "time_record_ms": durations_ms,
    }


def main():
    parser = argparse.ArgumentParser(description="Profile EE model with single_model_serve style queue/signals.")
    parser.add_argument("-b", "--batch", type=int, required=True, help="Batch size to profile.")
    parser.add_argument("-e", "--ee-head", type=int, required=True, help="EE head to profile (0-13).")
    parser.add_argument("-n", "--iterations", type=int, default=100, help="Measured iterations.")
    parser.add_argument("-w", "--warmup", type=int, default=10, help="Warmup iterations (not recorded).")
    parser.add_argument("-s", "--save", action="store_true", help="Save results under profile_results/results.")
    args = parser.parse_args()

    set_start_method("spawn", force=True)

    results = profile(args.batch, args.ee_head, args.iterations, args.warmup)
    print(f"batch={args.batch}, ee_head={args.ee_head}, mean={results['summary']['mean_ms']:.3f} ms, "
          f"std={results['summary']['std_ms']:.3f} ms over {results['summary']['count']} runs")

    if args.save:
        os.makedirs("./profile_results/results", exist_ok=True)
        save_name = f"serve_resnet50ee_batch_{args.batch}_eehead_{args.ee_head}.json"
        with open(os.path.join("./profile_results/results", save_name), "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"saved to ./profile_results/results/{save_name}")


if __name__ == "__main__":
    main()
