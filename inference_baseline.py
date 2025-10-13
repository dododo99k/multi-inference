import time,copy,os,pickle, argparse,json, subprocess
import numpy as np
import torch
from torch.multiprocessing import Manager, Pool,current_process
from torchvision import models


# main process device
device = torch.device('cpu')  # defer CUDA to child processes (MPS-friendly)

def duplicate_model_dataset(model, dataset, num_proc):
    models, datasets = [], []
    for i in range(num_proc):
        mdl = copy.deepcopy(model)
        mdl.eval()
        mdl.train_mode = False
        models.append(mdl)

        # keep tensors on CPU; child process will move to its own CUDA context
        data = copy.deepcopy(dataset)
        datasets.append(data)

    return models, datasets

def parallel_worker(model, data, results_list, idx_list, start_barrier, warmup=0,
                    enable_compile=False, compile_mode='max-autotune'):
    # Child process: set device mapping for MPS. Expect CUDA_VISIBLE_DEVICES to be set externally (e.g., to '1').
    os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
    # Robust device selection; fallback to CPU if CUDA is unavailable/problematic
    try:
        torch.cuda.set_device(0)
        dev = torch.device('cuda:0')
    except Exception:
        dev = torch.device('cpu')
    print(f'{current_process().name} using device {dev}')
    
    model = model.to(dev).eval()

    # Optionally compile the model for acceleration (PyTorch 2+)
    if enable_compile and hasattr(torch, 'compile'):
        try:
            kwargs = {}
            if compile_mode:
                kwargs['mode'] = compile_mode
            model = torch.compile(model, **kwargs)
            print(f"{current_process().name} compiled model with torch.compile (mode={compile_mode})")
        except Exception as e:
            print(f"{current_process().name} torch.compile failed ({e}); falling back to eager mode")
    data = data.to(dev, non_blocking=True) if dev.type == 'cuda' else data

    if warmup > 0:
        with torch.no_grad():
            for _ in range(warmup):
                _out = model(data)
                if dev.type == 'cuda':
                    torch.cuda.synchronize()
                _ = _out.detach().cpu()

    # Mark readiness and wait for global start event
    print(f"{current_process().name} ready; waiting for start_barrier ...")

    try:
        start_barrier.wait()
    except Exception as e:
        print(f"Error occurred while waiting for barrier: {e}") 
        
    worker_start_time = time.perf_counter()
    for idx in idx_list:
        with torch.no_grad():
            start_time = time.perf_counter() - worker_start_time
            outputs = model(data)
            if dev.type == 'cuda':
                torch.cuda.synchronize()
            outputs.detach().cpu()
            end_time = time.perf_counter() - worker_start_time
            temp_records = {
                'task_id': int(idx),
                'worker': current_process().name,
                'start': float(start_time),
                'end': float(end_time),
                'duration': float(end_time - start_time),
            }
            results_list.append(temp_records)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-s','--seed',
                           type=int,
                           default=42,
                           help='random seed')
    
    argparser.add_argument('-l','--length',
                           type = int,
                           default=3600,
                           help='num of iterations')
    
    argparser.add_argument('-m','--model',
                           type = str,
                           default='resnet50',
                           help='[resnet50, resnet101, resnet152, vgg11, vgg16, vgg19]')

    argparser.add_argument('-i','--save',
                           type = bool,
                           default=True,
                           help='if save results')
    
    argparser.add_argument('-p','--parallel',
                           type = int,
                           default=1,
                           help='parallel model inference number')
    
    argparser.add_argument('-b','--batch',
                           type = int,
                           default=1,
                           help='batch size for each inference subprocess')
    argparser.add_argument('-w','--warmup',
                           type = int,
                           default=100,
                           help='warmup iterations per worker before timing loop')
    argparser.add_argument('-nc','--no-compile',
                           action='store_true',
                           help='disable torch.compile acceleration (PyTorch 2+)')
    argparser.add_argument('--compile-mode',
                           type=str,
                           default='max-autotune',
                           help="torch.compile mode: 'default', 'reduce-overhead', or 'max-autotune'")
    # argparser.add_argument('--compile-backend',
    #                        type=str,
    #                        default=None,
    #                        help="torch.compile backend (e.g., 'inductor'); None uses default")
    
    args = argparser.parse_args()

    # MPS-friendly setup (parent avoids touching CUDA; children will pick cuda:0 via remapping)
    os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')  # map physical GPU 0 to logical cuda:0 in children
    # os.environ.setdefault('CUDA_MPS_PIPE_DIRECTORY', '/tmp/nvidia-mps')
    # os.environ.setdefault('CUDA_MPS_LOG_DIRECTORY', '/tmp/nvidia-mps')
    # os.makedirs('/tmp/nvidia-mps', exist_ok=True)
    # try:
    #     subprocess.run(['nvidia-cuda-mps-control', '-d'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    # except Exception:
    #     pass

    #### model ##################
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.multiprocessing.set_start_method('spawn')


    if args.model == 'resnet50':
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
    elif args.model == 'resnet101':
        weights = models.ResNet101_Weights.DEFAULT
        model = models.resnet101(weights=weights)
    elif args.model == 'resnet152':
        weights = models.ResNet152_Weights.DEFAULT
        model = models.resnet152(weights=weights)
    elif args.model == 'vgg11':
        weights = models.VGG11_Weights.DEFAULT
        model = models.vgg11(weights=weights)
    elif args.model == 'vgg16':
        weights = models.VGG16_Weights.DEFAULT
        model = models.vgg16(weights=weights)
    elif args.model == 'vgg19':
        weights = models.VGG19_Weights.DEFAULT
        model = models.vgg19(weights=weights)
    else:
        raise ValueError('no implemented models')
    
    #### test data ##############
    data = torch.randn(args.batch, 3, 224, 224)  # keep on CPU; child will move to its own device
    
    models, datasets = duplicate_model_dataset(model, data, args.parallel)

    # model
    total_time = 0
    time_record = []

    # shared records across processes (per-worker)
    manager = Manager()
    pool = Pool(processes=args.parallel)
    # Use manager-backed proxies so objects are pickleable under spawn

    start_barrier = manager.Barrier(args.parallel+1)

    worker_records = [manager.list() for _ in range(args.parallel)]

    # partition indices across the number of parallel workers dynamically
    total_inference_index = list(range(args.length))
    group_inference_index = [total_inference_index[i::args.parallel] for i in range(args.parallel)]

    # start worker processes
    # Use explicit spawn context for both Pool and Barrier to avoid cross-context sync issues
    # ctx =  torch.multiprocessing.get_context('spawn')
    # pool = ctx.Pool(processes=args.parallel)
    # barrier deprecated: using Event + counter instead

    async_results = []
    print(f'[DEBUG]: {args.no_compile}, {args.compile_mode}')
    for i in range(args.parallel):
        ar = pool.apply_async(
            parallel_worker,
            args=(models[i], datasets[i], worker_records[i], group_inference_index[i], start_barrier, args.warmup,
                  not args.no_compile, args.compile_mode),
        )
        async_results.append(ar)


    # Wait until all workers report readiness, then release together
    print('Waiting for workers to be ready ...')
    try:
        start_barrier.wait(timeout=600)  # 1 min timeout
        print('All workers ready; starting inference ...')
    except Exception as e:
        print(f'Error or timeout while waiting for workers to be ready: {e}')

    # Close and join the pool to wait for workers to finish
    pool.close() 
    for i, ar in enumerate(async_results):
            try:
                ar.get()
            except Exception as e:
                print(f"Worker {i} raised: {e}")
    pool.join() 

    # Build time_record from merged per-worker task records
    try:
        merged = []
        for wr in worker_records:
            merged.extend(list(wr))
        ordered = sorted(merged, key=lambda r: r['task_id'])
        time_record = [float(rec['duration']) for rec in ordered]
    except Exception:
        time_record = []

    # Optional warmup trimming
    if len(time_record) > 100:
        time_record = time_record[100:]

    # Compute statistics in milliseconds
    if len(time_record) == 0:
        mean_ms = 0.0
        std_ms = 0.0
        print('No inference records collected. Check multiprocessing setup or GPU availability.')
    else:
        mean_ms = float(np.mean(time_record) * 1000.0)
        std_ms = float(np.std(time_record) * 1000.0)
        print(f'average time: {mean_ms:.3f} ms, time std: {std_ms:.3f} ms')

    save_name = f"{args.model}_parallel_{args.parallel}_batch_{args.batch}"

    # Persist results as JSON (metadata + summary + series + per-task records)
    if args.save:
        os.makedirs('./profile_results/results', exist_ok=True)
        os.makedirs('./profile_results/figures', exist_ok=True)
        result_json = {
            'meta': {
                'model': args.model,
                'length': int(args.length),
                'parallel': int(args.parallel),
                'batch': int(args.batch),
                'seed': int(args.seed),
                'device': str(device),
                'backend': 'torch',
                'compile': bool(not args.no_compile),
                'compile_mode': args.compile_mode,
                # 'compile_backend': args.compile_backend,
            },
            'summary': {
                'count': int(len(time_record)),
                'mean_ms': mean_ms,
                'std_ms': std_ms,
                'start_time': float(ordered[0]['start'] if ordered else 0.0),
                'end_time': float(ordered[-1]['end'] if ordered else 0.0),
            },
            'time_record_s': time_record,  # durations in seconds
            'tasks': ordered if 'ordered' in locals() else [],
            'per_worker': [list(wr) for wr in worker_records],
        }

        with open('./profile_results/results/' + save_name + '.json', 'w') as f:
            json.dump(result_json, f, ensure_ascii=False, indent=2)

        # Optional plot (lazy import to avoid numpy/matplotlib ABI issues)
        if len(time_record) > 0:
            try:
                import matplotlib
                matplotlib.use('Agg', force=True)  # headless backend
                import matplotlib.pyplot as plt

                y = [float(x) for x in time_record]
                fig, ax = plt.subplots(figsize=(8, 4), facecolor='white')
                ax.plot(y, label='Inference Times (s)')
                ax.legend()
                fig.tight_layout()
                fig.savefig('./profile_results/figures/' + save_name + '.png')
                plt.close(fig)
            except Exception as e:
                print(f'Plotting failed: {e}')
    else:
        pass

    ######### see plot_parallel.py for result plotting ##############
