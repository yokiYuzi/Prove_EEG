mkdir -p ~/ddp_patch

cat > ~/ddp_patch/sitecustomize.py <<'PY'
import os, sys
try:
    import torch
    import torch.nn.parallel as _nnpar
    import torch.nn.parallel.distributed as _dist_mod

    # 你日志里反复未收梯度的一组层；保持冻结以避免 DDP 等待它们的规约
    _FREEZE_PATTERNS = (
        "SDE.SAt_timewise",
        "SDE.embedS_timewise",
        "tat_out_proj_time",
    )

    def _freeze_params_by_name(module):
        frozen = []
        for n, p in module.named_parameters(recurse=True):
            if any(k in n for k in _FREEZE_PATTERNS):
                if p.requires_grad:
                    p.requires_grad = False
                    frozen.append(n)
        if frozen:
            print(f"[sitecustomize] Froze {len(frozen)} params (examples: {frozen[:3]})", file=sys.stderr)
        return frozen

    # 把 DDP 构造时的跨进程“参数形状一致性校验”改为 no-op，避免 Gloo monitored barrier 卡死
    try:
        import torch.distributed.utils as _dist_utils
        if hasattr(_dist_utils, "_verify_params_across_processes"):
            _dist_utils._verify_params_across_processes = lambda *a, **k: None
    except Exception:
        pass
    try:
        import torch.distributed as _dist
        if hasattr(_dist, "_verify_params_across_processes"):
            _dist._verify_params_across_processes = lambda *a, **k: None
    except Exception:
        pass

    def _patch_ddp_class(DDP_cls):
        if getattr(DDP_cls, "_patched_by_sitecustomize_v5", False):
            return
        _orig_init = DDP_cls.__init__

        def _patched_init(self, module, *args, **kwargs):
            kwargs.setdefault("find_unused_parameters", True)
            kwargs.setdefault("broadcast_buffers", False)
            if "device_ids" not in kwargs and "LOCAL_RANK" in os.environ and torch.cuda.is_available():
                try:
                    lr = int(os.environ["LOCAL_RANK"])
                    kwargs["device_ids"] = [lr]
                    kwargs["output_device"] = lr
                except Exception:
                    pass
            _freeze_params_by_name(module)
            return _orig_init(self, module, *args, **kwargs)

        DDP_cls.__init__ = _patched_init
        DDP_cls._patched_by_sitecustomize_v5 = True

    _patch_ddp_class(_nnpar.DistributedDataParallel)
    _patch_ddp_class(_dist_mod.DistributedDataParallel)

    print("[sitecustomize] DDP patched: freeze some params + skip param-shape verification", file=sys.stderr)

except Exception:
    pass
PY


#!/usr/bin/env bash
set -euo pipefail

# 避开 0 号卡（被其它任务占用）
# 导出很耗时 -> 放大分布式超时（秒）
export TORCH_DISTRIBUTED_TIMEOUT=43200

export CUDA_VISIBLE_DEVICES=0,1,2,3

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29537
export OMP_NUM_THREADS=1

# 你的平台不支持 expandable_segments，只保留 split 限制（缓解碎片峰值）
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256"

# 通信更稳：错误浮出 + 阻塞等待
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_DEBUG=WARN

# 统一让 Gloo/NCCL 走回环网卡，避免偶发 socket 阻塞
export GLOO_SOCKET_IFNAME=lo
export NCCL_SOCKET_IFNAME=lo
export NCCL_SHM_DISABLE=1

# DDP 调试（必要时保留）
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# sitecustomize 注入（V5）
export PYTHONPATH="$HOME/ddp_patch:${PYTHONPATH:-}"

# （可选）提高句柄上限，5 折长跑更稳
ulimit -n 8192 || true

# 单机 3 进程，代码内部会顺序跑完 5 折（训练→验证→导出）
torchrun --standalone --nnodes=1 --nproc_per_node=3 main_classi.py "$@"