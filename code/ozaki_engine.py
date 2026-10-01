"""CRT (Ozaki-II) INT8 emulation of FP64 matrix multiplication with inside scaling, an INT8 throughput
diagnosis, and an accuracy-matched head-to-head against the cuBLAS 13.1 FP64 emulation.
Method and measurements: docs/engine.md.  Data: results/engine/.

Usage (run from the repository root):
  python code/ozaki_engine.py diag                      # INT8/FP16/FP8 throughput diagnosis -> int8_diag.json
  python code/ozaki_engine.py selftest                  # integer-exactness of the CRT core (Python big integers)
  python code/ozaki_engine.py check n [fam,..]          # accuracy vs the correctly rounded sample, N sweep, bound audit
  python code/ozaki_engine.py race n1,n2 fam1,fam2      # full race cells -> race_results.jsonl
"""
import ctypes
import hashlib
import json
import math
import os
import subprocess
import sys
import time

# ---- cuda / cuBLASLt ctypes wrappers (boilerplate) ----
import os
import math
import ctypes
from ctypes import POINTER, c_int, c_uint, c_size_t, c_char_p, c_void_p, c_uint64, c_int64, c_float, c_int32

CUDA13 = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64"
if hasattr(os, "add_dll_directory") and os.path.exists(CUDA13):
    os.add_dll_directory(CUDA13)


def ptr(t):
    return c_void_p(t.data_ptr())


class Driver:
    def __init__(self):
        import sys
        try:
            if sys.platform == "win32":
                self.lib = ctypes.WinDLL("nvcuda.dll")
            else:
                self.lib = ctypes.CDLL("libcuda.so.1")
        except OSError as e:
            raise RuntimeError("CUDA driver library not found. GPU subcommands require an NVIDIA driver.") from e

        self.cuInit = self.lib.cuInit
        self.cuInit.argtypes = [c_uint]
        self.cuInit.restype = c_int

        self.cuCtxGetCurrent = self.lib.cuCtxGetCurrent
        self.cuCtxGetCurrent.argtypes = [POINTER(c_void_p)]
        self.cuCtxGetCurrent.restype = c_int

        self.cuModuleLoadData = self.lib.cuModuleLoadData
        self.cuModuleLoadData.argtypes = [POINTER(c_void_p), c_void_p]
        self.cuModuleLoadData.restype = c_int

        self.cuModuleGetFunction = self.lib.cuModuleGetFunction
        self.cuModuleGetFunction.argtypes = [POINTER(c_void_p), c_void_p, c_char_p]
        self.cuModuleGetFunction.restype = c_int

        self.cuModuleGetGlobal_v2 = self.lib.cuModuleGetGlobal_v2
        self.cuModuleGetGlobal_v2.argtypes = [POINTER(c_uint64), POINTER(c_size_t), c_void_p, c_char_p]
        self.cuModuleGetGlobal_v2.restype = c_int

        self.cuMemcpyHtoD_v2 = self.lib.cuMemcpyHtoD_v2
        self.cuMemcpyHtoD_v2.argtypes = [c_uint64, c_void_p, c_size_t]
        self.cuMemcpyHtoD_v2.restype = c_int

        self.cuLaunchKernel = self.lib.cuLaunchKernel
        self.cuLaunchKernel.argtypes = [
            c_void_p,
            c_uint, c_uint, c_uint,
            c_uint, c_uint, c_uint,
            c_uint, c_void_p,
            POINTER(c_void_p), POINTER(c_void_p)
        ]
        self.cuLaunchKernel.restype = c_int

        self.cuFuncSetAttribute = self.lib.cuFuncSetAttribute
        self.cuFuncSetAttribute.argtypes = [c_void_p, c_int, c_int]
        self.cuFuncSetAttribute.restype = c_int

        self.cuGetErrorString = self.lib.cuGetErrorString
        self.cuGetErrorString.argtypes = [c_int, POINTER(c_char_p)]
        self.cuGetErrorString.restype = c_int

        self.cuStreamSynchronize = self.lib.cuStreamSynchronize
        self.cuStreamSynchronize.argtypes = [c_void_p]
        self.cuStreamSynchronize.restype = c_int

        self.check(self.cuInit(0), "cuInit")

    def check(self, status, what=""):
        if status != 0:
            err_str = c_char_p()
            self.cuGetErrorString(status, ctypes.byref(err_str))
            msg = err_str.value.decode("utf-8", errors="replace") if err_str.value else "unknown"
            raise RuntimeError(f"{what}: CUDA driver error {status}: {msg}")


_DRV = None
def get_drv():
    global _DRV
    if _DRV is None:
        _DRV = Driver()
    return _DRV



class Nvrtc:
    def __init__(self):
        dll_path = os.path.join(CUDA13, "nvrtc64_130_0.dll")
        self.lib = ctypes.WinDLL(dll_path)

        self.nvrtcCreateProgram = self.lib.nvrtcCreateProgram
        self.nvrtcCreateProgram.argtypes = [POINTER(c_void_p), c_char_p, c_char_p, c_int, POINTER(c_char_p), POINTER(c_char_p)]
        self.nvrtcCreateProgram.restype = c_int

        self.nvrtcCompileProgram = self.lib.nvrtcCompileProgram
        self.nvrtcCompileProgram.argtypes = [c_void_p, c_int, POINTER(c_char_p)]
        self.nvrtcCompileProgram.restype = c_int

        self.nvrtcGetProgramLogSize = self.lib.nvrtcGetProgramLogSize
        self.nvrtcGetProgramLogSize.argtypes = [c_void_p, POINTER(c_size_t)]
        self.nvrtcGetProgramLogSize.restype = c_int

        self.nvrtcGetProgramLog = self.lib.nvrtcGetProgramLog
        self.nvrtcGetProgramLog.argtypes = [c_void_p, c_char_p]
        self.nvrtcGetProgramLog.restype = c_int

        self.nvrtcGetCUBINSize = self.lib.nvrtcGetCUBINSize
        self.nvrtcGetCUBINSize.argtypes = [c_void_p, POINTER(c_size_t)]
        self.nvrtcGetCUBINSize.restype = c_int

        self.nvrtcGetCUBIN = self.lib.nvrtcGetCUBIN
        self.nvrtcGetCUBIN.argtypes = [c_void_p, c_char_p]
        self.nvrtcGetCUBIN.restype = c_int

        self.nvrtcDestroyProgram = self.lib.nvrtcDestroyProgram
        self.nvrtcDestroyProgram.argtypes = [POINTER(c_void_p)]
        self.nvrtcDestroyProgram.restype = c_int

    def compile(self, src: str, name: str = "k.cu", arch: str = "sm_120", extra_opts=()) -> bytes:
        cuda_parent = os.path.dirname(os.path.dirname(CUDA13))
        inc_dir = os.path.join(cuda_parent, "include")
        opts = [f"--gpu-architecture={arch}", "-std=c++17", f"--include-path={inc_dir}"] + list(extra_opts)
        opt_bytes = [opt.encode("utf-8") for opt in opts]
        opt_array = (c_char_p * len(opt_bytes))(*opt_bytes)

        prog = c_void_p()
        res = self.nvrtcCreateProgram(ctypes.byref(prog), src.encode("utf-8"), name.encode("utf-8"), 0, None, None)
        if res != 0:
            raise RuntimeError(f"nvrtcCreateProgram failed: {res}")

        try:
            status = self.nvrtcCompileProgram(prog, len(opts), opt_array)
            if status != 0:
                log_size = c_size_t()
                self.nvrtcGetProgramLogSize(prog, ctypes.byref(log_size))
                log_buf = ctypes.create_string_buffer(log_size.value)
                self.nvrtcGetProgramLog(prog, log_buf)
                raise RuntimeError(f"NVRTC compilation failed:\n{log_buf.value.decode('utf-8', errors='replace')}")

            cubin_size = c_size_t()
            res = self.nvrtcGetCUBINSize(prog, ctypes.byref(cubin_size))
            if res != 0:
                raise RuntimeError(f"nvrtcGetCUBINSize failed: {res}")
            cubin_buf = ctypes.create_string_buffer(cubin_size.value)
            res = self.nvrtcGetCUBIN(prog, cubin_buf)
            if res != 0:
                raise RuntimeError(f"nvrtcGetCUBIN failed: {res}")
            return cubin_buf.raw[:cubin_size.value]
        finally:
            self.nvrtcDestroyProgram(ctypes.byref(prog))


class Module:
    def __init__(self, cubin: bytes):
        import torch
        torch.zeros(1, device="cuda")
        ctx = c_void_p()
        get_drv().check(get_drv().cuCtxGetCurrent(ctypes.byref(ctx)), "cuCtxGetCurrent")
        assert ctx.value is not None, "CUDA context is null"
        self._cubin_buf = ctypes.create_string_buffer(cubin)
        self.mod = c_void_p()
        get_drv().check(get_drv().cuModuleLoadData(ctypes.byref(self.mod), self._cubin_buf), "cuModuleLoadData")

    def get(self, name: str) -> "Kernel":
        func = c_void_p()
        get_drv().check(get_drv().cuModuleGetFunction(ctypes.byref(func), self.mod, name.encode("utf-8")), "cuModuleGetFunction")
        return Kernel(func, name)

    def set_global(self, name: str, data: bytes):
        dptr = c_uint64()
        nbytes = c_size_t()
        get_drv().check(get_drv().cuModuleGetGlobal_v2(ctypes.byref(dptr), ctypes.byref(nbytes), self.mod, name.encode("utf-8")), "cuModuleGetGlobal_v2")
        if len(data) > nbytes.value:
            raise ValueError(f"Data length {len(data)} exceeds symbol {name} size {nbytes.value}")
        buf = ctypes.create_string_buffer(data)
        get_drv().check(get_drv().cuMemcpyHtoD_v2(dptr, buf, len(data)), "cuMemcpyHtoD_v2")


class Kernel:
    def __init__(self, func, name: str):
        self.func = func
        self.name = name

    def launch(self, grid, block, args, smem=0, stream=0):
        def _pad3(t):
            if isinstance(t, int):
                return (t, 1, 1)
            t = tuple(t)
            return (t + (1, 1, 1))[:3]

        gx, gy, gz = _pad3(grid)
        bx, by, bz = _pad3(block)

        if smem > 48 * 1024:
            get_drv().check(get_drv().cuFuncSetAttribute(self.func, 8, smem), "cuFuncSetAttribute")

        c_args = (c_void_p * len(args))(*[ctypes.cast(ctypes.byref(a), c_void_p) for a in args])
        st = c_void_p(stream if isinstance(stream, int) else (stream.value if isinstance(stream, c_void_p) else stream.cuda_stream))
        get_drv().check(get_drv().cuLaunchKernel(
            self.func,
            c_uint(gx), c_uint(gy), c_uint(gz),
            c_uint(bx), c_uint(by), c_uint(bz),
            c_uint(smem), st, c_args, None
        ), "cuLaunchKernel")


CUDA_R_8I = 3
CUDA_R_32I = 10
CUBLAS_COMPUTE_32I = 72
CUBLAS_OP_N = 0
CUBLAS_OP_T = 1
CUBLASLT_MATMUL_DESC_TRANSA = 3
CUBLASLT_MATMUL_DESC_TRANSB = 4
CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES = 1


class Algo(ctypes.Structure):
    _fields_ = [("data", c_uint64 * 8)]


class HeurResult(ctypes.Structure):
    _fields_ = [
        ("algo", Algo),
        ("workspaceSize", c_size_t),
        ("state", c_int),
        ("wavesCount", c_float),
        ("reserved", c_int * 4),
    ]


class LtInt8:
    def __init__(self, workspace_bytes=64 << 20):
        dll_path = os.path.join(CUDA13, "cublasLt64_13.dll")
        self.lib = ctypes.WinDLL(dll_path)

        self.cublasLtCreate = self.lib.cublasLtCreate
        self.cublasLtCreate.argtypes = [POINTER(c_void_p)]
        self.cublasLtCreate.restype = c_int

        self.cublasLtDestroy = self.lib.cublasLtDestroy
        self.cublasLtDestroy.argtypes = [c_void_p]
        self.cublasLtDestroy.restype = c_int

        self.cublasLtMatmulDescCreate = self.lib.cublasLtMatmulDescCreate
        self.cublasLtMatmulDescCreate.argtypes = [POINTER(c_void_p), c_int, c_int]
        self.cublasLtMatmulDescCreate.restype = c_int

        self.cublasLtMatmulDescSetAttribute = self.lib.cublasLtMatmulDescSetAttribute
        self.cublasLtMatmulDescSetAttribute.argtypes = [c_void_p, c_int, c_void_p, c_size_t]
        self.cublasLtMatmulDescSetAttribute.restype = c_int

        self.cublasLtMatrixLayoutCreate = self.lib.cublasLtMatrixLayoutCreate
        self.cublasLtMatrixLayoutCreate.argtypes = [POINTER(c_void_p), c_int, c_uint64, c_uint64, c_int64]
        self.cublasLtMatrixLayoutCreate.restype = c_int

        self.cublasLtMatmulPreferenceCreate = self.lib.cublasLtMatmulPreferenceCreate
        self.cublasLtMatmulPreferenceCreate.argtypes = [POINTER(c_void_p)]
        self.cublasLtMatmulPreferenceCreate.restype = c_int

        self.cublasLtMatmulPreferenceSetAttribute = self.lib.cublasLtMatmulPreferenceSetAttribute
        self.cublasLtMatmulPreferenceSetAttribute.argtypes = [c_void_p, c_int, c_void_p, c_size_t]
        self.cublasLtMatmulPreferenceSetAttribute.restype = c_int

        self.cublasLtMatmulAlgoGetHeuristic = self.lib.cublasLtMatmulAlgoGetHeuristic
        self.cublasLtMatmulAlgoGetHeuristic.argtypes = [
            c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
            c_void_p, c_int, POINTER(HeurResult), POINTER(c_int)
        ]
        self.cublasLtMatmulAlgoGetHeuristic.restype = c_int

        self.cublasLtMatmul = self.lib.cublasLtMatmul
        self.cublasLtMatmul.argtypes = [
            c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
            c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
            POINTER(Algo), c_void_p, c_size_t, c_void_p
        ]
        self.cublasLtMatmul.restype = c_int

        self.handle = c_void_p()
        res = self.cublasLtCreate(ctypes.byref(self.handle))
        if res != 0:
            raise RuntimeError(f"cublasLtCreate failed: {res}")

        import torch
        self.workspace_size = workspace_bytes
        self.ws = torch.empty(workspace_bytes, dtype=torch.uint8, device="cuda")
        self.plans = {}
        self.best = {}

    def _check(self, status, what=""):
        if status != 0:
            raise RuntimeError(f"{what}: cuBLASLt error {status}")

    def plan(self, m, n, k, top=16):
        desc = c_void_p()
        self._check(self.cublasLtMatmulDescCreate(ctypes.byref(desc), CUBLAS_COMPUTE_32I, CUDA_R_32I), "desc create")
        op_t = c_int(CUBLAS_OP_T)
        op_n = c_int(CUBLAS_OP_N)
        self._check(self.cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, ctypes.byref(op_t), ctypes.sizeof(op_t)), "desc TRANSA")
        self._check(self.cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, ctypes.byref(op_n), ctypes.sizeof(op_n)), "desc TRANSB")

        Adesc = c_void_p()
        self._check(self.cublasLtMatrixLayoutCreate(ctypes.byref(Adesc), CUDA_R_8I, c_uint64(k), c_uint64(n), c_int64(k)), "layout Adesc")
        Bdesc = c_void_p()
        self._check(self.cublasLtMatrixLayoutCreate(ctypes.byref(Bdesc), CUDA_R_8I, c_uint64(k), c_uint64(m), c_int64(k)), "layout Bdesc")
        Cdesc = c_void_p()
        self._check(self.cublasLtMatrixLayoutCreate(ctypes.byref(Cdesc), CUDA_R_32I, c_uint64(n), c_uint64(m), c_int64(n)), "layout Cdesc")
        Ddesc = Cdesc

        pref = c_void_p()
        self._check(self.cublasLtMatmulPreferenceCreate(ctypes.byref(pref)), "pref create")
        ws_bytes = c_size_t(self.workspace_size)
        self._check(self.cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, ctypes.byref(ws_bytes), ctypes.sizeof(ws_bytes)), "pref max_ws")

        heur_arr = (HeurResult * top)()
        ret_count = c_int(0)
        self._check(self.cublasLtMatmulAlgoGetHeuristic(self.handle, desc, Adesc, Bdesc, Cdesc, Ddesc, pref, top, heur_arr, ctypes.byref(ret_count)), "algo heuristic")

        valid_heur = [heur_arr[i] for i in range(ret_count.value) if heur_arr[i].state == 0]
        entry = {
            "desc": desc,
            "Adesc": Adesc,
            "Bdesc": Bdesc,
            "Cdesc": Cdesc,
            "Ddesc": Ddesc,
            "pref": pref,
            "heuristics": valid_heur,
        }
        self.plans[(m, n, k)] = entry
        if (m, n, k) not in self.best:
            self.best[(m, n, k)] = 0
        return entry

    def _to_void_p(self, p):
        if isinstance(p, c_void_p):
            return p
        if isinstance(p, int):
            return c_void_p(p)
        if hasattr(p, "data_ptr"):
            return c_void_p(p.data_ptr())
        return c_void_p(int(p))

    def gemm(self, A_ptr, Bt_ptr, C_ptr, m, n, k, algo_index=0, stream=0):
        plan = self.plans.get((m, n, k))
        if plan is None:
            plan = self.plan(m, n, k)
        algo_ptr = ctypes.byref(plan["heuristics"][algo_index].algo) if plan["heuristics"] and algo_index < len(plan["heuristics"]) else None
        alpha = c_int32(1)
        beta = c_int32(0)
        ws_ptr = c_void_p(self.ws.data_ptr())
        st = self._to_void_p(stream)
        return self.cublasLtMatmul(
            self.handle,
            plan["desc"],
            ctypes.byref(alpha),
            self._to_void_p(Bt_ptr),
            plan["Adesc"],
            self._to_void_p(A_ptr),
            plan["Bdesc"],
            ctypes.byref(beta),
            self._to_void_p(C_ptr),
            plan["Cdesc"],
            self._to_void_p(C_ptr),
            plan["Ddesc"],
            algo_ptr,
            ws_ptr,
            c_size_t(self.workspace_size),
            st
        )

    def autotune(self, A, Bt, C, m, n, k, reps=5):
        import torch
        if (m, n, k) not in self.plans:
            self.plan(m, n, k)
        plan = self.plans[(m, n, k)]
        A_p = ptr(A)
        Bt_p = ptr(Bt)
        C_p = ptr(C)
        timings = []
        for idx in range(len(plan["heuristics"])):
            status = self.gemm(A_p, Bt_p, C_p, m, n, k, algo_index=idx)
            if status != 0:
                continue
            torch.cuda.synchronize()
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            start_ev.record()
            failed = False
            for _ in range(reps):
                if self.gemm(A_p, Bt_p, C_p, m, n, k, algo_index=idx) != 0:
                    failed = True
                    break
            end_ev.record()
            torch.cuda.synchronize()
            if failed:
                continue
            t = (start_ev.elapsed_time(end_ev) / 1000.0) / reps
            timings.append((idx, t))
        timings.sort(key=lambda x: x[1])
        if timings:
            self.best[(m, n, k)] = timings[0][0]
        return timings

    def gemm_best(self, A_ptr, Bt_ptr, C_ptr, m, n, k, stream=0):
        algo_idx = self.best.get((m, n, k), 0)
        return self.gemm(A_ptr, Bt_ptr, C_ptr, m, n, k, algo_index=algo_idx, stream=stream)


# =====================================================================================================================
# harness pieces: input families and the exact sampled reference
# =====================================================================================================================
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import torch

DEV = "cuda"


def make_inputs(n, fam, seed=0):
    g = torch.Generator(device=DEV)
    g.manual_seed(1000 * seed + n)

    def rn():
        return torch.randn(n, n, device=DEV, dtype=torch.float64, generator=g)

    def ru():
        return torch.rand(n, n, device=DEV, dtype=torch.float64, generator=g)
    if fam == "G":
        A, B = rn(), rn()
    elif fam == "X":
        A = (ru() - 0.5) * torch.exp(2.0 * rn())
        B = (ru() - 0.5) * torch.exp(2.0 * rn())
    elif fam == "S":
        A, B = rn(), rn()
        u = torch.randint(-16, 17, (n,), device=DEV, generator=g).to(torch.float64)
        A = A * torch.exp2(u)[None, :]
        B = B * torch.exp2(-u)[:, None]  # keeps C on the scale of G; contraction-index similarity
    elif fam == "R":
        A, B = rn(), rn()
        s = torch.randint(-20, 21, (n,), device=DEV, generator=g).to(torch.float64)
        t = torch.randint(-20, 21, (n,), device=DEV, generator=g).to(torch.float64)
        A = A * torch.exp2(s)[:, None]
        B = B * torch.exp2(t)[None, :]
    else:
        raise ValueError(fam)
    return A.contiguous(), B.contiguous()


# ------------------------------------------------------------------ exact sampled reference
def _split(a):
    t = 134217729.0 * a
    hi = t - (t - a)
    return hi, a - hi


def exact_sample(Ar, Bc):
    """Correctly rounded (Ar @ Bc) via Dekker TwoProduct + math.fsum.  Ar r x n, Bc n x c (float64 numpy)."""
    r, n = Ar.shape
    c = Bc.shape[1]
    out = np.empty((r, c))
    bh, bl = _split(Bc)
    for i in range(r):
        a = Ar[i][:, None]
        ah, al = _split(a)
        p = a * Bc
        e = ((ah * bh - p) + ah * bl + al * bh) + al * bl
        for j in range(c):
            out[i, j] = math.fsum(p[:, j].tolist() + e[:, j].tolist())
    return out

class Sample:
    def __init__(self, A, B, Cnat, nr=48, nc=48, seed=7):
        n = A.shape[0]
        rng = np.random.default_rng(seed)
        ri = set(rng.choice(n, nr, replace=False).tolist())
        ci = set(rng.choice(n, nc, replace=False).tolist())
        am = torch.abs(Cnat).max(dim=1).values.argmax().item()
        bm = torch.abs(Cnat).max(dim=0).values.argmax().item()
        ri.add(am)
        ci.add(bm)
        self.ri = np.array(sorted(ri))
        self.ci = np.array(sorted(ci))
        Ar = A[torch.as_tensor(self.ri, device=DEV)].cpu().numpy()
        Bc = B[:, torch.as_tensor(self.ci, device=DEV)].cpu().numpy()
        self.ref = exact_sample(Ar, Bc)
        self.absab = np.abs(Ar) @ np.abs(Bc)
        self.den1 = np.abs(self.ref).max()
        self.tri = torch.as_tensor(self.ri, device=DEV)
        self.tci = torch.as_tensor(self.ci, device=DEV)

    def errors(self, C):
        Cs = C[self.tri][:, self.tci].cpu().numpy()
        d = np.abs(Cs - self.ref)
        return {"E1": float(d.max() / self.den1), "E2": float((d / self.absab).max())}


OUT = os.environ.get("ENGINE_OUT", os.path.join("results", "engine"))
os.makedirs(OUT, exist_ok=True)

try:
    import pynvml

    pynvml.nvmlInit()
    _NVH = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:  # pragma: no cover
    _NVH = None


def gpu_state():
    """SM clock (MHz), temperature (C), power (W) and clock-throttle reason mask from NVML (cheap, no GPU sync)."""
    d = {}
    if _NVH is None:
        return d
    try:
        d["sm_mhz"] = int(pynvml.nvmlDeviceGetClockInfo(_NVH, pynvml.NVML_CLOCK_SM))
        d["temp_c"] = int(pynvml.nvmlDeviceGetTemperature(_NVH, pynvml.NVML_TEMPERATURE_GPU))
        d["power_w"] = round(pynvml.nvmlDeviceGetPowerUsage(_NVH) / 1000.0, 1)
        d["thr"] = int(pynvml.nvmlDeviceGetCurrentClocksEventReasons(_NVH))
    except Exception:
        pass
    return d


def ev_time(fn):
    """Seconds for one call of fn, measured with CUDA events on the default stream (synchronised)."""
    a = torch.cuda.Event(enable_timing=True)
    b = torch.cuda.Event(enable_timing=True)
    a.record()
    fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / 1e3


def set_high_priority():
    try:
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00000080)  # HIGH
    except Exception:
        pass


def _q(v, p):
    return float(np.percentile(np.asarray(v, dtype=float), p))


# =====================================================================================================================
# cuBLAS 13.1 through ctypes (explicit argtypes): native DGEMM, DGEMM emulation with its controls, INT8 GemmEx
# =====================================================================================================================
class Vendor:
    STRAT_PERFORMANT, STRAT_EAGER = 1, 2

    def __init__(self):
        self.lib = ctypes.WinDLL(os.path.join(CUDA13, "cublas64_13.dll"))

        def fn(name, argtypes):
            f = getattr(self.lib, name)
            f.argtypes = argtypes
            f.restype = ctypes.c_int
            return f

        vp, ci = c_void_p, c_int
        self._create = fn("cublasCreate_v2", [POINTER(c_void_p)])
        self._math = fn("cublasSetMathMode", [vp, ci])
        self._strat = fn("cublasSetEmulationStrategy", [vp, ci])
        self._mctl = fn("cublasSetFixedPointEmulationMantissaControl", [vp, ci])
        self._mbits = fn("cublasSetFixedPointEmulationMaxMantissaBitCount", [vp, ci])
        self._ws = fn("cublasSetWorkspace_v2", [vp, vp, c_size_t])
        self._dgemm = fn("cublasDgemm_v2", [vp, ci, ci, ci, ci, ci, vp, vp, ci, vp, ci, vp, vp, ci])
        self._gemmex = fn("cublasGemmEx", [vp, ci, ci, ci, ci, ci, vp, vp, ci, ci, vp, ci, ci, vp, vp, ci, ci, ci, ci])
        self._sv = fn("cublasSetEmulationSpecialValuesSupport", [vp, ci])
        self._off = fn("cublasSetFixedPointEmulationMantissaBitOffset", [vp, ci])
        self._bitsptr = fn("cublasSetFixedPointEmulationMantissaBitCountPointer", [vp, vp])
        self.h = c_void_p()
        self._chk(self._create(ctypes.byref(self.h)), "cublasCreate")
        self.bits = torch.full((1,), -2, dtype=torch.int32, device=DEV)      # the library writes the retained mantissa bits (-1 = native fallback)
        self._chk(self._bitsptr(self.h, c_void_p(self.bits.data_ptr())), "SetMantissaBitCountPointer")
        self.one, self.zero = ctypes.c_double(1.0), ctypes.c_double(0.0)
        self.ws_keep = None

    @staticmethod
    def _chk(st, what=""):
        if st != 0:
            raise RuntimeError(f"cuBLAS {what} status {st}")

    def configure(self, emulate, strategy=1, cap=0, fixed_bits=0, ws=None, ws_bytes=0, sv=None, offset=0):
        """emulate=False: native DGEMM (math mode 0).  Dynamic mantissa by default; cap = max mantissa bits (0 = library default);
        fixed_bits > 0: FIXED mantissa control; offset: bias of the dynamic recommendation; sv: special-values mask (0 = none required);
        ws: user workspace tensor of ws_bytes bytes (None = library-managed default: SetWorkspace is NOT called, because (NULL, 0) forces the native fallback)."""
        self._chk(self._math(self.h, 8 if emulate else 0), "SetMathMode")
        self._chk(self._strat(self.h, strategy if emulate else 0), "SetEmulationStrategy")
        if emulate:
            self._chk(self._mctl(self.h, 1 if fixed_bits else 0), "SetMantissaControl")
            self._chk(self._mbits(self.h, int(fixed_bits or cap or 0)), "SetMaxMantissaBitCount")
            if sv is not None:
                self._chk(self._sv(self.h, int(sv)), "SetSpecialValuesSupport")
            if offset:
                self._chk(self._off(self.h, int(offset)), "SetMantissaBitOffset")
        if ws is not None:
            self.ws_keep = ws
            self._chk(self._ws(self.h, c_void_p(ws.data_ptr()), int(ws_bytes or ws.numel())), "SetWorkspace")
        else:
            self.ws_keep = None

    def bits_used(self):
        return int(self.bits.item())

    def dgemm(self, A, B, C):
        n = A.shape[0]
        # row-major C = A@B  <=>  col-major C^T = B^T A^T
        return self._dgemm(self.h, 0, 0, n, n, n, ctypes.byref(self.one), c_void_p(B.data_ptr()), n,
                           c_void_p(A.data_ptr()), n, ctypes.byref(self.zero), c_void_p(C.data_ptr()), n)

    def int8_gemmex(self, A8, Bt8, C32):
        """C32[m,n] = A8[m,k] @ Bt8[n,k]^T through cublasGemmEx (TN, INT32 compute)."""
        m, k = A8.shape
        n = Bt8.shape[0]
        al, be = c_int32(1), c_int32(0)
        return self._gemmex(self.h, 1, 0, n, m, k, ctypes.byref(al), c_void_p(Bt8.data_ptr()), CUDA_R_8I, k,
                            c_void_p(A8.data_ptr()), CUDA_R_8I, k, ctypes.byref(be), c_void_p(C32.data_ptr()),
                            CUDA_R_32I, n, CUBLAS_COMPUTE_32I, -1)


# =====================================================================================================================
# Task 2: INT8 throughput diagnosis
# =====================================================================================================================
MMA_SRC = r'''
#define XORS(s) (s ^= s << 13, s ^= s >> 17, s ^= s << 5, s)
#define MMA_S8(C) asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n" \
  : "+r"(C[0]), "+r"(C[1]), "+r"(C[2]), "+r"(C[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1))
#define MMA_F16(C) asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n" \
  : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1))
#define MMA_E4(C) asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n" \
  : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1))

extern "C" __global__ void mma_s8(int iters, long long* cyc, int* sink) {
  unsigned s = 2463534242u ^ (threadIdx.x * 2654435761u) ^ (blockIdx.x * 40503u);
  unsigned a0 = XORS(s), a1 = XORS(s), a2 = XORS(s), a3 = XORS(s), b0 = XORS(s), b1 = XORS(s);
  int c0[4] = {0,0,0,0}, c1[4] = {0,0,0,0}, c2[4] = {0,0,0,0}, c3[4] = {0,0,0,0};
  int c4[4] = {0,0,0,0}, c5[4] = {0,0,0,0}, c6[4] = {0,0,0,0}, c7[4] = {0,0,0,0};
  long long t0 = clock64();
  for (int i = 0; i < iters; i++) { MMA_S8(c0); MMA_S8(c1); MMA_S8(c2); MMA_S8(c3); MMA_S8(c4); MMA_S8(c5); MMA_S8(c6); MMA_S8(c7); }
  long long t1 = clock64();
  int r = 0; for (int j = 0; j < 4; j++) r += c0[j]+c1[j]+c2[j]+c3[j]+c4[j]+c5[j]+c6[j]+c7[j];
  sink[blockIdx.x * blockDim.x + threadIdx.x] = r;
  if (threadIdx.x == 0) cyc[blockIdx.x] = t1 - t0;
}
extern "C" __global__ void mma_f16(int iters, long long* cyc, int* sink) {
  unsigned s = 2463534242u ^ (threadIdx.x * 2654435761u) ^ (blockIdx.x * 40503u);
  unsigned a0 = 0x3c003c00u ^ (XORS(s) & 0x03ff03ffu), a1 = 0x3c003c00u ^ (XORS(s) & 0x03ff03ffu), a2 = 0x3c003c00u ^ (XORS(s) & 0x03ff03ffu),
           a3 = 0x3c003c00u ^ (XORS(s) & 0x03ff03ffu), b0 = 0x3c003c00u ^ (XORS(s) & 0x03ff03ffu), b1 = 0x3c003c00u ^ (XORS(s) & 0x03ff03ffu);
  float c0[4] = {0,0,0,0}, c1[4] = {0,0,0,0}, c2[4] = {0,0,0,0}, c3[4] = {0,0,0,0};
  float c4[4] = {0,0,0,0}, c5[4] = {0,0,0,0}, c6[4] = {0,0,0,0}, c7[4] = {0,0,0,0};
  long long t0 = clock64();
  for (int i = 0; i < iters; i++) { MMA_F16(c0); MMA_F16(c1); MMA_F16(c2); MMA_F16(c3); MMA_F16(c4); MMA_F16(c5); MMA_F16(c6); MMA_F16(c7); }
  long long t1 = clock64();
  float r = 0; for (int j = 0; j < 4; j++) r += c0[j]+c1[j]+c2[j]+c3[j]+c4[j]+c5[j]+c6[j]+c7[j];
  sink[blockIdx.x * blockDim.x + threadIdx.x] = (int)r;
  if (threadIdx.x == 0) cyc[blockIdx.x] = t1 - t0;
}
extern "C" __global__ void mma_e4m3(int iters, long long* cyc, int* sink) {
  unsigned s = 2463534242u ^ (threadIdx.x * 2654435761u) ^ (blockIdx.x * 40503u);
  unsigned a0 = 0x38383838u ^ (XORS(s) & 0x07070707u), a1 = 0x38383838u ^ (XORS(s) & 0x07070707u), a2 = 0x38383838u ^ (XORS(s) & 0x07070707u),
           a3 = 0x38383838u ^ (XORS(s) & 0x07070707u), b0 = 0x38383838u ^ (XORS(s) & 0x07070707u), b1 = 0x38383838u ^ (XORS(s) & 0x07070707u);
  float c0[4] = {0,0,0,0}, c1[4] = {0,0,0,0}, c2[4] = {0,0,0,0}, c3[4] = {0,0,0,0};
  float c4[4] = {0,0,0,0}, c5[4] = {0,0,0,0}, c6[4] = {0,0,0,0}, c7[4] = {0,0,0,0};
  long long t0 = clock64();
  for (int i = 0; i < iters; i++) { MMA_E4(c0); MMA_E4(c1); MMA_E4(c2); MMA_E4(c3); MMA_E4(c4); MMA_E4(c5); MMA_E4(c6); MMA_E4(c7); }
  long long t1 = clock64();
  float r = 0; for (int j = 0; j < 4; j++) r += c0[j]+c1[j]+c2[j]+c3[j]+c4[j]+c5[j]+c6[j]+c7[j];
  sink[blockIdx.x * blockDim.x + threadIdx.x] = (int)r;
  if (threadIdx.x == 0) cyc[blockIdx.x] = t1 - t0;
}
'''


def mma_peak(target_s=0.25):
    """Bare mma.sync loops (no memory traffic): ops per clock per SM and the effective SM clock, for s8, f16 (f32 acc), e4m3."""
    nvrtc = Nvrtc()
    mod = Module(nvrtc.compile(MMA_SRC, "mma.cu"))
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    grid, block, warps = 2 * sms, 256, 8
    res = {}
    for name, opsm in (("s8", 8192), ("f16", 4096), ("e4m3", 8192)):
        k = mod.get("mma_" + name)
        cyc = torch.zeros(grid, dtype=torch.int64, device=DEV)
        sink = torch.zeros(grid * block, dtype=torch.int32, device=DEV)
        iters = 2000
        run = lambda it: k.launch((grid,), (block,), [c_int(it), ptr(cyc), ptr(sink)])
        t = ev_time(lambda: run(iters))
        iters = int(iters * target_s / t)
        rows = []
        for rep in range(5):
            st0 = gpu_state()
            t = ev_time(lambda: run(iters))
            c = float(cyc.float().mean().item())
            total_ops = grid * warps * iters * 8 * opsm
            rows.append({"t": t, "tops": total_ops / t / 1e12, "f_eff_mhz": c / t / 1e6,
                         "ops_per_clk_per_sm": total_ops / (c * sms), "state": st0})
        res[name] = {"iters": iters, "rows": rows,
                     "median_tops": _q([r["tops"] for r in rows], 50),
                     "median_ops_per_clk_per_sm": _q([r["ops_per_clk_per_sm"] for r in rows], 50),
                     "median_f_eff_mhz": _q([r["f_eff_mhz"] for r in rows], 50)}
    return res


def cmd_diag():
    set_high_priority()
    
    nvrtc_ok = True
    res = {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "sms": torch.cuda.get_device_properties(0).multi_processor_count,
           "datasheet_int8_dense_tops_at_1515mhz": 82 * 2048 * 1.515e9 / 1e12, "gpu_start": gpu_state(), "mma_loop": {}, "gemm": {}}
    print("datasheet INT8 dense TOPS at 1.515 GHz:", round(res["datasheet_int8_dense_tops_at_1515mhz"], 1), flush=True)
    res["mma_loop"] = mma_peak()
    for nm, v in res["mma_loop"].items():
        print(f"  mma loop {nm}: {v['median_tops']:.1f} T-ops/s, {v['median_ops_per_clk_per_sm']:.0f} ops/clk/SM, "
              f"f_eff {v['median_f_eff_mhz']:.0f} MHz", flush=True)
    vend = Vendor()
    vend.configure(False)
    lt = LtInt8()
    for n in (2048, 4096, 8192):
        g = torch.Generator(device=DEV)
        g.manual_seed(n)
        a8 = torch.randint(-127, 128, (n, n), device=DEV, dtype=torch.int8, generator=g)
        b8 = torch.randint(-127, 128, (n, n), device=DEV, dtype=torch.int8, generator=g)   # row-major k x n
        bt8 = b8.t().contiguous()                                                         # n x k (K contiguous)
        b_col = bt8.t()                                                                   # k x n column-major view
        c32 = torch.empty(n, n, dtype=torch.int32, device=DEV)
        a16 = torch.randn(n, n, device=DEV, dtype=torch.float16)
        b16 = torch.randn(n, n, device=DEV, dtype=torch.float16)
        a_bf = a16.to(torch.bfloat16)
        b_bf = b16.to(torch.bfloat16)
        a_f8 = (torch.randn(n, n, device=DEV) * 0.5).to(torch.float8_e4m3fn)
        b_f8 = (torch.randn(n, n, device=DEV) * 0.5).to(torch.float8_e4m3fn).t().contiguous().t()  # column-major operand
        one = torch.tensor(1.0, device=DEV)
        # bit-exactness of every INT8 path against torch._int_mm on this data
        ref = torch._int_mm(a8, b8)
        lt.plan(n, n, n)
        tune = lt.autotune(a8, bt8, c32, n, n, n, reps=3)
        lt.gemm_best(ptr(a8), ptr(bt8), ptr(c32), n, n, n)
        exact = {"lt_best": bool(torch.equal(c32, ref))}
        torch.cuda.synchronize()
        c2 = torch.empty_like(c32)
        st = vend.int8_gemmex(a8, bt8, c2)
        torch.cuda.synchronize()
        exact["gemmex"] = bool(st == 0 and torch.equal(c2, ref))
        exact["int_mm_colmajor"] = bool(torch.equal(torch._int_mm(a8, b_col), ref))
        del ref, c2
        paths = {
            "fp16_matmul": lambda: a16 @ b16,
            "bf16_matmul": lambda: a_bf @ b_bf,
            "fp8_e4m3_scaled_mm": lambda: torch._scaled_mm(a_f8, b_f8, scale_a=one, scale_b=one, out_dtype=torch.float16),
            "int_mm_rowmajorB": lambda: torch._int_mm(a8, b8),
            "int_mm_colmajorB": lambda: torch._int_mm(a8, b_col),
            "transpose_B_only": lambda: b8.t().contiguous(),
            "lt_int8_best": lambda: lt.gemm_best(ptr(a8), ptr(bt8), ptr(c32), n, n, n),
            "gemmex_int8": lambda: vend.int8_gemmex(a8, bt8, c32),
        }
        # loaded-state warm-up (FP16 GEMMs) then interleaved random-order rounds
        t_end = time.time() + 15
        while time.time() < t_end:
            ev_time(paths["fp16_matmul"])
        for fn in paths.values():
            fn()
        torch.cuda.synchronize()
        R = 11
        rng = np.random.RandomState(n)
        T = {k: [] for k in paths}
        S = {k: [] for k in paths}
        for r in range(R):
            for kname in rng.permutation(list(paths)):
                T[kname].append(ev_time(paths[kname]))
                S[kname].append(gpu_state())
        cell = {"exact_vs_int_mm": exact, "lt_autotune_top": [(i, round(2 * n ** 3 / t / 1e12, 1)) for i, t in tune[:8]],
                "lt_n_algos": len(tune)}
        for kname in paths:
            t = np.array(T[kname])
            cell[kname] = {"med_s": float(np.median(t)), "q1_s": _q(t, 25), "q3_s": _q(t, 75),
                           "tops_med": 2 * n ** 3 / float(np.median(t)) / 1e12,
                           "tops_q1": 2 * n ** 3 / _q(t, 75) / 1e12, "tops_q3": 2 * n ** 3 / _q(t, 25) / 1e12,
                           "sm_mhz_med": _q([s.get("sm_mhz", 0) for s in S[kname]], 50),
                           "temp_med": _q([s.get("temp_c", 0) for s in S[kname]], 50),
                           "power_med": _q([s.get("power_w", 0) for s in S[kname]], 50)}
        res["gemm"][n] = cell
        print(n, {k: (round(v["tops_med"], 1) if isinstance(v, dict) and "tops_med" in v else v) for k, v in cell.items()
                  if k not in ("lt_autotune_top",)}, flush=True)
        del a8, b8, bt8, b_col, a16, b16, a_bf, b_bf, a_f8, b_f8
        torch.cuda.empty_cache()
    res["gpu_end"] = gpu_state()
    json.dump(res, open(os.path.join(OUT, "int8_diag.json"), "w"), indent=1)


# =====================================================================================================================
# The engine: CRT (Ozaki-II) INT8 emulation of a float64 GEMM, specification in docs/engine.md
# =====================================================================================================================
from ctypes import c_longlong  # noqa: E402

MODS = [256, 255, 253, 251, 247, 241, 239, 233, 229, 227, 223, 217, 211, 199, 197, 193, 191, 181, 179, 173, 167, 163,
        157, 151, 149, 139, 137, 131, 127, 113, 109, 107]
NMAX = len(MODS)
NEG = -1000000
assert all(math.gcd(a, b) == 1 for i, a in enumerate(MODS) for b in MODS[i + 1:]), "moduli not pairwise coprime"
assert all(2 <= m <= 256 for m in MODS)

_MODP = "constexpr unsigned modp(int t) { return " + "".join(f"t=={i}?{m}u:" for i, m in enumerate(MODS[:-1])) + f"{MODS[-1]}u; }}"

CU_COMMON = r'''
typedef unsigned long long u64;
#define NEGINF (-1000000)
#define ABSMASK 0x7fffffffffffffffULL
''' + "__host__ __device__ " + _MODP + r'''
__device__ __forceinline__ int fexp(u64 b) {      // frexp exponent of |x| from its abs bit pattern; NEGINF for zero
  if (b == 0ULL) return NEGINF;
  unsigned E = (unsigned)(b >> 52);
  if (E == 0u) return (63 - __clzll((long long)b)) - 1073;
  return (int)E - 1022;
}
'''

CU_STATS = CU_COMMON + r'''
extern "C" __global__ void colmax_k(const u64* __restrict__ A, u64* __restrict__ out, int m, int k, int rows_per) {
  int l = blockIdx.x * blockDim.x + threadIdx.x;
  if (l >= k) return;
  int i0 = blockIdx.y * rows_per, i1 = min(m, i0 + rows_per);
  u64 mx = 0;
  for (int i = i0; i < i1; i++) { u64 b = A[(long long)i * k + l] & ABSMASK; if (b > mx) mx = b; }
  atomicMax(out + l, mx);
}
extern "C" __global__ void rowmax_k(const u64* __restrict__ B, u64* __restrict__ out, int n) {   // block per row of B
  int l = blockIdx.x;
  u64 mx = 0;
  for (int j = threadIdx.x; j < n; j += blockDim.x) { u64 b = B[(long long)l * n + j] & ABSMASK; if (b > mx) mx = b; }
  for (int o = 16; o > 0; o >>= 1) { u64 t = __shfl_down_sync(0xffffffffu, mx, o); if (t > mx) mx = t; }
  __shared__ u64 sm[32];
  int w = threadIdx.x >> 5, ln = threadIdx.x & 31;
  if (ln == 0) sm[w] = mx;
  __syncthreads();
  if (w == 0) {
    mx = (ln < (int)(blockDim.x >> 5)) ? sm[ln] : 0ULL;
    for (int o = 16; o > 0; o >>= 1) { u64 t = __shfl_down_sync(0xffffffffu, mx, o); if (t > mx) mx = t; }
    if (ln == 0) out[l] = mx;
  }
}
extern "C" __global__ void gamma_k(const u64* __restrict__ cA, const u64* __restrict__ rB, int* __restrict__ gam,
                                   int* __restrict__ flag, int k, int enable) {
  int l = blockIdx.x * blockDim.x + threadIdx.x;
  if (l >= k) return;
  u64 a = cA[l], b = rB[l];
  bool bad = ((a >> 52) == 0x7ffULL) || ((b >> 52) == 0x7ffULL);
  if (bad) atomicOr(flag, 1);
  int ea = fexp(a), eb = fexp(b);
  gam[l] = (!enable || bad || ea == NEGINF || eb == NEGINF) ? 0 : ((eb - ea) >> 1);
}
__device__ __forceinline__ int blockMaxI(int v, int* sm) {
  for (int o = 16; o > 0; o >>= 1) v = max(v, __shfl_down_sync(0xffffffffu, v, o));
  int w = threadIdx.x >> 5, ln = threadIdx.x & 31;
  __syncthreads();
  if (ln == 0) sm[w] = v;
  __syncthreads();
  v = (ln < (int)(blockDim.x >> 5)) ? sm[ln] : NEGINF;
  for (int o = 16; o > 0; o >>= 1) v = max(v, __shfl_down_sync(0xffffffffu, v, o));
  if (threadIdx.x == 0) sm[0] = v;
  __syncthreads();
  return sm[0];
}
__device__ __forceinline__ double blockSumD(double v, double* sm) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
  int w = threadIdx.x >> 5, ln = threadIdx.x & 31;
  __syncthreads();
  if (ln == 0) sm[w] = v;
  __syncthreads();
  v = (ln < (int)(blockDim.x >> 5)) ? sm[ln] : 0.0;
  for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
  if (threadIdx.x == 0) sm[0] = v;
  __syncthreads();
  return sm[0];
}
extern "C" __global__ void rowstatsA_k(const double* __restrict__ A, const int* __restrict__ gam, int k,
                                       int* __restrict__ r_out, double* __restrict__ s_out, double* __restrict__ n_out) {
  int i = blockIdx.x;
  const double* row = A + (long long)i * k;
  __shared__ int smi[32];
  __shared__ double smd[32];
  int mxe = NEGINF;
  for (int l = threadIdx.x; l < k; l += blockDim.x) {
    int e = fexp(((u64)__double_as_longlong(row[l])) & ABSMASK);
    if (e != NEGINF) { e += gam[l]; if (e > mxe) mxe = e; }
  }
  int rr = blockMaxI(mxe, smi);
  if (rr <= NEGINF / 2) rr = 0;
  double s = 0.0, nn = 0.0;
  for (int l = threadIdx.x; l < k; l += blockDim.x) {
    double x = fabs(row[l]);
    if (x != 0.0) { s += ldexp(x, gam[l]); double y = ldexp(x, gam[l] - rr); nn += y * y; }
  }
  s = blockSumD(s, smd);
  nn = blockSumD(nn, smd);
  if (threadIdx.x == 0) { r_out[i] = rr; s_out[i] = s; n_out[i] = nn; }
}
extern "C" __global__ void colexpB_k(const double* __restrict__ B, const int* __restrict__ gam, int k, int n, int l_per,
                                     int* __restrict__ c_out) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= n) return;
  int l0 = blockIdx.y * l_per, l1 = min(k, l0 + l_per);
  int mxe = NEGINF;
  for (int l = l0; l < l1; l++) {
    int e = fexp(((u64)__double_as_longlong(B[(long long)l * n + j])) & ABSMASK);
    if (e != NEGINF) { e -= gam[l]; if (e > mxe) mxe = e; }
  }
  if (mxe > NEGINF / 2) atomicMax(c_out + j, mxe);
}
extern "C" __global__ void colsumsB_k(const double* __restrict__ B, const int* __restrict__ gam, const int* __restrict__ c,
                                      int k, int n, int l_per, double* __restrict__ s_out, double* __restrict__ n_out) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= n) return;
  int cj = c[j]; if (cj <= NEGINF / 2) cj = 0;
  int l0 = blockIdx.y * l_per, l1 = min(k, l0 + l_per);
  double s = 0.0, nn = 0.0;
  for (int l = l0; l < l1; l++) {
    double x = fabs(B[(long long)l * n + j]);
    if (x != 0.0) { s += ldexp(x, -gam[l]); double y = ldexp(x, -gam[l] - cj); nn += y * y; }
  }
  atomicAdd(s_out + j, s);
  atomicAdd(n_out + j, nn);
}
template<int T> __device__ __forceinline__ unsigned modnn(int v) {
  constexpr unsigned P = modp(T);
  int r = v % (int)P;
  return (unsigned)(r < 0 ? r + (int)P : r);
}
extern "C" __global__ void reduce_k(const int* __restrict__ c, unsigned char* __restrict__ out, long long total4, int t) {
  long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= total4) return;
  int4 v = reinterpret_cast<const int4*>(c)[g];
  unsigned r0 = 0, r1 = 0, r2 = 0, r3 = 0;
  switch (t) {
''' + "".join(f"    case {i}: r0 = modnn<{i}>(v.x); r1 = modnn<{i}>(v.y); r2 = modnn<{i}>(v.z); r3 = modnn<{i}>(v.w); break;\n"
              for i in range(NMAX)) + r'''    default: break;
  }
  reinterpret_cast<unsigned*>(out)[g] = r0 | (r1 << 8) | (r2 << 16) | (r3 << 24);
}
'''

CU_RES = CU_COMMON + r'''
template<int T> __device__ __forceinline__ unsigned mod53(u64 x) {
  constexpr unsigned P = modp(T);
  constexpr unsigned R = (1u << 26) % P;
  unsigned hi = (unsigned)(x >> 26), lo = (unsigned)(x & 0x3ffffffULL);
  return ((hi % P) * R + (lo % P)) % P;
}
__device__ __forceinline__ void decomp(double v, int ag, u64& x, int& sh, unsigned& neg) {
  u64 b = (u64)__double_as_longlong(v);
  neg = (unsigned)(b >> 63);
  unsigned E = (unsigned)((b >> 52) & 0x7ffu);
  u64 f = b & 0xfffffffffffffULL;
  u64 mant; int e2;
  if (E == 0u) { mant = f; e2 = -1074; } else { mant = f | (1ULL << 52); e2 = (int)E - 1075; }
  int s = e2 + ag;                           // |v| 2^ag = mant 2^s
  if (mant == 0ULL) { x = 0ULL; sh = -1; return; }
  if (s >= 0) { x = mant; sh = s; }
  else { int q = -s; x = (q >= 54) ? 0ULL : ((mant + (1ULL << (q - 1))) >> q); sh = -1; }   // round half away from zero
}
template<int T> __device__ __forceinline__ int resid(u64 x, int sh, unsigned neg, const unsigned char* __restrict__ tab) {
  constexpr unsigned P = modp(T);
  unsigned r = mod53<T>(x);
  if (sh > 0) r = (r * (unsigned)tab[T * 256 + sh]) % P;
  if (neg && r != 0u) r = P - r;
  return (r >= (P + 1u) / 2u) ? (int)r - (int)P : (int)r;
}
extern "C" __global__ void resA_k(const double* __restrict__ A, const int* __restrict__ rr, const int* __restrict__ gam,
                                  signed char* __restrict__ out, const unsigned char* __restrict__ tab, int m, int k,
                                  const int* __restrict__ pp) {
  int k4 = k >> 2;
  long long gid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= (long long)m * k4) return;
  int i = (int)(gid / k4), l4 = (int)(gid - (long long)i * k4);
  const double2* rowp = reinterpret_cast<const double2*>(A + (long long)i * k);
  double2 v01 = rowp[2 * l4], v23 = rowp[2 * l4 + 1];
  int4 g = reinterpret_cast<const int4*>(gam)[l4];
  int base = *pp - rr[i];
  u64 x[4]; int sh[4]; unsigned ng[4];
  decomp(v01.x, base + g.x, x[0], sh[0], ng[0]);
  decomp(v01.y, base + g.y, x[1], sh[1], ng[1]);
  decomp(v23.x, base + g.z, x[2], sh[2], ng[2]);
  decomp(v23.y, base + g.w, x[3], sh[3], ng[3]);
  long long plane = (long long)m * k;
#define RA(T) if (T < NMOD) { unsigned w = 0; _Pragma("unroll") for (int q = 0; q < 4; q++) w |= ((unsigned)(resid<T>(x[q], sh[q], ng[q], tab) & 0xff)) << (8 * q); \
  reinterpret_cast<unsigned*>(out + (long long)T * plane + (long long)i * k)[l4] = w; }
''' + "".join(f"  RA({i})\n" for i in range(NMAX)) + r'''}
extern "C" __global__ void resB_k(const double* __restrict__ B, const int* __restrict__ cc, const int* __restrict__ gam,
                                  signed char* __restrict__ out, const unsigned char* __restrict__ tab, int k, int n,
                                  const int* __restrict__ pp) {
  __shared__ signed char sm[NMOD][32][36];
  int tx = threadIdx.x, ty = threadIdx.y;
  int j0 = blockIdx.x * 32, l0 = blockIdx.y * 32;
  int j = j0 + tx;
  int cj = cc[j]; if (cj <= NEGINF / 2) cj = 0;
  int base = *pp - cj;
  for (int rrow = 0; rrow < 4; rrow++) {
    int ll = ty + 8 * rrow;
    int l = l0 + ll;
    double v = B[(long long)l * n + j];
    u64 x; int sh; unsigned ng;
    decomp(v, base - gam[l], x, sh, ng);
#define RB(T) if (T < NMOD) sm[T][ll][tx] = (signed char)resid<T>(x, sh, ng, tab);
''' + "".join(f"    RB({i})\n" for i in range(NMAX)) + r'''  }
  __syncthreads();
  int tid = ty * 32 + tx;
  int jj = tid >> 3, l4 = tid & 7;
#define WB(T) if (T < NMOD) { unsigned w = 0; _Pragma("unroll") for (int q = 0; q < 4; q++) w |= ((unsigned)(unsigned char)sm[T][4 * l4 + q][jj]) << (8 * q); \
  *reinterpret_cast<unsigned*>(out + ((long long)T * n + (j0 + jj)) * (long long)k + l0 + 4 * l4) = w; }
''' + "".join(f"  WB({i})\n" for i in range(NMAX)) + "}\n"

CU_RECON = CU_COMMON + r'''
__device__ __constant__ unsigned char GC[32 * 32];
__device__ __constant__ unsigned char INVP[32];
__device__ __constant__ unsigned MLIM[LIMBS];
__device__ __constant__ unsigned MHALF[LIMBS];
extern "C" __global__ void recon_k(const unsigned char* __restrict__ X, const int* __restrict__ rr, const int* __restrict__ cc,
                                   double* __restrict__ C, long long mn, int n, const int* __restrict__ pp) {
  int i = blockIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= n) return;
  long long idx = (long long)i * n + j;
  int p = *pp;
  int cj = cc[j]; if (cj <= NEGINF / 2) cj = 0;
  unsigned x[NMOD];
#pragma unroll
  for (int t = 0; t < NMOD; t++) x[t] = X[(long long)t * mn + idx];
  unsigned v[NMOD];
  v[0] = x[0];
#define GS(T) if (T >= 1 && T < NMOD) { unsigned u = 0; _Pragma("unroll") for (int s = 0; s < T; s++) u += v[s] * (unsigned)GC[T * 32 + s]; \
  constexpr unsigned P = modp(T); int uu = (int)(u % P); int d = (int)x[T] - uu; if (d < 0) d += (int)P; v[T] = ((unsigned)d * (unsigned)INVP[T]) % P; }
''' + "".join(f"  GS({i})\n" for i in range(1, NMAX)) + r'''  unsigned a[LIMBS];
#pragma unroll
  for (int q = 0; q < LIMBS; q++) a[q] = 0u;
  a[0] = v[NMOD - 1];
#define HS(T) if (T <= NMOD - 2) { u64 carry = v[T]; _Pragma("unroll") for (int q = 0; q < LIMBS; q++) { u64 tt = (u64)a[q] * modp(T) + carry; a[q] = (unsigned)tt; carry = tt >> 32; } }
''' + "".join(f"  HS({i})\n" for i in range(NMAX - 2, -1, -1)) + r'''  bool gt = false, eq = true;
#pragma unroll
  for (int q = LIMBS - 1; q >= 0; q--) { unsigned mh = MHALF[q]; gt = gt | (eq & (a[q] > mh)); eq = eq & (a[q] == mh); }
  if (gt) {
    u64 borrow = 0ULL;
#pragma unroll
    for (int q = 0; q < LIMBS; q++) { u64 tt = (u64)MLIM[q] - (u64)a[q] - borrow; a[q] = (unsigned)tt; borrow = (tt >> 63) & 1ULL; }
  }
  unsigned sticky[LIMBS + 1];
  sticky[0] = 0u;
#pragma unroll
  for (int q = 0; q < LIMBS; q++) sticky[q + 1] = sticky[q] | a[q];
  bool found = false;
  unsigned w2 = 0u, w1 = 0u, w0 = 0u, st = 0u;
  int topi = 0;
#pragma unroll
  for (int q = LIMBS - 1; q >= 0; q--) {
    if (!found && a[q] != 0u) {
      found = true; topi = q; w2 = a[q]; w1 = (q >= 1) ? a[(q >= 1) ? q - 1 : 0] : 0u; w0 = (q >= 2) ? a[(q >= 2) ? q - 2 : 0] : 0u;
      st = (q >= 3) ? sticky[(q >= 3) ? q - 2 : 0] : 0u;
    }
  }
  double outv = 0.0;
  if (found) {
    int lz = __clz(w2);
    u64 hi = ((u64)w2 << 32) | (u64)w1;
    u64 qv; unsigned lost;
    if (lz == 0) { qv = hi; lost = w0; } else { qv = (hi << lz) | ((u64)w0 >> (32 - lz)); lost = w0 << lz; }
    if ((lost | st) != 0u) qv |= 1ULL;
    double d = __ull2double_rn(qv);
    int s = 32 * (topi - 1) - lz;
    outv = ldexp(d, s - 2 * p + rr[i] + cj);
    if (gt) outv = -outv;
  }
  C[idx] = outv;
}
'''


def _limbs_for(N):
    M = math.prod(MODS[:N])
    return max(2, (M.bit_length() + 31) // 32)


def _pow2tab():
    t = np.zeros((32, 256), dtype=np.uint8)
    for ti, p in enumerate(MODS):
        v = 1
        for s in range(256):
            t[ti, s] = v % p
            v = (v * 2) % p
    return t


def pmax_bits(N, nuA, nuB, k):
    """Largest integer p with 2 Xmax(p) + 1 <= M_N, Xmax(p) = (2^p nuA + sqrt(k)/2)(2^p nuB + sqrt(k)/2) (closed form + fix-up)."""
    M = float(math.prod(MODS[:N]))
    sk = math.sqrt(k) / 2

    def fits(p):
        return 2 * (2.0 ** p * nuA + sk) * (2.0 ** p * nuB + sk) * (1 + 1e-9) + 2 <= M

    if not fits(0):
        return 0
    a, bq, c = nuA * nuB, sk * (nuA + nuB), sk * sk - (M - 2) / (2 * (1 + 1e-9))
    if a == 0.0:
        y = -c / bq if bq > 0 else 2.0 ** 249
    else:
        y = (-bq + math.sqrt(bq * bq - 4 * a * c)) / (2 * a)
    p = max(0, min(249, int(math.floor(math.log2(y))) if y > 0 else 0))
    while p > 0 and not fits(p):
        p -= 1
    while p < 249 and fits(p + 1):
        p += 1
    return p


def bound_matrix(r, c, sA, sB, k, p, I, J):
    """E_ij(p) of Lemma L1 on the pairs I x J (r, c int exponents; sA_i = sum_l |A_il| 2^gamma_l, sB_j = sum_l |B_lj| 2^-gamma_l)."""
    rI = np.asarray(r, dtype=np.float64)[I][:, None]
    cJ = np.asarray(c, dtype=np.float64)[J][None, :]
    return (0.5 * np.exp2(rI - p) * np.asarray(sB)[J][None, :] + 0.5 * np.exp2(cJ - p) * np.asarray(sA)[I][:, None]
            + 0.25 * k * np.exp2(rI + cJ - 2 * p))


class OzakiCRT:
    def __init__(self, lt=None):
        self.nvrtc = Nvrtc()
        self.lt = lt or LtInt8()
        self.stats_mod = Module(self.nvrtc.compile(CU_STATS, "stats.cu"))
        self.kern = {n: self.stats_mod.get(n) for n in ("colmax_k", "rowmax_k", "gamma_k", "rowstatsA_k", "colexpB_k",
                                                         "colsumsB_k", "reduce_k")}
        self.tab = torch.from_numpy(_pow2tab()).to(DEV)
        self.per_n = {}
        self.bufs = {}
        self.graphs = {}
        self.probe = {}
        self.Mvec = torch.tensor([float(math.prod(MODS[:N])) for N in range(6, NMAX + 1)], dtype=torch.float64, device=DEV)
        self.cache = os.path.join(os.environ.get("TEMP", "."), "ozaki_cubin_cache")
        os.makedirs(self.cache, exist_ok=True)

    # ---- per-N compiled kernels (residue + reconstruction), cached in memory and on disk ------------------------------
    def _compile(self, name, src, opts):
        import hashlib
        key = hashlib.sha1((src + "|".join(opts)).encode()).hexdigest()[:16]
        fp = os.path.join(self.cache, f"{name}_{key}.cubin")
        if os.path.exists(fp):
            return open(fp, "rb").read()
        cub = self.nvrtc.compile(src, name + ".cu", extra_opts=opts)
        open(fp, "wb").write(cub)
        return cub

    def kernels(self, N):
        if N in self.per_n:
            return self.per_n[N]
        L = _limbs_for(N)
        opts = [f"-DNMOD={N}", f"-DLIMBS={L}"]
        mres = Module(self._compile("res", CU_RES, opts))
        mrec = Module(self._compile("recon", CU_RECON, opts))
        prod = [1]
        for p in MODS[:N]:
            prod.append(prod[-1] * p)
        GC = np.zeros((32, 32), dtype=np.uint8)
        INVP = np.zeros(32, dtype=np.uint8)
        for t in range(1, N):
            for s in range(t):
                GC[t, s] = prod[s] % MODS[t]
            INVP[t] = pow(prod[t], -1, MODS[t])
        M = prod[N]
        Mh = (M - 1) // 2
        mrec.set_global("GC", GC.tobytes())
        mrec.set_global("INVP", INVP.tobytes())
        mrec.set_global("MLIM", np.array([(M >> (32 * i)) & 0xffffffff for i in range(L)], dtype=np.uint32).tobytes())
        mrec.set_global("MHALF", np.array([(Mh >> (32 * i)) & 0xffffffff for i in range(L)], dtype=np.uint32).tobytes())
        d = {"resA": mres.get("resA_k"), "resB": mres.get("resB_k"), "recon": mrec.get("recon_k"), "mods": (mres, mrec), "L": L}
        self.per_n[N] = d
        return d

    def buffers(self, m, n, k, N):
        key = (m, n, k)
        b = self.bufs.get(key)
        if b is None or b["N"] < N:
            self.bufs.pop(key, None)
            self.graphs.clear()                       # every captured graph points into the old buffers
            b = None
            torch.cuda.empty_cache()
            cap = max(N, 20)
            bi = torch.empty(m + n + 1, dtype=torch.int32, device=DEV)            # r | c | flag
            bd = torch.empty(2 * m + 2 * n, dtype=torch.float64, device=DEV)      # sA | nA | sB | nB
            b = {"N": cap,
                 "Ares": torch.empty(cap * m * k, dtype=torch.int8, device=DEV),
                 "Bres": torch.empty(cap * n * k, dtype=torch.int8, device=DEV),
                 "X": torch.empty(cap * m * n, dtype=torch.uint8, device=DEV),
                 "C32": torch.empty(m * n, dtype=torch.int32, device=DEV),
                 "colmaxA": torch.empty(k, dtype=torch.int64, device=DEV), "rowmaxB": torch.empty(k, dtype=torch.int64, device=DEV),
                 "gam": torch.empty(k, dtype=torch.int32, device=DEV), "pdev": torch.zeros(1, dtype=torch.int32, device=DEV),
                 "bi": bi, "bd": bd, "r": bi[:m], "c": bi[m:m + n], "flag": bi[m + n:],
                 "sA": bd[:m], "nA": bd[m:2 * m], "sB": bd[2 * m:2 * m + n], "nB": bd[2 * m + n:],
                 "hi": torch.empty(m + n + 1, dtype=torch.int32).pin_memory(), "hd": torch.empty(2 * m + 2 * n, dtype=torch.float64).pin_memory(),
                 "dec": torch.zeros(4, dtype=torch.int32, device=DEV), "hdec": torch.zeros(4, dtype=torch.int32).pin_memory()}
            self.bufs[key] = b
        return b

    # ---- stage 1: statistics (device) ---------------------------------------------------------------------------------
    def stats(self, A, B, b, inside=True, stream=None):
        m, k = A.shape
        n = B.shape[1]
        K = self.kern
        if stream is None:
            stream = torch.cuda.current_stream().cuda_stream
        b["colmaxA"].zero_(); b["flag"].zero_(); b["c"].fill_(NEG); b["bd"][2 * m:].zero_()
        rows_per = 64
        K["colmax_k"].launch(((k + 255) // 256, (m + rows_per - 1) // rows_per), (256,),
                             [ptr(A), ptr(b["colmaxA"]), c_int(m), c_int(k), c_int(rows_per)], stream=stream)
        K["rowmax_k"].launch((k,), (256,), [ptr(B), ptr(b["rowmaxB"]), c_int(n)], stream=stream)
        K["gamma_k"].launch(((k + 255) // 256,), (256,), [ptr(b["colmaxA"]), ptr(b["rowmaxB"]), ptr(b["gam"]), ptr(b["flag"]),
                                                          c_int(k), c_int(1 if inside else 0)], stream=stream)
        K["rowstatsA_k"].launch((m,), (256,), [ptr(A), ptr(b["gam"]), c_int(k), ptr(b["r"]), ptr(b["sA"]), ptr(b["nA"])], stream=stream)
        l_per = 256
        g2 = ((n + 127) // 128, (k + l_per - 1) // l_per)
        K["colexpB_k"].launch(g2, (128,), [ptr(B), ptr(b["gam"]), c_int(k), c_int(n), c_int(l_per), ptr(b["c"])], stream=stream)
        K["colsumsB_k"].launch(g2, (128,), [ptr(B), ptr(b["gam"]), ptr(b["c"]), c_int(k), c_int(n), c_int(l_per),
                                            ptr(b["sB"]), ptr(b["nB"])], stream=stream)

    # ---- stage 2: the a-priori rule (host; synchronises) -----------------------------------------------------------------
    def choose(self, A, B, b, tau=2.0 ** -52, force_N=None, seed=0, safety=2.0, prof=None):
        m, k = A.shape
        n = B.shape[1]
        b["hi"].copy_(b["bi"], non_blocking=True)
        b["hd"].copy_(b["bd"], non_blocking=True)
        torch.cuda.current_stream().synchronize()
        t0 = time.perf_counter()
        hi, hd = b["hi"].numpy(), b["hd"].numpy()
        h = {"r": hi[:m].copy(), "c": np.where(hi[m:m + n] <= NEG // 2, 0, hi[m:m + n]), "sA": hd[:m].copy(), "nA": hd[m:2 * m].copy(),
             "sB": hd[2 * m:2 * m + n].copy(), "nB": hd[2 * m + n:].copy()}
        info = {"nonfinite": bool(hi[m + n] != 0)}
        if info["nonfinite"]:
            info["fallback"] = True
            return None, None, info, h
        nuA, nuB = math.sqrt(float(h["nA"].max())), math.sqrt(float(h["nB"].max()))
        info["nuA"], info["nuB"] = nuA, nuB
        if force_N is not None:
            return force_N, pmax_bits(force_N, nuA, nuB, k), info, h
        rs = np.random.RandomState(seed)
        Ir = rs.choice(m, min(64, m), replace=False)
        Jr = rs.choice(n, min(64, n), replace=False)
        with np.errstate(all="ignore"):
            hard_r = np.where(h["sA"] > 0, np.exp2(h["r"].astype(np.float64)) * k / h["sA"], 0.0)
            hard_c = np.where(h["sB"] > 0, np.exp2(h["c"].astype(np.float64)) * k / h["sB"], 0.0)
        I = np.unique(np.concatenate([Ir, np.argsort(-hard_r)[:32]]))
        J = np.unique(np.concatenate([Jr, np.argsort(-hard_c)[:32]]))
        It, Jt = torch.as_tensor(I, device=DEV), torch.as_tensor(J, device=DEV)
        Dh = (torch.abs(A.index_select(0, It)) @ torch.abs(B.index_select(1, Jt))).cpu().numpy()
        info["probe"] = [int(len(I)), int(len(J))]
        with np.errstate(all="ignore"):
            rI = h["r"][I].astype(np.float64)[:, None]
            cJ = h["c"][J].astype(np.float64)[None, :]
            F1 = 0.5 * (np.exp2(rI) * h["sB"][J][None, :] + np.exp2(cJ) * h["sA"][I][:, None])     # E = F1 2^-p + F2 2^-2p
            F2 = 0.25 * k * np.exp2(rI + cJ)
            mask = Dh > 0
            aq, bq, cq = F2[mask], F1[mask], (tau * Dh / safety)[mask]
            xmax = 2 * cq / (bq + np.sqrt(bq * bq + 4 * aq * cq))                                  # largest 2^-p meeting E <= tau D/2
            preq = int(np.ceil(np.max(-np.log2(xmax)))) if mask.any() else 1
            preq = max(preq, 1)

            def ok(p):
                E = F1 * 2.0 ** -p + F2 * 2.0 ** (-2 * p)
                return bool(np.all(~mask | (E * safety <= tau * Dh)))
            while preq > 1 and ok(preq - 1):
                preq -= 1
            while not ok(preq) and preq < 249:
                preq += 1
            E = F1 * 2.0 ** -preq + F2 * 2.0 ** (-2 * preq)
            info["worst_ratio_E_over_D"] = float(np.max(np.where(mask, E / np.where(mask, Dh, 1.0), 0.0)))
        info["p_req"] = preq
        for N in range(6, NMAX + 1):
            p = pmax_bits(N, nuA, nuB, k)
            if p >= preq:
                if prof is not None:
                    prof["rule_host"] = time.perf_counter() - t0
                return N, p, info, h
        info["fallback"] = True
        return None, None, info, h

    # ---- stage 3: residues, INT8 GEMMs, reconstruction (p is read from the device so the graph is p-independent) --------------
    def core(self, A, B, b, N, C, stream=None, prof=None, skipA=False):
        if stream is None:
            stream = torch.cuda.current_stream().cuda_stream
        m, k = A.shape
        n = B.shape[1]
        assert k % 4 == 0 and n % 32 == 0 and m % 4 == 0 and (m * n) % 4 == 0 and k % 32 == 0
        assert k <= 131071, "INT32 exactness bound k*128*128 < 2^31"
        kd = self.kernels(N)
        evs = []

        def mark(name):
            if prof is not None:
                e = torch.cuda.Event(enable_timing=True)
                e.record()
                evs.append((name, e))
        mark("start")
        tot = m * (k // 4)
        if not skipA:
            kd["resA"].launch(((tot + 255) // 256,), (256,), [ptr(A), ptr(b["r"]), ptr(b["gam"]), ptr(b["Ares"]), ptr(self.tab),
                                                              c_int(m), c_int(k), ptr(b["pdev"])], stream=stream)
        kd["resB"].launch((n // 32, k // 32), (32, 8), [ptr(B), ptr(b["c"]), ptr(b["gam"]), ptr(b["Bres"]), ptr(self.tab),
                                                         c_int(k), c_int(n), ptr(b["pdev"])], stream=stream)
        mark("residues")
        total4 = (m * n) // 4
        Ap, Bp = b["Ares"].data_ptr(), b["Bres"].data_ptr()
        c32p = c_void_p(b["C32"].data_ptr())
        for t in range(N):
            st = self.lt.gemm(c_void_p(Ap + t * m * k), c_void_p(Bp + t * n * k), c32p, m, n, k, stream=stream)
            if st != 0:
                raise RuntimeError(f"cublasLt INT8 gemm status {st}")
            self.kern["reduce_k"].launch(((total4 + 255) // 256,), (256,), [ptr(b["C32"]), c_void_p(b["X"].data_ptr() + t * m * n),
                                                                             c_longlong(total4), c_int(t)], stream=stream)
        mark("gemm+reduce")
        kd["recon"].launch(((n + 127) // 128, m), (128,), [ptr(b["X"]), ptr(b["r"]), ptr(b["c"]), ptr(C), c_longlong(m * n),
                                                            c_int(n), ptr(b["pdev"])], stream=stream)
        mark("recon")
        if prof is not None:
            torch.cuda.synchronize()
            for (n0, e0), (n1, e1) in zip(evs[:-1], evs[1:]):
                prof[n1] = prof.get(n1, 0.0) + e0.elapsed_time(e1) / 1e3
        return C

    def _capture(self, fn):
        """Canonical recipe: eager warm-up on a side stream, then capture on that stream; returns the CUDAGraph."""
        st = torch.cuda.Stream()
        with torch.cuda.stream(st):
            fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=st):
            fn()
        return g

    def core_graph(self, A, B, b, N, C):
        key = ("core", A.shape[0], B.shape[1], A.shape[1], N, A.data_ptr(), B.data_ptr(), C.data_ptr())
        g = self.graphs.get(key)
        if g is None:
            g = self._capture(lambda: self.core(A, B, b, N, C))
            self.graphs[key] = g
        g.replay()

    # ---- the rule on the device (same mathematics as choose(); no host arithmetic) -------------------------------------------
    def _rule_device(self, A, B, b, tau, safety, force_N=None, force_p=None):
        m, k = A.shape
        n = B.shape[1]
        f64 = torch.float64
        nuA, nuB = torch.sqrt(b["nA"].max()), torch.sqrt(b["nB"].max())
        sk = math.sqrt(k) / 2
        Mv = self.Mvec
        a_, bq = nuA * nuB, sk * (nuA + nuB)
        cq = sk * sk - (Mv - 2) / (2 * (1 + 1e-9))
        y = torch.where(a_ > 0, (-bq + torch.sqrt(torch.clamp(bq * bq - 4 * a_ * cq, min=0))) / (2 * torch.clamp(a_, min=1e-300)),
                        -cq / torch.clamp(bq, min=1e-300))
        p0 = torch.floor(torch.log2(torch.clamp(y, min=1e-300)))
        cand = torch.stack([p0 + 1, p0, p0 - 1, p0 - 2])
        fits = 2 * (torch.exp2(cand) * nuA + sk) * (torch.exp2(cand) * nuB + sk) * (1 + 1e-9) + 2 <= Mv[None, :]
        pv = torch.where(fits, cand, torch.full_like(cand, -1.0)).max(dim=0).values.clamp(0, 249)
        if force_N is not None:
            fb = b["flag"][0] != 0
            psel = pv[force_N - 6]
            if force_p is not None:                       # a table-supplied p above what the data range allows is unsafe -> fallback
                fb = fb | (psel < force_p)
                psel = torch.minimum(psel, torch.full_like(psel, float(force_p)))
            Nsel = torch.where(fb, torch.zeros_like(b["flag"][0]), torch.full_like(b["flag"][0], force_N))
            b["dec"].copy_(torch.stack([Nsel.to(torch.int32), psel.to(torch.int32), fb.to(torch.int32), torch.zeros_like(Nsel).to(torch.int32)]))
            b["pdev"].copy_(b["dec"][1:2])
            return
        key = ("probe", m, n)
        if key not in self.probe:
            rs = np.random.RandomState(0)
            Ir = rs.choice(m, min(64, m), replace=False)
            Jr = rs.choice(n, min(64, n), replace=False)
            self.probe[key] = (torch.as_tensor(Ir, device=DEV), torch.as_tensor(Jr, device=DEV))
        Ir, Jr = self.probe[key]
        r = b["r"].to(f64)
        c = torch.where(b["c"] <= NEG // 2, torch.zeros_like(b["c"]), b["c"]).to(f64)
        sA, sB = b["sA"], b["sB"]
        hard_r = torch.where(sA > 0, torch.exp2(r) * k / sA, torch.zeros_like(sA))
        hard_c = torch.where(sB > 0, torch.exp2(c) * k / sB, torch.zeros_like(sB))
        I = torch.cat([Ir, torch.topk(hard_r, 32).indices])
        J = torch.cat([Jr, torch.topk(hard_c, 32).indices])
        Dh = torch.abs(A.index_select(0, I)) @ torch.abs(B.index_select(1, J))
        rI, cJ = r[I][:, None], c[J][None, :]
        F1 = 0.5 * (torch.exp2(rI) * sB[J][None, :] + torch.exp2(cJ) * sA[I][:, None])
        F2 = 0.25 * k * torch.exp2(rI + cJ)
        thr = tau * Dh / safety
        xmax = 2 * thr / (F1 + torch.sqrt(F1 * F1 + 4 * F2 * thr))
        need = torch.where(Dh > 0, -torch.log2(xmax), torch.full_like(xmax, -1e30))
        preq = torch.clamp(torch.ceil(need.max() + 1e-9), min=1.0, max=1000.0)
        okN = pv >= preq
        has = okN.any()
        idx = torch.argmax(okN.to(torch.int32))
        fb = (~has) | (b["flag"][0] != 0) | (preq > 249)
        Nsel = torch.where(fb, torch.zeros_like(idx), idx + 6)
        psel = torch.gather(pv, 0, idx.reshape(1)).reshape(())          # NOT pv[idx]: a 0-dim tensor index syncs (illegal in capture)
        b["dec"].copy_(torch.stack([Nsel.to(torch.int32), psel.to(torch.int32), fb.to(torch.int32), preq.to(torch.int32)]))
        b["pdev"].copy_(b["dec"][1:2])

    def _stats_rule(self, A, B, b, tau, safety, inside, force_N=None, force_p=None):
        self.stats(A, B, b, inside)
        self._rule_device(A, B, b, tau, safety, force_N, force_p)

    def run_fast(self, A, B, C, tau=2.0 ** -52, safety=2.0, inside=True, force_N=None, force_p=None):
        """Timed path: [graph: statistics + rule on the device] -> one host read of (N, p) -> [graph: residues, GEMMs, reconstruction]."""
        m, k = A.shape
        n = B.shape[1]
        b = self.buffers(m, n, k, 20)
        key = ("sr", m, n, k, A.data_ptr(), B.data_ptr(), tau, safety, inside, force_N, force_p)
        g = self.graphs.get(key)
        if g is None:
            g = self._capture(lambda: self._stats_rule(A, B, b, tau, safety, inside, force_N, force_p))
            self.graphs[key] = g
        g.replay()
        b["hdec"].copy_(b["dec"], non_blocking=True)
        torch.cuda.current_stream().synchronize()
        N, p, fb, preq = [int(x) for x in b["hdec"].tolist()]
        if fb or N == 0:
            torch.matmul(A, B, out=C)
            return C, {"fallback": True, "preq": preq}
        if b["N"] < N:
            b = self.buffers(m, n, k, N)
            g = self._capture(lambda: self._stats_rule(A, B, b, tau, safety, inside, force_N, force_p))
            self.graphs[key] = g
            g.replay()
            b["hdec"].copy_(b["dec"], non_blocking=True)
            torch.cuda.current_stream().synchronize()
        self.core_graph(A, B, b, N, C)
        return C, {"N": N, "p": p, "preq": preq}

    # ---- full call ---------------------------------------------------------------------------------------------------------
    def run(self, A, B, C=None, tau=2.0 ** -52, force_N=None, inside=True, graph=False, prof=None, seed=0, config=None, force_p=None):
        """config: dict {'N':.., 'p':..} (e.g. picked from a JSON table by pick_config) overrides the a-priori rule."""
        if config is not None:
            force_N, force_p = int(config['N']), config.get('p')
        m, k = A.shape
        n = B.shape[1]
        if graph:
            if C is None:
                C = torch.empty(m, n, dtype=torch.float64, device=DEV)
            return self.run_fast(A, B, C, tau=tau, inside=inside, force_N=force_N, force_p=force_p)
        b = self.buffers(m, n, k, force_N or 20)
        if C is None:
            C = torch.empty(m, n, dtype=torch.float64, device=DEV)
        t0 = time.perf_counter()
        if prof is not None:
            e0 = torch.cuda.Event(enable_timing=True)
            e0.record()
        self.stats(A, B, b, inside)
        if prof is not None:
            e1 = torch.cuda.Event(enable_timing=True)
            e1.record()
        N, p, info, h = self.choose(A, B, b, tau, force_N, seed, prof=prof)
        if prof is not None:
            prof["stats_gpu"] = e0.elapsed_time(e1) / 1e3
            prof["stats+rule_wall"] = time.perf_counter() - t0
        if N is None:
            info["fallback"] = True
            torch.matmul(A, B, out=C)
            return C, info
        if b["N"] < N:
            b = self.buffers(m, n, k, N)                   # grow (statistics are recomputed)
            self.stats(A, B, b, inside)
        if force_p is not None:
            if force_p > p:                                   # unsafe table entry for this data: refuse (native fallback)
                info["fallback"] = True
                info["config_rejected"] = True
                torch.matmul(A, B, out=C)
                return C, info
            p = int(force_p)
        b["pdev"].fill_(p)
        info.update({"N": N, "p": p, "L": _limbs_for(N), "stats": h})
        if graph:
            self.core_graph(A, B, b, N, C)                 # forced-N graph path (oracle variant)
        else:
            self.core(A, B, b, N, C, prof=prof)
        return C, info


# =====================================================================================================================
# inputs, correctness tests, accuracy check
# =====================================================================================================================
def worst_case_p(N, k):
    """Data-independent p for N moduli and inner dimension k (nu_A = nu_B = sqrt(k) is the worst case of the range bound)."""
    return pmax_bits(N, math.sqrt(k), math.sqrt(k), k)


def sigma_of(p):
    """Scaled contract of Lemma L1 for finite inputs with normal-range results (an absolute allowance of 2^-1075 applies below it): |C_hat - C|_ij <= sigma k 2^(r_i + c_j) + 2^-53 |C_hat_ij|
    (s^A_i < k 2^r_i, s^B_j < k 2^c_j, r_i/c_j = row/column exponents after inside scaling)."""
    return 2.0 ** -p * (1 + 2.0 ** (-p - 2))


def make_config_table(path, ks=(1024, 2048, 4096, 8192, 16384)):
    tab = {"schema": "ozaki_crt_int8_config_v1", "value_set": "int8 signed residues, INT32 accumulation, pairwise-coprime moduli <= 256",
           "moduli": MODS,
           "contract": "for finite inputs with normal-range results: |C_hat_ij - C_ij| <= sigma * k * 2^(r_i + c_j) + 2^-53 |C_hat_ij| + 2^-1075; 2^r_i > max_l |A_il| 2^gamma_l, "
                       "2^c_j > max_l |B_lj| 2^-gamma_l (row/column scales after inside scaling); exact CRT stage needs 2 Xmax(p) + 1 <= prod(moduli[:N])",
           "rigor": "Lemma L1 (elementary; independent re-check pending); sigma is a worst-case guarantee, not a sample",
           "entries": [{"k": k, "N": N, "p": worst_case_p(N, k), "sigma": sigma_of(worst_case_p(N, k))} for k in ks for N in range(6, NMAX + 1)]}
    json.dump(tab, open(path, "w"), indent=1)
    return tab


def pick_config(table, k, sigma_target):
    """Smallest-N entry whose table k >= k and whose guaranteed sigma <= sigma_target (theory selects the configuration)."""
    ks = sorted({e["k"] for e in table["entries"] if e["k"] >= k})
    for e in sorted((e for e in table["entries"] if e["k"] == ks[0]), key=lambda e: e["N"]):
        if e["sigma"] <= sigma_target:
            return {"N": e["N"], "p": e["p"], "sigma": e["sigma"], "table_k": e["k"]}
    return None


def full_certificate(E, A, B, info, tau=2.0 ** -52, safety=2.0):
    """FULL-matrix certificate of E_ij(p) <= tau D_ij / safety for ALL (i, j): D_ij = sum_l |A_il||B_lj| is bounded below on every entry by
    one FP16 tensor-core GEMM of the scaled magnitudes rounded DOWN (x (1 - 2^-10) before RN-to-fp16) with FP32 accumulation, derated by
    8 k 2^-24 (positive terms).  Entries whose lower bound is 0 cannot be certified (counted).  Assumption: the accumulation error of the
    tensor-core FP32 accumulation of non-negative terms is below 8 k 2^-24 relative."""
    m, k = A.shape
    n = B.shape[1]
    h = info["stats"]
    b = E.buffers(m, n, k, 20)
    E.stats(A, B, b, True)                    # gam must belong to the configuration being certified (inside scaling on)
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    r = torch.as_tensor(h["r"], device=DEV, dtype=torch.float64)
    c = torch.as_tensor(h["c"], device=DEV, dtype=torch.float64)
    g = torch.exp2(b["gam"].double())
    As = torch.abs(A) * (g[None, :] * torch.exp2(-r)[:, None]) * (1 - 2.0 ** -10)
    Bs = torch.abs(B) * (torch.exp2(-c)[None, :] / g[:, None]) * (1 - 2.0 ** -10)
    A16, B16 = As.to(torch.float16), Bs.to(torch.float16)
    del As, Bs
    try:
        Dl = torch.mm(A16, B16, out_dtype=torch.float32).double()
    except Exception:
        Dl = (A16.float() @ B16.float()).double()
    rI, cJ = torch.exp2(r)[:, None], torch.exp2(c)[None, :]
    D = Dl * (1 - 8 * k * 2.0 ** -24) * rI * cJ
    del Dl
    F1 = 0.5 * (rI * torch.as_tensor(h["sB"], device=DEV)[None, :] + cJ * torch.as_tensor(h["sA"], device=DEV)[:, None])
    F2 = 0.25 * k * rI * cJ
    thr = tau * D / safety
    mask = D > 0
    xmax = 2 * thr / (F1 + torch.sqrt(F1 * F1 + 4 * F2 * thr))
    need = torch.where(mask, -torch.log2(xmax), torch.full_like(xmax, -1e30)).max()
    e1.record()
    torch.cuda.synchronize()
    unc = int((~mask).sum())
    preq = int(math.ceil(float(need) + 1e-9))
    nuA, nuB = info["nuA"], info["nuB"]
    N_full = next((N for N in range(6, NMAX + 1) if pmax_bits(N, nuA, nuB, k) >= preq), None)
    N_s = info.get("N")
    return {"N_full": N_full, "preq_full": preq, "preq_sampled": info.get("p_req"), "uncertified_entries": unc, "cost_ms": e0.elapsed_time(e1),
            "N_sampled": N_s, "sampled_config_fully_certified": bool(unc == 0 and N_full is not None and N_s is not None and N_s >= N_full)}


def cmd_config():
    """Write the data-independent (theory-facing) configuration table and test that an engine run driven by it meets its guaranteed contract."""
    set_high_priority()
    
    path = os.path.join(OUT, "config_table.json")
    tab = make_config_table(path)
    E = OzakiCRT()
    res = {"table": path, "cases": []}
    n = 1024
    for fam in ("G", "X", "W"):
        for sigma_t in (2.0 ** -40, 2.0 ** -50):
            cfg = pick_config(json.load(open(path)), n, sigma_t)
            A, B = make_inputs_ext(n, fam)
            Cn = A @ B
            S = Sample(A, B, Cn)
            C, info = E.run(A, B, C=torch.empty_like(Cn), config=cfg)
            if info.get("fallback"):
                res["cases"].append({"fam": fam, "sigma_target": sigma_t, "config": cfg, "fallback": True})
                continue
            h, p = info["stats"], info["p"]
            Cs = C[S.tri][:, S.tci].cpu().numpy()
            err = np.abs(Cs - S.ref)
            lim = cfg["sigma"] * n * np.exp2(np.asarray(h["r"], dtype=np.float64)[S.ri][:, None] + np.asarray(h["c"], dtype=np.float64)[S.ci][None, :]) \
                + 2.0 ** -53 * (np.abs(Cs) + np.abs(S.ref)) * (1 + 1e-9)
            res["cases"].append({"fam": fam, "sigma_target": sigma_t, "config": cfg, "violations": int((err > lim).sum()),
                                 "max_err_over_limit": float((err / lim).max()), "p_used": p})
            print(res["cases"][-1], flush=True)
    res["all_ok"] = bool(all((not c.get("fallback")) and c["violations"] == 0 for c in res["cases"]))
    json.dump(res, open(os.path.join(OUT, "config_test.json"), "w"), indent=1)
    print("CONFIG", "PASS" if res["all_ok"] else "FAIL", flush=True)


def make_inputs_ext(n, fam, seed=0):
    """G, X, S, R as above; W = rotated ill-scaled family (docs/engine.md)."""
    if fam in ("G", "X", "S", "R"):
        return make_inputs(n, fam, seed)
    assert fam == "W"
    g = torch.Generator(device=DEV)
    g.manual_seed(1000 * seed + n + 7)
    At = torch.randn(n, n, device=DEV, dtype=torch.float64, generator=g)
    Bt = torch.randn(n, n, device=DEV, dtype=torch.float64, generator=g)
    d = torch.exp2(20.0 * torch.arange(n, device=DEV, dtype=torch.float64) / (n - 1))
    ar = torch.arange(n, device=DEV)
    idx = (ar[None, :] + ar[:, None]) % n            # symmetric: idx[i, l] = (l + i) mod n
    A = (At * d[None, :]).gather(1, idx)             # A[i, l] <- A[i, (l + i) mod n]
    B = (Bt / d[:, None]).gather(0, idx)             # B[l, j] <- B[(l + j) mod n, j]
    return A.contiguous(), B.contiguous()


def np_stats(A, B, inside=True):
    """Reference statistics (numpy) with the same definitions as the device kernels."""
    def ex(X):
        return np.where(X == 0, NEG, np.frexp(np.abs(X))[1]).astype(np.int64)
    eA, eB = ex(A), ex(B)
    a, bq = eA.max(axis=0), eB.max(axis=1)
    gam = np.where((a <= NEG // 2) | (bq <= NEG // 2) | (not inside), 0, (bq - a) >> 1)
    r = np.where(eA <= NEG // 2, NEG, eA + gam[None, :]).max(axis=1)
    r = np.where(r <= NEG // 2, 0, r)
    c = np.where(eB <= NEG // 2, NEG, eB - gam[:, None]).max(axis=0)
    c = np.where(c <= NEG // 2, 0, c)
    sA = (np.abs(A) * np.exp2(gam)[None, :]).sum(axis=1)
    sB = (np.abs(B) * np.exp2(-gam)[:, None]).sum(axis=0)
    nA = ((np.abs(A) * np.exp2(gam[None, :] - r[:, None])) ** 2).sum(axis=1)
    nB = ((np.abs(B) * np.exp2(-gam[:, None] - c[None, :])) ** 2).sum(axis=0)
    return {"gam": gam, "r": r, "c": c, "sA": sA, "sB": sB, "nA": nA, "nB": nB}


def _rnd(q):
    """Round-half-away-from-zero of a Fraction to a Python int."""
    from fractions import Fraction
    s = -1 if q < 0 else 1
    return s * math.floor(abs(q) + Fraction(1, 2))


def expected_core(A, B, gam, r, c, p):
    """Exact CRT-stage result in Python integers: correctly rounded float(A'B' 2^-(2p - r_i - c_j))."""
    from fractions import Fraction
    m, k = A.shape
    n = B.shape[1]
    two = Fraction(2)
    Ai = np.empty((m, k), dtype=object)
    Bi = np.empty((k, n), dtype=object)
    for i in range(m):
        for l in range(k):
            Ai[i, l] = _rnd(Fraction(float(A[i, l])) * two ** int(p - r[i] + gam[l]))
    for l in range(k):
        for j in range(n):
            Bi[l, j] = _rnd(Fraction(float(B[l, j])) * two ** int(p - c[j] - gam[l]))
    X = Ai.dot(Bi)
    out = np.empty((m, n))
    for i in range(m):
        for j in range(n):
            out[i, j] = float(Fraction(int(X[i, j])) * two ** int(-(2 * p - r[i] - c[j])))
    return out


def cmd_selftest():
    set_high_priority()
    
    E = OzakiCRT()
    rng = np.random.RandomState(5)
    res = {"cases": [], "all_ok": True}

    def gen(kind, m, k, n):
        if kind == "gauss":
            A, B = rng.randn(m, k), rng.randn(k, n)
        elif kind == "graded":
            A = (rng.rand(m, k) - 0.5) * np.exp(2 * rng.randn(m, k))
            B = (rng.rand(k, n) - 0.5) * np.exp(2 * rng.randn(k, n))
        else:  # wide: zeros, tiny and huge entries, exact powers of two
            A, B = rng.randn(m, k), rng.randn(k, n)
            A *= 2.0 ** rng.randint(-90, 90, size=(m, k)) * (rng.rand(m, k) > 0.25)
            B *= 2.0 ** rng.randint(-90, 90, size=(k, n)) * (rng.rand(k, n) > 0.25)
            A[0, :3] = [1.0, -2.0 ** -40, 5e-324 * 3]
            B[:3, 0] = [-1.0, 2.0 ** 30, 1.0]
        return np.ascontiguousarray(A), np.ascontiguousarray(B)

    for (m, k, n) in ((64, 64, 64), (32, 96, 64)):
        for kind in ("gauss", "graded", "wide"):
            An, Bn = gen(kind, m, k, n)
            A = torch.from_numpy(An).to(DEV)
            B = torch.from_numpy(Bn).to(DEV)
            b = E.buffers(m, n, k, 32)
            E.stats(A, B, b, True)
            ref = np_stats(An, Bn, True)
            st_ok = (np.array_equal(b["gam"].cpu().numpy(), ref["gam"]) and np.array_equal(b["r"].cpu().numpy(), ref["r"])
                     and np.array_equal(np.where(b["c"].cpu().numpy() <= NEG // 2, 0, b["c"].cpu().numpy()), ref["c"])
                     and np.allclose(b["sA"].cpu().numpy(), ref["sA"], rtol=1e-11, atol=0)
                     and np.allclose(b["sB"].cpu().numpy(), ref["sB"], rtol=1e-11, atol=0)
                     and np.allclose(b["nA"].cpu().numpy(), ref["nA"], rtol=1e-11, atol=0)
                     and np.allclose(b["nB"].cpu().numpy(), ref["nB"], rtol=1e-11, atol=0))
            res["cases"].append({"shape": [m, k, n], "kind": kind, "stats_ok": bool(st_ok)})
            res["all_ok"] &= bool(st_ok)
            for Nsel in (8, 14, 20, 30):
                if _limbs_for(Nsel) > 9:
                    continue
                Nn, pm, info, h = E.choose(A, B, b, force_N=Nsel)
                for p in sorted({pm, max(pm - 9, 6)}):
                    b["pdev"].fill_(p)
                    C = torch.empty(m, n, dtype=torch.float64, device=DEV)
                    E.core(A, B, b, Nsel, C)
                    torch.cuda.synchronize()
                    exp = expected_core(An, Bn, ref["gam"], ref["r"], ref["c"], p)
                    got = C.cpu().numpy()
                    ok = bool(np.array_equal(got.view(np.uint64), exp.view(np.uint64)))
                    rec = {"shape": [m, k, n], "kind": kind, "N": Nsel, "p": int(p), "exact_bitwise": ok}
                    if not ok:
                        bad = np.argwhere(got.view(np.uint64) != exp.view(np.uint64))
                        rec["n_bad"] = int(len(bad))
                        rec["first_bad"] = [int(bad[0][0]), int(bad[0][1]), float(got[tuple(bad[0])]), float(exp[tuple(bad[0])])]
                    res["cases"].append(rec)
                    res["all_ok"] &= ok
                    print(rec, flush=True)
    # graph replay must equal eager
    m = k = n = 64
    An, Bn = gen("graded", m, k, n)
    A, B = torch.from_numpy(An).to(DEV), torch.from_numpy(Bn).to(DEV)
    C1, i1 = E.run(A, B)
    C2 = torch.empty_like(C1)
    E.run(A, B, C=C2, graph=True)
    E.run(A, B, C=C2, graph=True)
    torch.cuda.synchronize()
    res["graph_equals_eager"] = bool(torch.equal(C1, C2))
    res["all_ok"] &= res["graph_equals_eager"]
    print("graph == eager:", res["graph_equals_eager"], " N,p =", i1.get("N"), i1.get("p"), flush=True)
    json.dump(res, open(os.path.join(OUT, "selftest.json"), "w"), indent=1)
    print("SELFTEST", "PASS" if res["all_ok"] else "FAIL", flush=True)


def audit_bound(S, info, k, C, scale=1.0):
    """Lemma L1 audit on the shared sample: measured |C_hat - C_ref| against E_ij(p) + 2^-53 (|C_hat| + |C_ref|) (must never exceed;
    the second rounding term is the rounding of the correctly rounded reference C_ref = fl(C))."""
    h, p = info["stats"], info["p"]
    Eb = scale * bound_matrix(h["r"], h["c"], h["sA"], h["sB"], k, p, S.ri, S.ci)
    Cs = C[S.tri][:, S.tci].cpu().numpy()
    err = np.abs(Cs - S.ref)
    lim = Eb * (1 + 1e-9) + 2.0 ** -53 * (np.abs(Cs) + np.abs(S.ref)) * (1 + 1e-9)   # result rounding + rounding of the reference itself
    nz = err > 0
    return {"max_err_over_bound": float((err / lim).max()), "violations": int((err > lim).sum()),
            "median_bound_over_err": float(np.median(Eb[nz] / err[nz])) if nz.any() else None,
            "max_err_over_D": float((err / S.absab).max())}


def cmd_check(n, fams):
    set_high_priority()
    
    E = OzakiCRT()
    allres = {}
    for fam in fams:
        A, B = make_inputs_ext(n, fam)
        Cn = A @ B
        S = Sample(A, B, Cn)
        en = S.errors(Cn)
        match = lambda e: e["E1"] <= en["E1"] and e["E2"] <= en["E2"]
        row = {"native": en}
        C = torch.empty(n, n, dtype=torch.float64, device=DEV)
        for inside in ((True, False) if fam in ("S", "X", "W") else (True,)):
            Cc, info = E.run(A, B, C=C, inside=inside)
            key = "cert" if inside else "cert_no_inside"
            if inside:
                info_c = info
            if info.get("fallback"):
                row[key] = {"fallback": True}
                continue
            e = S.errors(Cc)
            row[key] = {"N": info["N"], "p": info["p"], "err": e, "matched": bool(match(e)), "contract_tau_2^-52_met": bool(e["E2"] <= 2.0 ** -52),
                        "worst_E_over_D": info.get("worst_ratio_E_over_D"), "audit": audit_bound(S, info, n, Cc)}
            print(fam, key, {k_: row[key][k_] for k_ in ("N", "p", "matched")}, e, row[key]["audit"], flush=True)
        if "N" in row.get("cert", {}):
            row["full_cert"] = full_certificate(E, A, B, info_c)
            print(fam, "full-matrix certificate", row["full_cert"], flush=True)
        # full-matrix cross-check (the 49x49 sample cannot see a localised error): compare EVERY entry with the N=32 result
        # (its quantisation error is ~2^-100 relative to |A||B|; the CRT stage is bit-exact by the selftest) and with native
        if "N" in row.get("cert", {}):
            Ccert, _ = E.run(A, B, C=torch.empty_like(C))
            Cref, _ = E.run(A, B, C=torch.empty_like(C), force_N=32)
            Dm = torch.abs(A) @ torch.abs(B)
            pos = Dm > 0
            den = Dm.clamp_min(1e-300)
            row["full_matrix"] = {"E2_cert_vs_N32": float(((Ccert - Cref).abs() / den)[pos].max()),
                                  "E2_native_vs_N32": float(((Cn - Cref).abs() / den)[pos].max()),
                                  "E1_cert_vs_N32": float((Ccert - Cref).abs().max() / Cref.abs().max()),
                                  "E1_native_vs_N32": float((Cn - Cref).abs().max() / Cref.abs().max())}
            row["full_matrix"]["cert_not_worse_than_native"] = bool(row["full_matrix"]["E2_cert_vs_N32"] <= row["full_matrix"]["E2_native_vs_N32"]
                                                                    and row["full_matrix"]["E1_cert_vs_N32"] <= row["full_matrix"]["E1_native_vs_N32"])
            bb_ = E.buffers(n, n, n, 20)
            E.stats(A, B, bb_, True)
            ref = np_stats(A.cpu().numpy(), B.cpu().numpy(), True)
            hi_, hd_ = bb_["bi"].cpu().numpy(), bb_["bd"].cpu().numpy()
            c_dev = np.where(hi_[n:2 * n] <= NEG // 2, 0, hi_[n:2 * n])
            row["stats_equal_numpy"] = bool(np.array_equal(hi_[:n], ref["r"]) and np.array_equal(c_dev, ref["c"])
                                           and np.allclose(hd_[:n], ref["sA"], rtol=1e-11, atol=0) and np.allclose(hd_[2 * n:3 * n], ref["sB"], rtol=1e-11, atol=0)
                                           and np.allclose(hd_[n:2 * n], ref["nA"], rtol=1e-11, atol=0) and np.allclose(hd_[3 * n:], ref["nB"], rtol=1e-11, atol=0))
            print(fam, "full-matrix", row["full_matrix"], "stats==numpy", row["stats_equal_numpy"], flush=True)
        # oracle N: smallest N whose measured errors match native
        row["oracle"] = None
        for N in range(6, NMAX + 1):
            Cq, iq = E.run(A, B, C=C, force_N=N)
            e = S.errors(Cq)
            if match(e):
                row["oracle"] = {"N": N, "p": iq["p"], "err": e}
                break
        print(fam, "oracle", row["oracle"], "native", en, flush=True)
        if "N" in row.get("cert", {}):
            Cg = torch.empty_like(C)
            E.run(A, B, C=Cg, graph=True)
            _, ig = E.run(A, B, C=Cg, graph=True)
            Ce, ie = E.run(A, B, C=C)
            torch.cuda.synchronize()
            row["graph_equals_eager"] = bool(torch.equal(Cg, Ce))
            row["device_rule_equals_host_rule"] = bool(ig.get("N") == ie.get("N") and ig.get("p") == ie.get("p"))
            row["device_rule"] = {"N": ig.get("N"), "p": ig.get("p"), "preq": ig.get("preq"), "host_preq": ie.get("p_req")}
            for _ in range(10):
                E.run(A, B, C=C, force_N=row["cert"]["N"])
            prof = {}
            E.run(A, B, C=C, force_N=row["cert"]["N"], prof=prof)
            row["prof_s"] = prof
            print(fam, "graph==eager", row["graph_equals_eager"], "device rule == host rule", row["device_rule_equals_host_rule"], row["device_rule"], "prof", {k_: round(v, 5) for k_, v in prof.items()}, flush=True)
        allres[fam] = row
        del A, B, Cn, C
        torch.cuda.empty_cache()
    json.dump(allres, open(os.path.join(OUT, f"check_{n}.json"), "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


# =====================================================================================================================
# the race (see docs/engine.md)
# =====================================================================================================================
GIB = 1 << 30
VD = "dyn_s1_ws0_cap0"          # out-of-the-box call
WS_BYTES = 2 << 30              # used by the controls; the race allocates 8 GiB once and passes 2 or 8 GiB as the size
GROUP_N = int(os.environ.get("OZAKI_GROUP_N", "8192"))   # split brackets from this n on


def build_variants():
    """Vendor variants (27 per cell; see docs/engine.md). cls: 'apriori' = the library's own dynamic rule (finite-input
    declaration and workspace are static settings); 'oracle' = needs a calibration against the native error (fixed mantissa, dynamic offset)."""
    v = []
    for s_ in (1, 2):
        for wsn, wsb in (("0", 0), ("2", 2 * GIB), ("8", 8 * GIB)):
            for cap in (0, 128):
                v.append((f"dyn_s{s_}_ws{wsn}_cap{cap}", dict(strategy=s_, ws=wsb, cap=cap, cls="apriori")))
    for s_ in (1, 2):
        for wsn, wsb in (("2", 2 * GIB), ("8", 8 * GIB)):
            for cap in (0, 128):
                v.append((f"dyn_s{s_}_ws{wsn}_cap{cap}_svnone", dict(strategy=s_, ws=wsb, cap=cap, sv=0, cls="apriori")))
    for bt in (40, 48, 56, 64):
        v.append((f"fix_b{bt}", dict(strategy=2, ws=8 * GIB, fixed=bt, sv=0, cls="oracle")))
    for off in (-8, -16, -24):
        v.append((f"off{off}", dict(strategy=2, ws=8 * GIB, cap=128, sv=0, offset=off, cls="oracle")))
    return v


VARIANTS = build_variants()
VCLS = {name: kw["cls"] for name, kw in VARIANTS}


def cpu_pct_fast():
    try:
        import psutil
        return psutil.cpu_percent(interval=None)
    except Exception:
        return None


def env_snapshot():
    d = {"gpu": gpu_state(), "cpu_pct": None, "gpu_util": None}
    try:
        import psutil
        d["cpu_pct"] = psutil.cpu_percent(interval=0.5)
    except Exception:
        pass
    try:
        d["gpu_util"] = int(pynvml.nvmlDeviceGetUtilizationRates(_NVH).gpu)
    except Exception:
        pass
    return d


def _primer():
    """A tiny kernel + sync.  The first submission after a long kernel (native DGEMM) is delayed by 30-90 ms in about 60% of rounds (WDDM
    scheduling); the delay lands on this primer instead of on the first timed method."""
    torch.cuda.synchronize()
    torch.empty(1, device=DEV).fill_(1.0)
    torch.cuda.synchronize()


def wait_cool(target=74, tmax=150.0):
    """Block until the GPU temperature (NVML) is <= target C or tmax seconds passed; returns (seconds waited, temperature, reached)."""
    assert _NVH is not None, "NVML unavailable: the cool-down wait cannot run"
    t0 = time.time()
    temp = gpu_state().get("temp_c", 0)
    while temp > target and time.time() - t0 < tmax:
        time.sleep(0.5)
        temp = gpu_state().get("temp_c", 0)
    return time.time() - t0, temp, temp <= target


GATE_C = None if os.environ.get("OZAKI_GATE_C", "74").lower() in ("off", "none") else float(os.environ.get("OZAKI_GATE_C", "74"))


def run_block(fns, ref, seed, min_valid=7, max_attempts=30, warm=2, gate_c="env", gate_max_s=150.0):
    """Interleaved rounds : [ref pre] [others in a fresh random order] [ref post]; a round is VALID iff the two ref
    timings agree within 5%.  Every round (warm-up rounds included) starts only after the GPU has cooled to gate_c
    degrees (NVML poll, at most gate_max_s), so that the round runs in the flat boost region; gate_c=None disables the wait."""
    if gate_c == "env":
        gate_c = GATE_C
    names = [k for k in fns if k != ref]
    rng = np.random.RandomState(seed)
    valid, invalid = [], []
    attempts = r = 0
    while True:
        gate = wait_cool(gate_c, gate_max_s) if gate_c is not None else (0.0, gpu_state().get("temp_c"), True)
        pre = ev_time(fns[ref])
        st_pre = gpu_state()
        _primer()
        ts, st = {}, {}
        for nm in rng.permutation(names):
            ts[str(nm)] = ev_time(fns[str(nm)])
            st[str(nm)] = gpu_state()
        post = ev_time(fns[ref])
        st_post = gpu_state()
        r += 1
        if r <= warm:
            continue
        attempts += 1
        drift = abs(post - pre) / min(pre, post)
        rnd = {"pre": pre, "post": post, "drift": drift, "t": ts, "state": st, "state_pre": st_pre, "state_post": st_post,
               "cpu_pct": cpu_pct_fast(), "gate_waited_s": gate[0], "gate_start_temp": gate[1], "gate_reached": bool(gate[2])}
        (valid if drift <= 0.05 else invalid).append(rnd)
        if len(valid) >= min_valid or attempts >= max_attempts:
            break
    return {"valid": valid, "invalid": invalid, "attempts": attempts, "warm_rounds": r - attempts}


def summarize(blk, names, ref):
    R = blk["valid"]
    out = {}
    for nm in [ref] + names:
        if nm == ref:
            t = [0.5 * (x["pre"] + x["post"]) for x in R]
            ratio = [1.0] * len(R)
        else:
            t = [x["t"][nm] for x in R]
            ratio = [x["t"][nm] / (0.5 * (x["pre"] + x["post"])) for x in R]
        out[nm] = {"med_s": _q(t, 50), "q1_s": _q(t, 25), "q3_s": _q(t, 75), "ratio_med": _q(ratio, 50),
                   "ratio_q1": _q(ratio, 25), "ratio_q3": _q(ratio, 75), "speedup_med": 1.0 / _q(ratio, 50)}
    return out


def overlap(a, b):
    return max(a["ratio_q1"], b["ratio_q1"]) <= min(a["ratio_q3"], b["ratio_q3"])


def is_matched(e, en):
    return bool(e["E1"] <= en["E1"] and e["E2"] <= en["E2"])


def decide_cell(R, summ, fam, dyn_names, cert_ok, fix_name, has_cert, has_oracle, vend_all=(), Rmap=None, vkey="med_s"):
    """H1a / H1b / oracle comparison of plus the paired-ratio table.  R = valid rounds of the (single) bracket; with
    several brackets (split brackets) Rmap maps every method to the valid-round list of ITS bracket: a pair timed in the same bracket is
    compared round by round ('paired'); a pair from different brackets is compared through the native-normalised ratios ('unpaired',
    q1 = own_q1 / v_q3, q3 = own_q3 / v_q1)."""
    if Rmap is None:
        Rmap = {k: R for k in summ if k != "native"}

    def pair(a, b):
        if b == "native":
            rr = [x["t"][a] / (0.5 * (x["pre"] + x["post"])) for x in Rmap[a]]
            return _q(rr, 50), _q(rr, 25), _q(rr, 75), "paired"
        if Rmap.get(a) is not None and Rmap.get(a) is Rmap.get(b):
            rr = [x["t"][a] / x["t"][b] for x in Rmap[a]]
            return _q(rr, 50), _q(rr, 25), _q(rr, 75), "paired"
        sa, sb = summ[a], summ[b]
        return sa["ratio_med"] / sb["ratio_med"], sa["ratio_q1"] / sb["ratio_q3"], sa["ratio_q3"] / sb["ratio_q1"], "unpaired"

    dec = {}
    if has_cert:
        if dyn_names:
            vstar = min(dyn_names, key=lambda x: summ[x][vkey])
            dec["vstar_kind"] = "vendor dynamic (a-priori class)"
        else:
            vstar = "native"
            dec["vstar_kind"] = "native (no dynamic vendor variant matched accuracy)"
        med, q1, q3, pairing = pair("own_cert", vstar)
        ov = overlap(summ["own_cert"], summ[vstar])
        dec.update({"vstar_dyn": vstar, "rho_med": med, "rho_q1": q1, "rho_q3": q3, "pairing": pairing,
                    "own_cert_matched": bool(cert_ok), "iqr_overlap_vs_vstar": bool(ov),
                    "H1a": bool(cert_ok and med <= 0.90 and not ov)})
        if fam in ("S", "W") and "vend:" + VD in summ:
            vd = summ["vend:" + VD]
            dec["Vd_speedup_over_native"] = vd["speedup_med"]
            dec["H1b"] = bool(vd["speedup_med"] <= 1.05 and cert_ok and summ["own_cert"]["ratio_med"] < 0.77
                              and summ["own_cert"]["ratio_q3"] < 0.77)
    if fix_name and has_oracle and "own_oracle" in summ and fix_name in summ:
        dec["oracle_rho_med_vs_fixed"] = pair("own_oracle", fix_name)[0]
        dec["oracle_pairing"] = pair("own_oracle", fix_name)[3]
    tab = {}
    for o in ("own_cert", "own_oracle"):
        if o in summ:
            for v in vend_all:
                if v in summ:
                    m_, a_, b_, pr = pair(o, v)
                    tab[f"{o}/{v}"] = [m_, a_, b_, pr]
    dec["rho_table"] = tab
    return dec


def cmd_instr():
    """Positive and negative controls for every detector that supports a claim (pre-run controls)."""
    set_high_priority()
    res = {}
    # 1. drift detector: alternating native duration must make every round INVALID (positive); constant duration VALID (negative)
    global ev_time
    real_ev = ev_time
    ev_time = lambda fn: fn()                # the fake methods return their 'measured' seconds directly
    cnt = {"n": 0}

    def native_drift():
        cnt["n"] += 1
        return 1.0 if cnt["n"] % 2 == 0 else 1.10        # pre/post differ by 10% -> every round must be invalid

    try:
        bad = run_block({"native": native_drift, "x": lambda: 0.5}, "native", 1, min_valid=3, max_attempts=6, warm=1, gate_c=None)
        good = run_block({"native": lambda: 1.0, "x": lambda: 0.5}, "native", 1, min_valid=3, max_attempts=6, warm=1, gate_c=None)
        edge = run_block({"native": (lambda c=[0]: (c.__setitem__(0, c[0] + 1), 1.0 if c[0] % 2 else 1.049)[1]), "x": lambda: 0.5}, "native", 1,
                         min_valid=3, max_attempts=6, warm=1, gate_c=None)          # 4.9% < 5% -> valid
    finally:
        ev_time = real_ev
    res["drift_positive_control"] = {"valid": len(bad["valid"]), "invalid": len(bad["invalid"]),
                                     "ok": len(bad["valid"]) == 0 and len(bad["invalid"]) > 0}
    res["drift_negative_control"] = {"valid": len(good["valid"]), "invalid": len(good["invalid"]),
                                     "ok": len(good["valid"]) >= 3 and len(good["invalid"]) == 0,
                                     "edge_4.9pct_valid": len(edge["valid"]) >= 3}
    res["drift_negative_control"]["ok"] = bool(res["drift_negative_control"]["ok"] and res["drift_negative_control"]["edge_4.9pct_valid"])

    # 2. decision logic on synthetic rounds
    def synth(tv, to, jit, nrounds=9, nat=1.0):
        rng = np.random.RandomState(0)
        R = [{"pre": nat, "post": nat, "t": {"vend:" + VD: tv * (1 + jit * rng.randn()), "own_cert": to * (1 + jit * rng.randn())}}
             for _ in range(nrounds)]
        return R, summarize({"valid": R}, ["vend:" + VD, "own_cert"], "native")

    dc = {}
    for label, (tv, to, jit, fam, exp_a) in {"win_nonoverlap": (0.5, 0.3, 0.02, "G", True), "tie": (0.5, 0.49, 0.05, "G", False),
                                              "slower": (0.5, 0.6, 0.02, "G", False)}.items():
        R, summ = synth(tv, to, jit)
        d = decide_cell(R, summ, fam, ["vend:" + VD], True, None, True, False)
        dc[label] = {"H1a": d["H1a"], "expected": exp_a, "ok": d["H1a"] == exp_a, "rho_med": round(d["rho_med"], 3)}
    # deterministic overlap case: a consistent 15% paired win (rho = 0.85) whose native-normalised IQRs overlap -> H1a must be False
    vt = [0.40, 0.45, 0.50, 0.55, 0.60, 0.45, 0.55]
    R = [{"pre": 1.0, "post": 1.0, "t": {"vend:" + VD: x, "own_cert": 0.85 * x}} for x in vt]
    summ = summarize({"valid": R}, ["vend:" + VD, "own_cert"], "native")
    d = decide_cell(R, summ, "G", ["vend:" + VD], True, None, True, False)
    dc["rho_0.85_but_iqr_overlap"] = {"H1a": d["H1a"], "rho_med": d["rho_med"], "overlap": d["iqr_overlap_vs_vstar"], "ok": d["H1a"] is False and d["iqr_overlap_vs_vstar"] is True}
    for label, (tv, to, fam, cert_ok, expected) in {"H1b_true": (0.98, 0.2, "S", True, True), "H1b_false_slow_own": (0.98, 0.8, "S", True, False),
                                                     "H1b_false_vendor_not_fallback": (0.5, 0.2, "S", True, False),
                                                     "H1b_false_unmatched_own": (0.98, 0.2, "S", False, False)}.items():
        R, summ = synth(tv, to, 0.02)
        d = decide_cell(R, summ, fam, ["vend:" + VD], cert_ok, None, True, False)
        dc[label] = {"H1b": d["H1b"], "expected": expected, "ok": d["H1b"] == expected}
    R, summ = synth(0.98, 0.2, 0.02)
    dc["H1b_absent_on_G"] = {"has_H1b": "H1b" in decide_cell(R, summ, "G", ["vend:" + VD], True, None, True, False)}
    dc["H1b_absent_on_G"]["ok"] = not dc["H1b_absent_on_G"]["has_H1b"]
    R, summ = synth(0.5, 0.3, 0.02)
    dc["unmatched_own_blocks_H1a"] = {"H1a": decide_cell(R, summ, "G", ["vend:" + VD], False, None, True, False)["H1a"]}
    dc["unmatched_own_blocks_H1a"]["ok"] = dc["unmatched_own_blocks_H1a"]["H1a"] is False
    res["decision_controls"] = dc
    dc_ok = all(v["ok"] for v in dc.values())
    # 2b. brackets: paired inside a bracket, unpaired (ratio-based) across brackets, V* by native-normalised ratio
    rng_ = np.random.RandomState(1)
    R1 = [{"pre": 1.0, "post": 1.0, "t": {"own_cert": 0.10 * (1 + 0.01 * rng_.randn()), "vend:a": 0.20 * (1 + 0.01 * rng_.randn())}} for _ in range(9)]
    R2 = [{"pre": 1.0, "post": 1.0, "t": {"vend:" + VD: 0.98 * (1 + 0.01 * rng_.randn()), "vend:b": 0.15 * (1 + 0.01 * rng_.randn())}} for _ in range(9)]
    summ2 = {"native": summarize({"valid": R1}, ["own_cert", "vend:a"], "native")["native"]}
    summ2.update(summarize({"valid": R1}, ["own_cert", "vend:a"], "native"))
    summ2.update(summarize({"valid": R2}, ["vend:" + VD, "vend:b"], "native"))
    Rm = {"own_cert": R1, "vend:a": R1, "vend:" + VD: R2, "vend:b": R2}
    d_a = decide_cell(R1, summ2, "S", ["vend:a", "vend:" + VD], True, None, True, False, vend_all=["vend:a", "vend:b"], Rmap=Rm, vkey="ratio_med")
    d_b = decide_cell(R1, summ2, "G", ["vend:a", "vend:b"], True, None, True, False, vend_all=["vend:a", "vend:b"], Rmap=Rm, vkey="ratio_med")
    res["bracket_controls"] = {"paired_in_bracket": d_a["pairing"] == "paired" and d_a["vstar_dyn"] == "vend:a",
                               "unpaired_across_brackets": d_b["pairing"] == "unpaired" and d_b["vstar_dyn"] == "vend:b" and d_b["H1a"] is True,
                               "H1b_uses_Vd_from_other_bracket": bool(d_a.get("H1b"))}
    # 3. matched-accuracy predicate and the sampled error estimator: exact / fp32-rounded / one perturbed entry
    n = 256
    A, B = make_inputs_ext(n, "G")
    Cn = A @ B
    S = Sample(A, B, Cn)
    en = S.errors(Cn)
    C32 = Cn.float().double()
    Cp = Cn.clone()
    Cp[int(S.ri[3]), int(S.ci[5])] += 1e-9
    res["match_controls"] = {"native_vs_itself_matched": is_matched(S.errors(Cn), en),
                             "fp32_rounded_not_matched": not is_matched(S.errors(C32), en),
                             "perturbed_entry_not_matched": not is_matched(S.errors(Cp), en),
                             "perturbed_entry_E1": S.errors(Cp)["E1"], "fp32_E1": S.errors(C32)["E1"], "native_E1": en["E1"]}
    # 4. vendor knobs actually act (positive controls): fixed-mantissa 40 bits is less accurate than 64; emulation is faster than native
    n = 2048
    A, B = make_inputs_ext(n, "G")
    Cn = A @ B
    S = Sample(A, B, Cn)
    wsb = torch.empty(WS_BYTES, dtype=torch.uint8, device=DEV)
    out = {}
    Cv = torch.empty_like(Cn)
    for name, kw in (("fix_b40", dict(fixed_bits=40)), ("fix_b64", dict(fixed_bits=64)), ("dyn_ws", dict()), ("native13", None)):
        v = Vendor()
        if kw is None:
            v.configure(False)
        else:
            v.configure(True, strategy=2, ws=wsb, **kw)
        st = v.dgemm(A, B, Cv)
        torch.cuda.synchronize()
        ts = [ev_time(lambda: v.dgemm(A, B, Cv)) for _ in range(5)]
        out[name] = {"status": st, "err": S.errors(Cv), "t_med": _q(ts, 50)}
    tn = _q([ev_time(lambda: torch.matmul(A, B, out=Cv)) for _ in range(5)], 50)
    out["torch_native_t_med"] = tn
    res["vendor_knob_controls"] = {"b40_less_accurate_than_b64": out["fix_b40"]["err"]["E1"] > out["fix_b64"]["err"]["E1"],
                                   "emulation_faster_than_native": out["dyn_ws"]["t_med"] < 0.8 * tn,
                                   "native13_close_to_torch_native": 0.6 < out["native13"]["t_med"] / tn < 1.6, "detail": out}
    vk = res["vendor_knob_controls"]
    # 5. timed-path fallback flag (M1): a NaN input must be reported as a fallback; a normal input must not
    n = 1024
    A, B = make_inputs_ext(n, "G")
    E = OzakiCRT()
    Cx = torch.empty(n, n, dtype=torch.float64, device=DEV)
    _, i_ok = E.run(A, B, C=Cx, graph=True)
    An = A.clone()
    An[3, 5] = float("nan")
    _, i_nan = E.run(An, B, C=Cx, graph=True)
    res["fallback_flag_controls"] = {"normal_not_fallback": not i_ok.get("fallback"), "nan_is_fallback": bool(i_nan.get("fallback"))}
    # 6. audit detector: the true bound must show 0 violations, a bound shrunk 100x must show violations
    Cn = A @ B
    S = Sample(A, B, Cn)
    Cc, ic = E.run(A, B, C=Cx, force_N=8)          # p ~ 21: quantisation error dominates the final rounding, so a too-small bound must be caught
    res["audit_controls"] = {"violations_true_bound": audit_bound(S, ic, n, Cc)["violations"], "violations_bound_x0.01": audit_bound(S, ic, n, Cc, scale=0.01)["violations"]}
    # 7. timed scope: a host sleep between the graph launches and a GPU sleep inside the eager path must both show up in ev_time
    base = _q([ev_time(lambda: E.run(A, B, C=Cx, graph=True)) for _ in range(5)], 50)
    orig_cg = E.core_graph

    def slow_cg(*a_, **k_):
        time.sleep(0.05)
        return orig_cg(*a_, **k_)
    E.core_graph = slow_cg
    slow_host = _q([ev_time(lambda: E.run(A, B, C=Cx, graph=True)) for _ in range(5)], 50)
    E.core_graph = orig_cg
    base_e = _q([ev_time(lambda: E.run(A, B, C=Cx)) for _ in range(5)], 50)
    orig_core = E.core

    def slow_core(*a_, **k_):
        torch.cuda._sleep(50_000_000)
        return orig_core(*a_, **k_)
    E.core = slow_core
    slow_gpu = _q([ev_time(lambda: E.run(A, B, C=Cx)) for _ in range(5)], 50)
    E.core = orig_core
    res["timed_scope_controls"] = {"host_gap_delta_s": slow_host - base, "gpu_sleep_delta_s": slow_gpu - base_e,
                                   "ok": bool(slow_host - base > 0.035 and slow_gpu - base_e > 0.012)}
    # 8. retained-mantissa-bits pointer: emulation on G reports > 0 bits, a wide-range input with the default cap reports -1 (fallback)
    vb = Vendor()
    vb.configure(True, strategy=2, ws=torch.empty(WS_BYTES, dtype=torch.uint8, device=DEV), ws_bytes=WS_BYTES)
    n2 = 2048
    Ag, Bg = make_inputs_ext(n2, "G")
    As_, Bs_ = make_inputs_ext(n2, "S")
    Cq = torch.empty(n2, n2, dtype=torch.float64, device=DEV)
    vb.dgemm(Ag, Bg, Cq)
    torch.cuda.synchronize()
    bits_g = vb.bits_used()
    vb.dgemm(As_, Bs_, Cq)
    torch.cuda.synchronize()
    bits_s = vb.bits_used()
    # the pointer reports the RECOMMENDED retained bits (62 for G, 91 for S); a default-cap fallback to native is visible only in the time
    res["vendor_bits_controls"] = {"bits_G": bits_g, "bits_S": bits_s, "ok": bool(40 <= bits_g <= 100 and bits_s > bits_g)}
    # 9. V* falls back to native when no dynamic vendor variant matched
    R, summ = synth(0.98, 0.3, 0.02)
    dnat = decide_cell(R, summ, "G", [], True, None, True, False)
    res["vstar_native_control"] = {"vstar": dnat.get("vstar_dyn"), "ok": dnat.get("vstar_dyn") == "native" and "H1a" in dnat}
    res["all_ok"] = bool(res["fallback_flag_controls"]["normal_not_fallback"] and res["fallback_flag_controls"]["nan_is_fallback"]
                         and res["audit_controls"]["violations_true_bound"] == 0 and res["audit_controls"]["violations_bound_x0.01"] > 0
                         and res["timed_scope_controls"]["ok"] and res["vendor_bits_controls"]["ok"] and res["vstar_native_control"]["ok"]
                         and all(res["bracket_controls"].values()) and res["drift_positive_control"]["ok"] and res["drift_negative_control"]["ok"] and dc_ok
                         and all(res["match_controls"][k] for k in ("native_vs_itself_matched", "fp32_rounded_not_matched", "perturbed_entry_not_matched"))
                         and vk["b40_less_accurate_than_b64"] and vk["emulation_faster_than_native"] and vk["native13_close_to_torch_native"])
    json.dump(res, open(os.path.join(OUT, "instr_controls.json"), "w"), indent=1,
              default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(json.dumps({k: v for k, v in res.items() if k not in ("vendor_knob_controls", "decision_controls")}, default=str)[:3000])
    print("vendor knobs:", {k: v for k, v in vk.items() if k != "detail"},
          {k: (v["status"], v["err"]["E1"], round(v["t_med"], 5)) for k, v in out.items() if isinstance(v, dict) and "err" in v})
    print("INSTR", "PASS" if res["all_ok"] else "FAIL")


def race_cell(E, vend, v13, wsbuf, n, fam, results_f):
    seed = 20260929 + n + ord(fam)
    E.graphs.clear()
    A, B = make_inputs_ext(n, fam)
    Cn = torch.empty(n, n, dtype=torch.float64, device=DEV)
    torch.matmul(A, B, out=Cn)
    torch.cuda.synchronize()
    S = Sample(A, B, Cn)
    en = S.errors(Cn)
    match = lambda e: is_matched(e, en)
    rec = {"n": n, "fam": fam, "native_err": en, "env_start": env_snapshot(), "gate_c": GATE_C,
           "code_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest()}
    Cv, Cc, Co, Cn2, Ce = (torch.empty_like(Cn) for _ in range(5))
    # ---- own engine: eager host rule (reference for the timed device path), oracle N -----------------------------------------
    E.run(A, B, C=Ce)
    _, ic = E.run(A, B, C=Ce)
    own = {}
    No = po = None
    if ic.get("fallback"):
        own["cert"] = {"fallback": True}
    else:
        e = S.errors(Ce)
        own["cert"] = {"N": ic["N"], "p": ic["p"], "err": e, "matched": match(e), "worst_E_over_D": ic.get("worst_ratio_E_over_D"),
                       "audit": audit_bound(S, ic, n, Ce)}
        for _ in range(10):
            E.run(A, B, C=Co, force_N=ic["N"])
        prof = {}
        E.run(A, B, C=Co, force_N=ic["N"], prof=prof)
        own["cert"]["prof_s"] = prof
        bb_ = E.bufs[(n, n, n)]
        own["cert"]["buffer_bytes"] = int(sum(bb_[k_].numel() * bb_[k_].element_size() for k_ in ("Ares", "Bres", "X", "C32")))
        if prof.get("gemm+reduce"):
            own["cert"]["int8_tops_gemm_stage"] = ic["N"] * 2.0 * n ** 3 / prof["gemm+reduce"] / 1e12
        own["oracle"] = None
        for N in range(max(6, ic["N"] - 6), NMAX + 1):
            _, iq = E.run(A, B, C=Co, force_N=N)
            e2 = S.errors(Co)
            if match(e2):
                own["oracle"] = {"N": N, "p": iq["p"], "err": e2}
                No, po = N, iq["p"]
                Co_eager = Co.clone()
                break
    rec["own"] = own
    # ---- vendor screening (3 interleaved runs of every accuracy-matching variant) ---------------------------------------------
    scr = {}
    for name, v in vend.items():
        st = v.dgemm(A, B, Cv)
        torch.cuda.synchronize()
        e = S.errors(Cv)
        scr[name] = {"status": st, "err": e, "matched": match(e) and st == 0, "bits": v.bits_used(), "cls": VCLS[name], "t": []}
    rng = np.random.RandomState(seed)
    ok_names = [nm for nm in vend if scr[nm]["matched"]]
    for rep in range(3):
        if GATE_C is not None:
            wait_cool(GATE_C)                             # each screening pass starts cool; the order inside a pass is random
        _primer()
        for nm in rng.permutation(ok_names + ["_native13", "_native"]):
            nm = str(nm)
            if nm == "_native13":
                scr.setdefault(nm, {"t": []})["t"].append(ev_time(lambda: v13.dgemm(A, B, Cv)))
            elif nm == "_native":
                scr.setdefault(nm, {"t": []})["t"].append(ev_time(lambda: torch.matmul(A, B, out=Cn2)))
            else:
                scr[nm]["t"].append(ev_time(lambda: vend[nm].dgemm(A, B, Cv)))
    for nm in scr:
        scr[nm]["t_med"] = _q(scr[nm]["t"], 50) if scr[nm]["t"] else None
    rec["vendor_screen"] = scr
    dyn = sorted([nm for nm in ok_names if scr[nm]["cls"] == "apriori" and nm != VD], key=lambda x: scr[x]["t_med"])[:3]
    fix = sorted([nm for nm in ok_names if scr[nm]["cls"] == "oracle"], key=lambda x: scr[x]["t_med"])[:2]
    carried_dyn = [VD] + dyn
    # ---- main rounds ---------------------------------------------------------------------------------------------------------
    log = {"cert": [], "oracle": []}
    fns = {"native": lambda: torch.matmul(A, B, out=Cn2)}
    for nm in carried_dyn + fix:
        fns["vend:" + nm] = (lambda nm=nm: vend[nm].dgemm(A, B, Cv))
    if not own["cert"].get("fallback"):
        def own_cert_fn():
            _, inf = E.run(A, B, C=Cc, graph=True)
            log["cert"].append((inf.get("N"), inf.get("p"), bool(inf.get("fallback"))))
        fns["own_cert"] = own_cert_fn
        if own.get("oracle") and No != own["cert"]["N"]:
            def own_oracle_fn():
                _, inf = E.run(A, B, C=Co, force_N=No, graph=True)
                log["oracle"].append((inf.get("N"), inf.get("p"), bool(inf.get("fallback"))))
            fns["own_oracle"] = own_oracle_fn
    # At n >= GROUP_N one native-bracketed round cannot stay within 5% (pre/post native drift +9% median from heating over ~3.3 s),
    # so the timed methods are split over short brackets, each with its own native pre/post pair, cool-down wait and validity count.
    if n >= GROUP_N:
        dn = ["vend:" + x for x in dyn]
        fx = ["vend:" + x for x in fix]
        groups = [["own_cert"] + dn[:2], (["own_oracle"] if "own_oracle" in fns else []) + fx[:1], ["vend:" + VD] + dn[2:3] + fx[1:2]]
        groups = [[k_ for k_ in g if k_ in fns] for g in groups]
        groups = [g for g in groups if g]
    else:
        groups = [[k_ for k_ in fns if k_ != "native"]]
    gres = []
    for gi, gnames in enumerate(groups):
        sub = {"native": fns["native"]}
        sub.update({k_: fns[k_] for k_ in gnames})
        blk_g = run_block(sub, "native", seed + 100 * gi)
        tries = [{"valid": len(blk_g["valid"]), "invalid": len(blk_g["invalid"]), "attempts": blk_g["attempts"]}]
        if len(blk_g["valid"]) < 7:
            time.sleep(60)
            blk_g = run_block(sub, "native", seed + 100 * gi + 1)
            tries.append({"valid": len(blk_g["valid"]), "invalid": len(blk_g["invalid"]), "attempts": blk_g["attempts"]})
        gres.append({"methods": gnames, "blk": blk_g, "tries": tries})
    blk = gres[0]["blk"]
    blocks = gres[0]["tries"]
    names = [k for k in fns if k != "native"]
    allrounds = [x for g in gres for x in g["blk"]["valid"] + g["blk"]["invalid"]]
    rec.update({"n_gate_missed": int(sum(1 for x in allrounds if not x.get("gate_reached", True))),
                "native_med_s": _q([0.5 * (x["pre"] + x["post"]) for x in allrounds], 50),
                "native_nominal_s": 2.0 * n ** 3 / 0.688e12})
    rec.update({"blocks": blocks, "n_valid": len(blk["valid"]), "n_invalid": len(blk["invalid"]), "attempts": blk["attempts"],
                "carried": list(fns), "env_end": env_snapshot(), "raw_valid": blk["valid"], "raw_invalid": blk["invalid"],
                "bracket_design": "single" if len(gres) == 1 else "split",
                "groups": [{"methods": g["methods"], "tries": g["tries"], "n_valid": len(g["blk"]["valid"]), "n_invalid": len(g["blk"]["invalid"]),
                            "raw_valid": g["blk"]["valid"], "raw_invalid": g["blk"]["invalid"]} for g in gres[1:]]})
    rec["status"] = "VALID" if len(blk["valid"]) >= 7 else "INVALID(drift)"
    rec["group_status"] = [len(g["blk"]["valid"]) >= 7 for g in gres]
    # ---- post-round re-checks: accuracy of the timed callables, retained mantissa bits, timed-path flags, graph == eager -------------
    post, post_bits = {}, {}
    for nm in carried_dyn + fix:
        vend[nm].dgemm(A, B, Cv)
        torch.cuda.synchronize()
        post["vend:" + nm] = S.errors(Cv)
        post_bits["vend:" + nm] = vend[nm].bits_used()
    tp = {}
    if "own_cert" in fns:
        fns["own_cert"]()
        torch.cuda.synchronize()
        post["own_cert"] = S.errors(Cc)
        tp["graph_equals_eager"] = bool(torch.equal(Cc, Ce))
        tp["cert_calls"] = len(log["cert"])
        tp["cert_ok"] = bool(all(x == (ic["N"], ic["p"], False) for x in log["cert"]))
    if "own_oracle" in fns:
        fns["own_oracle"]()
        torch.cuda.synchronize()
        post["own_oracle"] = S.errors(Co)
        tp["oracle_graph_equals_eager"] = bool(torch.equal(Co, Co_eager))
        tp["oracle_calls"] = len(log["oracle"])
        tp["oracle_ok"] = bool(all(x == (No, po, False) for x in log["oracle"]))
    rec["post_err"], rec["post_bits"], rec["timed_path"] = post, post_bits, tp
    if rec["status"] == "VALID" and not all(v for k_, v in tp.items() if k_.endswith("_ok") or k_.endswith("graph_equals_eager")):
        rec["status"] = "INVALID(timed_path)"
    if rec["status"] == "VALID":
        summ, summ_all, Rmap = {}, {}, {}
        for g in gres:
            if len(g["blk"]["valid"]) < 7:
                continue                                        # a bracket without 7 valid rounds contributes nothing to any decision
            sg = summarize(g["blk"], g["methods"], "native")
            sa = summarize({"valid": g["blk"]["valid"] + g["blk"]["invalid"]}, g["methods"], "native")
            for k_ in g["methods"]:
                summ[k_], summ_all[k_], Rmap[k_] = sg[k_], sa[k_], g["blk"]["valid"]
            summ.setdefault("native", sg["native"])
            summ_all.setdefault("native", sa["native"])
        rec["summary"], rec["summary_all_attempts"] = summ, summ_all
        dyn_names = ["vend:" + nm for nm in carried_dyn if match(post["vend:" + nm]) and ("vend:" + nm) in summ]
        fix_in = [("vend:" + nm) for nm in fix if ("vend:" + nm) in summ]
        dec = decide_cell(blk["valid"], summ, fam, dyn_names, match(post["own_cert"]) if "own_cert" in fns else False,
                          min(fix_in, key=lambda x: summ[x]["ratio_med"]) if fix_in else None, "own_cert" in fns, "own_oracle" in fns,
                          vend_all=["vend:" + nm for nm in carried_dyn + fix], Rmap=Rmap, vkey="med_s" if len(gres) == 1 else "ratio_med")
        rec["decision"] = dec
        print(json.dumps({"n": n, "fam": fam, "N_cert": own["cert"].get("N"), "N_oracle": No, "valid": [len(g["blk"]["valid"]) for g in gres],
                          "invalid": [len(g["blk"]["invalid"]) for g in gres], "speedup": {k: round(v["speedup_med"], 2) for k, v in summ.items()},
                          "decision": {k: v for k, v in dec.items() if k != "rho_table"}}), flush=True)
    else:
        print(json.dumps({"n": n, "fam": fam, "status": rec["status"], "valid": [len(g["blk"]["valid"]) for g in gres],
                          "invalid": [len(g["blk"]["invalid"]) for g in gres], "timed_path": tp}), flush=True)
    with open(results_f, "a") as f:
        f.write(json.dumps(rec, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
    del A, B, Cn, Cv, Cc, Co, Cn2, Ce
    torch.cuda.empty_cache()


def cmd_race(ns, fams):
    set_high_priority()
    
    E = OzakiCRT()
    wsbuf = torch.empty(8 * GIB, dtype=torch.uint8, device=DEV)
    vend = {}
    for name, kw in VARIANTS:
        v = Vendor()
        v.configure(True, strategy=kw["strategy"], cap=kw.get("cap", 0), fixed_bits=kw.get("fixed", 0), sv=kw.get("sv"), offset=kw.get("offset", 0),
                    ws=wsbuf if kw["ws"] else None, ws_bytes=kw["ws"])
        vend[name] = v
    v13 = Vendor()
    v13.configure(False)
    results_f = os.path.join(OUT, os.environ.get("OZAKI_RESULTS_NAME", "race_results.jsonl"))
    done = set()
    if os.path.exists(results_f) and os.environ.get("OZAKI_SKIP_DONE", "1") == "1":
        for line in open(results_f):
            r = json.loads(line)
            if r.get("status") == "VALID":
                done.add((r["n"], r["fam"]))
    for n in ns:
        for fam in fams:
            if (n, fam) in done:
                print("skip (already VALID):", n, fam, flush=True)
                continue
            race_cell(E, vend, v13, wsbuf, n, fam, results_f)



class PersistA:
    """Persistent fixed operand (fixed-operand mode): A is encoded ONCE at a fixed (N, p) (inside scaling off, gamma = 0); every product A B_j pays only the
    B-side statistics, a device check that p is still exact-safe for B_j (nu bound), the B residues, the N INT8 GEMMs, reductions and reconstruction.
    Finite B is assumed (no non-finite flag in this mode)."""

    def __init__(self, E, A, B0, B_shape_n):
        self.E, self.A = E, A
        m, k = A.shape
        self.n = B_shape_n
        C0 = torch.empty(m, self.n, dtype=torch.float64, device=DEV)
        _, info = E.run(A, B0, C=C0, inside=False)                      # a-priori rule on the first product fixes (N, p)
        self.N, self.p = info["N"], info["p"]
        self.b = E.buffers(m, self.n, k, max(self.N, 20))
        self.C = None
        self.graphs = {}

    def encode(self):
        E, A, b = self.E, self.A, self.b
        m, k = A.shape
        K = E.kern
        st = torch.cuda.current_stream().cuda_stream
        b["gam"].zero_()
        K["rowstatsA_k"].launch((m,), (256,), [ptr(A), ptr(b["gam"]), c_int(k), ptr(b["r"]), ptr(b["sA"]), ptr(b["nA"])], stream=st)
        b["pdev"].fill_(self.p)
        kd = E.kernels(self.N)
        tot = m * (k // 4)
        kd["resA"].launch(((tot + 255) // 256,), (256,), [ptr(A), ptr(b["r"]), ptr(b["gam"]), ptr(b["Ares"]), ptr(E.tab),
                                                          c_int(m), c_int(k), ptr(b["pdev"])], stream=st)

    def _stats_B(self, B):
        E, b = self.E, self.b
        k, n = B.shape
        K = E.kern
        st = torch.cuda.current_stream().cuda_stream
        b["c"].fill_(NEG)
        b["bd"][2 * self.A.shape[0]:].zero_()
        l_per = 256
        g2 = ((n + 127) // 128, (k + l_per - 1) // l_per)
        K["colexpB_k"].launch(g2, (128,), [ptr(B), ptr(b["gam"]), c_int(k), c_int(n), c_int(l_per), ptr(b["c"])], stream=st)
        K["colsumsB_k"].launch(g2, (128,), [ptr(B), ptr(b["gam"]), ptr(b["c"]), c_int(k), c_int(n), c_int(l_per), ptr(b["sB"]), ptr(b["nB"])], stream=st)
        E._rule_device(self.A, B, b, 2.0 ** -52, 2.0, self.N, self.p)    # forced (N, p): flags fallback if p exceeds the data range for this B

    def product(self, B, C):
        E, b = self.E, self.b
        key = (B.data_ptr(), C.data_ptr())
        gr = self.graphs.get(key)
        if gr is None:
            gr = E._capture(lambda: (self._stats_B(B), E.core(self.A, B, b, self.N, C, skipA=True)))
            self.graphs[key] = gr
        gr.replay()


def cmd_persist(n=4096, fams=("G", "R", "X", "S", "W"), qs=(1, 2, 4, 8, 16)):
    """Fixed operand: streams of q products A B_j with A encoded once, against own non-persistent and the best accuracy-matching vendor variant."""
    set_high_priority()
    
    E = OzakiCRT()
    E2 = OzakiCRT(lt=E.lt)
    wsbuf = torch.empty(8 * GIB, dtype=torch.uint8, device=DEV)
    race = {}
    for line in open(os.path.join(OUT, "race_results.jsonl")):
        r = json.loads(line)
        if r.get("status") == "VALID":
            race[(r["n"], r["fam"])] = r
    out_f = os.path.join(OUT, "persist_results.jsonl")
    vd = dict(VARIANTS)
    for fam in fams:
        A, B0 = make_inputs_ext(n, fam, 0)
        Bs = [make_inputs_ext(n, fam, j)[1] for j in range(max(qs))]
        Cn = torch.empty(n, n, dtype=torch.float64, device=DEV)
        torch.matmul(A, B0, out=Cn)
        S0 = Sample(A, B0, Cn)
        en0 = S0.errors(Cn)
        vname = race[(n, fam)]["decision"]["vstar_dyn"].replace("vend:", "")
        kw = vd[vname]
        v = Vendor()
        v.configure(True, strategy=kw["strategy"], cap=kw.get("cap", 0), sv=kw.get("sv"), ws=wsbuf if kw["ws"] else None, ws_bytes=kw["ws"])
        pa = PersistA(E2, A, B0, n)
        Cs = [torch.empty(n, n, dtype=torch.float64, device=DEV) for _ in range(2)]
        Cv, Cn2 = torch.empty_like(Cn), torch.empty_like(Cn)
        pa.encode()
        pa.product(B0, Cs[0])
        torch.cuda.synchronize()
        e_p = S0.errors(Cs[0])
        E.run(A, B0, C=Cs[1])
        e_plain = S0.errors(Cs[1])
        v.dgemm(A, B0, Cv)
        torch.cuda.synchronize()
        e_v = S0.errors(Cv)
        rec = {"n": n, "fam": fam, "vendor": vname, "N": pa.N, "p": pa.p, "err_native": en0, "err_persist": e_p, "err_plain": e_plain, "err_vendor": e_v,
               "matched": {"persist": is_matched(e_p, en0), "plain": is_matched(e_plain, en0), "vendor": is_matched(e_v, en0)}, "q": {}}
        Cq = [torch.empty(n, n, dtype=torch.float64, device=DEV) for _ in range(2)]

        def own_plain(q):
            def f():
                for j in range(q):
                    E.run(Bs[0].new_empty(0) if False else A, Bs[j], C=Cq[j % 2], graph=True)
            return f

        def own_persist(q):
            def f():
                pa.encode()
                for j in range(q):
                    pa.product(Bs[j], Cq[j % 2])
            return f

        def vend(q):
            def f():
                for j in range(q):
                    v.dgemm(A, Bs[j], Cv)
            return f
        for q in qs:
            if q in (2, 8) and fam not in ("G",):
                continue
            fns = {"native": lambda: torch.matmul(A, B0, out=Cn2), "own_persist": own_persist(q), "own_plain": own_plain(q), "vendor": vend(q)}
            for _ in range(2):
                for k_, f in fns.items():
                    f()
            torch.cuda.synchronize()
            blk = run_block(fns, "native", 7000 + q)
            if len(blk["valid"]) < 7:
                time.sleep(60)
                blk = run_block(fns, "native", 7100 + q)
            ok = len(blk["valid"]) >= 7
            R = blk["valid"]
            summ = summarize(blk, ["own_persist", "own_plain", "vendor"], "native") if ok else {}
            rho_v = [x["t"]["own_persist"] / x["t"]["vendor"] for x in R]
            rho_p = [x["t"]["own_persist"] / x["t"]["own_plain"] for x in R]
            rec["q"][q] = {"valid": len(R), "invalid": len(blk["invalid"]), "summary": summ,
                           "rho_persist_vs_vendor": [_q(rho_v, 50), _q(rho_v, 25), _q(rho_v, 75)] if R else None,
                           "rho_persist_vs_plain": [_q(rho_p, 50), _q(rho_p, 25), _q(rho_p, 75)] if R else None,
                           "iqr_overlap_vs_vendor": overlap(summ["own_persist"], summ["vendor"]) if ok else None,
                           "iqr_overlap_vs_plain": overlap(summ["own_persist"], summ["own_plain"]) if ok else None}
            print(json.dumps({"fam": fam, "q": q, **{k_: v_ for k_, v_ in rec["q"][q].items() if k_ != "summary"}}), flush=True)
        # accuracy of the LAST product of the longest stream (p fixed from the first product): sampled against the correctly rounded reference
        jl = max(qs) - 1
        pa.encode()
        pa.product(Bs[jl], Cs[0])
        torch.cuda.synchronize()
        Sl = Sample(A, Bs[jl], A @ Bs[jl])
        enl = Sl.errors(A @ Bs[jl])
        rec["last_product"] = {"err_native": enl, "err_persist": Sl.errors(Cs[0]), "matched": is_matched(Sl.errors(Cs[0]), enl)}
        with open(out_f, "a") as fh:
            fh.write(json.dumps(rec, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        del A, B0, Bs, Cs, Cq, Cn, Cv, Cn2, v, pa
        E2.graphs.clear()
        E.graphs.clear()
        torch.cuda.empty_cache()


def cmd_trace(n=4096, fams=("G", "W")):
    """Host-share trace: decompose the TIMED run_fast path: (a) stats/rule graph, (b) host read + sync + dispatch, (c) core graph; interleaved with native
    and V*dyn, with cool-down waits; host share = (end-to-end wall - GPU a - GPU c) / wall."""
    set_high_priority()
    E = OzakiCRT()
    wsbuf = torch.empty(8 * GIB, dtype=torch.uint8, device=DEV)
    race = {}
    for line in open(os.path.join(OUT, "race_results.jsonl")):
        r = json.loads(line)
        if r.get("status") == "VALID":
            race[(r["n"], r["fam"])] = r
    vd = dict(VARIANTS)
    for fam in fams:
        A, B = make_inputs_ext(n, fam)
        C, Cv, Cn = (torch.empty(n, n, dtype=torch.float64, device=DEV) for _ in range(3))
        kw = vd[race[(n, fam)]["decision"]["vstar_dyn"].replace("vend:", "")]
        v = Vendor()
        v.configure(True, strategy=kw["strategy"], cap=kw.get("cap", 0), sv=kw.get("sv"), ws=wsbuf if kw["ws"] else None, ws_bytes=kw["ws"])
        b = E.buffers(n, n, n, 20)
        E.run(A, B, C=C, graph=True)
        key = [k for k in E.graphs if k[0] == "sr"][0]
        gsr = E.graphs[key]
        rows = []

        def traced():
            ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
            t0 = time.perf_counter()
            ev[0].record(); gsr.replay(); ev[1].record()
            b["hdec"].copy_(b["dec"], non_blocking=True)
            torch.cuda.current_stream().synchronize()
            t1 = time.perf_counter()
            N = int(b["hdec"].tolist()[0])
            ev[2].record(); E.core_graph(A, B, b, N, C); ev[3].record()
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            rows.append({"a": ev[0].elapsed_time(ev[1]) / 1e3, "c": ev[2].elapsed_time(ev[3]) / 1e3, "wall": t2 - t0, "host_mid": t1 - t0})
        fns = {"native": lambda: torch.matmul(A, B, out=Cn), "traced": traced, "plain": lambda: E.run(A, B, C=C, graph=True), "vendor": lambda: v.dgemm(A, B, Cv)}
        for _ in range(3):
            for f in fns.values():
                f()
        rows.clear()
        blk = run_block(fns, "native", 9000)
        R = blk["valid"]
        k0 = len(rows) - len(blk["valid"]) - len(blk["invalid"])
        rr = rows[-(len(R) + len(blk["invalid"])):]
        host = [(x["wall"] - x["a"] - x["c"]) / x["wall"] for x in rr]
        rec = {"n": n, "fam": fam, "valid": len(R), "invalid": len(blk["invalid"]), "a_ms": 1e3 * _q([x["a"] for x in rr], 50), "c_ms": 1e3 * _q([x["c"] for x in rr], 50),
               "wall_traced_ms": 1e3 * _q([x["wall"] for x in rr], 50), "host_share_med": _q(host, 50), "host_share_q": [_q(host, 25), _q(host, 75)],
               "plain_ms": 1e3 * _q([x["t"]["plain"] for x in R], 50) if R else None, "vendor_ms": 1e3 * _q([x["t"]["vendor"] for x in R], 50) if R else None,
               "rule": "host share >= 20% -> device-resident dispatch next; < 10% -> reject premise; between -> repeat"}
        print(json.dumps(rec), flush=True)
        with open(os.path.join(OUT, "trace_results.jsonl"), "a") as fh:
            fh.write(json.dumps(rec) + chr(10))
        del A, B, C, Cv, Cn
        E.graphs.clear()
        torch.cuda.empty_cache()


def cmd_audit(n=4096, fams=("G", "W", "X"), blocks=5, block_s=15.0):
    """Audit: try to disprove the vendor-relative win.  Contenders: native, best accuracy-matching vendor DYNAMIC and FIXED/offset variant
    (selection charged), own a-priori (cert) and own oracle.  Randomised sweeps, `blocks` cooled blocks of ~block_s seconds, per-call timing, NVML state per block,
    full-matrix errors of every contender's output against the N=32 engine result (|A||B| componentwise) plus sampled correctly rounded reference.
    Promotion rule: lower 95% confidence bound (t, df = blocks-1, on block log-ratios) of the speedup of own vs the best matched vendor mode must exceed 1.25."""
    set_high_priority()
    E = OzakiCRT()
    wsbuf = torch.empty(8 * GIB, dtype=torch.uint8, device=DEV)
    race = {}
    for line in open(os.path.join(OUT, "race_results.jsonl")):
        r = json.loads(line)
        if r.get("status") == "VALID":
            race[(r["n"], r["fam"])] = r
    vd = dict(VARIANTS)
    for fam in fams:
        rr = race[(n, fam)]
        A, B = make_inputs_ext(n, fam)
        Cn = torch.empty(n, n, dtype=torch.float64, device=DEV)
        torch.matmul(A, B, out=Cn)
        names = {"vdyn": rr["decision"]["vstar_dyn"].replace("vend:", "")}
        fixc = [k for k in rr["summary"] if k.startswith("vend:fix") or k.startswith("vend:off")]
        names["vfix"] = max(fixc, key=lambda k: rr["summary"][k]["speedup_med"]).replace("vend:", "")
        outs = {k: torch.empty_like(Cn) for k in ("native", "vdyn", "vfix", "own_cert", "own_oracle")}
        vs = {}
        for key in ("vdyn", "vfix"):
            kw = vd[names[key]]
            v = Vendor()
            v.configure(True, strategy=kw["strategy"], cap=kw.get("cap", 0), fixed_bits=kw.get("fixed", 0), sv=kw.get("sv"), offset=kw.get("offset", 0),
                        ws=wsbuf if kw["ws"] else None, ws_bytes=kw["ws"])
            vs[key] = v
        No = rr["own"]["oracle"]["N"]
        fns = {"native": lambda: torch.matmul(A, B, out=outs["native"]),
               "vdyn": lambda: vs["vdyn"].dgemm(A, B, outs["vdyn"]), "vfix": lambda: vs["vfix"].dgemm(A, B, outs["vfix"]),
               "own_cert": lambda: E.run(A, B, C=outs["own_cert"], graph=True), "own_oracle": lambda: E.run(A, B, C=outs["own_oracle"], force_N=No, graph=True)}
        for _ in range(3):
            for f in fns.values():
                f()
        torch.cuda.synchronize()
        # selection charge: vendor screening calls actually made in the race cell, own oracle scan (calls x measured eager call time)
        scr = rr["vendor_screen"]
        vend_sel_calls = sum(1 + len(v.get("t", [])) for k, v in scr.items() if not k.startswith("_"))
        vend_sel_s = sum(sum(v.get("t", [])) + (v["t"][0] if v.get("t") else 0) for k, v in scr.items() if not k.startswith("_"))
        te = _q([ev_time(lambda: E.run(A, B, C=outs["own_oracle"], force_N=No)) for _ in range(3)], 50)
        own_sel_calls = rr["own"]["cert"]["N"] - max(6, rr["own"]["cert"]["N"] - 6) + 1
        # full-matrix accuracy against the N=32 result and |A||B|
        Cref, _ = E.run(A, B, C=torch.empty_like(Cn), force_N=32)
        Dm = (torch.abs(A) @ torch.abs(B)).clamp_min(1e-300)
        acc = {}
        for k, o in outs.items():
            d = (o - Cref).abs()
            acc[k] = {"E1_full": float(d.max() / Cref.abs().max()), "E2_full": float((d / Dm).max()),
                      "normwise_rel_fro": float(torch.linalg.norm(o - Cref) / torch.linalg.norm(Cref))}
        S = Sample(A, B, Cn)
        for k, o in outs.items():
            acc[k]["sampled_E1E2_vs_correctly_rounded"] = S.errors(o)
        rec = {"n": n, "fam": fam, "variants": names, "oracle_N": No, "cert_N": rr["own"]["cert"]["N"], "accuracy": acc,
               "selection_charge": {"vendor_calls": vend_sel_calls, "vendor_gpu_s": vend_sel_s, "own_oracle_calls": own_sel_calls, "own_oracle_gpu_s": own_sel_calls * te},
               "blocks": []}
        rng = np.random.RandomState(31337 + n)
        first = {k: o.clone() for k, o in outs.items()}
        for bi in range(blocks):
            wait_cool(GATE_C if GATE_C is not None else 74)
            _primer()
            st0 = gpu_state()
            ts = {k: [] for k in fns}
            sweeps, t_end = 0, time.time() + block_s
            while time.time() < t_end:
                for k in rng.permutation(list(fns)):
                    ts[str(k)].append(ev_time(fns[str(k)]))
                sweeps += 1
            st1 = gpu_state()
            nat = np.array(ts["native"])
            half = len(nat) // 2
            drift = float(np.median(nat[half:]) / np.median(nat[:half]) - 1) if half else None
            med = {k: float(np.median(v)) for k, v in ts.items()}
            cv = {k: float((np.percentile(v, 75) - np.percentile(v, 25)) / np.median(v)) for k, v in ts.items()}
            same = {k: bool(torch.equal(outs[k], first[k])) for k in outs}
            rec["blocks"].append({"sweeps": sweeps, "state_start": st0, "state_end": st1, "median_s": med, "iqr_over_median": cv, "native_half_drift": drift,
                                  "outputs_identical_to_first": same, "cpu_pct": cpu_pct_fast(), "reliable": bool(drift is not None and abs(drift) <= 0.05)})
            print(json.dumps({"fam": fam, "block": bi, "sweeps": sweeps, "median_ms": {k: round(1e3 * v, 2) for k, v in med.items()}, "native_half_drift": drift}), flush=True)
            if sum(1 for b_ in rec["blocks"] if not b_["reliable"]) > 2:
                rec["stopped_early"] = "more than 2 unreliable blocks"
                break
        bl = [b_ for b_ in rec["blocks"] if b_["reliable"]] or rec["blocks"]
        res = {}
        for own in ("own_cert", "own_oracle"):
            for ven in ("vdyn", "vfix"):
                lr = np.array([np.log(b_["median_s"][ven] / b_["median_s"][own]) for b_ in bl])
                m_, sd = lr.mean(), lr.std(ddof=1) if len(lr) > 1 else float("nan")
                tq = {1: 12.71, 2: 4.303, 3: 3.182, 4: 2.776}.get(len(lr) - 1, 1.96)
                res[f"{own}_vs_{ven}"] = {"speedup_mean": float(np.exp(m_)), "lcb95": float(np.exp(m_ - tq * sd / np.sqrt(len(lr)))), "blocks_used": len(lr)}
        best = {}
        for own in ("own_cert", "own_oracle"):
            lcb = min(res[f"{own}_vs_vdyn"]["lcb95"], res[f"{own}_vs_vfix"]["lcb95"])
            best[own] = {"lcb95_vs_best_matched_vendor_mode": lcb, "promote_if_gt_1.25": bool(lcb > 1.25)}
        rec["speedups"], rec["promotion"] = res, best
        print(json.dumps({"fam": fam, "speedups": res, "promotion": best}), flush=True)
        with open(os.path.join(OUT, "audit_results.jsonl"), "a") as fh:
            fh.write(json.dumps(rec, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + chr(10))
        del A, B, Cn, outs, first, Cref, Dm
        E.graphs.clear()
        torch.cuda.empty_cache()


def cmd_report(name="race_results.jsonl"):
    """Compact per-cell table from the raw results (median speedup over native)."""
    rows = [json.loads(l) for l in open(os.path.join(OUT, name))]
    latest = {}
    for r in rows:                      # INVALID cells are re-run and appended: keep the latest record per (n, fam)
        latest[(r["n"], r["fam"])] = r
    rows = list(latest.values())
    print("n    fam status   valid/inv Ncert Norac | speedup over native: Vd  V*dyn V*fix own_cert own_oracle | rho_med [q1,q3] vs V*dyn  H1a  H1b | matched | native/nominal gateT cpu%")
    for r in rows:
        if r.get("status") != "VALID":
            print(f"{r['n']:<5}{r['fam']:<4}{r['status']} valid={r['n_valid']} invalid={r['n_invalid']}")
            continue
        sm, d = r["summary"], r.get("decision", {})
        vd = sm.get("vend:" + VD, {}).get("speedup_med", float("nan"))
        vs = sm.get(d.get("vstar_dyn", ""), {}).get("speedup_med", float("nan"))
        fix_vals = [sm[k]["speedup_med"] for k in sm if k.startswith("vend:fix") or k.startswith("vend:off")]
        vf = max(fix_vals) if fix_vals else float("nan")
        oc = sm.get("own_cert", {}).get("speedup_med", float("nan"))
        oo = sm.get("own_oracle", {}).get("speedup_med", float("nan"))
        clk = [r.get("native_med_s", float("nan")) / r.get("native_nominal_s", float("nan"))]      # native time / nominal 0.688 TFLOPS time (regime witness)
        tmp = [x.get("gate_start_temp", 0) for x in r["raw_valid"]]
        cpu = [x["cpu_pct"] for x in r["raw_valid"] if x.get("cpu_pct") is not None]
        print(f"{r['n']:<5}{r['fam']:<4}{r['status']:<8}{r['n_valid']}/{r['n_invalid']:<6} {r['own']['cert'].get('N')!s:<5} {(r['own'].get('oracle') or {}).get('N')!s:<5}"
              f"| {vd:6.2f} {vs:6.2f} {vf:6.2f} {oc:7.2f} {oo:7.2f} | {d.get('rho_med', float('nan')):.3f} [{d.get('rho_q1', float('nan')):.3f},{d.get('rho_q3', float('nan')):.3f}]"
              f" {d.get('H1a')!s:<5} {d.get('H1b')!s:<5} | {d.get('own_cert_matched')} | {np.median(clk):.2f} {np.median(tmp):.0f} {np.median(cpu) if cpu else float('nan'):.0f}")


# =====================================================================================================================
if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "diag"
    if cmd == "diag":
        cmd_diag()
    elif cmd == "selftest":
        cmd_selftest()
    elif cmd == "check":
        cmd_check(int(sys.argv[2]), sys.argv[3].split(",") if len(sys.argv) > 3 else ["G", "X", "S", "R", "W"])
    elif cmd == "race":
        cmd_race([int(x) for x in sys.argv[2].split(",")], sys.argv[3].split(","))
    elif cmd == "instr":
        cmd_instr()
    elif cmd == "config":
        cmd_config()
    elif cmd == "audit":
        cmd_audit()
    elif cmd == "trace":
        cmd_trace()
    elif cmd == "persist":
        cmd_persist(int(sys.argv[2]) if len(sys.argv) > 2 else 4096, sys.argv[3].split(",") if len(sys.argv) > 3 else ("G", "R", "X", "S", "W"))
    elif cmd == "report":
        cmd_report(sys.argv[2] if len(sys.argv) > 2 else "race_results.jsonl")
    else:
        raise SystemExit(__doc__)
