#!/usr/bin/env python3
"""Запуск тяжёлого процесса с ЖЁСТКИМИ лимитами памяти и CPU.

Зачем: на этой машине (15.9 ГБ RAM, GPU делится с браузером) тяжёлый прогон,
запущенный без ограничения, съедает всю память и приводит к OOM у всей
системы — включая чужую работу. Windows даёт нужный механизм: Job Object с
лимитом памяти процесса. При его срабатывании процесс получает MemoryError
внутри себя и умирает, а система остаётся живой.

Использование:
    python tools/limited_run.py --ram-pct 60 --cpu-pct 60 --log out.log -- python tools/check_v20_generation.py

Лимиты по умолчанию берутся из переменных окружения ZIWAGO_RAM_PCT / ZIWAGO_CPU_PCT.
Пороги (85 %) — из правил проекта: VRAM ≤ 85 %, RAM ≤ 85 %, CPU ≤ 85 % ядер;
здесь по умолчанию 60 %, чтобы оставить запас пользователю.

Отказ закрыт (fail closed): если лимит памяти не удалось применить (Job Object
не назначен), процесс НЕ запускается и возвращается код 3. Запуск без лимита
возможен только с явным флагом --allow-unlimited.
"""
from __future__ import annotations

import argparse
import ctypes
import os
import subprocess
import sys
import time
from ctypes import wintypes

# --- константы Windows Job Object ---
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200
JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
JOB_OBJECT_CPU_RATE_CONTROL_INFORMATION = 15
# Значения из winnt.h: ENABLE=0x1, WEIGHT_BASED=0x2, HARD_CAP=0x4.
# Жёсткий потолок = ENABLE | HARD_CAP; один ENABLE без режима WinAPI отвергает.
JOB_OBJECT_CPU_RATE_CONTROL_ENABLE = 0x00000001
JOB_OBJECT_CPU_RATE_CONTROL_HARD_CAP = 0x00000004
CREATE_SUSPENDED = 0x00000004

class IO_COUNTERS(ctypes.Structure):
    _fields_ = [("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong)]


class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD)]


class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t)]


class JOBOBJECT_CPU_RATE_CONTROL_INFORMATION(ctypes.Structure):
    # {DWORD ControlFlags; union {DWORD CpuRate; DWORD Weight; struct {WORD MinRate, MaxRate;}}} = 8 байт.
    # CpuRate — в сотых долях процента (1..10000) при HARD_CAP.
    _fields_ = [("ControlFlags", wintypes.DWORD),
                ("CpuRate", wintypes.DWORD)]


def total_physical_memory() -> tuple[int, int]:
    """Возвращает (всего физической памяти, доступно сейчас) в байтах."""
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    return int(status.ullTotalPhys), int(status.ullAvailPhys)


def cpu_count() -> int:
    return os.cpu_count() or 4


def make_job(ram_bytes: int, cpu_pct: int) -> tuple[object, bool]:
    """Создаёт Job Object. Возвращает (job, cpu_cap_applied).

    Лимит памяти обязателен: его отказ — исключение. Жёсткий потолок CPU
    применяется по возможности, но результат больше не игнорируется.
    """
    kernel32 = ctypes.windll.kernel32
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        raise OSError("CreateJobObjectW failed")
    limits = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    limits.BasicLimitInformation.LimitFlags = (
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | JOB_OBJECT_LIMIT_JOB_MEMORY
        | JOB_OBJECT_LIMIT_PROCESS_MEMORY)
    limits.JobMemoryLimit = ram_bytes
    limits.ProcessMemoryLimit = ram_bytes
    if not kernel32.SetInformationJobObject(job, 9, ctypes.byref(limits),
                                            ctypes.sizeof(limits)):
        raise OSError("SetInformationJobObject(JobMemoryLimit) failed")
    cpu_cap_applied = False
    if cpu_pct and cpu_pct < 100:
        cpu = JOBOBJECT_CPU_RATE_CONTROL_INFORMATION()
        cpu.ControlFlags = (JOB_OBJECT_CPU_RATE_CONTROL_ENABLE
                            | JOB_OBJECT_CPU_RATE_CONTROL_HARD_CAP)
        cpu.CpuRate = max(1, int(cpu_pct * 100))
        cpu_cap_applied = bool(kernel32.SetInformationJobObject(
            job, JOB_OBJECT_CPU_RATE_CONTROL_INFORMATION,
            ctypes.byref(cpu), ctypes.sizeof(cpu)))
    return job, cpu_cap_applied


def assign(job, pid: int) -> tuple[bool, int]:
    """Назначает процесс в job object. Процесс должен быть создан приостановленным."""
    kernel32 = ctypes.windll.kernel32
    PROCESS_SET_QUOTA = 0x0100
    PROCESS_TERMINATE = 0x0001
    PROCESS_QUERY_INFORMATION = 0x0400
    handle = kernel32.OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE | PROCESS_QUERY_INFORMATION,
                                   False, pid)
    if not handle:
        return False, kernel32.GetLastError()
    ok = bool(kernel32.AssignProcessToJobObject(job, handle))
    error = kernel32.GetLastError() if not ok else 0
    kernel32.CloseHandle(handle)
    return ok, error


def job_peak(job) -> tuple[int, int]:
    """Пик памяти процесса в job и жёсткий лимит (байты)."""
    kernel32 = ctypes.windll.kernel32
    info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    ok = kernel32.QueryInformationJobObject(job, 9, ctypes.byref(info),
                                            ctypes.sizeof(info), None)
    if not ok:
        return 0, 0
    return int(info.PeakJobMemoryUsed), int(info.JobMemoryLimit)


def main() -> int:
    if os.name != "nt":
        print("[limited_run] поддерживается только Windows (Job Object); "
              "на этой платформе лимит применить нельзя, запуск отменён.", file=sys.stderr)
        return 2
    parser = argparse.ArgumentParser()
    parser.add_argument("--ram-pct", type=int,
                        default=int(os.environ.get("ZIWAGO_RAM_PCT", "60")))
    parser.add_argument("--ram-free-pct", type=int,
                        default=int(os.environ.get("ZIWAGO_RAM_FREE_PCT", "95")),
                        help="Потолок как доля СВОБОДНОЙ памяти в момент запуска. "
                             "Это и есть настоящая защита машины: лимит от общего "
                             "объёма не защищает, если память уже занята чужими "
                             "приложениями (у пользователя занято ~7 ГБ).")
    parser.add_argument("--cpu-pct", type=int,
                        default=int(os.environ.get("ZIWAGO_CPU_PCT", "60")))
    parser.add_argument("--allow-unlimited", action="store_true",
                        help="Разрешить запуск БЕЗ лимита памяти, если назначение в Job Object "
                             "не удалось. По умолчанию такой запуск отменяется (код 3).")
    parser.add_argument("--log", type=str, default=None)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = [part for part in args.command if part != "--"]
    if not command:
        parser.error("no command given")
    total, avail = total_physical_memory()
    cap_total = int(total * max(5, min(95, args.ram_pct)) / 100)
    cap_free = int(avail * max(5, min(100, args.ram_free_pct)) / 100)
    cap = min(cap_total, cap_free)
    job, cpu_cap_applied = make_job(cap, max(1, min(100, args.cpu_pct)))
    out = open(args.log, "ab", buffering=0) if args.log else None
    started = time.perf_counter()

    def report(line: str) -> None:
        # Консоль Windows по умолчанию cp1251 — символ «←» роняет печать и
        # ТЕРЯЕТ весь отчёт о прогоне. Печатаем безопасно, в лог — в UTF-8.
        try:
            print(line, flush=True)
        except UnicodeEncodeError:
            print(line.encode("ascii", "replace").decode("ascii"), flush=True)
        if out is not None:
            out.write((line + "\n").encode("utf-8"))

    report(f"[limited_run] RAM cap {cap/1e9:.2f} ГБ "
           f"(min({args.ram_pct}% от {total/1e9:.1f} ГБ, {args.ram_free_pct}% от свободных "
           f"{avail/1e9:.1f} ГБ)), CPU cap {args.cpu_pct}% "
           f"({max(1, args.cpu_pct*cpu_count()//100)} из {cpu_count()} ядер)")
    if args.cpu_pct < 100 and not cpu_cap_applied:
        report("[limited_run] ВНИМАНИЕ: жёсткий потолок CPU НЕ применён "
               "(SetInformationJobObject вернул ошибку); лимит памяти действует.")
    try:
        # Процесс создаётся ПРИОСТАНОВЛЕННЫМ: лимит памяти должен действовать
        # с первого байта его адресного пространства, а не после первого
        # размещения. Затем он назначается в job и только потом возобновляется.
        process = subprocess.Popen(command, stdout=out or subprocess.DEVNULL,
                                   stderr=subprocess.STDOUT if out else subprocess.DEVNULL,
                                   creationflags=CREATE_SUSPENDED)
    except OSError as exc:
        report(f"[limited_run] failed to start: {exc}")
        return 2
    ok, error = assign(job, process.pid)
    if not ok:
        if not args.allow_unlimited:
            report(f"[limited_run] ОШИБКА: не удалось назначить процесс в job object (Win32 {error}); "
                   "лимит памяти НЕ применён, запуск отменён (используйте --allow-unlimited, "
                   "только если осознанно принимаете риск OOM).")
            process.kill()
            return 3
        report(f"[limited_run] ВНИМАНИЕ: не удалось назначить job object (Win32 {error}) — "
               "лимит памяти НЕ применён (--allow-unlimited)")
        try:
            ctypes.windll.ntdll.NtResumeProcess(process._handle)
        except Exception:                                   # noqa: BLE001
            process.kill()
            return 3
    else:
        ntdll = ctypes.windll.ntdll
        status = ntdll.NtResumeProcess(process._handle)
        if status != 0:
            report(f"[limited_run] NtResumeProcess вернул {status}")
            process.kill()
            return 3
    code = process.wait()
    elapsed = time.perf_counter() - started
    peak, limit = job_peak(job)
    report(f"[limited_run] завершился с кодом {code} за {elapsed/60:.1f} мин; "
           f"пик памяти {peak/1e9:.2f} ГБ при лимите {limit/1e9:.2f} ГБ"
           + ("  ← УБИТ ЛИМИТОМ ПАМЯТИ" if code != 0 and peak >= limit * 0.95 else ""))
    if out:
        out.close()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
