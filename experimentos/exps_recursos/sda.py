import time
import psutil

def funcion_lenta():
    s = 0
    for i in range(10_000_000):
        s += i
    return s

process = psutil.Process()

t0 = time.perf_counter()
cpu_start = process.cpu_times()
funcion_lenta()
cpu_end = process.cpu_times()
t = time.perf_counter() - t0
cpu_used = (cpu_end.user + cpu_end.system) - (cpu_start.user + cpu_start.system)

print(f"Tiempo total: {t:.4f} s")
print(f"Tiempo CPU: {cpu_used:.4f} s")
