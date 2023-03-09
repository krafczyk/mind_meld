import timeit

def best_time_callable(f, total_est_time=20, min_executions=10, min_repeats=10, report=False):
    timer = timeit.Timer(f)
    number, time_taken = timer.autorange()
    num = int(total_est_time/time_taken)
    if num < min_repeats:
        num = min_repeats
    if number < min_executions:
        number = min_executions
    all_times = timeit.repeat(f, repeat=num, number=number)
    best_time = min(all_times)/number
    print(f"Estimated Best Time: {render_time(best_time)} from {num*number} executions")
    return best_time

def render_time(time_s):
    if time_s < 1e-9:
        return f"{time_s:.2e}"
    if time_s < 1e-6:
        return f"{time_s/1e-9:.2f}ns"
    if time_s < 1e-3:
        return f"{time_s/1e-6:.2f}μs"
    if time_s < 1:
        return f"{time_s/1e-3:.2f}ms"
    if time_s < 60:
        return f"{time_s:.2f}s"
    mins = int(time_s/60)
    secs = int(time_s-(mins*60))
    return f"{mins}m {secs}s"
