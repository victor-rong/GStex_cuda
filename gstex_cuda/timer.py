import time
import torch

class Timer:
    """
    Simple timer class that supports timing calls to CUDA kernels.
    """
    def __init__(self, cuda_support=True, disabled=False):
        self.time = time.time()
        self.times = {}
        self.start_events = {}
        self.stop_events = {}
        self.laps = 0
        self.key = None
        self.cuda_support = cuda_support
        self.disabled = disabled

    def start(self, key):
        if self.disabled:
            return
        if self.key is not None and self.cuda_support:
            self.start_events[self.key].pop()
        self.key = key
        if key not in self.times:
            if self.cuda_support:
                self.start_events[key] = []
                self.stop_events[key] = []
            self.times[key] = 0
        if self.cuda_support:
            self.start_events[key].append(torch.cuda.Event(enable_timing=True))
            self.start_events[key][-1].record()
        self.time = time.time()

    def stop(self):
        if self.disabled:
            return
        if self.key is None:
            return

        self.times[self.key] += time.time() - self.time
        if self.cuda_support:
            self.stop_events[self.key].append(torch.cuda.Event(enable_timing=True))
            self.stop_events[self.key][-1].record()
        self.key = None

    def lap(self):
        if self.disabled:
            return
        self.laps += 1

    def dump_key(self, key):
        if self.disabled:
            return
        time = 1000 * self.times[key]
        accurate_time = 0
        if self.cuda_support:
            assert len(self.start_events[key]) == len(self.stop_events[key])
            for i in range(len(self.start_events[key])):
                accurate_time += self.start_events[key][i].elapsed_time(self.stop_events[key][i])
            print(f"{key}\t{accurate_time:.3f}")
        else:
            print(f"{key}\t{time:.3f}")
        return time, accurate_time

    def dump(self):
        if self.disabled:
            return
        torch.cuda.synchronize()
        total = 0
        accurate_total = 0
        for key in self.times.keys():
            time, accurate_time = self.dump_key(key)
            accurate_total += accurate_time
            total += time
        print(f"TOTAL\t{accurate_total:.3f}")