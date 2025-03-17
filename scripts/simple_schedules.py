import torch
import numpy as np
from modules import scripts

LOG_DEBUG = True
INSTALLED = False
# Defined schedules
SCHEDULES = {
    ("minimalist", "Log,Lin"):  [14.615, 0.029],
    ("med-3", "Log"):           [14.615, 3, 0.029],
    ("med-1", "Log"):           [14.615, 1, 0.029],
    ("med-1/10", "Log"):        [14.615, 0.1, 0.029],
}

MIN_DROPOFF_STEPS = 3
MAX_DROPOFF_RATIO = 0.2
LINEAR_SMOOTHING_STEPS = 3

def debug_print(message):
    if LOG_DEBUG:
        print(message)

def create_fixed_schedule(values, n, sigma_min, sigma_max, device, match_minmax=True):
    # Ensure that values is a numpy array
    values = np.array(values)
    isZSNR = False
    ogSigmaMax = values[0]
    ogSigmaMin = values[-1]
    ogValues = values.copy()

    # Adjust sigma_max and sigma_min if needed
    if match_minmax:
        if values[0] < sigma_max:
            debug_print(f"\t Adjusting sigma_max: {values[0]} -> {sigma_max}")
            values[0] = sigma_max  # Ensure first value is sigma_max
            isZSNR = True
            
        if values[-1] > sigma_min:
            debug_print(f"\t Adjusting sigma_min: {values[-1]} -> {sigma_min}")
            values[-1] = sigma_min  # Ensure last value is sigma_min

    # Handle zero-term SNR (zsnr) case (high sigma_max)
    if isZSNR:
        debug_print(f"\t Very noisy model detected, forcing schedule to drop quickly to {ogSigmaMax}.")
        
        dropoff_steps = max(MIN_DROPOFF_STEPS, int(n * MAX_DROPOFF_RATIO))  # Hard drop withing the first few steps, proportional
        remaining_steps = n - dropoff_steps
        
        if remaining_steps < 1:
            # Fallback if n is extremely small
            sigmas = np.exp(np.linspace(np.log(sigma_max), np.log(sigma_min), n))
        else:
            # Generate the schedule with a quick drop followed by the normal loglinear progression
            drop_schedule = np.exp(np.linspace(np.log(sigma_max), np.log(ogSigmaMax), dropoff_steps))       
            # Create a domain for the original values:
            x_old = np.linspace(0, 1, len(values))
            # For the remaining portion, generate a new domain covering [0, 1] with (remaining_steps+1) points,
            # and skip the first to avoid duplicating the drop endpoint.
            remaining_schedule = np.exp(np.interp(np.linspace(0, 1, remaining_steps + 1)[1:], x_old, np.log(ogValues)))
     
            # Combine the two parts of the schedule
            sigmas = np.concatenate([drop_schedule, remaining_schedule[1:]])
    else:
        # Fallback to normal loglinear interpolation
        if n == len(values):
            sigmas = values
        else:
            x_old = np.linspace(0, 1, len(values))
            x_new = np.linspace(0, 1, n - 1)
            sigmas = np.exp(np.interp(x_new, x_old, np.log(values)))
    
    return torch.tensor(list(sigmas) + [0.0], device=device)
    
def create_fixed_schedule_linear(values, n, sigma_min, sigma_max, device, match_minmax=True):
    # Ensure that values is a numpy array
    values = np.array(values)
    isZSNR = False
    ogSigmaMax = values[0]
    ogSigmaMin = values[-1]
    ogValues = values.copy()

    # Adjust sigma_max and sigma_min if needed
    if match_minmax:
        if values[0] < sigma_max:
            debug_print(f"\t Adjusting sigma_max: {values[0]} -> {sigma_max}")
            values[0] = sigma_max  # Ensure first value is sigma_max
            isZSNR = True
            
        if values[-1] > sigma_min:
            debug_print(f"\t Adjusting sigma_min: {values[-1]} -> {sigma_min}")
            values[-1] = sigma_min  # Ensure last value is sigma_min

    # Handle zero-term SNR (zsnr) case (high sigma_max)
    if isZSNR:
        debug_print(f"\t Very noisy model detected, forcing schedule to drop quickly to around {ogSigmaMax}.")
        
        dropoff_steps = max(MIN_DROPOFF_STEPS, int(n * MAX_DROPOFF_RATIO)) 
        remaining_steps = n - dropoff_steps
        
        if remaining_steps < 1:
            # Fallback if n is extremely small
            sigmas = np.exp(np.linspace(np.log(sigma_max), np.log(sigma_min), n))
        else:
            # Generate the schedule with a quick drop followed by the normal linear progression
            drop_schedule = np.exp(np.linspace(np.log(sigma_max), np.log(ogSigmaMax), dropoff_steps))

            # Create a domain for the original values:
            x_old = np.linspace(0, 1, len(values))
            # For the remaining portion, generate a new domain covering [0, 1] with (remaining_steps+1) points,
            # and skip the first to avoid duplicating the drop endpoint.
            remaining_schedule = np.interp(np.linspace(0, 1, remaining_steps + 1)[1:], x_old, ogValues)
        
            # Combine the two parts of the schedule
            sigmas = np.concatenate([drop_schedule, remaining_schedule[1:]])
    else:
        # Fallback to normal linear interpolation
        if n == len(values):
            sigmas = values
        else:
            sigmas = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(values)), values)

    # Final region smoothing: replace the last few steps with a loglinear decay from the current value to sigma_min.
    start_index = max(0, n - LINEAR_SMOOTHING_STEPS - 1)
    start_val = sigmas[start_index]
    new_final = np.exp(np.linspace(np.log(start_val), np.log(sigma_min), LINEAR_SMOOTHING_STEPS + 1))[1:]
    sigmas = np.concatenate([sigmas[:start_index], new_final])
    
    return torch.tensor(list(sigmas) + [0.0], device=device)

def fixed_scheduler(n, sigma_min, sigma_max, device, sched_key):
    debug_print(f"\t Schedule: {sched_key} Loglinear")
    # sched_key can be a tuple or string
    if isinstance(sched_key, (list, tuple)):
        sched_name = sched_key[0]
    else:
        sched_name = sched_key
    # Get the values from SCHEDULES using the key.
    # If sched_key is a tuple, use it; if it's a string, use it directly.
    values = SCHEDULES.get(sched_key, None)
    if values is None:
        # Fallback: try using the sched_name as key.
        values = SCHEDULES.get(sched_name, [])
    debug_print(f"\t Sigmas: {values}")
    tensor = create_fixed_schedule(values, n, sigma_min, sigma_max, device)
    debug_print("\t Adjusted: [" + ", ".join(["{:.3f}".format(x.item()) for x in tensor]) + f"] (Total: {len(tensor)})")
    return tensor
    
def fixed_scheduler_linear(n, sigma_min, sigma_max, device, sched_key):
    debug_print(f"\t Schedule: {sched_key} Linear")
    if isinstance(sched_key, (list, tuple)):
        sched_name = sched_key[0]
    else:
        sched_name = sched_key
    values = SCHEDULES.get(sched_key, None)
    if values is None:
        values = SCHEDULES.get(sched_name, [])
    debug_print(f"\t Sigmas: {values}")
    tensor = create_fixed_schedule_linear(values, n, sigma_min, sigma_max, device)
    debug_print("\t Adjusted: [" + ", ".join(["{:.3f}".format(x.item()) for x in tensor]) + f"] (Total: {len(tensor)})")
    return tensor

from modules import sd_samplers

class MinimalScheduler(scripts.Script):
    sorting_priority = 99
    
    def title(self):
        return "Minimalist Scheduler"

    def show(self, is_img2img):
        return scripts.AlwaysVisible if INSTALLED else False

try:
    if INSTALLED != True:
        import modules.sd_schedulers as schedulers
        debug_print("Extension: Minimalist Scheduler: Registering new schedules")

        # Iterate over SCHEDULES items and register only the requested types.
        for key in SCHEDULES.keys():
            if isinstance(key, (list, tuple)):
                sched_name = key[0]
                types = key[1].split(",")
                types = [t.strip().lower() for t in types]
            else:
                sched_name = key
                types = ["log", "lin"]
            
            if "log" in types:
                schedulers.schedulers.append(schedulers.Scheduler(
                    sched_name + " LOG",
                    (sched_name + " LOG").replace('_', ' ').title(),
                    lambda n, sigma_min, sigma_max, device, name=key: fixed_scheduler(n, sigma_min, sigma_max, device, name)
                ))
            if "lin" in types:
                schedulers.schedulers.append(schedulers.Scheduler(
                    sched_name + " LIN",
                    (sched_name + " LIN").replace('_', ' ').title(),
                    lambda n, sigma_min, sigma_max, device, name=key: fixed_scheduler_linear(n, sigma_min, sigma_max, device, name)
                ))
            
        schedulers.schedulers_map = {**{x.name: x for x in schedulers.schedulers}, **{x.label: x for x in schedulers.schedulers}}
        INSTALLED = True

except Exception as e:
    debug_print("Extension: Minimalist Schedulers: Unsupported WebUI", str(e))
    INSTALLED = False
