import torch
import numpy as np
from modules import scripts

def create_fixed_schedule(values, n, sigma_min, sigma_max, device, match_minmax = True):
    values = np.array(values)
    
    if match_minmax == True:
        if values[0] < sigma_max:
            print(f"\t Adjusting sigma_max: {values[0]} -> {sigma_max}")
            values[0] = sigma_max  # Ensure first value is sigma_max
            
        if values[-1] > sigma_min:
            print(f"\t Adjusting sigma_min: {values[-1]} -> {sigma_min}")
            values[-1] = sigma_min  # Ensure last value is sigma_min
    
    if n == len(values):
        interpolated = values
    else:
        x_old = np.linspace(0, 1, len(values))
        x_new = np.linspace(0, 1, n)
        interpolated = np.exp(np.interp(x_new, x_old, np.log(values)))
    return torch.tensor(list(interpolated) + [0.0], device=device)

# Defined schedules
SCHEDULES = {
    "stairstep":        [14.615, 12.0, 12.0, 9.0, 9.0, 6.0, 6.0, 3.0, 3.0, 1.5, 1.5, 0.7, 0.7, 0.3, 0.3, 0.1, 0.1, 0.029],
    "simple_loglinear": [14.615, 9.652, 6.375, 4.21, 2.781, 1.837, 1.213, 0.801, 0.529, 0.349, 0.231, 0.152, 0.101, 0.066, 0.044, 0.029],
    "med_3_loglinear":  [14.615, 12.2, 10.184, 8.501, 7.096, 5.923, 4.945, 4.127, 2.726, 1.424, 0.744, 0.389, 0.203, 0.106, 0.055, 0.029],
    "med_1_loglinear":  [14.615, 10.63, 7.731, 5.623, 4.09, 2.975, 2.164, 1.574, 1.039, 0.623, 0.374, 0.224, 0.134, 0.081, 0.048, 0.029],
    "minimalist":       [14.615, 0.029]
}

def fixed_scheduler(n, sigma_min, sigma_max, device, name):
    print(f"\t Schedule: {name}")
    values = SCHEDULES.get(name, [])
    print(f"\t Sigmas: {values}")
    tensor = create_fixed_schedule(values, n, sigma_min, sigma_max, device)
    print("\t Adjusted: [" + ", ".join(["{:.3f}".format(x.item()) for x in tensor]) + "]")

    return tensor

from modules import sd_samplers

class ExtraScheduler(scripts.Script):
    sorting_priority = 99
    installed = False

    def title(self):
        return "Extra Schedulers (Custom)"

    def show(self, is_img2img):
        return scripts.AlwaysVisible if ExtraScheduler.installed else False

try:
    import modules.sd_schedulers as schedulers
    print("Extension: Extra Schedulers: Registering new schedules")

    #schedulers.schedulers = []
    for name in SCHEDULES.keys():
        schedulers.schedulers.append(schedulers.Scheduler(name, name.replace('_', ' ').title(), lambda n, sigma_min, sigma_max, device, name=name: fixed_scheduler(n, sigma_min, sigma_max, device, name)))

    schedulers.schedulers_map = {**{x.name: x for x in schedulers.schedulers}, **{x.label: x for x in schedulers.schedulers}}
    
    ExtraScheduler.installed = True
except Exception as e:
    print("Extension: Extra Schedulers: Unsupported WebUI", str(e))
    ExtraScheduler.installed = False
