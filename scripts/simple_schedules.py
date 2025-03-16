import torch
import numpy as np
from modules import scripts

INSTALLED = False
# Defined schedules
SCHEDULES = {
    "minimalist":       [14.615, 0.029],
    "med-1":            [14.615, 10.63, 7.731, 5.623, 4.09, 2.975, 2.164, 1.574, 1.039, 0.623, 0.374, 0.224, 0.134, 0.081, 0.048, 0.029],
}

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
    
def create_fixed_schedule_linear(values, n, sigma_min, sigma_max, device, match_minmax = True):
    values = np.array(values)
    
    if match_minmax == True:
        if values[0] < sigma_max:
            print(f"\t Adjusting sigma_max: {values[0]} -> {sigma_max}")
            values[0] = sigma_max  # Ensure first value is sigma_max
            
        if values[-1] > sigma_min:
            print(f"\t Adjusting sigma_min: {values[-1]} -> {sigma_min}")
            values[-1] = sigma_min  # Ensure last value is sigma_min
    if n == len(values):
        sigmas = values
    else:
        sigmas = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(values)), values)
    return torch.tensor(list(sigmas) + [0.0], device=device)

def fixed_scheduler(n, sigma_min, sigma_max, device, name):
    print(f"\t Schedule: {name} Loglinear")
    values = SCHEDULES.get(name, [])
    print(f"\t Sigmas: {values}")
    tensor = create_fixed_schedule(values, n, sigma_min, sigma_max, device)
    print("\t Adjusted: [" + ", ".join(["{:.3f}".format(x.item()) for x in tensor]) + "]")
    return tensor
    
def fixed_scheduler_linear(n, sigma_min, sigma_max, device, name):
    print(f"\t Schedule: {name} Linear")
    values = SCHEDULES.get(name, [])
    print(f"\t Sigmas: {values}")
    tensor = create_fixed_schedule_linear(values, n, sigma_min, sigma_max, device)
    print("\t Adjusted: [" + ", ".join(["{:.3f}".format(x.item()) for x in tensor]) + "]")
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
        print("Extension: Minimalist Scheduler: Registering new schedules")

        #schedulers.schedulers = []
        for name in SCHEDULES.keys():
            schedulers.schedulers.append(schedulers.Scheduler(
                name+" LOG",
                (name+" LOG").replace('_', ' ').title(),
                lambda n, sigma_min, sigma_max, device, name=name: fixed_scheduler(n, sigma_min, sigma_max, device, name)
            ))
            schedulers.schedulers.append(schedulers.Scheduler(
                name+" LIN",
                (name+" LIN").replace('_', ' ').title(),
                lambda n, sigma_min, sigma_max, device, name=name: fixed_scheduler_linear(n, sigma_min, sigma_max, device, name)
            ))
            
        schedulers.schedulers_map = {**{x.name: x for x in schedulers.schedulers}, **{x.label: x for x in schedulers.schedulers}}
        INSTALLED = True

except Exception as e:
    print("Extension: Minimalist Schedulers: Unsupported WebUI", str(e))
    INSTALLED = False
