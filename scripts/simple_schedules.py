import torch
from modules import scripts

def create_fixed_schedule(values, device):
    return torch.tensor(values + [0.0], device=device)

# Defined schedules
SCHEDULES = {
    "stairstep": [14.615, 12.0, 12.0, 9.0, 9.0, 6.0, 6.0, 3.0, 3.0, 1.5, 1.5, 0.7, 0.7, 0.3, 0.3, 0.1, 0.1, 0.029],
    "simple_loglinear": [14.615, 9.652, 6.375, 4.21, 2.781, 1.837, 1.213, 0.801, 0.529, 0.349, 0.231, 0.152, 0.101, 0.066, 0.044, 0.029],
    "med_3_loglinear": [14.615, 12.2, 10.184, 8.501, 7.096, 5.923, 4.945, 4.127, 2.726, 1.424, 0.744, 0.389, 0.203, 0.106, 0.055, 0.029],
    "med_1_loglinear": [14.615, 10.63, 7.731, 5.623, 4.09, 2.975, 2.164, 1.574, 1.039, 0.623, 0.374, 0.224, 0.134, 0.081, 0.048, 0.029],
    "minimalist": [14.615, 0.029]
}

def fixed_scheduler(name, n, sigma_min, sigma_max, device):
    values = SCHEDULES.get(name, [])
    return create_fixed_schedule(values, device)

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

    schedulers.schedulers = []
    for name in SCHEDULES.keys():
        schedulers.schedulers.append(schedulers.Scheduler(name, name.replace('_', ' ').title(), lambda n, sigma_min, sigma_max, device, name=name: fixed_scheduler(name, n, sigma_min, sigma_max, device)))
    
    schedulers.schedulers_map = {x.name: x for x in schedulers.schedulers}
    sd_samplers.set_samplers()
    ExtraScheduler.installed = True
except Exception as e:
    print("Extension: Extra Schedulers: Unsupported WebUI", str(e))
    ExtraScheduler.installed = False
