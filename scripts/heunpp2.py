import modules.sd_samplers
from modules import sd_samplers_kdiffusion, sd_samplers_common
from ldm_patched.k_diffusion import sampling as k_diffusion_sampling

class HeunPP2Sampler(sd_samplers_kdiffusion.KDiffusionSampler):
    def __init__(self, sd_model):
        super().__init__(k_diffusion_sampling.sample_heunpp2, sd_model, None)

def build_constructor(model):
    return HeunPP2Sampler(model)

samplers_data_heunpp2 = [
    sd_samplers_common.SamplerData('HeunPP2', build_constructor, ['heunpp2'], {}),
]

modules.sd_samplers.all_samplers.extend(samplers_data_heunpp2)
modules.sd_samplers.all_samplers_map = {x.name: x for x in modules.sd_samplers.all_samplers}
modules.sd_samplers.set_samplers()
