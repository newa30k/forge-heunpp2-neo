from modules import sd_samplers_kdiffusion, sd_samplers_common
import modules.sd_samplers

def find_heunpp2():
    candidates = [
        'backend.modules.k_diffusion_extra',
        'backend.k_diffusion.sampling',
        'k_diffusion.sampling',
        'ldm_patched.k_diffusion.sampling',
        'comfy.k_diffusion.sampling',
    ]
    for modname in candidates:
        try:
            mod = __import__(modname, fromlist=['sample_heunpp2'])
            if hasattr(mod, 'sample_heunpp2'):
                print(f"[HeunPP2] Encontrado em: {modname}")
                return getattr(mod, 'sample_heunpp2')
        except ImportError:
            continue
    return None

sample_heunpp2 = find_heunpp2()

if sample_heunpp2 is not None:
    class HeunPP2Sampler(sd_samplers_kdiffusion.KDiffusionSampler):
        def __init__(self, sd_model):
            super().__init__(sample_heunpp2, sd_model, None)

    def build_constructor(model):
        return HeunPP2Sampler(model)

    samplers_data_heunpp2 = [
        sd_samplers_common.SamplerData('HeunPP2', build_constructor, ['heunpp2'], {}),
    ]

    modules.sd_samplers.all_samplers.extend(samplers_data_heunpp2)
    modules.sd_samplers.all_samplers_map = {x.name: x for x in modules.sd_samplers.all_samplers}
    modules.sd_samplers.set_samplers()
    print("[HeunPP2] Sampler registrado com sucesso!")
else:
    print("[HeunPP2] AVISO: não encontrei sample_heunpp2 em nenhum módulo conhecido. Sampler NAO adicionado.")
