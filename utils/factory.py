def get_model(model_name, args):
    name = model_name.lower()
    if name=="simplecil":
        from models.simplecil import Learner
        return Learner(args)
    elif name=="adam_finetune":
        from models.adam_finetune import Learner
        return Learner(args)
    elif name=="adam_ssf":
        from models.adam_ssf import Learner
        return Learner(args)
    elif name=="adam_vpt":
        from models.adam_vpt import Learner
        return Learner(args) 
    elif name=="adam_adapter":
        from models.adam_adapter import Learner
        return Learner(args)
    elif name=="adam_adapter_lora":
        from models.adam_adapter_LoRA import Learner
        return Learner(args)
    elif name=="adam_adapter_lora_fc":
        from models.adam_adapter_LoRA_fc import Learner
        return Learner(args)
    elif name=="adam_adapter_lora_fc_linear":
        from models.adam_adapter_LoRA_fc_linear import Learner
        return Learner(args)
    elif name=="adam_emsemble":
        from models.adam_ensemble import Learner
        return Learner(args)
    elif name=="adam_emsemble_linear":
        from models.adam_ensemble_linear import Learner
        return Learner(args)
    else:
        assert 0
