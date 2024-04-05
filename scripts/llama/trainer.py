import torch, os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from trl import SFTTrainer
from transformers.modeling_utils import unwrap_model
from peft.utils import get_peft_model_state_dict
from accelerate import Accelerator

class NiceSFTTrainer(SFTTrainer):
    
    def save_model(
        self, 
        output_dir: Optional[str] = None, 
        _internal_call: bool = False
    ):
        if Accelerator().process_index: return

        if output_dir is None: 
            output_dir = self.args.output_dir

        # If we are executing this function, we are the process zero, so we don't check for that. <-- This is a lie (as least with accelerate launch)
        os.makedirs(output_dir, exist_ok=True)
        self.model.create_or_update_model_card(output_dir)

        # save only the trainable weights
        output_state_dict = get_peft_model_state_dict(
            unwrap_model(self.model), 
            adapter_name="default"
        )

        torch.save(output_state_dict, os.path.join(output_dir, "adapter_model.bin"))
        print(f"----- save model to {output_dir} -----")




