import os
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple 
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer 
from ..rome.layer_stats import layer_stats
from ...util import nethook
from ...util.generate import generate_fast
from ...util.globals import * 
from .compute_ks import compute_ks
from .compute_z import compute_z, get_module_input_output_at_words, find_fact_lookup_idx
from .AlphaEdit_hparams import AlphaEditHyperParams 
import torch.nn as nn
import torch.nn.functional as F

# Cache variable(s)
CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}
counter = 0
P_loaded = False
cache_c_new = False

def apply_AlphaEdit_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: AlphaEditHyperParams,
    copy=False,
    return_orig_weights=False,
    cache_template: Optional[str] = None,
    keep_original_weight=False,
    **kwargs
) -> Dict[str, Tuple[torch.Tensor]]:
  #-> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Returns a model with the desired changes.
    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.
    :return: (1) the updated model, (2) an original copy of the weights that changed
    """

    global P, P_loaded, cache_c, cache_c_new 

    weights_copy = {}
    if copy:
        model = deepcopy(model)
    
    # Calculate the null-space projection matrix P
    # Please ensure that you have downloaded "null_space_project.pt" to the easyedit folder beforehand, or get the P by following calculation
    if not os.path.exists(hparams.P_loc):
        print(f"The null-space projection matrix P does not exist and now calculate.")
        W_out = nethook.get_parameter(model, f"{hparams.rewrite_module_tmp.format(hparams.layers[-1])}.weight")
        if "llama" in hparams.model_name.lower() or "gpt-j-6b" in hparams.model_name.lower():
            P = torch.zeros((len(hparams.layers), W_out.shape[1], W_out.shape[1]), device="cpu")
        elif "gpt2-xl" in hparams.model_name.lower():
            P = torch.zeros((len(hparams.layers), W_out.shape[0], W_out.shape[0]), device="cpu")
        del W_out
        for i, layer in enumerate(hparams.layers):
            P[i,:,:] = get_project(model, tok, layer, hparams)
        # torch.save(P, "null_space_project.pt")
        model_name = hparams.model_name.split("/")[-1]  # 提取模型名最后一部分
        filename = f"/home/liujinzhe/code/KnowledgeEdit/AlphaEdit/examples/outputs/prefile/AlphaEdit/P_loc/{model_name}_null_space_project.pt" 
        torch.save(P, filename)
        print("Saved to:", filename)
        P_loaded = True
    elif P_loaded == False:
        P = torch.load(hparams.P_loc)
        P_loaded = True

    # Maintain the global variable cache_c to avoid redundant computations.
    # If this is the first calculation (i.e., cache_c_new == false), then initialize cache_c first
    if not cache_c_new:
        W_out = nethook.get_parameter(model, f"{hparams.rewrite_module_tmp.format(hparams.layers[-1])}.weight")
        if "llama" in hparams.model_name.lower() or "gpt-j-6b" in hparams.model_name.lower():
            cache_c = torch.zeros((len(hparams.layers), W_out.shape[1], W_out.shape[1]), device="cpu")
        elif "gpt2-xl" in hparams.model_name.lower():
            cache_c = torch.zeros((len(hparams.layers), W_out.shape[0], W_out.shape[0]), device="cpu")
        del W_out
        cache_c_new = True
    
    deltas = execute_AlphaEdit(model, tok, requests, hparams, cache_template=cache_template)

    with torch.no_grad():
        for w_name, upd_m in deltas.items():
            upd_matrix = upd_m.to(f"cuda:{hparams.device}")
            w = nethook.get_parameter(model, w_name)
            upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)

            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()
            w[...] += upd_matrix.float()

    print(f"New weights successfully inserted into {list(deltas.keys())}")

    return model, weights_copy


def execute_AlphaEdit(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: AlphaEditHyperParams,
    cache_template: Optional[str] = None,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the AlphaEdit update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """
    global counter
    deltas = {}

    # Update target and print info
    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        if request["target_new"][0] != " ":
            # Space required for correct tokenization
            requests[i]["target_new"] = " " + request["target_new"]
        if '{}' not in request['prompt']:
            assert request['subject'] in request['prompt'] or \
                   print(f"Subject:{request['subject']} do not exist in prompt: {request['prompt']}")
        requests[i]['prompt'] = requests[i]['prompt'].replace(requests[i]['subject'], '{}')
        print(
            f"Executing AlphaEdit algo for: "
            f"[{request['prompt']}] -> [{request['target_new']}]"
        )

    # Retrieve weights that user desires to change
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }

    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    # Compute z for final layer
    context_templates = get_context_templates(model, tok)
    z_layer = hparams.layers[-1]
    z_list = []

    for request in requests:
        # Retrieve k/v pair if already stored in cache
        cache_fname = (
            Path(
                str(cache_template).format(
                    z_layer, hparams.clamp_norm_factor, request["case_id"]
                )
            )
            if cache_template is not None
            else None
        )
        data_loaded = False
        if (
            cache_fname is not None  # Require cache template
            and cache_fname.exists()  # Cache file must exist
        ):
            try:
                data = np.load(cache_fname)
                z_list.append(torch.from_numpy(data["v_star"]).to(f"cuda:{hparams.device}"))
                data_loaded = True
            except Exception as e:
                print(f"Error reading cache file due to {e}. Recomputing...")

        # Compute k/v pair if not loaded from cache
        if not data_loaded:
            cur_z = compute_z(
                model,
                tok,
                request,
                hparams,
                z_layer,
                context_templates,
            )

            z_list.append(cur_z) # torch.Size([4096])

            if cache_fname is not None:
                cache_fname.parent.mkdir(exist_ok=True, parents=True)
                np.savez(
                    cache_fname,
                    **{
                        "v_star": cur_z.detach().cpu().numpy(),
                    },
                )
                print(f"Cached k/v pair at {cache_fname}")
    zs = torch.stack(z_list, dim=1)

    # print("Computing neuron attribution masks...")
    importance_masks = {}
 
    input_prompts = [context.format(request["prompt"].replace("{}", request["subject"])) for request in requests for context_list in context_templates for context in context_list]
    input_tok = tok(input_prompts, return_tensors="pt", padding=True).to(f"cuda:{hparams.device}")
 
    with torch.no_grad():    
        model_output = model(**input_tok) 
    
    # 提前释放 tokenizer tensor
    del input_tok
    torch.cuda.empty_cache()
# # ================================================================================================================
    # #  0-1_mask
    # for layer in hparams.layers: 
    #     score_matrix = get_importance_scores(model_output, model, layer, hparams)
    #     pooled_score = score_matrix.max(dim=0).values  # [11008]  
    #     threshold = torch.quantile(pooled_score, 1 - 0.7)
    #     mask = (pooled_score >= threshold).float()
    #     importance_masks[layer] = mask
    #     print(f"Layer {layer} neuron mask: {int(mask.sum().item())}/{len(mask)} neurons kept.")
    #     print("====== Layer {} attribution stats ======".format(layer))
    #     print("Raw score_matrix shape:", score_matrix.shape)
    #     print("Score matrix mean/std: {:.4f} / {:.4f}".format(score_matrix.mean().item(), score_matrix.std().item()))
    #     print("Score matrix min/max: {:.4f} / {:.4f}".format(score_matrix.min().item(), score_matrix.max().item()))
    #     print("Example scores (first 5 neurons across prompts):")
    #     print(score_matrix[:, :5])  # shape [n_prompts, 5]
    #     print("Pooled max per neuron (first 10):", pooled_score[:10])
    #     print("Final mask: {}/{} neurons kept.".format(int(mask.sum().item()), len(mask)))

# # ================================================================================================================
    # #  0-1_mask_resonant  
    # for layer in hparams.layers: 
    #     score_matrix = get_importance_scores(model_output, model, layer, hparams) 
    #     mask, resonance_counts, burst_score, resonance_cut, burst_cut = compute_hybrid_resonant_mask(
    #         score_matrix, resonance_ratio=0.35, burst_ratio=0.25,
    #         use_resonance=False,     
    #         use_burst=True)       
    #     importance_masks[layer] = mask 
    #     total_neurons = score_matrix.shape[1]
    #     total_mask = int(mask.sum().item())   
    #     mask_activation = mask.sum().item() / total_neurons 
    use_resonance = True
    use_burst = True
    for layer in hparams.layers:
        if 'llama' in hparams.model_name.lower():
            score_matrix = get_importance_scores(model_output, model, layer, hparams)
        if 'gpt' in hparams.model_name.lower():
            score_matrix = get_gpt_importance_scores(model_output, model, layer, hparams)
        
        # # 获取动态的掩码比例
        resonance_ratio, burst_ratio = entropy_adaptive_mask_ratio(score_matrix)
        # resonance_ratio = 0.5
        # burst_ratio = 0.3
        # 选择是否仅使用共振神经元或爆发神经元
        final_mask, resonance_mask, burst_mask, resonance_counts, burst_score, resonance_cut, burst_cut = compute_hybrid_resonant_mask(
            score_matrix, resonance_ratio=resonance_ratio, burst_ratio=burst_ratio,
            use_resonance=use_resonance,     # 仅使用共振神经元
            use_burst=use_burst)        # 不使用爆发神经元 
        importance_masks[layer] = final_mask

        del score_matrix, final_mask, resonance_mask, burst_mask
        del resonance_counts, burst_score
        torch.cuda.empty_cache()
    
    del model_output
    torch.cuda.empty_cache()
        # total_neurons = score_matrix.shape[1]   
        # resonance_neurons = int(resonance_mask.sum().item())   
        # burst_neurons = int(burst_mask.sum().item())  
        # overlapping_neurons = int((resonance_mask * burst_mask).sum().item())  
        # final_activated_neurons = int(final_mask.sum().item()) 
        # resonance_ratio_in_layer = resonance_neurons / total_neurons   
        # burst_ratio_in_layer = burst_neurons / total_neurons   
        # overlap_ratio_in_layer = overlapping_neurons / total_neurons
        # final_mask_ratio_in_layer = final_activated_neurons / total_neurons 
        # print("resonance_ratio_in_layer: ", resonance_ratio_in_layer)  
        # print("burst_ratio_in_layer: ", burst_ratio_in_layer)
        # print("overlap_ratio_in_layer: ", overlap_ratio_in_layer) 
        # print("final_mask_ratio_in_layer: ", final_mask_ratio_in_layer) 

# # ================================================================================================================ 
    # Insert
    for i, layer in enumerate(hparams.layers):
        print(f"\n\nLAYER {layer}\n")

        # Get current model activations   layer_ks torch.Size([11008, 1])
        layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
        print(f"Writing {layer_ks.size(1)} key/value pair(s) into layer {layer}")

        # Compute residual error  # cur_zs  torch.Size([4096, 1])
        cur_zs = get_module_input_output_at_words(
            model,
            tok,
            z_layer,
            context_templates=[request["prompt"] for request in requests],
            words=[request["subject"] for request in requests],
            module_template=hparams.layer_module_tmp,
            fact_token_strategy=hparams.fact_token,
        )[1].T
        targets = zs - cur_zs   # torch.Size([4096, 1])
        print("z error", torch.linalg.norm(targets, dim=0).mean())

        repeat_factor = (layer_ks.size(1) // targets.size(1))
        targets = targets.repeat_interleave(repeat_factor, dim=1) 
        resid = targets / (len(hparams.layers) - i)  # Distribute residual across layers 
        # post mask
        upd_matrix = torch.linalg.solve(
                P[i,:,:].to(f"cuda:{hparams.device}") @ (layer_ks.to(f"cuda:{hparams.device}") @ layer_ks.T.to(f"cuda:{hparams.device}") + cache_c[i,:,:].to(f"cuda:{hparams.device}")) + hparams.L2*torch.eye(layer_ks.shape[0], dtype=torch.float,device=f"cuda:{hparams.device}"),
                P[i,:,:].to(f"cuda:{hparams.device}") @ layer_ks.to(f"cuda:{hparams.device}") @ resid.T.to(f"cuda:{hparams.device}")
        )  # upd_matrix shape = torch.Size([ , 4096]) 
        # Apply neuron mask  
        neuron_mask = importance_masks[layer].to(upd_matrix.device)  # shape: [d1]
        upd_matrix = neuron_mask[:, None] * upd_matrix  # apply to rows only
        # Adjust update matrix shape
        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)  # upd_matrix shape = torch.Size([4096, 11008])

        print("orig norm", torch.linalg.norm(weights[weight_name]))
        print("upd norm", torch.linalg.norm(upd_matrix))

        # Update model weights and record desired changes in `delta` variable
        with torch.no_grad():
            weights[weight_name][...] = weights[weight_name] + upd_matrix.float()
            deltas[weight_name] = (upd_matrix.detach().cpu())
        
        # Clear GPU memory
        #del U,S,cov
        for x in [layer_ks, cur_zs, targets]:
            x.cpu()
            del x
        torch.cuda.empty_cache()
    
    for i, layer in enumerate(hparams.layers):
        layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
        cache_c[i,:,:] += layer_ks.cpu() @ layer_ks.cpu().T

    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]
    
    print(f"Deltas successfully computed for {list(weights.keys())}")

    return deltas

def get_cov(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer_name: str,
    mom2_dataset: str,
    mom2_n_samples: str,
    mom2_dtype: str,
    inv: bool = False,
    force_recompute: bool = False,
    hparams=None,
) -> torch.Tensor:
    """
    Retrieves covariance statistics, then computes the algebraic inverse.
    Caches result for future use.
    """

    model_name = model.config._name_or_path.replace("/", "_")
    key = (model_name, layer_name)

    print(f"Retrieving covariance statistics for {model_name} @ {layer_name}.")
    if key not in COV_CACHE or force_recompute:
        stat = layer_stats(
            model,
            tok,
            layer_name,
            hparams.stats_dir,
            mom2_dataset,
            to_collect=["mom2"],
            sample_size=mom2_n_samples,
            precision=mom2_dtype,
            hparams=hparams,
            force_recompute=force_recompute,
        )
        COV_CACHE[key] = stat.mom2.moment().float().to("cpu")

    return (
        torch.inverse(COV_CACHE[key].to(f"cuda:{hparams.device}")) if inv else COV_CACHE[key].to(f"cuda:{hparams.device}")
    )
 
def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    GPT-2 and GPT-J have transposed weight representations.
    Returns a matrix that matches the desired shape, else raises a ValueError
    """

    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError(
            "Update matrix computed by AlphaEdit does not match original weight shape. "
            "Check for bugs in the code?"
        )
 
def get_context_templates(model, tok):
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = [["{}"]] + [
            [
                f.replace("{", " ").replace("}", " ") + ". {}"
                for f in generate_fast(
                    model,
                    tok,
                    ["The", "Therefore", "Because", "I", "You"],
                    n_gen_per_prompt=n_gen // 5,
                    max_out_len=length,
                )
            ]
            for length, n_gen in [(10, 5)]  # Be careful about changing this.
        ]
        print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE

def get_project(model, tok, layer, hparams):
    force_recompute = False
    cov = get_cov(
        model,
        tok,
        hparams.rewrite_module_tmp.format(layer),
        hparams.mom2_dataset,
        hparams.mom2_n_samples
        if not force_recompute
        else hparams.mom2_n_samples // 10,
        hparams.mom2_dtype,
        force_recompute=force_recompute,
        hparams=hparams
    ).cpu()
    U, S, _ = torch.linalg.svd(cov, full_matrices=False)
    threshold = hparams.nullspace_threshold
    small_singular_indices = (S < threshold).nonzero(as_tuple=True)[0]
    print(len(small_singular_indices))
    return U[:, small_singular_indices] @ U[:, small_singular_indices].T


def get_importance_scores(model_output, model, layer, hparams):
    def get_prob(vector):    return torch.nn.Softmax(dim=-1)(vector)
    def get_bsvalues(vector, model, final_var):
        vector = vector * torch.rsqrt(final_var + 1e-6)
        vector = vector.to(model.model.norm.weight.device)
        vector_rmsn = vector * model.model.norm.weight
        return model.lm_head(vector_rmsn)
        # vector_rmsn = vector * model.model.norm.weight.data
        # vector_bsvalues = model.lm_head(vector_rmsn).data
        # return vector_bsvalues
 
    predicted_list = [model_output[0][i][-1] for i in range(model_output[0].shape[0])]
    predicted_top10_list = [torch.argsort(predicted, descending=True)[:10] for predicted in predicted_list]
    predicted_indices_list = [predicted_indices[0].item() for predicted_indices in predicted_top10_list]
 
    # 提取第 layer 层的所有batch信息
    residual_outputs = model_output[-1][layer][1].tolist()  # 保留所有batch
    coefficient_scores_all = model_output[-1][layer][0].tolist()  # 保留所有batch 
    final_var_list = model_output[-1][-1][2].tolist()  # 保留所有batch
 
    def safe_log_prob(vector, eps=1e-8):
        return torch.log(torch.clamp(vector, min=eps, max=1-eps))
    
    AMPLIFY_FACTOR = 30  # 从100调整为10
    importance_scores_all = []
    for batch_idx in range(len(residual_outputs)):
        # final_var = torch.tensor(final_var_list[batch_idx][-1], dtype=torch.float).pow(2).mean(-1, keepdim=True)
        # final_var = final_var.cuda()
        final_var_tensor = torch.tensor(final_var_list[batch_idx][-1], dtype=torch.float32).cuda()
        final_var = final_var_tensor.pow(2).mean(-1, keepdim=True)
        coefficient_scores = torch.tensor(coefficient_scores_all[batch_idx][-1], dtype=torch.float)
        fc2_vectors = model.model.layers[layer].mlp.down_proj.weight.data
        fc2_vectors = fc2_vectors.cuda()
        coefficient_scores = coefficient_scores.cuda()
        ffn_subvalues = (coefficient_scores * fc2_vectors).T    # ffn_subvalues [11008, 4096]
        cur_residual = torch.tensor(residual_outputs[batch_idx][-1], dtype=torch.float)
        cur_residual = cur_residual.cuda()
        with torch.no_grad():
            origin_prob = get_prob(get_bsvalues(cur_residual, model, final_var))[predicted_indices_list[batch_idx]]
            cur_ffn_subvalues_plus = ffn_subvalues * AMPLIFY_FACTOR + cur_residual
            cur_ffn_subvalues_probs = get_prob(get_bsvalues(cur_ffn_subvalues_plus, model, final_var))[:, predicted_indices_list[batch_idx]]
            origin_prob = origin_prob.clamp(min=1e-8, max=1.0)
            cur_ffn_subvalues_probs = cur_ffn_subvalues_probs.clamp(min=1e-8, max=1.0)
            importance_scores = safe_log_prob(cur_ffn_subvalues_probs) - safe_log_prob(origin_prob)
        importance_scores_all.append(importance_scores)
 
    # importance_scores_all = torch.stack(importance_scores_all).to(f"cuda:{hparams.device}")
    importance_scores_all = torch.stack(importance_scores_all).to(f"cuda:{hparams.device}")
    importance_scores_all = torch.nan_to_num(importance_scores_all, nan=0.0, posinf=0.0, neginf=0.0)
    # print("Importance scores statistics:")
    # print(f"Mean: {importance_scores_all.mean().item():.4f}, Std: {importance_scores_all.std().item():.4f}, "
    #       f"Min: {importance_scores_all.min().item():.4f}, Max: {importance_scores_all.max().item():.4f}")
    return importance_scores_all   
 
def get_gpt_importance_scores(model_output, model, layer, hparams):
    def get_prob(vector):    return torch.nn.Softmax(dim=-1)(vector) 
    def get_bsvalues(vector, model, final_var):
        E = torch.mean(vector, -1)
        vector_ln = (vector-E.unsqueeze(-1))/final_var * model.transformer.ln_f.weight.data
        vector_bsvalues = model.lm_head(vector_ln).data
        return vector_bsvalues
 
    predicted_list = [model_output[0][i][-1] for i in range(model_output[0].shape[0])]
    predicted_top10_list = [torch.argsort(predicted, descending=True)[:10] for predicted in predicted_list]
    predicted_indices_list = [predicted_indices[0].item() for predicted_indices in predicted_top10_list]
  
    residual_outputs = model_output[-1][layer][1].tolist()   
    coefficient_scores_all = model_output[-1][layer][0].tolist()  
    layer_outputs = model_output[-1][layer][2].tolist()   
    final_var_list = model_output[-1][-1][2].tolist()   
 
    def safe_log_prob(vector, eps=1e-8):
        return torch.log(torch.clamp(vector, min=eps, max=1-eps))
    
    AMPLIFY_FACTOR = 30  
    importance_scores_all = []
    for batch_idx in range(len(residual_outputs)):
        # final_var = torch.tensor(final_var_list[batch_idx][-1], dtype=torch.float32).pow(2).mean(-1, keepdim=True)
        # final_var = torch.tensor(all_pos_layer_output[-1][-1]).pow(2).mean(-1, keepdim=True) 
        final_var_tensor = torch.tensor(final_var_list[batch_idx][-1], dtype=torch.float32).cuda()
        final_var = torch.var(final_var_tensor, dim=-1, unbiased=False, keepdim=True).sqrt() + 1e-5

        # final_var = ((torch.var(torch.tensor(final_var_list[batch_idx][-1], dtype=torch.float), -1, unbiased=False)+1e-5)**0.5).item()
        # final_var = final_var.cuda()
        coefficient_scores = torch.tensor(coefficient_scores_all[batch_idx][-1], dtype=torch.float) 
        fc2_vectors = model.transformer.h[layer].mlp.c_proj.weight.data.T
        fc2_vectors = fc2_vectors.cuda()
        coefficient_scores = coefficient_scores.cuda()
        ffn_subvalues = (coefficient_scores * fc2_vectors).T    # ffn_subvalues [11008, 4096]
        cur_residual = torch.tensor(residual_outputs[batch_idx][-1], dtype=torch.float)
        cur_residual = cur_residual.cuda()
        with torch.no_grad():
            origin_prob = get_prob(get_bsvalues(cur_residual, model, final_var))[predicted_indices_list[batch_idx]]
            cur_ffn_subvalues_plus = ffn_subvalues * AMPLIFY_FACTOR + cur_residual
            cur_ffn_subvalues_probs = get_prob(get_bsvalues(cur_ffn_subvalues_plus, model, final_var))[:, predicted_indices_list[batch_idx]]
            origin_prob = origin_prob.clamp(min=1e-8, max=1.0)
            cur_ffn_subvalues_probs = cur_ffn_subvalues_probs.clamp(min=1e-8, max=1.0)
            importance_scores = safe_log_prob(cur_ffn_subvalues_probs) - safe_log_prob(origin_prob)
        importance_scores_all.append(importance_scores)
 
    # importance_scores_all = torch.stack(importance_scores_all).to(f"cuda:{hparams.device}")
    importance_scores_all = torch.stack(importance_scores_all).to(f"cuda:{hparams.device}")
    importance_scores_all = torch.nan_to_num(importance_scores_all, nan=0.0, posinf=0.0, neginf=0.0)
    # print("Importance scores statistics:")
    # print(f"Mean: {importance_scores_all.mean().item():.4f}, Std: {importance_scores_all.std().item():.4f}, "
    #       f"Min: {importance_scores_all.min().item():.4f}, Max: {importance_scores_all.max().item():.4f}")
    return importance_scores_all   

def compute_hybrid_resonant_mask(score_matrix: torch.Tensor, 
                                 resonance_ratio: float = 0.25,
                                 burst_ratio: float = 0.15,
                                 use_resonance: bool = True,    
                                 use_burst: bool = True        
                                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # 1. Z-score normalize per prompt
    normalized = (score_matrix - score_matrix.mean(dim=1, keepdim=True)) / (score_matrix.std(dim=1, keepdim=True) + 1e-6)
    
    # 2. Resonance score: count how many prompts each neuron is "important"
    resonance_counts = (normalized > 0.0).float().sum(dim=0)  # [n_neurons] torch.Size([14336])
    resonance_cut = torch.quantile(resonance_counts, 1 - resonance_ratio)
    resonance_mask = (resonance_counts >= resonance_cut).float() 
    # print((torch.bincount(resonance_mask.long(), minlength=2)).tolist()) 

    # 3. Burst score: max activation across prompts
    burst_score = score_matrix.max(dim=0).values
    burst_cut = torch.quantile(burst_score, 1 - burst_ratio)
    burst_mask = (burst_score >= burst_cut).float() 
    # print((torch.bincount(burst_mask.long(), minlength=2)).tolist()) 
    final_mask = torch.clamp(resonance_mask + burst_mask, max=1.0)
    # if use_resonance and use_burst:
    #     final_mask = torch.clamp(resonance_mask + burst_mask, max=1.0)
    # elif use_resonance:
    #     final_mask = resonance_mask
    # elif use_burst:
    #     final_mask = burst_mask
    # else:
    #     final_mask = torch.zeros_like(resonance_mask)  # 如果不使用任何掩码，返回全0

    return final_mask, resonance_mask, burst_mask, resonance_counts, burst_score, resonance_cut, burst_cut
def entropy_adaptive_mask_ratio(
    score_matrix: torch.Tensor, 
    resonance_bounds: Tuple[float, float] = (0.3, 0.4),
    burst_bounds: Tuple[float, float] = (0.3, 0.4),
    gamma_r: float = 3.0,
    gamma_b: float = 2.0,
    alpha: float = 30.0,  
) -> Tuple[float, float]:
 
    D = score_matrix.shape[1]
    logD = np.log(D)

    # print("\n******************======= [Entropy Mask Ratio Debug Info] =======*********************")
    # print(f"[Score Matrix] shape = {score_matrix.shape}")
    # print(f"- min = {score_matrix.min().item():.6f}")
    # print(f"- max = {score_matrix.max().item():.6f}")
    # print(f"- mean = {score_matrix.mean().item():.6f}")
    # print(f"- std = {score_matrix.std().item():.6f}")
 
    shifted = score_matrix - score_matrix.max(dim=1, keepdim=True)[0]
    scaled = shifted * alpha  
    softmax_scores = torch.softmax(scaled, dim=1)

    entropy_prompt = -(softmax_scores * (softmax_scores + 1e-8).log()).sum(dim=1)
    entropy_r = entropy_prompt.mean() / logD

    # print(f"[Resonance] prompt-wise entropy:")
    # print(f"- min = {entropy_prompt.min().item():.6f}")
    # print(f"- max = {entropy_prompt.max().item():.6f}")
    # print(f"- mean = {entropy_prompt.mean().item():.6f}")
    # print(f"- normalized = {entropy_r:.6f}")

    resonance_ratio = resonance_bounds[0] + (resonance_bounds[1] - resonance_bounds[0]) * entropy_r.clamp(0, 1).pow(gamma_r)
    # print(f"[Resonance] ratio = {resonance_ratio:.4f} ∈ [{resonance_bounds[0]}, {resonance_bounds[1]}]")
 
    max_acts = score_matrix.max(dim=0).values
    max_acts = torch.clamp(max_acts, min=0.0)  # 🔧 防止负数 → NaN

    if max_acts.sum() < 1e-8:
        print("⚠️ [Burst] All max_acts ≈ 0 → fallback ratio")
        return float(resonance_ratio), burst_bounds[0]

    burst_probs = max_acts / (max_acts.sum() + 1e-8)
    entropy_burst = -(burst_probs * (burst_probs + 1e-8).log()).sum()
    entropy_b = entropy_burst / logD

    # print(f"[Burst] max_acts stats:")
    # print(f"- min = {max_acts.min().item():.6f}")
    # print(f"- max = {max_acts.max().item():.6f}")
    # print(f"- mean = {max_acts.mean().item():.6f}")
    # print(f"- std = {max_acts.std().item():.6f}")
    # print(f"- entropy = {entropy_burst:.6f}, normalized = {entropy_b:.6f}")

    burst_ratio = burst_bounds[0] + (burst_bounds[1] - burst_bounds[0]) * entropy_b.clamp(0, 1).pow(gamma_b)
    # print(f"[Burst] ratio = {burst_ratio:.4f} ∈ [{burst_bounds[0]}, {burst_bounds[1]}]")
 
    # if torch.isnan(entropy_r): print("❌ NaN in entropy_r")
    # if torch.isnan(entropy_b): print("❌ NaN in entropy_b")
    # if torch.isnan(burst_ratio): print("❌ NaN in burst_ratio")
    # if torch.isnan(resonance_ratio): print("❌ NaN in resonance_ratio")

    return float(resonance_ratio), float(burst_ratio)
 
 