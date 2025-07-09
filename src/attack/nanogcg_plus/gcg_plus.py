import copy
import gc
import logging
import time

from dataclasses import dataclass
from tqdm import tqdm
from typing import List, Optional, Union

import torch
import transformers
from torch import Tensor
from transformers import set_seed

from src.attack.nanogcg_plus.nano_utils import (
    INIT_CHARS,
    find_executable_batch_size,
    get_nonascii_toks,
    mellowmax,
)

logger = logging.getLogger("nanogcg")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@dataclass
class GCGConfig:
    num_steps: int = 250
    optim_str_init: Union[str, List[str]] = "x x x x x x x x x x x x x x x x x x x x x x x x x x x x x"
    search_width: int = 512
    batch_size: int = None
    topk: int = 256
    n_replace: int = 1
    buffer_size: int = 0
    universal: bool = False
    use_mellowmax: bool = False
    mellowmax_alpha: float = 1.0
    use_cw_loss: bool = True
    early_stop: bool = False
    allow_non_ascii: bool = False
    filter_ids: bool = True
    add_space_before_target: bool = False
    seed: int = None
    verbosity: str = "INFO"


@dataclass
class GCGResult:
    best_loss: float
    best_string: str
    losses: List[float]
    strings: List[str]
    time_to_find_s: float = None
    num_steps: int = None


class AttackBuffer:
    def __init__(self, size: int):
        self.buffer = []  # elements are (loss: float, optim_ids: Tensor)
        self.size = size

    def add(self, loss: float, optim_ids: Tensor) -> None:
        if self.size == 0:
            self.buffer = [(loss, optim_ids)]
            return

        if len(self.buffer) < self.size:
            self.buffer.append((loss, optim_ids))
        else:
            self.buffer[-1] = (loss, optim_ids)

        self.buffer.sort(key=lambda x: x[0])

    def get_best_ids(self) -> Tensor:
        return self.buffer[0][1]

    def get_lowest_loss(self) -> float:
        return self.buffer[0][0]

    def get_highest_loss(self) -> float:
        return self.buffer[-1][0]

    def log_buffer(self, tokenizer):
        message = "buffer:"
        for loss, ids in self.buffer:
            optim_str = tokenizer.batch_decode(ids)[0]
            optim_str = optim_str.replace("\\", "\\\\")
            optim_str = optim_str.replace("\n", "\\n")
            message += f"\nloss: {loss}" + f" | string: {optim_str}"
        logger.info(message)


def sample_ids_from_grad(
    ids: Tensor,
    grad: Tensor,
    search_width: int,
    topk: int = 256,
    n_replace: int = 1,
    not_allowed_ids: Tensor = False,
):
    """Returns `search_width` combinations of token ids based on the token gradient.

    Args:
        ids : Tensor, shape = (n_optim_ids)
            the sequence of token ids that are being optimized
        grad : Tensor, shape = (n_optim_ids, vocab_size)
            the gradient of the GCG loss computed with respect to the one-hot token embeddings
        search_width : int
            the number of candidate sequences to return
        topk : int
            the topk to be used when sampling from the gradient
        n_replace : int
            the number of token positions to update per sequence
        not_allowed_ids : Tensor, shape = (n_ids)
            the token ids that should not be used in optimization

    Returns:
        sampled_ids : Tensor, shape = (search_width, n_optim_ids)
            sampled token ids
    """
    n_optim_tokens = len(ids)
    original_ids = ids.repeat(search_width, 1)

    if not_allowed_ids is not None:
        grad[:, not_allowed_ids.to(grad.device)] = float("inf")

    # gradient descent
    topk_ids = (-grad).topk(topk, dim=1).indices

    sampled_ids_pos = torch.argsort(torch.rand((search_width, n_optim_tokens), device=grad.device))[..., :n_replace]
    sampled_ids_val = torch.gather(
        topk_ids[sampled_ids_pos],
        2,
        torch.randint(0, topk, (search_width, n_replace, 1), device=grad.device),
    ).squeeze(2)

    new_ids = original_ids.scatter_(1, sampled_ids_pos, sampled_ids_val)

    return new_ids


def filter_ids(ids: Tensor, tokenizer: transformers.PreTrainedTokenizer):
    """Filters out sequeneces of token ids that change after retokenization.

    Args:
        ids : Tensor, shape = (search_width, n_optim_ids)
            token ids
        tokenizer : ~transformers.PreTrainedTokenizer
            the model's tokenizer

    Returns:
        filtered_ids : Tensor, shape = (new_search_width, n_optim_ids)
            all token ids that are the same after retokenization
    """
    ids_decoded = tokenizer.batch_decode(ids)
    filtered_ids = []

    for i in range(len(ids_decoded)):
        # Retokenize the decoded token ids
        ids_encoded = tokenizer(ids_decoded[i], return_tensors="pt", add_special_tokens=False).to(ids.device)["input_ids"][0]
        if torch.equal(ids[i], ids_encoded):
            filtered_ids.append(ids[i])

    if not filtered_ids:
        # This occurs in some cases, e.g. using the Llama-3 tokenizer with a bad initialization
        return False
    else:
        return torch.stack(filtered_ids)


class GCG:
    def __init__(
        self,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizer,
        config: GCGConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        self.embedding_layer = model.get_input_embeddings()
        self.not_allowed_ids = None if config.allow_non_ascii else get_nonascii_toks(tokenizer, device=model.device)
        
        if model.dtype in (torch.float32, torch.float64):
            logger.warning(f"Model is in {model.dtype}. Use a lower precision data type, if possible, for much faster optimization.")

        if model.device == torch.device("cpu"):
            logger.warning("Model is on the CPU. Use a hardware accelerator for faster optimization.")

        if not tokenizer.chat_template:
            logger.warning("Tokenizer does not have a chat template. Assuming base model and setting chat template to empty.")
            tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"


    def run(
        self,
        messages: Union[str, List[dict], List[str]],
        target: Union[str, List[str]],
    ) -> GCGResult:
        model = self.model
        tokenizer = self.tokenizer
        config = self.config

        self.prompt_index = 0

        if config.seed is not None:
            set_seed(config.seed)
            torch.use_deterministic_algorithms(True, warn_only=True)

        if config.universal:
            assert isinstance(messages, list) and isinstance(messages[0], list) and isinstance(messages[0][0], dict), "Universal GCG requires a list of conversations, each conversation being a list of dictionaries, like [[{'role': 'system', 'content': '...'}, {'role': 'user', 'content': '...'}], [], ...]"
            assert isinstance(target, list) and isinstance(target[0], str), "Universal GCG requires a list of target strings, like ['target1', 'target2', ...]"
            assert isinstance(target, list) and len(messages) == len(target), f"The number of messages and targets must be the same for universal GCG, but got len(messages) = {len(messages)} and len(target) = {len(target)}."
        else:
            if isinstance(messages, str):
                messages = [[{"role": "user", "content": messages}]] # single conversation
            elif isinstance(messages, dict):
                messages = [[copy.deepcopy(messages)]]
            elif isinstance(messages, list):
                messages = [copy.deepcopy(messages)]

        # Append the GCG string at the end of the prompt if location not specified
        # Assert optim_str is present if universal optimization
        for conversation in messages:
            if not any(["{optim_str}" in d["content"] for d in conversation]):
                raise ValueError("GCG string ({optim_str}) must be present in the messages.")

        targets = target if config.universal else [target]

        before_embeds_list = []
        after_embeds_list = []
        target_embeds_list = []

        before_str_list = []
        after_str_list = []

        target_ids_list = []
        target_ids_local_list = []


        for message, targ in zip(messages, targets):
            template = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            # Remove the BOS token -- this will get added when tokenizing, if necessary
            if tokenizer.bos_token and template.startswith(tokenizer.bos_token):
                template = template.replace(tokenizer.bos_token, "")
            before_str, after_str = template.split("{optim_str}")

            targ = " " + targ if config.add_space_before_target else targ

            # Tokenize everything that doesn't get optimized
            before_ids = tokenizer([before_str], padding=False, return_tensors="pt")["input_ids"].to(model.device, torch.int64)
            after_ids = tokenizer([after_str], add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device, torch.int64)
            target_ids = tokenizer([targ], add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device, torch.int64)
            # Create a local copy for a loop concatenation with another set of local ids
            target_ids_local = tokenizer([targ], add_special_tokens=False, return_tensors="pt")["input_ids"].to(torch.int64)

            # Embed everything that doesn't get optimized
            embedding_layer = self.embedding_layer
            # [1, len(seq), embed_dim]
            before_embeds, after_embeds, target_embeds = [embedding_layer(ids) for ids in (before_ids, after_ids, target_ids)]

            before_embeds_list.append(before_embeds)
            after_embeds_list.append(after_embeds)
            target_embeds_list.append(target_embeds)

            before_str_list.append(before_str)
            after_str_list.append(after_str)

            target_ids_list.append(target_ids)
            target_ids_local_list.append(target_ids_local)

        # Make em all class variables
        self.before_embeds_list = before_embeds_list
        self.after_embeds_list = after_embeds_list
        self.target_embeds_list = target_embeds_list

        self.before_str_list = before_str_list
        self.after_str_list = after_str_list

        self.target_ids_list = target_ids_list

        # Initialize the attack buffer
        buffer = self.init_buffer()
        optim_ids = buffer.get_best_ids() # starting with the trigger ids

        losses = []
        optim_strings = []

        for step_num in tqdm(range(config.num_steps)):
            # Compute the gradients for every possible token at every position - this is linearized loss approximation
            optim_ids_onehot_grad = self.compute_token_gradient(optim_ids)

            with torch.no_grad():
                # Sample candidate token sequences based on the token gradient - [search_width, n_optim_ids]
                sampled_ids = sample_ids_from_grad(
                    optim_ids.squeeze(0),
                    optim_ids_onehot_grad.squeeze(0),
                    config.search_width,
                    config.topk,
                    config.n_replace,
                    not_allowed_ids=self.not_allowed_ids,
                )

                if config.filter_ids:
                    sampled_ids = filter_ids(sampled_ids, tokenizer)
                    if sampled_ids is False:
                        logger.warning(
                            "No token sequences are the same after decoding and re-encoding. "
                            "Consider setting `filter_ids=False` or trying a different `optim_str_init`"
                        )
                        continue

                new_search_width = sampled_ids.shape[0]

                total_loss = torch.zeros(new_search_width, device=model.device, dtype=model.dtype)
                induced_target_all = None

                batch_size = new_search_width if config.batch_size is None else config.batch_size
                sampled_strs = tokenizer.batch_decode(sampled_ids)

                for i in range(self.prompt_index+1):
                    # creating the prompt that will be tokenized together at inference time
                    prompts = [before_str_list[i] + samp_str + after_str_list[i] for samp_str in sampled_strs]

                    # tokenizing the prompt and appending tokenized target
                    prompts_ids = []
                    tokenized_prompt_lens = []
                    for prompt in prompts:
                        prompt_ids = torch.cat([tokenizer(prompt, add_special_tokens=True, padding=False, return_tensors="pt")["input_ids"], target_ids_local_list[i]], dim=1)
                        tokenized_prompt_lens.append(prompt_ids.shape[1])
                        prompts_ids.append(prompt_ids.squeeze())

                    # padding the prompt + target ids for embedding all together
                    prompts_ids = torch.nn.utils.rnn.pad_sequence(prompts_ids, batch_first=True, padding_value=tokenizer.eos_token_id).to(model.device, torch.int64)

                    # some have eos token (2) at the end
                    input_embeds = self.embedding_layer(prompts_ids)

                    # compute loss on all candidate sequences for a single prompt
                    loss, induced_target = find_executable_batch_size(self._compute_candidates_loss_original, batch_size)(input_embeds, self.target_ids_list[i], tokenized_prompt_lens)
                    total_loss = total_loss + loss

                    induced_target_all = induced_target.unsqueeze(0) if induced_target_all is None else torch.cat((induced_target_all, induced_target.unsqueeze(0)), dim=0)

            # select the best candidate sequence
            av_loss = total_loss / (self.prompt_index + 1)
            current_loss = av_loss.min().item()
            optim_ids = sampled_ids[av_loss.argmin()].unsqueeze(0)

            # Update the buffer based on the loss
            losses.append(current_loss)
            if buffer.size == 0 or current_loss < buffer.get_highest_loss():
                buffer.add(current_loss, optim_ids)

            optim_ids = buffer.get_best_ids()
            optim_str = tokenizer.batch_decode(optim_ids)[0]
            optim_strings.append(optim_str)

            buffer.log_buffer(tokenizer)

            if config.early_stop and not config.universal:
                if torch.any(induced_target_all.squeeze(0)).item():
                    success_idx = torch.where(induced_target_all.squeeze(0))[0][0]
                    success_loss = av_loss[success_idx].item()
                    success_ids = sampled_ids[success_idx]
                    success_str = tokenizer.decode(success_ids)

                    losses.append(success_loss)
                    optim_strings.append(success_str)

                    logger.info("Early stopping triggered.")
                    result = GCGResult(
                        best_loss=success_loss,
                        best_string=success_str,
                        losses=losses,
                        strings=optim_strings,
                        num_steps=step_num + 1,
                    )
                    return result

                
            if config.universal:
                induced_all_targets = torch.all(induced_target_all, dim=0)
                if torch.any(induced_all_targets).item():
                    print("\n\n######## TRIGGER WORKED, ADDING ANOTHER PROMPT ############\n\n", flush=True)
                    success_idx = torch.where(induced_all_targets)[0][0]
                    self.prompt_index += 1
                if self.prompt_index == len(messages): # this means we have succeed at attacking all prompts
                    logger.info("Early stopping triggered.")
                    success_loss = av_loss[success_idx].item()
                    success_ids = sampled_ids[success_idx]
                    success_str = tokenizer.decode(success_ids)

                    losses.append(success_loss)
                    optim_strings.append(success_str)

                    result = GCGResult(
                        best_loss=success_loss,
                        best_string=success_str,
                        losses=losses,
                        strings=optim_strings,
                        num_steps=step_num + 1,
                    )
                    return result

            del optim_ids_onehot_grad
            del prompts
            del prompts_ids
            del input_embeds
            torch.cuda.empty_cache()
            gc.collect()


        min_loss_index = losses.index(min(losses))

        result = GCGResult(
            best_loss=losses[min_loss_index],
            best_string=optim_strings[min_loss_index],
            losses=losses,
            strings=optim_strings,
            num_steps=config.num_steps,
        )

        return result


    def init_buffer(self) -> AttackBuffer:
        model = self.model
        tokenizer = self.tokenizer
        config = self.config

        logger.info(f"Initializing attack buffer of size {config.buffer_size}...")

        # Create the attack buffer and initialize the buffer ids
        buffer = AttackBuffer(config.buffer_size)

        if isinstance(config.optim_str_init, str):
            init_optim_ids = tokenizer(config.optim_str_init, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
            if config.buffer_size > 1:
                init_buffer_ids = tokenizer(INIT_CHARS, add_special_tokens=False, return_tensors="pt")["input_ids"].squeeze().to(model.device)
                init_indices = torch.randint(0, init_buffer_ids.shape[0], (config.buffer_size - 1, init_optim_ids.shape[1]))
                init_buffer_ids = torch.cat([init_optim_ids, init_buffer_ids[init_indices]], dim=0)
            else:
                init_buffer_ids = init_optim_ids

        else:  # assume list
            if len(config.optim_str_init) != config.buffer_size:
                logger.warning(f"Using {len(config.optim_str_init)} initializations but buffer size is set to {config.buffer_size}")
            try:
                init_buffer_ids = tokenizer(config.optim_str_init, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
            except ValueError:
                logger.error("Unable to create buffer. Ensure that all initializations tokenize to the same length.")

        true_buffer_size = max(1, config.buffer_size)

        init_buffer_prompt = self.before_str_list[0] + ' '.join(tokenizer.batch_decode(init_buffer_ids.squeeze())) + self.after_str_list[0]
        init_buffer_prompt_ids = self.tokenizer(init_buffer_prompt, add_special_tokens=True, return_tensors="pt")["input_ids"].to(model.device)
        init_buffer_embeds = self.embedding_layer(init_buffer_prompt_ids)

        # gets loss on the initial prompt+target. Target is added just as an efficient way to get autoregressive logits
        init_buffer_losses, _ = find_executable_batch_size(self._compute_candidates_loss_original, true_buffer_size)(init_buffer_embeds, self.target_ids_list[0])

        # Populate the buffer
        for i in range(true_buffer_size):
            buffer.add(init_buffer_losses[i], init_buffer_ids[[i]])

        buffer.log_buffer(tokenizer)

        logger.info("Initialized attack buffer.")

        return buffer


    def compute_token_gradient(
        self,
        optim_ids: Tensor,
    ) -> Tensor:
        """Computes the gradient of the GCG loss w.r.t the one-hot token matrix.

        Args:
            optim_ids : Tensor, shape = (1, n_optim_ids)
                the sequence of token ids that are being optimized
        """

        model = self.model
        embedding_layer = self.embedding_layer

        for i in range(self.prompt_index+1):

            # Create the one-hot encoding matrix of our optimized token ids
            optim_ids_onehot = torch.nn.functional.one_hot(optim_ids, num_classes=embedding_layer.num_embeddings)
            optim_ids_onehot = optim_ids_onehot.to(model.device, model.dtype)
            optim_ids_onehot.requires_grad_()

            # (1, num_optim_tokens, vocab_size) @ (vocab_size, embed_dim) -> (1, num_optim_tokens, embed_dim)
            optim_embeds = optim_ids_onehot @ embedding_layer.weight

            # total_loss = torch.zeros(optim_ids.shape[0], device=model.device, dtype=model.dtype)
            total_optim_ids_onehot_grad = torch.zeros((optim_ids.shape[0], optim_ids.shape[1], embedding_layer.num_embeddings), device=model.device, dtype=model.dtype)

            input_embeds = torch.cat(
                [
                    self.before_embeds_list[i],
                    optim_embeds,
                    self.after_embeds_list[i],
                    self.target_embeds_list[i],
                ],
                dim=1,
            ) # [batch_size, seq_len, embed_dim]

            output = model(inputs_embeds=input_embeds)

            logits = output.logits

            # Shift logits so token n-1 predicts token n
            shift = input_embeds.shape[1] - self.target_ids_list[i].shape[1]
            shift_logits = logits[..., shift - 1 : -1, :].contiguous()  # (1, num_target_ids, vocab_size)
            shift_labels = self.target_ids_list[i]

            if self.config.use_mellowmax:
                label_logits = torch.gather(shift_logits, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
                loss = mellowmax(-label_logits, alpha=self.config.mellowmax_alpha, dim=-1)
            elif self.config.use_cw_loss:
                loss = self._cw_loss(shift_logits, shift_labels)
            else:
                loss = torch.nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            # should free computational graph
            total_optim_ids_onehot_grad += torch.autograd.grad(outputs=[loss], inputs=[optim_ids_onehot])[0]

            # total_loss = total_loss + loss

            del output
            del loss
            del optim_embeds
            del optim_ids_onehot
            del input_embeds
            gc.collect()
            torch.cuda.empty_cache()

        # av_loss = total_loss / (self.prompt_index+1)

        # optim_ids_onehot_grad = torch.autograd.grad(outputs=[av_loss], inputs=[optim_ids_onehot])[0]

        optim_ids_onehot_grad = total_optim_ids_onehot_grad / (self.prompt_index+1)

        return optim_ids_onehot_grad


    def _compute_candidates_loss_original(
        self,
        search_batch_size: int,
        input_embeds: Tensor,
        target_ids: Tensor,
        pre_padded_prompt_lens: List[int] = None,
    ) -> Tensor:
        """Computes the GCG loss on all candidate token id sequences.

        Args:
            search_batch_size : int
                the number of candidate sequences to evaluate in a given batch
            input_embeds : Tensor, shape = (search_width, seq_len, embd_dim)
                the embeddings of the `search_width` candidate sequences to evaluate
            target_ids : Tensor, shape = (1, seq_len)
                the token ids of the target sequence
            pre_padded_prompt_lens : List[int], optional
                the lengths of the tokenized prompts (with target) for each candidate sequence before padding was applied
        """
        all_loss = []
        all_induced_target = []

        if pre_padded_prompt_lens is None:
            pre_padded_prompt_lens = [input_embeds.shape[1]] * input_embeds.shape[0]

        for i in range(0, input_embeds.shape[0], search_batch_size):
            with torch.no_grad():
                input_embeds_batch = input_embeds[i:i + search_batch_size]
                pre_padded_lens_batch = pre_padded_prompt_lens[i:i + search_batch_size]

                current_batch_size = input_embeds_batch.shape[0]

                outputs = self.model(inputs_embeds=input_embeds_batch)
                
                logits = outputs.logits

                starts = [pre - target_ids.shape[1] - 1 for pre in pre_padded_lens_batch] # original prompt lengths -1
                stops = [start + target_ids.shape[1] for start in starts]
                shift_logits_list = [logits[i:i+1, start:stop, :].contiguous() for i, (start, stop) in enumerate(zip(starts, stops))] # (search_batch_size, num_target_ids, vocab_size)
                shift_logits = torch.cat(shift_logits_list, dim=0)

                shift_labels = target_ids.repeat(current_batch_size, 1)

                if self.config.use_mellowmax:
                    label_logits = torch.gather(shift_logits, -1, shift_labels.unsqueeze(-1)).squeeze(-1)
                    loss = mellowmax(-label_logits, alpha=self.config.mellowmax_alpha, dim=-1)
                    loss = loss.view(current_batch_size, -1).mean(dim=-1)
                elif self.config.use_cw_loss:
                    loss = self._cw_loss(shift_logits, shift_labels)
                else:
                    loss = torch.nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="none")
                    loss = loss.view(current_batch_size, -1).mean(dim=-1)
                
                all_loss.append(loss)

                induced_target = torch.all(torch.argmax(shift_logits, dim=-1) == shift_labels, dim=-1)
                all_induced_target.append(induced_target)

                del outputs
                gc.collect()
                torch.cuda.empty_cache()

        return torch.cat(all_loss, dim=0), torch.cat(all_induced_target, dim=0)


    def _cw_loss(
        self,
        logits: torch.FloatTensor,
        target_ids: torch.LongTensor,
        cw_margin: float = 1e-3,
        dim: int = -1,
    ) -> torch.FloatTensor:
        """CW loss.
        Hinge loss on the difference between the largest and the target logits.
        """
        input_shape = target_ids.shape
        assert logits.shape[:-1] == input_shape, (logits.shape, input_shape)

        target_ids = target_ids.unsqueeze(-1)
        tgt_logits = logits.gather(dim, target_ids).squeeze(-1)

        # Set logits of target tok very low (-1e3) so it cannot be the largest
        tmp_logits = logits.clone()
        tmp_logits.scatter_(dim, target_ids, -1e3)

        largest_non_tgt_logits = tmp_logits.max(dim).values
        loss = largest_non_tgt_logits - tgt_logits
        loss = loss.clamp_min(-cw_margin).mean(-1)

        if len(input_shape) == 1:
            assert loss.ndim == 0, loss.shape
        else:
            assert loss.shape == input_shape[:1], loss.shape

        return loss


# A wrapper around the GCG `run` method that provides a simple API
def run(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    messages: Union[str, List[dict], List[str]],
    target: Union[str, List[str]],
    config: Optional[GCGConfig] = None,
) -> GCGResult:
    """Generates a single optimized string using GCG.

    Args:
        model: The model to use for optimization.
        tokenizer: The model's tokenizer.
        messages: The conversation to use for optimization. If config.universal is True, this should be a list of messages.
        target: The target generation. If config.universal is True, this should be a list of targets.
        config: The GCG configuration to use.

    Returns:
        A GCGResult object that contains losses and the optimized strings.
    """
    if config is None:
        config = GCGConfig()

    logger.setLevel(getattr(logging, config.verbosity))

    start_optim = time.time()
    gcg = GCG(model, tokenizer, config)
    result = gcg.run(messages, target)
    end_optim = time.time()
    result.time_to_find_s = end_optim - start_optim
    return result