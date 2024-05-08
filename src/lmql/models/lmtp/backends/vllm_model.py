from typing import Tuple

import numpy as np

from lmql.models.lmtp.backends.lmtp_model import LMTPModel, TokenStreamer, LMTPModelResult


class VllmModel(LMTPModel):
    def __init__(self, model_identifier, **kwargs) -> None:
        from vllm import LLM

        self.model_identifier = model_identifier
        self.kwargs = kwargs

        self.max_batch_size = 1

        print("[Loading vLLM model from", self.model_identifier, " with ", kwargs, "]", flush=True)

        self.llm = LLM(model=model_identifier[len("vllm:"):], **kwargs)

    def model_info(self):
        import vllm
        return {
            "model_identifier": self.model_identifier[len("vllm:"):],
            "model_type": "vllm",
            "constructor": "LLM(model='{}'{})".format(self.model_identifier[len("vllm:"):], ", " + ", ".join(["{}={}".format(k, v) for k,v in self.kwargs.items()]) if len(self.kwargs) > 0 else ""),
            "vllm": vllm.__version__,
        }

    @property
    def eos_token_id(self):
        return 2

    def score(self, input_ids: np.ndarray, attention_mask: np.ndarray, **model_kwargs) -> Tuple[np.ndarray, np.ndarray]:
        return super().score(input_ids, attention_mask, **model_kwargs)

    def generate(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        temperature: float,
        max_new_tokens: int,
        bias_tensor: np.ndarray,
        streamer: TokenStreamer) -> LMTPModelResult:
        """
        input_ids: The input token ids; "Hello there!" becomes [[50256 15496 612 0]]
        """
        from vllm import SamplingParams
        """Sampling parameters for text generation.

        Overall, we follow the sampling parameters from the OpenAI text completion
        API (https://platform.openai.com/docs/api-reference/completions/create).
        In addition, we support beam search, which is not supported by OpenAI.

        Args:
            n: Number of output sequences to return for the given prompt.
            presence_penalty: Float that penalizes new tokens based on whether they
                appear in the generated text so far. Values > 0 encourage the model
                to use new tokens, while values < 0 encourage the model to repeat
                tokens.
            frequency_penalty: Float that penalizes new tokens based on their
                frequency in the generated text so far. Values > 0 encourage the
                model to use new tokens, while values < 0 encourage the model to
                repeat tokens.
            repetition_penalty: Float that penalizes new tokens based on whether
                they appear in the prompt and the generated text so far. Values > 1
                encourage the model to use new tokens, while values < 1 encourage
                the model to repeat tokens.
            temperature: Float that controls the randomness of the sampling. Lower
                values make the model more deterministic, while higher values make
                the model more random. Zero means greedy sampling.
            top_p: Float that controls the cumulative probability of the top tokens
                to consider. Must be in (0, 1]. Set to 1 to consider all tokens.
            top_k: Integer that controls the number of top tokens to consider. Set
                to -1 to consider all tokens.
            min_p: Float that represents the minimum probability for a token to be
                considered, relative to the probability of the most likely token.
                Must be in [0, 1]. Set to 0 to disable this.
            seed: Random seed to use for the generation.
            use_beam_search: Whether to use beam search instead of sampling.
            length_penalty: Float that penalizes sequences based on their length.
                Used in beam search.
            early_stopping: Controls the stopping condition for beam search. It
                accepts the following values: `True`, where the generation stops as
                soon as there are `best_of` complete candidates; `False`, where an
                heuristic is applied and the generation stops when is it very
                unlikely to find better candidates; `"never"`, where the beam search
                procedure only stops when there cannot be better candidates
                (canonical beam search algorithm).
            stop: List of strings that stop the generation when they are generated.
                The returned output will not contain the stop strings.
            stop_token_ids: List of tokens that stop the generation when they are
                generated. The returned output will contain the stop tokens unless
                the stop tokens are special tokens.
            include_stop_str_in_output: Whether to include the stop strings in output
                text. Defaults to False.
            ignore_eos: Whether to ignore the EOS token and continue generating
                tokens after the EOS token is generated.
            max_tokens: Maximum number of tokens to generate per output sequence.
            logprobs: Number of log probabilities to return per output token.
                Note that the implementation follows the OpenAI API: The return
                result includes the log probabilities on the `logprobs` most likely
                tokens, as well the chosen tokens. The API will always return the
                log probability of the sampled token, so there  may be up to
                `logprobs+1` elements in the response.
            prompt_logprobs: Number of log probabilities to return per prompt token.
            skip_special_tokens: Whether to skip special tokens in the output.
            spaces_between_special_tokens: Whether to add spaces between special
                tokens in the output.  Defaults to True.
            logits_processors: List of functions that modify logits based on
                previously generated tokens.
        """

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_new_tokens,
            logprobs=self.llm.get_tokenizer().vocab_size)

        input_ids = input_ids.reshape(-1).tolist()

        outputs = self.llm.generate(prompt_token_ids=input_ids, sampling_params=sampling_params)

        #for output in outputs:
        #    prompt = output.prompt
        #    generated_text = output.outputs[0].text
        #    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

        return super().generate(input_ids, attention_mask, temperature, max_new_tokens, bias_tensor, streamer)

    def make_bias_tensor(self, logit_biases, vocab_size):
        return super().make_bias_tensor(logit_biases, vocab_size)


LMTPModel.registry["vllm"] = VllmModel
