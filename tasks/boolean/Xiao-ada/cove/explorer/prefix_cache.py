"""Reusable exact prompt prefix, excluding the last token, for MLX generation."""
import copy


def build_prefix(model, prompt_ids):
    import mlx.core as mx
    from mlx_lm.generate import generation_stream
    from mlx_lm.models.cache import make_prompt_cache
    cache=make_prompt_cache(model)
    # Match generate_step's default 2048-token prefill chunking exactly.
    with mx.stream(generation_stream):
        prefix=prompt_ids[:-1]
        for start in range(0,len(prefix),2048):
            model(mx.array(prefix[start:start+2048])[None],cache=cache)
            mx.eval([c.state for c in cache])
            mx.clear_cache()
    return cache


def fresh_prefix(cache):
    return copy.deepcopy(cache)
