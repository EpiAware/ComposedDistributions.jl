# [The leaf protocol](@id leaf-protocol)

This page is the reference for the leaf protocol, the stable public contract a downstream leaf-wrapper package implements so its wrapped delays are transparent to composition.
A leaf is a univariate distribution at the tip of a composed tree.
Any `Distributions.jl` distribution is a valid leaf with no extra work, so a plain leaf needs none of these methods.
A wrapper leaf, a type that carries fixed structure or extra parameters around an inner delay (censoring in CensoredDistributions, the modifiers in ModifiedDistributions), implements the methods below so the introspection and reconstruction layers see through the wrapper to the inner free delay.

The names are `public` but not exported, so a downstream package reaches them by the qualified name (`ComposedDistributions.free_leaf` and friends) and adds methods dispatching on its own wrapper type.
Extending only `free_leaf` and `rewrap_leaf` is enough for a fixed-structure wrapper; a wrapper that attaches priors or owns extra parameters extends the rest so those reach `params_table` and `build_priors`.

## The methods

The protocol splits into peel and rebuild, names, reconstruction, uncertainty, the shared tag, moments, and extra parameters.

| Method | Role |
|---|---|
| `free_leaf(leaf)` | peel to the innermost free delay |
| `rewrap_leaf(leaf, inner)` | rebuild the wrapper around a new inner delay |
| `component_names(node)` | a node's child names |
| `param_names(leaf)` | the inner delay's native parameter names |
| `leaf_param_names(leaf)` | the estimable names, native then extra |
| `leaf_ctor(leaf)` | the constructor that rebuilds the inner delay |
| `uncertain_specs(leaf)` | attached priors, or `nothing` when fixed |
| `shared_tag(leaf)` | the shared tie tag, or `nothing` |
| `leaf_mean(leaf)`, `leaf_var(leaf)` | per-leaf moments |
| `extra_leaf_params(leaf)` | modifier-owned free parameters and supports |
| `set_extra_leaf_params(leaf, vals)` | rebuild with new extra values |
| `leaf_detail_lines(leaf)` | `inspect` rendering lines |

The peel and rebuild pair is the base of the protocol.
`free_leaf` reaches the inner free delay whose parameters are the leaf's free parameters, and `rewrap_leaf` re-applies the fixed structure around a rebuilt delay.
A plain leaf is its own free leaf and `rewrap_leaf` returns the new inner delay, so the identity holds without a method.

Names and reconstruction fix the coordinates the parameter table and the codec work in.
`param_names` labels the native family parameters, `leaf_param_names` appends any extra names, and `leaf_ctor` rebuilds the inner delay from a positional tuple of native values.

## Extra parameters

Most wrappers carry only fixed structure, so their extra-parameter map is empty and `params_table` shows just the inner delay's rows.
A wrapper that owns a free parameter which is not one of the inner delay's native parameters reports it through `extra_leaf_params`, a `NamedTuple` mapping each extra name to a `(value, support)` pair.
The thinning factor of `thin(d, p)` is the first instance, reported as a `:thin` entry with support `(0.0, 1.0)`.
The support drives the default prior, so a `:thin` factor picks up a `Uniform(0, 1)` default the same way a `branch_probs` row does.
`set_extra_leaf_params` is the dual that rebuilds the leaf from new extra values by name.

A worked example with a plain leaf, where every peel is the identity and the extra map is empty.

```@example leaf-protocol
using ComposedDistributions, Distributions

leaf = Gamma(2.0, 1.0)
(free = ComposedDistributions.free_leaf(leaf),
    names = ComposedDistributions.leaf_param_names(leaf),
    specs = ComposedDistributions.uncertain_specs(leaf),
    tag = ComposedDistributions.shared_tag(leaf),
    mean = ComposedDistributions.leaf_mean(leaf),
    extras = ComposedDistributions.extra_leaf_params(leaf))
```

A censored or truncated leaf peels its fixed structure to the inner delay, so the parameter table lists only the inner free parameters.

```@example leaf-protocol
inner = ComposedDistributions.free_leaf(truncated(Gamma(2.0, 1.0); upper = 10.0))
rebuilt = ComposedDistributions.rewrap_leaf(
    truncated(Gamma(2.0, 1.0); upper = 10.0), Gamma(3.0, 1.5))
(inner = inner, rebuilt = rebuilt)
```

## Adding a wrapper leaf

1. Implement `free_leaf` and `rewrap_leaf` so the wrapper peels to and rebuilds around the inner free delay.
2. Add `shared_tag` and `uncertain_specs` methods that forward to the inner delay, so a tie or an attached prior under the wrapper still reaches the parameter table.
3. Override `leaf_mean` and `leaf_var` only when the wrapper's transform changes the moment, as an affine scale and shift does.
4. Implement `extra_leaf_params` and `set_extra_leaf_params` only when the wrapper owns a free parameter beyond the inner delay's native ones.
5. Add a `leaf_detail_lines` method when the wrapper's raw struct dump would clutter an `inspect` tree.
6. Register the wrapper with `register_leaf_wrapper!` (see below) so the generated flat-vector codec (`flat_dimension`/`unflatten`/`flatten`/`reconstruct`) sees through it too — steps 1-5 alone cover `params_table`/`build_priors`, but not the generated codec.

The docstrings for each method, with runnable examples, are in the public API reference.

## Registering with the generated codec (#189)

The methods above are all INSTANCE-level: they dispatch on a leaf value.
The generated flat-vector codec (`flat_dimension`, `unflatten`, `flatten`, `reconstruct`) works from the tree's TYPE alone, before any instance exists, so it needs a TYPE-level answer to the same two questions `free_leaf`/`extra_leaf_params` answer at the instance level: what does this wrapper peel to, and what extra parameters does it own?

A package-external leaf wrapper (defined in a DIFFERENT package to ComposedDistributions, loaded through a package extension) cannot answer these by adding a direct dispatch method the way a core wrapper (`Truncated`) does: the codec's `@generated` functions call these hooks from their GENERATOR, and calling ANY user-defined function or closure from a generator — dispatched or not, `Base.invokelatest`-wrapped or not — can hit a world-age wall once the generator's own machinery has been compiled against an earlier world (confirmed by direct experiment, not just theory).
`register_leaf_wrapper!` sidesteps this entirely by taking PLAIN DATA, not a callable: a type-parameter index for the one-level peel, and a fixed tuple of extra names.
Reading a type parameter and a struct field involve no dispatch and no world-age concern at all.
Call it from your extension's `__init__` — never at module top level, since `__init__` is what guarantees the registry is populated before any code that could construct one of your leaf types runs:

```julia
function __init__()
    ComposedDistributions.register_leaf_wrapper!(YourWrapper; free_index = 1)
    # `extra_names` defaults to `()` ("no extras, keep peeling"); pass a
    # tuple to claim extras and stop peeling further, e.g. for a case that
    # depends on a further type parameter:
    # ComposedDistributions.register_leaf_wrapper!(
    #     YourWrapper{D, <:YourSpecialOp} where D;
    #     free_index = 1, extra_names = (:your_extra,))
end
```

`free_index` need only name the ONE-LEVEL peeled type parameter's position, even for a leaf nested under several wrappers — the codec's own resolver recurses through as many further layers as there are, core or registered, mixed in any order.
A wrapper family whose answer depends on a further type parameter (`thin(...)` owns a `:thin` extra and does not peel further; every OTHER forward-transform op owns none and does peel through) registers ONE ENTRY PER CASE, most specific pattern registered LAST (a later, more specific registration takes precedence over an earlier, more general one).

See `register_leaf_wrapper!`'s docstring in the public API reference for the full contract and a worked example.
