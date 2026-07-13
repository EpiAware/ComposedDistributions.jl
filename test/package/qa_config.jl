# PACKAGE-OWNED — scaffold writes this once and never overwrites it.
#
# QA configuration values the managed `quality.jl` testset reads. Fill in the
# package-specific inputs the shared helpers need; the standard testset logic
# stays in `quality.jl` (managed). Edit freely.

using ComposedDistributions

const QA_CONFIG = (
    # The module under test.
    mod = ComposedDistributions,

    # Path to the isolated JET environment (see test/jet/Project.toml).
    jet_env = joinpath(@__DIR__, "..", "jet"),

    # Per-check Aqua relaxations, e.g. (; ambiguities = false). Empty = all on.
    aqua = (;),

    # ExplicitImports `ignore`: symbols imported non-publicly.
    # (`logccdf_ad_safe` is a public EpiAwareADTools export the racing-hazard node
    # reuses and extends, so it needs no ignore. `_uncertain_specs` and
    # `_leaf_detail_lines` are `public` (#142), so they need no ignore either.)
    # `_shared_tag`, `_thin_factor` and `_set_thin_factor` are the internal
    # tag/thin-factor protocols, and `_leaf_mean`/`_leaf_var` the internal
    # per-leaf moment hooks, that the ModifiedDistributions extension extends
    # so a modified leaf peels its shared tag and uncertain parameter specs, a
    # `thin` reporting probability round-trips through `params_table`/`update`,
    # and a `thin(...)`/`affine(...)` leaf honours its analytic moment, inside
    # a composed tree; `_leaf_param_names` and `_collect_shared` are the
    # internal leaf-naming and shared-group lookups the FlexiChains extension
    # reuses to read a fitted chain back onto a composed tree;
    # `CentredPoolPrior` and `_population_template` are the internal pooling
    # marker type and population-family lookup the Bijectors extension reuses
    # to read a centred-pooled row's constraint off its population instead of
    # a fixed prior.
    ei_ignore = (:_shared_tag, :_thin_factor,
        :_set_thin_factor, :_leaf_mean, :_leaf_var, :_leaf_param_names,
        :_collect_shared, :CentredPoolPrior, :_population_template),

    # Docstring `crossref_ignore`: upstream names docstrings link to via
    # `[`name`](@ref)`. Distributions functions plus the censoring / PPL surface
    # that stays in CensoredDistributions (referenced from ported prose).
    crossref_ignore = (:pdf, :cdf, :logpdf, :mean, :var, :std, :logcdf, :ccdf,
        :logccdf, :quantile, :latent, :primary_censored, :interval_censored,
        :double_interval_censored, :truncate_to_horizon,
        :composed_distribution_model, :composed_parameters_model,
        :record_distributions),

    # Extra docstring-format options, e.g.
    # (; exported_only_examples = true, require_field_docs = true).
    docstring = (;),

    # README section-structure check. `path` is the package root (its
    # README.md). Override `required`/`order` to extend or relax the standard
    # section set, e.g.
    #   (; required = vcat(STANDARD_README_SECTIONS, [("Benchmarks",)]))
    # Empty `(;)` uses the standard structure in standard order.
    readme = (; path = joinpath(@__DIR__, "..", "..")),

    # Package extensions to ambiguity-check. Each entry:
    #   (; name = :MyPkgSomeTriggerExt,
    #      triggers = ("SomeTrigger",),       # packages to load first
    #      prefixes = ("MyPkg", "SomeTrigger"),
    #      expect_phantoms = false,    # true if a third party adds phantoms
    #      broken = false)             # true to quarantine a known ambiguity
    extensions = ()
)
