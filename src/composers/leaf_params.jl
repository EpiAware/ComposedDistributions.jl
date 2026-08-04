# The single leaf-parameter table (#332): one source of truth for a native
# Distributions.jl family's scalar parameter names, generating both the
# instance-level (`param_names`) and type-level (`_param_names_of`,
# `_params_arity_of`) dispatch tables so they cannot drift apart (the two were
# previously hand-maintained in step, in introspection.jl and codec_gen.jl
# respectively, with nothing enforcing agreement).
#
# Included before both introspection.jl and codec_gen.jl: the `@eval` loop at
# the bottom of this file adds methods to `param_names`/`_param_names_of`/
# `_params_arity_of` before either file defines those functions' `::Any`/
# `::Type` fallback methods. Julia does not require a generic function to
# exist before a method is added to it -- the fallback methods defined later
# simply extend the same generic functions -- so definition order across the
# two files does not matter, only that every method is in place before the
# module finishes loading (well before any tree is built).

# One family's native scalar-parameter names and each parameter's own domain
# (the value the parameter itself may range over, not the variate's support --
# a Beta's shape parameters live on `(0, Inf)` even though the variate lives on
# `(0, 1)`). `supports` has one `(lo, hi)` pair per name, same order.
struct LeafParamSpec
    names::Tuple{Vararg{Symbol}}
    supports::Tuple
end

# Keyed on the UnionAll wrapper type (`Distributions.Gamma`, not a concrete
# `Distributions.Gamma{Float64}`), matched with `<:` wherever it is consulted.
#
# The first six entries are the pre-existing hand-written table
# (introspection.jl/codec_gen.jl before #332), values unchanged so no row in
# any existing `params_table` output renames.
#
# The remaining entries cover the rest of Distributions.jl's univariate
# library, enumerated by walking `names(Distributions)` for every concrete
# `UnivariateDistribution` subtype and constructing a default instance to read
# `length(params(d))`. A family is omitted when `params` is not scalar
# (`Categorical`, `DiscreteNonParametric`, `PoissonBinomial`,
# `UnivariateGMM` -- see the non-scalar guard in `_walk_rows!`), when
# `Distributions.params` has no method at all for it (`Dirac`, `Chernoff`,
# `Kolmogorov`'s siblings `KSDist`/`KSOneSided`, the `Edgeworth*` family), or
# when it is a wrapper over an arbitrary inner distribution rather than a flat
# family (`Truncated`, already peeled by `free_leaf` before dispatch ever
# reaches here; `OrderStatistic`, whose `params` flattens an arbitrary inner
# distribution's own parameters). `Kolmogorov` itself has zero parameters and
# is included with an empty spec.
#
# Names are spelled out in ASCII, after the family's own documented
# parameterisation (a family's own accessor names -- `shape`/`scale`,
# `location`, `df` -- where Distributions.jl's docstring defines one, a direct
# transliteration of the docstring's own symbol otherwise). This is a
# judgement call flagged for Sam in PR1: Distributions.jl exposes no parameter
# names at all (`params` is a bare tuple), and `fieldnames` is actively wrong
# for several families (`InverseGamma`'s shape field is `:invd`; see the
# fieldcount note below), so ASCII names describing each parameter's role are
# the only available option, but the exact word chosen per family is this
# table's judgement, not Distributions.jl's.
const LEAF_PARAM_SPECS = Pair{Type, LeafParamSpec}[
    # --- pre-existing six (unchanged) ---------------------------------------
    Distributions.Normal => LeafParamSpec((:mu, :sigma),
        ((-Inf, Inf), (0.0, Inf))),
    Distributions.LogNormal => LeafParamSpec((:mu, :sigma),
        ((-Inf, Inf), (0.0, Inf))),
    Distributions.Gamma => LeafParamSpec((:shape, :scale),
        ((0.0, Inf), (0.0, Inf))),
    Distributions.Weibull => LeafParamSpec((:shape, :scale),
        ((0.0, Inf), (0.0, Inf))),
    Distributions.Exponential => LeafParamSpec((:scale,), ((0.0, Inf),)),
    Distributions.Uniform => LeafParamSpec((:lower, :upper),
        ((-Inf, Inf), (-Inf, Inf))),

    # --- the rest of the univariate library ---------------------------------
    Distributions.Arcsine => LeafParamSpec((:lower, :upper),
        ((-Inf, Inf), (-Inf, Inf))),
    Distributions.Bernoulli => LeafParamSpec((:p,), ((0.0, 1.0),)),
    Distributions.BernoulliLogit => LeafParamSpec((:logitp,), ((-Inf, Inf),)),
    Distributions.Beta => LeafParamSpec((:alpha, :beta),
        ((0.0, Inf), (0.0, Inf))),
    Distributions.BetaBinomial => LeafParamSpec((:n, :alpha, :beta),
        ((0.0, Inf), (0.0, Inf), (0.0, Inf))),
    Distributions.BetaPrime => LeafParamSpec((:alpha, :beta),
        ((0.0, Inf), (0.0, Inf))),
    Distributions.Binomial => LeafParamSpec((:n, :p), ((0.0, Inf), (0.0, 1.0))),
    Distributions.Biweight => LeafParamSpec((:mu, :sigma),
        ((-Inf, Inf), (0.0, Inf))),
    Distributions.Cauchy => LeafParamSpec((:mu, :sigma),
        ((-Inf, Inf), (0.0, Inf))),
    Distributions.Chi => LeafParamSpec((:df,), ((0.0, Inf),)),
    Distributions.Chisq => LeafParamSpec((:df,), ((0.0, Inf),)),
    Distributions.Cosine => LeafParamSpec((:mu, :sigma),
        ((-Inf, Inf), (0.0, Inf))),
    Distributions.DiscreteUniform => LeafParamSpec((:lower, :upper),
        ((-Inf, Inf), (-Inf, Inf))),
    Distributions.Epanechnikov => LeafParamSpec((:mu, :sigma),
        ((-Inf, Inf), (0.0, Inf))),
    Distributions.Erlang => LeafParamSpec((:shape, :scale),
        ((0.0, Inf), (0.0, Inf))),
    Distributions.FDist => LeafParamSpec((:df1, :df2),
        ((0.0, Inf), (0.0, Inf))),
    Distributions.FisherNoncentralHypergeometric => LeafParamSpec(
        (:ns, :nf, :n, :omega),
        ((0.0, Inf), (0.0, Inf), (0.0, Inf), (0.0, Inf))),
    Distributions.Frechet => LeafParamSpec((:shape, :scale),
        ((0.0, Inf), (0.0, Inf))),
    Distributions.GeneralizedExtremeValue => LeafParamSpec(
        (:location, :scale, :shape), ((-Inf, Inf), (0.0, Inf), (-Inf, Inf))),
    Distributions.GeneralizedPareto => LeafParamSpec(
        (:location, :scale, :shape), ((-Inf, Inf), (0.0, Inf), (-Inf, Inf))),
    Distributions.Geometric => LeafParamSpec((:p,), ((0.0, 1.0),)),
    Distributions.Gumbel => LeafParamSpec((:mu, :sigma),
        ((-Inf, Inf), (0.0, Inf))),
    Distributions.Hypergeometric => LeafParamSpec((:s, :f, :n),
        ((0.0, Inf), (0.0, Inf), (0.0, Inf))),
    Distributions.InverseGamma => LeafParamSpec((:shape, :scale),
        ((0.0, Inf), (0.0, Inf))),
    Distributions.InverseGaussian => LeafParamSpec((:mu, :lambda),
        ((0.0, Inf), (0.0, Inf))),
    Distributions.JohnsonSU => LeafParamSpec((:xi, :lambda, :gamma, :delta),
        ((-Inf, Inf), (0.0, Inf), (-Inf, Inf), (0.0, Inf))),
    Distributions.Kolmogorov => LeafParamSpec((), ()),
    Distributions.Kumaraswamy => LeafParamSpec((:shape1, :shape2),
        ((0.0, Inf), (0.0, Inf))),
    Distributions.Laplace => LeafParamSpec((:mu, :sigma),
        ((-Inf, Inf), (0.0, Inf))),
    Distributions.Levy => LeafParamSpec((:mu, :sigma),
        ((-Inf, Inf), (0.0, Inf))),
    Distributions.Lindley => LeafParamSpec((:theta,), ((0.0, Inf),)),
    Distributions.LogLogistic => LeafParamSpec((:scale, :shape),
        ((0.0, Inf), (0.0, Inf))),
    Distributions.LogUniform => LeafParamSpec((:lower, :upper),
        ((0.0, Inf), (0.0, Inf))),
    Distributions.Logistic => LeafParamSpec((:mu, :sigma),
        ((-Inf, Inf), (0.0, Inf))),
    Distributions.LogitNormal => LeafParamSpec((:mu, :sigma),
        ((-Inf, Inf), (0.0, Inf))),
    Distributions.NegativeBinomial => LeafParamSpec((:r, :p),
        ((0.0, Inf), (0.0, 1.0))),
    Distributions.NoncentralBeta => LeafParamSpec((:alpha, :beta, :lambda),
        ((0.0, Inf), (0.0, Inf), (0.0, Inf))),
    Distributions.NoncentralChisq => LeafParamSpec((:df, :lambda),
        ((0.0, Inf), (0.0, Inf))),
    Distributions.NoncentralF => LeafParamSpec((:df1, :df2, :lambda),
        ((0.0, Inf), (0.0, Inf), (0.0, Inf))),
    Distributions.NoncentralT => LeafParamSpec((:df, :lambda),
        ((0.0, Inf), (-Inf, Inf))),
    # Cached-field exception: `fieldcount` is 3 (`η, λ, μ`, the last a derived
    # mean cached from the other two) against `length(params(d)) == 2`. The
    # generated `_params_arity_of` method below reads its arity off this
    # spec's `names`, not `fieldcount`, so the mismatch never reaches the
    # codec.
    Distributions.NormalCanon => LeafParamSpec((:eta, :lambda),
        ((-Inf, Inf), (0.0, Inf))),
    Distributions.NormalInverseGaussian => LeafParamSpec(
        (:mu, :alpha, :beta, :delta),
        ((-Inf, Inf), (0.0, Inf), (-Inf, Inf), (0.0, Inf))),
    Distributions.Pareto => LeafParamSpec((:shape, :scale),
        ((0.0, Inf), (0.0, Inf))),
    Distributions.PGeneralizedGaussian => LeafParamSpec(
        (:location, :scale, :shape), ((-Inf, Inf), (0.0, Inf), (0.0, Inf))),
    Distributions.Poisson => LeafParamSpec((:lambda,), ((0.0, Inf),)),
    Distributions.Rayleigh => LeafParamSpec((:scale,), ((0.0, Inf),)),
    Distributions.Rician => LeafParamSpec((:nu, :sigma),
        ((0.0, Inf), (0.0, Inf))),
    Distributions.Semicircle => LeafParamSpec((:radius,), ((0.0, Inf),)),
    Distributions.Skellam => LeafParamSpec((:mu1, :mu2),
        ((0.0, Inf), (0.0, Inf))),
    Distributions.SkewNormal => LeafParamSpec((:location, :scale, :shape),
        ((-Inf, Inf), (0.0, Inf), (-Inf, Inf))),
    Distributions.SkewedExponentialPower => LeafParamSpec(
        (:location, :scale, :shape, :skew),
        ((-Inf, Inf), (0.0, Inf), (0.0, Inf), (0.0, 1.0))),
    Distributions.Soliton => LeafParamSpec((:k, :m, :delta, :atol),
        ((0.0, Inf), (0.0, Inf), (0.0, 1.0), (0.0, 1.0))),
    Distributions.StudentizedRange => LeafParamSpec((:df, :k),
        ((0.0, Inf), (0.0, Inf))),
    Distributions.SymTriangularDist => LeafParamSpec((:mu, :sigma),
        ((-Inf, Inf), (0.0, Inf))),
    Distributions.TDist => LeafParamSpec((:df,), ((0.0, Inf),)),
    Distributions.TriangularDist => LeafParamSpec((:lower, :upper, :mode),
        ((-Inf, Inf), (-Inf, Inf), (-Inf, Inf))),
    Distributions.Triweight => LeafParamSpec((:mu, :sigma),
        ((-Inf, Inf), (0.0, Inf))),
    # Cached-field exception: `fieldcount` is 3 (`μ, κ, I0κx`, the last a
    # cached Bessel-function normaliser) against `length(params(d)) == 2`.
    # Same fix as `NormalCanon` above.
    Distributions.VonMises => LeafParamSpec((:mu, :kappa),
        ((-Inf, Inf), (0.0, Inf))),
    Distributions.WalleniusNoncentralHypergeometric => LeafParamSpec(
        (:ns, :nf, :n, :omega),
        ((0.0, Inf), (0.0, Inf), (0.0, Inf), (0.0, Inf)))
]

# Generate both dispatch tables (instance-level `param_names`, type-level
# `_param_names_of`/`_params_arity_of`) from the one table above, so they
# cannot drift apart the way the two hand-maintained versions did (#332).
# `param_supports` is generated too, ready for the walk to consume once it
# reads parameter (not variate) domains.
#
# Two more families than the design review's own scan flagged
# (`DiscreteUniform`, whose `pv` normalising field is cached, and
# `NormalInverseGaussian`, whose `γ` field is a cached
# `sqrt(α² - β²)`) turned out to have the same cached-derived-field arity
# mismatch as `VonMises`/`NormalCanon` above. No special-casing was needed for
# them: every entry in `LEAF_PARAM_SPECS` gets an explicit `_params_arity_of`
# method here, so `fieldcount` is never consulted for a registered family --
# only for a leaf type with no entry in this table at all.
for (T, spec) in LEAF_PARAM_SPECS
    @eval param_names(::$T) = $(spec.names)
    @eval _param_names_of(::Type{<:$T}) = $(spec.names)
    @eval _params_arity_of(::Type{<:$T}) = $(length(spec.names))
    @eval param_supports(::$T) = $(spec.supports)
end
