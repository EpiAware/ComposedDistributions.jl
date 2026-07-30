# `elapsed_between`: the distribution-level elapsed distance between two named
# events of a chain, the counterpart to the sample-level difference of two event
# positions (#274). Components are continuous throughout; discrete-component
# behaviour follows whatever Convolved#85 rules for `convolved`'s value support.

@testitem "elapsed_between: elapsed-distance law between two named events (#274)" begin
    using ComposedDistributions: update
    using Distributions

    # Events in order: origin, onset, admit, exit.
    chain = sequential(:origin_onset => Gamma(2.0, 1.0),
        :onset_admit => LogNormal(0.5, 0.4),
        :admit_exit => Gamma(1.5, 1.0))
    @test event_names(chain) == (:origin, :onset, :admit, :exit)

    # Between two named events: the convolution of the intervening steps — here
    # a single step, so the leaf itself.
    @test elapsed_between(chain, :onset, :admit) == LogNormal(0.5, 0.4)

    # Origin to a named intermediate event: the convolution of the prefix. It is
    # a scoreable univariate law.
    law = elapsed_between(chain, :admit)
    @test mean(law) ≈ mean(Gamma(2.0, 1.0)) + mean(LogNormal(0.5, 0.4))
    @test rand(law) isa Real
    @test isfinite(logpdf(law, 3.2))

    # A multi-step interval convolves only the steps strictly between the two
    # events, not a difference of totals (which would double-count the shared
    # origin prefix and give the wrong law).
    span = elapsed_between(chain, :onset, :exit)
    @test mean(span) ≈ mean(LogNormal(0.5, 0.4)) + mean(Gamma(1.5, 1.0))
end

@testitem "elapsed_between: rejects reversed, unknown, uncertain and non-chain (#274)" begin
    using ComposedDistributions: update
    using Distributions

    chain = sequential(:origin_onset => Gamma(2.0, 1.0),
        :onset_admit => LogNormal(0.5, 0.4),
        :admit_exit => Gamma(1.5, 1.0))

    # `from` must come strictly before `to`.
    @test_throws ArgumentError elapsed_between(chain, :admit, :onset)
    @test_throws ArgumentError elapsed_between(chain, :onset, :onset)
    # An event name that is not on the chain.
    @test_throws ArgumentError elapsed_between(chain, :nope)

    # An uncertain-leaf chain must be pinned first (the template is not the
    # marginal).
    uncertain_chain = sequential(
        :a_b => (@uncertain Gamma(Normal(2.0, 0.3), 1.0)),
        :b_c => Gamma(1.0, 1.0))
    @test_throws ArgumentError elapsed_between(uncertain_chain, :b, :c)

    # A non-chain operand has no single per-event elapsed layout.
    @test_throws ArgumentError elapsed_between(
        parallel(:a => Gamma(1.0, 1.0), :b => Gamma(2.0, 1.0)), :a)
    @test_throws ArgumentError elapsed_between(Gamma(2.0, 1.0), :x)
end
