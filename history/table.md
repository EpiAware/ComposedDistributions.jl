|                                                                           | d887b413c6d609...   |
|:--------------------------------------------------------------------------|:-------------------:|
| AD gradients/Choose selected-branch logpdf/Enzyme reverse                 | 2.58 ± 0.096 μs     |
| AD gradients/Choose selected-branch logpdf/ForwardDiff                    | 0.596 ± 0.086 μs    |
| AD gradients/Choose selected-branch logpdf/Mooncake reverse               | 21.9 ± 6.5 μs       |
| AD gradients/Choose selected-branch logpdf/ReverseDiff (tape)             | 11.1 ± 0.46 μs      |
| AD gradients/Compete racing-hazard marginal logpdf/Enzyme reverse         | 8.16 ± 0.12 μs      |
| AD gradients/Compete racing-hazard marginal logpdf/ForwardDiff            | 7.8 ± 1.7 μs        |
| AD gradients/Compete racing-hazard marginal logpdf/Mooncake reverse       | 0.0538 ± 0.009 ms   |
| AD gradients/Compete racing-hazard marginal logpdf/ReverseDiff (tape)     | 0.0401 ± 0.007 ms   |
| AD gradients/Pool non-centred reconstruction logpdf/Enzyme reverse        | 1.54 ± 0.02 μs      |
| AD gradients/Pool non-centred reconstruction logpdf/ForwardDiff           | 0.914 ± 0.15 μs     |
| AD gradients/Pool non-centred reconstruction logpdf/Mooncake reverse      | 24.1 ± 5.5 μs       |
| AD gradients/Pool non-centred reconstruction logpdf/ReverseDiff (tape)    | 21 ± 0.89 μs        |
| AD gradients/Resolve mixture marginal logpdf/Enzyme reverse               | 18 ± 0.8 μs         |
| AD gradients/Resolve mixture marginal logpdf/ForwardDiff                  | 5.62 ± 0.93 μs      |
| AD gradients/Resolve mixture marginal logpdf/Mooncake reverse             | 0.0808 ± 0.013 ms   |
| AD gradients/Resolve mixture marginal logpdf/ReverseDiff (tape)           | 0.0543 ± 0.0084 ms  |
| AD gradients/Resolve stick-breaking branch-prob logpdf/Enzyme reverse     | 18.1 ± 0.88 μs      |
| AD gradients/Resolve stick-breaking branch-prob logpdf/ForwardDiff        | 5.94 ± 1 μs         |
| AD gradients/Resolve stick-breaking branch-prob logpdf/Mooncake reverse   | 0.0804 ± 0.013 ms   |
| AD gradients/Resolve stick-breaking branch-prob logpdf/ReverseDiff (tape) | 0.0441 ± 0.0077 ms  |
| AD gradients/Sequential Gamma+LogNormal logpdf/Enzyme reverse             | 0.0846 ± 0.018 ms   |
| AD gradients/Sequential Gamma+LogNormal logpdf/ForwardDiff                | 8.93 ± 1.1 μs       |
| AD gradients/Sequential Gamma+LogNormal logpdf/Mooncake reverse           | 0.255 ± 0.053 ms    |
| AD gradients/Sequential Gamma+LogNormal logpdf/ReverseDiff (tape)         | 22.4 ± 0.9 μs       |
| Composition/Choose/construct                                              | 1.35 ± 0 ns         |
| Composition/Choose/logpdf                                                 | 22 ± 0.04 ns        |
| Composition/Compete/construct                                             | 1.35 ± 0 ns         |
| Composition/Compete/logccdf                                               | 0.266 ± 0.0018 μs   |
| Composition/Compete/rand                                                  | 0.437 ± 0.066 μs    |
| Composition/Nested/compose                                                | 2.3 ± 0.1 μs        |
| Composition/Nested/logpdf                                                 | 1.28 ± 0.088 μs     |
| Composition/Nested/rand                                                   | 3.6 ± 0.17 μs       |
| Composition/Parallel/construct                                            | 1.24 ± 0.07 μs      |
| Composition/Parallel/logpdf                                               | 0.0789 ± 0.023 μs   |
| Composition/Parallel/rand                                                 | 1.08 ± 0.034 μs     |
| Composition/Resolve/construct                                             | 2.7 ± 0.01 ns       |
| Composition/Resolve/logpdf                                                | 0.0853 ± 0.00087 μs |
| Composition/Resolve/rand                                                  | 0.424 ± 0.066 μs    |
| Composition/Sequential/construct                                          | 1.25 ± 0.072 μs     |
| Composition/Sequential/logpdf                                             | 0.0791 ± 0.023 μs   |
| Composition/Sequential/rand                                               | 1.08 ± 0.035 μs     |
| time_to_load                                                              | 0.719 ± 0.0054 s    |

|                                                                           | d887b413c6d609...         |
|:--------------------------------------------------------------------------|:-------------------------:|
| AD gradients/Choose selected-branch logpdf/Enzyme reverse                 | 24  allocs: 1 kB          |
| AD gradients/Choose selected-branch logpdf/ForwardDiff                    | 7  allocs: 0.266 kB       |
| AD gradients/Choose selected-branch logpdf/Mooncake reverse               | 0.28 k allocs: 0.0321 MB  |
| AD gradients/Choose selected-branch logpdf/ReverseDiff (tape)             | 0.228 k allocs: 9.92 kB   |
| AD gradients/Compete racing-hazard marginal logpdf/Enzyme reverse         | 24  allocs: 1.3 kB        |
| AD gradients/Compete racing-hazard marginal logpdf/ForwardDiff            | 0.132 k allocs: 6.2 kB    |
| AD gradients/Compete racing-hazard marginal logpdf/Mooncake reverse       | 0.582 k allocs: 0.0475 MB |
| AD gradients/Compete racing-hazard marginal logpdf/ReverseDiff (tape)     | 0.691 k allocs: 30.5 kB   |
| AD gradients/Pool non-centred reconstruction logpdf/Enzyme reverse        | 10  allocs: 0.297 kB      |
| AD gradients/Pool non-centred reconstruction logpdf/ForwardDiff           | 7  allocs: 0.484 kB       |
| AD gradients/Pool non-centred reconstruction logpdf/Mooncake reverse      | 0.337 k allocs: 29.6 kB   |
| AD gradients/Pool non-centred reconstruction logpdf/ReverseDiff (tape)    | 0.446 k allocs: 17.7 kB   |
| AD gradients/Resolve mixture marginal logpdf/Enzyme reverse               | 0.244 k allocs: 12.6 kB   |
| AD gradients/Resolve mixture marginal logpdf/ForwardDiff                  | 0.052 k allocs: 3.64 kB   |
| AD gradients/Resolve mixture marginal logpdf/Mooncake reverse             | 1.14 k allocs: 0.0482 MB  |
| AD gradients/Resolve mixture marginal logpdf/ReverseDiff (tape)           | 0.815 k allocs: 0.0352 MB |
| AD gradients/Resolve stick-breaking branch-prob logpdf/Enzyme reverse     | 0.248 k allocs: 12.8 kB   |
| AD gradients/Resolve stick-breaking branch-prob logpdf/ForwardDiff        | 0.053 k allocs: 3.73 kB   |
| AD gradients/Resolve stick-breaking branch-prob logpdf/Mooncake reverse   | 1.15 k allocs: 0.0489 MB  |
| AD gradients/Resolve stick-breaking branch-prob logpdf/ReverseDiff (tape) | 0.768 k allocs: 0.033 MB  |
| AD gradients/Sequential Gamma+LogNormal logpdf/Enzyme reverse             | 1.18 k allocs: 0.0498 MB  |
| AD gradients/Sequential Gamma+LogNormal logpdf/ForwardDiff                | 0.139 k allocs: 7.02 kB   |
| AD gradients/Sequential Gamma+LogNormal logpdf/Mooncake reverse           | 4.26 k allocs: 0.155 MB   |
| AD gradients/Sequential Gamma+LogNormal logpdf/ReverseDiff (tape)         | 0.405 k allocs: 16.8 kB   |
| Composition/Choose/construct                                              | 0  allocs: 0 B            |
| Composition/Choose/logpdf                                                 | 0  allocs: 0 B            |
| Composition/Compete/construct                                             | 0  allocs: 0 B            |
| Composition/Compete/logccdf                                               | 0  allocs: 0 B            |
| Composition/Compete/rand                                                  | 8  allocs: 0.219 kB       |
| Composition/Nested/compose                                                | 0.034 k allocs: 1.5 kB    |
| Composition/Nested/logpdf                                                 | 0.043 k allocs: 1.78 kB   |
| Composition/Nested/rand                                                   | 0.056 k allocs: 2.22 kB   |
| Composition/Parallel/construct                                            | 20  allocs: 0.828 kB      |
| Composition/Parallel/logpdf                                               | 2  allocs: 0.0781 kB      |
| Composition/Parallel/rand                                                 | 20  allocs: 0.672 kB      |
| Composition/Resolve/construct                                             | 0  allocs: 0 B            |
| Composition/Resolve/logpdf                                                | 0  allocs: 0 B            |
| Composition/Resolve/rand                                                  | 8  allocs: 0.219 kB       |
| Composition/Sequential/construct                                          | 20  allocs: 0.828 kB      |
| Composition/Sequential/logpdf                                             | 2  allocs: 0.0781 kB      |
| Composition/Sequential/rand                                               | 20  allocs: 0.672 kB      |
| time_to_load                                                              | 0.149 k allocs: 11.2 kB   |

