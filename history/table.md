|                                                                           | 2fa68e7570a753...  |
|:--------------------------------------------------------------------------|:------------------:|
| AD gradients/Choose selected-branch logpdf/Enzyme reverse                 | 3.37 ± 0.086 μs    |
| AD gradients/Choose selected-branch logpdf/ForwardDiff                    | 0.743 ± 0.085 μs   |
| AD gradients/Choose selected-branch logpdf/Mooncake reverse               | 27.8 ± 11 μs       |
| AD gradients/Choose selected-branch logpdf/ReverseDiff (tape)             | 14.5 ± 0.43 μs     |
| AD gradients/Compete racing-hazard marginal logpdf/Enzyme reverse         | 9.97 ± 0.09 μs     |
| AD gradients/Compete racing-hazard marginal logpdf/ForwardDiff            | 8.29 ± 1.7 μs      |
| AD gradients/Compete racing-hazard marginal logpdf/Mooncake reverse       | 0.0709 ± 0.015 ms  |
| AD gradients/Compete racing-hazard marginal logpdf/ReverseDiff (tape)     | 0.0527 ± 0.0071 ms |
| AD gradients/Pool non-centred reconstruction logpdf/Enzyme reverse        | 1.95 ± 0.015 μs    |
| AD gradients/Pool non-centred reconstruction logpdf/ForwardDiff           | 1.01 ± 0.031 μs    |
| AD gradients/Pool non-centred reconstruction logpdf/Mooncake reverse      | 31.5 ± 8.9 μs      |
| AD gradients/Pool non-centred reconstruction logpdf/ReverseDiff (tape)    | 28.4 ± 0.67 μs     |
| AD gradients/Resolve mixture marginal logpdf/Enzyme reverse               | 21.9 ± 0.63 μs     |
| AD gradients/Resolve mixture marginal logpdf/ForwardDiff                  | 6.97 ± 0.7 μs      |
| AD gradients/Resolve mixture marginal logpdf/Mooncake reverse             | 0.12 ± 0.017 ms    |
| AD gradients/Resolve mixture marginal logpdf/ReverseDiff (tape)           | 0.0744 ± 0.011 ms  |
| AD gradients/Resolve stick-breaking branch-prob logpdf/Enzyme reverse     | 22.4 ± 0.69 μs     |
| AD gradients/Resolve stick-breaking branch-prob logpdf/ForwardDiff        | 7.03 ± 0.71 μs     |
| AD gradients/Resolve stick-breaking branch-prob logpdf/Mooncake reverse   | 0.122 ± 0.018 ms   |
| AD gradients/Resolve stick-breaking branch-prob logpdf/ReverseDiff (tape) | 0.0622 ± 0.0093 ms |
| AD gradients/Sequential Gamma+LogNormal logpdf/Enzyme reverse             | 0.116 ± 0.02 ms    |
| AD gradients/Sequential Gamma+LogNormal logpdf/ForwardDiff                | 10.7 ± 0.42 μs     |
| AD gradients/Sequential Gamma+LogNormal logpdf/Mooncake reverse           | 0.453 ± 0.06 ms    |
| AD gradients/Sequential Gamma+LogNormal logpdf/ReverseDiff (tape)         | 28.5 ± 0.58 μs     |
| Composition/Choose/construct                                              | 1.55 ± 0.009 ns    |
| Composition/Choose/logpdf                                                 | 29.7 ± 0.1 ns      |
| Composition/Compete/construct                                             | 2.47 ± 0.01 ns     |
| Composition/Compete/logccdf                                               | 0.307 ± 0.0018 μs  |
| Composition/Compete/rand                                                  | 0.561 ± 0.069 μs   |
| Composition/Nested/compose                                                | 3.12 ± 0.084 μs    |
| Composition/Nested/logpdf                                                 | 1.66 ± 0.075 μs    |
| Composition/Nested/rand                                                   | 4.91 ± 0.14 μs     |
| Composition/Parallel/construct                                            | 1.76 ± 0.071 μs    |
| Composition/Parallel/logpdf                                               | 0.101 ± 0.021 μs   |
| Composition/Parallel/rand                                                 | 1.27 ± 0.028 μs    |
| Composition/Resolve/construct                                             | 3.1 ± 0.01 ns      |
| Composition/Resolve/logpdf                                                | 0.113 ± 0.00079 μs |
| Composition/Resolve/rand                                                  | 0.55 ± 0.071 μs    |
| Composition/Sequential/construct                                          | 1.49 ± 0.057 μs    |
| Composition/Sequential/logpdf                                             | 0.101 ± 0.021 μs   |
| Composition/Sequential/rand                                               | 1.27 ± 0.027 μs    |
| time_to_load                                                              | 0.904 ± 0.0074 s   |

|                                                                           | 2fa68e7570a753...         |
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

