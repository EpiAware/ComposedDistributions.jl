|                                                                           | c627d3f156288c...   |
|:--------------------------------------------------------------------------|:-------------------:|
| AD gradients/Compete racing-hazard marginal logpdf/Enzyme reverse         | 10.1 ± 0.08 μs      |
| AD gradients/Compete racing-hazard marginal logpdf/ForwardDiff            | 8.5 ± 1.4 μs        |
| AD gradients/Compete racing-hazard marginal logpdf/Mooncake reverse       | 0.0748 ± 0.016 ms   |
| AD gradients/Compete racing-hazard marginal logpdf/ReverseDiff (tape)     | 0.0542 ± 0.0073 ms  |
| AD gradients/Pool non-centred reconstruction logpdf/Enzyme reverse        | 1.97 ± 0.016 μs     |
| AD gradients/Pool non-centred reconstruction logpdf/ForwardDiff           | 1.04 ± 0.032 μs     |
| AD gradients/Pool non-centred reconstruction logpdf/Mooncake reverse      | 0.032 ± 0.0087 ms   |
| AD gradients/Pool non-centred reconstruction logpdf/ReverseDiff (tape)    | 27.8 ± 1.7 μs       |
| AD gradients/Resolve mixture marginal logpdf/Enzyme reverse               | 3.16 ± 0.07 μs      |
| AD gradients/Resolve mixture marginal logpdf/ForwardDiff                  | 9.6 ± 2 μs          |
| AD gradients/Resolve mixture marginal logpdf/Mooncake reverse             | 0.268 ± 0.049 ms    |
| AD gradients/Resolve mixture marginal logpdf/ReverseDiff (tape)           | 0.0667 ± 0.0095 ms  |
| AD gradients/Resolve stick-breaking branch-prob logpdf/Enzyme reverse     | 3.37 ± 0.22 μs      |
| AD gradients/Resolve stick-breaking branch-prob logpdf/ForwardDiff        | 9.85 ± 1.9 μs       |
| AD gradients/Resolve stick-breaking branch-prob logpdf/Mooncake reverse   | 0.259 ± 0.044 ms    |
| AD gradients/Resolve stick-breaking branch-prob logpdf/ReverseDiff (tape) | 0.0569 ± 0.0083 ms  |
| AD gradients/Sequential Gamma+LogNormal logpdf/Enzyme reverse             | 0.105 ± 0.02 ms     |
| AD gradients/Sequential Gamma+LogNormal logpdf/ForwardDiff                | 7.52 ± 1.5 μs       |
| AD gradients/Sequential Gamma+LogNormal logpdf/Mooncake reverse           | 0.362 ± 0.062 ms    |
| AD gradients/Sequential Gamma+LogNormal logpdf/ReverseDiff (tape)         | 25.2 ± 0.72 μs      |
| Composition/Choose/construct                                              | 1.55 ± 0.009 ns     |
| Composition/Choose/logpdf                                                 | 0.0316 ± 9.1e-05 μs |
| Composition/Compete/construct                                             | 1.55 ± 0.009 ns     |
| Composition/Compete/logccdf                                               | 0.315 ± 0.0019 μs   |
| Composition/Compete/rand                                                  | 0.572 ± 0.11 μs     |
| Composition/Nested/compose                                                | 1.96 ± 0.08 μs      |
| Composition/Nested/logpdf                                                 | 1.72 ± 0.076 μs     |
| Composition/Nested/rand                                                   | 5.18 ± 0.17 μs      |
| Composition/Parallel/construct                                            | 1.09 ± 0.066 μs     |
| Composition/Parallel/logpdf                                               | 0.107 ± 0.029 μs    |
| Composition/Parallel/rand                                                 | 0.845 ± 0.23 μs     |
| Composition/Resolve/construct                                             | 3.41 ± 0.01 ns      |
| Composition/Resolve/logpdf                                                | 0.114 ± 0.00078 μs  |
| Composition/Resolve/rand                                                  | 0.562 ± 0.11 μs     |
| Composition/Sequential/construct                                          | 0.928 ± 0.18 μs     |
| Composition/Sequential/logpdf                                             | 0.106 ± 0.03 μs     |
| Composition/Sequential/rand                                               | 2.03 ± 0.033 μs     |
| time_to_load                                                              | 0.868 ± 0.0038 s    |

|                                                                           | c627d3f156288c...         |
|:--------------------------------------------------------------------------|:-------------------------:|
| AD gradients/Compete racing-hazard marginal logpdf/Enzyme reverse         | 24  allocs: 1.3 kB        |
| AD gradients/Compete racing-hazard marginal logpdf/ForwardDiff            | 0.132 k allocs: 6.2 kB    |
| AD gradients/Compete racing-hazard marginal logpdf/Mooncake reverse       | 0.596 k allocs: 0.0522 MB |
| AD gradients/Compete racing-hazard marginal logpdf/ReverseDiff (tape)     | 0.691 k allocs: 30.5 kB   |
| AD gradients/Pool non-centred reconstruction logpdf/Enzyme reverse        | 10  allocs: 0.297 kB      |
| AD gradients/Pool non-centred reconstruction logpdf/ForwardDiff           | 7  allocs: 0.484 kB       |
| AD gradients/Pool non-centred reconstruction logpdf/Mooncake reverse      | 0.343 k allocs: 30 kB     |
| AD gradients/Pool non-centred reconstruction logpdf/ReverseDiff (tape)    | 0.446 k allocs: 17.7 kB   |
| AD gradients/Resolve mixture marginal logpdf/Enzyme reverse               | 14  allocs: 1.11 kB       |
| AD gradients/Resolve mixture marginal logpdf/ForwardDiff                  | 0.052 k allocs: 6.06 kB   |
| AD gradients/Resolve mixture marginal logpdf/Mooncake reverse             | 2.45 k allocs: 0.148 MB   |
| AD gradients/Resolve mixture marginal logpdf/ReverseDiff (tape)           | 0.775 k allocs: 0.0337 MB |
| AD gradients/Resolve stick-breaking branch-prob logpdf/Enzyme reverse     | 18  allocs: 1.28 kB       |
| AD gradients/Resolve stick-breaking branch-prob logpdf/ForwardDiff        | 0.053 k allocs: 6.16 kB   |
| AD gradients/Resolve stick-breaking branch-prob logpdf/Mooncake reverse   | 2.44 k allocs: 0.13 MB    |
| AD gradients/Resolve stick-breaking branch-prob logpdf/ReverseDiff (tape) | 0.723 k allocs: 31.9 kB   |
| AD gradients/Sequential Gamma+LogNormal logpdf/Enzyme reverse             | 1.18 k allocs: 0.0516 MB  |
| AD gradients/Sequential Gamma+LogNormal logpdf/ForwardDiff                | 0.114 k allocs: 5.92 kB   |
| AD gradients/Sequential Gamma+LogNormal logpdf/Mooncake reverse           | 3.91 k allocs: 0.144 MB   |
| AD gradients/Sequential Gamma+LogNormal logpdf/ReverseDiff (tape)         | 0.38 k allocs: 15.8 kB    |
| Composition/Choose/construct                                              | 0  allocs: 0 B            |
| Composition/Choose/logpdf                                                 | 0  allocs: 0 B            |
| Composition/Compete/construct                                             | 0  allocs: 0 B            |
| Composition/Compete/logccdf                                               | 0  allocs: 0 B            |
| Composition/Compete/rand                                                  | 9  allocs: 0.25 kB        |
| Composition/Nested/compose                                                | 24  allocs: 1.16 kB       |
| Composition/Nested/logpdf                                                 | 0.044 k allocs: 1.98 kB   |
| Composition/Nested/rand                                                   | 0.057 k allocs: 2.45 kB   |
| Composition/Parallel/construct                                            | 15  allocs: 0.641 kB      |
| Composition/Parallel/logpdf                                               | 2  allocs: 0.0781 kB      |
| Composition/Parallel/rand                                                 | 18  allocs: 0.578 kB      |
| Composition/Resolve/construct                                             | 0  allocs: 0 B            |
| Composition/Resolve/logpdf                                                | 0  allocs: 0 B            |
| Composition/Resolve/rand                                                  | 9  allocs: 0.25 kB        |
| Composition/Sequential/construct                                          | 15  allocs: 0.641 kB      |
| Composition/Sequential/logpdf                                             | 2  allocs: 0.0781 kB      |
| Composition/Sequential/rand                                               | 19  allocs: 0.625 kB      |
| time_to_load                                                              | 0.149 k allocs: 11.2 kB   |

