|                                                                           | 80f4f67e0a6810...  |
|:--------------------------------------------------------------------------|:------------------:|
| AD gradients/Compete racing-hazard marginal logpdf/Enzyme reverse         | 10.1 ± 0.11 μs     |
| AD gradients/Compete racing-hazard marginal logpdf/ForwardDiff            | 8.58 ± 1.7 μs      |
| AD gradients/Compete racing-hazard marginal logpdf/Mooncake reverse       | 0.0746 ± 0.016 ms  |
| AD gradients/Compete racing-hazard marginal logpdf/ReverseDiff (tape)     | 0.0536 ± 0.0073 ms |
| AD gradients/Pool non-centred reconstruction logpdf/Enzyme reverse        | 1.97 ± 0.019 μs    |
| AD gradients/Pool non-centred reconstruction logpdf/ForwardDiff           | 1.07 ± 0.032 μs    |
| AD gradients/Pool non-centred reconstruction logpdf/Mooncake reverse      | 0.0323 ± 0.0089 ms |
| AD gradients/Pool non-centred reconstruction logpdf/ReverseDiff (tape)    | 27.6 ± 1 μs        |
| AD gradients/Resolve mixture marginal logpdf/Enzyme reverse               | 3.26 ± 0.088 μs    |
| AD gradients/Resolve mixture marginal logpdf/ForwardDiff                  | 9.88 ± 2.1 μs      |
| AD gradients/Resolve mixture marginal logpdf/Mooncake reverse             | 0.25 ± 0.04 ms     |
| AD gradients/Resolve mixture marginal logpdf/ReverseDiff (tape)           | 0.0678 ± 0.0099 ms |
| AD gradients/Resolve stick-breaking branch-prob logpdf/Enzyme reverse     | 3.27 ± 0.24 μs     |
| AD gradients/Resolve stick-breaking branch-prob logpdf/ForwardDiff        | 9.83 ± 2 μs        |
| AD gradients/Resolve stick-breaking branch-prob logpdf/Mooncake reverse   | 0.245 ± 0.041 ms   |
| AD gradients/Resolve stick-breaking branch-prob logpdf/ReverseDiff (tape) | 0.0531 ± 0.0084 ms |
| AD gradients/Sequential Gamma+LogNormal logpdf/Enzyme reverse             | 0.103 ± 0.021 ms   |
| AD gradients/Sequential Gamma+LogNormal logpdf/ForwardDiff                | 7.17 ± 1.5 μs      |
| AD gradients/Sequential Gamma+LogNormal logpdf/Mooncake reverse           | 0.375 ± 0.061 ms   |
| AD gradients/Sequential Gamma+LogNormal logpdf/ReverseDiff (tape)         | 24.6 ± 0.81 μs     |
| Composition/Choose/construct                                              | 1.55 ± 0.01 ns     |
| Composition/Choose/logpdf                                                 | 0.0318 ± 0.0001 μs |
| Composition/Compete/construct                                             | 1.55 ± 0.01 ns     |
| Composition/Compete/logccdf                                               | 0.315 ± 0.0017 μs  |
| Composition/Compete/rand                                                  | 0.596 ± 0.11 μs    |
| Composition/Nested/compose                                                | 1.65 ± 0.084 μs    |
| Composition/Nested/logpdf                                                 | 1.75 ± 0.086 μs    |
| Composition/Nested/rand                                                   | 5.07 ± 0.15 μs     |
| Composition/Parallel/construct                                            | 0.948 ± 0.13 μs    |
| Composition/Parallel/logpdf                                               | 0.107 ± 0.028 μs   |
| Composition/Parallel/rand                                                 | 0.903 ± 0.25 μs    |
| Composition/Resolve/construct                                             | 3.41 ± 0.01 ns     |
| Composition/Resolve/logpdf                                                | 0.113 ± 0.00047 μs |
| Composition/Resolve/rand                                                  | 0.569 ± 0.1 μs     |
| Composition/Sequential/construct                                          | 0.879 ± 0.19 μs    |
| Composition/Sequential/logpdf                                             | 0.107 ± 0.028 μs   |
| Composition/Sequential/rand                                               | 2.01 ± 0.034 μs    |
| time_to_load                                                              | 0.911 ± 0.0055 s   |

|                                                                           | 80f4f67e0a6810...         |
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

