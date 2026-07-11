|                                                                           | 44cac2d42fae7f...  |
|:--------------------------------------------------------------------------|:------------------:|
| AD gradients/Compete racing-hazard marginal logpdf/Enzyme reverse         | 10 ± 0.15 μs       |
| AD gradients/Compete racing-hazard marginal logpdf/ForwardDiff            | 8.54 ± 1.1 μs      |
| AD gradients/Compete racing-hazard marginal logpdf/Mooncake reverse       | 0.0718 ± 0.016 ms  |
| AD gradients/Compete racing-hazard marginal logpdf/ReverseDiff (tape)     | 0.0531 ± 0.0081 ms |
| AD gradients/Pool non-centred reconstruction logpdf/Enzyme reverse        | 2.03 ± 0.023 μs    |
| AD gradients/Pool non-centred reconstruction logpdf/ForwardDiff           | 1.03 ± 0.041 μs    |
| AD gradients/Pool non-centred reconstruction logpdf/Mooncake reverse      | 0.0329 ± 0.01 ms   |
| AD gradients/Pool non-centred reconstruction logpdf/ReverseDiff (tape)    | 27.9 ± 0.87 μs     |
| AD gradients/Resolve mixture marginal logpdf/Enzyme reverse               | 3.27 ± 0.09 μs     |
| AD gradients/Resolve mixture marginal logpdf/ForwardDiff                  | 11 ± 0.49 μs       |
| AD gradients/Resolve mixture marginal logpdf/Mooncake reverse             | 0.222 ± 0.042 ms   |
| AD gradients/Resolve mixture marginal logpdf/ReverseDiff (tape)           | 0.0632 ± 0.01 ms   |
| AD gradients/Resolve stick-breaking branch-prob logpdf/Enzyme reverse     | 3.32 ± 0.26 μs     |
| AD gradients/Resolve stick-breaking branch-prob logpdf/ForwardDiff        | 11 ± 0.46 μs       |
| AD gradients/Resolve stick-breaking branch-prob logpdf/Mooncake reverse   | 0.217 ± 0.041 ms   |
| AD gradients/Resolve stick-breaking branch-prob logpdf/ReverseDiff (tape) | 0.0509 ± 0.0093 ms |
| AD gradients/Sequential Gamma+LogNormal logpdf/Enzyme reverse             | 0.104 ± 0.017 ms   |
| AD gradients/Sequential Gamma+LogNormal logpdf/ForwardDiff                | 7.66 ± 1.7 μs      |
| AD gradients/Sequential Gamma+LogNormal logpdf/Mooncake reverse           | 0.283 ± 0.059 ms   |
| AD gradients/Sequential Gamma+LogNormal logpdf/ReverseDiff (tape)         | 24.9 ± 0.86 μs     |
| Composition/Choose/construct                                              | 1.74 ± 0.01 ns     |
| Composition/Choose/logpdf                                                 | 29.7 ± 0.06 ns     |
| Composition/Compete/construct                                             | 1.74 ± 0.01 ns     |
| Composition/Compete/logccdf                                               | 0.339 ± 0.0019 μs  |
| Composition/Compete/rand                                                  | 0.547 ± 0.11 μs    |
| Composition/Nested/compose                                                | 1.37 ± 0.1 μs      |
| Composition/Nested/logpdf                                                 | 1.71 ± 0.12 μs     |
| Composition/Nested/rand                                                   | 4.72 ± 0.2 μs      |
| Composition/Parallel/construct                                            | 0.911 ± 0.22 μs    |
| Composition/Parallel/logpdf                                               | 0.105 ± 0.023 μs   |
| Composition/Parallel/rand                                                 | 0.784 ± 0.24 μs    |
| Composition/Resolve/construct                                             | 3.48 ± 0.01 ns     |
| Composition/Resolve/logpdf                                                | 0.108 ± 0.00078 μs |
| Composition/Resolve/rand                                                  | 0.524 ± 0.096 μs   |
| Composition/Sequential/construct                                          | 0.925 ± 0.24 μs    |
| Composition/Sequential/logpdf                                             | 0.103 ± 0.022 μs   |
| Composition/Sequential/rand                                               | 1.83 ± 0.044 μs    |
| time_to_load                                                              | 0.889 ± 0.0069 s   |

|                                                                           | 44cac2d42fae7f...         |
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

