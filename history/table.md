|                                                                           | 2c15541087c3dc...  |
|:--------------------------------------------------------------------------|:------------------:|
| AD gradients/Compete racing-hazard marginal logpdf/Enzyme reverse         | 10.3 ± 0.091 μs    |
| AD gradients/Compete racing-hazard marginal logpdf/ForwardDiff            | 8.41 ± 1.3 μs      |
| AD gradients/Compete racing-hazard marginal logpdf/Mooncake reverse       | 0.0748 ± 0.016 ms  |
| AD gradients/Compete racing-hazard marginal logpdf/ReverseDiff (tape)     | 0.0531 ± 0.0072 ms |
| AD gradients/Pool non-centred reconstruction logpdf/Enzyme reverse        | 1.94 ± 0.016 μs    |
| AD gradients/Pool non-centred reconstruction logpdf/ForwardDiff           | 1.01 ± 0.026 μs    |
| AD gradients/Pool non-centred reconstruction logpdf/Mooncake reverse      | 0.0324 ± 0.009 ms  |
| AD gradients/Pool non-centred reconstruction logpdf/ReverseDiff (tape)    | 27.1 ± 0.97 μs     |
| AD gradients/Resolve mixture marginal logpdf/Enzyme reverse               | 3.41 ± 0.081 μs    |
| AD gradients/Resolve mixture marginal logpdf/ForwardDiff                  | 9.4 ± 2 μs         |
| AD gradients/Resolve mixture marginal logpdf/Mooncake reverse             | 0.258 ± 0.038 ms   |
| AD gradients/Resolve mixture marginal logpdf/ReverseDiff (tape)           | 0.0664 ± 0.0096 ms |
| AD gradients/Resolve stick-breaking branch-prob logpdf/Enzyme reverse     | 3.21 ± 0.23 μs     |
| AD gradients/Resolve stick-breaking branch-prob logpdf/ForwardDiff        | 9.61 ± 2 μs        |
| AD gradients/Resolve stick-breaking branch-prob logpdf/Mooncake reverse   | 0.249 ± 0.042 ms   |
| AD gradients/Resolve stick-breaking branch-prob logpdf/ReverseDiff (tape) | 0.055 ± 0.0081 ms  |
| AD gradients/Sequential Gamma+LogNormal logpdf/Enzyme reverse             | 0.105 ± 0.02 ms    |
| AD gradients/Sequential Gamma+LogNormal logpdf/ForwardDiff                | 7.97 ± 1.5 μs      |
| AD gradients/Sequential Gamma+LogNormal logpdf/Mooncake reverse           | 0.352 ± 0.062 ms   |
| AD gradients/Sequential Gamma+LogNormal logpdf/ReverseDiff (tape)         | 24.6 ± 0.67 μs     |
| Composition/Choose/construct                                              | 1.55 ± 0.01 ns     |
| Composition/Choose/logpdf                                                 | 31.5 ± 0.15 ns     |
| Composition/Compete/construct                                             | 1.55 ± 0.01 ns     |
| Composition/Compete/logccdf                                               | 0.312 ± 0.002 μs   |
| Composition/Compete/rand                                                  | 0.589 ± 0.097 μs   |
| Composition/Nested/compose                                                | 1.47 ± 0.079 μs    |
| Composition/Nested/logpdf                                                 | 1.71 ± 0.079 μs    |
| Composition/Nested/rand                                                   | 5.12 ± 0.17 μs     |
| Composition/Parallel/construct                                            | 0.94 ± 0.24 μs     |
| Composition/Parallel/logpdf                                               | 0.106 ± 0.028 μs   |
| Composition/Parallel/rand                                                 | 0.82 ± 0.21 μs     |
| Composition/Resolve/construct                                             | 21.6 ± 0.02 ns     |
| Composition/Resolve/logpdf                                                | 0.113 ± 0.0007 μs  |
| Composition/Resolve/rand                                                  | 0.561 ± 0.096 μs   |
| Composition/Sequential/construct                                          | 0.929 ± 0.18 μs    |
| Composition/Sequential/logpdf                                             | 0.106 ± 0.028 μs   |
| Composition/Sequential/rand                                               | 1.87 ± 0.037 μs    |
| time_to_load                                                              | 0.861 ± 0.013 s    |

|                                                                           | 2c15541087c3dc...         |
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

