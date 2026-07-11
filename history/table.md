|                                                                           | 8d44dfb071892e...  |
|:--------------------------------------------------------------------------|:------------------:|
| AD gradients/Compete racing-hazard marginal logpdf/Enzyme reverse         | 10.1 ± 0.091 μs    |
| AD gradients/Compete racing-hazard marginal logpdf/ForwardDiff            | 8.33 ± 1.2 μs      |
| AD gradients/Compete racing-hazard marginal logpdf/Mooncake reverse       | 0.0776 ± 0.018 ms  |
| AD gradients/Compete racing-hazard marginal logpdf/ReverseDiff (tape)     | 0.0535 ± 0.0077 ms |
| AD gradients/Pool non-centred reconstruction logpdf/Enzyme reverse        | 2.07 ± 0.024 μs    |
| AD gradients/Pool non-centred reconstruction logpdf/ForwardDiff           | 1.1 ± 0.033 μs     |
| AD gradients/Pool non-centred reconstruction logpdf/Mooncake reverse      | 0.033 ± 0.0089 ms  |
| AD gradients/Pool non-centred reconstruction logpdf/ReverseDiff (tape)    | 0.0339 ± 0.0014 ms |
| AD gradients/Resolve mixture marginal logpdf/Enzyme reverse               | 3.18 ± 0.089 μs    |
| AD gradients/Resolve mixture marginal logpdf/ForwardDiff                  | 9.68 ± 2 μs        |
| AD gradients/Resolve mixture marginal logpdf/Mooncake reverse             | 0.264 ± 0.043 ms   |
| AD gradients/Resolve mixture marginal logpdf/ReverseDiff (tape)           | 0.0669 ± 0.0096 ms |
| AD gradients/Resolve stick-breaking branch-prob logpdf/Enzyme reverse     | 3.34 ± 0.2 μs      |
| AD gradients/Resolve stick-breaking branch-prob logpdf/ForwardDiff        | 9.81 ± 1.9 μs      |
| AD gradients/Resolve stick-breaking branch-prob logpdf/Mooncake reverse   | 0.268 ± 0.048 ms   |
| AD gradients/Resolve stick-breaking branch-prob logpdf/ReverseDiff (tape) | 0.0572 ± 0.0085 ms |
| AD gradients/Sequential Gamma+LogNormal logpdf/Enzyme reverse             | 0.109 ± 0.021 ms   |
| AD gradients/Sequential Gamma+LogNormal logpdf/ForwardDiff                | 8.54 ± 0.57 μs     |
| AD gradients/Sequential Gamma+LogNormal logpdf/Mooncake reverse           | 0.394 ± 0.061 ms   |
| AD gradients/Sequential Gamma+LogNormal logpdf/ReverseDiff (tape)         | 25.9 ± 0.68 μs     |
| Composition/Choose/construct                                              | 1.55 ± 0.009 ns    |
| Composition/Choose/logpdf                                                 | 31.3 ± 0.14 ns     |
| Composition/Compete/construct                                             | 1.55 ± 0.01 ns     |
| Composition/Compete/logccdf                                               | 0.315 ± 0.002 μs   |
| Composition/Compete/rand                                                  | 0.584 ± 0.11 μs    |
| Composition/Nested/compose                                                | 1.57 ± 0.08 μs     |
| Composition/Nested/logpdf                                                 | 1.75 ± 0.094 μs    |
| Composition/Nested/rand                                                   | 5.28 ± 0.17 μs     |
| Composition/Parallel/construct                                            | 0.995 ± 0.034 μs   |
| Composition/Parallel/logpdf                                               | 0.109 ± 0.028 μs   |
| Composition/Parallel/rand                                                 | 1.03 ± 0.029 μs    |
| Composition/Resolve/construct                                             | 3.41 ± 0.01 ns     |
| Composition/Resolve/logpdf                                                | 0.113 ± 0.00027 μs |
| Composition/Resolve/rand                                                  | 0.547 ± 0.11 μs    |
| Composition/Sequential/construct                                          | 0.977 ± 0.19 μs    |
| Composition/Sequential/logpdf                                             | 0.109 ± 0.028 μs   |
| Composition/Sequential/rand                                               | 2.23 ± 0.041 μs    |
| time_to_load                                                              | 0.87 ± 0.012 s     |

|                                                                           | 8d44dfb071892e...         |
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

