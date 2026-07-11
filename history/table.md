|                                                                           | 7be65715efeb5e...  |
|:--------------------------------------------------------------------------|:------------------:|
| AD gradients/Compete racing-hazard marginal logpdf/Enzyme reverse         | 10.3 ± 0.14 μs     |
| AD gradients/Compete racing-hazard marginal logpdf/ForwardDiff            | 8.47 ± 1.1 μs      |
| AD gradients/Compete racing-hazard marginal logpdf/Mooncake reverse       | 0.0728 ± 0.0063 ms |
| AD gradients/Compete racing-hazard marginal logpdf/ReverseDiff (tape)     | 0.052 ± 0.0081 ms  |
| AD gradients/Pool non-centred reconstruction logpdf/Enzyme reverse        | 2.09 ± 0.029 μs    |
| AD gradients/Pool non-centred reconstruction logpdf/ForwardDiff           | 1.03 ± 0.042 μs    |
| AD gradients/Pool non-centred reconstruction logpdf/Mooncake reverse      | 0.0322 ± 0.0093 ms |
| AD gradients/Pool non-centred reconstruction logpdf/ReverseDiff (tape)    | 26.9 ± 0.95 μs     |
| AD gradients/Resolve mixture marginal logpdf/Enzyme reverse               | 3.39 ± 0.087 μs    |
| AD gradients/Resolve mixture marginal logpdf/ForwardDiff                  | 8.7 ± 0.44 μs      |
| AD gradients/Resolve mixture marginal logpdf/Mooncake reverse             | 0.225 ± 0.046 ms   |
| AD gradients/Resolve mixture marginal logpdf/ReverseDiff (tape)           | 0.0602 ± 0.01 ms   |
| AD gradients/Resolve stick-breaking branch-prob logpdf/Enzyme reverse     | 3.45 ± 0.26 μs     |
| AD gradients/Resolve stick-breaking branch-prob logpdf/ForwardDiff        | 10.8 ± 2.4 μs      |
| AD gradients/Resolve stick-breaking branch-prob logpdf/Mooncake reverse   | 0.22 ± 0.043 ms    |
| AD gradients/Resolve stick-breaking branch-prob logpdf/ReverseDiff (tape) | 0.0502 ± 0.0095 ms |
| AD gradients/Sequential Gamma+LogNormal logpdf/Enzyme reverse             | 0.106 ± 0.019 ms   |
| AD gradients/Sequential Gamma+LogNormal logpdf/ForwardDiff                | 7.51 ± 1.6 μs      |
| AD gradients/Sequential Gamma+LogNormal logpdf/Mooncake reverse           | 0.279 ± 0.063 ms   |
| AD gradients/Sequential Gamma+LogNormal logpdf/ReverseDiff (tape)         | 24.7 ± 0.7 μs      |
| Composition/Choose/construct                                              | 1.74 ± 0.01 ns     |
| Composition/Choose/logpdf                                                 | 29.8 ± 0.061 ns    |
| Composition/Compete/construct                                             | 1.74 ± 0.01 ns     |
| Composition/Compete/logccdf                                               | 0.336 ± 0.0021 μs  |
| Composition/Compete/rand                                                  | 0.54 ± 0.12 μs     |
| Composition/Nested/compose                                                | 1.42 ± 0.095 μs    |
| Composition/Nested/logpdf                                                 | 1.72 ± 0.11 μs     |
| Composition/Nested/rand                                                   | 4.87 ± 0.18 μs     |
| Composition/Parallel/construct                                            | 0.899 ± 0.2 μs     |
| Composition/Parallel/logpdf                                               | 0.105 ± 0.023 μs   |
| Composition/Parallel/rand                                                 | 0.913 ± 0.043 μs   |
| Composition/Resolve/construct                                             | 3.5 ± 0.01 ns      |
| Composition/Resolve/logpdf                                                | 0.109 ± 0.0014 μs  |
| Composition/Resolve/rand                                                  | 0.521 ± 0.11 μs    |
| Composition/Sequential/construct                                          | 0.904 ± 0.21 μs    |
| Composition/Sequential/logpdf                                             | 0.104 ± 0.024 μs   |
| Composition/Sequential/rand                                               | 2.24 ± 0.041 μs    |
| time_to_load                                                              | 0.865 ± 0.0028 s   |

|                                                                           | 7be65715efeb5e...         |
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

