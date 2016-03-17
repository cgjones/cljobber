# cljobber

Pronounced "clobber".  This is a OpenCL benchmarking program based on krrishnarraj/clpeak.

## Usage

```bash
$ npm install
[...]
$ npm test
====================
[0] Platform Apple
  [0.0] Intel(R) Core(TM) i7-4980HQ CPU @ 2.80GHz
  [0.1] Iris Pro
  [0.2] GeForce GT 750M
====================

$ npm test 1 2
[...]
```

`npm test` accepts the IDs of the compute devices you want to benchmark.

## Caveats

The results seem to be comparable to krrishnarraj/clpeak, but no effort was made to formally verify them.
