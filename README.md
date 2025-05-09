# Black-Scholes-Numerical-Solvers

This repository contains various numerical solvers for the Black-Scholes equation using different methods and approaches. The solvers are implemented for the Fractional Black-Scholes equation, with any choice of parameters you would like.

## Overview

The following solvers are included in this repository:

1. **LieSplittingTwoSteps**: 
   - This solver implements the **Lie Splitting** method using **two steps**. It separates the Black-Scholes equation into two parts and solves each part separately.
   
2. **LieSplittingThreeSteps**: 
   - This solver implements the **Lie Splitting** method using **three steps**. This approach has more error, but is still included for completeness.

3. **CarrMadan**: 
   - This solver implements the **Carr-Madan integral** method. It's a widely used method for option pricing, leveraging Fourier transforms for efficient pricing. Note that this method was not written by us, but it has been included for completeness.

4. **LogTransform**: 
   - This solver uses the **Lie Splitting method** with a **logarithmic transformation**. The log transformation simplifies the Black-Scholes equation by converting the problem into a simpler form for numerical solving.

## Getting Started

### Requirements

To run the solvers, you'll need to have the following Python libraries installed:

- `numpy`
- `scipy`
- `matplotlib`
- `fft`

You can install them via `pip`:

```bash
pip install numpy scipy matplotlib
