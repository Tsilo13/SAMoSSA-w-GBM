There is a better version of the SAMoSSA code at the creator's github here https://github.com/AbdullahO/SAMoSSA

I made this because I wanted to try my own hand at coding out the individual layers of the process for myself. 
  You'll find this code only works on univariate data (hence why the creator's code is superior), using single value decomposition as opposed to
  HSVT (High-dimensional Singular Value Thresholding), which they use for multivariate input.
I wanted to experiment with Autoregression as applied to the time series data by comparing it, side by side, to another method of modeling the stochastic element.
I chose to compare it to Geometric Brownian Motion, which seemed like a good idea at the time. 

If you want to use the code just install the environment and fill out which stock ticker and time you want to try. This code will then use the yfinance library to retrieve closing prices 
and fill out the matrices using that. 

Here is a link to the original paper, it is worth a read. 
https://arxiv.org/pdf/2305.16491

Please let me know If you find any errors in my code, or maybe any suggestions on how i could do it better.
Thank you!
