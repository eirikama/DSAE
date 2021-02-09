# Descattering Autoencoder



<img border="0" align="Right" src="/img/architecture.png" alt="Your image title" width="350"/>
Implementation of Descattering Autoencoder (DSAE) for Mie-scatter correction of FT-IR 
spectra of cells and tissues. The DSAE takes measured scatter-distorted spectra as input and yields 
scatter corrected spectra. The model is intended to be trained on a subset of measured spectra 
which have been scatter-corrected by the state-of-the-art Mie-scatter correction method ME-EMSC.


Related paper
---------------
For further explanation of the approach see: 

> "Deep convolutional neural network recovers pure absorbance spectrum from highly scatter-distorted spectra of cells", 
> Magnussen E.A., Solheim J.H, Blazhko U, Tafintseva V., Tøndel K, Liland K. H.,  Dzurendova S.,  Shapaval V.,  Sandt C.,  Borondics F.,  Kohler A.,
> J. Biophotonics. 2020; 13:e202000204.
> https://doi.org/10.1002/jbio.202000204


License
---------
CC BY 4.0
