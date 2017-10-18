This folder contains a simple benchmark case on how to run the program and how the 
input file should look like.

The best test case is to run the simplefit.dat file using the simplefit.f90 code.
This input file is set up as follows:

     8 2    #  number of data entries and order of polynomial
     0.001 	-2.89017 	0.00073621
     0.002 	-2.88946 	0.00052732
     0.005	-2.89067  	0.00055038
     0.010 	-2.89091 	0.00040973
     0.015 	-2.89084 	0.00034278
     0.02  	-2.89086 	0.00029315
     0.025 	-2.89059	0.00034278
     0.03 	-2.89077 	0.00025017
    The present input file  stems from a variational Monte Carlo calculation of the ground
    state energy (second column). It contains also the standard deviation (3rd column)
    The first column is the time step used in importance sampling. Extrapolating to zero 
    should give the best estimate for the VMC energy.


Run the code as executable  < inputfile > outputfile after having compiled and linked the source
files.
