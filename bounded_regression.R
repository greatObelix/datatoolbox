# bounded regression, upper bound = %50 above the coefficient, start = %10 above, lower = point
############### Bounds ###################
startPtsupper <- 	list( beta1 = 5.0325 , beta2 = 5.0835 , beta3 = 5.664 ,  beta4 = 0.1965 , beta5 = 2.19 , beta6 = 2.0865 , beta7 = 2.0865 , beta8 = 1.335 , beta9 = 0.9135 , beta10 = 0.2385 , beta11 = 1.8285 , beta12 = 1.8285 , beta13 = 3.018 , beta14 = 0.063 ,  beta15 = 3.018 ) 
startPts <- 		list( beta1 = 3.6905 , beta2 = 3.7279 , beta3 = 4.1536 , beta4 = 0.1441 , beta5 = 1.606 ,beta6 = 1.5301 , beta7 = 1.5301 , beta8 = 0.979 , beta9 = 0.6699 , beta10 = 0.1749 , beta11 = 1.3409 , beta12 = 1.3409 , beta13 = 2.2132 , beta14 = 0.0462 ,beta15 = 2.2132 )
startPtslower <-  	list( beta1 = 3.355 ,  beta2 = 3.389 ,  beta3 = 3.776 ,  beta4 = 0.131 ,  beta5 = 1.46 , beta6 = 1.391 ,  beta7 = 1.391 ,  beta8 = 0.89 ,  beta9 = 0.609 ,  beta10 = 0.159 ,  beta11 = 1.219 ,  beta12 = 1.219 ,  beta13 = 2.012 ,  beta14 = 0.042 , beta15 = 2.012 ) 
############### Bounds ###################

ourFormula <- y ~ (beta1 * x[,1]) + (beta2 * x[,2]) + (beta3 * x[,3]) + (beta4 * x[,4]) + (beta5 * x[,5]) +  (beta6 * x[,6]) + (beta7 * x[,7]) + (beta8 * x[,8]) + (beta9 * x[,9]) + (beta10 * x[,10]) + (beta11 * x[,11]) + (beta12 *  x[,12]) + (beta13 * x[,13]) + (beta14 * x[,14]) + (beta15 * x[,15])

out <- file("C:\\cygwin\\home\\","w");

usedVars = list(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);

# input directory, 1 file = 1 RNC
filePath = "C:\\cygwin\\home\\"
setwd(filePath);
dirList = list.files(pattern=".csv");

numTrain = 1000; #num of row in input file
for (file in dirList){
	xdata <- read.csv(file, head=TRUE,sep=","); # 'sep' = separator like csv file
	x = xdata[1:numTrain,1:15] #input to regression , intensity of events
	y = (xdata[1:numTrain,16]) #empirical point

	RNCid = substr(file,1,14)[1] #RNC ID extract filename, char1 to char14

        #actual regression
        #alg must keep as "port" to use bounded
        #control => maxiter = max iteration to get a stable set of coefficients
	ourlin <- nls(ourFormula,start=startPts ,alg="port", lower=startPtslower , upper=startPtsupper, control=list(maxiter = 500));

	pvec <- c(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
	idx = 1;
	for (j in usedVars){
		pvec[j] = coefficients(ourlin)[idx];
		idx = idx + 1;
	}
	ourLine = sprintf("%s,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f",RNCid,pvec[1],pvec[2],pvec[3],pvec[4],pvec[5],pvec[6],pvec[7],pvec[8],pvec[9],pvec[10],pvec[11],pvec[12],pvec[13],pvec[14],pvec[15]);


	writeLines(ourLine,con=out,sep = "\n")
}

close(out);



