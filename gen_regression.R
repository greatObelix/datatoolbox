library(ggplot2)

outSummary <- file("C:\\Users\\mh636c\\Desktop\\","w");  
outFit <- file("C:\\Users\\mh636c\\Desktop\\","w");  
infile = "C:\\Users\\mh636c\\Desktop\\"

prettyprint <- function(kpi, ourlin) {

  #output summary
  outLine = sprintf("%s,%f,%f",kpi,summary(ourlin)$r.squared,coefficients(ourlin));
  writeLines(outLine,con=outSummary,sep = "\n");
  
  #output fitted line and confidence intervalc
  writeLines(kpi,con=outFit,sep = "\n");
  fitted_x <- data.frame(PRB=seq(0,100,10));
  if (attributes(ourlin)$class == "lm") {
    #used if lm model is used
    outLine = sprintf("%s", capture.output(predict(ourlin, fitted_x, interval="confidence"))); 
  }
  else if (attributes(ourlin)$class == "nls") {
    #used if nls model is used
    outLine = sprintf("%s", capture.output(predictNLS(ourlin,fitted_x))); 
  }
  else {
    outLine = "Error! regression function for prediction not available";
  }
  writeLines(outLine,con=outFit,sep = "\n");
}

call_plot <- function(ourlin) {
  #Plot scatter
  plot(data[,4],data[,5]);
  #points(rrcConnUe,y,col='red') #use only if plot 2 graphs
  
  #Plot for different fitting fcn
  DF <- data.frame(data[,4],data[,5]);
  ggplot(DF, aes(x = data[,4], y = data[,5])) + geom_point() +
    stat_smooth(method = 'lm', aes(colour = 'linear'), se = FALSE) +
    stat_smooth(method = 'lm', formula = y ~ x + I(x^2), aes(colour = 'polynomial'), se= FALSE) +
    stat_smooth(method = 'nls', formula = y ~ a * log(x) +b, aes(colour = 'logarithmic'), se = FALSE, start = list(a=-1,b=5), control = list(maxiter = 50), trace = TRUE) +
    stat_smooth(method = 'nls', formula = y ~ a*exp(b *x) + b, aes(colour = 'Exponential'), se = FALSE, start = list(a=1,b=1)) +
    theme_bw() +
    scale_colour_brewer(name = 'Trendline', palette = 'Set2');
  
  
  #plot confident interval
  PRB_perct <- data.frame(PRB=seq(0,100,10));
  pred.w.clim <- predict(ourlin, PRB_perct, interval="confidence")
  matplot(PRB_perct, pred.w.clim,lty = c(1,2,2), type = "l", xlab = "PRB%", ylab = "predicted y");
  legend("topleft",c("Predicted","Lower Bound","Upper Bound"), lty = c(1,2,2), col=c("black","red","green"));
  abline(0,0); #straight line
}

#predict NLS model with confidence interval using Monte Carlo simulation approach
predictNLS <- function(object, newdata,level = 0.95, nsim = 10000, ...){
  require(MASS, quietly = TRUE)
  
  ## get right-hand side of formula
  RHS <- as.list(object$call$formula)[[3]]
  EXPR <- as.expression(RHS)
  
  ## all variables in model
  VARS <- all.vars(EXPR)
  
  ## coefficients
  COEF <- coef(object)
  
  ## extract predictor variable    
  predNAME <- setdiff(VARS, names(COEF))  
  
  ## take fitted values, if 'newdata' is missing
  if (missing(newdata)) {
    newdata <- eval(object$data)[predNAME]
    colnames(newdata) <- predNAME
  }
  
  ## check that 'newdata' has same name as predVAR
  if (names(newdata)[1] != predNAME) stop("newdata should have name '", predNAME, "'!")
  
  ## get parameter coefficients
  COEF <- coef(object)
  
  ## get variance-covariance matrix
  VCOV <- vcov(object)
  
  ## augment variance-covariance matrix for 'mvrnorm' 
  ## by adding a column/row for 'error in x'
  NCOL <- ncol(VCOV)
  ADD1 <- c(rep(0, NCOL))
  ADD1 <- matrix(ADD1, ncol = 1)
  colnames(ADD1) <- predNAME
  VCOV <- cbind(VCOV, ADD1)
  ADD2 <- c(rep(0, NCOL + 1))
  ADD2 <- matrix(ADD2, nrow = 1)
  rownames(ADD2) <- predNAME
  VCOV <- rbind(VCOV, ADD2) 
  
  ## iterate over all entries in 'newdata' as in usual 'predict.' functions
  NR <- nrow(newdata)
  respVEC <- numeric(NR)
  seVEC <- numeric(NR)
  varPLACE <- ncol(VCOV)   
  
  ## define counter function
  counter <- function (i) 
  {
    if (i%%10 == 0) 
      cat(i)
    else cat(".")
    if (i%%50 == 0) 
      cat("\n")
    flush.console()
  }
  
  outMAT <- NULL 
  
  for (i in 1:NR) {
    counter(i)
    
    ## get predictor values and optional errors
    predVAL <- newdata[i, 1]
    if (ncol(newdata) == 2) predERROR <- newdata[i, 2] else predERROR <- 0
    names(predVAL) <- predNAME  
    names(predERROR) <- predNAME  
    
    ## create mean vector for 'mvrnorm'
    MU <- c(COEF, predVAL)
    
    ## create variance-covariance matrix for 'mvrnorm'
    ## by putting error^2 in lower-right position of VCOV
    newVCOV <- VCOV
    newVCOV[varPLACE, varPLACE] <- predERROR^2
    
    ## create MC simulation matrix
    simMAT <- mvrnorm(n = nsim, mu = MU, Sigma = newVCOV, empirical = TRUE)
    
    ## evaluate expression on rows of simMAT
    EVAL <- try(eval(EXPR, envir = as.data.frame(simMAT)), silent = TRUE)
    if (inherits(EVAL, "try-error")) stop("There was an error evaluating the simulations!")
    
    ## collect statistics
    PRED <- data.frame(predVAL)
    colnames(PRED) <- predNAME   
    FITTED <- predict(object, newdata = data.frame(PRED))
    MEAN.sim <- mean(EVAL, na.rm = TRUE)
    SD.sim <- sd(EVAL, na.rm = TRUE)
    MEDIAN.sim <- median(EVAL, na.rm = TRUE)
    MAD.sim <- mad(EVAL, na.rm = TRUE)
    QUANT <- quantile(EVAL, c((1 - level)/2, level + (1 - level)/2))
    RES <- c(FITTED, MEAN.sim, SD.sim, MEDIAN.sim, MAD.sim, QUANT[1], QUANT[2])
    outMAT <- rbind(outMAT, RES)
  }
  
  colnames(outMAT) <- c("fit", "mean", "sd", "median", "mad", names(QUANT[1]), names(QUANT[2]))
  rownames(outMAT) <- NULL
  
  cat("\n")
  
  return(outMAT)  
}

######## Main Function #######
#get data
data <- read.csv(infile, head=TRUE,sep=","); # 'sep' = separator like csv file
#print header
writeLines("metrics,R_squared,coefficient",con=outSummary,sep = "\n");

metrics <- c("UserThrput","Pktloss","RTT");
#y_index <- c(5,7,9); #index of the column of y-value in the data
y_index <- c(5); #index of the column of y-value in the data
idx = 1;

for (i in y_index) {
  #select empirial data, and clean up the data for N/A
  data[,y_index[idx]] = as.numeric( gsub('[^eE0-9.-]', '', data[,y_index[idx]])) #remove symbol and non_numeric values
  good = complete.cases(data[,y_index[idx]]) #remove empty row with empty cell or NA
  y = data[good,y_index[idx]]
  PRB = data[good,4]
  #actual regression
  #ourlin <- lm(y ~ PRB); #linear fit
  ourlin <- lm(y ~ PRB + I(PRB^2)); #2nd degree polymonimal fit
  #ourlin <- nls(y ~ a * log(PRB) + b, start=list(a=-1, b=5), control = list(maxiter = 500), trace = TRUE); #log fit
  #ourlin <- nls(y ~ a*exp(b *PRB) + b); #exponential fit
  
  prettyprint(metrics[idx], ourlin);
  idx = idx + 1;
}


close(outSummary);
close(outFit);

####### End Main Function #####
