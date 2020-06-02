#
# Some useful keyboard shortcuts for package authoring:
#
#   Build and Reload Package:  'Ctrl + Shift + B'
#   Check Package:             'Ctrl + Shift + E'
#   Test Package:              'Ctrl + Shift + T'

# parse the solver type from the filepath
getSolverType <- function(fullPath){
  # input: full path name to daily Summary
  # output: "stochastic" or "determinsitc"
  #
  # if the path contains stochstic, return that
  # likewise with deterministic

  if(grepl("*stochastic*",fullPath)==T){
    output <- "stochastic"
  }else{
    output <- "deterministic"
  }
  return(output)
}

# parse the scenario# from the filepath
getPrescientNumber <-function(fullpath){
  #input: fullpathname
  #output: the number from the filepath

  allLoc <- gregexpr("*prescient_*",ignore.case=T,fullpath)[[1]]
  loc <- allLoc[length(allLoc)]
  scenario <- substr(fullpath,loc+10,loc+10)


  return(scenario)
}

# get the wind scaling factor from the pathname
getWindLevel<-function(fullpath){
  #input: fullpathname
  #output: the number from the filepath

  allLoc <- gregexpr("*presc_*",ignore.case=T,fullpath)[[1]]
  loc <- allLoc[length(allLoc)]
  end <- gregexpr("_",substr(fullpath,loc+7,1000000))[[1]][1]
  wind <- substr(fullpath,loc+7,loc+7+end-2)


  return(wind)
}

# get the cutpoints from the pathname
getCutpoints<-function(fullPath){
  #input: fullpathname
  #output: the number from the filepath

  #if it has wang0827 return that

  if(grepl("*wang0827*",fullPath)==T){
    output <- "wang_27"
    return(output)
  } else if(grepl("*wang0864*",fullPath)==T){
    output <- "wang_64"
    return(output)
  } else if(grepl("*quant_reg27*",fullPath)==T){
    output <- "quant_reg_27"
    return(output)
  } else if(grepl("*quant_reg64*",fullPath)==T){
    output <- "quant_reg_64"
    return(output)
  }

  allLoc <- gregexpr("*_cp*",ignore.case=T,fullPath)[[1]]


  loc <- allLoc[length(allLoc)]
  end <- gregexpr("/",substr(fullPath,loc+1,1000000))[[1]][1]
  cutpoints <- substr(fullPath,loc+1,loc+1+end-2)


  return(cutpoints)
}

# get the path x numbers of directories above the file or folder
getParentPath <- function(fullpath, levelsUp=0){
  #input: fullpathname and number of parents to go up. 0 takes you to the same
  # place
  #output: the path of the directory that levels up

  if(file.exists(fullpath)){
    # so we know it's a file, not a directory

    slashLocations <- gregexpr("/",fullpath)[1][[1]]
    totalSlashes <- length(slashLocations)

    output <- substr(fullpath,0,slashLocations[totalSlashes-levelsUp])

  }else{

    slashLocations <- gregexpr("/",fullpath)[1][[1]]
    totalSlashes <- length(slashLocations)

    output <- substr(fullpath,0,slashLocations[totalSlashes-levelsUp])

  }
  return(output)
}

# multi-gsub
mgsub <- function(pattern, replacement, x, ...) {
  if (length(pattern)!=length(replacement)) {
    stop("pattern and replacement do not have the same length.")
  }
  result <- x
  for (i in 1:length(pattern)) {
    result <- gsub(pattern[i], replacement[i], fixed=T,result, ...)
  }
  result
}

getMetricsCutpoints<-function(fullpath){
  #input: fullpathname
  #output: the number from the filepath

  allLoc <- gregexpr("*_cp_*",ignore.case=T,fullpath)[[1]]


  if(allLoc<0){
    #print(fullpath)
    return("quant_reg")
  }

  loc <- allLoc[length(allLoc)]
  end <-nchar(fullpath)
  cutpoints <- substr(fullpath,loc+1, end-4)


  return(cutpoints)
}

scaleColumn <- function(col){
  min <- min(col)
  max <- max(col)
  range <- max-min

  returnCol <- data.frame(output=numeric(length(col)))
  for(i in 1:length(col)){
    returnCol$output[i] <-(col[i]-min)/range
  }

  return(returnCol)
}


#' Plot Prescient scenario
#'
#' @param type Type 1 indicates we are taking the data from a dataframe in the
#'   prescient output format. Type 2 indicates we are pointing the function at a
#'   path to the pyspdir output directory
#' @param dataFrame If type 1 is slected, this is the dataframe. The structure
#'   should be the same as the Prescient output format
#' @param pyspDirPath If type 2 is selected, this is required as the path to the
#'   specific date's folder in the pyspdir_2stage directory
#' @param plot T or F. Do we plot?
#' @param saveID T or F, do we save?
#' @param upperBound What do we use to scale the data by? Note: this has to be
#'   adjusted if we change the wind scaling factors
#' @param plotFileName If we are plotting, what's our file name? Saved in the
#'   current wd unless modified by a relative path.
#' @param title Title of the plot
#'
#' @return One kickass scenario plot
#' @export
#'
#' @examples
plotScenario <- function ( type=1,dataFrame=NULL,pyspDirPath=NULL, plot=T,saveID=FALSE,upperBound = 4782,plotFileName= "scenario_plot.pdf",title="default title"){

  #devtools::use_package("scales")
  #library("scales")
  # If type 1 is selected, we need a dataframe. Check.
  if (type == 1 &&
      is.null(dataFrame)) {
    return(
      "Type 1 means read a dataframe, you must specify the dataframe object"
    )

  }

  # If type 2 is selected, we need a pyspDir path. Check.
  if (type == 2 &&
      is.null(pyspDirPath)) {
    return("Type 2 means read from pyspDir output (legacy AF). You must specify a pyspDirPath value")

  }




  if(type==2) {
    # set the path based on the pyspPath argument
    path <- pyspDirPath

    # check if the path ends with a /
    if (substr(path, start = nchar(path), stop = nchar(path) + 1) != "/") {
      path <- paste0(path, "/")
    }

    # now get the scenarios dataframe
    scenarios <- getDfFromPySpDir(path)
  }else{

    #if we are in type 1, just set scenarios as the dataFrame
    scenarios <- dataFrame

  }


  # now we have the 'scenario' dataframe so let's plot it.
  if(plot==T) {
    divideby <- upperBound
    temp <- scenarios[(2:nrow(scenarios)),(2:ncol(scenarios))] / divideby
    scenarios[(2:nrow(scenarios)),(2:ncol(scenarios))] <-
      temp

    if (saveID == T) {
      pdf(
        width = 6,
        height = 6,
        file = paste0(plotFileName)
      )
    }

    plot(
      x = scenarios[2:25, 1],
      y = scenarios[2:25, 3],
      type = 'n',
      ylim = c(0, 1),
      xlab = 'Hour',
      ylab = 'Wind Power',
      main = title
    )

    for (k in 4:length(scenarios[2,])) {
      prob <- scenarios[1, k]
      lines(
        x = scenarios[2:25, 1],
        y = scenarios[2:25, k],
        col = scales::alpha('dodgerblue', alpha = 0.7)
      )
    }

    points(
      x = scenarios[2:25, 1],
      y = scenarios[2:25, 3],
      ylim = c(0, 1),
      type = 'b',
      pch = 15,
      col = 'black'
    )
    points(
      x = scenarios[2:25, 1],
      y = scenarios[2:25, 2],
      type = 'b',
      pch = 15,
      col = 'red'
    )
    legend(
      3,
      1,
      c('actual', 'forecast'),
      col = c('red', 'black'),
      lty = c(1, 1),
      lwd = c(1, 1),
      pch = c(15, 15),
      box.lty = 0
      #title = title
    )

    if (saveID == T) {
      dev.off()
    }
    p1 <- recordPlot()

  }
  return(list(plot=p1))
}


#' Create R dataframe from the Presient 1.0 Populator output (reads the pysp format)
#'
#' @param fullPath The path to the directory full of scenarios and actuals.
#'
#' @return Returns a dataframe of the actuals, forecast, and all scenarios with their probabilities.
#' @export
#'
#' @examples
getDfFromPySpDir <- function(fullPath){


  #hardcoded stuff:
  # the line numbers which correspond with retreiving the scenarios
  # the search string for finding the probabilities


  scenarios <-
    data.frame(Hour = numeric(48),
               Forecast = numeric(48),
               Actual = numeric(48))

  file.list <-
    sort(list.files(
      path = fullPath,
      pattern = "Scenario*",
      full.names = T
    ))



  file.list <-
    file.list[-c(1, 2, length(file.list))] # clean out the actuals/forecasts
  probabilities <-  data.frame(prob = numeric(length(file.list)))
  actual.file <-
    list.files(path = fullPath,
               pattern = "*actuals*",
               full.names = T)
  forecast.file <-
    list.files(path = fullPath,
               pattern = "*expected*",
               full.names = T)
  prob.file     <-
    list.files(path = fullPath,
               pattern = "*Structure*",
               full.names = T)

  # read in forecast
  forecast <- as.data.frame(readLines(forecast.file)[165:212])
  tmpfcst <-
    matrix(unlist(strsplit(as.character(forecast[, 1]), split = " ")),
           ncol = 4,
           byrow = T)


  scenarios$Hour <- as.numeric(tmpfcst[, 2])
  scenarios$Forecast <- as.numeric(tmpfcst[, 4])

  # read in actual
  actual <- as.data.frame(readLines(actual.file)[165:212])
  tmpact <-
    matrix(unlist(strsplit(as.character(actual[, 1]), split = " ")),
           ncol = 4,
           byrow = T)

  scenarios$Actual <- as.numeric(tmpact[, 4])

  # read in probability and populate probability dataframe
  prob.lines <- suppressWarnings(readLines(prob.file))
  start.line <- grep("param ConditionalProbability :", prob.lines)
  prob.lines <-
    as.data.frame(prob.lines[(start.line):(start.line + length(file.list) +
                                             1)])
  prob <-
    matrix(unlist(strsplit(as.character(prob.lines[(3:length(prob.lines[, 1])), ]), split =
                             " ")),
           ncol = 2,
           byrow = T)
  probabilities$prob <- as.numeric(prob[, 2])

  for (d in 1:(length(file.list))) {
    this.file <- file.list[d]
    scenario.dat <-  as.data.frame(readLines(this.file)[165:212])
    tmpscen <-
      matrix(unlist(strsplit(as.character(scenario.dat[, 1]), split = " ")),
             ncol = 4,
             byrow = T)

    scenarios <- cbind(scenarios, as.numeric(tmpscen[, 4]))
    names(scenarios)[d + 3] <- (paste0("s", d))
  }

  row1 <- c(NA,NA,NA,probabilities$prob)
  output <- rbind(row1,scenarios)

  return(output)
}



#' Return Variogram score from a day's worth of scenarios
#'
#' @param type Type 1 means we are reading a dataframe in the output format specified in prescient 1.0. Type 2 read in a pysp directory and get the info and make the dataframe
#' @param dataFrame If type 1 is specified, this is the dataframe object. Not required for type 2
#' @param pyspDirPath If type 2 is specified, this is the path to the directory
#' @param p P paramter as specified by the variogram score documentation
#' @param upperBound If scale is true, this is the upper bound we are using to scale the data.
#'
#' @return Variogram score as a number.
#' @export
#'
#' @examples
getVariogramScore <- function(type=1,dataFrame=NULL,pyspDirPath, p, upperBound=0){
  # Implement variogram score of order p
  # From: Scheuerer, Michael, and Thomas M. Hamill. "Variogram-based
  #         proper scoring rules for probabilistic forecasts of multivariate quantities."
  #         Monthly Weather Review 143.4 (2015): 1321-1334.


  if (type == 1 &&
      is.null(dataFrame)) {
    return(
      "Type 1 means read a dataframe, you must specify the dataframe object"
    )

  }

  if (type == 2 &&
      is.null(pyspDirPath)) {
    return("Type 2 means read from pyspDir output (legacy AF). You must specify a pyspDirPath value")

  }





  if(type==2) {
    # set the path based on the path or the
    path <- pyspDirPath

    # check if the path ends with a /
    if (substr(path, start = nchar(path), stop = nchar(path) + 1) != "/") {
      path <- paste0(path, "/")
    }

  }


  VSpDF <- function(scenario.set) {
    act <- as.matrix(scenario.set[(2:nrow(scenario.set)),2]/upperBound)
    scens <- as.matrix(scenario.set[(2:nrow(scenario.set)),(4:ncol(scenario.set))])
    scens <- scens/upperBound
    probs <- as.matrix(scenario.set[1,(4:ncol(scenario.set))])
    S.score <- 0
    for (i in 1:24) {
      for (j in 1:24) {
        # weight <- 1/nrow(scens)
        weight <- actweights[i, j]
        calc.act <- abs(act[i] - act[j]) ^ p
        exp.scen <- 0
        for (k in 1:ncol(scens)) {
          exp.scen <- exp.scen + (probs[k] * abs(scens[i, k] - scens[j, k]) ^ p)
        }
        S.score <- S.score + (weight * (calc.act - exp.scen) ^ 2)
      }
    }
    return(S.score)
  }

  #TODO: fix this so it's not arbitrary and pulling from nothing
  # Determine weights (This is hardcoded AF at the moment. )

  data("actuals")
  actuals <- actuals[,-2]
  colnames(actuals) <- c('datetime', 'actual')

  idx <- which(substr(actuals[,1], 10, 11) == '00')
  storeact <- data.frame()
  for (i in 1:(length(idx) - 1)){
    storeact <- rbind(storeact, actuals[(idx[i]:(idx[i] + 23)),2])
  }
  colnames(storeact) <- paste0('Hr', seq(0, 23, 1))

  actcor <- cor(storeact)


  # Weights from actual correlation or error correlation? Actual for now
  actweights <- actcor
  for (line in 1:nrow(actweights)){
    actweights[line,] <- actweights[line,] / sum(actweights[line,])
  }


  #works with the pyspDir Path
  if(type==2){
    date.df <- getDfFromPySpDir(pyspDirPath)
    #print(head(date.df))



    return( VSpDF(date.df))
  }


  #works with just the input dataframe
  if(type==1){

    date.df <- as.data.frame(dataFrame,head=T)


    return(VSpDF(date.df))
  }


}


#' Title
#'
#' @param type Type 1 means we are reading a dataframe in the output format specified in prescient 1.0. Type 2 read in a pysp directory and get the info and make the dataframe
#' @param dataFrame If we use type 1, this is the R dataframe. Format is Prescient 1.0 output format.
#' @param pyspDirPath IF we are using type 2, this is the fullpath to the directory which contains the scenarios, actuals, and forecasts
#' @param upperBound The upper bound for scaling the scores
#'
#' @return Returns the energy score for the given day.
#' @export
#'
#' @examples
getEnergyScore <- function(type=1,dataFrame=NULL,pyspDirPath,upperBound=0){


  if (type == 1 &&
      is.null(dataFrame)) {
    return(
      "Type 1 means read a dataframe, you must specify the dataframe object"
    )

  }

  if (type == 2 &&
      is.null(pyspDirPath)) {
    return("Type 2 means read from pyspDir output (legacy AF). You must specify a pyspDirPath value")

  }

  if(type==2) {
    # set the path based on the path or the
    path <- pyspDirPath

    # check if the path ends with a /
    if (substr(path, start = nchar(path), stop = nchar(path) + 1) != "/") {
      path <- paste0(path, "/")
    }
  }


  EnergyScoreDF <- function(scenario.df){
    saveA <- c()
    saveB <- c()
    length <- length(scenario.df[,2])
    probList <- as.matrix(scenario.df[1,(4:ncol(scenario.df))])
    for (i in 4:ncol(scenario.df)){
      a <- probList[i-3] * norm((scenario.df[2:nrow(scenario.df),2]- scenario.df[2:nrow(scenario.df),i]), type='2')
      saveC <- c()
      for (j in 4:ncol(scenario.df)){
        partial <- probList[i-3] * probList[j-3] * norm((scenario.df[2:nrow(scenario.df),i] - scenario.df[2:nrow(scenario.df),j]), type='2')
        saveC[(j-3)] <- partial
      }
      saveA[(i-3)] <- unlist(a)
      saveB[(i-3)] <- sum(unlist(saveC))
    }
    ES <- sum(saveA) - 0.5 * sum(saveB)
    return(ES)
  }


  if(type==2) {

    date.df <- getDfFromPySpDir(path)

    date.df[(2:nrow(date.df)), (2:ncol(date.df))] <-
      date.df[(2:nrow(date.df)), (2:ncol(date.df))] / upperBound


    energyScore <-
      EnergyScoreDF(date.df)

  }

  if(type==1){


      dataFrame[(2:nrow(dataFrame)), (2:ncol(dataFrame))] <-
        dataFrame[(2:nrow(dataFrame)), (2:ncol(dataFrame))] / upperBound


    energyScore <- EnergyScoreDF(dataFrame)

  }

  return(energyScore)
}


getBrierScore <- function(pyspDirPath,from,to,k=3,threshold=0.2,event="Significant Gradient",upperBound){

  path <- pyspDirPath
  #if the path doesn't end with a / add one
  if(substr(path,start=nchar(path),stop = nchar(path)+1)!="/"){
    path <- paste0(path,"/")
  }

  n.day <- length(seq.Date(from=as.Date(from),to=as.Date(to),by="day"))
  n.hour <- 24
  max.scen <- length(list.files(path=paste0(path,from)))

  event.obs <- function(){
    if (event == 'Significant Gradient'){
      return(ifelse(max(observation[t:(t+k)]) - min(observation[t:(t+k)]) >= threshold, 1, 0))
    } else if (event == 'Ramp Up'){
      return(ifelse(!is.unsorted(observation[t:(t+k)]) & observation[(t+k)] - observation[t] >= threshold, 1, 0))
    } else if (event == 'Ramp Down'){
      return(ifelse(!is.unsorted(-observation[t:(t+k)]) & observation[t] - observation[(t+k)] >= threshold, 1, 0))
    }
  }

  event.scen <- function(){
    if (event == 'Significant Gradient'){
      return(ifelse(max(scenario[j,t:(t+k)]) - min(scenario[j, t:(t+k)]) >= threshold, 1, 0))
    } else if (event == 'Ramp Up'){
      return(ifelse(!is.unsorted(scenario[j, t:(t+k)]) & scenario[j,(t+k)] - scenario[j,t] >= threshold, 1, 0))
    } else if (event == 'Ramp Down'){
      return(ifelse(!is.unsorted(-scenario[j, t:(t+k)]) & scenario[j,t] - scenario[j,(t+k)] >= threshold, 1, 0))
    }
  }

  ## Store results for Brier Score terms
  g.obs <- matrix(data=NA, nrow=n.day, ncol=(n.hour-k))
  g.scen <- array(data=NA, dim=c(n.day, n.hour-k, max.scen))
  P.scen <- matrix(data=NA, nrow=n.day, ncol=(n.hour-k))
  brier <- as.data.frame(matrix(data=NA, nrow=n.day, ncol=2))

  n <- 1
  for (i in 0:n.day){
    date <- as.Date(from) + i
    if (file.exists(paste0(set.path, date, '.RData'))){
      load(paste0(set.path, date, '.RData'))
      n.scen <- ncol(scenarios) - 3
      sdata <- as.matrix(scenarios[2:25, -c(1,3)])
      # Normalize data
      sdata <- sdata / 4782

      scenario <- t(sdata[,-1])
      observation <- sdata[,1]
      probs <- scenarios[1, 4:ncol(scenarios)]
      # Calculate observed gradient event
      for (t in 1:(n.hour - k)){
        g.obs[n,t] <- event.obs()
        for (j in 1:n.scen){
          g.scen[n,t,j] <- event.scen()
        }
        P.scen[n,t] <- sum(probs * na.omit(g.scen[n,t,]))
        brier[n,1] <- as.character(date)
        brier[n,2] <- sum((P.scen[n,] - g.obs[n,])^2) / (n.hour - k)
      }
      n <- n + 1
    }
  }





}
